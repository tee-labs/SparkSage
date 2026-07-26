"""Semantic cache for the QA pipeline: near-duplicate queries short-circuit.

The biggest cost lever in a RAG system is the LLM calls (rewrite + rerank +
generate + judge). A *semantic* cache keyed on query *meaning* (not exact text)
short-circuits the whole pipeline when a near-duplicate prior query was already
answered -- "how do I deploy" hits the cached answer for "how to deploy".

:class:`InMemorySemanticCache` embeds each stored query via a pluggable
:class:`~sparksage.embed.client.EmbeddingClient` and, on lookup, returns the
cached result whose embedding is within ``threshold`` cosine of the query.
This is exactly the dependency-free similarity judgement
:func:`~sparksage.find_similar_pairs` uses, applied to the cache -- pure
stdlib, unit-testable with :class:`~sparksage.embed.FakeEmbeddingClient`.

The cache implements the :class:`~sparksage.qa.QACache` protocol structurally
(``lookup`` / ``store``), so it drops straight into a
:class:`~sparksage.qa.QAEngine`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from sparksage.embed.client import EmbeddingClient

_logger = logging.getLogger(__name__)

V = TypeVar("V")

#: Default cosine threshold above which two queries are treated as duplicates.
DEFAULT_CACHE_THRESHOLD = 0.90

#: Default max cached entries (LRU-style eviction, FIFO actually -- oldest first).
DEFAULT_MAX_ENTRIES = 1024


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (na * nb)


@dataclass
class _Entry(Generic[V]):
    vector: list[float]
    value: V


class InMemorySemanticCache(Generic[V]):
    """Embedding-similarity cache over arbitrary values.

    Each key is embedded once on :meth:`store`; :meth:`lookup` embeds the query
    and returns the value of the most similar stored key whose cosine
    similarity is at or above ``threshold``. Eviction is oldest-first once
    ``max_entries`` is exceeded.

    Parameters
    ----------
    embedder:
        Any :class:`EmbeddingClient` (real :class:`OpenAIEmbeddingClient` in
        production, :class:`FakeEmbeddingClient` in tests).
    threshold:
        Minimum cosine similarity for a cache hit (default ``0.90``). Lower is
        more aggressive (more hits, more false matches); higher is stricter.
    max_entries:
        Soft cap on stored entries (default ``1024``).
    key_extractor:
        Optional callable to derive the cache key from the stored value (e.g.
        pull the query out of a :class:`~sparksage.qa.QAResult`). Defaults to
        using the ``query`` argument passed to :meth:`store`.

    Examples
    --------
    >>> from sparksage import FakeEmbeddingClient
    >>> from sparksage.query import InMemorySemanticCache
    >>> cache = InMemorySemanticCache(FakeEmbeddingClient(dimension=64))
    >>> cache.store("how to deploy", value="answer")
    >>> cache.lookup("how to deploy")   # doctest: +SKIP
    'answer'
    """

    def __init__(
        self,
        embedder: EmbeddingClient,
        *,
        threshold: float = DEFAULT_CACHE_THRESHOLD,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        key_extractor: Callable[[V], str] | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._embedder = embedder
        self._threshold = float(threshold)
        self._max_entries = int(max_entries)
        self._key_extractor = key_extractor
        self._entries: list[_Entry[V]] = []
        self._keys: list[str] = []
        self.hits = 0
        self.misses = 0
        self.stores = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Drop every cached entry (counters are kept)."""
        self._entries.clear()
        self._keys.clear()

    def store(self, query: str, value: V) -> None:
        """Embed ``query`` and cache ``value`` under it.

        If ``key_extractor`` was supplied, ``query`` is treated as the value
        and the actual key is extracted from it -- this is the shape the
        :class:`~sparksage.qa.QACache` protocol uses (``store(query, result)``).
        """
        key = self._resolve_key(query, value)
        vec = self._embed_one(key)
        if vec is None:
            return
        self._entries.append(_Entry(vector=vec, value=value))
        self._keys.append(key)
        self.stores += 1
        if len(self._entries) > self._max_entries:
            self._entries.pop(0)
            self._keys.pop(0)

    def lookup(self, query: str) -> V | None:
        """Return the value of the nearest cached key at/above threshold."""
        qvec = self._embed_one(query)
        if qvec is None or not self._entries:
            self.misses += 1
            return None
        best_idx = -1
        best_score = -1.0
        for i, entry in enumerate(self._entries):
            score = _cosine(qvec, entry.vector)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0 and best_score >= self._threshold:
            self.hits += 1
            return self._entries[best_idx].value
        self.misses += 1
        return None

    def _resolve_key(self, query: str, value: V) -> str:
        if self._key_extractor is not None:
            try:
                return str(self._key_extractor(value))
            except Exception as exc:  # pragma: no cover - defensive
                _logger.warning("semantic cache key_extractor failed: %s", exc)
                return str(query)
        return str(query)

    def _embed_one(self, text: str) -> list[float] | None:
        text = str(text)
        if not text.strip():
            return None
        vecs = self._embedder.embed_batch([text])
        if not vecs:
            return None
        return list(vecs[0])


@dataclass
class CacheStats:
    """Snapshot of semantic-cache hit/miss counters."""

    hits: int = 0
    misses: int = 0
    stores: int = 0
    size: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.lookups) if self.lookups else 0.0


def semantic_cache_stats(cache: InMemorySemanticCache) -> CacheStats:
    """Return a :class:`CacheStats` snapshot of ``cache``."""
    return CacheStats(
        hits=cache.hits,
        misses=cache.misses,
        stores=cache.stores,
        size=len(cache),
    )


__all__ = [
    "CacheStats",
    "DEFAULT_CACHE_THRESHOLD",
    "DEFAULT_MAX_ENTRIES",
    "InMemorySemanticCache",
    "semantic_cache_stats",
]
