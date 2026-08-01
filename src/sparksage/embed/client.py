"""Embedding client abstraction for IdeaBlock vectorization.

The embedding core depends *only* on the :class:`EmbeddingClient` protocol, so it
is fully unit-testable with the deterministic :class:`FakeEmbeddingClient`. A
concrete :class:`OpenAIEmbeddingClient` (backed by the ``openai`` SDK's
embeddings endpoint) is provided for production use against any
OpenAI-compatible embeddings API.

The ``openai`` package is an *optional* dependency -- install it with
``pip install 'sparksage[embed]'``.
"""

from __future__ import annotations

import hashlib
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

_logger = logging.getLogger(__name__)

#: Per-request input cap for the OpenAI embeddings endpoint. The API hard limit
#: is 2048 inputs; 1000 is the battle-tested safe value used by the Distill
#: pipeline.
DEFAULT_BATCH_SIZE = 1000

#: Concurrency for concurrent embedding batches.
DEFAULT_MAX_WORKERS = 10

#: Known vector dimensionalities for common OpenAI embedding models, so the
#: :attr:`EmbeddingClient.dimension` can be reported without a probe call.
_KNOWN_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@runtime_checkable
class EmbeddingClient(Protocol):
    """Minimal embedding interface the indexer depends on.

    Any callable producing a fixed-length dense vector per input string
    implements this -- the OpenAI embeddings API in production, a local
    sentence-transformers model, or a deterministic fake for tests.
    """

    @property
    def dimension(self) -> int:
        """The length of every vector this client produces."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per input text, preserving order.

        Implementations should batch internally (see
        :data:`DEFAULT_BATCH_SIZE`) and must return exactly ``len(texts)``
        vectors. Empty input returns an empty list.
        """
        ...


def _l2_normalize(vec: list[float]) -> list[float]:
    """Scale ``vec`` to unit length (pure-Python, no numpy)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


@dataclass
class FakeEmbeddingClient:
    """Deterministic, dependency-free embedding client for tests.

    Uses a signed feature-hashing ("hashing trick") over word/char n-grams so
    that *semantically overlapping texts produce overlapping hash buckets* and
    therefore non-trivial cosine similarity -- useful for testing
    similarity-driven code (e.g. the future Distill pipeline) with zero
    external dependencies.

    All vectors are L2-normalized to unit length so cosine similarity reduces to
    a plain dot product.

    Parameters
    ----------
    dimension:
        Length of the produced vectors (default 128).
    """

    dimension: int = 128

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        lowered = text.lower()
        tokens = lowered.split()
        grams: list[str] = list(tokens)
        grams.extend(" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1))
        for tok in tokens:
            n = len(tok)
            if n >= 3:
                grams.extend(tok[i : i + 3] for i in range(n - 2))
            else:
                grams.append(tok)
        for gram in grams:
            digest = hashlib.sha256(gram.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if (digest[4] & 1) else -1.0
            vec[idx] += sign
        return _l2_normalize(vec)


class OpenAIEmbeddingClient:
    """Embedding client backed by an OpenAI-compatible embeddings endpoint.

    Works with OpenAI, Azure OpenAI, and anything else that speaks the
    ``embeddings.create`` protocol. Point it at a self-hosted / non-OpenAI
    endpoint via ``base_url``.

    Requests are batched at :data:`DEFAULT_BATCH_SIZE` inputs per API call and
    run concurrently over a :class:`~concurrent.futures.ThreadPoolExecutor` for
    throughput. Vectors are L2-normalized by default so cosine similarity is a
    plain dot product (the convention used by FAISS ``IndexFlatIP``).

    The ``openai`` package is an *optional* dependency -- install it with
    ``pip install 'sparksage[embed]'``.

    Parameters
    ----------
    model:
        Embedding model name (default ``text-embedding-3-small``).
    dimension:
        Vector dimensionality. If omitted it is looked up from
        :data:`_KNOWN_DIMS`, then probed lazily on the first
        :meth:`embed_batch` call. Pass it explicitly to avoid the probe or for
        non-standard models.
    batch_size:
        Max inputs sent per API request (default :data:`DEFAULT_BATCH_SIZE`).
        Must be ``<= 2048`` (the OpenAI API limit).
    max_workers:
        Concurrency for concurrent batches (default
        :data:`DEFAULT_MAX_WORKERS`).
    normalize:
        L2-normalize every vector (default ``True``).
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimension: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_workers: int = DEFAULT_MAX_WORKERS,
        normalize: bool = True,
        timeout: float | None = None,
        **client_kwargs: Any,
    ) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "OpenAIEmbeddingClient requires the 'openai' package. "
                "Install it with: pip install 'sparksage[embed]'"
            ) from exc
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if batch_size > 2048:
            raise ValueError("batch_size must be <= 2048 (OpenAI API limit)")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, **client_kwargs
        )
        self._model = model
        self._batch_size = batch_size
        self._max_workers = max_workers
        self._normalize = normalize
        self._dim = dimension if dimension is not None else _KNOWN_DIMS.get(model)

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._probe_dimension()
        if self._dim is None:
            raise RuntimeError(
                "embedding dimension is unknown for model "
                f"{self._model!r}; pass dimension= to the constructor"
            )
        return self._dim

    def _probe_dimension(self) -> None:
        """Probe the endpoint with a single short text to discover the dimension.

        Called lazily the first time :attr:`dimension` is accessed for a model
        not in :data:`_KNOWN_DIMS` and without an explicit ``dimension=`` (e.g.
        when a :class:`~sparksage.kb.KnowledgeBase` sizes its vector store at
        construction). A single short text is embedded; the length of the
        returned vector fixes ``_dim``. Errors are swallowed so callers see the
        :class:`RuntimeError` from :attr:`dimension` with actionable guidance.
        """
        try:
            sample = self.embed_batch(["dimension probe"])
            if sample and sample[0]:
                self._dim = len(sample[0])
                _logger.debug(
                    "probed embedding dimension for %r: %d", self._model, self._dim
                )
        except Exception as exc:
            _logger.debug(
                "dimension probe failed for %r: %s", self._model, exc
            )

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        def _embed_slice(start: int) -> tuple[int, list[list[float]]]:
            chunk = texts[start : start + self._batch_size]
            resp = self._client.embeddings.create(
                model=self._model, input=chunk
            )
            return start, [list(d.embedding) for d in resp.data]

        starts = list(range(0, len(texts), self._batch_size))
        by_start: dict[int, list[list[float]]] = {}
        if len(starts) <= 1:
            s, vecs = _embed_slice(starts[0])
            by_start[s] = vecs
        else:
            with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
                futures = [ex.submit(_embed_slice, s) for s in starts]
                for fut in futures:
                    s, vecs = fut.result()
                    by_start[s] = vecs

        out: list[list[float]] = []
        for s in starts:
            out.extend(by_start[s])

        if self._dim is None and out:
            self._dim = len(out[0])
            _logger.debug("probed embedding dimension for %r: %d", self._model, self._dim)

        if self._normalize:
            out = [_l2_normalize(v) for v in out]
        return out
