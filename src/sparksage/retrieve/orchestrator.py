"""Retrieval orchestration: hybrid recall -> fusion -> re-rank -> filter.

:class:`Retriever` is the query-time counterpart of the ingest pipeline. It
consumes the vectors / keywords the ingest side has been producing and returns
ranked :class:`~sparksage.retrieve.models.RetrievedChunk` lists ready to feed a
reader. It finally wires up the three "designed but unconsumed" IdeaBlock
fields the analysis flagged:

* ``keywords`` -> :class:`~sparksage.retrieve.lexical.BM25Retriever` (lexical
  recall boosting, now actually used).
* ``tags`` / ``entities`` / ``language`` / ``kb_id`` -> :class:`RetrievalFilter`
  metadata scoping (multi-tenant / permission retrieval).
* ``source.locator`` -> surfaced on each chunk via
  :meth:`RetrievedChunk.to_citation`, ready for grounded citations.

The orchestrator depends only on protocols it already owns plus the existing
:class:`~sparksage.embed.store.VectorStore`,
:class:`~sparksage.embed.indexer.BlockEmbedder` and the
:class:`~sparksage.retrieve.lexical.LexicalRetriever` /
:class:`~sparksage.retrieve.reranker.Reranker` protocols -- so it is fully
unit-testable with
:class:`~sparksage.embed.FakeEmbeddingClient` and zero network calls.

Because the :class:`VectorStore` is deliberately text-agnostic, metadata
filtering is a *post-filter* over an over-fetched dense pool (``fetch_factor``
x ``k``). This keeps the store decoupled from embedding exactly as designed;
for exact filtered kNN swap in a backend that supports native metadata filters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sparksage.embed.indexer import BlockEmbedder
from sparksage.embed.store import SearchHit, VectorStore
from sparksage.retrieve.fusion import reciprocal_rank_fusion
from sparksage.retrieve.lexical import LexicalRetriever, NullLexicalRetriever
from sparksage.retrieve.models import RetrievalFilter, RetrievalResult, RetrievedChunk
from sparksage.retrieve.reranker import IdentityReranker, Reranker
from sparksage.schema.ideablock import IdeaBlock

_logger = logging.getLogger(__name__)

#: How large a dense pool to fetch relative to ``k`` before post-filtering.
DEFAULT_FETCH_FACTOR = 8
#: Absolute floor on the dense pool size (so tiny ``k`` still over-fetches).
#:
#: Empirically a reranker needs a candidate pool of at least ~50 before it has
#: enough selection space to move the needle -- with the previous ``30`` a
#: typical ``k=5`` query produced a pool of only ``40``, just under that line.
#: The ``50`` floor keeps the reranker effective for small ``k``.
DEFAULT_MIN_FETCH = 50

#: Default RRF weight for the dense leg (equal weighting preserves the original
#: score-free RRF). WeKnora favours ``0.7`` here; sweep it with
#: :func:`~sparksage.retrieve.fusion.tune_rrf_weights` on labelled data.
DEFAULT_DENSE_WEIGHT = 1.0
#: Default RRF weight for the lexical leg. WeKnora favours ``0.3`` here; only
#: the ratio to :data:`DEFAULT_DENSE_WEIGHT` affects the fused ordering.
DEFAULT_LEXICAL_WEIGHT = 1.0

#: Default cosine threshold above which two post-rerank chunks are treated as
#: semantic near-duplicates and the weaker one is dropped. ``0.9`` is strict --
#: only near-identical blocks (paraphrases / copies) collapse -- so distinct
#: content is never wrongly merged. See :func:`_dedup_chunks_by_embedding`.
DEFAULT_DEDUP_THRESHOLD = 0.9

#: Default score-floor fallback decay (WeKnora): when a score floor wipes out
#: every candidate, relax the threshold by multiplying it by this factor and
#: retry, down to :data:`DEFAULT_SCORE_RETRY_FLOOR`.
DEFAULT_SCORE_RETRY_FACTOR = 0.7
#: Default lower bound for the retry threshold (WeKnora ``0.3``): retrying stops
#: once the decayed threshold would drop below this.
DEFAULT_SCORE_RETRY_FLOOR = 0.3
#: Default last-resort floor for the top-1 fallback (WeKnora ``0.15``): when the
#: decayed threshold still yields nothing, the single best chunk is kept only if
#: its score is at least this high -- otherwise the result is genuinely empty.
DEFAULT_SCORE_MIN_TOP1 = 0.15


class Retriever:
    """Hybrid + re-rank retrieval orchestrator.

    Parameters
    ----------
    registry:
        A ``{block_id (str): IdeaBlock}`` mapping used to resolve hit ids to
        blocks and to apply :class:`RetrievalFilter` scoping. The retriever
        does not own this registry -- it is shared with the
        :class:`~sparksage.kb.KnowledgeBase` (or built directly for tests).
    store:
        The dense :class:`VectorStore`.
    embedder:
        The :class:`BlockEmbedder` used to embed the query text into a vector.
    lexical:
        Optional :class:`LexicalRetriever` (e.g. :class:`BM25Retriever`). When
        ``None``, retrieval is dense-only. Pass a
        :class:`~sparksage.retrieve.lexical.NullLexicalRetriever` explicitly to
        keep "lexical disabled" as a uniform protocol object.
    reranker:
        Optional :class:`Reranker`. Defaults to
        :class:`~sparksage.retrieve.reranker.IdentityReranker` so reranking is
        always configurable as "off" without branching.
    fetch_factor, min_fetch:
        The dense pool size is ``max(min_fetch, k * fetch_factor)`` -- large
        enough that post-filtering rarely starves the final ``k``.
    dense_weight, lexical_weight:
        Per-leg RRF weights (generalized / weighted RRF, the WeKnora-style
        ``weight / (k + rank)``). Defaults are equal (``1.0`` / ``1.0``) so the
        fused ordering is identical to the original score-free RRF; only the
        *ratio* matters, so WeKnora's ``0.7`` / ``0.3`` is a drop-in starting
        point. Sweep it with
        :func:`~sparksage.retrieve.fusion.tune_rrf_weights` on labelled data.
    dedup_threshold:
        Optional cosine threshold above which two post-rerank chunks are treated
        as semantic near-duplicates and the weaker one is dropped (default
        :data:`DEFAULT_DEDUP_THRESHOLD` = ``0.9``). Blocks whose embeddings
        reach the threshold collapse, keeping the best-scoring representative
        and saving reader context / token budget. ``None`` disables dedup.
    min_score:
        Optional absolute floor on each chunk's final ``score`` (the WeKnora
        rerank-threshold guard). Chunks scoring below it are dropped so weak /
        irrelevant blocks no longer fill the top-``k`` just because the pool was
        over-fetched. When that filter would empty the result, the threshold
        decays by ``score_retry_factor`` (down to ``score_retry_floor``); if
        still empty the single best chunk survives when it clears
        ``score_min_top1``. Calibrated for normalized scores -- it applies on
        the re-rank path and the dense-only (cosine) path; it is **skipped**
        when the final score is an un-reranked RRF fusion score (RRF scores are
        rank-based, not comparable to an absolute threshold). ``None`` (the
        default) disables the guard, preserving the original "fill k" behaviour.
    score_retry_factor, score_retry_floor, score_min_top1:
        The three WeKnora fallback knobs (defaults ``0.7`` / ``0.3`` / ``0.15``).
        See ``min_score``.

    Examples
    --------
    >>> from sparksage import (
    ...     BlockEmbedder, FakeEmbeddingClient, InMemoryVectorStore,
    ... )
    >>> from sparksage.retrieve import Retriever, BM25Retriever   # doctest: +SKIP
    >>> store = InMemoryVectorStore(dimension=64)                 # doctest: +SKIP
    >>> retriever = Retriever(                                    # doctest: +SKIP
    ...     registry=registry, store=store,
    ...     embedder=BlockEmbedder(FakeEmbeddingClient(dimension=64)),
    ...     lexical=BM25Retriever(),
    ... )
    >>> result = retriever.search("how to deploy", k=5)           # doctest: +SKIP
    """

    def __init__(
        self,
        registry: dict[str, IdeaBlock],
        store: VectorStore,
        embedder: BlockEmbedder,
        *,
        lexical: LexicalRetriever | None = None,
        reranker: Reranker | None = None,
        fetch_factor: int = DEFAULT_FETCH_FACTOR,
        min_fetch: int = DEFAULT_MIN_FETCH,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
        dedup_threshold: float | None = DEFAULT_DEDUP_THRESHOLD,
        min_score: float | None = None,
        score_retry_factor: float = DEFAULT_SCORE_RETRY_FACTOR,
        score_retry_floor: float = DEFAULT_SCORE_RETRY_FLOOR,
        score_min_top1: float = DEFAULT_SCORE_MIN_TOP1,
    ) -> None:
        if fetch_factor < 1:
            raise ValueError("fetch_factor must be >= 1")
        if min_fetch < 1:
            raise ValueError("min_fetch must be >= 1")
        dense_weight = _check_weight(dense_weight, "dense_weight")
        lexical_weight = _check_weight(lexical_weight, "lexical_weight")
        if dedup_threshold is not None:
            if isinstance(dedup_threshold, bool) or not isinstance(
                dedup_threshold, (int, float)
            ):
                raise TypeError("dedup_threshold must be a float or None")
            if not 0.0 <= dedup_threshold <= 1.0:
                raise ValueError("dedup_threshold must be in [0.0, 1.0] or None")
        score_retry_factor = _check_score_param(
            score_retry_factor, "score_retry_factor", allow_zero=False
        )
        score_retry_floor = _check_score_param(
            score_retry_floor, "score_retry_floor", allow_zero=False
        )
        score_min_top1 = _check_score_param(
            score_min_top1, "score_min_top1", allow_zero=True
        )
        if not score_min_top1 <= score_retry_floor:
            raise ValueError("score_min_top1 must be <= score_retry_floor")
        if min_score is not None:
            min_score = _check_score_param(min_score, "min_score", allow_zero=False)
            if min_score < score_retry_floor:
                raise ValueError("min_score must be >= score_retry_floor")
        self._registry = registry
        self._store = store
        self._embedder = embedder
        self._lexical: LexicalRetriever = lexical if lexical is not None else NullLexicalRetriever()
        self._reranker: Reranker = reranker if reranker is not None else IdentityReranker()
        self._fetch_factor = fetch_factor
        self._min_fetch = min_fetch
        self._dense_weight = dense_weight
        self._lexical_weight = lexical_weight
        self._dedup_threshold: float | None = dedup_threshold
        self._min_score: float | None = min_score
        self._score_retry_factor = score_retry_factor
        self._score_retry_floor = score_retry_floor
        self._score_min_top1 = score_min_top1

    @property
    def store(self) -> VectorStore:
        return self._store

    @property
    def embedder(self) -> BlockEmbedder:
        return self._embedder

    @property
    def lexical(self) -> LexicalRetriever:
        return self._lexical

    @property
    def reranker(self) -> Reranker:
        return self._reranker

    @property
    def dedup_threshold(self) -> float | None:
        """The post-rerank semantic-dedup cosine threshold, or ``None`` if off."""
        return self._dedup_threshold

    @property
    def dense_weight(self) -> float:
        """RRF weight applied to the dense leg."""
        return self._dense_weight

    @property
    def lexical_weight(self) -> float:
        """RRF weight applied to the lexical leg."""
        return self._lexical_weight

    @property
    def min_score(self) -> float | None:
        """The post-rerank score floor, or ``None`` when the guard is off."""
        return self._min_score

    @property
    def score_retry_factor(self) -> float:
        """Per-retry decay applied to :attr:`min_score` (WeKnora ``0.7``)."""
        return self._score_retry_factor

    @property
    def score_retry_floor(self) -> float:
        """Lower bound for the decayed retry threshold (WeKnora ``0.3``)."""
        return self._score_retry_floor

    @property
    def score_min_top1(self) -> float:
        """Top-1 last-resort floor (WeKnora ``0.15``)."""
        return self._score_min_top1

    def index(self, blocks: list[IdeaBlock]) -> None:
        """(Re)build the dense + lexical indexes from ``blocks``.

        Mutates the passed-in ``registry`` to mirror the new corpus and fills
        each block's ``embedding`` in place. A no-op for an empty list.
        """
        if not blocks:
            self._registry.clear()
            if hasattr(self._store, "clear"):
                self._store.clear()
            if hasattr(self._lexical, "_n"):
                self._lexical.index([])
            return
        self._registry.clear()
        for b in blocks:
            self._registry[str(b.id)] = b
        self._embedder.embed_blocks(blocks)
        self._store.add_many(self._embedder.vectors_for(blocks))
        self._lexical.index(blocks)

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        filter: RetrievalFilter | None = None,
        use_lexical: bool = True,
        use_rerank: bool = True,
    ) -> RetrievalResult:
        """Run hybrid + fusion + re-rank retrieval for ``query``.

        Parameters
        ----------
        query:
            The (possibly rewritten) query text. Embedded once for the dense
            leg and tokenized once for the lexical leg.
        k:
            Final number of chunks to return.
        filter:
            Optional :class:`RetrievalFilter` metadata scope.
        use_lexical:
            When ``True`` (default) and a real lexical retriever is wired, run
            the lexical leg and fuse. ``False`` forces dense-only.
        use_rerank:
            When ``True`` (default) and a non-identity reranker is wired,
            re-rank the fused pool. ``False`` skips re-ranking.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if not str(query).strip():
            return RetrievalResult(query=str(query))

        flt = filter if filter is not None else RetrievalFilter()
        pool = max(self._min_fetch, k * self._fetch_factor)

        raw_dense = self._dense_search_raw(query, pool)
        dense_hits = self._apply_filter_to_hits(raw_dense, flt)
        raw_lexical: list[SearchHit] = []
        lexical_hits: list[SearchHit] = []
        has_lexical = use_lexical and not isinstance(self._lexical, NullLexicalRetriever)
        if has_lexical and len(self._lexical) > 0:
            raw_lexical = self._lexical.search(query, k=pool)
            lexical_hits = self._apply_filter_to_hits(raw_lexical, flt)

        rankings: list[list[SearchHit]] = [dense_hits]
        if lexical_hits:
            rankings.append(lexical_hits)

        if len(rankings) > 1:
            # Weighted RRF: only the dense/lexical weight *ratio* affects the
            # ordering; the equal-weight default (1.0 / 1.0) reproduces the
            # original score-free RRF exactly.
            fused = reciprocal_rank_fusion(
                rankings,
                weights=[self._dense_weight, self._lexical_weight],
                top_n=pool,
            )
            did_fuse = True
        else:
            fused = dense_hits[:pool]
            did_fuse = False
        fused = fused[: max(pool, k)]

        resolved = self._resolve(fused, dense_hits, lexical_hits)
        filtered_out = (len(raw_dense) - len(dense_hits)) + (
            len(raw_lexical) - len(lexical_hits)
        )

        dedup_on = self._dedup_threshold is not None
        will_rerank = use_rerank and not isinstance(self._reranker, IdentityReranker)
        # The score floor is calibrated for normalized scores (reranker output
        # or a dense cosine). It is meaningless on an un-reranked RRF fusion
        # score -- RRF scores are rank-based, not on an absolute scale -- so it
        # is skipped in exactly that one path.
        floor_applies = (
            self._min_score is not None and not (did_fuse and not will_rerank)
        )
        # When dedup or the score floor needs to look past rank ``k`` (to recover
        # deeper chunks after weaker ones are dropped), keep the full reranked
        # pool; the slice to ``k`` happens at the very end.
        keep_full_pool = dedup_on or floor_applies

        if will_rerank:
            rerank_top = None if keep_full_pool else k
            resolved = self._reranker.rerank(query, resolved, top_n=rerank_top)
            reranked = True
        else:
            reranked = False

        if dedup_on:
            before = len(resolved)
            resolved = _dedup_chunks_by_embedding(resolved, self._dedup_threshold)  # type: ignore[arg-type]
            if len(resolved) < before:
                _logger.debug(
                    "semantic dedup %d -> %d chunks (threshold=%.2f)",
                    before,
                    len(resolved),
                    self._dedup_threshold,
                )
            resolved = [_replaced(c, rank=i) for i, c in enumerate(resolved)]
        elif not reranked and not keep_full_pool:
            resolved = [_replaced(c, rank=i) for i, c in enumerate(resolved[:k])]

        if floor_applies:
            assert self._min_score is not None  # noqa: S101 (narrowed by floor_applies)
            before = len(resolved)
            resolved = _apply_score_floor(
                resolved,
                self._min_score,
                retry_factor=self._score_retry_factor,
                retry_floor=self._score_retry_floor,
                min_top1=self._score_min_top1,
            )
            if len(resolved) < before:
                _logger.debug(
                    "score floor %d -> %d chunks (min_score=%.2f)",
                    before,
                    len(resolved),
                    self._min_score,
                )
            resolved = [_replaced(c, rank=i) for i, c in enumerate(resolved)]

        resolved = resolved[:k]

        return RetrievalResult(
            query=str(query),
            chunks=resolved,
            dense_hits=dense_hits,
            lexical_hits=lexical_hits,
            fused=did_fuse,
            reranked=reranked,
            filtered_out=max(filtered_out, 0),
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _dense_search_raw(self, query: str, pool: int) -> list[SearchHit]:
        if len(self._store) == 0:
            return []
        qvecs = self._embedder.embed_texts([query])
        if not qvecs:
            return []
        return self._store.search(qvecs[0], k=pool)

    def _apply_filter_to_hits(
        self,
        hits: list[SearchHit],
        flt: RetrievalFilter,
    ) -> list[SearchHit]:
        if flt.is_empty:
            return list(hits)
        out: list[SearchHit] = []
        for h in hits:
            block = self._registry.get(h.block_id)
            if block is not None and flt.matches(block):
                out.append(h)
        return out

    def _resolve(
        self,
        fused: list[SearchHit],
        dense: list[SearchHit],
        lexical: list[SearchHit],
    ) -> list[RetrievedChunk]:
        dense_by = {h.block_id: h.score for h in dense}
        lex_by = {h.block_id: h.score for h in lexical}
        out: list[RetrievedChunk] = []
        for hit in fused:
            block = self._registry.get(hit.block_id)
            if block is None:
                continue
            out.append(
                RetrievedChunk(
                    block=block,
                    score=hit.score,
                    dense_score=dense_by.get(hit.block_id),
                    lexical_score=lex_by.get(hit.block_id),
                )
            )
        return out


def _replaced(chunk: RetrievedChunk, *, rank: int) -> RetrievedChunk:
    from dataclasses import replace

    return replace(chunk, rank=rank)


def _dot(a: list[float], b: list[float]) -> float:
    """Pure-Python dot product (vectors are L2-normalized, so cosine)."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def _dedup_chunks_by_embedding(
    chunks: list[RetrievedChunk],
    threshold: float,
) -> list[RetrievedChunk]:
    """Greedy near-duplicate removal over an ordered (best-first) chunk list.

    Walks ``chunks`` in order and keeps a chunk unless its block embedding is
    at or above ``threshold`` cosine similarity to any *already kept* chunk --
    so the best-scoring representative survives and its near-duplicate
    paraphrases / copies are dropped. Blocks without an embedding are always
    kept (nothing to compare against). This is the query-time counterpart of
    :func:`sparksage.embed.find_similar_pairs`, applied to an *ordered* single
    retrieval result rather than an all-pairs corpus scan.
    """
    kept: list[RetrievedChunk] = []
    kept_vecs: list[list[float]] = []
    for chunk in chunks:
        vec = chunk.block.embedding
        if vec is None:
            kept.append(chunk)
            continue
        if any(_dot(vec, kv) >= threshold for kv in kept_vecs):
            continue
        kept.append(chunk)
        kept_vecs.append(list(vec))
    return kept


def _check_weight(value: float, name: str) -> float:
    """Validate a non-negative RRF leg weight (``bool`` rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return float(value)


def _check_score_param(value: float, name: str, *, allow_zero: bool) -> float:
    """Validate a score-threshold knob in ``[0, 1]`` (``bool`` rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0]")
    if not allow_zero and value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return float(value)


def _apply_score_floor(
    chunks: list[RetrievedChunk],
    min_score: float,
    *,
    retry_factor: float,
    retry_floor: float,
    min_top1: float,
) -> list[RetrievedChunk]:
    """WeKnora-style score floor with decayed-retry + top-1 fallback.

    ``chunks`` is assumed best-first by ``score``. Chunks scoring below
    ``min_score`` are dropped; when that would empty the list the threshold
    decays by ``retry_factor`` (floored at ``retry_floor``) and the filter
    retries, until the threshold stops changing. If every decayed level still
    yields nothing, the single best chunk is kept only when it clears
    ``min_top1`` -- otherwise the result is genuinely empty. This prevents a
    too-strict floor from silently returning nothing while still blocking
    irrelevant context that would otherwise fill the top-``k``.
    """
    threshold = min_score
    seen: set[float] = set()
    while threshold not in seen:
        seen.add(threshold)
        kept = [c for c in chunks if c.score >= threshold]
        if kept:
            return kept
        threshold = max(threshold * retry_factor, retry_floor)
    if chunks and chunks[0].score >= min_top1:
        return [chunks[0]]
    return []


@dataclass
class RetrievalConfig:
    """Convenience bundle for the knobs :class:`Retriever.search` exposes.

    Lets callers (e.g. :class:`~sparksage.qa.QAEngine`) hold one typed config
    object instead of threading kwargs through every call.
    """

    k: int = 5
    fetch_factor: int = DEFAULT_FETCH_FACTOR
    min_fetch: int = DEFAULT_MIN_FETCH
    dense_weight: float = DEFAULT_DENSE_WEIGHT
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT
    dedup_threshold: float | None = DEFAULT_DEDUP_THRESHOLD
    min_score: float | None = None
    score_retry_factor: float = DEFAULT_SCORE_RETRY_FACTOR
    score_retry_floor: float = DEFAULT_SCORE_RETRY_FLOOR
    score_min_top1: float = DEFAULT_SCORE_MIN_TOP1
    use_lexical: bool = True
    use_rerank: bool = True


__all__ = [
    "DEFAULT_DEDUP_THRESHOLD",
    "DEFAULT_DENSE_WEIGHT",
    "DEFAULT_FETCH_FACTOR",
    "DEFAULT_LEXICAL_WEIGHT",
    "DEFAULT_MIN_FETCH",
    "DEFAULT_SCORE_MIN_TOP1",
    "DEFAULT_SCORE_RETRY_FACTOR",
    "DEFAULT_SCORE_RETRY_FLOOR",
    "RetrievalConfig",
    "Retriever",
]
