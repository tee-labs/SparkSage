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

#: Default cosine threshold above which two post-rerank chunks are treated as
#: semantic near-duplicates and the weaker one is dropped. ``0.9`` is strict --
#: only near-identical blocks (paraphrases / copies) collapse -- so distinct
#: content is never wrongly merged. See :func:`_dedup_chunks_by_embedding`.
DEFAULT_DEDUP_THRESHOLD = 0.9


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
    dedup_threshold:
        Optional cosine threshold above which two post-rerank chunks are treated
        as semantic near-duplicates and the weaker one is dropped (default
        :data:`DEFAULT_DEDUP_THRESHOLD` = ``0.9``). Blocks whose embeddings
        reach the threshold collapse, keeping the best-scoring representative
        and saving reader context / token budget. ``None`` disables dedup.

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
        dedup_threshold: float | None = DEFAULT_DEDUP_THRESHOLD,
    ) -> None:
        if fetch_factor < 1:
            raise ValueError("fetch_factor must be >= 1")
        if min_fetch < 1:
            raise ValueError("min_fetch must be >= 1")
        if dedup_threshold is not None:
            if isinstance(dedup_threshold, bool) or not isinstance(
                dedup_threshold, (int, float)
            ):
                raise TypeError("dedup_threshold must be a float or None")
            if not 0.0 <= dedup_threshold <= 1.0:
                raise ValueError("dedup_threshold must be in [0.0, 1.0] or None")
        self._registry = registry
        self._store = store
        self._embedder = embedder
        self._lexical: LexicalRetriever = lexical if lexical is not None else NullLexicalRetriever()
        self._reranker: Reranker = reranker if reranker is not None else IdentityReranker()
        self._fetch_factor = fetch_factor
        self._min_fetch = min_fetch
        self._dedup_threshold: float | None = dedup_threshold

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
            fused = reciprocal_rank_fusion(rankings, top_n=pool)
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

        if use_rerank and not isinstance(self._reranker, IdentityReranker):
            # When dedup is on, keep the full reranked pool so dedup does not
            # starve the final ``k``; the slice to ``k`` happens at the end.
            rerank_top = None if dedup_on else k
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
        elif not reranked:
            resolved = [_replaced(c, rank=i) for i, c in enumerate(resolved[:k])]

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


@dataclass
class RetrievalConfig:
    """Convenience bundle for the knobs :class:`Retriever.search` exposes.

    Lets callers (e.g. :class:`~sparksage.qa.QAEngine`) hold one typed config
    object instead of threading kwargs through every call.
    """

    k: int = 5
    fetch_factor: int = DEFAULT_FETCH_FACTOR
    min_fetch: int = DEFAULT_MIN_FETCH
    dedup_threshold: float | None = DEFAULT_DEDUP_THRESHOLD
    use_lexical: bool = True
    use_rerank: bool = True


__all__ = [
    "DEFAULT_DEDUP_THRESHOLD",
    "DEFAULT_FETCH_FACTOR",
    "DEFAULT_MIN_FETCH",
    "RetrievalConfig",
    "Retriever",
]
