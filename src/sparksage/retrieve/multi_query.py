"""Multi-query retrieval: one primary query + variants, fused via RRF.

This is the shared recall-boost path used by both :class:`~sparksage.qa.QAEngine`
(the single-shot baseline, for sub-query decomposition / multi-query expansion)
and :class:`~sparksage.agent.AgenticQAEngine` (the per-step retrieval inside the
agent loop). Extracting it keeps the two orchestrators in lockstep: the same
``retrieve each variant -> RRF-fuse -> build RetrievedChunks`` semantics, with
no duplicated logic.

The function depends only on the :class:`~sparksage.retrieve.Retriever` and the
RRF helper in :mod:`sparksage.retrieve.fusion` -- it is pure stdlib beyond
those, so it stays unit-testable with :class:`~sparksage.generator.FakeLLMClient`
/ :class:`~sparksage.embed.FakeEmbeddingClient`.
"""

from __future__ import annotations

from sparksage.retrieve.fusion import reciprocal_rank_fusion
from sparksage.retrieve.models import RetrievalResult, RetrievedChunk
from sparksage.retrieve.orchestrator import Retriever


def multi_query_retrieve(
    retriever: Retriever,
    primary: str,
    sub_queries: list[str],
    *,
    k: int,
    filter: object = None,
    use_lexical: bool = True,
    use_rerank: bool = False,
) -> RetrievalResult:
    """Run retrieval for the primary + each sub-query and RRF-fuse the ranked lists.

    Each variant gets its own recall (without the expensive rerank pass -- RRF
    needs ranks only), the ranked lists are fused via RRF, and the fused pool is
    rebuilt as :class:`RetrievedChunk` objects carrying the fused score plus the
    underlying dense/lexical scores for transparency.

    A single-query call (no usable sub-queries) short-circuits to one plain
    :meth:`Retriever.search` so there is no RRF overhead when expansion is off.

    Parameters
    ----------
    retriever:
        Any :class:`~sparksage.retrieve.Retriever`. Its block registry is the
        source of truth for resolving fused block ids back to chunks.
    primary:
        The seed / call-level query. Always included as the first RRF input.
    sub_queries:
        Additional variants (paraphrases, HyDE hypothesis, decomposed
        sub-questions). De-duplicated against ``primary``.
    k:
        Top-k to finally return (and the per-variant fetch depth scales with
        ``max(k * 2, 5)`` so the fusion pool is generous).
    filter, use_lexical, use_rerank:
        Forwarded to each per-variant retrieval. ``use_rerank`` is forced to
        ``False`` on the per-variant fetches (rerank happens after fusion).
    """
    queries = [primary] + [q for q in (sub_queries or []) if q and q != primary]
    if len(queries) == 1:
        return retriever.search(
            queries[0],
            k=k,
            filter=filter,
            use_lexical=use_lexical,
            use_rerank=use_rerank,
        )

    fetch_depth = max(k * 2, 5)
    registry = retriever._registry  # noqa: SLF001 (shared registry, as in QAEngine)
    fused_ids: dict[str, float] = {}
    dense_by: dict[str, float] = {}
    lex_by: dict[str, float] = {}
    rankings = []
    for q in queries:
        res = retriever.search(
            q,
            k=fetch_depth,
            filter=filter,
            use_lexical=use_lexical,
            use_rerank=False,
        )
        rankings.append(res.dense_hits)
        for h in res.dense_hits:
            dense_by.setdefault(h.block_id, h.score)
        for h in res.lexical_hits:
            lex_by.setdefault(h.block_id, h.score)

    fused = (
        reciprocal_rank_fusion(rankings, top_n=fetch_depth) if any(rankings) else []
    )
    for h in fused:
        fused_ids[h.block_id] = h.score

    chunks = [
        RetrievedChunk(
            block=registry[bid],
            score=score,
            dense_score=dense_by.get(bid),
            lexical_score=lex_by.get(bid),
            rank=i,
        )
        for i, (bid, score) in enumerate(
            sorted(fused_ids.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        if bid in registry
    ][:k]

    return RetrievalResult(
        query=primary,
        chunks=chunks,
        dense_hits=[h for h in fused if h.block_id in dense_by] or fused,
        lexical_hits=[],
        fused=True,
        reranked=False,
        filtered_out=0,
    )


__all__ = ["multi_query_retrieve"]
