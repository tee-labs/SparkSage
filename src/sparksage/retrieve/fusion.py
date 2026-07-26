"""Rank fusion for hybrid / multi-query retrieval (pure stdlib).

When retrieval produces more than one ranked candidate list -- a dense list and
a lexical list (hybrid), or several lists from multi-query expansion -- the
lists must be merged into one without comparing incomparable scores (a cosine
``0.8`` and a BM25 ``12.3`` are not on the same scale).

:func:`reciprocal_rank_fusion` implements Reciprocal Rank Fusion (RRF), the
standard score-free merge: each candidate's fused score is the sum over input
lists of ``1 / (k + rank)``, where ``rank`` is its 1-indexed position in that
list and ``k`` (default ``60``, the canonical value from the original paper) is
a smoothing constant. RRF only needs *ranks*, never scores, so it is robust to
arbitrary score distributions and trivially combines any number of lists.

This is the dependency-free fusion step the retrieval orchestrator and the
multi-query expander both consume.
"""

from __future__ import annotations

from sparksage.embed.store import SearchHit

#: Canonical RRF smoothing constant (from Cormack, Clarke & Buettcher, 2009).
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[SearchHit]],
    *,
    k_const: int = DEFAULT_RRF_K,
    top_n: int | None = None,
) -> list[SearchHit]:
    """Fuse multiple ranked hit lists into one via Reciprocal Rank Fusion.

    Parameters
    ----------
    rankings:
        One or more ranked :class:`~sparksage.embed.store.SearchHit` lists
        (each best-first). Lists need not be the same length; a candidate
        absent from a list simply contributes nothing for that list.
    k_const:
        RRF smoothing constant (default ``60``). Larger dampens the advantage
        of the very top ranks; the canonical ``60`` works well in practice.
    top_n:
        If given, return only the top-``n`` fused hits. ``None`` returns all.

    Returns
    -------
    list[SearchHit]
        Fused hits sorted by descending fused score, then by ``block_id`` for
        determinism. Each ``block_id`` appears at most once. The reported
        ``score`` is the fused RRF score (not a cosine).

    Examples
    --------
    >>> from sparksage.embed.store import SearchHit
    >>> from sparksage.retrieve import reciprocal_rank_fusion
    >>> dense = [SearchHit("a", 0.9), SearchHit("b", 0.8)]
    >>> lex = [SearchHit("b", 11.0), SearchHit("c", 9.0)]
    >>> fused = reciprocal_rank_fusion([dense, lex])
    >>> fused[0].block_id
    'b'
    """
    if k_const < 1:
        raise ValueError("k_const must be >= 1")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be >= 1")

    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.block_id] = scores.get(hit.block_id, 0.0) + 1.0 / (
                k_const + rank
            )
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_n is not None:
        fused = fused[:top_n]
    return [SearchHit(block_id=bid, score=s) for bid, s in fused]


__all__ = ["DEFAULT_RRF_K", "reciprocal_rank_fusion"]
