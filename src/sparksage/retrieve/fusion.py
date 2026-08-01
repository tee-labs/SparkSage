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

The ``k=60`` default is the most-copied value in the RAG literature -- and that
is exactly why it should not be trusted blindly. It originated in the original
email-retrieval setting; on a new corpus the optimal ``k`` can differ, and the
only honest way to pick it is to measure. :func:`tune_rrf_k` is the dependency-
free tuning entrypoint: given a small set of labelled queries (the ranked lists
each produced + which block ids are relevant), it sweeps a grid of ``k``
candidates and returns the one maximizing mean recall@``top_n`` -- so the
default is an empirical choice, not a superstition.

This is the dependency-free fusion step the retrieval orchestrator and the
multi-query expander both consume.
"""

from __future__ import annotations

from sparksage.embed.store import SearchHit

#: Canonical RRF smoothing constant (from Cormack, Clarke & Buettcher, 2009).
#:
#: This is the most widely *copied* value in the RAG literature, which also
#: makes it the one most likely to be wrong for a new corpus. It is a sensible
#: starting point, but treat it as a placeholder: sweep the grid with
#: :func:`tune_rrf_k` on your own labelled queries to pick the value that
#: actually maximizes recall on *your* data. The smoothing constant trades off
#: how much the very top ranks dominate -- a smaller ``k`` favours rank-1 hits,
#: a larger ``k`` spreads the weight more evenly across the top ranks.
DEFAULT_RRF_K = 60

#: Default ``k_const`` grid swept by :func:`tune_rrf_k` when none is supplied.
DEFAULT_TUNE_K_CANDIDATES: tuple[int, ...] = (10, 20, 40, 60, 80, 100)


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
        of the very top ranks; the canonical ``60`` works well in practice but
        is widely copied -- sweep it with :func:`tune_rrf_k` on labelled data
        to find the value that maximizes recall for *your* corpus.
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


def tune_rrf_k(
    queries: list[list[list[SearchHit]]],
    relevant: list[set[str]],
    *,
    k_candidates: list[int] | None = None,
    top_n: int = 10,
) -> int:
    """Pick the RRF smoothing constant that maximizes mean recall@``top_n``.

    The canonical ``k=60`` is the most-copied RRF constant -- and the one most
    likely to be wrong on a new corpus. This dependency-free tuner settles the
    choice empirically: for each candidate ``k`` it fuses the ranked lists of
    every labelled query, measures how many of the *known-relevant* block ids
    surface in the fused top-``top_n``, and returns the ``k`` with the best mean
    recall. Ties are broken by the smallest ``k`` (cheaper, favours the top).

    Parameters
    ----------
    queries:
        One entry per labelled query: the list of ranked :class:`SearchHit`
        lists to fuse (e.g. ``[dense_hits, lexical_hits]`` for hybrid, or the
        multi-query variants). Each inner list is best-first.
    relevant:
        One set of relevant ``block_id`` strings per query, parallel to
        ``queries``. A block id in the fused top-``top_n`` that is also in this
        set counts as a hit.
    k_candidates:
        The ``k_const`` values to sweep (default :data:`DEFAULT_TUNE_K_CANDIDATES`).
        Each must be ``>= 1``.
    top_n:
        The cutoff at which recall is measured (default ``10``). Must be ``>= 1``.

    Returns
    -------
    int
        The ``k_const`` value with the highest mean recall@``top_n``. When every
        candidate scores identically, the smallest candidate wins. When no
        labelled data is supplied, :data:`DEFAULT_RRF_K` is returned as a safe
        default.

    Raises
    ------
    ValueError
        If ``queries`` and ``relevant`` have different lengths, or any
        ``k_candidates`` / ``top_n`` value is ``< 1``.

    Examples
    --------
    >>> from sparksage.embed.store import SearchHit
    >>> from sparksage.retrieve import tune_rrf_k
    >>> dense = [SearchHit("a", 0.9), SearchHit("b", 0.8)]
    >>> lex = [SearchHit("b", 11.0), SearchHit("c", 9.0)]
    >>> best_k = tune_rrf_k([[dense, lex]], [{"b"}], top_n=2)
    >>> best_k >= 1
    True
    """
    if len(queries) != len(relevant):
        raise ValueError("queries and relevant must be the same length")
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise TypeError("top_n must be an int")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    grid = list(k_candidates) if k_candidates is not None else list(DEFAULT_TUNE_K_CANDIDATES)
    if not grid:
        raise ValueError("k_candidates must not be empty")
    for kc in grid:
        if isinstance(kc, bool) or not isinstance(kc, int):
            raise TypeError("k_candidates must be ints")
        if kc < 1:
            raise ValueError("each k_candidate must be >= 1")

    if not queries:
        return DEFAULT_RRF_K

    best_k = grid[0]
    best_score = -1.0
    for kc in grid:
        total = 0.0
        for ranked_lists, rel in zip(queries, relevant, strict=True):
            if not rel:
                continue
            fused = reciprocal_rank_fusion(ranked_lists, k_const=kc, top_n=top_n)
            hits = sum(1 for h in fused if h.block_id in rel)
            total += hits / len(rel)
        mean_recall = total / len(queries)
        if mean_recall > best_score or (
            mean_recall == best_score and kc < best_k
        ):
            best_score = mean_recall
            best_k = kc
    return best_k


__all__ = [
    "DEFAULT_RRF_K",
    "DEFAULT_TUNE_K_CANDIDATES",
    "reciprocal_rank_fusion",
    "tune_rrf_k",
]
