"""Retrieval-quality metrics for the benchmark suite (pure stdlib).

The benchmark scores two indexing strategies (IdeaBlock vs naive chunking) on
the *same* set of queries with known ground truth. These helpers compute the
standard top-k retrieval metrics -- hit@k, MRR, mean top score -- plus a
token-efficiency measure, so a report can quantify both "does the right answer
surface?" and "what does it cost?".

No external metric library is needed: everything is computed from the ranked
``SearchHit`` lists the existing :class:`~sparksage.embed.store.InMemoryVectorStore`
already returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sparksage.embed.store import SearchHit


@dataclass
class TokenStats:
    """Token-budget summary for one indexing strategy.

    Attributes
    ----------
    unit_count:
        Number of indexed units (IdeaBlocks or baseline chunks).
    total_chars:
        Sum of characters across all indexed units (the embedded text length).
    avg_chars:
        Mean characters per unit.
    total_tokens:
        Estimated tokens across all units (``chars / chars_per_token``).
    avg_tokens:
        Estimated tokens per unit.
    chars_per_token:
        The heuristic ratio used (default ``4.0`` -- the standard OpenAI-ish
        approximation). Override via ``token_counter`` on the runner for a real
        tokenizer.
    """

    unit_count: int = 0
    total_chars: int = 0
    avg_chars: float = 0.0
    total_tokens: float = 0.0
    avg_tokens: float = 0.0
    chars_per_token: float = 4.0


@dataclass
class RetrievalMetrics:
    """Top-k retrieval quality for one indexing strategy over a query set.

    Attributes
    ----------
    query_count:
        Number of queries evaluated.
    hit_at_k:
        Mapping ``k -> hit rate`` (fraction of queries whose ground truth appears
        in the top ``k``).
    mrr:
        Mean Reciprocal Rank. ``1.0`` means the ground truth is always ranked
        first; ``0.0`` means it never appears.
    avg_top_score:
        Mean cosine similarity of the top-1 hit per query. Higher means the
        index produces more confident retrievals.
    k_values:
        The ``k`` values the metrics were computed at.
    """

    query_count: int = 0
    hit_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    avg_top_score: float = 0.0
    k_values: tuple[int, ...] = (1, 3, 5)

    @property
    def hit_at_1(self) -> float:
        return self.hit_at_k.get(1, 0.0)

    @property
    def hit_at_3(self) -> float:
        return self.hit_at_k.get(3, 0.0)

    @property
    def hit_at_5(self) -> float:
        return self.hit_at_k.get(5, 0.0)


def approx_tokens(text: str, chars_per_token: float = 4.0) -> float:
    """Cheap token estimate (``len / chars_per_token``), no tokenizer needed.

    The ``4`` default is the widely-used approximation for English text on
    BPE tokenizers. It is deliberately approximate -- the benchmark compares
    *relative* token efficiency between strategies using the same heuristic, so
    the approximation cancels out. Plug in a real tokenizer via
    :class:`~sparksage.bench.runner.BenchmarkRunner`'s ``token_counter`` for
    absolute numbers.
    """
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be > 0")
    return len(text) / chars_per_token


def token_stats(
    texts: list[str],
    *,
    chars_per_token: float = 4.0,
) -> TokenStats:
    """Summarize the token budget of an index built from ``texts``."""
    if not texts:
        return TokenStats(chars_per_token=chars_per_token)
    total_chars = sum(len(t) for t in texts)
    n = len(texts)
    return TokenStats(
        unit_count=n,
        total_chars=total_chars,
        avg_chars=total_chars / n,
        total_tokens=total_chars / chars_per_token,
        avg_tokens=total_chars / n / chars_per_token,
        chars_per_token=chars_per_token,
    )


def evaluate_retrieval(
    rankings: list[list[SearchHit]],
    ground_truth: list[set[str]],
    *,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> RetrievalMetrics:
    """Compute :class:`RetrievalMetrics` from ranked hits + per-query truth.

    Parameters
    ----------
    rankings:
        One ranked list of :class:`~sparksage.embed.store.SearchHit` per query
        (best first), as returned by
        :meth:`~sparksage.embed.store.InMemoryVectorStore.search`.
    ground_truth:
        One set of relevant ids per query, aligned with ``rankings``. A query is
        a "hit@k" when *any* ground-truth id appears in the top ``k``.
    k_values:
        The ``k`` cutoffs to report (default ``(1, 3, 5)``).
    """
    if len(rankings) != len(ground_truth):
        raise ValueError(
            f"rankings ({len(rankings)}) and ground_truth ({len(ground_truth)}) "
            "must have equal length"
        )
    if not k_values:
        raise ValueError("k_values must not be empty")

    n = len(rankings)
    if n == 0:
        return RetrievalMetrics(query_count=0, k_values=tuple(k_values))

    max_k = max(k_values)
    hit_counts = {k: 0 for k in k_values}
    reciprocal_ranks: list[float] = []
    top_scores: list[float] = []

    for hits, truth in zip(rankings, ground_truth, strict=True):
        if hits:
            top_scores.append(float(hits[0].score))
        ranked_ids = [h.block_id for h in hits[:max_k]]
        first_relevant_rank: int | None = None
        for rank, bid in enumerate(ranked_ids, start=1):
            if bid in truth:
                if first_relevant_rank is None:
                    first_relevant_rank = rank
                for k in k_values:
                    if rank <= k:
                        hit_counts[k] += 1
                break
        if first_relevant_rank is not None:
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        else:
            reciprocal_ranks.append(0.0)

    return RetrievalMetrics(
        query_count=n,
        hit_at_k={k: hit_counts[k] / n for k in k_values},
        mrr=sum(reciprocal_ranks) / n,
        avg_top_score=sum(top_scores) / len(top_scores) if top_scores else 0.0,
        k_values=tuple(k_values),
    )


__all__ = [
    "RetrievalMetrics",
    "TokenStats",
    "approx_tokens",
    "evaluate_retrieval",
    "token_stats",
]
