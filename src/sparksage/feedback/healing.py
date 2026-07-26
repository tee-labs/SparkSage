"""Corpus self-healing: turn user feedback back into ingest-side signals.

This is the loop that closes the quality flywheel -- the thing the project
analysis called the "P4 feedback loop". Query-side signals (which the ingest
side could never see before) flow back as concrete ingest actions:

* **Low-recall queries** -- a question that repeatedly retrieved zero / too few
  blocks means the corpus has a *coverage gap* on that topic: the source
  document needs re-chunking, or new content needs ingesting. Surfaced as
  :class:`LowRecallSignal`.
* **Split candidates** -- a block that repeatedly attracts negative / corrected
  feedback is probably trying to answer too much (or the wrong thing): it is a
  candidate to *split* into sharper IdeaBlocks or to re-author. Surfaced as
  :class:`SplitCandidateSignal` (the inverse of the Distill *merge*).

The extractors depend only on the :class:`~sparksage.feedback.store.FeedbackStore`
protocol -- pure stdlib, fully unit-testable with an
:class:`~sparksage.feedback.store.InMemoryFeedbackStore` and zero network calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sparksage.feedback.models import FeedbackRecord
from sparksage.feedback.store import FeedbackStats, FeedbackStore


@dataclass
class LowRecallSignal:
    """A query topic the corpus under-covers.

    Attributes
    ----------
    query:
        The (normalized) query that repeatedly retrieved too little.
    occurrences:
        How many feedback records reported low recall for it.
    avg_block_count:
        Mean number of blocks retrieved across those records.
    """

    query: str
    occurrences: int = 0
    avg_block_count: float = 0.0


@dataclass
class SplitCandidateSignal:
    """A block that should be split or re-authored.

    Attributes
    ----------
    block_id:
        The block attracting the bad feedback.
    stats:
        :class:`~sparksage.feedback.store.FeedbackStats` for that block.
    bad_ratio:
        ``(negative + corrected) / total`` -- the headline re-author signal.
    """

    block_id: str
    stats: FeedbackStats
    bad_ratio: float = 0.0


@dataclass
class HealingReport:
    """The combined self-healing signal set for one feedback store.

    Attributes
    ----------
    low_recall:
        Queries with repeated under-coverage, sorted by occurrences desc.
    split_candidates:
        Blocks with a high bad-feedback ratio, sorted by bad_ratio desc.
    approval:
        Overall approval ratio across all feedback (headline health metric).
    """

    low_recall: list[LowRecallSignal] = field(default_factory=list)
    split_candidates: list[SplitCandidateSignal] = field(default_factory=list)
    approval: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.low_recall and not self.split_candidates


def _normalize_query(query: str) -> str:
    return " ".join(str(query).lower().split())


def extract_low_recall(
    records: list[FeedbackRecord],
    *,
    min_hits: int = 1,
    min_occurrences: int = 2,
) -> list[LowRecallSignal]:
    """Find queries that repeatedly retrieved fewer than ``min_hits`` blocks.

    Groups feedback records by their normalized query and keeps those that (a)
    reported a below-``min_hits`` block count and (b) occurred at least
    ``min_occurrences`` times. Sorted by occurrences descending.
    """
    buckets: dict[str, list[int]] = {}
    for rec in records:
        key = _normalize_query(rec.query)
        if not key:
            continue
        n = len(rec.block_ids)
        if n < min_hits:
            buckets.setdefault(key, []).append(n)
    signals = [
        LowRecallSignal(
            query=key,
            occurrences=len(counts),
            avg_block_count=(sum(counts) / len(counts)) if counts else 0.0,
        )
        for key, counts in buckets.items()
        if len(counts) >= min_occurrences
    ]
    signals.sort(key=lambda s: (-s.occurrences, s.query))
    return signals


def extract_split_candidates(
    block_breakdown: dict[str, FeedbackStats],
    *,
    min_records: int = 2,
    min_bad_ratio: float = 0.5,
) -> list[SplitCandidateSignal]:
    """Find blocks with a high negative/corrected feedback ratio.

    Parameters
    ----------
    block_breakdown:
        The ``{block_id: FeedbackStats}`` mapping from
        :meth:`InMemoryFeedbackStore.block_breakdown`.
    min_records:
        Minimum total feedback records on a block to consider it (avoids
        one-off noise).
    min_bad_ratio:
        ``(negative + corrected) / total`` threshold above which a block is a
        split / re-author candidate.
    """
    signals: list[SplitCandidateSignal] = []
    for bid, stats in block_breakdown.items():
        if stats.total < min_records:
            continue
        bad = stats.negative + stats.corrected
        ratio = bad / stats.total if stats.total else 0.0
        if ratio >= min_bad_ratio:
            signals.append(SplitCandidateSignal(block_id=bid, stats=stats, bad_ratio=ratio))
    signals.sort(key=lambda s: (-s.bad_ratio, -s.stats.total, s.block_id))
    return signals


def extract_healing_signals(
    store: FeedbackStore,
    *,
    min_hits: int = 1,
    min_occurrences: int = 2,
    min_records_per_block: int = 2,
    min_bad_ratio: float = 0.5,
) -> HealingReport:
    """Compute the full :class:`HealingReport` from a feedback store.

    Convenience entry point that pulls the records + block breakdown from
    ``store`` and runs both extractors. The ingest side (e.g. a
    :class:`~sparksage.kb.KnowledgeBase` operator) consumes the report to
    decide what to re-chunk / split / re-author.
    """
    records: list[FeedbackRecord]
    if hasattr(store, "list"):
        records = store.list(limit=10**6)
    else:  # pragma: no cover - defensive
        records = []
    low_recall = extract_low_recall(
        records, min_hits=min_hits, min_occurrences=min_occurrences
    )

    split_candidates: list[SplitCandidateSignal] = []
    approval = 0.0
    if hasattr(store, "block_breakdown"):
        split_candidates = extract_split_candidates(
            store.block_breakdown(),  # type: ignore[attr-defined]
            min_records=min_records_per_block,
            min_bad_ratio=min_bad_ratio,
        )
    if hasattr(store, "stats"):
        approval = store.stats().approval  # type: ignore[attr-defined]

    return HealingReport(
        low_recall=low_recall,
        split_candidates=split_candidates,
        approval=approval,
    )


__all__ = [
    "HealingReport",
    "LowRecallSignal",
    "SplitCandidateSignal",
    "extract_healing_signals",
    "extract_low_recall",
    "extract_split_candidates",
]
