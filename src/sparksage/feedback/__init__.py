"""User feedback + corpus self-healing (the Phase-4 quality flywheel).

This package closes the loop between the query side and the ingest side. Where
the rest of SparkSage flows ingest -> query, this flows query -> ingest:

* :class:`FeedbackRecord` captures what the user thought of a surfaced answer
  (thumbs up / down / corrected).
* :class:`FeedbackStore` persists and aggregates those signals (approval ratio,
  per-block / per-query breakdowns).
* :func:`extract_healing_signals` turns the aggregate back into concrete
  ingest actions -- low-recall queries flag a *coverage gap* (re-chunk / new
  content); blocks with a high bad-feedback ratio become *split candidates*
  (the inverse of the Distill merge).

Everything depends only on the :class:`FeedbackStore` protocol, so it is fully
unit-testable with :class:`InMemoryFeedbackStore` and zero network calls.

Example
-------
::

    from sparksage.feedback import (
        FeedbackRecord, FeedbackRating, InMemoryFeedbackStore,
        extract_healing_signals,
    )

    store = InMemoryFeedbackStore()
    store.add(FeedbackRecord(query="how to deploy", answer_text="...",
                             rating=FeedbackRating.NEGATIVE, block_ids=[...]))
    report = extract_healing_signals(store)
"""

from sparksage.feedback.healing import (
    HealingReport,
    LowRecallSignal,
    SplitCandidateSignal,
    extract_healing_signals,
    extract_low_recall,
    extract_split_candidates,
)
from sparksage.feedback.models import FeedbackRating, FeedbackRecord
from sparksage.feedback.store import (
    FeedbackStats,
    FeedbackStore,
    InMemoryFeedbackStore,
)

__all__ = [
    "FeedbackRating",
    "FeedbackRecord",
    "FeedbackStats",
    "FeedbackStore",
    "HealingReport",
    "InMemoryFeedbackStore",
    "LowRecallSignal",
    "SplitCandidateSignal",
    "extract_healing_signals",
    "extract_low_recall",
    "extract_split_candidates",
]
