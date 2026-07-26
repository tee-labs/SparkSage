"""Feedback storage abstraction (:class:`FeedbackStore` protocol).

The persistence layer for the Phase-4 quality flywheel. Where
:class:`~sparksage.documents.store.DocumentStore` stores documents and
:class:`~sparksage.kb.store.KnowledgeBaseStore` stores KB metadata, this stores
:class:`FeedbackRecord`s -- the user signals that feed back into corpus
self-healing. The default :class:`InMemoryFeedbackStore` is a plain list backed
by a dict; a future durable backend implements the same protocol.

The store also owns the *aggregate* roll-ups the self-healing extractors
consume: approval ratio, per-block and per-query breakdowns, and recent-
negative windows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sparksage.feedback.models import FeedbackRating, FeedbackRecord


@runtime_checkable
class FeedbackStore(Protocol):
    """CRUD + aggregation over :class:`FeedbackRecord` instances."""

    def add(self, record: FeedbackRecord) -> FeedbackRecord:
        """Append ``record``; returns the stored record."""
        ...

    def get(self, feedback_id: str) -> FeedbackRecord | None:
        """Return the record for ``feedback_id`` (a copy), or ``None``."""
        ...

    def list(
        self,
        *,
        rating: FeedbackRating | None = None,
        kb_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FeedbackRecord]:
        """Return a slice of records, newest-first, optionally filtered."""
        ...

    def delete(self, feedback_id: str) -> bool:
        """Remove ``feedback_id``. Return whether a record was deleted."""
        ...

    def count(self, *, kb_id: str | None = None) -> int:
        """Number of records, optionally restricted to a KB."""
        ...

    def __len__(self) -> int:
        ...


@dataclass
class FeedbackStats:
    """Aggregate roll-up over a :class:`FeedbackStore` (or a slice).

    Attributes
    ----------
    total:
        Total records in the window.
    positive, negative, corrected:
        Counts per :class:`FeedbackRating`.
    approval:
        ``positive / total`` (``0.0`` when empty). The headline quality signal.
    """

    total: int = 0
    positive: int = 0
    negative: int = 0
    corrected: int = 0

    @property
    def approval(self) -> float:
        return (self.positive / self.total) if self.total else 0.0


class InMemoryFeedbackStore:
    """Dict-backed :class:`FeedbackStore` (tests / single-node demos).

    Records are stored by ``feedback_id`` and served newest-first by
    ``created_at``. Defensive copies are taken on add / get so caller mutation
    cannot corrupt the store.
    """

    def __init__(self) -> None:
        self._records: dict[str, FeedbackRecord] = {}

    def add(self, record: FeedbackRecord) -> FeedbackRecord:
        stored = record.model_copy()
        self._records[stored.feedback_id] = stored
        return stored.model_copy()

    def get(self, feedback_id: str) -> FeedbackRecord | None:
        rec = self._records.get(str(feedback_id))
        return rec.model_copy() if rec is not None else None

    def list(
        self,
        *,
        rating: FeedbackRating | None = None,
        kb_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FeedbackRecord]:
        if limit < 1:
            raise ValueError("limit must be a positive int")
        if offset < 0:
            raise ValueError("offset must be a non-negative int")
        ordered = sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)
        out = [
            r
            for r in ordered
            if (rating is None or r.rating == rating)
            and (kb_id is None or r.kb_id == kb_id)
        ]
        return [r.model_copy() for r in out[offset : offset + limit]]

    def delete(self, feedback_id: str) -> bool:
        return self._records.pop(str(feedback_id), None) is not None

    def count(self, *, kb_id: str | None = None) -> int:
        if kb_id is None:
            return len(self._records)
        return sum(1 for r in self._records.values() if r.kb_id == kb_id)

    def stats(self, *, kb_id: str | None = None) -> FeedbackStats:
        """Aggregate :class:`FeedbackStats` over (a slice of) the store."""
        records = self.list(kb_id=kb_id, limit=len(self._records) or 1) if kb_id else list(
            self._records.values()
        )
        counts = Counter(r.rating for r in records)
        return FeedbackStats(
            total=len(records),
            positive=counts.get(FeedbackRating.POSITIVE, 0),
            negative=counts.get(FeedbackRating.NEGATIVE, 0),
            corrected=counts.get(FeedbackRating.CORRECTED, 0),
        )

    def block_breakdown(self) -> dict[str, FeedbackStats]:
        """Per-block :class:`FeedbackStats` (which blocks attract bad feedback)."""
        per_block: dict[str, list[FeedbackRecord]] = {}
        for rec in self._records.values():
            for bid in rec.block_ids:
                per_block.setdefault(bid, []).append(rec)
        out: dict[str, FeedbackStats] = {}
        for bid, recs in per_block.items():
            counts = Counter(r.rating for r in recs)
            out[bid] = FeedbackStats(
                total=len(recs),
                positive=counts.get(FeedbackRating.POSITIVE, 0),
                negative=counts.get(FeedbackRating.NEGATIVE, 0),
                corrected=counts.get(FeedbackRating.CORRECTED, 0),
            )
        return out

    def __contains__(self, feedback_id: object) -> bool:
        return str(feedback_id) in self._records

    def __len__(self) -> int:
        return len(self._records)


__all__ = [
    "FeedbackStats",
    "FeedbackStore",
    "InMemoryFeedbackStore",
]
