"""Cleaning-rule storage abstraction (:class:`CleaningRuleStore` protocol).

The persistence layer for custom cleaning rules -- the cleaning counterpart of
:class:`~sparksage.documents.store.DocumentStore` /
:class:`~sparksage.feedback.store.FeedbackStore`. Where those store documents /
feedback signals, this stores :class:`~sparksage.clean.models.CleaningRuleRecord`
instances: the source-code + routing definitions that get rebuilt into the live
:class:`~sparksage.clean.cleaner.TextCleaner` on load.

The default :class:`InMemoryCleaningRuleStore` is an ordered list backed by a
dict; a durable backend (e.g.
:class:`~sparksage.clean.backends.sqlite.SqliteCleaningRuleStore`) implements
the same protocol so rules survive a restart.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sparksage.clean.models import CleaningRuleRecord


@runtime_checkable
class CleaningRuleStore(Protocol):
    """Ordered CRUD over :class:`CleaningRuleRecord` instances."""

    def add(self, record: CleaningRuleRecord) -> CleaningRuleRecord:
        """Insert ``record``; returns the stored record."""
        ...

    def get(self, rule_id: str) -> CleaningRuleRecord | None:
        """Return the record for ``rule_id`` (a copy), or ``None``."""
        ...

    def list(self, *, limit: int = 100, offset: int = 0) -> list[CleaningRuleRecord]:
        """Return a slice of records in application order (insertion / updated)."""
        ...

    def update(self, record: CleaningRuleRecord) -> CleaningRuleRecord:
        """Replace the stored record for ``record.rule_id``. Raises if absent."""
        ...

    def delete(self, rule_id: str) -> bool:
        """Remove ``rule_id``. Return whether a record was deleted."""
        ...

    def count(self) -> int:
        """Number of stored records."""
        ...

    def __contains__(self, rule_id: object) -> bool:
        ...

    def __len__(self) -> int:
        ...


class InMemoryCleaningRuleStore:
    """Dict-backed :class:`CleaningRuleStore` (tests / single-node demos).

    Records are stored by ``rule_id`` and served in insertion order (the order
    the registry applies them in). Defensive copies are taken on add / get /
    update so caller mutation cannot corrupt the store.
    """

    def __init__(self) -> None:
        self._records: dict[str, CleaningRuleRecord] = {}

    def add(self, record: CleaningRuleRecord) -> CleaningRuleRecord:
        stored = record.model_copy()
        self._records[stored.rule_id] = stored
        return stored.model_copy()

    def get(self, rule_id: str) -> CleaningRuleRecord | None:
        rec = self._records.get(str(rule_id))
        return rec.model_copy() if rec is not None else None

    def list(self, *, limit: int = 100, offset: int = 0) -> list[CleaningRuleRecord]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive int")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a non-negative int")
        ordered = list(self._records.values())
        return [r.model_copy() for r in ordered[offset : offset + limit]]

    def update(self, record: CleaningRuleRecord) -> CleaningRuleRecord:
        rid = record.rule_id
        if rid not in self._records:
            raise KeyError(f"cleaning rule not found: {rid}")
        stored = record.model_copy()
        self._records[rid] = stored
        return stored.model_copy()

    def delete(self, rule_id: str) -> bool:
        return self._records.pop(str(rule_id), None) is not None

    def count(self) -> int:
        return len(self._records)

    def __contains__(self, rule_id: object) -> bool:
        return str(rule_id) in self._records

    def __len__(self) -> int:
        return len(self._records)


__all__ = ["CleaningRuleStore", "InMemoryCleaningRuleStore"]
