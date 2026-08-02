"""QA conversation-history persistence (the query log the QA page restores).

The Q&A page keeps its conversation in frontend state, which disappears on a
page refresh -- while :class:`~sparksage.feedback.FeedbackRecord`s (the user's
ratings) persist server-side and show up in the feedback statistics. That
asymmetry is the bug this module fixes: :class:`QATurn` is the unit of a
persisted conversation log, and :class:`QASessionStore` is the storage
abstraction behind it. Recording happens inside
:meth:`~sparksage.api.qa_service.QAService.ask`, so every asked question --
and its answer, with the full serialized answer payload -- survives restarts
and can be re-rendered by the UI without re-running the pipeline.

Mirrors the established schema conventions: Pydantic v2,
``ConfigDict(extra="forbid")``, a closed ``TurnRole`` enum, defensive copies on
read / write.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class TurnRole(str, Enum):
    """Closed vocabulary for who produced a :class:`QATurn`."""

    USER = "user"
    ASSISTANT = "assistant"


class QATurn(BaseModel):
    """One entry in the QA conversation history: a question or an answer.

    Attributes
    ----------
    turn_id:
        Stable unique id (UUID4 string). Auto-generated when omitted.
    role:
        :class:`TurnRole` -- the user question or the assistant answer.
    content:
        The user query (``role=user``) or the surfaced answer text
        (``role=assistant``). May be empty when the answer abstained.
    kb_id:
        The knowledge base the turn belongs to (multi-tenant attribution).
    result:
        The serialized answer payload (the HTTP ``AskResponse`` shape) for
        assistant turns, so a UI can re-render citations / retrieved chunks /
        confidence without re-running the pipeline. ``None`` for user turns.
    created_at:
        UTC timestamp.
    """

    model_config = ConfigDict(extra="forbid")

    turn_id: str = Field(default_factory=_new_id, description="Stable unique id.")
    role: TurnRole = Field(..., description="Who produced this turn.")
    content: str = Field(default="", description="The user query or answer text.")
    kb_id: str | None = Field(default=None, description="Source knowledge base.")
    result: dict[str, Any] | None = Field(
        default=None,
        description="Serialized answer payload (assistant turns only).",
    )
    created_at: datetime = Field(default_factory=_utcnow, description="UTC timestamp.")

    @field_validator("content")
    @classmethod
    def _strip_content(cls, value: str) -> str:
        return (value or "").strip()

    @property
    def query(self) -> str | None:
        """The user query this assistant turn answered (``None`` for user turns)."""
        if self.result and isinstance(self.result, dict):
            q = self.result.get("query")
            return str(q) if q else None
        return None


@runtime_checkable
class QASessionStore(Protocol):
    """CRUD over :class:`QATurn` instances (the persisted conversation log)."""

    def add_turn(self, record: QATurn) -> QATurn:
        """Append ``record``; returns the stored record."""
        ...

    def list(
        self,
        *,
        kb_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[QATurn]:
        """Return a slice of turns, newest-first, optionally filtered by KB."""
        ...

    def count(self, *, kb_id: str | None = None) -> int:
        """Number of turns, optionally restricted to a KB."""
        ...

    def clear(self, *, kb_id: str | None = None) -> None:
        """Remove turns (optionally only those belonging to ``kb_id``)."""
        ...

    def __len__(self) -> int:
        ...


class InMemoryQASessionStore:
    """Dict-backed :class:`QASessionStore` (tests / single-node demos).

    Turns are stored by ``turn_id`` and served newest-first by ``created_at``.
    Defensive copies are taken on add / list so caller mutation cannot corrupt
    the store.
    """

    def __init__(self) -> None:
        self._turns: dict[str, QATurn] = {}

    def add_turn(self, record: QATurn) -> QATurn:
        stored = record.model_copy()
        self._turns[stored.turn_id] = stored
        return stored.model_copy()

    def list(
        self,
        *,
        kb_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[QATurn]:
        if limit < 1:
            raise ValueError("limit must be a positive int")
        if offset < 0:
            raise ValueError("offset must be a non-negative int")
        ordered = sorted(self._turns.values(), key=lambda t: t.created_at, reverse=True)
        filtered = ordered if kb_id is None else [t for t in ordered if t.kb_id == kb_id]
        return [t.model_copy() for t in filtered[offset : offset + limit]]

    def count(self, *, kb_id: str | None = None) -> int:
        if kb_id is None:
            return len(self._turns)
        return sum(1 for t in self._turns.values() if t.kb_id == kb_id)

    def clear(self, *, kb_id: str | None = None) -> None:
        if kb_id is None:
            self._turns.clear()
            return
        for turn_id in [t.turn_id for t in self._turns.values() if t.kb_id == kb_id]:
            self._turns.pop(turn_id, None)

    def __len__(self) -> int:
        return len(self._turns)


__all__ = [
    "InMemoryQASessionStore",
    "QASessionStore",
    "QATurn",
    "TurnRole",
]
