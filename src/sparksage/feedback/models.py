"""Feedback models: the user-signal record for the quality flywheel.

:class:`FeedbackRecord` is the answer-side counterpart of
:class:`~sparksage.reader.schema.GeneratedAnswer`: it captures what the user
thought of a surfaced answer -- a thumbs-up/down rating, an optional corrected
answer, and the block ids the answer was built from. This is the raw signal the
Phase-4 quality flywheel consumes: aggregate negative ratings surface answer
quality regressions, repeated low-recall queries flag documents for
re-chunking, and frequently-corrected blocks become split candidates.

Mirrors the established schema conventions: Pydantic v2,
``ConfigDict(extra="forbid")``, a closed enum for the controlled rating
vocabulary.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class FeedbackRating(str, Enum):
    """Closed vocabulary for a user's verdict on a surfaced answer.

    A small, unambiguous set keeps aggregate statistics reliable: the
    :class:`~sparksage.feedback.store.FeedbackStore` rolls these up into an
    approval ratio, and the self-healing extractors branch on ``NEGATIVE`` /
    ``CORRECTED``.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTED = "corrected"


class FeedbackRecord(BaseModel):
    """One user-signal datum: a rating (and optional correction) on an answer.

    Attributes
    ----------
    feedback_id:
        Stable unique id (UUID4 string). Auto-generated when omitted.
    query:
        The user query that produced the answer (the signal key).
    answer_text:
        The surfaced answer text (for re-judging / diffs).
    rating:
        :class:`FeedbackRating` verdict.
    correction:
        Optional user-supplied corrected answer. Present with
        ``CORRECTED`` (and sometimes with ``NEGATIVE``).
    block_ids:
        The block ids the answer was built from (ties the feedback to the
        corpus for self-healing).
    kb_id:
        The knowledge base the answer came from (multi-tenant attribution).
    created_at:
        UTC timestamp.
    metadata:
        Free-form caller metadata (user id, session id, ...).
    """

    model_config = ConfigDict(extra="forbid")

    feedback_id: str = Field(default_factory=_new_id, description="Stable unique id.")
    query: str = Field(..., min_length=1, description="The query that produced the answer.")
    answer_text: str = Field(default="", description="The surfaced answer text.")
    rating: FeedbackRating = Field(..., description="The user's verdict.")
    correction: str | None = Field(default=None, description="Optional corrected answer.")
    block_ids: list[str] = Field(
        default_factory=list, description="Block ids the answer was built from."
    )
    kb_id: str | None = Field(default=None, description="Source knowledge base.")
    created_at: datetime = Field(default_factory=_utcnow, description="UTC timestamp.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Free-form caller metadata."
    )

    @field_validator("query")
    @classmethod
    def _nonempty_query(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("query must not be empty")
        return value

    @field_validator("correction")
    @classmethod
    def _strip_correction(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @property
    def is_negative(self) -> bool:
        """Whether this feedback signals a problem (negative or corrected)."""
        return self.rating in (FeedbackRating.NEGATIVE, FeedbackRating.CORRECTED)


__all__ = ["FeedbackRating", "FeedbackRecord"]
