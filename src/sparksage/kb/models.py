"""KnowledgeBase metadata record.

:class:`KnowledgeBaseInfo` is the serializable metadata for a knowledge base
(id / name / description / language / ACL / timestamps) -- the part you can
persist and list without dragging along the live vector index. The runtime
aggregate root (:class:`~sparksage.kb.knowledge_base.KnowledgeBase`) owns both
this info *and* the live block registry + vector store + retriever.

Mirrors the relationship :class:`~sparksage.documents.DocumentRecord` has to
:class:`~sparksage.documents.DocumentStore`: a Pydantic v2 model
(``ConfigDict(extra="forbid")`` like every schema model) carrying exactly the
fields an enterprise knowledge-management service needs, with free-form
``metadata`` for ACL / department / tenant-specific fields.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class KnowledgeBaseInfo(BaseModel):
    """Serializable metadata for a knowledge base (the multi-tenant entity).

    Attributes
    ----------
    kb_id:
        Stable unique id (UUID4 string). Auto-generated when omitted.
    name:
        Human-readable name. Required.
    description:
        Optional free-text description.
    language:
        Default ISO-639/BCP-47 language code for blocks in this KB.
    tags:
        Free-form labels on the KB itself (not block-level :class:`Tag`s).
    metadata:
        Free-form dict for ACL / tenant / department / ...
    created_at, updated_at:
        UTC timestamps. ``updated_at`` refreshed on every store write.
    """

    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(default_factory=_new_id, description="Stable unique id.")
    name: str = Field(..., min_length=1, description="Human-readable KB name.")
    description: str | None = Field(default=None, description="Free-text description.")
    language: str = Field(
        default="en", min_length=2, max_length=16, description="Default block language."
    )
    tags: list[str] = Field(
        default_factory=list, description="Free-form KB-level labels."
    )
    created_at: datetime = Field(default_factory=_utcnow, description="Creation time (UTC).")
    updated_at: datetime = Field(default_factory=_utcnow, description="Last write time (UTC).")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Free-form caller-specific metadata."
    )

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("name must not be empty")
        return value

    @field_validator("kb_id")
    @classmethod
    def _nonempty_id(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("kb_id must be a non-empty string")
        return str(value)

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for raw in value:
            if raw is None:
                continue
            tag = str(raw).strip()
            if not tag or tag in seen:
                continue
            seen.add(tag)
            out.append(tag)
        return out


__all__ = ["KnowledgeBaseInfo"]
