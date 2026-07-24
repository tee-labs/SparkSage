"""The :class:`Document`: an enterprise knowledge-management record.

SparkSage's core data unit is the question-aligned :class:`IdeaBlock`, but a
knowledge-management platform works at the *document* granularity: users upload
a Markdown document, the platform extracts its title / summary / body, attaches
tags, stores it, and exposes CRUD over it. :class:`Document` is that record.

Design notes:

* **Tags are free-form strings**, not the controlled :class:`Tag` enum. The KM
  workflow lets users tag freely *and* lets the system auto-generate tags from
  content (keyword extraction). The :class:`~sparksage.schema.enums.TagSource`
  field records which path produced the current tag set.
* Like every schema model it uses ``ConfigDict(extra="forbid")`` to fail fast on
  typos, and carries provenance (:class:`SourceRef`) plus lifecycle timestamps.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sparksage.schema.enums import TagSource
from sparksage.schema.source import SourceRef

#: Soft cap (in characters) for the auto-extracted ``summary``.
SUMMARY_MAX = 500

#: Hard cap (in characters) for the document ``title``.
TITLE_MAX = 300

#: Maximum number of tags a single document may carry.
TAGS_MAX = 32

#: Maximum length of a single tag string.
TAG_MAX = 64


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(BaseModel):
    """A single stored knowledge document with title, summary, body and tags."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    # --- identity ----------------------------------------------------------
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Stable unique id of the document.",
    )

    # --- content -----------------------------------------------------------
    title: str | None = Field(
        default=None,
        max_length=TITLE_MAX,
        description="Document title (auto-extracted from the first H1 when absent).",
    )
    summary: str | None = Field(
        default=None,
        max_length=SUMMARY_MAX,
        description="Short summary (auto-extracted from the first paragraph).",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The document body as Markdown (the canonical payload).",
    )

    # --- classification ----------------------------------------------------
    tags: list[str] = Field(
        default_factory=list,
        max_length=TAGS_MAX,
        description="Free-form tags for filtering and classification.",
    )
    tag_source: TagSource = Field(
        default=TagSource.AUTO,
        description="Whether tags came from the user, auto-extraction, or both.",
    )

    # --- provenance & bookkeeping -----------------------------------------
    source: SourceRef | None = Field(
        default=None,
        description="Where the document was uploaded from (auditability).",
    )
    language: str = Field(
        default="en",
        min_length=2,
        max_length=16,
        description="ISO-639/BCP-47 language code of the document content.",
    )
    version: int = Field(default=1, ge=1, description="Monotonic content version.")

    # --- timestamps --------------------------------------------------------
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    # ------------------------------------------------------------------ #
    # validators
    # ------------------------------------------------------------------ #
    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @field_validator("summary")
    @classmethod
    def _strip_summary(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @field_validator("content")
    @classmethod
    def _strip_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("content must not be empty or whitespace")
        return v

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for tag in v:
            tag = tag.strip()
            if not tag:
                continue
            if len(tag) > TAG_MAX:
                raise ValueError(
                    f"tag '{tag[:16]}…' is {len(tag)} chars; max is {TAG_MAX}."
                )
            lowered = {t.lower() for t in out}
            if tag.lower() not in lowered:
                out.append(tag)
        return out

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def touch(self) -> Document:
        """Bump ``updated_at`` and ``version`` after an edit."""
        self.updated_at = _utcnow()
        self.version += 1
        return self

    def add_tags(self, tags: list[str], *, source: TagSource) -> Document:
        """Merge ``tags`` in, updating :attr:`tag_source` and bumping the version.

        Existing user-provided tags are always preserved; the incoming tags are
        de-duplicated against them. ``tag_source`` becomes ``MIXED`` when a user
        tag set is         extended by auto-extracted tags (or vice-versa).
        """
        existing = list(self.tags)
        merged = existing[:]
        for tag in tags:
            tag = tag.strip()
            if tag and tag.lower() not in {t.lower() for t in merged}:
                merged.append(tag)

        changed = merged != existing
        if not changed and source == self.tag_source:
            return self

        self.tags = merged
        if source != self.tag_source:
            self.tag_source = TagSource.MIXED
        return self.touch()

    def replace_tags(self, tags: list[str], *, source: TagSource) -> Document:
        """Overwrite the tag set wholesale and record ``source``."""
        self.tags = tags
        self.tag_source = source
        return self.touch()
