"""The :class:`DocumentRecord`: the document-level entity for tag management.

SparkSage's existing schema jumps straight from a Markdown blob to a list of
:class:`~sparksage.schema.IdeaBlock` chunks; there is no *document* object to
attach a title, summary, tags or lifecycle to. :class:`DocumentRecord` fills that
gap. It is a Pydantic v2 model (``ConfigDict(extra="forbid")``, like every schema
model) carrying exactly the fields an enterprise knowledge-management service
needs:

* ``title`` / ``summary`` / ``body_markdown`` -- the parsed document content.
* ``tags`` -- **free-form** ``list[str]`` (not the closed
  :class:`~sparksage.schema.enums.Tag` enum, which keeps its coarse-grained
  semantic-filtering role). This is where algorithm-extracted / user-supplied
  business tags (``"k8s"``, ``"报销流程"``, ``"Q3财报"``) live.
* ``source`` -- a :class:`~sparksage.schema.source.SourceRef` for provenance.
* ``created_at`` / ``updated_at`` / ``content_hash`` -- lifecycle / change
  detection.

A :class:`DocumentRecord` is what a :class:`~sparksage.documents.store.DocumentStore`
saves/lists/queries, and what the ``/api/v1/documents`` route serializes.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sparksage.schema.source import SourceRef


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


def content_hash_of(text: str) -> str:
    """Return a stable SHA-256 hex digest of ``text`` (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DocumentRecord(BaseModel):
    """A single managed document: parsed content + free-form tags + provenance.

    Attributes
    ----------
    doc_id:
        Stable unique id (UUID4 string). Auto-generated when omitted.
    title:
        Document title (extracted from the source or set by the caller). May be
        ``None`` when no title could be inferred.
    summary:
        Document-level summary. Extractive by default (see
        :class:`~sparksage.documents.summarizer.ExtractiveSummarizer`); may be
        LLM-generated or hand-written. ``None`` until produced.
    body_markdown:
        The full parsed Markdown body -- the canonical document content.
    tags:
        Free-form tag list (business / topic labels). De-duplicated on
        validation, order preserved. Not the closed
        :class:`~sparksage.schema.enums.Tag` enum.
    source:
        :class:`~sparksage.schema.source.SourceRef` provenance (URI / title /
        locator / ...).
    created_at, updated_at:
        UTC timestamps. ``updated_at`` is refreshed on every store write.
    content_hash:
        SHA-256 of the body, for cheap change detection. Auto-computed from
        ``body_markdown`` when omitted.
    metadata:
        Free-form ``dict`` for caller-specific fields (author, department,
        ACL, ...).
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(default_factory=_new_id, description="Stable unique id.")
    title: str | None = Field(default=None, description="Document title, if known.")
    summary: str | None = Field(
        default=None, description="Document-level summary (extractive / LLM)."
    )
    body_markdown: str = Field(..., description="Full parsed Markdown body.")
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form business / topic tags (de-duplicated).",
    )
    source: SourceRef = Field(..., description="Provenance of the document.")
    created_at: datetime = Field(default_factory=_utcnow, description="Creation time (UTC).")
    updated_at: datetime = Field(default_factory=_utcnow, description="Last write time (UTC).")
    content_hash: str | None = Field(
        default=None, description="SHA-256 of the body for change detection."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Free-form caller-specific metadata."
    )

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

    @field_validator("doc_id")
    @classmethod
    def _nonempty_doc_id(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("doc_id must be a non-empty string")
        return str(value)

    def model_post_init(self, __context: Any) -> None:
        if self.content_hash is None and self.body_markdown is not None:
            object.__setattr__(self, "content_hash", content_hash_of(self.body_markdown))


def new_record(
    *,
    body_markdown: str,
    source: SourceRef | str,
    title: str | None = None,
    summary: str | None = None,
    tags: list[str] | None = None,
    doc_id: str | None = None,
    content_hash: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> DocumentRecord:
    """Convenience factory for a :class:`DocumentRecord`.

    ``source`` accepts a :class:`SourceRef` or a bare URI string (wrapped in a
    minimal :class:`SourceRef`). ``content_hash`` defaults to the SHA-256 of the
    body when omitted.
    """
    src = source if isinstance(source, SourceRef) else SourceRef(uri=str(source))
    return DocumentRecord(
        doc_id=doc_id if doc_id is not None else _new_id(),
        title=title,
        summary=summary,
        body_markdown=body_markdown,
        tags=list(tags) if tags is not None else [],
        source=src,
        content_hash=content_hash,
        metadata=dict(metadata) if metadata is not None else {},
    )


__all__ = [
    "DocumentRecord",
    "content_hash_of",
    "new_record",
]
