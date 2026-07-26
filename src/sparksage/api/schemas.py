"""HTTP request/response models for the SparkSage API.

These are plain Pydantic v2 models -- no FastAPI import -- so they are
reusable outside the web layer (e.g. by a CLI or test harness) and stay
consistent with the project's ``ConfigDict(extra="forbid")`` convention.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sparksage.api.pipeline import GenerateOutput


class SourceInfo(BaseModel):
    """Provenance echoed back to the caller."""

    model_config = ConfigDict(extra="forbid")

    uri: str | None = Field(default=None, description="Source URI / filename.")
    title: str | None = Field(default=None, description="Document title, if known.")


class ConvertResponse(BaseModel):
    """Response body for ``POST /api/v1/convert``."""

    model_config = ConfigDict(extra="forbid")

    markdown: str = Field(description="The (optionally cleaned) Markdown text.")
    title: str | None = Field(default=None, description="Document title, if extracted.")
    source: SourceInfo = Field(description="Provenance of the converted document.")
    cleaned: bool = Field(
        default=False,
        description="Whether the text-cleaning pipeline was applied.",
    )


class GenerationStatsOut(BaseModel):
    """Diagnostic counters from block generation."""

    model_config = ConfigDict(extra="forbid")

    raw_block_count: int = Field(default=0, description="Blocks the LLM produced.")
    emitted: int = Field(default=0, description="Blocks that passed validation.")
    skipped: int = Field(default=0, description="Blocks dropped as invalid.")
    errors: list[str] = Field(
        default_factory=list,
        description="Per-block coercion errors (non-strict mode).",
    )


class GenerateResponse(BaseModel):
    """Response body for ``POST /api/v1/generate``."""

    model_config = ConfigDict(extra="forbid")

    blocks: list[dict[str, Any]] = Field(
        description=(
            "Generated IdeaBlocks serialized as flat JSON dicts "
            "(via ``IdeaBlock.model_dump(mode='json')``)."
        )
    )
    title: str | None = Field(default=None, description="Document title, if extracted.")
    source: SourceInfo = Field(description="Provenance of the source document.")
    cleaned: bool = Field(
        default=True,
        description="Whether the text-cleaning pipeline was applied.",
    )
    stats: GenerationStatsOut | None = Field(
        default=None,
        description="Generation diagnostics (present when ``with_stats`` was set).",
    )


class HealthResponse(BaseModel):
    """Response body for ``GET /api/v1/health``."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="ok")
    version: str = Field(description="SparkSage library version.")
    generator_configured: bool = Field(
        description="Whether block generation is available on this instance."
    )
    converter_configured: bool = Field(
        description="Whether file conversion is available on this instance.",
        default=True,
    )


def to_convert_response(out: object) -> ConvertResponse:
    """Build a :class:`ConvertResponse` from a service :class:`ConvertOutput`."""
    from sparksage.api.pipeline import ConvertOutput  # local to avoid cycle

    assert isinstance(out, ConvertOutput)
    return ConvertResponse(
        markdown=out.markdown,
        title=out.title,
        source=SourceInfo(uri=out.source.uri, title=out.source.title),
        cleaned=out.cleaned,
    )


def to_generate_response(out: GenerateOutput) -> GenerateResponse:
    """Build a :class:`GenerateResponse` from a service :class:`GenerateOutput`."""
    from sparksage.api.pipeline import _block_to_dict

    stats = None
    if out.stats is not None:
        stats = GenerationStatsOut(
            raw_block_count=out.stats.raw_block_count,
            emitted=out.stats.emitted,
            skipped=out.stats.skipped,
            errors=list(out.stats.errors),
        )
    return GenerateResponse(
        blocks=[_block_to_dict(b) for b in out.blocks],
        title=out.title,
        source=SourceInfo(uri=out.source.uri, title=out.source.title),
        cleaned=out.cleaned,
        stats=stats,
    )


# ---------------------------------------------------------------------------- #
# Document management
# ---------------------------------------------------------------------------- #
class DocumentSourceInfo(BaseModel):
    """Provenance of a stored document."""

    model_config = ConfigDict(extra="forbid")

    uri: str | None = Field(default=None, description="Source URI / filename.")
    title: str | None = Field(default=None, description="Document title, if known.")
    locator: str | None = Field(default=None, description="Position within source.")
    system: str | None = Field(default=None, description="Originating system.")


class DocumentResponse(BaseModel):
    """Full representation of a stored document (detail / create / update)."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(description="Stable unique id.")
    title: str | None = Field(default=None, description="Document title, if known.")
    summary: str | None = Field(default=None, description="Document-level summary.")
    body_markdown: str = Field(description="Full parsed Markdown body.")
    tags: list[str] = Field(default_factory=list, description="Free-form tags.")
    source: DocumentSourceInfo = Field(description="Provenance of the document.")
    created_at: datetime = Field(description="Creation time (UTC, ISO 8601).")
    updated_at: datetime = Field(description="Last write time (UTC, ISO 8601).")
    content_hash: str | None = Field(
        default=None, description="SHA-256 of the body for change detection."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Free-form caller metadata."
    )


class DocumentSummary(BaseModel):
    """Lightweight document row for list responses (no ``body_markdown``)."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(description="Stable unique id.")
    title: str | None = Field(default=None, description="Document title, if known.")
    summary: str | None = Field(default=None, description="Document-level summary.")
    tags: list[str] = Field(default_factory=list, description="Free-form tags.")
    source: DocumentSourceInfo = Field(description="Provenance of the document.")
    created_at: datetime = Field(description="Creation time (UTC, ISO 8601).")
    updated_at: datetime = Field(description="Last write time (UTC, ISO 8601).")
    content_hash: str | None = Field(
        default=None, description="SHA-256 of the body for change detection."
    )


class DocumentListResponse(BaseModel):
    """Paginated document listing."""

    model_config = ConfigDict(extra="forbid")

    items: list[DocumentSummary] = Field(description="The current page of documents.")
    count: int = Field(description="Number of items in this page.")
    total: int = Field(description="Total documents matching the filter.")
    tag: str | None = Field(default=None, description="Active tag filter, if any.")
    q: str | None = Field(default=None, description="Active text query, if any.")
    limit: int = Field(description="Page size used.")
    offset: int = Field(description="Offset used.")


class TagsResponse(BaseModel):
    """The distinct tag vocabulary across stored documents."""

    model_config = ConfigDict(extra="forbid")

    tags: list[str] = Field(description="Distinct tags, sorted ascending.")


class DocumentUpdateRequest(BaseModel):
    """JSON body for ``PATCH /api/v1/documents/{doc_id}``."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, description="New title.")
    tags: list[str] | None = Field(
        default=None, description="Replacement tag list (when supplied)."
    )
    summary: str | None = Field(default=None, description="New summary.")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Replacement metadata (when supplied)."
    )


class RetagRequest(BaseModel):
    """JSON body for ``POST /api/v1/documents/{doc_id}/tags``."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=8, ge=1, description="Number of tags to extract.")
    replace: bool = Field(
        default=True, description="Replace existing tags (True) or append (False)."
    )
    extra_tags: list[str] | None = Field(
        default=None, description="Additional tags to merge in."
    )


def _to_source_info(source: object) -> DocumentSourceInfo:
    return DocumentSourceInfo(
        uri=getattr(source, "uri", None),
        title=getattr(source, "title", None),
        locator=getattr(source, "locator", None),
        system=getattr(source, "system", None),
    )


def to_document_response(record: object) -> DocumentResponse:
    """Build a :class:`DocumentResponse` from a :class:`DocumentRecord`."""
    return DocumentResponse(
        doc_id=record.doc_id,
        title=record.title,
        summary=record.summary,
        body_markdown=record.body_markdown,
        tags=list(record.tags),
        source=_to_source_info(record.source),
        created_at=record.created_at,
        updated_at=record.updated_at,
        content_hash=record.content_hash,
        metadata=dict(record.metadata),
    )


def to_document_summary(record: object) -> DocumentSummary:
    """Build a :class:`DocumentSummary` from a :class:`DocumentRecord`."""
    return DocumentSummary(
        doc_id=record.doc_id,
        title=record.title,
        summary=record.summary,
        tags=list(record.tags),
        source=_to_source_info(record.source),
        created_at=record.created_at,
        updated_at=record.updated_at,
        content_hash=record.content_hash,
    )


def to_document_list_response(
    items: list[object],
    *,
    total: int,
    tag: str | None,
    q: str | None,
    limit: int,
    offset: int,
) -> DocumentListResponse:
    """Build a :class:`DocumentListResponse` from a page of records."""
    summaries = [to_document_summary(r) for r in items]
    return DocumentListResponse(
        items=summaries,
        count=len(summaries),
        total=total,
        tag=tag,
        q=q,
        limit=limit,
        offset=offset,
    )
