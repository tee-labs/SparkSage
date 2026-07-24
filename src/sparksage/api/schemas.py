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
from sparksage.documents.schema import Document
from sparksage.schema.enums import TagSource


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
# Document management (knowledge base)
# ---------------------------------------------------------------------------- #
class DocumentOut(BaseModel):
    """Response body for a single knowledge document.

    A flat, JSON-safe projection of a stored :class:`~sparksage.documents.Document`.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable UUID of the document.")
    title: str | None = Field(default=None, description="Document title.")
    summary: str | None = Field(default=None, description="Auto-extracted summary.")
    content: str = Field(description="Document body as Markdown.")
    tags: list[str] = Field(default_factory=list, description="Document tags.")
    tag_source: TagSource = Field(description="Origin of the current tag set.")
    source: SourceInfo = Field(description="Provenance of the uploaded document.")
    language: str = Field(description="BCP-47 language code of the content.")
    version: int = Field(default=1, ge=1, description="Monotonic content version.")
    created_at: datetime = Field(description="Creation timestamp (UTC).")
    updated_at: datetime = Field(description="Last edit timestamp (UTC).")


class DocumentCreateResponse(BaseModel):
    """Response body for ``POST /api/v1/documents``."""

    model_config = ConfigDict(extra="forbid")

    document: DocumentOut = Field(description="The newly created document.")


class DocumentListResponse(BaseModel):
    """Response body for ``GET /api/v1/documents``."""

    model_config = ConfigDict(extra="forbid")

    items: list[DocumentOut] = Field(default_factory=list)
    count: int = Field(default=0, description="Number of items in this page.")
    total: int = Field(default=0, description="Total documents matching the query.")
    tag: str | None = Field(default=None, description="Active tag filter, if any.")


class DocumentUpdateRequest(BaseModel):
    """Request body for ``PATCH /api/v1/documents/{id}``.

    All fields are optional; only provided fields are changed. Supplying
    ``tags`` replaces the tag set wholesale and records it as user-sourced.
    """

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=300)
    summary: str | None = Field(default=None, max_length=500)
    tags: list[str] | None = Field(default=None)


class AddTagsRequest(BaseModel):
    """Request body for ``POST /api/v1/documents/{id}/tags``."""

    model_config = ConfigDict(extra="forbid")

    tags: list[str] = Field(..., min_length=1, description="Tags to merge in.")


class AutoTagRequest(BaseModel):
    """Request body for ``POST /api/v1/documents/{id}/tags:auto``."""

    model_config = ConfigDict(extra="forbid")

    top_n: int = Field(default=5, ge=1, le=32, description="Max tags to generate.")
    replace_existing: bool = Field(
        default=False,
        description="Replace the tag set instead of merging onto it.",
    )


class TagCount(BaseModel):
    """A tag and how many documents carry it."""

    model_config = ConfigDict(extra="forbid")

    tag: str
    count: int


class TagsResponse(BaseModel):
    """Response body for ``GET /api/v1/tags``."""

    model_config = ConfigDict(extra="forbid")

    tags: list[TagCount] = Field(default_factory=list)
    total: int = Field(default=0, description="Number of distinct tags.")


def to_document_out(document: Document) -> DocumentOut:
    """Project a stored :class:`Document` to its JSON response form."""
    source = document.source
    src_uri = source.uri if source else None
    src_title = source.title if source else None
    return DocumentOut(
        id=str(document.id),
        title=document.title,
        summary=document.summary,
        content=document.content,
        tags=list(document.tags),
        tag_source=document.tag_source,
        source=SourceInfo(uri=src_uri, title=src_title),
        language=document.language,
        version=document.version,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def to_document_list_response(
    items: list[Document], *, total: int, tag: str | None
) -> DocumentListResponse:
    """Build a :class:`DocumentListResponse` from a page of documents."""
    return DocumentListResponse(
        items=[to_document_out(d) for d in items],
        count=len(items),
        total=total,
        tag=tag,
    )


def to_tags_response(counts: dict[str, int]) -> TagsResponse:
    """Build a :class:`TagsResponse` from a ``{tag: count}`` mapping."""
    tag_counts = [TagCount(tag=t, count=c) for t, c in counts.items()]
    return TagsResponse(tags=tag_counts, total=len(tag_counts))
