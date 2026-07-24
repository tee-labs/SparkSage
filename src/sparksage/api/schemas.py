"""HTTP request/response models for the SparkSage API.

These are plain Pydantic v2 models -- no FastAPI import -- so they are
reusable outside the web layer (e.g. by a CLI or test harness) and stay
consistent with the project's ``ConfigDict(extra="forbid")`` convention.
"""

from __future__ import annotations

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
