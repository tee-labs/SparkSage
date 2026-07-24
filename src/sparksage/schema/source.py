"""Provenance model: where a chunk came from."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceRef(BaseModel):
    """Pointer back to the original document/location a chunk was built from.

    Provenance is what makes a knowledge base auditable: you can always trace a
    retrieved answer to its source line. Without it, ``trusted_answer`` cannot
    actually be *trusted*.
    """

    model_config = ConfigDict(extra="forbid")

    uri: str = Field(
        ...,
        description="Stable identifier of the source (URL, file path, doc id).",
    )
    title: str | None = Field(default=None, description="Human-readable source title.")
    locator: str | None = Field(
        default=None,
        description="Position within the source (page, line range, anchor, ...).",
    )
    content_hash: str | None = Field(
        default=None,
        description="Hash of the source content slice for change detection.",
    )
    system: str | None = Field(
        default=None,
        description="Originating system (e.g. 'confluence', 'github', 'pdf').",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provenance fields the ingester may carry.",
    )
