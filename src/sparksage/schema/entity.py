"""Entity model: a real-world thing referenced by a knowledge chunk."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sparksage.schema.enums import EntityRelation, EntityType


class Entity(BaseModel):
    """A named real-world thing that a chunk talks about.

    Entities turn unstructured prose into a queryable graph: they power
    metadata filtering, permission scoping, and hybrid retrieval. Keeping the
    type/relation to a controlled vocabulary avoids the "same thing, many
    spellings" problem.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    entity_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Canonical, human-readable name of the entity.",
    )
    entity_type: EntityType = Field(
        ...,
        description="Controlled vocabulary type of the entity.",
    )
    relation: EntityRelation = Field(
        default=EntityRelation.MENTIONS,
        description="How this entity relates to the referencing chunk.",
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternate spellings/names for normalization at query time.",
    )
    external_id: str | None = Field(
        default=None,
        description="Optional stable id in an upstream system (e.g. CRM/Jira).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form provenance/attributes (kept open deliberately).",
    )

    @field_validator("entity_name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("entity_name must not be empty or whitespace")
        return v

    @field_validator("aliases")
    @classmethod
    def _dedupe_aliases(cls, v: list[str]) -> list[str]:
        seen: list[str] = []
        for a in v:
            a = a.strip()
            if a and a not in seen:
                seen.append(a)
        return seen
