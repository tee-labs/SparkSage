"""Lenient intermediate models + enum coercion for LLM-produced block data.

LLMs are unpredictable: they emit arbitrary tag spellings, forget the ``?`` on a
question, or wrap output in prose. So we never feed raw model output straight
into :class:`IdeaBlock` (which is strict by design -- ``extra="forbid"``).

Instead, the model output is parsed into the *lenient* models here (everything
is a plain string, extras ignored), then :func:`coerce_block` normalizes and
maps it through the controlled vocabularies before constructing a real
:class:`IdeaBlock`. This keeps the strict schema as the single source of truth
while staying robust to messy model output.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sparksage.schema.entity import Entity
from sparksage.schema.enums import EntityType, Tag
from sparksage.schema.ideablock import RECOMMENDED_ANSWER_MAX, IdeaBlock
from sparksage.schema.source import SourceRef


class RawEntity(BaseModel):
    """Lenient entity as emitted by an LLM (strings, not enums)."""

    model_config = ConfigDict(extra="ignore")

    entity_name: str = ""
    entity_type: str = "CONCEPT"
    aliases: list[str] = Field(default_factory=list)


class RawIdeaBlock(BaseModel):
    """Lenient block as emitted by an LLM (strings, not enums)."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    critical_question: str = ""
    trusted_answer: str = ""
    tags: list[str] = Field(default_factory=list)
    entities: list[RawEntity] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class RawGenerationResult(BaseModel):
    """Top-level lenient envelope: a list of raw blocks."""

    model_config = ConfigDict(extra="ignore")

    blocks: list[RawIdeaBlock] = Field(default_factory=list)


#: Mapping used when an unknown entity type needs a safe fallback.
DEFAULT_ENTITY_TYPE = EntityType.CONCEPT


def _map_tag(raw: str, *, strict: bool) -> Tag | None:
    """Map a raw tag string to a :class:`Tag`; ``None`` if unknown.

    Case-insensitive. In ``strict`` mode an unknown tag raises ``ValueError``.
    """
    value = raw.strip()
    if not value:
        return None
    needle = value.upper()
    for member in Tag:
        if member.value.upper() == needle:
            return member
    if strict:
        raise CoercionError(f"unknown tag: {raw!r}")
    return None


def _map_entity_type(raw: str, *, strict: bool) -> EntityType:
    """Map a raw entity-type string to :class:`EntityType`.

    Falls back to :data:`DEFAULT_ENTITY_TYPE` in non-strict mode so the entity
    (whose name is still valuable) is retained rather than dropped.
    """
    value = (raw or "").strip()
    if value:
        needle = value.upper()
        for member in EntityType:
            if member.value.upper() == needle:
                return member
    if strict:
        raise CoercionError(f"unknown entity_type: {raw!r}")
    return DEFAULT_ENTITY_TYPE


def _ensure_question(text: str) -> str:
    """Guarantee the question ends with a ``?`` (light, deterministic repair)."""
    stripped = text.strip().rstrip(".!。")
    if not stripped:
        return text
    if stripped.endswith(("?", "？")):
        return text.strip()
    return stripped + "?"


class CoercionError(ValueError):
    """Raised when a raw block cannot be coerced into a valid IdeaBlock."""


def coerce_block(
    raw: RawIdeaBlock,
    *,
    strict: bool,
    source: SourceRef | None = None,
    language: str = "en",
) -> IdeaBlock:
    """Normalize a :class:`RawIdeaBlock` into a strict :class:`IdeaBlock`.

    - maps ``tags`` / ``entity_type`` strings through the controlled vocabularies
      (unknown tags dropped unless ``strict``; unknown entity types fall back to
      ``CONCEPT`` unless ``strict``);
    - ensures ``critical_question`` ends with ``?`` (non-strict only);
    - attaches provenance (``source``) and ``language`` to every block;
    - raises :class:`CoercionError` (wrapping the pydantic detail) if the
      normalized block still fails validation -- e.g. an oversized answer.
    """
    if not isinstance(raw, RawIdeaBlock):
        raise CoercionError("expected a RawIdeaBlock")

    question = raw.critical_question.strip()
    answer = raw.trusted_answer.strip()
    name = raw.name.strip()
    if not strict:
        question = _ensure_question(question)

    if not name:
        raise CoercionError("block is missing a 'name'")
    if not question:
        raise CoercionError("block is missing a 'critical_question'")
    if not answer:
        raise CoercionError("block is missing a 'trusted_answer'")
    if len(answer) > RECOMMENDED_ANSWER_MAX:
        if strict:
            raise CoercionError(
                f"trusted_answer is {len(answer)} chars "
                f"(>{RECOMMENDED_ANSWER_MAX}); split into more blocks"
            )
        raise CoercionError(
            f"trusted_answer is {len(answer)} chars (>={RECOMMENDED_ANSWER_MAX}); "
            "split into more blocks instead of truncating"
        )

    tags: list[Tag] = []
    for raw_tag in raw.tags:
        mapped = _map_tag(raw_tag, strict=strict)
        if mapped is not None and mapped not in tags:
            tags.append(mapped)

    entities: list[Entity] = []
    for raw_entity in raw.entities:
        ent_name = raw_entity.entity_name.strip()
        if not ent_name:
            continue
        entities.append(
            Entity(
                entity_name=ent_name,
                entity_type=_map_entity_type(raw_entity.entity_type, strict=strict),
                aliases=raw_entity.aliases,
            )
        )

    keywords = [kw.strip() for kw in raw.keywords if kw and kw.strip()]

    try:
        return IdeaBlock(
            name=name,
            critical_question=question,
            trusted_answer=answer,
            tags=tags,
            entities=entities,
            keywords=keywords,
            source=source,
            language=language,
        )
    except ValidationError as exc:  # pragma: no cover - defensive
        raise CoercionError(str(exc)) from exc


def parse_raw_result(data: Any) -> RawGenerationResult:
    """Validate decoded JSON into :class:`RawGenerationResult`.

    Accepts either ``{"blocks": [...]}`` or a bare ``[...]`` list for
    robustness against models that return a top-level array.
    """
    if isinstance(data, list):
        data = {"blocks": data}
    if not isinstance(data, dict):
        raise CoercionError(f"expected a JSON object or array, got {type(data).__name__}")
    return RawGenerationResult.model_validate(data)
