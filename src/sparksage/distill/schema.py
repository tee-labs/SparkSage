"""Lenient intermediate model + coercion for LLM-produced *merged* blocks.

Mirrors the established two-stage pattern of :mod:`sparksage.generator.schema`:
raw merge-model output is parsed into the *lenient* :class:`RawMergedBlock`
(plain strings, ``extra="ignore"``), then coerced through the controlled
vocabularies (:class:`Tag`, :class:`EntityType`) into a strict
:class:`~sparksage.schema.IdeaBlock` whose lifecycle fields are set for Distill:

* :attr:`~sparksage.schema.IdeaBlock.status` -> :attr:`BlockStatus.ACTIVE`
  (the canonical, merged block is live);
* :attr:`~sparksage.schema.IdeaBlock.parents` -> the UUIDs of every block merged
  into it (the provenance chain);
* :attr:`~sparksage.schema.IdeaBlock.confidence` -> the cluster's mean pairwise
  similarity (set by the caller, not the model).

The tag/entity-type mapping helpers are reused verbatim from the generator so the
controlled vocabularies stay the single source of truth.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sparksage.generator.schema import RawEntity, _map_entity_type, _map_tag
from sparksage.schema.entity import Entity
from sparksage.schema.enums import BlockStatus
from sparksage.schema.ideablock import RECOMMENDED_ANSWER_MAX, IdeaBlock
from sparksage.schema.source import SourceRef


class RawMergedBlock(BaseModel):
    """Lenient merged-block payload as emitted by the merge LLM.

    Same shape as the generator's :class:`RawIdeaBlock`, plus a ``reasoning``
    field the model may emit to justify the merge (kept for diagnostics, not
    written to the block).
    """

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    critical_question: str = ""
    trusted_answer: str = ""
    tags: list[str] = Field(default_factory=list)
    entities: list[RawEntity] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    reasoning: str = ""


class MergeCoercionError(ValueError):
    """Raised when a raw merged block cannot be coerced into a valid IdeaBlock."""


def _ensure_question(text: str) -> str:
    stripped = text.strip().rstrip(".!。")
    if not stripped:
        return text
    if stripped.endswith(("?", "？")):
        return text.strip()
    return stripped + "?"


def coerce_merged_block(
    raw: RawMergedBlock,
    *,
    parents: list[uuid.UUID],
    confidence: float,
    source: SourceRef | None = None,
    language: str = "en",
    strict: bool = False,
) -> IdeaBlock:
    """Normalize a :class:`RawMergedBlock` into a canonical, merged :class:`IdeaBlock`.

    - maps ``tags`` / ``entity_type`` strings through the controlled vocabularies
      (reusing the generator's mapping helpers so vocabularies cannot drift);
    - ensures ``critical_question`` ends with ``?`` (non-strict only);
    - sets the Distill lifecycle fields: ``status=ACTIVE``, ``parents``, and
      ``confidence`` (clamped to ``[0, 1]``);
    - attaches provenance (``source``) and ``language``.

    Raises :class:`MergeCoercionError` if the normalized block still fails
    validation (e.g. an oversized merged answer -- the model was asked to
    *compress*, so this is a real failure worth surfacing).
    """
    if not isinstance(raw, RawMergedBlock):
        raise MergeCoercionError("expected a RawMergedBlock")

    question = raw.critical_question.strip()
    answer = raw.trusted_answer.strip()
    name = raw.name.strip()
    if not strict:
        question = _ensure_question(question)

    if not name:
        raise MergeCoercionError("merged block is missing a 'name'")
    if not question:
        raise MergeCoercionError("merged block is missing a 'critical_question'")
    if not answer:
        raise MergeCoercionError("merged block is missing a 'trusted_answer'")
    if len(answer) > RECOMMENDED_ANSWER_MAX:
        raise MergeCoercionError(
            f"merged trusted_answer is {len(answer)} chars (>{RECOMMENDED_ANSWER_MAX}); "
            "the model should compress the cluster into one concise answer"
        )

    tags = []
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

    if not 0.0 <= confidence <= 1.0:
        confidence = max(0.0, min(1.0, float(confidence)))

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
            status=BlockStatus.ACTIVE,
            parents=list(parents),
            confidence=confidence,
        )
    except ValidationError as exc:
        raise MergeCoercionError(str(exc)) from exc


def parse_raw_merged(data: Any) -> RawMergedBlock:
    """Validate decoded JSON into a :class:`RawMergedBlock`.

    Accepts the merge object directly, or an envelope ``{"block": {...}}`` /
    ``{"merged": {...}}`` for robustness against prompt drift.
    """
    if isinstance(data, dict) and "block" in data and isinstance(data["block"], dict):
        data = data["block"]
    elif (
        isinstance(data, dict)
        and "merged" in data
        and isinstance(data["merged"], dict)
    ):
        data = data["merged"]
    if not isinstance(data, dict):
        raise MergeCoercionError(
            f"expected a JSON object, got {type(data).__name__}"
        )
    try:
        return RawMergedBlock.model_validate(data)
    except ValidationError as exc:
        raise MergeCoercionError(str(exc)) from exc


__all__ = [
    "MergeCoercionError",
    "RawMergedBlock",
    "coerce_merged_block",
    "parse_raw_merged",
]
