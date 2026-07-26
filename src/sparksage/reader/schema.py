"""Lenient intermediate models + coercion for LLM-produced answers.

Mirrors the established two-stage pattern of :mod:`sparksage.query.schema` and
:mod:`sparksage.generator.schema`: raw model output is parsed into the *lenient*
models here (plain strings, ``extra="ignore"``), then coerced into the strict,
``extra="forbid"`` result models. This keeps the strict result the single
source of truth while staying robust to messy LLM output (prose-wrapped JSON,
missing fields, hallucinated block ids).

The reader is where :class:`~sparksage.retrieve.models.Citation` finally
*consumes* the schema's ``source.locator`` field -- a grounded citation carries
``uri`` / ``locator`` / ``title`` straight from the backing IdeaBlock's
:class:`~sparksage.schema.source.SourceRef`, the provenance the ingest side has
been filling but no consumer has read until now.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sparksage.retrieve.models import Citation

#: Confidence assumed when the generator omits a usable confidence value.
DEFAULT_ANSWER_CONFIDENCE = 0.5

#: Confidence assumed when the faithfulness judge omits a usable score.
DEFAULT_FAITHFULNESS = 0.5

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$", re.MULTILINE)


class CoercionError(ValueError):
    """Raised when raw answer / faithfulness data cannot be coerced."""


# --------------------------------------------------------------------------- #
# Lenient models (what the LLM emits)
# --------------------------------------------------------------------------- #
class RawCitation(BaseModel):
    """Lenient citation as emitted by an LLM."""

    model_config = ConfigDict(extra="ignore")

    block_id: str | int = ""
    quote: str = ""


class RawAnswer(BaseModel):
    """Lenient generated answer as emitted by an LLM."""

    model_config = ConfigDict(extra="ignore")

    reasoning: str = ""
    answer: str = ""
    citations: list[RawCitation] = Field(default_factory=list)
    confidence: float = DEFAULT_ANSWER_CONFIDENCE


class RawFaithfulness(BaseModel):
    """Lenient faithfulness verdict as emitted by an LLM."""

    model_config = ConfigDict(extra="ignore")

    reasoning: str = ""
    score: float = DEFAULT_FAITHFULNESS
    supported_claims: int = 0
    unsupported_claims: int = 0


# --------------------------------------------------------------------------- #
# Strict result models (what the rest of the system consumes)
# --------------------------------------------------------------------------- #
class FaithfulnessResult(BaseModel):
    """Strict verdict on whether a generated answer is supported by evidence.

    Attributes
    ----------
    score:
        Faithfulness in ``[0, 1]`` -- ``1.0`` means every claim is grounded,
        ``0.0`` means nothing is. Below the reader's ``min_faithfulness`` the
        orchestrator abstains rather than surface a hallucination.
    supported_claims, unsupported_claims:
        Counts the judge used to reach ``score`` (for transparency / dashboards).
    reasoning:
        The judge's brief explanation.
    """

    model_config = ConfigDict(extra="forbid")

    score: float
    supported_claims: int = 0
    unsupported_claims: int = 0
    reasoning: str = ""


class GeneratedAnswer(BaseModel):
    """A reader-produced answer with grounded citations.

    Attributes
    ----------
    text:
        The generated natural-language answer.
    citations:
        :class:`~sparksage.retrieve.models.Citation` list, each tied to a
        backing block id and carrying that block's ``source.uri`` /
        ``source.locator`` for traceability.
    grounded_block_ids:
        The distinct backing block ids (superset of citation block ids).
    confidence:
        The generator's self-reported confidence in ``[0, 1]``.
    faithfulness:
        The judge's faithfulness score, set after judging; ``None`` until then.
    abstained:
        ``True`` when the reader chose not to answer (insufficient evidence or
        low faithfulness). See ``abstention_reason``.
    abstention_reason:
        Why the reader abstained (``None`` when it answered).
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[Citation] = Field(default_factory=list)
    grounded_block_ids: list[str] = Field(default_factory=list)
    confidence: float = DEFAULT_ANSWER_CONFIDENCE
    faithfulness: float | None = None
    abstained: bool = False
    abstention_reason: str | None = None


# --------------------------------------------------------------------------- #
# JSON extraction (from possibly-noisy model responses)
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> str:
    """Pull the JSON object out of a possibly-noisy model response.

    Handles plain JSON, ```json fences, and JSON embedded in prose (extracted
    via outermost brace matching). Raises :class:`CoercionError` on an empty or
    unparseable response.
    """
    cleaned = text.strip()
    if not cleaned:
        raise CoercionError("empty model response")
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    raise CoercionError("model response was not valid JSON")


def _coerce_score(raw: float, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value != value or value in (float("inf"), float("-inf")):
        return default
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _coerce_int(raw: Any, default: int = 0) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(value, 0)


def _to_block_id(raw: str | int) -> str:
    """Coerce a raw citation id (string or int) to a clean string."""
    if isinstance(raw, int):
        return str(raw)
    return str(raw).strip()


# --------------------------------------------------------------------------- #
# Parsing (decoded dict -> lenient model)
# --------------------------------------------------------------------------- #
def parse_raw_answer(data: Any) -> RawAnswer:
    if not isinstance(data, dict):
        raise CoercionError(f"expected a JSON object, got {type(data).__name__}")
    try:
        return RawAnswer.model_validate(data)
    except ValidationError as exc:
        raise CoercionError(str(exc)) from exc


def parse_raw_faithfulness(data: Any) -> RawFaithfulness:
    if not isinstance(data, dict):
        raise CoercionError(f"expected a JSON object, got {type(data).__name__}")
    try:
        return RawFaithfulness.model_validate(data)
    except ValidationError as exc:
        raise CoercionError(str(exc)) from exc


def parse_answer_response(text: str) -> RawAnswer:
    return parse_raw_answer(json.loads(extract_json(text)))


def parse_faithfulness_response(text: str) -> RawFaithfulness:
    return parse_raw_faithfulness(json.loads(extract_json(text)))


# --------------------------------------------------------------------------- #
# Coercion (lenient -> strict)
# --------------------------------------------------------------------------- #
def coerce_citations(
    raw: list[RawCitation],
    valid_ids: set[str],
) -> list[Citation]:
    """Drop citations whose ``block_id`` is not in ``valid_ids``.

    Guards against the model hallucinating a block id that was never in the
    context -- a grounded citation must point at a real retrieved block.
    """
    out: list[Citation] = []
    seen: set[str] = set()
    for rc in raw:
        bid = _to_block_id(rc.block_id)
        if not bid or bid not in valid_ids or bid in seen:
            continue
        seen.add(bid)
        out.append(Citation(block_id=bid, quote=(rc.quote or "").strip()))
    return out


def coerce_answer(
    raw: RawAnswer,
    valid_ids: set[str],
    id_to_citation: dict[str, Citation],
    *,
    strict: bool,
) -> GeneratedAnswer:
    """Normalize a :class:`RawAnswer` into a strict :class:`GeneratedAnswer`.

    ``id_to_citation`` maps each candidate block id to a pre-built
    :class:`~sparksage.retrieve.models.Citation` carrying that block's
    provenance (``uri`` / ``locator`` / ``title``) -- so the coerced answer's
    citations stay grounded in the schema's source fields even when the model
    emits only a bare id. An empty answer triggers abstention rather than an
    empty reply.
    """
    if not isinstance(raw, RawAnswer):
        raise CoercionError("expected a RawAnswer")
    text = (raw.answer or "").strip()
    if not text:
        if strict:
            raise CoercionError("generator produced an empty answer")
        return GeneratedAnswer(
            text="",
            citations=[],
            grounded_block_ids=[],
            confidence=0.0,
            abstained=True,
            abstention_reason="no answer produced",
        )

    base = coerce_citations(raw.citations, valid_ids)
    citations: list[Citation] = []
    for c in base:
        prov = id_to_citation.get(c.block_id)
        if prov is not None and (prov.uri or prov.locator or prov.title):
            from dataclasses import replace

            citations.append(
                replace(c, uri=prov.uri, locator=prov.locator, title=prov.title)
            )
        else:
            citations.append(c)

    grounded = list({c.block_id for c in citations})
    return GeneratedAnswer(
        text=text,
        citations=citations,
        grounded_block_ids=grounded,
        confidence=_coerce_score(raw.confidence, DEFAULT_ANSWER_CONFIDENCE),
    )


def coerce_faithfulness(raw: RawFaithfulness, *, strict: bool) -> FaithfulnessResult:
    """Normalize a :class:`RawFaithfulness` into a strict result."""
    if not isinstance(raw, RawFaithfulness):
        raise CoercionError("expected a RawFaithfulness")
    return FaithfulnessResult(
        score=_coerce_score(raw.score, DEFAULT_FAITHFULNESS),
        supported_claims=_coerce_int(raw.supported_claims),
        unsupported_claims=_coerce_int(raw.unsupported_claims),
        reasoning=(raw.reasoning or "").strip(),
    )


__all__ = [
    "CoercionError",
    "DEFAULT_ANSWER_CONFIDENCE",
    "DEFAULT_FAITHFULNESS",
    "FaithfulnessResult",
    "GeneratedAnswer",
    "RawAnswer",
    "RawCitation",
    "RawFaithfulness",
    "coerce_answer",
    "coerce_citations",
    "coerce_faithfulness",
    "extract_json",
    "parse_answer_response",
    "parse_faithfulness_response",
    "parse_raw_answer",
    "parse_raw_faithfulness",
]
