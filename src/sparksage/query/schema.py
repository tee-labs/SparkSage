"""Lenient intermediate models + enum coercion for LLM-produced query data.

Mirrors the established two-stage pattern of :mod:`sparksage.generator.schema`:
raw model output is parsed into the *lenient* models here (plain strings,
``extra="ignore"``), then coerced through the :class:`QueryIntent` controlled
vocabulary into the strict, ``extra="forbid"`` result models. This keeps the
strict result as the single source of truth while staying robust to messy LLM
output (arbitrary casing, missing fields, prose-wrapped JSON).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sparksage.llmutil import extract_json as _extract_json
from sparksage.schema.enums import QueryIntent

#: Default intent assumed when a model emits an unrecognised intent string.
DEFAULT_INTENT = QueryIntent.BUSINESS_ANALYSIS

#: Confidence assumed when the model omits a usable confidence value.
DEFAULT_CONFIDENCE = 0.5


class CoercionError(ValueError):
    """Raised when raw query data cannot be coerced into a valid result."""


# --------------------------------------------------------------------------- #
# Lenient models (what the LLM emits)
# --------------------------------------------------------------------------- #
class RawIntent(BaseModel):
    """Lenient intent classification as emitted by an LLM (strings, not enums)."""

    model_config = ConfigDict(extra="ignore")

    reasoning: str = ""
    intent: str = ""
    confidence: float = DEFAULT_CONFIDENCE


class RawRewrite(BaseModel):
    """Lenient rewrite as emitted by an LLM (plain strings, extras ignored)."""

    model_config = ConfigDict(extra="ignore")

    reasoning: str = ""
    rewritten_query: str = ""
    sub_queries: list[str] = Field(default_factory=list)
    extracted_companies: list[str] = Field(default_factory=list)
    extracted_years: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Strict result models (what the rest of the system consumes)
# --------------------------------------------------------------------------- #
class IntentResult(BaseModel):
    """Strict, validated outcome of intent classification."""

    model_config = ConfigDict(extra="forbid")

    intent: QueryIntent
    confidence: float
    reasoning: str = ""


class RewriteResult(BaseModel):
    """Strict, validated outcome of query rewriting."""

    model_config = ConfigDict(extra="forbid")

    rewritten_query: str
    sub_queries: list[str] = Field(default_factory=list)
    extracted_companies: list[str] = Field(default_factory=list)
    extracted_years: list[str] = Field(default_factory=list)
    reasoning: str = ""


# --------------------------------------------------------------------------- #
# JSON extraction (from possibly-noisy model responses)
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> str:
    """Pull the JSON object out of a possibly-noisy model response.

    Handles plain JSON, JSON wrapped in ```json fences, and JSON embedded in
    surrounding prose (extracted via outermost brace matching). Raises
    :class:`CoercionError` on an empty or unparseable response.
    """
    return _extract_json(text, error_type=CoercionError)


# --------------------------------------------------------------------------- #
# Parsing (decoded dict -> lenient model)
# --------------------------------------------------------------------------- #
def parse_raw_intent(data: Any) -> RawIntent:
    """Validate decoded JSON into :class:`RawIntent`."""
    if not isinstance(data, dict):
        raise CoercionError(
            f"expected a JSON object, got {type(data).__name__}"
        )
    try:
        return RawIntent.model_validate(data)
    except ValidationError as exc:
        raise CoercionError(str(exc)) from exc


def parse_raw_rewrite(data: Any) -> RawRewrite:
    """Validate decoded JSON into :class:`RawRewrite`."""
    if not isinstance(data, dict):
        raise CoercionError(
            f"expected a JSON object, got {type(data).__name__}"
        )
    try:
        return RawRewrite.model_validate(data)
    except ValidationError as exc:
        raise CoercionError(str(exc)) from exc


def parse_intent_response(text: str) -> RawIntent:
    """Extract JSON from a raw model response and parse it into :class:`RawIntent`."""
    return parse_raw_intent(json.loads(extract_json(text)))


def parse_rewrite_response(text: str) -> RawRewrite:
    """Extract JSON from a raw model response and parse it into :class:`RawRewrite`."""
    return parse_raw_rewrite(json.loads(extract_json(text)))


# --------------------------------------------------------------------------- #
# Coercion (lenient -> strict)
# --------------------------------------------------------------------------- #
def _map_intent(raw: str, *, strict: bool) -> QueryIntent:
    """Map a raw intent string to :class:`QueryIntent`.

    Case-insensitive on the enum *value*. In ``strict`` mode an unknown intent
    raises :class:`CoercionError`; otherwise falls back to :data:`DEFAULT_INTENT`
    so a bad label does not wreck the whole pipeline.
    """
    value = (raw or "").strip()
    if value:
        needle = value.lower()
        for member in QueryIntent:
            if member.value.lower() == needle:
                return member
    if strict:
        raise CoercionError(f"unknown intent: {raw!r}")
    return DEFAULT_INTENT


def _coerce_confidence(raw: float) -> float:
    """Clamp a raw confidence into the inclusive ``[0.0, 1.0]`` range.

    NaN / non-finite values fall back to :data:`DEFAULT_CONFIDENCE`.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE
    if value != value or value in (float("inf"), float("-inf")):  # NaN/inf
        return DEFAULT_CONFIDENCE
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def coerce_intent(raw: RawIntent, *, strict: bool) -> IntentResult:
    """Normalize a :class:`RawIntent` into a strict :class:`IntentResult`."""
    if not isinstance(raw, RawIntent):
        raise CoercionError("expected a RawIntent")
    return IntentResult(
        intent=_map_intent(raw.intent, strict=strict),
        confidence=_coerce_confidence(raw.confidence),
        reasoning=raw.reasoning.strip(),
    )


def _clean_list(items: list[str]) -> list[str]:
    """Deduplicate a list of strings while preserving order and dropping blanks."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = (item or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def coerce_rewrite(
    raw: RawRewrite,
    original_query: str,
    *,
    strict: bool,
) -> RewriteResult:
    """Normalize a :class:`RawRewrite` into a strict :class:`RewriteResult`.

    Falls back to ``original_query`` when the model produced an empty
    ``rewritten_query`` (a rewrite that drops the query is never an improvement),
    so callers always get something searchable. ``sub_queries``,
    ``extracted_companies`` and ``extracted_years`` are de-duplicated and
    stripped; in ``strict`` mode an empty rewrite still raises.
    """
    if not isinstance(raw, RawRewrite):
        raise CoercionError("expected a RawRewrite")

    rewritten = raw.rewritten_query.strip()
    if not rewritten:
        if strict:
            raise CoercionError("rewrite produced an empty rewritten_query")
        rewritten = original_query.strip()

    try:
        return RewriteResult(
            rewritten_query=rewritten,
            sub_queries=_clean_list(raw.sub_queries),
            extracted_companies=_clean_list(raw.extracted_companies),
            extracted_years=_clean_list(raw.extracted_years),
            reasoning=raw.reasoning.strip(),
        )
    except ValidationError as exc:  # pragma: no cover - defensive
        raise CoercionError(str(exc)) from exc
