"""Lenient intermediate models + coercion for the agent controller's decisions.

Mirrors the established two-stage pattern of :mod:`sparksage.reader.schema` and
:mod:`sparksage.query.schema`: raw model output (the controller's JSON
``{"thought", "action", "query", "k"}``) is parsed into the *lenient*
:class:`RawAgentAction` here (plain strings, ``extra="ignore"``), then coerced
into the strict :class:`~sparksage.agent.models.AgentAction`
(``extra="forbid"`` semantics via the :class:`ActionType` enum). This keeps the
enum the single source of truth while staying robust to messy LLM output
(prose-wrapped JSON, unknown action labels, missing query on a retrieve).

The ``_FENCE_RE`` / :func:`extract_json` helpers are duplicated here on purpose
-- each schema module stays standalone, exactly as the reader / query / distill
schema modules do.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from sparksage.agent.models import ActionType, AgentAction

#: Action assumed when the controller emits an unknown / empty action label.
DEFAULT_ACTION = ActionType.SYNTHESIZE


_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$", re.MULTILINE)


class CoercionError(ValueError):
    """Raised when a raw controller decision cannot be coerced."""


# --------------------------------------------------------------------------- #
# Lenient model (what the LLM emits)
# --------------------------------------------------------------------------- #
class RawAgentAction(BaseModel):
    """Lenient controller decision as emitted by an LLM."""

    model_config = ConfigDict(extra="ignore")

    thought: str = ""
    action: str = ""
    query: str | None = None
    k: Any = None


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


def _map_action(raw: str, *, strict: bool) -> ActionType:
    value = (raw or "").strip().lower()
    if value:
        aliases = {
            "retrieve": ActionType.RETRIEVE,
            "search": ActionType.RETRIEVE,
            "lookup": ActionType.RETRIEVE,
            "query": ActionType.RETRIEVE,
            "synthesize": ActionType.SYNTHESIZE,
            "synthesise": ActionType.SYNTHESIZE,
            "answer": ActionType.SYNTHESIZE,
            "finish": ActionType.SYNTHESIZE,
            "done": ActionType.SYNTHESIZE,
            "final": ActionType.SYNTHESIZE,
        }
        mapped = aliases.get(value)
        if mapped is not None:
            return mapped
        for member in ActionType:
            if member.value == value:
                return member
    if strict:
        raise CoercionError(f"unknown action: {raw!r}")
    return DEFAULT_ACTION


def _coerce_k(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


def coerce_action(raw: RawAgentAction, *, strict: bool) -> AgentAction:
    """Turn a lenient :class:`RawAgentAction` into a strict :class:`AgentAction`.

    * ``action`` is mapped to :class:`ActionType` (case-insensitive, with a few
      common aliases); unknown labels fall back to ``SYNTHESIZE`` unless strict.
    * a ``RETRIEVE`` with an empty query falls back to ``SYNTHESIZE`` (there is
      nothing to retrieve) unless strict, which raises.
    * ``k`` is clamped to ``>= 1`` or dropped when unparseable.
    """
    if not isinstance(raw, RawAgentAction):
        raise CoercionError("expected a RawAgentAction")
    action = _map_action(raw.action, strict=strict)
    query = (raw.query or "").strip() or None
    if action is ActionType.RETRIEVE and not query:
        if strict:
            raise CoercionError("retrieve action requires a non-empty query")
        action = ActionType.SYNTHESIZE
    return AgentAction(
        action=action,
        thought=(raw.thought or "").strip(),
        query=query if action is ActionType.RETRIEVE else None,
        k=_coerce_k(raw.k) if action is ActionType.RETRIEVE else None,
    )


# --------------------------------------------------------------------------- #
# Parsing (raw text -> lenient model -> strict action)
# --------------------------------------------------------------------------- #
def parse_raw_action(data: Any) -> RawAgentAction:
    if not isinstance(data, dict):
        raise CoercionError(f"expected a JSON object, got {type(data).__name__}")
    try:
        return RawAgentAction.model_validate(data)
    except ValidationError as exc:
        raise CoercionError(str(exc)) from exc


def parse_action_response(text: str, *, strict: bool = False) -> AgentAction:
    """Decode a controller response string into a strict :class:`AgentAction`.

    Pipeline: :func:`extract_json` -> ``json.loads`` -> :func:`parse_raw_action`
    -> :func:`coerce_action`.
    """
    payload = json.loads(extract_json(text))
    raw = parse_raw_action(payload)
    return coerce_action(raw, strict=strict)


__all__ = [
    "DEFAULT_ACTION",
    "CoercionError",
    "RawAgentAction",
    "coerce_action",
    "extract_json",
    "parse_action_response",
    "parse_raw_action",
]
