"""Lenient intermediate models + coercion for the agent controller's decisions.

Mirrors the established two-stage pattern of :mod:`sparksage.reader.schema` and
:mod:`sparksage.query.schema`: raw model output (the controller's JSON
``{"thought", "action", "query", "k"}``) is parsed into the *lenient*
:class:`RawAgentAction` here (plain strings, ``extra="ignore"``), then coerced
into the strict :class:`~sparksage.agent.models.AgentAction`
(``extra="forbid"`` semantics via the :class:`ActionType` enum). This keeps the
enum the single source of truth while staying robust to messy LLM output
(prose-wrapped JSON, unknown action labels, missing query on a retrieve).

The :func:`extract_json` helper is shared from :mod:`sparksage.llmutil` --
the single copy every schema module imports.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, ValidationError

from sparksage.agent.models import ActionType, AgentAction
from sparksage.llmutil import extract_json as _extract_json

if TYPE_CHECKING:
    from sparksage.retrieve.models import RetrievalFilter

#: Action assumed when the controller emits an unknown / empty action label.
DEFAULT_ACTION = ActionType.SYNTHESIZE


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
    sub_queries: list[str] | None = None
    k: Any = None
    filter: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# JSON extraction (from possibly-noisy model responses)
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> str:
    """Pull the JSON object out of a possibly-noisy model response.

    Handles plain JSON, ```json fences, and JSON embedded in prose (extracted
    via outermost brace matching). Raises :class:`CoercionError` on an empty or
    unparseable response.
    """
    return _extract_json(text, error_type=CoercionError)


def _map_action(raw: str, *, strict: bool) -> ActionType:
    value = (raw or "").strip().lower()
    if value:
        aliases = {
            "retrieve": ActionType.RETRIEVE,
            "search": ActionType.RETRIEVE,
            "lookup": ActionType.RETRIEVE,
            "query": ActionType.RETRIEVE,
            "plan": ActionType.PLAN,
            "decompose": ActionType.PLAN,
            "break_down": ActionType.PLAN,
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


def _coerce_sub_queries(raw: list[str] | None) -> list[str] | None:
    """De-duplicate + strip a PLAN's sub-queries; ``None`` when nothing left."""
    if not raw:
        return None
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        text = (str(item) if item is not None else "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out or None


def coerce_action(raw: RawAgentAction, *, strict: bool) -> AgentAction:
    """Turn a lenient :class:`RawAgentAction` into a strict :class:`AgentAction`.

    * ``action`` is mapped to :class:`ActionType` (case-insensitive, with a few
      common aliases); unknown labels fall back to ``SYNTHESIZE`` unless strict.
    * a ``RETRIEVE`` with an empty query falls back to ``SYNTHESIZE`` (there is
      nothing to retrieve) unless strict, which raises.
    * a ``PLAN`` with no usable ``sub_queries`` falls back to ``SYNTHESIZE``
      (nothing to decompose) unless strict, which raises.
    * ``k`` is clamped to ``>= 1`` or dropped when unparseable.
    """
    if not isinstance(raw, RawAgentAction):
        raise CoercionError("expected a RawAgentAction")
    action = _map_action(raw.action, strict=strict)
    query = (raw.query or "").strip() or None
    sub_queries = _coerce_sub_queries(raw.sub_queries)
    if action is ActionType.RETRIEVE and not query:
        if strict:
            raise CoercionError("retrieve action requires a non-empty query")
        action = ActionType.SYNTHESIZE
    if action is ActionType.PLAN and not sub_queries:
        if strict:
            raise CoercionError("plan action requires a non-empty sub_queries list")
        action = ActionType.SYNTHESIZE
    flt = _coerce_filter(raw.filter)
    return AgentAction(
        action=action,
        thought=(raw.thought or "").strip(),
        query=query if action is ActionType.RETRIEVE else None,
        sub_queries=sub_queries if action is ActionType.PLAN else None,
        k=_coerce_k(raw.k) if action is ActionType.RETRIEVE else None,
        filter=flt,
    )


def _coerce_filter(raw: dict[str, Any] | None) -> RetrievalFilter | None:
    """Best-effort coercion of a controller ``filter`` blob.

    Accepts a dict with any of ``tags`` / ``entities`` / ``languages`` /
    ``kb_id``. Unknown tag values are dropped (rather than failing) so a
    controller that hallucinate a tag cannot abort a multi-step run. Returns
    ``None`` when no usable field survives.
    """
    if not isinstance(raw, dict) or not raw:
        return None
    from sparksage.retrieve.models import RetrievalFilter
    from sparksage.schema.enums import Tag

    tags: set[Tag] = set()
    for raw_tag in raw.get("tags") or []:
        try:
            tags.add(Tag(str(raw_tag)))
        except ValueError:
            continue
    entities = {str(e) for e in raw.get("entities") or []} or None
    languages = {str(lang) for lang in raw.get("languages") or []} or None
    kb_id = raw.get("kb_id")
    kb_id = str(kb_id) if kb_id else None
    if not tags and not entities and not languages and not kb_id:
        return None
    return RetrievalFilter(
        tags=tags or None,
        entities=entities,
        languages=languages,
        kb_id=kb_id,
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
