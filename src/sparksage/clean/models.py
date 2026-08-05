"""Cleaning-rule records: the persisted definition of a custom cleaning rule.

Declarative normalization lives in :mod:`sparksage.clean.rules`; the script
escape hatch (:class:`~sparksage.clean.script.RestrictedScriptRule`) compiles a
user ``clean(text, source)`` at construction. :class:`CleaningRuleRecord` is the
*stored* form of such a rule -- the source code plus the routing + safety knobs
-- so a rule survives a restart and can be rebuilt into the live
:class:`~sparksage.clean.cleaner.TextCleaner` on load.

Mirrors the established schema conventions: Pydantic v2, ``ConfigDict(extra=
"forbid")``, a closed enum for the controlled pattern vocabulary.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class PatternKind(str, Enum):
    """How a rule's ``source_pattern`` is interpreted when routing.

    ``NONE`` makes the rule global (applies to every source); ``GLOB`` matches
    via :func:`fnmatch.fnmatch` against the path and basename; ``REGEX`` matches
    via :func:`re.search`. Mirrors :class:`CleaningRegistry.add_for_glob` /
    :meth:`add_for_regex` so a stored rule maps cleanly onto the registry.
    """

    NONE = "none"
    GLOB = "glob"
    REGEX = "regex"


class CleaningRuleRecord(BaseModel):
    """The stored definition of one custom cleaning rule.

    Attributes
    ----------
    rule_id:
        Stable unique id (UUID4 string). Auto-generated when omitted.
    name:
        Human-readable label (shown in the UI / logs).
    code:
        RestrictedPython source defining ``clean(text, source=None) -> str``.
        Compiled into a :class:`~sparksage.clean.script.RestrictedScriptRule`
        when the rule is applied.
    source_pattern:
        Source-routing pattern. Interpreted per :attr:`pattern_kind`. ``None``
        (with ``pattern_kind=NONE``) makes the rule global.
    pattern_kind:
        How :attr:`source_pattern` is matched.
    enabled:
        When ``False`` the rule is stored but not applied (the rebuild skips
        it). The UI toggle.
    timeout:
        Wall-clock seconds per ``clean`` call (and per regex operation).
    max_input_chars:
        Texts larger than this are skipped without invoking the script.
    max_output_chars:
        Outputs larger than this are treated as a failure (fail-open).
    created_at, updated_at:
        UTC timestamps.
    """

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(default_factory=_new_id, description="Stable unique id.")
    name: str = Field(..., min_length=1, max_length=128, description="Rule label.")
    code: str = Field(..., min_length=1, description="RestrictedPython source.")
    source_pattern: str | None = Field(
        default=None, description="Source-routing pattern (None = global)."
    )
    pattern_kind: PatternKind = Field(
        default=PatternKind.NONE, description="How source_pattern is matched."
    )
    enabled: bool = Field(default=True, description="Whether the rule is applied.")
    timeout: float = Field(default=5.0, gt=0, description="Per-call wall-clock seconds.")
    max_input_chars: int = Field(
        default=1_000_000, ge=1, description="Skip scripts above this input size."
    )
    max_output_chars: int = Field(
        default=2_000_000, ge=1, description="Fail scripts above this output size."
    )
    created_at: datetime = Field(default_factory=_utcnow, description="UTC timestamp.")
    updated_at: datetime = Field(default_factory=_utcnow, description="UTC timestamp.")

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("name must not be empty")
        return value

    @field_validator("source_pattern")
    @classmethod
    def _normalize_pattern(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


__all__ = ["CleaningRuleRecord", "PatternKind"]
