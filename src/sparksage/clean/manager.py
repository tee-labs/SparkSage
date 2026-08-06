"""The cleaning-rule manager: store <-> live cleaner <-> script compilation.

:class:`CleaningRuleManager` is the small orchestrator that ties three pieces
together so the rest of the system sees one object:

* the :class:`~sparksage.clean.store.CleaningRuleStore` (durable rule
  definitions),
* the live :class:`~sparksage.clean.cleaner.TextCleaner` (rebuilt from the
  store on every mutation), and
* :class:`~sparksage.clean.script.RestrictedScriptRule` (the sandboxed script
  each stored definition compiles into).

The manager preserves the *base* cleaner's existing rules (defaults + whatever
the caller layered on at construction) and layers the store's enabled script
rules on top, so wiring the manager into an already-configured cleaner never
drops caller-added rules. It is pure-stdlib apart from the lazy
:class:`~sparksage.clean.script.RestrictedScriptRule` import (the optional
``[clean-script]`` extra), so CRUD + persistence are fully unit-testable without
RestrictedPython installed -- only :meth:`reload` / :meth:`test_rule` need it,
and both degrade to a clear error instead of raising.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from sparksage.clean.cleaner import TextCleaner
from sparksage.clean.models import CleaningRuleRecord, PatternKind
from sparksage.clean.store import CleaningRuleStore, InMemoryCleaningRuleStore

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleStatus:
    """A record plus its current compile status (computed on the fly)."""

    record: CleaningRuleRecord
    compiled: bool
    error: str | None


@dataclass(frozen=True)
class CleaningTestResult:
    """The output of running a code snippet over sample text.

    Attributes
    ----------
    ok:
        ``True`` when the script compiled, ran within the limits, and returned
        (failures leave the text unchanged and set ``error``).
    output:
        The text the script returned (the unchanged input on a failure).
    error:
        ``None`` on success; otherwise the compile / runtime / timeout / size
        error.
    elapsed_ms:
        Wall-clock milliseconds the ``clean`` call took.
    """

    ok: bool
    output: str
    error: str | None
    elapsed_ms: float


class CleaningRuleManager:
    """Owns the cleaning-rule store + the live :class:`TextCleaner`.

    Construction materializes a live cleaner from ``base_cleaner`` plus every
    *enabled* store rule; every mutating CRUD call rebuilds the cleaner so the
    pipeline picks up the change without a restart. Disabled rules and rules
    that fail to compile are kept in the store (so the UI can show + fix them)
    but skipped at rebuild time.

    Parameters
    ----------
    store:
        The :class:`CleaningRuleStore`. Defaults to an
        :class:`InMemoryCleaningRuleStore` so rule CRUD works with zero config.
    base_cleaner:
        The cleaner whose rules form the immutable base layer (defaults +
        caller-added rules). The manager never mutates it; rebuilds layer store
        rules on top of a copy of its bindings. Defaults to a fresh
        :class:`TextCleaner`.
    """

    def __init__(
        self,
        store: CleaningRuleStore | None = None,
        *,
        base_cleaner: TextCleaner | None = None,
    ) -> None:
        self._store: CleaningRuleStore = (
            store if store is not None else InMemoryCleaningRuleStore()
        )
        self._base_cleaner: TextCleaner = (
            base_cleaner if base_cleaner is not None else TextCleaner()
        )
        self._cleaner: TextCleaner = self._base_cleaner
        self.reload()

    # ------------------------------------------------------------------ #
    # accessors
    # ------------------------------------------------------------------ #
    @property
    def cleaner(self) -> TextCleaner:
        """The live cleaner (rebuilt on every CRUD mutation)."""
        return self._cleaner

    @property
    def store(self) -> CleaningRuleStore:
        return self._store

    # ------------------------------------------------------------------ #
    # CRUD (each mutation rebuilds the live cleaner)
    # ------------------------------------------------------------------ #
    def list_rules(
        self, *, limit: int = 100, offset: int = 0
    ) -> list[CleaningRuleRecord]:
        return self._store.list(limit=limit, offset=offset)

    def list_with_status(
        self, *, limit: int = 100, offset: int = 0
    ) -> list[RuleStatus]:
        """Records + their compile status (the list view the UI shows)."""
        out: list[RuleStatus] = []
        for rec in self._store.list(limit=limit, offset=offset):
            out.append(self.status_of(rec))
        return out

    def status_of(self, record: CleaningRuleRecord) -> RuleStatus:
        """The compile status of one record (disabled rules report no error)."""
        if not record.enabled:
            return RuleStatus(record=record, compiled=False, error=None)
        error = self._compile_error(record)
        return RuleStatus(record=record, compiled=error is None, error=error)

    def get_rule(self, rule_id: str) -> CleaningRuleRecord | None:
        return self._store.get(rule_id)

    def create_rule(self, record: CleaningRuleRecord) -> CleaningRuleRecord:
        stored = self._store.add(record)
        self.reload()
        return stored

    def update_rule(self, record: CleaningRuleRecord) -> CleaningRuleRecord:
        stored = self._store.update(record)
        self.reload()
        return stored

    def delete_rule(self, rule_id: str) -> bool:
        removed = self._store.delete(rule_id)
        if removed:
            self.reload()
        return removed

    # ------------------------------------------------------------------ #
    # test (transient compile + run; never mutates the store / cleaner)
    # ------------------------------------------------------------------ #
    def test_rule(
        self,
        code: str,
        text: str,
        *,
        source: str | None = None,
        timeout: float = 5.0,
        max_input_chars: int = 1_000_000,
        max_output_chars: int = 2_000_000,
    ) -> CleaningTestResult:
        """Run ``code`` over ``text`` without persisting or applying it."""
        rule = self._build_rule_raw(
            code,
            name="<test>",
            timeout=timeout,
            max_input_chars=max_input_chars,
            max_output_chars=max_output_chars,
        )
        if isinstance(rule, str):
            return CleaningTestResult(ok=False, output=text, error=rule, elapsed_ms=0.0)
        start = time.monotonic()
        try:
            output = rule.clean(text, source)
        except Exception as exc:  # pragma: no cover - the rule fails-open, but guard
            elapsed = (time.monotonic() - start) * 1000.0
            return CleaningTestResult(
                ok=False, output=text, error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=elapsed,
            )
        elapsed = (time.monotonic() - start) * 1000.0
        error = rule.last_error
        return CleaningTestResult(
            ok=error is None, output=output, error=error, elapsed_ms=round(elapsed, 1)
        )

    # ------------------------------------------------------------------ #
    # rebuild
    # ------------------------------------------------------------------ #
    def reload(self) -> None:
        """Rebuild the live cleaner from the base + every enabled store rule."""
        cleaner = TextCleaner(use_defaults=False)
        # ponytail: reach into the sibling registry's binding list (same package)
        # to clone the base cleaner's defaults + caller-added rules verbatim, so
        # wiring the manager in never drops a caller-configured rule.
        cleaner.registry._bindings = list(self._base_cleaner.registry._bindings)

        for rec in self._store.list(limit=10**9):
            if not rec.enabled:
                continue
            rule = self._build_rule(rec)
            if isinstance(rule, str):
                _logger.warning(
                    "cleaning rule '%s' (%s) skipped: %s", rec.name, rec.rule_id, rule
                )
                continue
            if rec.pattern_kind == PatternKind.NONE or not rec.source_pattern:
                cleaner.add(rule)
            elif rec.pattern_kind == PatternKind.REGEX:
                cleaner.add_for(rec.source_pattern, rule, regex=True)
            else:
                cleaner.add_for(rec.source_pattern, rule)
        self._cleaner = cleaner

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _compile_error(self, record: CleaningRuleRecord) -> str | None:
        """Return ``None`` if the record compiles, else the error message."""
        rule = self._build_rule(record)
        return rule if isinstance(rule, str) else None

    @staticmethod
    def _build_rule_raw(
        code: str,
        *,
        name: str,
        timeout: float,
        max_input_chars: int,
        max_output_chars: int,
    ) -> object | str:
        """Build a :class:`RestrictedScriptRule` or return an error string."""
        try:
            from sparksage.clean.script import RestrictedScriptRule
        except ImportError as exc:  # pragma: no cover - env-dependent
            return (
                "RestrictedPython not installed; pip install 'sparksage[clean-script]'"
                f" ({exc})"
            )
        try:
            return RestrictedScriptRule(
                code,
                name=name,
                timeout=timeout,
                max_input_chars=max_input_chars,
                max_output_chars=max_output_chars,
            )
        except Exception as exc:
            # ponytail: broad catch is the fail-open pattern -- construction can
            # raise ImportError (missing [clean-script] extra: RestrictedPython
            # or regex) or ValueError (compile/exec); all should surface as a
            # helpful error string, never a 500 that crashes the test endpoint.
            return str(exc) or f"{type(exc).__name__}: {exc}"

    @classmethod
    def _build_rule(cls, record: CleaningRuleRecord) -> object | str:
        """Build a rule from a stored record (or return an error string)."""
        return cls._build_rule_raw(
            record.code,
            name=record.name,
            timeout=record.timeout,
            max_input_chars=record.max_input_chars,
            max_output_chars=record.max_output_chars,
        )


__all__ = [
    "CleaningRuleManager",
    "CleaningTestResult",
    "RuleStatus",
]
