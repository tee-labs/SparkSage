"""Source-aware routing of cleaning rules to documents.

Cleaning is strongly business-dependent, so which rules apply often depends on
*where a document came from*: strip page footers only for PDFs, remove Confluence
macros only for Confluence exports, redact PII only for support tickets, ...

:class:`CleaningRegistry` is the routing layer that makes this composable. Rules
can be registered globally (apply to every document) or conditionally, keyed by
the document's ``source`` descriptor (its path / URI). Conditionals match by
:mod:`fnmatch` glob (default) or :mod:`re` regex. The registry is a pure,
side-effect-free dispatcher -- it has no knowledge of any specific business
domain.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from sparksage.clean.rules import CleaningRule


@dataclass(frozen=True)
class _Binding:
    rule: CleaningRule
    match: Callable[[str], bool] | None


def _glob_match(source: str, pattern: str) -> bool:
    """True if ``source`` or its basename matches the glob ``pattern``.

    Matching against both the full path and the file name lets callers write
    either ``"*.pdf"`` or ``"tickets/*"`` and have it "just work".
    """
    name = Path(source).name
    return fnmatch.fnmatch(source, pattern) or fnmatch.fnmatch(name, pattern)


class CleaningRegistry:
    """Ordered, source-aware collection of cleaning rules.

    Rules are applied in registration order. A rule with no matcher runs for
    every document; a rule with a matcher runs only when its predicate accepts
    the document's ``source``.

    The registry is intentionally framework-free: it stores callables and
    strings, nothing more.
    """

    def __init__(self) -> None:
        self._bindings: list[_Binding] = []

    def add(self, rule: CleaningRule) -> CleaningRule:
        """Register ``rule`` as a global rule (runs for every source).

        Returns the rule unchanged for fluent composition.
        """
        self._bindings.append(_Binding(rule=rule, match=None))
        return rule

    def add_for_glob(
        self, pattern: str, rule: CleaningRule
    ) -> CleaningRule:
        """Register ``rule`` for sources whose path/name matches ``pattern``."""
        self._bindings.append(
            _Binding(rule=rule, match=lambda s: _glob_match(s, pattern))
        )
        return rule

    def add_for_regex(
        self, pattern: str, rule: CleaningRule, *, flags: int = 0
    ) -> CleaningRule:
        """Register ``rule`` for sources matching the regex ``pattern``."""
        compiled = re.compile(pattern, flags)

        def _m(s: str, _rx: re.Pattern[str] = compiled) -> bool:
            return _rx.search(s) is not None

        self._bindings.append(_Binding(rule=rule, match=_m))
        return rule

    def rules_for(self, source: str | None) -> list[CleaningRule]:
        """Rules applicable to ``source``, in registration order."""
        s = source or ""
        return [
            binding.rule
            for binding in self._bindings
            if binding.match is None or binding.match(s)
        ]

    def clean(self, text: str, source: str | None = None) -> str:
        """Apply every applicable rule to ``text`` in order and return it."""
        for rule in self.rules_for(source):
            text = rule.clean(text, source)
        return text

    def __len__(self) -> int:
        return len(self._bindings)

    def __iter__(self) -> Iterable[_Binding]:
        return iter(self._bindings)
