"""Cleaning rules: small, composable, idempotent text transforms.

A :class:`CleaningRule` turns one piece of raw text into a cleaner piece of
text. Rules are intentionally tiny so business-specific cleaning can be built by
composition and ordering rather than a monolithic function. Every rule receives
both the ``text`` **and** the optional ``source`` descriptor (file path / URI),
so a rule can branch on the *filename* -- the customization point the ingest
pipeline needs, since cleaning is strongly business-dependent.

The built-in rules cover the normalization that helps almost every document
(BOM, line endings, control chars, trailing whitespace, blank-line collapsing).
:class:`RegexReplaceRule` and :class:`CallableRule` are the escape hatches for
ad-hoc business logic without writing a class.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class CleaningRule(Protocol):
    """A single text-cleaning step.

    Implementations must be **pure** (no side effects, no I/O) and ideally
    **idempotent** -- applying the rule twice yields the same result as once.
    ``source`` is the originating file path / URI (may be ``None``); rules that
    only care about the text simply ignore it.
    """

    def clean(self, text: str, source: str | None = None) -> str:
        """Return the cleaned ``text``. ``source`` is metadata for routing."""
        ...


# --------------------------------------------------------------------------- #
# Built-in normalization rules
# --------------------------------------------------------------------------- #
class RemoveBomRule:
    """Strip a leading Unicode byte-order mark (U+FEFF)."""

    def clean(self, text: str, source: str | None = None) -> str:
        return text.lstrip("\ufeff")


class NormalizeLineEndingsRule:
    """Collapse ``\\r\\n`` and bare ``\\r`` to ``\\n``."""

    def clean(self, text: str, source: str | None = None) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")


class RemoveControlCharsRule:
    """Delete C0 control characters and DEL (keep tab / newline).

    Useful for PDF/Office extracts that leak form feeds, vertical tabs, NULs,
    etc. into the Markdown.
    """

    _RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def clean(self, text: str, source: str | None = None) -> str:
        return self._RE.sub("", text)


class StripTrailingWhitespaceRule:
    """Remove trailing whitespace on each line and trim the whole text."""

    def clean(self, text: str, source: str | None = None) -> str:
        lines = [line.rstrip() for line in text.split("\n")]
        joined = "\n".join(lines)
        return joined.strip()


class CollapseBlankLinesRule:
    """Collapse runs of blank lines beyond ``max_blanks`` empty lines.

    ``max_blanks=1`` (the Markdown convention) keeps a single blank line between
    paragraphs and removes the multi-blank-line noise converters often emit.
    """

    def __init__(self, max_blanks: int = 1) -> None:
        if max_blanks < 0:
            raise ValueError("max_blanks must be >= 0")
        self.max_blanks = max_blanks
        keep = max_blanks + 1
        self._keep_newlines = keep
        self._re = re.compile(r"\n{" + str(keep + 1) + r",}")

    def clean(self, text: str, source: str | None = None) -> str:
        return self._re.sub("\n" * self._keep_newlines, text)


class RemoveHtmlCommentsRule:
    """Strip ``<!-- ... -->`` comments (including multi-line)."""

    _RE = re.compile(r"<!--.*?-->", re.DOTALL)

    def clean(self, text: str, source: str | None = None) -> str:
        return self._RE.sub("", text)


# --------------------------------------------------------------------------- #
# Configurable / escape-hatch rules
# --------------------------------------------------------------------------- #
class RegexReplaceRule:
    """Replace every match of ``pattern`` with ``replacement``.

    The most convenient tool for business-specific surgery: drop watermarks,
    strip page headers/footers, redact tokens, normalize terminology, ...

    Parameters
    ----------
    pattern:
        A regex string (compiled with ``flags``) or a precompiled pattern.
    replacement:
        Backreference-aware replacement string (default: empty -> remove).
    flags:
        Passed to :func:`re.compile` when ``pattern`` is a string.
    count:
        Maximum number of substitutions (``0`` = all).
    """

    def __init__(
        self,
        pattern: str | re.Pattern[str],
        replacement: str = "",
        *,
        flags: int = 0,
        count: int = 0,
    ) -> None:
        self._pattern = (
            re.compile(pattern, flags) if isinstance(pattern, str) else pattern
        )
        self._replacement = replacement
        self._count = count

    def clean(self, text: str, source: str | None = None) -> str:
        return self._pattern.sub(self._replacement, text, count=self._count)


class CallableRule:
    """Wrap a plain function ``(text, source) -> text`` as a rule.

    The fastest way to add one-off business logic without subclassing::

        cleaner.add(CallableRule(lambda t, s: t.replace("CONFIDENTIAL", "")))

    The wrapped callable may accept either ``(text)`` or ``(text, source)``.
    """

    def __init__(
        self,
        fn: Callable[..., str],
    ) -> None:
        self._fn = fn
        try:
            nparams = len(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            nparams = 1
        self._accepts_source = nparams >= 2

    def clean(self, text: str, source: str | None = None) -> str:
        if self._accepts_source:
            return self._fn(text, source)
        return self._fn(text)
