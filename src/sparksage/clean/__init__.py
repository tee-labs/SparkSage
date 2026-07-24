"""Customizable text cleaning: raw document text -> generation-ready text.

Cleaning is the business-dependent step between conversion and generation.
Inject rules into a :class:`TextCleaner` -- global rules via
:meth:`~TextCleaner.add`, source/filename-specific rules via
:meth:`~TextCleaner.add_for` -- then call :meth:`~TextCleaner.clean` (or
:meth:`~TextCleaner.clean_result` to chain straight off a
:class:`~sparksage.convert.ConversionResult`).

The emitted :class:`CleaningResult` chains straight into block generation: feed
``result.text`` as the text and ``result.source_ref`` as provenance to
:class:`~sparksage.generator.IdeaBlockGenerator`.
"""

from sparksage.clean.cleaner import DEFAULT_RULES, CleaningResult, TextCleaner
from sparksage.clean.registry import CleaningRegistry
from sparksage.clean.rules import (
    CallableRule,
    CleaningRule,
    CollapseBlankLinesRule,
    NormalizeLineEndingsRule,
    RegexReplaceRule,
    RemoveBomRule,
    RemoveControlCharsRule,
    RemoveHtmlCommentsRule,
    StripTrailingWhitespaceRule,
)

__all__ = [
    "DEFAULT_RULES",
    "CallableRule",
    "CleaningRegistry",
    "CleaningResult",
    "CleaningRule",
    "CollapseBlankLinesRule",
    "NormalizeLineEndingsRule",
    "RegexReplaceRule",
    "RemoveBomRule",
    "RemoveControlCharsRule",
    "RemoveHtmlCommentsRule",
    "StripTrailingWhitespaceRule",
    "TextCleaner",
]
