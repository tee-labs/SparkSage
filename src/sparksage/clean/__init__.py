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
from sparksage.clean.manager import (
    CleaningRuleManager,
    CleaningTestResult,
    RuleStatus,
)
from sparksage.clean.models import CleaningRuleRecord, PatternKind
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
from sparksage.clean.script import RestrictedScriptRule
from sparksage.clean.store import CleaningRuleStore, InMemoryCleaningRuleStore

__all__ = [
    "DEFAULT_RULES",
    "CallableRule",
    "CleaningRegistry",
    "CleaningResult",
    "CleaningRule",
    "CleaningRuleManager",
    "CleaningRuleRecord",
    "CleaningRuleStore",
    "CleaningTestResult",
    "CollapseBlankLinesRule",
    "InMemoryCleaningRuleStore",
    "NormalizeLineEndingsRule",
    "PatternKind",
    "RegexReplaceRule",
    "RemoveBomRule",
    "RemoveControlCharsRule",
    "RemoveHtmlCommentsRule",
    "RestrictedScriptRule",
    "RuleStatus",
    "StripTrailingWhitespaceRule",
    "TextCleaner",
]
