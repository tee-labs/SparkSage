"""The text-cleaning orchestrator: raw text -> final document text.

:class:`TextCleaner` sits between conversion and generation:

    file -> MarkdownConverter -> [TextCleaner] -> IdeaBlockGenerator -> blocks

Conversion produces *raw* Markdown (faithful to the source bytes). That text is
seldom generation-ready: it carries BOMs, mixed line endings, leaked control
chars, page headers/footers, watermarks, boilerplate, etc. -- and *which* of
those are noise depends on the business. :class:`TextCleaner` gives one place to
declare that policy through composable, source-aware rules.

Design mirrors the rest of SparkSage: the cleaner depends only on the
:class:`~sparksage.clean.rules.CleaningRule` protocol and a
:class:`~sparksage.clean.registry.CleaningRegistry`, so it is deterministic and
fully unit-testable with no external dependencies.

The emitted :class:`CleaningResult` chains straight into block generation: feed
``result.text`` as the text and ``result.source_ref`` as provenance to
:class:`~sparksage.generator.IdeaBlockGenerator`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from sparksage.clean.registry import CleaningRegistry
from sparksage.clean.rules import (
    CleaningRule,
    CollapseBlankLinesRule,
    NormalizeLineEndingsRule,
    RemoveBomRule,
    RemoveControlCharsRule,
    StripTrailingWhitespaceRule,
)
from sparksage.convert.converter import ConversionResult
from sparksage.schema.source import SourceRef

#: Sensible, format-agnostic normalization applied by default. Business-specific
#: rules are layered on top via :meth:`TextCleaner.add` / :meth:`TextCleaner.add_for`.
#: Order matters: normalize bytes/line-endings first, then control chars, then
#: whitespace, then blank-line collapsing.
DEFAULT_RULES: tuple[CleaningRule, ...] = (
    RemoveBomRule(),
    NormalizeLineEndingsRule(),
    RemoveControlCharsRule(),
    StripTrailingWhitespaceRule(),
    CollapseBlankLinesRule(max_blanks=1),
)


@dataclass
class CleaningResult:
    """The cleaned text of a single document plus its provenance.

    Attributes
    ----------
    text:
        The cleaned Markdown (the only payload downstream cares about).
    source:
        Stable string descriptor of the input (file path, URI, ...). Used as the
        :attr:`~sparksage.schema.source.SourceRef.uri` and, crucially, to route
        source-specific cleaning rules.
    title:
        Document title carried through from conversion, if any.
    """

    text: str
    source: str
    title: str | None = None

    @property
    def source_ref(self) -> SourceRef:
        """A :class:`SourceRef` pointing back at the original document."""
        return SourceRef(uri=self.source, title=self.title)


class TextCleaner:
    r"""Clean raw document text via composable, source-aware rules.

    Parameters
    ----------
    rules:
        Extra global rules appended *after* the defaults (unless
        ``use_defaults`` is ``False``).
    registry:
        A pre-built :class:`CleaningRegistry` to use instead of a fresh one.
        When supplied, ``rules`` / ``use_defaults`` are still layered on top, so
        callers can share one registry across cleaners.
    use_defaults:
        When ``True`` (default) prepend :data:`DEFAULT_RULES`. Set to ``False``
        for full manual control over the rule set.

    Examples
    --------
    >>> from sparksage import TextCleaner, RegexReplaceRule
    >>> cleaner = TextCleaner()
    >>> cleaner.add(RegexReplaceRule("CONFIDENTIAL", ""))   # global rule
    >>> cleaner.add_for("*.pdf", RegexReplaceRule(r"Page \d+", ""))  # PDF only

    Chain straight from conversion::

        result = cleaner.clean_result(conv_result)
        blocks = IdeaBlockGenerator(client).generate(
            result.text, source=result.source_ref,
        )
    """

    def __init__(
        self,
        rules: Iterable[CleaningRule] | None = None,
        *,
        registry: CleaningRegistry | None = None,
        use_defaults: bool = True,
    ) -> None:
        self._registry = registry if registry is not None else CleaningRegistry()
        if use_defaults:
            for rule in DEFAULT_RULES:
                self._registry.add(rule)
        if rules:
            for rule in rules:
                self._registry.add(rule)

    @property
    def registry(self) -> CleaningRegistry:
        """The underlying rule registry (mainly for inspection/testing)."""
        return self._registry

    def add(self, rule: CleaningRule) -> TextCleaner:
        """Append a global rule (applies to every source). Returns ``self``."""
        self._registry.add(rule)
        return self

    def add_for(
        self,
        pattern: str,
        rule: CleaningRule,
        *,
        regex: bool = False,
    ) -> TextCleaner:
        """Append a rule that applies only to matching sources.

        Parameters
        ----------
        pattern:
            A glob by default (matched against both the full path and the file
            name), or a regex when ``regex=True``.
        """
        if regex:
            self._registry.add_for_regex(pattern, rule)
        else:
            self._registry.add_for_glob(pattern, rule)
        return self

    def rules_for(self, source: str | None) -> list[CleaningRule]:
        """Rules that would apply to ``source``, in execution order."""
        return self._registry.rules_for(source)

    def clean(
        self,
        text: str,
        *,
        source: str | None = None,
        title: str | None = None,
    ) -> CleaningResult:
        """Clean ``text`` and wrap it in a :class:`CleaningResult`.

        ``source`` both labels the result (provenance) and selects which
        source-specific rules run.
        """
        cleaned = self._registry.clean(text, source)
        return CleaningResult(text=cleaned, source=source or "", title=title)

    def clean_text(self, text: str, source: str | None = None) -> str:
        """Convenience: return just the cleaned text for ``source``."""
        return self._registry.clean(text, source)

    def clean_result(self, result: ConversionResult) -> CleaningResult:
        """Clean a :class:`~sparksage.convert.ConversionResult` end-to-end.

        Feeds ``result.markdown`` through the rules, carrying ``source`` and
        ``title`` straight from the converter. The natural bridge between
        conversion and generation.
        """
        return self.clean(
            result.markdown, source=result.source, title=result.title
        )

    def clean_results(
        self, results: Iterable[ConversionResult]
    ) -> list[CleaningResult]:
        """Batch-clean an iterable of conversion results."""
        return [self.clean_result(r) for r in results]
