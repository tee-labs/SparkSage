"""Tests for the customizable text-cleaning layer.

All tests run fully offline -- cleaning has no optional dependencies. The suite
covers the built-in rules, the escape-hatch rules, source-aware routing in the
registry, the :class:`TextCleaner` orchestrator (defaults + customization), and
end-to-end chaining from :class:`~sparksage.convert.ConversionResult` through to
:class:`~sparksage.generator.IdeaBlockGenerator` via the deterministic
:class:`~sparksage.generator.FakeLLMClient`.
"""

from __future__ import annotations

import pytest

from sparksage.clean import (
    DEFAULT_RULES,
    CallableRule,
    CleaningRegistry,
    CleaningResult,
    CleaningRule,
    CollapseBlankLinesRule,
    NormalizeLineEndingsRule,
    RegexReplaceRule,
    RemoveBomRule,
    RemoveControlCharsRule,
    RemoveHtmlCommentsRule,
    StripTrailingWhitespaceRule,
    TextCleaner,
)
from sparksage.clean.registry import _glob_match
from sparksage.convert import ConversionResult, FakeConverterBackend, MarkdownConverter
from sparksage.generator import FakeLLMClient, IdeaBlockGenerator


# ---------------------------------------------------------------------------- #
# built-in rules
# ---------------------------------------------------------------------------- #
def test_remove_bom_rule():
    assert RemoveBomRule().clean("\ufeffhello") == "hello"
    assert RemoveBomRule().clean("hello") == "hello"


def test_normalize_line_endings_rule():
    rule = NormalizeLineEndingsRule()
    assert rule.clean("a\r\nb\rc") == "a\nb\nc"


def test_remove_control_chars_rule():
    rule = RemoveControlCharsRule()
    assert rule.clean("a\x00b\x07c") == "abc"
    assert rule.clean("a\x07b") == "ab"
    # tab and newline are preserved
    assert rule.clean("a\tb\nc") == "a\tb\nc"
    # bell, form feed, vertical tab, delete removed
    assert rule.clean("a\x0bb\x0cc\x7fd") == "abcd"


def test_strip_trailing_whitespace_rule():
    rule = StripTrailingWhitespaceRule()
    assert rule.clean("a   \nb\t\n  c  ") == "a\nb\n  c"
    # whole-text trim too
    assert rule.clean("  \n hi \n  ") == "hi"


def test_collapse_blank_lines_default_one_blank():
    rule = CollapseBlankLinesRule()  # max_blanks=1
    assert rule.clean("a\n\n\n\nb") == "a\n\nb"
    # already <= 1 blank line: unchanged
    assert rule.clean("a\n\nb") == "a\n\nb"
    assert rule.clean("a\nb") == "a\nb"


def test_collapse_blank_lines_custom_max():
    rule = CollapseBlankLinesRule(max_blanks=2)
    assert rule.clean("a\n\n\n\n\n\nb") == "a\n\n\nb"
    rule0 = CollapseBlankLinesRule(max_blanks=0)
    assert rule0.clean("a\n\n\nb") == "a\nb"


def test_collapse_blank_lines_negative_raises():
    with pytest.raises(ValueError):
        CollapseBlankLinesRule(max_blanks=-1)


def test_remove_html_comments_rule():
    rule = RemoveHtmlCommentsRule()
    assert rule.clean("a<!-- comment -->b") == "ab"
    assert rule.clean("a<!-- multi\nline\ncomment -->b") == "ab"


# ---------------------------------------------------------------------------- #
# escape-hatch rules
# ---------------------------------------------------------------------------- #
def test_regex_replace_rule_remove():
    rule = RegexReplaceRule(r"CONFIDENTIAL")
    assert rule.clean("This is CONFIDENTIAL data") == "This is  data"


def test_regex_replace_rule_with_replacement_and_flags():
    rule = RegexReplaceRule(r"foo", "bar", flags=0)
    assert rule.clean("foo foo") == "bar bar"


def test_regex_replace_rule_count_limit():
    rule = RegexReplaceRule(r"x", "y", count=1)
    assert rule.clean("xx") == "yx"


def test_regex_replace_rule_precompiled_pattern():
    import re

    pat = re.compile(r"\d+")
    rule = RegexReplaceRule(pat, "#")
    assert rule.clean("a1b22c333") == "a#b#c#"


def test_callable_rule_two_args():
    rule = CallableRule(lambda t, s: f"[{s}]{t}")
    assert rule.clean("hi", "a.pdf") == "[a.pdf]hi"


def test_callable_rule_single_arg():
    rule = CallableRule(lambda t: t.upper())
    assert rule.clean("hi", "a.pdf") == "HI"


# ---------------------------------------------------------------------------- #
# cleaning rule protocol
# ---------------------------------------------------------------------------- #
def test_rules_satisfy_protocol():
    for rule in (
        RemoveBomRule(),
        NormalizeLineEndingsRule(),
        RemoveControlCharsRule(),
        StripTrailingWhitespaceRule(),
        CollapseBlankLinesRule(),
        RemoveHtmlCommentsRule(),
        RegexReplaceRule("x"),
        CallableRule(lambda t: t),
    ):
        assert isinstance(rule, CleaningRule)


# ---------------------------------------------------------------------------- #
# CleaningRegistry -- source-aware routing
# ---------------------------------------------------------------------------- #
def test_registry_global_rules_apply_to_everything():
    reg = CleaningRegistry()
    reg.add(CallableRule(lambda t: t + "G"))
    assert reg.clean("x", "a.pdf") == "xG"
    assert reg.clean("x", None) == "xG"


def test_registry_rules_in_registration_order():
    reg = CleaningRegistry()
    reg.add(CallableRule(lambda t: t + "1"))
    reg.add(CallableRule(lambda t: t + "2"))
    assert reg.clean("x") == "x12"


def test_registry_glob_match_basename_and_path():
    reg = CleaningRegistry()
    reg.add_for_glob("*.pdf", CallableRule(lambda t: t + "P"))
    reg.add_for_glob("*.docx", CallableRule(lambda t: t + "D"))
    assert reg.clean("x", "/docs/report.pdf") == "xP"
    assert reg.clean("x", "note.docx") == "xD"
    # non-matching source -> rule skipped
    assert reg.clean("x", "data.csv") == "x"


def test_registry_regex_match():
    reg = CleaningRegistry()
    reg.add_for_regex(r"confluence/.*", CallableRule(lambda t: t + "C"))
    assert reg.clean("x", "confluence/page1") == "xC"
    assert reg.clean("x", "wiki/page1") == "x"


def test_registry_rules_for_lists_applicable_only():
    reg = CleaningRegistry()
    g = CallableRule(lambda t: t)
    p = CallableRule(lambda t: t)
    reg.add(g)
    reg.add_for_glob("*.pdf", p)
    assert reg.rules_for("a.csv") == [g]
    assert reg.rules_for("a.pdf") == [g, p]
    assert reg.rules_for(None) == [g]


def test_registry_len_and_iter():
    reg = CleaningRegistry()
    assert len(reg) == 0
    reg.add(CallableRule(lambda t: t))
    reg.add_for_glob("*.md", CallableRule(lambda t: t))
    assert len(reg) == 2
    assert len(list(reg)) == 2


def test_glob_match_helper_both_forms():
    assert _glob_match("/a/b/report.pdf", "*.pdf") is True
    assert _glob_match("/a/b/report.pdf", "report.pdf") is True
    assert _glob_match("notes.md", "*.pdf") is False


# ---------------------------------------------------------------------------- #
# TextCleaner -- defaults + customization
# ---------------------------------------------------------------------------- #
def test_default_cleaner_normalizes_common_noise():
    cleaner = TextCleaner()
    raw = "\ufeffHello\r\nWorld\x00   \n\n\n\nDone"
    result = cleaner.clean(raw, source="doc.md")
    # BOM gone, CRLF normalized, control char gone, trailing ws stripped, blanks collapsed
    assert result.text == "Hello\nWorld\n\nDone"


def test_clean_result_fields_and_source_ref():
    cleaner = TextCleaner()
    result = cleaner.clean("hi", source="file://x.md", title="X")
    assert isinstance(result, CleaningResult)
    assert result.text == "hi"
    assert result.source == "file://x.md"
    assert result.title == "X"
    ref = result.source_ref
    assert ref.uri == "file://x.md"
    assert ref.title == "X"


def test_clean_text_returns_string_only():
    cleaner = TextCleaner()
    assert cleaner.clean_text("a\n\n\n\nb") == "a\n\nb"


def test_default_rules_are_prepended():
    cleaner = TextCleaner(rules=[CallableRule(lambda t: t + "Z")])
    # defaults first, then custom
    rules = cleaner.rules_for("any.md")
    assert rules[-1].clean("") == "Z"
    assert len(rules) == len(DEFAULT_RULES) + 1


def test_use_defaults_false_gives_empty_pipeline():
    cleaner = TextCleaner(use_defaults=False)
    assert cleaner.clean_text("anything\r\n\x00") == "anything\r\n\x00"


def test_add_global_rule_fluent():
    cleaner = TextCleaner(use_defaults=False)
    out = cleaner.add(CallableRule(lambda t: t.upper()))
    assert out is cleaner
    assert cleaner.clean_text("hi") == "HI"


def test_add_for_glob_filename_routing():
    cleaner = TextCleaner(use_defaults=False)
    cleaner.add(CallableRule(lambda t: t + "-global"))
    cleaner.add_for("*.pdf", CallableRule(lambda t: t + "-pdf"))
    assert cleaner.clean_text("x", source="r.pdf") == "x-global-pdf"
    assert cleaner.clean_text("x", source="r.docx") == "x-global"
    assert cleaner is cleaner.add_for("*.md", CallableRule(lambda t: t))


def test_add_for_regex_filename_routing():
    cleaner = TextCleaner(use_defaults=False)
    cleaner.add_for(r"^tickets/", CallableRule(lambda t: t + "-t"), regex=True)
    assert cleaner.clean_text("x", source="tickets/1") == "x-t"
    assert cleaner.clean_text("x", source="docs/1") == "x"


def test_shared_registry_is_layered():
    reg = CleaningRegistry()
    reg.add(CallableRule(lambda t: t + "-shared"))
    cleaner = TextCleaner(use_defaults=False, registry=reg)
    cleaner.add(CallableRule(lambda t: t + "-extra"))
    assert cleaner.clean_text("x") == "x-shared-extra"


def test_business_scenario_watermark_and_pii_redaction():
    cleaner = TextCleaner()
    cleaner.add(RegexReplaceRule(r"CONFIDENTIAL", ""))
    cleaner.add(RegexReplaceRule(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]"))
    cleaner.add_for("*.pdf", RegexReplaceRule(r"Page \d+ of \d+", ""))
    raw = "CONFIDENTIAL\nMy SSN is 123-45-6789.\nPage 1 of 5\nHello"
    pdf = cleaner.clean(raw, source="report.pdf")
    assert "CONFIDENTIAL" not in pdf.text
    assert "[REDACTED-SSN]" in pdf.text
    assert "Page 1 of 5" not in pdf.text
    # same rules, but a docx does NOT get the page-stripper
    docx = cleaner.clean(raw, source="report.docx")
    assert "Page 1 of 5" in docx.text
    assert "[REDACTED-SSN]" in docx.text


# ---------------------------------------------------------------------------- #
# Pipeline integration: convert -> clean -> generate
# ---------------------------------------------------------------------------- #
def test_clean_result_chains_from_conversion_result():
    raw = "Line one\n\n\n\n\nLine two\x00"
    conv_result = ConversionResult(markdown=raw, source="docs/note.md", title="Note")
    cleaner = TextCleaner()
    cleaned = cleaner.clean_result(conv_result)
    assert cleaned.text == "Line one\n\nLine two"
    assert cleaned.source == "docs/note.md"
    assert cleaned.title == "Note"


def test_clean_results_batch():
    results = [
        ConversionResult(markdown="a\n\n\n\nb", source="a.md"),
        ConversionResult(markdown="\ufeffc", source="b.md"),
    ]
    cleaner = TextCleaner()
    cleaned = cleaner.clean_results(results)
    assert [c.text for c in cleaned] == ["a\n\nb", "c"]
    assert [c.source for c in cleaned] == ["a.md", "b.md"]


def test_full_pipeline_convert_clean_generate():
    raw = "\ufeff# Title\r\n\n\n\nWhat is SparkSage? It is a RAG library.\x00"
    conv = MarkdownConverter(backend=FakeConverterBackend(markdown=raw))
    conv_result = conv.convert("spec.pdf")

    cleaner = TextCleaner()
    cleaner.add_for("*.pdf", RegexReplaceRule(r"# Title", "# Spec"))
    cleaned = cleaner.clean_result(conv_result)

    fake = FakeLLMClient(
        responses=["""[
      {
        "name": "SparkSage",
        "critical_question": "What is SparkSage?",
        "trusted_answer": "A RAG library.",
        "tags": ["technology"],
        "keywords": ["rag"]
      }
    ]"""]
    )
    gen = IdeaBlockGenerator(fake)
    blocks = gen.generate(cleaned.text, source=cleaned.source_ref)
    assert len(blocks) == 1
    assert blocks[0].critical_question == "What is SparkSage?"
    assert blocks[0].source.uri == "spec.pdf"
    # the cleaning actually ran (control char + BOM removed, title rewritten)
    assert "\x00" not in cleaned.text
    assert "\ufeff" not in cleaned.text
    assert "# Spec" in cleaned.text
