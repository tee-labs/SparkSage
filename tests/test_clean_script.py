"""Tests for :class:`RestrictedScriptRule`, the sandboxed script cleaning layer.

The script layer is the escape hatch for multi-branch, source-aware business
cleaning that declarative rules cannot express. These tests exercise the
RestrictedPython integration end-to-end: multi-branch rules, source-based
routing, sandbox escapes blocked at compile time, and the safety rails
(wall-clock timeout, time-boxed regex, output caps, fail-open error isolation).

Guarded by ``importorskip`` so the suite stays runnable without the optional
``[clean-script]`` extra.
"""

from __future__ import annotations

import time

import pytest

from sparksage.clean import CleaningRegistry, RestrictedScriptRule

pytest.importorskip("RestrictedPython")

DROP_CHAPTERS = """
def clean(text, source=None):
    parts = []
    keep = False
    for line in text.split("\\n"):
        if line.startswith("# "):
            keep = line.startswith(("# 第三章", "# 第五章"))
        if keep:
            parts.append(line)
    return "\\n".join(parts)
"""


def test_basic_replacement_rule():
    rule = RestrictedScriptRule(
        "def clean(text, source=None):\n"
        "    return text.replace('CONFIDENTIAL', '[REDACTED]')\n"
    )
    assert rule.clean("CONFIDENTIAL report") == "[REDACTED] report"
    assert rule.last_error is None


def test_multi_branch_section_extraction():
    text = "# 第一章\nintro\n# 第三章\nkeep me\n# 第二章\ndrop\n# 第五章\nkeep too\n"
    rule = RestrictedScriptRule(DROP_CHAPTERS)
    assert rule.clean(text) == "# 第三章\nkeep me\n# 第五章\nkeep too\n"


def test_source_aware_branching():
    code = (
        "def clean(text, source=None):\n"
        "    if source and source.endswith('.xlsx'):\n"
        "        return text.replace('|', ' ')\n"
        "    return text.replace('TODO', '')\n"
    )
    rule = RestrictedScriptRule(code)
    assert rule.clean("a|b", source="ops/report.xlsx") == "a b"
    assert rule.clean("TODO: fix", source="ops/manual.md") == ": fix"


def test_sandbox_blocks_import_at_runtime():
    # RestrictedPython rejects `import` by simply not providing __import__; the
    # failure surfaces at construction as a clear ValueError (fail-fast at
    # config time, before any ingest).
    code = "import os\ndef clean(text, source=None):\n    return text\n"
    with pytest.raises(ValueError, match="__import__"):
        RestrictedScriptRule(code)


def test_sandbox_blocks_eval_and_dunder_access():
    with pytest.raises(ValueError):
        RestrictedScriptRule("def clean(text, source=None):\n    return eval(text)\n")
    with pytest.raises(ValueError):
        RestrictedScriptRule("def clean(text, source=None):\n    return text.__class__\n")


def test_runtime_escape_blocked_fail_open():
    # getattr/open/globals are simply absent from safe_builtins -> NameError,
    # which the rule isolates by returning the input unchanged.
    code = "def clean(text, source=None):\n    return open('/etc/passwd').read()\n"
    rule = RestrictedScriptRule(code)
    out = rule.clean("hello")
    assert out == "hello"
    assert rule.last_error is not None and "open" in rule.last_error


def test_infinite_loop_times_out_and_fails_open():
    code = "def clean(text, source=None):\n    while True:\n        pass\n"
    rule = RestrictedScriptRule(code, timeout=0.3)
    start = time.time()
    out = rule.clean("hello")
    assert out == "hello"
    assert rule.last_error == "timed out after 0.3s"
    assert time.time() - start < 2.0


def test_regex_is_time_boxed_against_redos():
    # catastrophic backtrace on stdlib `re`/bare `regex` would hang the process;
    # the sandboxed `re` forces a timeout so the match aborts and fails open.
    code = "def clean(text, source=None):\n    return re.sub(r'(a+)+b', '', text)\n"
    rule = RestrictedScriptRule(code, timeout=0.5)
    start = time.time()
    out = rule.clean("a" * 200_000)
    assert out == "a" * 200_000
    assert rule.last_error is not None and "TimeoutError" in rule.last_error
    assert time.time() - start < 5.0


def test_output_size_cap_fails_open():
    rule = RestrictedScriptRule(
        "def clean(text, source=None):\n    return text * 10\n",
        max_output_chars=100,
    )
    out = rule.clean("x" * 20)
    assert out == "x" * 20
    assert "output too large" in (rule.last_error or "")


def test_input_size_cap_skips_script():
    rule = RestrictedScriptRule(
        "def clean(text, source=None):\n    return 'ran'\n",
        max_input_chars=10,
    )
    assert rule.clean("x" * 20) == "x" * 20
    assert "input too large" in (rule.last_error or "")


def test_missing_clean_raises():
    with pytest.raises(ValueError, match="must define a callable 'clean"):
        RestrictedScriptRule("x = 1\n")


def test_single_argument_clean_supported():
    rule = RestrictedScriptRule("def clean(text):\n    return text.strip()\n")
    assert rule.clean("  hi  ") == "hi"


def test_usable_in_registry_with_matcher():
    registry = CleaningRegistry()
    registry.add_for_glob("*.md", RestrictedScriptRule(DROP_CHAPTERS))
    text = "# 第一章\nintro\n# 第三章\nkeep\n"
    assert registry.clean(text, source="ops/guide.md") == "# 第三章\nkeep\n"
    assert registry.clean(text, source="ops/guide.xlsx") == text


def test_comprehensions_and_builtins_available():
    code = (
        "def clean(text, source=None):\n"
        "    return ' '.join(sorted(w for w in text.split() if len(w) > 3))\n"
    )
    rule = RestrictedScriptRule(code)
    assert rule.clean("ab cd longword other") == "longword other"
