"""Tests for the cleaning-rule store / manager / persistence layer.

The custom cleaning layer's persistence + orchestration: CRUD over
:class:`CleaningRuleRecord`, the :class:`CleaningRuleManager` rebuild of the
live :class:`TextCleaner` from the store, and the SQLite backend's reload
semantics. The store / manager are pure stdlib, so these run without the
optional ``[clean-script]`` extra; the paths that compile scripts (reload +
test) are additionally guarded by ``importorskip("RestrictedPython")`` so the
full suite stays green when the extra is absent.
"""

from __future__ import annotations

import pytest

from sparksage import RegexReplaceRule
from sparksage.clean.cleaner import TextCleaner
from sparksage.clean.manager import CleaningRuleManager
from sparksage.clean.models import CleaningRuleRecord, PatternKind
from sparksage.clean.store import InMemoryCleaningRuleStore

CODE = "def clean(text, source=None):\n    return text.replace('CONFIDENTIAL', '[REDACTED]')\n"


def _rec(**kw) -> CleaningRuleRecord:
    base = dict(name="r", code=CODE)
    base.update(kw)
    return CleaningRuleRecord(**base)


# ---------------------------------------------------------------------------- #
# store CRUD
# ---------------------------------------------------------------------------- #
class TestInMemoryStore:
    def test_add_get_roundtrip(self):
        s = InMemoryCleaningRuleStore()
        r = s.add(_rec(name="a"))
        assert s.get(r.rule_id) is not None
        assert s.get("missing") is None

    def test_list_preserves_insertion_order(self):
        s = InMemoryCleaningRuleStore()
        a = s.add(_rec(name="a"))
        b = s.add(_rec(name="b"))
        ids = [r.rule_id for r in s.list(limit=10)]
        assert ids == [a.rule_id, b.rule_id]

    def test_list_validates_paging(self):
        s = InMemoryCleaningRuleStore()
        s.add(_rec())
        with pytest.raises(ValueError):
            s.list(limit=0)
        with pytest.raises(ValueError):
            s.list(offset=-1)

    def test_update_replaces_fields(self):
        s = InMemoryCleaningRuleStore()
        r = s.add(_rec(name="a"))
        changed = r.model_copy(update={"name": "b", "code": "def clean(text):\n    return text\n"})
        updated = s.update(changed)
        assert updated.name == "b"
        assert s.get(r.rule_id).name == "b"

    def test_update_missing_raises(self):
        with pytest.raises(KeyError):
            InMemoryCleaningRuleStore().update(_rec())

    def test_delete(self):
        s = InMemoryCleaningRuleStore()
        r = s.add(_rec())
        assert s.delete(r.rule_id) is True
        assert s.delete(r.rule_id) is False
        assert len(s) == 0

    def test_contains_and_len(self):
        s = InMemoryCleaningRuleStore()
        r = s.add(_rec())
        assert r.rule_id in s
        assert len(s) == 1

    def test_defensive_copies(self):
        s = InMemoryCleaningRuleStore()
        stored = s.add(_rec(name="a"))
        stored.name = "mutated"
        assert s.get(stored.rule_id).name == "a"


# ---------------------------------------------------------------------------- #
# manager CRUD + cleaner rebuild (no script compilation needed for store ops)
# ---------------------------------------------------------------------------- #
class TestManagerCRUD:
    def test_defaults_to_in_memory_store(self):
        mgr = CleaningRuleManager()
        assert isinstance(mgr.store, InMemoryCleaningRuleStore)
        assert isinstance(mgr.cleaner, TextCleaner)

    def test_create_then_list(self):
        mgr = CleaningRuleManager()
        r = mgr.create_rule(_rec(name="a"))
        assert len(mgr.list_rules()) == 1
        assert mgr.get_rule(r.rule_id).name == "a"

    def test_update_and_delete(self):
        mgr = CleaningRuleManager()
        r = mgr.create_rule(_rec(name="a"))
        mgr.update_rule(r.model_copy(update={"name": "b"}))
        assert mgr.get_rule(r.rule_id).name == "b"
        assert mgr.delete_rule(r.rule_id) is True
        assert mgr.get_rule(r.rule_id) is None

    def test_base_cleaner_rules_preserved(self):
        base = TextCleaner()
        base.add(RegexReplaceRule("FOO", "BAR"))
        mgr = CleaningRuleManager(base_cleaner=base)
        assert "BAR" in mgr.cleaner.clean_text("FOO")

    def test_disabled_rule_not_flagged(self):
        mgr = CleaningRuleManager()
        r = mgr.create_rule(
            _rec(name="bad", code="def clean(text):\n    return eval(text)\n", enabled=False)
        )
        st = mgr.status_of(mgr.get_rule(r.rule_id))
        assert st.compiled is False
        assert st.error is None  # disabled -> no compile attempt, no error


# ---------------------------------------------------------------------------- #
# script-compilation paths (need the optional [clean-script] extra)
# ---------------------------------------------------------------------------- #
class TestScriptRebuild:
    def test_global_rule_applied_after_create(self):
        pytest.importorskip("RestrictedPython")
        mgr = CleaningRuleManager()
        mgr.create_rule(_rec(name="redact"))
        assert mgr.cleaner.clean_text("CONFIDENTIAL report") == "[REDACTED] report"

    def test_source_specific_rule_routed(self):
        pytest.importorskip("RestrictedPython")
        mgr = CleaningRuleManager()
        mgr.create_rule(_rec(name="pdf", source_pattern="*.pdf", pattern_kind=PatternKind.GLOB))
        assert mgr.cleaner.clean_text("CONFIDENTIAL", source="x.pdf") == "[REDACTED]"
        assert mgr.cleaner.clean_text("CONFIDENTIAL", source="x.docx") == "CONFIDENTIAL"

    def test_compile_error_rule_skipped_not_stored_applied(self):
        pytest.importorskip("RestrictedPython")
        mgr = CleaningRuleManager()
        mgr.create_rule(_rec(name="bad", code="def clean(text):\n    return eval(text)\n"))
        # the bad rule does not break the cleaner (fail-open); other text passes through
        assert mgr.cleaner.clean_text("hello") == "hello"
        st = [s for s in mgr.list_with_status(limit=100) if not s.compiled]
        assert len(st) == 1

    def test_test_rule_returns_output(self):
        pytest.importorskip("RestrictedPython")
        mgr = CleaningRuleManager()
        res = mgr.test_rule(
            "def clean(text, source=None):\n    return text.upper()\n",
            "hello",
        )
        assert res.ok is True
        assert res.output == "HELLO"
        assert res.error is None
        assert res.elapsed_ms >= 0

    def test_test_rule_reports_compile_error(self):
        pytest.importorskip("RestrictedPython")
        mgr = CleaningRuleManager()
        res = mgr.test_rule("def clean(text):\n    return eval(text)\n", "hi")
        assert res.ok is False
        assert res.error is not None
        assert res.output == "hi"  # unchanged on failure


# ---------------------------------------------------------------------------- #
# sqlite backend
# ---------------------------------------------------------------------------- #
class TestSqliteStore:
    def test_roundtrip_and_reload(self, tmp_path):
        from sparksage.clean.backends import SqliteCleaningRuleStore

        p = tmp_path / "c.db"
        s1 = SqliteCleaningRuleStore(p)
        a = s1.add(_rec(name="a"))
        b = s1.add(_rec(name="b", source_pattern="*.pdf", pattern_kind=PatternKind.GLOB))
        assert len(s1) == 2

        # a fresh instance over the same file reloads every record
        s2 = SqliteCleaningRuleStore(p)
        assert len(s2) == 2
        assert s2.get(a.rule_id).name == "a"
        ids = [r.rule_id for r in s2.list(limit=10)]
        assert ids == [a.rule_id, b.rule_id]  # insertion order preserved across restart

    def test_update_preserves_order(self, tmp_path):
        from sparksage.clean.backends import SqliteCleaningRuleStore

        s = SqliteCleaningRuleStore(tmp_path / "c.db")
        a = s.add(_rec(name="a"))
        s.add(_rec(name="b"))
        s.update(a.model_copy(update={"name": "a2"}))
        order = [r.name for r in s.list(limit=10)]
        assert order == ["a2", "b"]

    def test_delete_and_count(self, tmp_path):
        from sparksage.clean.backends import SqliteCleaningRuleStore

        s = SqliteCleaningRuleStore(tmp_path / "c.db")
        a = s.add(_rec(name="a"))
        assert s.count() == 1
        assert s.delete(a.rule_id) is True
        assert s.count() == 0
        assert s.delete(a.rule_id) is False

    def test_invalid_table_rejected(self, tmp_path):
        from sparksage.clean.backends import SqliteCleaningRuleStore

        with pytest.raises(ValueError):
            SqliteCleaningRuleStore(tmp_path / "c.db", table="bad name!")
