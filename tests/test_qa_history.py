"""Tests for the QA conversation-history layer (the persisted query log)."""

from __future__ import annotations

import pytest

from sparksage.qa import InMemoryQASessionStore, QATurn, TurnRole


def _turn(role=TurnRole.USER, content="q", kb_id="kb1", result=None):
    return QATurn(role=role, content=content, kb_id=kb_id, result=result)


class TestQATurn:
    def test_content_stripped(self):
        t = _turn(content="   hello   ")
        assert t.content == "hello"

    def test_query_property_from_result(self):
        t = _turn(role=TurnRole.ASSISTANT, content="ans", result={"query": "q?"})
        assert t.query == "q?"

    def test_query_property_none_for_user_turn(self):
        assert _turn().query is None

    def test_extra_field_rejected(self):
        with pytest.raises(ValueError):
            QATurn(role=TurnRole.USER, content="q", bogus=1)


class TestStore:
    def test_add_list_newest_first(self):
        store = InMemoryQASessionStore()
        store.add_turn(_turn(content="first"))
        store.add_turn(_turn(content="second"))
        page = store.list()
        assert [t.content for t in page] == ["second", "first"]

    def test_kb_scoping(self):
        store = InMemoryQASessionStore()
        store.add_turn(_turn(content="kb1-turn", kb_id="kb1"))
        store.add_turn(_turn(content="kb2-turn", kb_id="kb2"))
        assert store.count(kb_id="kb1") == 1
        assert [t.content for t in store.list(kb_id="kb2")] == ["kb2-turn"]
        assert len(store) == 2

    def test_clear_all(self):
        store = InMemoryQASessionStore()
        store.add_turn(_turn())
        store.clear()
        assert len(store) == 0

    def test_clear_scoped_per_kb(self):
        store = InMemoryQASessionStore()
        store.add_turn(_turn(content="a", kb_id="kb1"))
        store.add_turn(_turn(content="b", kb_id="kb2"))
        store.clear(kb_id="kb1")
        assert store.count(kb_id="kb1") == 0
        assert store.count(kb_id="kb2") == 1

    def test_pagination(self):
        store = InMemoryQASessionStore()
        for i in range(5):
            store.add_turn(_turn(content=f"t{i}"))
        page = store.list(limit=2, offset=0)
        assert [t.content for t in page] == ["t4", "t3"]
        page = store.list(limit=2, offset=2)
        assert [t.content for t in page] == ["t2", "t1"]

    def test_invalid_pagination_rejected(self):
        store = InMemoryQASessionStore()
        store.add_turn(_turn())
        with pytest.raises(ValueError):
            store.list(limit=0)
        with pytest.raises(ValueError):
            store.list(offset=-1)

    def test_defensive_copy(self):
        store = InMemoryQASessionStore()
        original = _turn(content="q")
        stored = store.add_turn(original)
        stored.content = "mutated"
        assert store.list()[0].content == "q"
        page = store.list()
        page[0].content = "mutated"
        assert store.list()[0].content == "q"
