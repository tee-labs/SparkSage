"""Tests for :class:`SqliteFeedbackStore` -- durable feedback persistence.

Mirrors ``test_feedback.py`` but exercises the SQLite backend, sharing the
store-behaviour assertions (CRUD + filtering + stats + defensive copies) so the
durable backend stays a drop-in for :class:`InMemoryFeedbackStore`.
"""

from __future__ import annotations

import pytest

from sparksage.feedback import (
    FeedbackRating,
    FeedbackRecord,
    FeedbackStore,
    SqliteFeedbackStore,
)


def _rec(
    query="how to deploy",
    rating=FeedbackRating.POSITIVE,
    blocks=None,
    correction=None,
    kb_id="kb1",
):
    return FeedbackRecord(
        query=query,
        answer_text="ans",
        rating=rating,
        block_ids=blocks or [],
        correction=correction,
        kb_id=kb_id,
    )


class TestSqliteFeedbackStore:
    def test_protocol(self):
        assert isinstance(SqliteFeedbackStore(), FeedbackStore)

    def test_add_get_list_delete(self):
        store = SqliteFeedbackStore()
        r = store.add(_rec())
        assert r.feedback_id in store
        assert store.get(r.feedback_id).query == "how to deploy"
        assert len(store.list()) == 1
        assert store.delete(r.feedback_id)
        assert len(store) == 0

    def test_filtering(self):
        store = SqliteFeedbackStore()
        store.add(_rec(query="a", rating=FeedbackRating.POSITIVE))
        store.add(_rec(query="b", rating=FeedbackRating.NEGATIVE))
        assert len(store.list(rating=FeedbackRating.NEGATIVE)) == 1
        assert store.count(kb_id="kb1") == 2
        assert store.count(kb_id="other") == 0

    def test_stats(self):
        store = SqliteFeedbackStore()
        store.add(_rec(rating=FeedbackRating.POSITIVE))
        store.add(_rec(rating=FeedbackRating.NEGATIVE))
        s = store.stats()
        assert s.total == 2
        assert s.positive == 1
        assert s.approval == 0.5

    def test_stats_scoped_by_kb(self):
        store = SqliteFeedbackStore()
        store.add(_rec(rating=FeedbackRating.POSITIVE, kb_id="kb1"))
        store.add(_rec(rating=FeedbackRating.NEGATIVE, kb_id="kb2"))
        s = store.stats(kb_id="kb1")
        assert s.total == 1
        assert s.positive == 1

    def test_block_breakdown(self):
        store = SqliteFeedbackStore()
        store.add(_rec(blocks=["b1"], rating=FeedbackRating.NEGATIVE))
        store.add(_rec(blocks=["b1"], rating=FeedbackRating.POSITIVE))
        bb = store.block_breakdown()
        assert "b1" in bb
        assert bb["b1"].total == 2

    def test_defensive_copy(self):
        store = SqliteFeedbackStore()
        r = store.add(_rec())
        r.query = "mutated"
        assert store.get(r.feedback_id).query == "how to deploy"

    def test_bad_pagination(self):
        store = SqliteFeedbackStore()
        with pytest.raises(ValueError):
            store.list(limit=0)
        with pytest.raises(ValueError):
            store.list(offset=-1)

    def test_get_missing_is_none(self):
        assert SqliteFeedbackStore().get("nope") is None

    def test_invalid_table_name(self):
        with pytest.raises(ValueError):
            SqliteFeedbackStore(table="bad name!")

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "feedback.db"
        s1 = SqliteFeedbackStore(path)
        r = s1.add(_rec(query="persisted"))
        s1.close()
        s2 = SqliteFeedbackStore(path)
        got = s2.get(r.feedback_id)
        assert got is not None
        assert got.query == "persisted"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "fb.db"
        s = SqliteFeedbackStore(path)
        assert path.exists()
        s.close()

    def test_newest_first_ordering(self):
        store = SqliteFeedbackStore()
        first = store.add(_rec(query="first"))
        second = store.add(_rec(query="second"))
        ordered = store.list()
        assert ordered[0].feedback_id == second.feedback_id
        assert ordered[1].feedback_id == first.feedback_id
