"""Tests for the feedback layer: store, stats, and self-healing signals."""

from __future__ import annotations

import pytest

from sparksage.feedback import (
    FeedbackRating,
    FeedbackRecord,
    InMemoryFeedbackStore,
    extract_healing_signals,
    extract_low_recall,
    extract_split_candidates,
)


def _rec(query="how to deploy", rating=FeedbackRating.POSITIVE, blocks=None, correction=None):
    return FeedbackRecord(
        query=query,
        answer_text="ans",
        rating=rating,
        block_ids=blocks or [],
        correction=correction,
        kb_id="kb1",
    )


class TestFeedbackRecord:
    def test_empty_query_rejected(self):
        with pytest.raises(ValueError):
            FeedbackRecord(query="  ", rating=FeedbackRating.POSITIVE)

    def test_correction_stripped_to_none(self):
        r = _rec(correction="   ")
        assert r.correction is None

    def test_is_negative(self):
        assert _rec(rating=FeedbackRating.NEGATIVE).is_negative
        assert _rec(rating=FeedbackRating.CORRECTED).is_negative
        assert not _rec(rating=FeedbackRating.POSITIVE).is_negative


class TestStore:
    def test_add_get_list_delete(self):
        store = InMemoryFeedbackStore()
        r = store.add(_rec())
        assert r.feedback_id in store
        assert store.get(r.feedback_id).query == "how to deploy"
        assert len(store.list()) == 1
        assert store.delete(r.feedback_id)
        assert len(store) == 0

    def test_filtering(self):
        store = InMemoryFeedbackStore()
        store.add(_rec(query="a", rating=FeedbackRating.POSITIVE))
        store.add(_rec(query="b", rating=FeedbackRating.NEGATIVE))
        assert len(store.list(rating=FeedbackRating.NEGATIVE)) == 1
        assert store.count(kb_id="kb1") == 2
        assert store.count(kb_id="other") == 0

    def test_stats(self):
        store = InMemoryFeedbackStore()
        store.add(_rec(rating=FeedbackRating.POSITIVE))
        store.add(_rec(rating=FeedbackRating.NEGATIVE))
        s = store.stats()
        assert s.total == 2
        assert s.positive == 1
        assert s.approval == 0.5

    def test_block_breakdown(self):
        store = InMemoryFeedbackStore()
        store.add(_rec(blocks=["b1"], rating=FeedbackRating.NEGATIVE))
        store.add(_rec(blocks=["b1"], rating=FeedbackRating.POSITIVE))
        bb = store.block_breakdown()
        assert "b1" in bb
        assert bb["b1"].total == 2

    def test_defensive_copy(self):
        store = InMemoryFeedbackStore()
        r = store.add(_rec())
        r.query = "mutated"
        assert store.get(r.feedback_id).query == "how to deploy"

    def test_bad_pagination(self):
        store = InMemoryFeedbackStore()
        with pytest.raises(ValueError):
            store.list(limit=0)


class TestHealing:
    def test_low_recall_groups_repeated(self):
        records = [
            _rec(query="how to deploy", blocks=[]),
            _rec(query="How To Deploy", blocks=[]),
            _rec(query="other question", blocks=[]),
        ]
        signals = extract_low_recall(records, min_hits=1, min_occurrences=2)
        assert len(signals) == 1
        assert signals[0].query == "how to deploy"
        assert signals[0].occurrences == 2

    def test_split_candidates_threshold(self):
        from sparksage.feedback import FeedbackStats

        breakdown = {
            "b1": FeedbackStats(total=4, positive=1, negative=2, corrected=1),  # 0.75 bad
            "b2": FeedbackStats(total=5, positive=4, negative=1, corrected=0),  # 0.2 bad
        }
        signals = extract_split_candidates(breakdown, min_records=2, min_bad_ratio=0.5)
        assert [s.block_id for s in signals] == ["b1"]
        assert signals[0].bad_ratio == 0.75

    def test_extract_healing_signals_end_to_end(self):
        store = InMemoryFeedbackStore()
        # two low-recall feedbacks on the same query
        store.add(_rec(query="how to deploy", blocks=[]))
        store.add(_rec(query="how to deploy", blocks=[]))
        # a block with bad feedback
        store.add(_rec(query="q2", rating=FeedbackRating.NEGATIVE, blocks=["b1"]))
        store.add(_rec(query="q3", rating=FeedbackRating.CORRECTED, blocks=["b1"]))

        report = extract_healing_signals(
            store, min_hits=1, min_occurrences=2, min_records_per_block=2, min_bad_ratio=0.5
        )
        assert len(report.low_recall) == 1
        assert report.low_recall[0].query == "how to deploy"
        assert len(report.split_candidates) == 1
        assert report.split_candidates[0].block_id == "b1"
