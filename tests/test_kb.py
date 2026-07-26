"""Tests for the KnowledgeBase aggregate root: consistency + reindex + scoping.

All tests run offline via :class:`FakeEmbeddingClient`.
"""

from __future__ import annotations

import pytest

from sparksage import BlockEmbedder, FakeEmbeddingClient, IdeaBlock, Tag, new_record
from sparksage.kb import (
    InMemoryKnowledgeBaseStore,
    KnowledgeBase,
    KnowledgeBaseInfo,
)
from sparksage.retrieve.models import RetrievalFilter


def _block(name="A", body="deploy via pip"):
    return IdeaBlock(
        name=name,
        critical_question=f"what is {name}?",
        trusted_answer=body,
        tags=[Tag.IMPORTANT],
    )


def _make_kb():
    return KnowledgeBase(
        info=KnowledgeBaseInfo(name="ops"),
        embedder=BlockEmbedder(FakeEmbeddingClient(dimension=64)),
    )


class TestKnowledgeBaseInfo:
    def test_defaults(self):
        info = KnowledgeBaseInfo(name="x")
        assert info.kb_id
        assert info.language == "en"

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            KnowledgeBaseInfo(name="  ")

    def test_tags_normalized(self):
        info = KnowledgeBaseInfo(name="x", tags=["a", "a", " b "])
        assert info.tags == ["a", "b"]


class TestStore:
    def test_crud(self):
        store = InMemoryKnowledgeBaseStore()
        info = KnowledgeBaseInfo(name="x")
        saved = store.save(info)
        assert saved.kb_id in store
        assert len(store) == 1
        assert store.get(saved.kb_id).name == "x"
        assert store.delete(saved.kb_id)
        assert saved.kb_id not in store
        assert len(store) == 0

    def test_list_pagination(self):
        store = InMemoryKnowledgeBaseStore()
        for i in range(5):
            store.save(KnowledgeBaseInfo(name=f"k{i}"))
        assert len(store.list(limit=2)) == 2
        assert len(store.list(limit=10, offset=3)) == 2

    def test_defensive_copy(self):
        store = InMemoryKnowledgeBaseStore()
        info = KnowledgeBaseInfo(name="x")
        saved = store.save(info)
        saved.name = "mutated"
        assert store.get(info.kb_id).name == "x"


class TestKnowledgeBase:
    def test_add_blocks_stamps_kb_id_and_indexes(self):
        kb = _make_kb()
        b = _block()
        kb.add_blocks([b])
        assert b.kb_id == kb.kb_id
        assert kb.block_count() == 1
        assert len(kb.store) == 1
        hits = kb.search("deploy", k=3)
        assert len(hits.chunks) == 1
        assert hits.chunks[0].block.kb_id == kb.kb_id

    def test_remove_block_cleans_all_indexes(self):
        kb = _make_kb()
        b = _block()
        kb.add_blocks([b])
        assert kb.remove_block(str(b.id))
        assert kb.block_count() == 0
        assert len(kb.store) == 0
        assert str(b.id) not in kb.lexical

    def test_remove_nonexistent_returns_false(self):
        kb = _make_kb()
        assert not kb.remove_block("nope")

    def test_document_cascade_delete(self):
        kb = _make_kb()
        rec = new_record(body_markdown="body", source="file://d.md")
        b1, b2 = _block("A"), _block("B")
        kb.add_document(rec, blocks=[b1, b2])
        assert kb.block_count() == 2
        assert len(kb.store) == 2
        assert kb.document_count() == 1

        assert kb.remove_document(rec.doc_id)
        assert kb.block_count() == 0
        assert len(kb.store) == 0
        assert kb.document_count() == 0

    def test_blocks_for_document(self):
        kb = _make_kb()
        rec = new_record(body_markdown="body", source="file://d.md")
        b1, b2 = _block("A"), _block("B")
        kb.add_document(rec, blocks=[b1, b2])
        linked = kb.blocks_for_document(rec.doc_id)
        assert {b.name for b in linked} == {"A", "B"}

    def test_update_document_no_change_skips_reindex(self):
        kb = _make_kb()
        rec = new_record(body_markdown="same body", source="file://d.md")
        b1 = _block("A")
        kb.add_document(rec, blocks=[b1])
        assert kb.block_count() == 1

        # update with same content_hash and no new blocks -> blocks untouched
        kb.update_document(rec.doc_id)
        assert kb.block_count() == 1

    def test_update_document_content_change_reindexes(self):
        kb = _make_kb()
        rec = new_record(body_markdown="body v1", source="file://d.md")
        b_old = _block("A", "old answer")
        kb.add_document(rec, blocks=[b_old])
        assert kb.block_count() == 1

        new_rec = rec.model_copy(update={"body_markdown": "body v2"})
        b_new = _block("A", "new answer")
        kb.update_document(rec.doc_id, record=new_rec, blocks=[b_new])
        assert kb.block_count() == 1  # old removed, new added
        assert kb.get_block(str(b_old.id)) is None
        assert kb.get_block(str(b_new.id)) is not None

    def test_update_nonexistent_raises(self):
        kb = _make_kb()
        with pytest.raises(KeyError):
            kb.update_document("nope")

    def test_reindex_rebuilds_from_registry(self):
        kb = _make_kb()
        b1, b2 = _block("A"), _block("B")
        kb.add_blocks([b1, b2])
        # corrupt the store by removing one vector manually
        kb.store.remove(str(b1.id))
        assert len(kb.store) == 1
        n = kb.reindex()
        assert n == 2
        assert len(kb.store) == 2

    def test_search_filter_by_kb(self):
        kb = _make_kb()
        kb.add_blocks([_block("A")])
        result = kb.search("deploy", k=3, filter=RetrievalFilter(kb_id=kb.kb_id))
        assert len(result.chunks) == 1
