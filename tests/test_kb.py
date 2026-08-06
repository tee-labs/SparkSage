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
        assert info.language == "zh"

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

    def test_incremental_lexical_matches_full_rebuild(self):
        # add_blocks uses the incremental BM25Retriever.add path; the resulting
        # lexical index must be byte-for-byte identical to a full rebuild.
        kb = _make_kb()
        blocks = [
            _block("A", "deploy the service"),
            _block("B", "scale the cluster"),
            _block("C", "monitor metrics"),
        ]
        for b in blocks:
            kb.add_blocks([b])

        lex = kb.lexical
        full_ids = list(lex._ids)
        full_df = dict(lex._df)
        full_avgdl = lex._avgdl

        kb.lexical.index(kb.blocks())
        assert list(kb.lexical._ids) == full_ids
        assert kb.lexical._df == full_df
        assert kb.lexical._avgdl == pytest.approx(full_avgdl)

    def test_remove_block_uses_incremental_remove(self):
        kb = _make_kb()
        blocks = [_block("A", "deploy"), _block("B", "scale"), _block("C", "monitor")]
        kb.add_blocks(blocks)
        assert kb.block_count() == 3

        removed_id = str(blocks[1].id)
        assert kb.remove_block(removed_id)
        assert kb.block_count() == 2
        assert removed_id not in kb.lexical
        result = kb.search("deploy", k=3)
        ids = {str(c.block.id) for c in result.chunks}
        assert removed_id not in ids
        assert str(blocks[0].id) in ids

    def test_repeated_add_does_not_quadratic_rebuild(self):
        # sanity: adding K documents across an existing registry must not raise
        # and must leave the lexical index consistent with the registry size.
        kb = _make_kb()
        n = 25
        for i in range(n):
            kb.add_blocks([_block(f"B{i}", f"content number {i} deploy")])
        assert kb.block_count() == n
        assert len(kb.lexical) == n

    def test_search_filter_by_kb(self):
        kb = _make_kb()
        kb.add_blocks([_block("A")])
        result = kb.search("deploy", k=3, filter=RetrievalFilter(kb_id=kb.kb_id))
        assert len(result.chunks) == 1

    def test_orphaned_blocks_none_for_linked_documents(self):
        kb = _make_kb()
        rec = new_record(body_markdown="body", source="file://d.md")
        kb.add_document(rec, blocks=[_block("A")])
        assert kb.orphaned_blocks() == []
        assert kb.remove_orphaned_blocks() == 0

    def test_orphaned_block_kept_when_document_present(self):
        # a block relinked to a doc that still exists in the store must survive
        kb = _make_kb()
        rec = new_record(body_markdown="body", source="file://d.md")
        b = _block("A")
        kb.add_blocks([b], doc_id=rec.doc_id)
        kb.document_store.save(rec)
        assert kb.orphaned_blocks() == []

    def test_remove_orphaned_blocks_cleans_deleted_documents(self):
        kb = _make_kb()
        rec = new_record(body_markdown="body", source="file://d.md")
        b1, b2 = _block("A"), _block("B")
        kb.add_document(rec, blocks=[b1, b2])
        # simulate the "document deleted but cascade bypassed" drift: drop the
        # record from the store directly, leaving the blocks behind
        kb.document_store.delete(rec.doc_id)
        kb._doc_ids.discard(rec.doc_id)
        assert kb.block_count() == 2
        assert {str(b.id) for b in kb.orphaned_blocks()} == {
            str(b1.id),
            str(b2.id),
        }
        assert kb.remove_orphaned_blocks() == 2
        assert kb.block_count() == 0
        assert len(kb.store) == 0
        assert len(kb.lexical) == 0

    def test_remove_orphaned_blocks_unlinked(self):
        kb = _make_kb()
        kb.add_blocks([_block("A")])  # added without any doc link
        assert len(kb.orphaned_blocks()) == 1
        assert kb.remove_orphaned_blocks() == 1
        assert kb.block_count() == 0
