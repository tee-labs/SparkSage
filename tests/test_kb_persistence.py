"""Tests for the durable knowledge-base backends.

Covers:

* :class:`SqliteKnowledgeBaseStore` -- KB metadata CRUD + cross-instance
  persistence (the multi-tenant counterpart of ``SqliteDocumentStore``).
* :class:`SqliteKbStateStore` -- live block registry + vectors + doc-links.
* :class:`KnowledgeBase` with ``state_store=`` -- write-through on mutate +
  hydrate on construction (the restart-survives mechanic).

All offline via :class:`FakeEmbeddingClient`.
"""

from __future__ import annotations

import pytest

from sparksage import BlockEmbedder, FakeEmbeddingClient, IdeaBlock, Tag, new_record
from sparksage.kb import (
    KnowledgeBase,
    KnowledgeBaseInfo,
    SqliteKbStateStore,
    SqliteKnowledgeBaseStore,
)
from sparksage.kb.backends.state import KbStateSnapshot, KbStateStore
from sparksage.kb.store import KnowledgeBaseStore


def _block(name="A", body="deploy via pip"):
    return IdeaBlock(
        name=name,
        critical_question=f"what is {name}?",
        trusted_answer=body,
        tags=[Tag.IMPORTANT],
    )


def _embedder():
    return BlockEmbedder(FakeEmbeddingClient(dimension=16))


# ---------------------------------------------------------------------------- #
# SqliteKnowledgeBaseStore
# ---------------------------------------------------------------------------- #
class TestSqliteKnowledgeBaseStore:
    def test_protocol(self):
        store = SqliteKnowledgeBaseStore()
        assert isinstance(store, KnowledgeBaseStore)

    def test_crud_round_trip(self):
        store = SqliteKnowledgeBaseStore()
        info = KnowledgeBaseInfo(name="ops")
        saved = store.save(info)
        assert saved.kb_id in store
        assert len(store) == 1
        got = store.get(saved.kb_id)
        assert got is not None
        assert got.name == "ops"
        assert store.delete(saved.kb_id)
        assert saved.kb_id not in store
        assert len(store) == 0

    def test_list_newest_first_pagination(self):
        store = SqliteKnowledgeBaseStore()
        for i in range(5):
            store.save(KnowledgeBaseInfo(name=f"k{i}"))
        assert len(store.list(limit=2)) == 2
        assert len(store.list(limit=10, offset=3)) == 2

    def test_defensive_copy(self):
        store = SqliteKnowledgeBaseStore()
        info = KnowledgeBaseInfo(name="x")
        saved = store.save(info)
        saved.name = "mutated"
        assert store.get(info.kb_id).name == "x"

    def test_get_missing_is_none(self):
        assert SqliteKnowledgeBaseStore().get("nope") is None

    def test_invalid_table_name(self):
        with pytest.raises(ValueError):
            SqliteKnowledgeBaseStore(table="bad name!")

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "kb.db"
        s1 = SqliteKnowledgeBaseStore(path)
        info = s1.save(KnowledgeBaseInfo(name="persisted"))
        s1.close()
        s2 = SqliteKnowledgeBaseStore(path)
        got = s2.get(info.kb_id)
        assert got is not None
        assert got.name == "persisted"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "kb.db"
        s = SqliteKnowledgeBaseStore(path)
        assert path.exists()
        s.close()

    def test_upsert_replaces(self):
        store = SqliteKnowledgeBaseStore()
        info = KnowledgeBaseInfo(name="v1")
        saved = store.save(info)
        info2 = saved.model_copy(update={"name": "v2"})
        store.save(info2)
        assert store.get(saved.kb_id).name == "v2"
        assert len(store) == 1


# ---------------------------------------------------------------------------- #
# SqliteKbStateStore
# ---------------------------------------------------------------------------- #
class TestSqliteKbStateStore:
    def test_protocol(self):
        store = SqliteKbStateStore()
        assert isinstance(store, KbStateStore)

    def test_upsert_load_round_trip(self):
        store = SqliteKbStateStore()
        embedder = _embedder()
        b = _block()
        embedder.embed_blocks([b])
        store.upsert_block("kb1", b, "doc1")
        snap = store.load("kb1")
        assert len(snap.blocks) == 1
        assert snap.blocks[0].name == "A"
        assert snap.blocks[0].embedding is not None
        assert "doc1" in snap.doc_links
        assert str(b.id) in snap.doc_links["doc1"]

    def test_load_missing_kb_is_empty(self):
        snap = SqliteKbStateStore().load("absent")
        assert isinstance(snap, KbStateSnapshot)
        assert snap.blocks == []
        assert snap.doc_links == {}

    def test_delete_block(self):
        store = SqliteKbStateStore()
        b = _block()
        store.upsert_block("kb1", b, "doc1")
        assert store.delete_block("kb1", str(b.id))
        assert not store.delete_block("kb1", str(b.id))
        assert store.load("kb1").blocks == []

    def test_delete_block_also_drops_link(self):
        store = SqliteKbStateStore()
        b = _block()
        store.upsert_block("kb1", b, "doc1")
        store.delete_block("kb1", str(b.id))
        assert store.load("kb1").doc_links == {}

    def test_unlink_doc(self):
        store = SqliteKbStateStore()
        b1, b2 = _block("A"), _block("B")
        store.upsert_block("kb1", b1, "doc1")
        store.upsert_block("kb1", b2, "doc1")
        n = store.unlink_doc("kb1", "doc1")
        assert n == 2
        snap = store.load("kb1")
        assert snap.doc_links == {}
        # blocks themselves are NOT removed by unlink_doc
        assert len(snap.blocks) == 2

    def test_clear(self):
        store = SqliteKbStateStore()
        store.upsert_block("kb1", _block("A"), "doc1")
        store.upsert_block("kb2", _block("B"), "doc2")
        store.clear("kb1")
        assert store.load("kb1").blocks == []
        # kb2 untouched
        assert len(store.load("kb2").blocks) == 1

    def test_kb_isolation(self):
        store = SqliteKbStateStore()
        store.upsert_block("kb1", _block("A"), None)
        store.upsert_block("kb2", _block("B"), None)
        assert len(store.load("kb1").blocks) == 1
        assert len(store.load("kb2").blocks) == 1

    def test_upsert_replaces_block(self):
        store = SqliteKbStateStore()
        b = _block("A", "v1")
        store.upsert_block("kb1", b, None)
        b2 = IdeaBlock(
            id=b.id,
            name="A",
            critical_question="what is A?",
            trusted_answer="v2",
            tags=[Tag.IMPORTANT],
        )
        store.upsert_block("kb1", b2, None)
        snap = store.load("kb1")
        assert len(snap.blocks) == 1
        assert snap.blocks[0].trusted_answer == "v2"

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "state.db"
        embedder = _embedder()
        b = _block()
        embedder.embed_blocks([b])
        s1 = SqliteKbStateStore(path)
        s1.upsert_block("kb1", b, "doc1")
        s1.close()
        s2 = SqliteKbStateStore(path)
        snap = s2.load("kb1")
        assert len(snap.blocks) == 1
        assert snap.blocks[0].embedding is not None

    def test_invalid_table_name(self):
        with pytest.raises(ValueError):
            SqliteKbStateStore(table="bad name!")


# ---------------------------------------------------------------------------- #
# KnowledgeBase with state_store (write-through + hydrate on construct)
# ---------------------------------------------------------------------------- #
class TestKnowledgeBaseStateStore:
    def test_add_blocks_persists(self):
        state = SqliteKbStateStore()
        kb = KnowledgeBase(
            info=KnowledgeBaseInfo(name="ops"),
            embedder=_embedder(),
            state_store=state,
        )
        b = _block()
        kb.add_blocks([b])
        snap = state.load(kb.kb_id)
        assert len(snap.blocks) == 1
        assert snap.blocks[0].embedding is not None

    def test_remove_block_writes_through(self):
        state = SqliteKbStateStore()
        kb = KnowledgeBase(
            info=KnowledgeBaseInfo(name="ops"),
            embedder=_embedder(),
            state_store=state,
        )
        b = _block()
        kb.add_blocks([b])
        kb.remove_block(str(b.id))
        assert state.load(kb.kb_id).blocks == []

    def test_remove_document_cascades_to_state(self):
        state = SqliteKbStateStore()
        kb = KnowledgeBase(
            info=KnowledgeBaseInfo(name="ops"),
            embedder=_embedder(),
            state_store=state,
        )
        rec = new_record(body_markdown="body", source="file://d.md")
        b1, b2 = _block("A"), _block("B")
        kb.add_document(rec, blocks=[b1, b2])
        snap = state.load(kb.kb_id)
        assert len(snap.blocks) == 2
        assert str(rec.doc_id) in snap.doc_links

        kb.remove_document(rec.doc_id)
        snap = state.load(kb.kb_id)
        assert snap.blocks == []
        assert snap.doc_links == {}

    def test_new_kb_hydrates_from_state_on_construct(self):
        state = SqliteKbStateStore()
        info = KnowledgeBaseInfo(name="ops")
        kb1 = KnowledgeBase(
            info=info, embedder=_embedder(), state_store=state
        )
        rec = new_record(body_markdown="body", source="file://d.md")
        b1, b2 = _block("A"), _block("B")
        kb1.add_document(rec, blocks=[b1, b2])
        kb1_id = kb1.kb_id
        assert kb1.block_count() == 2

        # simulate a restart: brand-new aggregate over the SAME state store
        kb2 = KnowledgeBase(
            info=KnowledgeBaseInfo(name="ops", kb_id=kb1_id),
            embedder=_embedder(),
            state_store=state,
        )
        assert kb2.block_count() == 2
        assert kb2.document_count() == 1
        assert len(kb2.store) == 2  # vectors restored too

    def test_restored_kb_search_works_without_re_embedding(self):
        state = SqliteKbStateStore()
        info = KnowledgeBaseInfo(name="ops")
        embedder = _embedder()
        kb1 = KnowledgeBase(info=info, embedder=embedder, state_store=state)
        b = _block("A", "deploy with docker")
        kb1.add_blocks([b])

        # New aggregate over the same state. Wrap the embedder in a spy that
        # raises on embed_blocks (restore must read vectors off disk, never
        # re-embed) while still letting the query path call embed_texts.
        real = _embedder()
        calls = {"embed_blocks": 0}

        class _SpyEmbedder:
            dimension = real.dimension

            def embed_blocks(self, blocks, **kwargs):
                calls["embed_blocks"] += 1
                return real.embed_blocks(blocks)

            def vectors_for(self, blocks, **kwargs):
                return real.vectors_for(blocks)

            def embed_texts(self, texts):
                return real.embed_texts(list(texts))

        kb2 = KnowledgeBase(
            info=KnowledgeBaseInfo(name="ops", kb_id=kb1.kb_id),
            embedder=_SpyEmbedder(),  # type: ignore[arg-type]
            state_store=state,
        )
        result = kb2.search("docker", k=3)
        assert len(result.chunks) == 1
        assert calls["embed_blocks"] == 0

    def test_restored_kb_document_links_survive(self):
        state = SqliteKbStateStore()
        info = KnowledgeBaseInfo(name="ops")
        kb1 = KnowledgeBase(info=info, embedder=_embedder(), state_store=state)
        rec = new_record(body_markdown="body", source="file://d.md")
        kb1.add_document(rec, blocks=[_block("A"), _block("B")])

        kb2 = KnowledgeBase(
            info=KnowledgeBaseInfo(name="ops", kb_id=kb1.kb_id),
            embedder=_embedder(),
            state_store=state,
        )
        linked = kb2.blocks_for_document(rec.doc_id)
        assert {b.name for b in linked} == {"A", "B"}

    def test_restored_kb_remove_document_cascades(self):
        from sparksage.documents.backends.memory import InMemoryDocumentStore

        state = SqliteKbStateStore()
        # A shared document store mirrors the QAService wiring (one store for
        # all KBs); restore only rebuilds the block index, not the doc records.
        doc_store = InMemoryDocumentStore()
        info = KnowledgeBaseInfo(name="ops")
        kb1 = KnowledgeBase(
            info=info, embedder=_embedder(), state_store=state,
            document_store=doc_store,
        )
        rec = new_record(body_markdown="body", source="file://d.md")
        kb1.add_document(rec, blocks=[_block("A"), _block("B")])

        kb2 = KnowledgeBase(
            info=KnowledgeBaseInfo(name="ops", kb_id=kb1.kb_id),
            embedder=_embedder(),
            state_store=state,
            document_store=doc_store,
        )
        assert kb2.remove_document(rec.doc_id)
        assert kb2.block_count() == 0
        assert state.load(kb1.kb_id).blocks == []

    def test_no_state_store_is_ephemeral(self):
        kb = KnowledgeBase(
            info=KnowledgeBaseInfo(name="ops"),
            embedder=_embedder(),
        )
        kb.add_blocks([_block()])
        assert kb.state_store is None
