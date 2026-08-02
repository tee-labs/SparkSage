"""Integration tests: QAService reloads persisted state across a "restart".

These exercise the full persistence story the Docker-restart issue is about:
with durable stores wired, a brand-new :class:`QAService` built over the SAME
SQLite files sees the ingested documents + indexed blocks + KB metadata +
feedback that a prior instance wrote -- no re-ingest, no re-embedding, no
re-asking.

This is the regression guard for the "docker重启之后数据就没有了" issue.
"""

from __future__ import annotations

import json

import pytest

from sparksage import (
    BlockEmbedder,
    FakeConverterBackend,
    FakeEmbeddingClient,
    FakeLLMClient,
    FeedbackRating,
    IdeaBlockGenerator,
    LLMAnswerGenerator,
    LLMFaithfulnessJudge,
    MarkdownConverter,
    Reader,
    SparkSageService,
    TextCleaner,
)
from sparksage.api.qa_service import QAService
from sparksage.feedback import SqliteFeedbackStore
from sparksage.kb import SqliteKbStateStore, SqliteKnowledgeBaseStore

SAMPLE_MD = (
    "# Guide\nSparkSage is a library. Install with pip install sparksage. "
    "Deploy with uvicorn."
)

GEN_JSON = json.dumps(
    {
        "blocks": [
            {
                "name": "Install",
                "critical_question": "How to install?",
                "trusted_answer": "Install with pip install sparksage.",
                "tags": ["technology"],
                "keywords": ["install", "pip"],
            },
            {
                "name": "Deploy",
                "critical_question": "How to deploy?",
                "trusted_answer": "Deploy with uvicorn.",
                "tags": ["process"],
                "keywords": ["deploy", "uvicorn"],
            },
        ]
    }
)


def _answer_json(text="Use pip install sparksage.", cid="ID"):
    return json.dumps(
        {
            "answer": text,
            "citations": [{"block_id": cid, "quote": "pip install"}],
            "confidence": 0.9,
        }
    )


def _faith_json(score=0.9):
    return json.dumps({"score": score, "supported_claims": 1, "unsupported_claims": 0})


def _make_service(*, kb_store, state_store, feedback_store, embed_dim=16):
    converter = MarkdownConverter(
        backend=FakeConverterBackend(markdown=SAMPLE_MD, title="Guide")
    )
    spark = SparkSageService(
        converter=converter,
        cleaner=TextCleaner(),
        generator=IdeaBlockGenerator(FakeLLMClient(responses=[GEN_JSON])),
    )
    embedder = BlockEmbedder(FakeEmbeddingClient(dimension=embed_dim))
    answer_client = FakeLLMClient(
        responses=[_answer_json(), _faith_json()]
    )
    reader = Reader(
        generator=LLMAnswerGenerator(answer_client),
        faithfulness_judge=LLMFaithfulnessJudge(answer_client),
    )
    return QAService(
        service=spark,
        embedder=embedder,
        reader=reader,
        kb_store=kb_store,
        state_store=state_store,
        feedback_store=feedback_store,
    )


class TestReloadPersistedState:
    def test_blocks_and_vectors_survive_restart(self, tmp_path):
        kb_store = SqliteKnowledgeBaseStore(tmp_path / "kb.db")
        state_store = SqliteKbStateStore(tmp_path / "state.db")
        feedback_store = SqliteFeedbackStore(tmp_path / "fb.db")

        svc1 = _make_service(
            kb_store=kb_store, state_store=state_store, feedback_store=feedback_store
        )
        svc1.ingest_and_index(b"data", "guide.md")
        active_id = svc1.active_kb_id
        assert svc1.knowledge_base.block_count() == 2

        # "restart": a brand new service over the SAME files.
        svc2 = _make_service(
            kb_store=kb_store, state_store=state_store, feedback_store=feedback_store
        )
        # the previously-active KB was reloaded (not a fresh "default")
        assert active_id in svc2._kbs
        assert svc2.knowledge_base.block_count() == 2
        assert svc2.knowledge_base.document_count() == 1
        # vectors restored too -- search works with zero re-embedding
        res = svc2.knowledge_base.search("install", k=2)
        assert len(res.chunks) > 0

    def test_kb_metadata_survives_restart(self, tmp_path):
        kb_store = SqliteKnowledgeBaseStore(tmp_path / "kb.db")
        state_store = SqliteKbStateStore(tmp_path / "state.db")
        feedback_store = SqliteFeedbackStore(tmp_path / "fb.db")

        svc1 = _make_service(
            kb_store=kb_store, state_store=state_store, feedback_store=feedback_store
        )
        info = svc1.create_knowledge_base("ops", description="ops docs")
        svc1.ingest_and_index(b"data", "a.md", kb_id=info.kb_id)

        svc2 = _make_service(
            kb_store=kb_store, state_store=state_store, feedback_store=feedback_store
        )
        # both the default KB and "ops" survive
        names = {kb.name for kb in svc2._kbs.values()}
        assert "ops" in names
        ops = svc2._kbs[info.kb_id]
        assert ops.block_count() == 2

    def test_feedback_survives_restart(self, tmp_path):
        kb_store = SqliteKnowledgeBaseStore(tmp_path / "kb.db")
        state_store = SqliteKbStateStore(tmp_path / "state.db")
        feedback_store = SqliteFeedbackStore(tmp_path / "fb.db")

        svc1 = _make_service(
            kb_store=kb_store, state_store=state_store, feedback_store=feedback_store
        )
        svc1.ingest_and_index(b"data", "guide.md")
        svc1.add_feedback(
            "how?", "ans", FeedbackRating.POSITIVE, block_ids=["b1"]
        )
        assert svc1.feedback_stats().total == 1

        svc2 = _make_service(
            kb_store=kb_store, state_store=state_store, feedback_store=feedback_store
        )
        assert svc2.feedback_stats().total == 1
        assert svc2.feedback_stats().positive == 1

    def test_documents_survive_restart(self, tmp_path):
        # documents use the SparkSageService.document_store (SqliteDocumentStore)
        from sparksage.documents.backends import SqliteDocumentStore

        doc_store = SqliteDocumentStore(tmp_path / "docs.db")
        kb_store = SqliteKnowledgeBaseStore(tmp_path / "kb.db")
        state_store = SqliteKbStateStore(tmp_path / "state.db")
        feedback_store = SqliteFeedbackStore(tmp_path / "fb.db")

        converter = MarkdownConverter(
            backend=FakeConverterBackend(markdown=SAMPLE_MD, title="Guide")
        )
        spark1 = SparkSageService(
            converter=converter,
            cleaner=TextCleaner(),
            generator=IdeaBlockGenerator(FakeLLMClient(responses=[GEN_JSON])),
            document_store=doc_store,
        )
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=16))
        answer_client = FakeLLMClient(responses=[_answer_json(), _faith_json()])
        reader = Reader(
            generator=LLMAnswerGenerator(answer_client),
            faithfulness_judge=LLMFaithfulnessJudge(answer_client),
        )
        svc1 = QAService(
            service=spark1,
            embedder=embedder,
            reader=reader,
            kb_store=kb_store,
            state_store=state_store,
            feedback_store=feedback_store,
        )
        svc1.ingest_and_index(b"data", "guide.md")
        assert svc1.service.count_documents() == 1

        # restart: same doc store + same KB stores
        spark2 = SparkSageService(
            converter=converter,
            cleaner=TextCleaner(),
            generator=IdeaBlockGenerator(FakeLLMClient(responses=[GEN_JSON])),
            document_store=doc_store,
        )
        svc2 = QAService(
            service=spark2,
            embedder=embedder,
            reader=reader,
            kb_store=kb_store,
            state_store=state_store,
            feedback_store=feedback_store,
        )
        assert svc2.service.count_documents() == 1
        docs = svc2.service.list_documents()
        assert docs[0].title == "Guide"

    def test_delete_kb_clears_state(self, tmp_path):
        kb_store = SqliteKnowledgeBaseStore(tmp_path / "kb.db")
        state_store = SqliteKbStateStore(tmp_path / "state.db")
        feedback_store = SqliteFeedbackStore(tmp_path / "fb.db")

        svc1 = _make_service(
            kb_store=kb_store, state_store=state_store, feedback_store=feedback_store
        )
        info = svc1.create_knowledge_base("temp")
        svc1.ingest_and_index(b"data", "a.md", kb_id=info.kb_id)
        assert state_store.load(info.kb_id).blocks

        svc1.delete_knowledge_base(info.kb_id)
        # state for the deleted KB is cleared
        assert state_store.load(info.kb_id).blocks == []

        svc2 = _make_service(
            kb_store=kb_store, state_store=state_store, feedback_store=feedback_store
        )
        assert info.kb_id not in svc2._kbs

    def test_ephemeral_mode_unchanged(self):
        # No durable stores -> behaves exactly as before (in-memory, fresh KB).
        converter = MarkdownConverter(
            backend=FakeConverterBackend(markdown=SAMPLE_MD, title="Guide")
        )
        spark = SparkSageService(
            converter=converter,
            cleaner=TextCleaner(),
            generator=IdeaBlockGenerator(FakeLLMClient(responses=[GEN_JSON])),
        )
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=16))
        answer_client = FakeLLMClient(responses=[_answer_json(), _faith_json()])
        reader = Reader(
            generator=LLMAnswerGenerator(answer_client),
            faithfulness_judge=LLMFaithfulnessJudge(answer_client),
        )
        svc = QAService(service=spark, embedder=embedder, reader=reader)
        assert svc.state_store is None
        assert svc.knowledge_base.block_count() == 0


@pytest.mark.parametrize("validator", ["build_qa_service", "build_default_service"])
def test_data_dir_wires_durable_backends(tmp_path, monkeypatch, validator):
    # SPARKSAGE_DATA_DIR should wire every durable backend from one knob.
    from sparksage.api import app as app_module

    data_dir = tmp_path / "data"
    monkeypatch.setenv("SPARKSAGE_DATA_DIR", str(data_dir))
    # keep the rest of the env clean so nothing else is configured
    monkeypatch.delenv("SPARKSAGE_DOC_STORE", raising=False)
    monkeypatch.delenv("SPARKSAGE_KB_STORE", raising=False)
    monkeypatch.delenv("SPARKSAGE_KB_STATE_STORE", raising=False)
    monkeypatch.delenv("SPARKSAGE_FEEDBACK_STORE", raising=False)

    doc_store = app_module._build_document_store()
    kb_store, state_store = app_module._build_kb_stores()
    feedback_store = app_module._build_feedback_store()

    assert doc_store is not None
    assert kb_store is not None
    assert state_store is not None
    assert feedback_store is not None
    # the data dir was created
    assert data_dir.is_dir()
