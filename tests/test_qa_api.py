"""Tests for the end-to-end QA service + HTTP routes.

Two layers are tested (mirroring ``test_api.py``):

1. **QAService unit tests** -- framework-agnostic, fully offline with
   :class:`FakeConverterBackend` / :class:`FakeLLMClient` /
   :class:`FakeEmbeddingClient`.
2. **HTTP integration tests** -- the QA routes via ``TestClient``.

The QA pipeline is: ingest (convert -> chunk -> index) -> query (retrieve ->
answer) -> feedback (record -> aggregate).
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
from sparksage.api.pipeline import GenerationNotConfiguredError
from sparksage.api.qa_service import IngestResult, QAService

# ---------------------------------------------------------------------------- #
# Shared fakes
# ---------------------------------------------------------------------------- #
SAMPLE_MD = (
    "# Product Guide\n"
    "SparkSage is a library for question-aligned RAG. "
    "Install it with pip install sparksage. "
    "Deploy the API with uvicorn."
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
                "trusted_answer": "Deploy the API with uvicorn.",
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


def _make_qa_service(
    *,
    markdown: str = SAMPLE_MD,
    gen_json: str = GEN_JSON,
    answer_text: str = "Use pip install sparksage.",
    with_generator: bool = True,
    faith_score: float = 0.9,
) -> QAService:
    converter = MarkdownConverter(
        backend=FakeConverterBackend(markdown=markdown, title="Guide")
    )
    generator = (
        IdeaBlockGenerator(FakeLLMClient(responses=[gen_json]))
        if with_generator
        else None
    )
    spark_service = SparkSageService(
        converter=converter,
        cleaner=TextCleaner(),
        generator=generator,
    )
    embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))

    answer_client = FakeLLMClient(
        responses=[_answer_json(text=answer_text), _faith_json(faith_score)]
    )
    reader = Reader(
        generator=LLMAnswerGenerator(answer_client),
        faithfulness_judge=LLMFaithfulnessJudge(answer_client),
    )
    return QAService(
        service=spark_service,
        embedder=embedder,
        reader=reader,
    )


# ---------------------------------------------------------------------------- #
# QAService.ingest_and_index
# ---------------------------------------------------------------------------- #
class TestIngestAndIndex:
    def test_returns_ingest_result(self):
        svc = _make_qa_service()
        result = svc.ingest_and_index(b"data", "guide.md")
        assert isinstance(result, IngestResult)
        assert result.block_count == 2
        assert result.title == "Guide"
        assert result.doc_id

    def test_blocks_are_indexed_in_kb(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        assert svc.knowledge_base.block_count() == 2
        assert svc.knowledge_base.document_count() == 1

    def test_indexed_blocks_are_retrievable(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        res = svc.knowledge_base.search("install", k=2)
        assert len(res.chunks) > 0

    def test_tags_auto_extracted(self):
        svc = _make_qa_service()
        result = svc.ingest_and_index(b"data", "guide.md", top_k=3)
        assert result.tags
        assert len(result.tags) <= 3

    def test_explicit_tags_kept(self):
        svc = _make_qa_service()
        result = svc.ingest_and_index(
            b"data", "guide.md", tags=["custom"], auto_tag=True
        )
        assert result.tags == ["custom"]

    def test_summary_produced(self):
        svc = _make_qa_service()
        result = svc.ingest_and_index(b"data", "guide.md")
        assert result.summary is not None

    def test_no_generator_raises(self):
        svc = _make_qa_service(with_generator=False)
        with pytest.raises(GenerationNotConfiguredError):
            svc.ingest_and_index(b"data", "guide.md")

    def test_provenance_uses_filename(self):
        svc = _make_qa_service()
        result = svc.ingest_and_index(b"data", "docs/report.pdf")
        assert result.source.uri == "docs/report.pdf"


# ---------------------------------------------------------------------------- #
# QAService.ask
# ---------------------------------------------------------------------------- #
class TestAsk:
    def test_returns_qa_result(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        result = svc.ask("How to install?", use_lexical=False)
        assert result.query == "How to install?"
        assert not result.abstained
        assert "pip" in result.text

    def test_retrieval_scoped_to_kb(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        result = svc.ask("How to deploy?", use_lexical=False)
        assert result.retrieval is not None
        assert len(result.retrieval.chunks) > 0

    def test_abstention_on_no_kb_content(self):
        svc = _make_qa_service()
        # ask before any ingest -> empty retrieval -> abstention
        result = svc.ask("anything?", use_lexical=False)
        assert result.abstained

    def test_citation_bound_to_block(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        svc2 = _make_qa_service(answer_text="Use pip install sparksage.")
        svc2.ingest_and_index(b"data", "guide.md")
        result = svc2.ask("How to install?", use_lexical=False)
        assert not result.abstained


# ---------------------------------------------------------------------------- #
# QAService feedback
# ---------------------------------------------------------------------------- #
class TestFeedback:
    def test_add_feedback_string_rating(self):
        svc = _make_qa_service()
        record = svc.add_feedback("q", "a", "positive")
        assert record.rating == FeedbackRating.POSITIVE
        assert record.kb_id == svc.knowledge_base.kb_id

    def test_add_feedback_enum_rating(self):
        svc = _make_qa_service()
        record = svc.add_feedback("q", "a", FeedbackRating.NEGATIVE)
        assert record.rating == FeedbackRating.NEGATIVE

    def test_feedback_stats(self):
        svc = _make_qa_service()
        svc.add_feedback("q1", "a1", "positive")
        svc.add_feedback("q2", "a2", "negative")
        stats = svc.feedback_stats()
        assert stats.total == 2
        assert stats.positive == 1
        assert stats.negative == 1
        assert stats.approval == 0.5


# ---------------------------------------------------------------------------- #
# QAService.knowledge_base_info
# ---------------------------------------------------------------------------- #
class TestKnowledgeBaseInfo:
    def test_info_before_ingest(self):
        svc = _make_qa_service()
        info = svc.knowledge_base_info()
        assert info["block_count"] == 0
        assert info["document_count"] == 0
        assert info["name"] == "default"

    def test_info_after_ingest(self):
        svc = _make_qa_service()
        svc.ingest_and_index(b"data", "guide.md")
        info = svc.knowledge_base_info()
        assert info["block_count"] == 2
        assert info["document_count"] == 1


# ---------------------------------------------------------------------------- #
# HTTP integration tests
# ---------------------------------------------------------------------------- #
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from sparksage.api.app import create_app  # noqa: E402


@pytest.fixture
def qa_client():
    svc = _make_qa_service()
    app = create_app(qa_service=svc)
    return TestClient(app)


def _ingest(client, filename="guide.md"):
    """Helper: ingest a doc and return the response body."""
    resp = client.post(
        "/api/v1/knowledge_base/ingest",
        files={"file": (filename, b"data", "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestKBIngestRoute:
    def test_ingest_returns_blocks(self, qa_client):
        body = _ingest(qa_client)
        assert body["block_count"] == 2
        assert len(body["blocks"]) == 2
        assert body["title"] == "Guide"
        assert body["source"]["uri"] == "guide.md"
        assert body["tags"]
        assert body["summary"]

    def test_ingest_with_explicit_tags(self, qa_client):
        resp = qa_client.post(
            "/api/v1/knowledge_base/ingest",
            files={"file": ("guide.md", b"data", "text/plain")},
            data={"tags": "alpha,beta"},
        )
        assert resp.status_code == 200
        assert resp.json()["tags"] == ["alpha", "beta"]

    def test_ingest_missing_file(self, qa_client):
        resp = qa_client.post("/api/v1/knowledge_base/ingest")
        assert resp.status_code == 422

    def test_ingest_503_without_generator(self):
        svc = _make_qa_service(with_generator=False)
        app = create_app(qa_service=svc)
        client = TestClient(app)
        resp = client.post(
            "/api/v1/knowledge_base/ingest",
            files={"file": ("guide.md", b"data", "text/plain")},
        )
        assert resp.status_code == 503


class TestQueryRoute:
    def test_ask_returns_answer(self, qa_client):
        _ingest(qa_client)
        resp = qa_client.post(
            "/api/v1/query",
            json={"query": "How to install?", "use_lexical": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "How to install?"
        assert not body["abstained"]
        assert "pip" in body["answer"]
        assert body["retrieved"]

    def test_ask_abstains_on_empty_kb(self, qa_client):
        resp = qa_client.post(
            "/api/v1/query",
            json={"query": "anything?", "use_lexical": False},
        )
        assert resp.status_code == 200
        assert resp.json()["abstained"] is True

    def test_ask_with_tag_filter(self, qa_client):
        _ingest(qa_client)
        resp = qa_client.post(
            "/api/v1/query",
            json={
                "query": "install",
                "use_lexical": False,
                "tags": ["technology"],
            },
        )
        assert resp.status_code == 200

    def test_ask_missing_query_body(self, qa_client):
        resp = qa_client.post("/api/v1/query", json={})
        assert resp.status_code == 422

    def test_ask_with_history(self, qa_client):
        _ingest(qa_client)
        resp = qa_client.post(
            "/api/v1/query",
            json={
                "query": "how?",
                "use_lexical": False,
                "history": [
                    {"role": "user", "content": "how to install?"},
                ],
            },
        )
        assert resp.status_code == 200


class TestKnowledgeBaseRoute:
    def test_kb_info(self, qa_client):
        _ingest(qa_client)
        resp = qa_client.get("/api/v1/knowledge_base")
        assert resp.status_code == 200
        body = resp.json()
        assert body["block_count"] == 2
        assert body["document_count"] == 1
        assert body["name"] == "default"

    def test_remove_document(self, qa_client):
        body = _ingest(qa_client)
        doc_id = body["doc_id"]
        resp = qa_client.delete(f"/api/v1/knowledge_base/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        # block count should drop to 0
        resp = qa_client.get("/api/v1/knowledge_base")
        assert resp.json()["block_count"] == 0

    def test_remove_document_404(self, qa_client):
        resp = qa_client.delete("/api/v1/knowledge_base/documents/missing")
        assert resp.status_code == 404


class TestFeedbackRoute:
    def test_record_feedback(self, qa_client):
        resp = qa_client.post(
            "/api/v1/feedback",
            json={
                "query": "How to install?",
                "answer_text": "pip install sparksage",
                "rating": "positive",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["feedback_id"]
        assert body["rating"] == "positive"
        assert body["acknowledged"] is True

    def test_feedback_stats(self, qa_client):
        qa_client.post(
            "/api/v1/feedback",
            json={"query": "q1", "answer_text": "a", "rating": "positive"},
        )
        qa_client.post(
            "/api/v1/feedback",
            json={"query": "q2", "answer_text": "a", "rating": "negative"},
        )
        resp = qa_client.get("/api/v1/feedback")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert body["positive"] == 1
        assert body["negative"] == 1
        assert body["approval"] == 0.5

    def test_invalid_rating(self, qa_client):
        resp = qa_client.post(
            "/api/v1/feedback",
            json={"query": "q", "answer_text": "a", "rating": "bogus"},
        )
        assert resp.status_code == 422

    def test_feedback_with_correction(self, qa_client):
        resp = qa_client.post(
            "/api/v1/feedback",
            json={
                "query": "q",
                "answer_text": "wrong",
                "rating": "corrected",
                "correction": "right answer",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["rating"] == "corrected"


# ---------------------------------------------------------------------------- #
# create_app without qa_service should NOT mount QA routes
# ---------------------------------------------------------------------------- #
class TestRouteMounting:
    def test_qa_routes_absent_without_qa_service(self):
        """When no qa_service is passed, the QA routes are not mounted."""
        from sparksage.clean.cleaner import TextCleaner
        from sparksage.convert.backend import FakeConverterBackend
        from sparksage.convert.converter import MarkdownConverter

        svc = SparkSageService(
            converter=MarkdownConverter(backend=FakeConverterBackend("# hi")),
            cleaner=TextCleaner(),
        )
        app = create_app(service=svc)
        client = TestClient(app)

        resp = client.post("/api/v1/query", json={"query": "test"})
        assert resp.status_code == 404

        resp = client.get("/api/v1/knowledge_base")
        assert resp.status_code == 404

    def test_env_var_auto_builds_qa_service(self, monkeypatch):
        """SPARKSAGE_ENABLE_QA=1 auto-builds a QA service and mounts QA routes."""
        import sparksage.api.app as app_module

        qa_svc = _make_qa_service()
        monkeypatch.setenv("SPARKSAGE_ENABLE_QA", "1")
        monkeypatch.setattr(app_module, "build_qa_service", lambda: qa_svc)
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/knowledge_base")
        assert resp.status_code == 200
        assert app.state.qa_service is qa_svc

    def test_env_var_falsy_does_not_mount_qa_routes(self, monkeypatch):
        """When SPARKSAGE_ENABLE_QA is unset, QA routes stay unmounted."""
        import sparksage.api.app as app_module
        from sparksage.clean.cleaner import TextCleaner
        from sparksage.convert.backend import FakeConverterBackend
        from sparksage.convert.converter import MarkdownConverter

        monkeypatch.delenv("SPARKSAGE_ENABLE_QA", raising=False)
        base_svc = SparkSageService(
            converter=MarkdownConverter(backend=FakeConverterBackend("# hi")),
            cleaner=TextCleaner(),
        )
        monkeypatch.setattr(
            app_module, "build_default_service", lambda: base_svc
        )
        app = create_app()
        client = TestClient(app)

        resp = client.get("/api/v1/knowledge_base")
        assert resp.status_code == 404
