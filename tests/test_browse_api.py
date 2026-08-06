"""Tests for the knowledge-base block listing + feedback record listing routes.

These are the read-only browsing endpoints backing the WEB UI's knowledge-base
and feedback pages. They sit on top of :class:`QAService.list_blocks` /
:meth:`QAService.list_feedback`, both tested here via ``TestClient``.
"""

from __future__ import annotations

import json

import pytest

from sparksage import (
    BlockEmbedder,
    FakeConverterBackend,
    FakeEmbeddingClient,
    FakeLLMClient,
    IdeaBlockGenerator,
    LLMAnswerGenerator,
    LLMFaithfulnessJudge,
    MarkdownConverter,
    Reader,
    SparkSageService,
    TextCleaner,
)
from sparksage.api.qa_service import QAService

SAMPLE_MD = "# Guide\nSparkSage installs with pip. Deploy with uvicorn."

GEN_JSON = json.dumps(
    {
        "blocks": [
            {
                "name": "Install",
                "critical_question": "How to install?",
                "trusted_answer": "Install with pip install sparksage.",
                "tags": ["TECHNOLOGY"],
                "keywords": ["install", "pip"],
            },
            {
                "name": "Deploy",
                "critical_question": "How to deploy?",
                "trusted_answer": "Deploy the API with uvicorn.",
                "tags": ["PROCESS"],
                "keywords": ["deploy", "uvicorn"],
            },
        ]
    }
)


def _make_qa_service() -> QAService:
    converter = MarkdownConverter(
        backend=FakeConverterBackend(markdown=SAMPLE_MD, title="Guide")
    )
    generator = IdeaBlockGenerator(FakeLLMClient(responses=[GEN_JSON]))
    spark_service = SparkSageService(
        converter=converter, cleaner=TextCleaner(), generator=generator
    )
    embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
    answer_client = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "answer": "Use pip.",
                    "citations": [{"block_id": "x", "quote": "pip"}],
                    "confidence": 0.9,
                }
            ),
            json.dumps({"score": 0.9}),
        ]
    )
    reader = Reader(
        generator=LLMAnswerGenerator(answer_client),
        faithfulness_judge=LLMFaithfulnessJudge(answer_client),
    )
    return QAService(service=spark_service, embedder=embedder, reader=reader)


def _ingest(svc: QAService) -> str:
    result = svc.ingest_and_index(b"data", "guide.md")
    return result.doc_id


# ---------------------------------------------------------------------------- #
# QAService.list_blocks
# ---------------------------------------------------------------------------- #
class TestListBlocks:
    def test_lists_all_blocks(self):
        svc = _make_qa_service()
        _ingest(svc)
        page, total = svc.list_blocks(limit=10, offset=0)
        assert total == 2
        assert len(page) == 2

    def test_filter_by_tag(self):
        svc = _make_qa_service()
        _ingest(svc)
        page, total = svc.list_blocks(tags=["TECHNOLOGY"])
        assert total == 1
        assert page[0].name == "Install"

    def test_filter_by_tag_any_match(self):
        svc = _make_qa_service()
        _ingest(svc)
        page, total = svc.list_blocks(tags=["TECHNOLOGY", "PROCESS"])
        assert total == 2

    def test_filter_by_status(self):
        svc = _make_qa_service()
        _ingest(svc)
        # freshly generated blocks default to DRAFT per the schema
        page, total = svc.list_blocks(status="ACTIVE")
        assert total == 0
        page, total = svc.list_blocks(status="DRAFT")
        assert total == 2
        _ = page

    def test_pagination(self):
        svc = _make_qa_service()
        _ingest(svc)
        page, total = svc.list_blocks(limit=1, offset=0)
        assert total == 2
        assert len(page) == 1


# ---------------------------------------------------------------------------- #
# HTTP route integration
# ---------------------------------------------------------------------------- #
@pytest.fixture
def http_client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from sparksage.api.app import create_app

    svc = _make_qa_service()
    app = create_app(qa_service=svc)
    return TestClient(app), svc


class TestBlocksRoutes:
    def test_list_blocks_route(self, http_client):
        client, svc = http_client
        _ingest(svc)
        resp = client.get("/api/v1/knowledge_base/blocks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["count"] == 2
        names = {b["name"] for b in data["items"]}
        assert names == {"Install", "Deploy"}
        # block shape
        b0 = data["items"][0]
        for field in (
            "id",
            "name",
            "critical_question",
            "trusted_answer",
            "tags",
            "keywords",
            "language",
            "status",
            "parents",
        ):
            assert field in b0

    def test_list_blocks_tag_filter_route(self, http_client):
        client, svc = http_client
        _ingest(svc)
        resp = client.get("/api/v1/knowledge_base/blocks", params={"tag": "PROCESS"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["name"] == "Deploy"

    def test_list_blocks_status_filter_route(self, http_client):
        client, svc = http_client
        _ingest(svc)
        resp = client.get(
            "/api/v1/knowledge_base/blocks", params={"status": "ACTIVE"}
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_kb_info_route(self, http_client):
        client, svc = http_client
        _ingest(svc)
        resp = client.get("/api/v1/knowledge_base")
        assert resp.status_code == 200
        data = resp.json()
        assert data["block_count"] == 2
        assert data["document_count"] == 1

    def test_kb_tags_route(self, http_client):
        client, svc = http_client
        _ingest(svc)
        resp = client.get("/api/v1/knowledge_base/tags")
        assert resp.status_code == 200
        assert resp.json()["tags"] == ["PROCESS", "TECHNOLOGY"]

    def test_prune_orphaned_blocks_route(self, http_client):
        client, svc = http_client
        doc_id = _ingest(svc)
        # simulate the legacy drift: delete the document record through the
        # non-cascading path, leaving the indexed blocks behind
        svc.service.delete_document(doc_id)
        assert svc.service.document_store.get(doc_id) is None
        resp = client.post("/api/v1/knowledge_base/prune_orphaned_blocks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["removed"] == 2
        assert svc.knowledge_base.block_count() == 0
        assert svc.knowledge_base.document_count() == 0  # doc id reconciled too

    def test_prune_orphaned_blocks_404(self, http_client):
        client, _ = http_client
        resp = client.post(
            "/api/v1/knowledge_base/prune_orphaned_blocks", params={"kb_id": "nope"}
        )
        assert resp.status_code == 404


class TestDocumentsRoutes:
    def test_documents_multi_tag_any_match_route(self, http_client):
        client, qa_svc = http_client
        docs = qa_svc.service
        docs.ingest_document(b"a", "a.md", tags=["alpha"])
        docs.ingest_document(b"b", "b.md", tags=["beta"])
        docs.ingest_document(b"c", "c.md", tags=["gamma"])
        resp = client.get("/api/v1/documents", params={"tag": "alpha,beta"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["count"] == 2
        # single literal "alpha,beta" tag must NOT match (regression guard)
        resp_single = client.get("/api/v1/documents", params={"tag": "alpha"})
        assert resp_single.json()["total"] == 1


class TestFeedbackRoutes:
    def test_feedback_records_route(self, http_client):
        client, svc = http_client
        svc.add_feedback("how to install?", "Use pip.", "positive")
        svc.add_feedback("how to deploy?", "wrong", "negative")
        resp = client.get("/api/v1/feedback/records")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["count"] == 2
        assert {r["rating"] for r in data["items"]} == {"positive", "negative"}
        r0 = data["items"][0]
        for field in (
            "feedback_id",
            "query",
            "answer_text",
            "rating",
            "correction",
            "block_ids",
            "created_at",
        ):
            assert field in r0

    def test_feedback_records_pagination(self, http_client):
        client, svc = http_client
        for i in range(5):
            svc.add_feedback(f"q{i}", "a", "positive")
        resp = client.get("/api/v1/feedback/records", params={"limit": 2, "offset": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert data["count"] == 2

    def test_feedback_stats_route(self, http_client):
        client, svc = http_client
        svc.add_feedback("q", "a", "positive")
        svc.add_feedback("q2", "a2", "negative")
        resp = client.get("/api/v1/feedback")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["positive"] == 1
        assert data["negative"] == 1
