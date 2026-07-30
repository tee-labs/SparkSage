"""Demo: the full SparkSage knowledge-QA pipeline end-to-end.

Runs fully offline using deterministic fakes
(:class:`FakeConverterBackend` / :class:`FakeLLMClient` /
:class:`FakeEmbeddingClient`) -- no ``markitdown``, no API key, no server. It
exercises every stage of a production knowledge-QA system:

    1. INGEST   -- upload a document -> convert -> clean -> chunk into IdeaBlocks
                   -> embed + index into the knowledge base.
    2. QUERY    -- ask a question -> hybrid retrieval -> grounded answer with
                   citations (or a principled abstention).
    3. FEEDBACK -- record a user verdict -> aggregate stats (the quality flywheel).

For production use, install the extras and set env vars:

    pip install 'sparksage[api,convert,llm,embed]'
    export SPARKSAGE_API_KEY=sk-...
    export SPARKSAGE_EMBEDDING_API_KEY=sk-...      # falls back to the LLM key
    uvicorn sparksage.api.app:create_app --factory --port 8000

Then drive the same three stages over HTTP:

    # 1) upload knowledge (parse -> chunk -> embed -> index)
    curl -F "file=@report.pdf" http://localhost:8000/api/v1/knowledge_base/ingest

    # 2) ask a question
    curl -X POST http://localhost:8000/api/v1/query \\
         -H 'Content-Type: application/json' \\
         -d '{"query": "How did revenue change?"}'

    # 3) leave feedback
    curl -X POST http://localhost:8000/api/v1/feedback \\
         -H 'Content-Type: application/json' \\
         -d '{"query": "How did revenue change?", "answer_text": "...", "rating": "positive"}'

Run with:  PYTHONPATH=src python3 examples/qa_full_pipeline.py
"""

from __future__ import annotations

import json

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

SAMPLE_MARKDOWN = (
    "# SparkSage Product Guide\n\n"
    "SparkSage is a Python library for question-aligned RAG chunking. "
    "It turns documents into IdeaBlocks -- small knowledge units aligned to "
    "how users ask questions.\n\n"
    "## Installation\n"
    "Install SparkSage with pip: `pip install sparksage`. "
    "The core has zero dependencies beyond Pydantic. Optional extras pull in "
    "file conversion, LLM generation, and embedding clients.\n\n"
    "## Deployment\n"
    "Run the API server with uvicorn: "
    "`uvicorn sparksage.api.app:create_app --factory --port 8000`. "
    "Interactive docs are available at /docs.\n"
)

GEN_RESPONSE = json.dumps(
    {
        "blocks": [
            {
                "name": "What SparkSage is",
                "critical_question": "What is SparkSage?",
                "trusted_answer": (
                    "SparkSage is a Python library for question-aligned RAG "
                    "chunking that turns documents into IdeaBlocks."
                ),
                "tags": ["important"],
                "keywords": ["sparksage", "rag", "ideablock"],
            },
            {
                "name": "Installation",
                "critical_question": "How do I install SparkSage?",
                "trusted_answer": (
                    "Install SparkSage with pip: pip install sparksage. "
                    "The core has zero dependencies beyond Pydantic."
                ),
                "tags": ["technology"],
                "keywords": ["install", "pip", "dependencies"],
            },
            {
                "name": "Deployment",
                "critical_question": "How do I deploy the SparkSage API?",
                "trusted_answer": (
                    "Run the API server with uvicorn: "
                    "uvicorn sparksage.api.app:create_app --factory --port 8000."
                ),
                "tags": ["technology", "process"],
                "keywords": ["deploy", "uvicorn", "server"],
            },
        ]
    }
)

ANSWER_RESPONSE = json.dumps(
    {
        "answer": "You can install SparkSage using pip: run `pip install sparksage`.",
        "citations": [
            {"block_id": "ID", "quote": "pip install sparksage"}
        ],
        "confidence": 0.95,
    }
)

FAITH_RESPONSE = json.dumps(
    {"score": 0.9, "supported_claims": 1, "unsupported_claims": 0}
)


def build_demo_qa_service() -> QAService:
    """Wire a full :class:`QAService` with deterministic fakes."""
    converter = MarkdownConverter(
        backend=FakeConverterBackend(markdown=SAMPLE_MARKDOWN, title="SparkSage Guide")
    )
    cleaner = TextCleaner()
    generator = IdeaBlockGenerator(FakeLLMClient(responses=[GEN_RESPONSE]))

    spark_service = SparkSageService(
        converter=converter,
        cleaner=cleaner,
        generator=generator,
    )

    embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))

    answer_client = FakeLLMClient(responses=[ANSWER_RESPONSE, FAITH_RESPONSE])
    reader = Reader(
        generator=LLMAnswerGenerator(answer_client),
        faithfulness_judge=LLMFaithfulnessJudge(answer_client),
    )

    return QAService(
        service=spark_service,
        embedder=embedder,
        reader=reader,
    )


def main() -> None:
    qa_svc = build_demo_qa_service()

    print("=" * 70)
    print("STAGE 1: INGEST  (upload -> convert -> chunk -> embed -> index)")
    print("=" * 70)
    result = qa_svc.ingest_and_index(b"raw bytes", "guide.md")
    print(f"  doc_id:     {result.doc_id}")
    print(f"  title:      {result.title}")
    print(f"  blocks:     {result.block_count} IdeaBlocks indexed")
    print(f"  tags:       {result.tags}")
    for b in result.blocks:
        print(f"    - {b.critical_question}")
        print(f"      {b.trusted_answer[:60]}...")

    print()
    print("=" * 70)
    print("STAGE 2: QUERY   (ask -> retrieve -> grounded answer)")
    print("=" * 70)
    qa_result = qa_svc.ask("How do I install SparkSage?", use_lexical=False)
    print(f"  question:   {qa_result.query}")
    print(f"  abstained:  {qa_result.abstained}")
    print(f"  cached:     {qa_result.cached}")
    print(f"  answer:     {qa_result.text}")
    print(f"  confidence: {qa_result.answer.confidence:.2f}")
    if qa_result.citations:
        print(f"  citations:  {len(qa_result.citations)}")
        for c in qa_result.citations:
            print(f"    [{c.block_id[:8]}] {c.uri}  \"{c.quote}\"")
    if qa_result.retrieval:
        print(f"  retrieved:  {len(qa_result.retrieval.chunks)} chunks")

    print()
    print("=" * 70)
    print("STAGE 3: FEEDBACK  (record verdict -> aggregate)")
    print("=" * 70)
    fb = qa_svc.add_feedback(
        query="How do I install SparkSage?",
        answer_text=qa_result.text,
        rating="positive",
        block_ids=[str(b.id) for b in result.blocks[:2]],
    )
    print(f"  recorded:   {fb.feedback_id} ({fb.rating.value})")
    qa_svc.add_feedback(
        query="How do I deploy?",
        answer_text="wrong answer",
        rating="negative",
    )
    stats = qa_svc.feedback_stats()
    print(f"  total:      {stats.total}")
    print(f"  positive:   {stats.positive}")
    print(f"  negative:   {stats.negative}")
    print(f"  approval:   {stats.approval:.0%}")

    print()
    print("=" * 70)
    print("KNOWLEDGE BASE SNAPSHOT")
    print("=" * 70)
    kb = qa_svc.knowledge_base_info()
    print(f"  kb_id:          {kb['kb_id']}")
    print(f"  name:           {kb['name']}")
    print(f"  blocks:         {kb['block_count']}")
    print(f"  documents:      {kb['document_count']}")

    print()
    print("Full pipeline completed successfully.")

    print()
    print("=" * 70)
    print("BONUS: same pipeline over HTTP (FastAPI TestClient)")
    print("=" * 70)
    _demo_http()


def _demo_http() -> None:
    """Exercise the same three stages via the FastAPI routes."""
    from fastapi.testclient import TestClient

    from sparksage.api.app import create_app

    qa_svc = build_demo_qa_service()
    app = create_app(qa_service=qa_svc)
    client = TestClient(app)

    resp = client.post(
        "/api/v1/knowledge_base/ingest",
        files={"file": ("guide.md", b"data", "text/plain")},
    )
    body = resp.json()
    print(f"  POST /ingest -> {resp.status_code}, {body['block_count']} blocks")

    resp = client.post(
        "/api/v1/query",
        json={"query": "How do I install SparkSage?", "use_lexical": False},
    )
    body = resp.json()
    print(f"  POST /query  -> {resp.status_code}, abstained={body['abstained']}")
    print(f"    answer: {body['answer'][:70]}...")
    print(f"    citations: {len(body['citations'])}")

    resp = client.post(
        "/api/v1/feedback",
        json={
            "query": "How do I install SparkSage?",
            "answer_text": body["answer"],
            "rating": "positive",
            "block_ids": [c["block_id"] for c in body["citations"]],
        },
    )
    print(f"  POST /feedback -> {resp.status_code}, {resp.json()['rating']}")

    resp = client.get("/api/v1/feedback")
    print(f"  GET  /feedback -> {resp.status_code}, approval={resp.json()['approval']:.0%}")

    resp = client.get("/api/v1/knowledge_base")
    print(f"  GET  /kb      -> {resp.status_code}, blocks={resp.json()['block_count']}")


if __name__ == "__main__":
    main()
