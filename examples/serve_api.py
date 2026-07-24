"""Demo: serve the SparkSage WEB API and exercise every endpoint.

Runs fully offline using deterministic fakes (:class:`FakeConverterBackend` /
:class:`FakeLLMClient`) -- no ``markitdown``, no API key, no real HTTP server.

For production use, install the web + convert + llm extras and set env vars:

    pip install 'sparksage[api,convert,llm]'
    export SPARKSAGE_API_KEY=sk-...
    export SPARKSAGE_MODEL=gpt-4o-mini       # optional
    uvicorn sparksage.api.app:create_app --factory --port 8000

Then call the endpoints:

    # 1) file -> Markdown (optional cleaning)
    curl -F "file=@report.pdf" -F "clean=true" http://localhost:8000/api/v1/convert

    # 2) file -> IdeaBlock list
    curl -F "file=@report.pdf" -F "with_stats=true" http://localhost:8000/api/v1/generate

    # 3) knowledge-document management (CRUD + auto-tagging)
    curl -F "file=@onboarding.md" http://localhost:8000/api/v1/documents
    curl http://localhost:8000/api/v1/tags

Run with:  PYTHONPATH=src python3 examples/serve_api.py
"""

from __future__ import annotations

import json

from sparksage import (
    DocumentService,
    FakeConverterBackend,
    FakeLLMClient,
    IdeaBlockGenerator,
    MarkdownConverter,
    RegexReplaceRule,
    SparkSageService,
    TextCleaner,
)

SAMPLE_MARKDOWN = (
    "\ufeff# Annual Report\n"
    "CONFIDENTIAL\n\n\n\n"
    "Revenue grew 12% year over year. The company plans to expand to APAC.\n"
    "Page 1 of 5\n"
)

FAKE_LLM_RESPONSE = json.dumps(
    {
        "blocks": [
            {
                "name": "Revenue growth",
                "critical_question": "How did revenue change?",
                "trusted_answer": "Revenue grew 12% year over year.",
                "tags": ["IMPORTANT"],
                "keywords": ["revenue"],
            },
            {
                "name": "Expansion plan",
                "critical_question": "What is the expansion strategy?",
                "trusted_answer": "The company plans to expand to the APAC region.",
                "tags": ["PROCESS"],
                "keywords": ["apac", "expansion"],
            },
        ]
    }
)


def build_demo_converter() -> MarkdownConverter:
    return MarkdownConverter(
        backend=FakeConverterBackend(markdown=SAMPLE_MARKDOWN, title="Annual Report")
    )


def build_demo_cleaner() -> TextCleaner:
    cleaner = TextCleaner()
    cleaner.add(RegexReplaceRule(r"CONFIDENTIAL", ""))
    cleaner.add_for("*.pdf", RegexReplaceRule(r"Page \d+ of \d+", ""))
    return cleaner


def build_demo_service() -> SparkSageService:
    return SparkSageService(
        converter=build_demo_converter(),
        cleaner=build_demo_cleaner(),
        generator=IdeaBlockGenerator(FakeLLMClient(responses=[FAKE_LLM_RESPONSE])),
    )


def build_demo_document_service() -> DocumentService:
    return DocumentService(
        converter=build_demo_converter(), cleaner=build_demo_cleaner()
    )


def main() -> None:
    from fastapi.testclient import TestClient

    from sparksage.api.app import create_app

    app = create_app(
        service=build_demo_service(),
        document_service=build_demo_document_service(),
    )
    client = TestClient(app)

    print("=== GET /api/v1/health ===")
    resp = client.get("/api/v1/health")
    print(json.dumps(resp.json(), indent=2))

    print("\n=== POST /api/v1/convert (clean=true) ===")
    resp = client.post(
        "/api/v1/convert",
        files={"file": ("annual.pdf", b"fake-pdf-bytes", "application/pdf")},
        data={"clean": "true"},
    )
    body = resp.json()
    print(f"status: {resp.status_code}")
    print(f"source: {body['source']}")
    print(f"cleaned: {body['cleaned']}")
    print(f"markdown:\n{body['markdown']}")

    print("\n=== POST /api/v1/generate (with_stats=true) ===")
    resp = client.post(
        "/api/v1/generate",
        files={"file": ("annual.pdf", b"fake-pdf-bytes", "application/pdf")},
        data={"with_stats": "true"},
    )
    body = resp.json()
    print(f"status: {resp.status_code}")
    print(f"source: {body['source']}")
    print(f"stats: {body['stats']}")
    for i, block in enumerate(body["blocks"], 1):
        print(f"\n--- Block {i} ---")
        print(f"  name:              {block['name']}")
        print(f"  critical_question: {block['critical_question']}")
        print(f"  trusted_answer:    {block['trusted_answer']}")
        print(f"  tags:              {block['tags']}")
        print(f"  keywords:          {block['keywords']}")
        print(f"  source.uri:        {block['source']['uri']}")

    print("\n=== POST /api/v1/documents (auto-tagged from content) ===")
    resp = client.post(
        "/api/v1/documents",
        files={"file": ("annual.pdf", b"fake-pdf-bytes", "application/pdf")},
    )
    body = resp.json()["document"]
    print(f"status: {resp.status_code}")
    print(f"title:      {body['title']}")
    print(f"tags:       {body['tags']}  ({body['tag_source']})")
    print(f"total docs: {client.get('/api/v1/documents').json()['total']}")


if __name__ == "__main__":
    main()
