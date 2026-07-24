"""Demo: serve the SparkSage WEB API and exercise both endpoints.

Runs fully offline using deterministic fakes (:class:`FakeConverterBackend` /
:class:`FakeLLMClient`) -- no ``markitdown``, no API key, no real HTTP server.

For production use, install the web + convert + llm extras and set env vars:

    pip install 'sparksage[api,convert,llm]'
    export SPARKSAGE_API_KEY=sk-...
    export SPARKSAGE_MODEL=gpt-4o-mini       # optional
    uvicorn sparksage.api.app:create_app --factory --port 8000

Then call the two endpoints:

    # 1) file -> Markdown (optional cleaning)
    curl -F "file=@report.pdf" -F "clean=true" http://localhost:8000/api/v1/convert

    # 2) file -> IdeaBlock list
    curl -F "file=@report.pdf" -F "with_stats=true" http://localhost:8000/api/v1/generate

Run with:  PYTHONPATH=src python3 examples/serve_api.py
"""

from __future__ import annotations

import json

from sparksage import (
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


def build_demo_service() -> SparkSageService:
    converter = MarkdownConverter(
        backend=FakeConverterBackend(markdown=SAMPLE_MARKDOWN, title="Annual Report")
    )
    cleaner = TextCleaner()
    cleaner.add(RegexReplaceRule(r"CONFIDENTIAL", ""))
    cleaner.add_for("*.pdf", RegexReplaceRule(r"Page \d+ of \d+", ""))
    generator = IdeaBlockGenerator(FakeLLMClient(responses=[FAKE_LLM_RESPONSE]))
    return SparkSageService(
        converter=converter, cleaner=cleaner, generator=generator
    )


def main() -> None:
    from fastapi.testclient import TestClient

    from sparksage.api.app import create_app

    svc = build_demo_service()
    app = create_app(service=svc)
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


if __name__ == "__main__":
    main()
