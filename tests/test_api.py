"""Tests for the SparkSage WEB API layer.

Two layers are tested:

1. **Service unit tests** -- :class:`SparkSageService` is framework-agnostic and
   tested directly with :class:`FakeConverterBackend` / :class:`FakeLLMClient`
   (no HTTP, no optional deps).
2. **HTTP integration tests** -- the FastAPI routes are exercised end-to-end via
   :class:`fastapi.testclient.TestClient` (guarded by ``importorskip``).

The HTTP layer is a thin shell over the service, so the bulk of the assertions
live in the service tests.
"""

from __future__ import annotations

import json

import pytest

from sparksage.api import (
    ConvertOutput,
    GenerateOutput,
    GenerationNotConfiguredError,
    SparkSageService,
)
from sparksage.api.schemas import (
    ConvertResponse,
    GenerateResponse,
    to_convert_response,
    to_generate_response,
)
from sparksage.clean.cleaner import TextCleaner
from sparksage.convert.backend import FakeConverterBackend
from sparksage.convert.converter import MarkdownConverter
from sparksage.generator.client import FakeLLMClient
from sparksage.generator.generator import GenerationStats, IdeaBlockGenerator
from sparksage.schema.enums import BlockStatus, Tag

# ---------------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------------- #
VALID_BLOCKS_JSON = json.dumps(
    {
        "blocks": [
            {
                "name": "Revenue",
                "critical_question": "How did revenue grow?",
                "trusted_answer": "Revenue grew 12% year over year.",
                "tags": ["important"],
                "keywords": ["revenue"],
            },
            {
                "name": "Strategy",
                "critical_question": "What is the expansion plan?",
                "trusted_answer": "Expand into the APAC region next year.",
                "tags": ["process"],
                "keywords": ["strategy", "apac"],
            },
        ]
    }
)

NOISY_MARKDOWN = (
    "\ufeff# Report\n"
    "CONFIDENTIAL\n\n\n\n"
    "Revenue grew 12%.\n"
    "Page 1 of 5\n"
)


def _fake_converter(markdown: str = NOISY_MARKDOWN, title: str | None = "Report"):
    return MarkdownConverter(backend=FakeConverterBackend(markdown=markdown, title=title))


def _fake_generator(response: str = VALID_BLOCKS_JSON):
    return IdeaBlockGenerator(FakeLLMClient(responses=[response]))


def _service(
    *,
    markdown: str = NOISY_MARKDOWN,
    title: str | None = "Report",
    llm_response: str = VALID_BLOCKS_JSON,
    with_generator: bool = True,
    cleaner: TextCleaner | None = None,
) -> SparkSageService:
    generator = _fake_generator(llm_response) if with_generator else None
    return SparkSageService(
        converter=_fake_converter(markdown=markdown, title=title),
        cleaner=cleaner or TextCleaner(),
        generator=generator,
    )


# ---------------------------------------------------------------------------- #
# SparkSageService.convert
# ---------------------------------------------------------------------------- #
class TestServiceConvert:
    def test_convert_returns_markdown(self):
        svc = _service()
        out = svc.convert(b"raw bytes", "doc.md")
        assert isinstance(out, ConvertOutput)
        assert out.markdown == NOISY_MARKDOWN
        assert out.title == "Report"
        assert out.cleaned is False

    def test_convert_provenance_uses_filename_not_temp_path(self):
        svc = _service()
        out = svc.convert(b"data", "reports/annual.pdf")
        assert out.source.uri == "reports/annual.pdf"
        assert "/tmp" not in out.source.uri

    def test_convert_accepts_str(self):
        svc = _service()
        out = svc.convert("raw text", "note.txt")
        assert out.markdown == NOISY_MARKDOWN

    def test_convert_no_filename(self):
        svc = _service()
        out = svc.convert(b"data")
        assert out.source.uri

    def test_convert_with_cleaning(self):
        svc = _service()
        out = svc.convert(b"data", "doc.pdf", clean=True)
        assert out.cleaned is True
        assert "\ufeff" not in out.markdown
        assert "\n\n\n" not in out.markdown

    def test_convert_with_business_cleaning_rule(self):
        from sparksage import RegexReplaceRule
        from sparksage.clean.cleaner import TextCleaner

        cleaner = TextCleaner()
        cleaner.add(RegexReplaceRule(r"CONFIDENTIAL", ""))
        svc = _service(cleaner=cleaner)
        out = svc.convert(b"data", "doc.pdf", clean=True)
        assert out.cleaned is True
        assert "\ufeff" not in out.markdown
        assert "CONFIDENTIAL" not in out.markdown

    def test_convert_without_cleaning_keeps_noise(self):
        svc = _service()
        out = svc.convert(b"data", "doc.pdf", clean=False)
        assert out.cleaned is False
        assert "\ufeff" in out.markdown
        assert "CONFIDENTIAL" in out.markdown

    def test_convert_does_not_require_generator(self):
        svc = _service(with_generator=False)
        out = svc.convert(b"data", "doc.md")
        assert out.markdown == NOISY_MARKDOWN

    def test_convert_temp_file_extension_preserved(self, tmp_path):
        """The temp file must carry the original extension for format detection."""
        from sparksage.api.pipeline import _named_temp_file, _temp_suffix

        assert _temp_suffix("a.pdf") == ".pdf"
        assert _temp_suffix("no_ext") == ""

        with _named_temp_file(b"x", "note.docx") as p:
            assert p.suffix == ".docx"
        assert not p.exists()


# ---------------------------------------------------------------------------- #
# SparkSageService.generate
# ---------------------------------------------------------------------------- #
class TestServiceGenerate:
    def test_generate_returns_blocks(self):
        svc = _service()
        out = svc.generate(b"data", "doc.md")
        assert isinstance(out, GenerateOutput)
        assert len(out.blocks) == 2
        assert out.blocks[0].name == "Revenue"
        assert out.blocks[1].name == "Strategy"

    def test_generate_attaches_provenance(self):
        svc = _service()
        out = svc.generate(b"data", "docs/report.pdf")
        for block in out.blocks:
            assert block.source is not None
            assert block.source.uri == "docs/report.pdf"

    def test_generate_defaults_clean_true(self):
        svc = _service()
        out = svc.generate(b"data", "doc.md")
        assert out.cleaned is True

    def test_generate_without_cleaning(self):
        svc = _service()
        out = svc.generate(b"data", "doc.md", clean=False)
        assert out.cleaned is False

    def test_generate_with_stats(self):
        svc = _service()
        out = svc.generate(b"data", "doc.md", with_stats=True)
        assert out.stats is not None
        assert isinstance(out.stats, GenerationStats)
        assert out.stats.emitted == 2
        assert out.stats.raw_block_count == 2

    def test_generate_without_stats(self):
        svc = _service()
        out = svc.generate(b"data", "doc.md", with_stats=False)
        assert out.stats is None

    def test_generate_requires_generator(self):
        svc = _service(with_generator=False)
        with pytest.raises(GenerationNotConfiguredError):
            svc.generate(b"data", "doc.md")

    def test_generate_with_max_blocks_and_language(self):
        svc = _service()
        out = svc.generate(
            b"data", "doc.md", max_blocks=5, language="zh", with_stats=True
        )
        for block in out.blocks:
            assert block.language == "zh"

    def test_generate_empty_blocks(self):
        svc = _service(llm_response='{"blocks": []}')
        out = svc.generate(b"data", "doc.md")
        assert out.blocks == []


# ---------------------------------------------------------------------------- #
# SparkSageService properties
# ---------------------------------------------------------------------------- #
class TestServiceProperties:
    def test_has_generator_true(self):
        svc = _service(with_generator=True)
        assert svc.has_generator is True
        assert svc.generator is not None

    def test_has_generator_false(self):
        svc = _service(with_generator=False)
        assert svc.has_generator is False
        assert svc.generator is None

    def test_default_cleaner_when_none(self):
        svc = SparkSageService(
            converter=_fake_converter(), cleaner=None, generator=None
        )
        assert isinstance(svc.cleaner, TextCleaner)

    def test_converter_property(self):
        conv = _fake_converter()
        svc = SparkSageService(converter=conv)
        assert svc.converter is conv


# ---------------------------------------------------------------------------- #
# Response-schema mappers
# ---------------------------------------------------------------------------- #
class TestSchemaMappers:
    def test_to_convert_response(self):
        svc = _service()
        out = svc.convert(b"data", "doc.md", clean=True)
        resp = to_convert_response(out)
        assert isinstance(resp, ConvertResponse)
        assert resp.cleaned is True
        assert resp.source.uri == "doc.md"
        assert resp.title == "Report"
        assert resp.markdown == out.markdown

    def test_to_generate_response(self):
        svc = _service()
        out = svc.generate(b"data", "doc.md", with_stats=True)
        resp = to_generate_response(out)
        assert isinstance(resp, GenerateResponse)
        assert len(resp.blocks) == 2
        assert resp.blocks[0]["name"] == "Revenue"
        assert resp.stats is not None
        assert resp.stats.emitted == 2
        assert resp.source.uri == "doc.md"
        assert resp.cleaned is True

    def test_generate_response_blocks_are_json_serializable(self):
        svc = _service()
        out = svc.generate(b"data", "doc.md")
        resp = to_generate_response(out)
        blob = json.dumps(resp.model_dump(mode="json"))
        assert "Revenue" in blob

    def test_blocks_contain_expected_fields(self):
        svc = _service()
        out = svc.generate(b"data", "doc.md")
        resp = to_generate_response(out)
        block = resp.blocks[0]
        for key in (
            "id",
            "name",
            "critical_question",
            "trusted_answer",
            "tags",
            "keywords",
            "language",
            "status",
            "version",
        ):
            assert key in block
        assert block["status"] == BlockStatus.DRAFT.value
        assert Tag.IMPORTANT.value in [t for t in block["tags"]]


# ---------------------------------------------------------------------------- #
# HTTP integration tests (FastAPI TestClient)
# ---------------------------------------------------------------------------- #
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from sparksage.api.app import create_app  # noqa: E402


@pytest.fixture
def http_client():
    """A TestClient wired to a service using deterministic fakes."""
    svc = _service()
    app = create_app(service=svc)
    return TestClient(app)


class TestHealthRoute:
    def test_health_ok(self, http_client):
        resp = http_client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["generator_configured"] is True


class TestConvertRoute:
    def test_convert_returns_markdown(self, http_client):
        resp = http_client.post(
            "/api/v1/convert",
            files={"file": ("doc.md", b"hello", "text/plain")},
            data={"clean": "false"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["markdown"] == NOISY_MARKDOWN
        assert body["cleaned"] is False
        assert body["source"]["uri"] == "doc.md"
        assert body["title"] == "Report"

    def test_convert_with_cleaning(self, http_client):
        resp = http_client.post(
            "/api/v1/convert",
            files={"file": ("doc.pdf", b"hello", "application/pdf")},
            data={"clean": "true"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["cleaned"] is True
        assert "\ufeff" not in body["markdown"]
        assert "\n\n\n" not in body["markdown"]

    def test_convert_missing_file(self, http_client):
        resp = http_client.post("/api/v1/convert", data={"clean": "false"})
        assert resp.status_code == 422

    def test_convert_works_without_generator(self):
        svc = _service(with_generator=False)
        app = create_app(service=svc)
        client = TestClient(app)
        resp = client.post(
            "/api/v1/convert",
            files={"file": ("doc.md", b"x", "text/plain")},
        )
        assert resp.status_code == 200


class TestGenerateRoute:
    def test_generate_returns_blocks(self, http_client):
        resp = http_client.post(
            "/api/v1/generate",
            files={"file": ("doc.md", b"hello", "text/plain")},
            data={"clean": "true", "with_stats": "true"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["blocks"]) == 2
        assert body["blocks"][0]["name"] == "Revenue"
        assert body["source"]["uri"] == "doc.md"
        assert body["cleaned"] is True
        assert body["stats"]["emitted"] == 2

    def test_generate_without_stats(self, http_client):
        resp = http_client.post(
            "/api/v1/generate",
            files={"file": ("doc.md", b"hello", "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.json()["stats"] is None

    def test_generate_503_when_no_generator(self):
        svc = _service(with_generator=False)
        app = create_app(service=svc)
        client = TestClient(app)
        resp = client.post(
            "/api/v1/generate",
            files={"file": ("doc.md", b"hello", "text/plain")},
        )
        assert resp.status_code == 503

    def test_generate_missing_file(self, http_client):
        resp = http_client.post("/api/v1/generate")
        assert resp.status_code == 422

    def test_generate_max_blocks_validation(self, http_client):
        resp = http_client.post(
            "/api/v1/generate",
            files={"file": ("doc.md", b"x", "text/plain")},
            data={"max_blocks": "0"},
        )
        assert resp.status_code == 422

    def test_generate_language(self, http_client):
        resp = http_client.post(
            "/api/v1/generate",
            files={"file": ("doc.md", b"hello", "text/plain")},
            data={"language": "zh"},
        )
        assert resp.status_code == 200
        for block in resp.json()["blocks"]:
            assert block["language"] == "zh"
