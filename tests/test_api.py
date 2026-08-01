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


# ---------------------------------------------------------------------------- #
# SparkSageService document management
# ---------------------------------------------------------------------------- #
DOC_MARKDOWN = (
    "# Revenue Report\n"
    "Revenue grew 12 percent this quarter. The growth came from APAC expansion. "
    "Machine learning models improved retrieval ranking. Next year we enter Europe."
)


def _doc_service(*, markdown: str = DOC_MARKDOWN) -> SparkSageService:
    return SparkSageService(
        converter=_fake_converter(markdown=markdown, title="Revenue Report"),
        cleaner=TextCleaner(),
        generator=None,
    )


class TestServiceDocuments:
    def test_auto_tag_extracts_tags(self):
        svc = _doc_service()
        tags = svc.auto_tag(DOC_MARKDOWN, top_k=5, max_tag_words=3)
        assert tags
        assert len(tags) <= 5
        # tags are tag-shaped (<= 3 words for latin)
        for t in tags:
            assert len(t.split()) <= 3 or any(ord(c) >= 0x3000 for c in t)

    def test_auto_tag_dedupes(self):
        svc = _doc_service()
        tags = svc.auto_tag("revenue revenue revenue ranking ranking", top_k=5)
        assert len(set(tags)) == len(tags)

    def test_auto_tag_validation(self):
        svc = _doc_service()
        with pytest.raises(ValueError, match="top_k"):
            svc.auto_tag("x", top_k=0)

    def test_summarize_text(self):
        svc = _doc_service()
        summary = svc.summarize_text(DOC_MARKDOWN, max_sentences=2)
        assert isinstance(summary, str) and "#" not in summary

    def test_ingest_auto_tags_when_no_tags(self):
        svc = _doc_service()
        rec = svc.ingest_document(b"data", "report.md", top_k=4)
        assert rec.tags
        assert rec.title == "Revenue Report"
        assert rec.summary
        assert rec.source.uri == "report.md"
        assert rec.doc_id in svc.document_store

    def test_ingest_keeps_supplied_tags(self):
        svc = _doc_service()
        rec = svc.ingest_document(
            b"data", "report.md", tags=["manual", "tags"], auto_tag=True
        )
        assert rec.tags == ["manual", "tags"]

    def test_ingest_disables_summary(self):
        svc = _doc_service()
        rec = svc.ingest_document(b"data", "report.md", summarize=False)
        assert rec.summary is None

    def test_ingest_explicit_title_and_id(self):
        svc = _doc_service()
        rec = svc.ingest_document(
            b"data", "report.md", title="Custom", doc_id="my-id"
        )
        assert rec.title == "Custom"
        assert rec.doc_id == "my-id"

    def test_list_filter_and_count(self):
        svc = _doc_service()
        a = svc.ingest_document(b"a", "a.md", tags=["alpha"])
        b = svc.ingest_document(b"b", "b.md", tags=["beta"])
        assert svc.count_documents() == 2
        assert svc.count_documents(tag="alpha") == 1
        tag_a = svc.list_documents(tag="alpha")
        assert [r.doc_id for r in tag_a] == [a.doc_id]
        q = svc.list_documents(q="report")
        assert {r.doc_id for r in q} == {a.doc_id, b.doc_id}

    def test_list_multi_tag_any_match(self):
        svc = _doc_service()
        a = svc.ingest_document(b"a", "a.md", tags=["alpha"])
        b = svc.ingest_document(b"b", "b.md", tags=["beta"])
        svc.ingest_document(b"c", "c.md", tags=["gamma"])
        assert {r.doc_id for r in svc.list_documents(tags=["alpha", "beta"])} == {
            a.doc_id,
            b.doc_id,
        }
        assert svc.count_documents(tags=["alpha", "beta"]) == 2
        assert svc.list_documents(tags=["missing"]) == []

    def test_get_missing_is_none(self):
        assert _doc_service().get_document("nope") is None

    def test_delete(self):
        svc = _doc_service()
        rec = svc.ingest_document(b"data", "x.md")
        assert svc.delete_document(rec.doc_id) is True
        assert svc.delete_document(rec.doc_id) is False
        assert svc.get_document(rec.doc_id) is None

    def test_update_document_partial(self):
        svc = _doc_service()
        rec = svc.ingest_document(b"data", "x.md")
        upd = svc.update_document(rec.doc_id, title="New", tags=["t1", "t2"])
        assert upd.title == "New"
        assert upd.tags == ["t1", "t2"]
        assert upd.body_markdown == rec.body_markdown  # untouched

    def test_update_document_missing_raises(self):
        with pytest.raises(KeyError):
            _doc_service().update_document("nope", title="x")

    def test_retag_replace_and_append(self):
        svc = _doc_service()
        rec = svc.ingest_document(b"data", "report.md", tags=["seed"])
        replaced = svc.retag_document(rec.doc_id, top_k=3, replace=True)
        assert "seed" not in replaced.tags
        appended = svc.retag_document(
            rec.doc_id, top_k=3, replace=False, extra_tags=["custom"]
        )
        assert "custom" in appended.tags

    def test_retag_missing_raises(self):
        with pytest.raises(KeyError):
            _doc_service().retag_document("nope")

    def test_list_document_tags(self):
        svc = _doc_service()
        svc.ingest_document(b"a", "a.md", tags=["zeta", "alpha"])
        svc.ingest_document(b"b", "b.md", tags=["mu"])
        assert svc.list_document_tags() == ["alpha", "mu", "zeta"]

    def test_has_document_store_after_access(self):
        svc = _doc_service()
        assert svc.has_document_store is False
        _ = svc.document_store
        assert svc.has_document_store is True

    def test_default_store_is_in_memory(self):
        from sparksage import InMemoryDocumentStore

        svc = _doc_service()
        assert isinstance(svc.document_store, InMemoryDocumentStore)


# ---------------------------------------------------------------------------- #
# HTTP integration tests: /documents + /tags routes
# ---------------------------------------------------------------------------- #
class TestDocumentRoutes:
    @pytest.fixture
    def doc_client(self):
        svc = _doc_service()
        app = create_app(service=svc)
        return TestClient(app)

    def _create_document(self, client) -> str:
        """Create a document and return its id (shared helper)."""
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("report.md", b"data", "text/plain")},
            data={"top_k": "5"},
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["doc_id"]

    def test_create_document(self, doc_client):
        resp = doc_client.post(
            "/api/v1/documents",
            files={"file": ("report.md", b"data", "text/plain")},
            data={"top_k": "5"},
        )
        assert resp.status_code == 200, resp.text
        doc = resp.json()
        assert doc["title"] == "Revenue Report"
        assert doc["tags"]
        assert doc["summary"]
        assert "body_markdown" in doc
        assert doc["source"]["uri"] == "report.md"

    def test_create_with_explicit_tags_skips_auto(self, doc_client):
        resp = doc_client.post(
            "/api/v1/documents",
            files={"file": ("report.md", b"data", "text/plain")},
            data={"tags": "a,b,c", "auto_tag": "true"},
        )
        assert resp.status_code == 200
        assert resp.json()["tags"] == ["a", "b", "c"]

    def test_create_without_summary(self, doc_client):
        resp = doc_client.post(
            "/api/v1/documents",
            files={"file": ("report.md", b"data", "text/plain")},
            data={"summarize": "false"},
        )
        assert resp.status_code == 200
        assert resp.json()["summary"] is None

    def test_create_missing_file(self, doc_client):
        resp = doc_client.post("/api/v1/documents", data={"top_k": "5"})
        assert resp.status_code == 422

    def test_create_top_k_validation(self, doc_client):
        resp = doc_client.post(
            "/api/v1/documents",
            files={"file": ("report.md", b"data", "text/plain")},
            data={"top_k": "0"},
        )
        assert resp.status_code == 422

    def test_list_and_filter(self, doc_client):
        # seed two docs with distinct tags
        for tags, name in (("alpha", "a.md"), ("beta", "b.md")):
            doc_client.post(
                "/api/v1/documents",
                files={"file": (name, b"data", "text/plain")},
                data={"tags": tags},
            )
        resp = doc_client.get("/api/v1/documents")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2 and body["total"] == 2
        assert all("body_markdown" not in item for item in body["items"])

        resp = doc_client.get("/api/v1/documents", params={"tag": "alpha"})
        assert resp.json()["count"] == 1

        resp = doc_client.get("/api/v1/documents", params={"q": "revenue"})
        assert resp.json()["count"] == 2

    def test_get_and_404(self, doc_client):
        doc_id = self._create_document(doc_client)
        resp = doc_client.get(f"/api/v1/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["doc_id"] == doc_id

        resp = doc_client.get("/api/v1/documents/missing")
        assert resp.status_code == 404

    def test_patch_and_404(self, doc_client):
        doc_id = self._create_document(doc_client)
        resp = doc_client.patch(
            f"/api/v1/documents/{doc_id}",
            json={"title": "Patched", "tags": ["x", "y"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["title"] == "Patched" and body["tags"] == ["x", "y"]

        resp = doc_client.patch(
            "/api/v1/documents/missing", json={"title": "x"}
        )
        assert resp.status_code == 404

    def test_delete_and_404(self, doc_client):
        doc_id = self._create_document(doc_client)
        resp = doc_client.delete(f"/api/v1/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        resp = doc_client.delete(f"/api/v1/documents/{doc_id}")
        assert resp.status_code == 404

    def test_retag_and_404(self, doc_client):
        doc_id = self._create_document(doc_client)
        resp = doc_client.post(
            f"/api/v1/documents/{doc_id}/tags",
            json={"top_k": 3, "replace": True},
        )
        assert resp.status_code == 200
        assert resp.json()["tags"]

        resp = doc_client.post(
            "/api/v1/documents/missing/tags", json={"top_k": 3}
        )
        assert resp.status_code == 404

    def test_tags_route(self, doc_client):
        doc_client.post(
            "/api/v1/documents",
            files={"file": ("a.md", b"data", "text/plain")},
            data={"tags": "zeta,alpha"},
        )
        resp = doc_client.get("/api/v1/tags")
        assert resp.status_code == 200
        assert resp.json()["tags"] == ["alpha", "zeta"]
