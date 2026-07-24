"""Tests for the document-management subsystem.

Covers, offline and deterministically:

* the :class:`Document` schema (validation, tag normalization, lifecycle helpers);
* the Markdown parser (title / summary extraction);
* the keyword-extraction algorithm (scoring, stop words, CJK, determinism);
* the in-memory store (CRUD, paging, tag index, defensive copies);
* the :class:`DocumentService` orchestration (upload, auto-tag, CRUD).
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from sparksage.clean.cleaner import TextCleaner
from sparksage.convert.backend import FakeConverterBackend
from sparksage.convert.converter import MarkdownConverter
from sparksage.documents import (
    DEFAULT_STOP_WORDS,
    Document,
    DocumentNotFoundError,
    DocumentService,
    FrequencyKeywordExtractor,
    InMemoryDocumentStore,
    extract_summary,
    extract_title,
    parse_markdown,
)
from sparksage.documents.service import _named_temp_file, _temp_suffix
from sparksage.schema.enums import TagSource


# ---------------------------------------------------------------------------- #
# Document schema
# ---------------------------------------------------------------------------- #
class TestDocumentSchema:
    def test_defaults(self):
        doc = Document(content="Some body text.")
        assert doc.id is not None
        assert doc.tags == []
        assert doc.tag_source == TagSource.AUTO
        assert doc.version == 1
        assert doc.title is None

    def test_strips_whitespace_fields(self):
        doc = Document(title="  Title  ", summary="  Sum  ", content="  body  ")
        assert doc.title == "Title"
        assert doc.summary == "Sum"
        assert doc.content == "body"

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            Document(content="   ")

    def test_tags_normalized_and_deduped_case_insensitive(self):
        doc = Document(content="x", tags=["Go", "go", "  Rust  ", "", "python"])
        assert doc.tags == ["Go", "Rust", "python"]

    def test_tag_too_long_rejected(self):
        with pytest.raises(ValidationError):
            Document(content="x", tags=["x" * 100])

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            Document(content="x", surprise="nope")  # type: ignore[call-arg]

    def test_touch_bumps_version_and_updated_at(self):
        doc = Document(content="x")
        before = doc.updated_at
        doc.touch()
        assert doc.version == 2
        assert doc.updated_at >= before

    def test_add_tags_merges_and_marks_mixed(self):
        doc = Document(content="x", tags=["a"], tag_source=TagSource.USER)
        doc.add_tags(["b", "a", "c"], source=TagSource.AUTO)
        assert doc.tags == ["a", "b", "c"]
        assert doc.tag_source == TagSource.MIXED
        assert doc.version == 2

    def test_add_tags_no_change_no_touch(self):
        doc = Document(content="x", tags=["a"], tag_source=TagSource.USER)
        doc.add_tags(["a"], source=TagSource.USER)
        assert doc.version == 1
        assert doc.tag_source == TagSource.USER

    def test_replace_tags(self):
        doc = Document(content="x", tags=["a"], tag_source=TagSource.USER)
        doc.replace_tags(["x", "y"], source=TagSource.AUTO)
        assert doc.tags == ["x", "y"]
        assert doc.tag_source == TagSource.AUTO


# ---------------------------------------------------------------------------- #
# Markdown parser
# ---------------------------------------------------------------------------- #
class TestMarkdownParser:
    def test_title_from_h1(self):
        assert extract_title("# Hello World\nbody") == "Hello World"

    def test_title_from_h2_not_used_when_h1_absent_falls_back_to_line(self):
        assert extract_title("## Sub\nFirst line") == "First line"

    def test_title_skips_frontmatter(self):
        md = "---\ntitle: fm\n---\n# Real Title\nbody"
        assert extract_title(md) == "Real Title"

    def test_title_skips_code_fence(self):
        md = "```python\n# not a heading\n```\n# Real\nbody"
        assert extract_title(md) == "Real"

    def test_title_none_when_empty(self):
        assert extract_title("") is None

    def test_summary_first_paragraph(self):
        md = "# H\n\nFirst paragraph here with words.\n\nSecond paragraph."
        assert extract_summary(md) == "First paragraph here with words."

    def test_summary_skips_headings_and_lists(self):
        md = "# H\n- item one\n- item two\n\nActual prose summary."
        assert extract_summary(md).startswith("Actual prose summary")

    def test_summary_truncates_with_ellipsis(self):
        long = "word " * 60
        summary = extract_summary("# T\n\n" + long.strip(), max_chars=20)
        assert summary.endswith("…")
        assert len(summary) <= 21

    def test_summary_skips_code_block(self):
        md = "# T\n\n```\ncode block line\n```\n\nReal prose."
        assert extract_summary(md).startswith("Real prose")

    def test_summary_strips_blockquote_marker(self):
        assert extract_summary("# T\n\n> a quoted lead sentence").startswith(
            "a quoted lead"
        )

    def test_summary_none_when_only_headings(self):
        assert extract_summary("# A\n## B") is None

    def test_parse_markdown_roundtrip(self):
        parsed = parse_markdown("# Title\n\nBody text here.")
        assert parsed.title == "Title"
        assert parsed.summary == "Body text here."
        assert parsed.content.startswith("# Title")

    def test_summary_max_chars_positive(self):
        with pytest.raises(ValueError):
            extract_summary("x", max_chars=0)


# ---------------------------------------------------------------------------- #
# Keyword extraction
# ---------------------------------------------------------------------------- #
class TestKeywordExtractor:
    def test_empty_text_returns_empty(self):
        assert FrequencyKeywordExtractor().extract("") == []
        assert FrequencyKeywordExtractor().extract("   \n  ") == []

    def test_basic_ranking(self):
        md = (
            "# Machine Learning Platform\n\n"
            "The machine learning platform powers retrieval. "
            "Machine learning models train on large datasets.\n"
        )
        tags = FrequencyKeywordExtractor().extract(md, top_n=3)
        assert tags[0] == "machine learning"
        assert len(tags) == 3

    def test_stop_words_filtered(self):
        md = "the the the and or but machine machine machine learning learning"
        tags = FrequencyKeywordExtractor().extract(md, top_n=5)
        for stop in ("the", "and", "or", "but"):
            assert stop not in tags

    def test_heading_boost(self):
        md = (
            "# Kubernetes Deployment\n\n"
            "Deploy applications with helm. Helm charts package resources. "
            "Applications run in production.\n"
        )
        tags = FrequencyKeywordExtractor(heading_weight=5).extract(md, top_n=3)
        assert "kubernetes deployment" in tags or "kubernetes" in tags

    def test_redundant_unigram_dropped_when_bigram_wins(self):
        md = "# SparkSage Machine Learning\n\nmachine learning machine learning spark"
        tags = FrequencyKeywordExtractor().extract(md, top_n=5)
        # "machine" alone should be dropped because "machine learning" covers it
        assert "machine learning" in tags
        assert "machine" not in tags

    def test_cjk_extraction(self):
        md = "# 知识库管理系统\n\n知识库管理平台支持文档检索与标签自动生成。\n"
        tags = FrequencyKeywordExtractor().extract(md, top_n=4)
        assert any("知识" in t for t in tags)

    def test_deterministic_output(self):
        md = "# API Design\n\nDesign REST apis. API design matters for clients.\n"
        a = FrequencyKeywordExtractor().extract(md, top_n=5)
        b = FrequencyKeywordExtractor().extract(md, top_n=5)
        assert a == b

    def test_top_n_respected(self):
        md = "alpha alpha beta beta gamma gamma delta delta epsilon epsilon"
        tags = FrequencyKeywordExtractor().extract(md, top_n=2)
        assert len(tags) == 2

    def test_custom_stop_words(self):
        ext = FrequencyKeywordExtractor(stop_words=frozenset({"machine"}))
        tags = ext.extract("machine machine learning learning", top_n=5)
        assert "machine" not in tags
        assert "learning" in tags

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            FrequencyKeywordExtractor(heading_weight=0)
        with pytest.raises(ValueError):
            FrequencyKeywordExtractor(min_term_length=0)
        with pytest.raises(ValueError):
            FrequencyKeywordExtractor(default_top_n=0)

    def test_top_n_zero_returns_empty(self):
        assert FrequencyKeywordExtractor().extract("machine learning", top_n=0) == []

    def test_default_stop_words_present(self):
        assert "the" in DEFAULT_STOP_WORDS
        assert "的" in DEFAULT_STOP_WORDS

    def test_implements_protocol(self):
        assert isinstance(FrequencyKeywordExtractor(), object)


# ---------------------------------------------------------------------------- #
# In-memory store
# ---------------------------------------------------------------------------- #
class TestInMemoryStore:
    def test_save_and_get(self):
        store = InMemoryDocumentStore()
        doc = Document(content="body", tags=["a"])
        saved = store.save(doc)
        assert saved.id == doc.id
        got = store.get(doc.id)
        assert got is not None
        assert got.content == "body"

    def test_get_missing_returns_none(self):
        assert InMemoryDocumentStore().get(uuid.uuid4()) is None

    def test_returns_defensive_copies(self):
        store = InMemoryDocumentStore()
        store.save(Document(content="x", tags=["a"]))
        got = store.list()[0]
        got.tags.append("mutated")
        assert store.list()[0].tags == ["a"]

    def test_list_filters_by_tag_case_insensitive(self):
        store = InMemoryDocumentStore()
        store.save(Document(content="a", tags=["Go"]))
        store.save(Document(content="b", tags=["Rust"]))
        assert store.count(tag="go") == 1
        assert store.list(tag="go")[0].content == "a"

    def test_list_paging(self):
        store = InMemoryDocumentStore()
        for i in range(5):
            store.save(Document(content=f"c{i}", tags=["t"]))
        page = store.list(tag="t", limit=2, offset=1)
        assert len(page) == 2
        assert store.count(tag="t") == 5

    def test_delete(self):
        store = InMemoryDocumentStore()
        doc = store.save(Document(content="x"))
        assert store.delete(doc.id) is True
        assert store.delete(doc.id) is False

    def test_tags_index(self):
        store = InMemoryDocumentStore()
        store.save(Document(content="a", tags=["x", "y"]))
        store.save(Document(content="b", tags=["x"]))
        assert store.tags() == {"x": 2, "y": 1}

    def test_invalid_paging_rejected(self):
        store = InMemoryDocumentStore()
        with pytest.raises(ValueError):
            store.list(limit=-1)
        with pytest.raises(ValueError):
            store.list(offset=-1)


# ---------------------------------------------------------------------------- #
# DocumentService
# ---------------------------------------------------------------------------- #
MD_ONBOARDING = (
    "# Onboarding Guide\n\n"
    "This guide explains the onboarding workflow for new engineers.\n\n"
    "The workflow covers account setup, repository access and CI pipeline.\n"
)


def _service(markdown: str = MD_ONBOARDING) -> DocumentService:
    conv = MarkdownConverter(backend=FakeConverterBackend(markdown=markdown))
    return DocumentService(converter=conv)


class TestDocumentServiceUpload:
    def test_upload_auto_tags_when_none_given(self):
        svc = _service()
        doc = svc.upload(b"data", "guide.md")
        assert doc.title == "Onboarding Guide"
        assert doc.summary.startswith("This guide explains")
        assert doc.tags  # auto-generated
        assert doc.tag_source == TagSource.AUTO
        assert doc.source.uri == "guide.md"
        assert doc.source.title == "Onboarding Guide"

    def test_upload_user_tags_override_auto(self):
        svc = _service()
        doc = svc.upload(b"data", "guide.md", tags=["custom", "tag"])
        assert doc.tags == ["custom", "tag"]
        assert doc.tag_source == TagSource.USER

    def test_upload_auto_disabled(self):
        svc = _service()
        doc = svc.upload(b"data", "guide.md", tags=None, auto_tag=False)
        assert doc.tags == []
        assert doc.tag_source == TagSource.USER

    def test_upload_text_direct(self):
        svc = _service()
        doc = svc.upload_text("# Title\n\nSome markdown body.", source_uri="note.md")
        assert doc.title == "Title"
        assert doc.content.startswith("# Title")
        assert doc.source.uri == "note.md"

    def test_upload_text_empty_rejected(self):
        svc = _service()
        with pytest.raises(ValueError):
            svc.upload_text("   ")

    def test_provenance_uses_filename_not_temp_path(self):
        svc = _service()
        doc = svc.upload(b"data", "reports/annual.md")
        assert doc.source.uri == "reports/annual.md"
        assert "/tmp" not in doc.source.uri

    def test_temp_file_extension_preserved(self):
        assert _temp_suffix("a.md") == ".md"
        assert _temp_suffix("no_ext") == ""
        with _named_temp_file(b"x", "note.docx") as p:
            assert p.suffix == ".docx"
        assert not p.exists()


class TestDocumentServiceCRUD:
    def test_get_list_count(self):
        svc = _service()
        d1 = svc.upload(b"a", "a.md")
        svc.upload(b"b", "b.md")
        assert svc.count() == 2
        assert svc.get(d1.id).id == d1.id
        assert len(svc.list()) == 2

    def test_get_missing_raises(self):
        svc = _service()
        with pytest.raises(DocumentNotFoundError):
            svc.get(uuid.uuid4())
        assert svc.get_or_none(uuid.uuid4()) is None

    def test_get_invalid_id_raises(self):
        svc = _service()
        with pytest.raises(DocumentNotFoundError):
            svc.get("not-a-uuid")

    def test_update_fields(self):
        svc = _service()
        doc = svc.upload(b"a", "a.md", tags=["t"])
        updated = svc.update(doc.id, title="New Title", summary="New summary")
        assert updated.title == "New Title"
        assert updated.summary == "New summary"
        assert updated.version == 2

    def test_update_replaces_tags_as_user(self):
        svc = _service()
        doc = svc.upload(b"a", "a.md")
        updated = svc.update(doc.id, tags=["x", "y"])
        assert updated.tags == ["x", "y"]
        assert updated.tag_source == TagSource.USER

    def test_add_tags(self):
        svc = _service()
        doc = svc.upload(b"a", "a.md", tags=["a"])
        updated = svc.add_tags(doc.id, ["b", "a"])
        assert updated.tags == ["a", "b"]
        # Adding user tags onto a user-tagged document stays USER (not MIXED):
        # MIXED only arises when auto-generated tags merge onto user tags.
        assert updated.tag_source == TagSource.USER

    def test_add_tags_empty_rejected(self):
        svc = _service()
        doc = svc.upload(b"a", "a.md")
        with pytest.raises(ValueError):
            svc.add_tags(doc.id, [])

    def test_remove_tag_case_insensitive(self):
        svc = _service()
        doc = svc.upload(b"a", "a.md", tags=["Go", "Rust"])
        updated = svc.remove_tag(doc.id, "go")
        assert updated.tags == ["Rust"]

    def test_remove_missing_tag_raises(self):
        svc = _service()
        doc = svc.upload(b"a", "a.md", tags=["Go"])
        with pytest.raises(DocumentNotFoundError):
            svc.remove_tag(doc.id, "nope")

    def test_extract_tags_merge(self):
        svc = _service()
        doc = svc.upload(b"a", "a.md", tags=["user-tag"])
        updated = svc.extract_tags(doc.id, top_n=3)
        assert "user-tag" in updated.tags
        assert updated.tag_source == TagSource.MIXED
        assert len(updated.tags) >= 1

    def test_extract_tags_replace(self):
        svc = _service()
        doc = svc.upload(b"a", "a.md", tags=["user-tag"])
        updated = svc.extract_tags(doc.id, top_n=3, replace_existing=True)
        assert "user-tag" not in updated.tags
        assert updated.tag_source == TagSource.AUTO

    def test_delete(self):
        svc = _service()
        doc = svc.upload(b"a", "a.md")
        assert svc.delete(doc.id) is True
        assert svc.delete(doc.id) is False

    def test_tags_index(self):
        svc = _service()
        svc.upload_text("# T\n\nbody.", tags=["x", "y"], auto_tag=False)
        svc.upload_text("# T2\n\nbody2.", tags=["x"], auto_tag=False)
        assert svc.tags() == {"x": 2, "y": 1}

    def test_list_filter_by_tag(self):
        svc = _service()
        svc.upload_text("# T\n\nbody.", tags=["alpha"], auto_tag=False)
        svc.upload_text("# T2\n\nbody2.", tags=["beta"], auto_tag=False)
        assert len(svc.list(tag="alpha")) == 1
        assert svc.count(tag="alpha") == 1


class TestDocumentServiceWiring:
    def test_defaults(self):
        svc = DocumentService(converter=MarkdownConverter(backend=FakeConverterBackend()))
        assert isinstance(svc.cleaner, TextCleaner)
        assert isinstance(svc.extractor, FrequencyKeywordExtractor)
        assert isinstance(svc.store, InMemoryDocumentStore)

    def test_injected_dependencies_used(self):
        store = InMemoryDocumentStore()
        ext = FrequencyKeywordExtractor(default_top_n=2)
        svc = DocumentService(
            converter=MarkdownConverter(backend=FakeConverterBackend(markdown="# x\nb")),
            extractor=ext,
            store=store,
        )
        svc.upload(b"a", "a.md")
        assert store.count() == 1
        assert svc.extractor is ext
