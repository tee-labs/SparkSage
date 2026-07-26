"""Tests for the document-management core.

Covers :class:`DocumentRecord` validation, the :class:`DocumentStore` protocol
compliance, the in-memory and SQLite backends (including cross-instance
persistence), and the extractive :class:`ExtractiveSummarizer`. Everything runs
offline with stdlib only (``sqlite3`` ships with Python).
"""

from __future__ import annotations

import pytest

from sparksage import (
    DocumentRecord,
    DocumentStore,
    ExtractiveSummarizer,
    InMemoryDocumentStore,
    SqliteDocumentStore,
    Summarizer,
    content_hash_of,
    new_record,
    split_sentences,
)
from sparksage.schema.source import SourceRef


# ---------------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------------- #
def _record(
    *,
    doc_id: str | None = None,
    body: str = "body markdown content",
    title: str | None = "Title",
    tags: list[str] | None = None,
    source: str = "doc.md",
) -> DocumentRecord:
    kwargs: dict = dict(
        title=title,
        body_markdown=body,
        tags=tags if tags is not None else ["alpha", "beta"],
        source=SourceRef(uri=source, title=title),
    )
    if doc_id is not None:
        kwargs["doc_id"] = doc_id
    return DocumentRecord(**kwargs)


# ---------------------------------------------------------------------------- #
# DocumentRecord
# ---------------------------------------------------------------------------- #
class TestDocumentRecord:
    def test_tags_deduped_and_stripped(self):
        r = _record(tags=["alpha", " alpha ", "beta", "", "alpha"])
        assert r.tags == ["alpha", "beta"]

    def test_content_hash_autocomputed(self):
        r = _record(body="hello world")
        assert r.content_hash == content_hash_of("hello world")

    def test_content_hash_kept_when_provided(self):
        r = DocumentRecord(
            body_markdown="x", source={"uri": "u"}, content_hash="deadbeef"
        )
        assert r.content_hash == "deadbeef"

    def test_doc_id_auto_generated(self):
        r = _record()
        assert r.doc_id and len(r.doc_id) > 0

    def test_doc_id_empty_rejected(self):
        with pytest.raises(ValueError, match="doc_id"):
            DocumentRecord(doc_id="  ", body_markdown="x", source={"uri": "u"})

    def test_extra_forbidden(self):
        with pytest.raises(Exception, match="extra"):
            DocumentRecord(body_markdown="x", source={"uri": "u"}, bogus=1)  # type: ignore[call-arg]

    def test_source_coerced_from_dict(self):
        r = DocumentRecord(body_markdown="x", source={"uri": "u", "title": "t"})
        assert isinstance(r.source, SourceRef)
        assert r.source.uri == "u" and r.source.title == "t"

    def test_new_record_accepts_uri_string(self):
        r = new_record(body_markdown="x", source="file://a.md", tags=["t1", "t1"])
        assert isinstance(r.source, SourceRef)
        assert r.source.uri == "file://a.md"
        assert r.tags == ["t1"]

    def test_mutation_of_returned_does_not_affect_store_input(self):
        r = _record()
        original_tags = list(r.tags)
        store = InMemoryDocumentStore()
        store.save(r)
        r.tags.append("mutated")
        assert store.get(r.doc_id).tags == original_tags


# ---------------------------------------------------------------------------- #
# store behaviour shared by both backends
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "factory",
    [
        lambda: InMemoryDocumentStore(),
        lambda: SqliteDocumentStore(),  # in-memory db
    ],
    ids=["memory", "sqlite-memory"],
)
class TestStoreCore:
    def test_protocol(self, factory):
        assert isinstance(factory(), DocumentStore)

    def test_save_get_round_trip(self, factory):
        store = factory()
        r = _record(tags=["x", "y"], title="T")
        saved = store.save(r)
        got = store.get(saved.doc_id)
        assert got is not None
        assert got.body_markdown == r.body_markdown
        assert got.tags == ["x", "y"]
        assert got.title == "T"
        assert got.source.uri == "doc.md"

    def test_save_returns_copy(self, factory):
        store = factory()
        saved = store.save(_record(tags=["a"]))
        saved.tags.append("hack")
        assert store.get(saved.doc_id).tags == ["a"]

    def test_upsert_by_doc_id(self, factory):
        store = factory()
        r = _record(doc_id="fixed", body="v1", tags=["t1"])
        store.save(r)
        store.save(_record(doc_id="fixed", body="v2", tags=["t2"]))
        got = store.get("fixed")
        assert got.body_markdown == "v2"
        assert got.tags == ["t2"]

    def test_get_missing_is_none(self, factory):
        assert factory().get("nope") is None

    def test_contains_and_len(self, factory):
        store = factory()
        assert len(store) == 0
        r = store.save(_record(doc_id="a"))
        assert "a" in store
        assert r.doc_id in store
        assert "missing" not in store
        assert len(store) == 1

    def test_delete(self, factory):
        store = factory()
        saved = store.save(_record())
        assert store.delete(saved.doc_id) is True
        assert store.delete(saved.doc_id) is False
        assert store.get(saved.doc_id) is None

    def test_list_newest_first(self, factory):
        store = factory()
        a = store.save(_record(doc_id="a"))
        b = store.save(_record(doc_id="b"))
        ids = [r.doc_id for r in store.list()]
        assert ids.index(b.doc_id) < ids.index(a.doc_id)

    def test_tag_filter(self, factory):
        store = factory()
        store.save(_record(doc_id="a", tags=["alpha", "shared"]))
        store.save(_record(doc_id="b", tags=["beta", "shared"]))
        assert {r.doc_id for r in store.list(tag="shared")} == {"a", "b"}
        assert {r.doc_id for r in store.list(tag="alpha")} == {"a"}
        assert store.list(tag="missing") == []

    def test_text_query(self, factory):
        store = factory()
        store.save(_record(doc_id="a", title="Revenue Q3", body="quarterly numbers"))
        store.save(_record(doc_id="b", title="Onboarding", body="new hires"))
        assert [r.doc_id for r in store.list(q="revenue")] == ["a"]
        assert [r.doc_id for r in store.list(q="hires")] == ["b"]
        assert store.list(q="zzzzz") == []

    def test_count(self, factory):
        store = factory()
        store.save(_record(doc_id="a", tags=["t"]))
        store.save(_record(doc_id="b", tags=["t"]))
        store.save(_record(doc_id="c", tags=["other"]))
        assert store.count() == 3
        assert store.count(tag="t") == 2
        assert store.count(tag="other") == 1

    def test_list_tags_sorted(self, factory):
        store = factory()
        store.save(_record(doc_id="a", tags=["zeta", "alpha"]))
        store.save(_record(doc_id="b", tags=["mu"]))
        assert store.list_tags() == ["alpha", "mu", "zeta"]

    def test_pagination(self, factory):
        store = factory()
        for i in range(5):
            store.save(_record(doc_id=f"d{i}", tags=[f"t{i}"]))
        page = store.list(limit=2, offset=1)
        assert len(page) == 2

    def test_pagination_validation(self, factory):
        store = factory()
        with pytest.raises(ValueError, match="limit"):
            store.list(limit=0)
        with pytest.raises(ValueError, match="offset"):
            store.list(offset=-1)


# ---------------------------------------------------------------------------- #
# SQLite-specific: persistence across instances + validation
# ---------------------------------------------------------------------------- #
class TestSqliteSpecific:
    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "persist.db"
        s1 = SqliteDocumentStore(path)
        saved = s1.save(_record(tags=["keep", "me"], body="durable body"))
        s1.close()

        s2 = SqliteDocumentStore(path)
        got = s2.get(saved.doc_id)
        assert got is not None
        assert got.tags == ["keep", "me"]
        assert got.body_markdown == "durable body"
        assert len(s2) == 1
        s2.close()

    def test_invalid_table_name(self):
        with pytest.raises(ValueError, match="invalid table name"):
            SqliteDocumentStore(table="bad name!")

    def test_custom_table_name(self, tmp_path):
        s = SqliteDocumentStore(tmp_path / "c.db", table="my_docs")
        saved = s.save(_record(tags=["t"]))
        assert s.get(saved.doc_id).tags == ["t"]
        assert s.count(tag="t") == 1
        s.close()

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "store.db"
        s = SqliteDocumentStore(path)
        assert path.exists()
        s.close()

    def test_tags_junction_stays_consistent_on_overwrite(self, tmp_path):
        s = SqliteDocumentStore(tmp_path / "o.db")
        rid = s.save(_record(doc_id="d", tags=["a", "b", "c"])).doc_id
        s.save(_record(doc_id="d", tags=["x"]))
        assert s.get(rid).tags == ["x"]
        assert s.list_tags() == ["x"]
        assert s.count(tag="a") == 0
        s.close()

    def test_delete_clears_tags(self, tmp_path):
        s = SqliteDocumentStore(tmp_path / "d.db")
        rid = s.save(_record(tags=["only"])).doc_id
        assert s.delete(rid) is True
        assert s.list_tags() == []
        s.close()

    def test_close_is_idempotent(self, tmp_path):
        s = SqliteDocumentStore(tmp_path / "cl.db")
        s.close()
        s.close()  # second close must not raise


# ---------------------------------------------------------------------------- #
# summarizer
# ---------------------------------------------------------------------------- #
class TestSummarizer:
    def test_is_summarizer(self):
        assert isinstance(ExtractiveSummarizer(), Summarizer)

    def test_returns_string(self):
        text = "First sentence. Second sentence here. Third one now. End."
        out = ExtractiveSummarizer().summarize(text, max_sentences=2)
        assert isinstance(out, str) and out

    def test_strips_markdown_headings(self):
        text = (
            "# Revenue Report\n"
            "Revenue grew 12 percent this quarter.\n"
            "The growth came from APAC expansion."
        )
        out = ExtractiveSummarizer().summarize(text, max_sentences=1)
        assert not out.lstrip().startswith("#")

    def test_respects_max_sentences(self):
        text = (
            "Alpha sentence here. Beta sentence there. Gamma sentence now. "
            "Delta sentence end."
        )
        out = ExtractiveSummarizer().summarize(text, max_sentences=2)
        assert out.count(" ") <= 20  # at most 2 sentences worth of words

    def test_short_text_fallback(self):
        out = ExtractiveSummarizer().summarize("tiny", max_sentences=3)
        assert out == "tiny"

    def test_validation(self):
        with pytest.raises(ValueError, match="max_sentences"):
            ExtractiveSummarizer().summarize("x y z", max_sentences=0)
        with pytest.raises(TypeError, match="max_sentences"):
            ExtractiveSummarizer().summarize("x y z", max_sentences=2.5)  # type: ignore[arg-type]

    def test_split_sentences_handles_cjk(self):
        s = split_sentences("收入增长。亚太扩张。下一步进入欧洲。")
        assert len(s) == 3

    def test_split_sentences_strips_emphasis(self):
        parts = split_sentences("## Heading\n**Bold** start. Normal end.")
        joined = " ".join(parts)
        assert "#" not in joined
