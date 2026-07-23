"""Tests for uniform file-to-Markdown conversion.

All core tests run offline via :class:`FakeConverterBackend`, so no ``markitdown``
installation is required. One integration test (guarded by ``importorskip``)
exercises the real :class:`MarkItDownBackend` end-to-end when the optional
dependency is present.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from sparksage.convert import (
    DEFAULT_EXTENSIONS,
    ConversionResult,
    ConverterBackend,
    FakeConverterBackend,
    MarkdownConverter,
    MarkItDownBackend,
)
from sparksage.convert.converter import (
    _iter_source_files,
    _normalize_extension,
    _output_path_for,
)
from sparksage.schema.source import SourceRef


# ---------------------------------------------------------------------------- #
# ConversionResult
# ---------------------------------------------------------------------------- #
def test_conversion_result_source_ref():
    result = ConversionResult(markdown="# Hi", source="file://docs/a.md", title="A")
    ref = result.source_ref
    assert isinstance(ref, SourceRef)
    assert ref.uri == "file://docs/a.md"
    assert ref.title == "A"


def test_conversion_result_default_title_is_none():
    result = ConversionResult(markdown="x", source="s")
    assert result.title is None
    assert result.source_ref.title is None


# ---------------------------------------------------------------------------- #
# FakeConverterBackend
# ---------------------------------------------------------------------------- #
def test_fake_backend_protocol():
    fake = FakeConverterBackend(markdown="# md")
    assert isinstance(fake, ConverterBackend)


def test_fake_backend_returns_preset_and_records_calls():
    fake = FakeConverterBackend(markdown="# md", title="T")
    md, title = fake.convert("a.pdf")
    assert md == "# md"
    assert title == "T"
    assert fake.calls == ["a.pdf"]
    fake.convert("b.docx")
    assert fake.calls == ["a.pdf", "b.docx"]


def test_fake_backend_by_source_overrides_default(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_text("raw")
    fake = FakeConverterBackend(
        markdown="default",
        by_source={str(p): "per-file"},
    )
    assert fake.convert(p)[0] == "per-file"
    assert fake.convert("other")[0] == "default"


# ---------------------------------------------------------------------------- #
# MarkdownConverter single-source (offline)
# ---------------------------------------------------------------------------- #
def test_convert_returns_result_with_source_descriptor(tmp_path):
    p = tmp_path / "note.pdf"
    p.write_text("bytes")
    conv = MarkdownConverter(backend=FakeConverterBackend(markdown="# Title"))
    result = conv.convert(p)
    assert isinstance(result, ConversionResult)
    assert result.markdown == "# Title"
    assert result.source == str(p)
    assert result.title is None


def test_convert_to_markdown_returns_text_only():
    conv = MarkdownConverter(backend=FakeConverterBackend(markdown="body"))
    assert conv.convert_to_markdown("any://thing") == "body"


def test_convert_to_file_writes_md(tmp_path):
    src = tmp_path / "report.pdf"
    src.write_text("x")
    out_dir = tmp_path / "out"
    conv = MarkdownConverter(backend=FakeConverterBackend(markdown="# Hello"))
    written = conv.convert_to_file(src, out_dir)
    assert written == out_dir / "report.md"
    assert written.read_text(encoding="utf-8") == "# Hello"
    # parent created automatically
    assert out_dir.is_dir()


def test_convert_to_file_custom_suffix(tmp_path):
    src = tmp_path / "data.xlsx"
    src.write_text("x")
    out_dir = tmp_path / "out"
    conv = MarkdownConverter(backend=FakeConverterBackend(markdown="m"))
    written = conv.convert_to_file(src, out_dir, suffix=".markdown")
    assert written.name == "data.markdown"


# ---------------------------------------------------------------------------- #
# MarkdownConverter batch directory (offline)
# ---------------------------------------------------------------------------- #
def _make_tree(tmp_path):
    root = tmp_path / "src"
    (root / "sub").mkdir(parents=True)
    (root / "a.pdf").write_text("a")
    (root / "b.docx").write_text("b")
    (root / "c.bin").write_text("c")
    (root / "sub" / "d.pdf").write_text("d")
    return root


def test_convert_directory_collects_results_in_memory(tmp_path):
    root = _make_tree(tmp_path)
    fake = FakeConverterBackend(markdown="# md")
    conv = MarkdownConverter(backend=fake)
    results = conv.convert_directory(root)
    sources = [r.source for r in results]
    # default extensions include .pdf/.docx but NOT .bin
    assert any(s.endswith("a.pdf") for s in sources)
    assert any(s.endswith("b.docx") for s in sources)
    assert not any(s.endswith("c.bin") for s in sources)
    assert all(r.markdown == "# md" for r in results)


def test_convert_directory_recursive_includes_subdirs(tmp_path):
    root = _make_tree(tmp_path)
    conv = MarkdownConverter(backend=FakeConverterBackend(markdown="m"))
    recursive = conv.convert_directory(root, recursive=True)
    assert any(s.endswith("sub/d.pdf") for s in [r.source for r in recursive])

    flat = conv.convert_directory(root, recursive=False)
    assert not any(s.endswith("sub/d.pdf") for s in [r.source for r in flat])


def test_convert_directory_extension_filter(tmp_path):
    root = _make_tree(tmp_path)
    conv = MarkdownConverter(backend=FakeConverterBackend(markdown="m"))
    results = conv.convert_directory(root, extensions=[".bin"])
    assert [r.source for r in results] == [str(root / "c.bin")]


def test_convert_directory_extensions_normalized(tmp_path):
    root = _make_tree(tmp_path)
    conv = MarkdownConverter(backend=FakeConverterBackend(markdown="m"))
    # uppercase, no dot, mixed -- should still match .pdf only
    results = conv.convert_directory(root, extensions=["PDF", ".DOCX"])
    names = sorted(Path(r.source).name for r in results)
    assert names == ["a.pdf", "b.docx", "d.pdf"]


def test_convert_directory_writes_to_dest(tmp_path):
    root = _make_tree(tmp_path)
    dest = tmp_path / "md_out"
    fake = FakeConverterBackend(markdown="# x")
    conv = MarkdownConverter(backend=fake)
    conv.convert_directory(root, dest_dir=dest)
    assert (dest / "a.md").read_text(encoding="utf-8") == "# x"
    assert (dest / "b.md").exists()
    assert (dest / "d.md").exists()
    assert not (dest / "c.md").exists()


def test_convert_directory_skips_failed_files(tmp_path, caplog):
    root = tmp_path / "src"
    root.mkdir()
    (root / "good.pdf").write_text("g")
    (root / "bad.pdf").write_text("b")

    class Flaky:
        def __init__(self):
            self.n = 0

        def convert(self, source, **kwargs):
            self.n += 1
            if "bad" in str(source):
                raise RuntimeError("boom")
            return "ok", None

    conv = MarkdownConverter(backend=Flaky())
    with caplog.at_level("WARNING"):
        results = conv.convert_directory(root)
    assert [r.source for r in results] == [str(root / "good.pdf")]
    assert results[0].markdown == "ok"
    assert any("bad.pdf" in rec.getMessage() for rec in caplog.records)


def test_convert_directory_missing_dir_raises(tmp_path):
    conv = MarkdownConverter(backend=FakeConverterBackend())
    with pytest.raises(FileNotFoundError):
        conv.convert_directory(tmp_path / "nope")


def test_convert_directory_on_file_raises(tmp_path):
    f = tmp_path / "f.pdf"
    f.write_text("x")
    conv = MarkdownConverter(backend=FakeConverterBackend())
    with pytest.raises(NotADirectoryError):
        conv.convert_directory(f)


# ---------------------------------------------------------------------------- #
# pure helpers
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("pdf", ".pdf"),
        (".PDF", ".pdf"),
        (" .Docx ", ".docx"),
        ("", ""),
    ],
)
def test_normalize_extension(raw, expected):
    assert _normalize_extension(raw) == expected


def test_output_path_for(tmp_path):
    out = _output_path_for(tmp_path / "a" / "b.pdf", tmp_path / "out", suffix=".md")
    assert out == tmp_path / "out" / "b.md"


def test_default_extensions_cover_common_formats():
    for ext in (".pdf", ".docx", ".pptx", ".xlsx", ".html", ".csv", ".json", ".md"):
        assert ext in DEFAULT_EXTENSIONS


def test_iter_source_files_none_extensions_matches_all(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "a.pdf").write_text("a")
    (root / "weird.xyz").write_text("b")
    files = _iter_source_files(root, recursive=True, extensions=None)
    assert sorted(p.name for p in files) == ["a.pdf", "weird.xyz"]


# ---------------------------------------------------------------------------- #
# MarkItDownBackend optional-dependency behaviour
# ---------------------------------------------------------------------------- #
def test_markitdown_backend_missing_dependency(monkeypatch):
    # Make `import markitdown` fail without uninstalling it.
    monkeypatch.setitem(sys.modules, "markitdown", None)
    with pytest.raises(ImportError, match=r"sparksage\[convert\]"):
        MarkItDownBackend()


def test_markitdown_default_backend_instantiated_when_no_backend_given(monkeypatch):
    """MarkdownConverter() must lazily build a MarkItDownBackend."""
    created = {}

    class FakeMI:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

        def convert(self, source, **kwargs):
            return type("R", (), {"markdown": "fake", "title": None})()

    import types

    fake_mod = types.ModuleType("markitdown")
    fake_mod.MarkItDown = FakeMI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "markitdown", fake_mod)

    conv = MarkdownConverter(llm_model="gpt-4o", enable_plugins=False)
    assert created["kwargs"] == {"llm_model": "gpt-4o", "enable_plugins": False}
    assert conv.convert_to_markdown("anything.pdf") == "fake"


# ---------------------------------------------------------------------------- #
# Integration: real markitdown (only when installed)
# ---------------------------------------------------------------------------- #
@pytest.mark.skipif(
    importlib.util.find_spec("markitdown") is None,
    reason="markitdown not installed",
)
def test_real_markitdown_backend_converts_json(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('{"greeting": "hello", "count": 3}', encoding="utf-8")
    conv = MarkdownConverter()
    result = conv.convert(str(p))
    assert "hello" in result.markdown
    assert result.source == str(p)
