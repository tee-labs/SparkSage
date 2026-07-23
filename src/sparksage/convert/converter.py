"""Uniform file-to-Markdown conversion powered by a pluggable backend.

:class:`MarkdownConverter` normalizes heterogeneous source documents (PDF, Word,
PowerPoint, Excel, HTML, CSV/JSON/XML, images, audio, EPub, ...) into Markdown,
the lingua franca downstream chunking/generation expects. It delegates the
actual per-format work to a :class:`~sparksage.convert.backend.ConverterBackend`
(Microsoft ``markitdown`` by default) and adds:

* a typed :class:`ConversionResult` that slots straight into
  :class:`~sparksage.generator.IdeaBlockGenerator` -- feed ``result.markdown`` as
  the text and build a :class:`~sparksage.schema.source.SourceRef` from
  ``result.source`` (or just use the :attr:`ConversionResult.source_ref`
  property);
* batch directory conversion with extension filtering and recursion;
* one-shot write-to-disk helpers.

The core never imports ``markitdown`` directly, so it is deterministic and
unit-testable without the optional dependency.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sparksage.convert.backend import ConverterBackend, MarkItDownBackend
from sparksage.schema.source import SourceRef

_logger = logging.getLogger(__name__)

#: File extensions included by :meth:`MarkdownConverter.convert_directory` by
#: default. Covers every format the ``markitdown`` built-ins ship a converter
#: for. Pass an explicit ``extensions`` to narrow or widen it.
DEFAULT_EXTENSIONS: frozenset[str] = frozenset(
    {
        # documents
        ".pdf", ".docx", ".doc",
        # presentations
        ".pptx", ".ppt",
        # spreadsheets
        ".xlsx", ".xls",
        # web / markup
        ".html", ".htm", ".xhtml",
        # structured data
        ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",
        # plain text
        ".txt", ".md", ".markdown", ".rst",
        # ebooks
        ".epub",
        # email
        ".msg", ".eml",
        # archives
        ".zip",
        # images
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
        # audio
        ".mp3", ".wav", ".m4a",
    }
)


@dataclass
class ConversionResult:
    """The Markdown rendering of a single source plus its provenance.

    Attributes
    ----------
    markdown:
        The converted Markdown text (the only payload downstream cares about).
    source:
        Stable string descriptor of the input (file path, URI, ...). Used as the
        :attr:`~sparksage.schema.source.SourceRef.uri`.
    title:
        Document title extracted by the backend, if any.
    """

    markdown: str
    source: str
    title: str | None = None

    @property
    def source_ref(self) -> SourceRef:
        """A :class:`SourceRef` pointing back at the original document."""
        return SourceRef(uri=self.source, title=self.title)


def _normalize_extension(ext: str) -> str:
    """Return a lowercased, dot-prefixed extension (``"PDF"`` -> ``".pdf"``)."""
    cleaned = ext.lower().strip()
    if not cleaned:
        return cleaned
    return cleaned if cleaned.startswith(".") else "." + cleaned


def _extension_set(extensions: Iterable[str] | None) -> set[str] | None:
    if extensions is None:
        return None
    return {_normalize_extension(e) for e in extensions}


def _matches(path: Path, ext_set: set[str] | None) -> bool:
    return ext_set is None or path.suffix.lower() in ext_set


def _iter_source_files(
    src_dir: str | Path,
    *,
    recursive: bool,
    extensions: Iterable[str] | None,
) -> list[Path]:
    """Return the sorted list of convertible files under ``src_dir``."""
    root = Path(src_dir)
    if not root.exists():
        raise FileNotFoundError(f"source directory not found: {src_dir}")
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {src_dir}")
    ext_set = _extension_set(extensions)
    iterator = root.rglob("*") if recursive else root.glob("*")
    files = [p for p in iterator if p.is_file() and _matches(p, ext_set)]
    files.sort()
    return files


def _output_path_for(
    source: str | Path, dest_dir: str | Path, *, suffix: str
) -> Path:
    """Compute the destination path for ``source`` under ``dest_dir``.

    ``/docs/a/b.pdf`` written into ``/out`` with ``suffix=".md"`` -> ``/out/b.md``.
    """
    src = Path(source)
    name = src.stem or src.name or "converted"
    return Path(dest_dir) / (name + suffix)


def _source_descriptor(source: Any) -> str:
    """Best-effort stable string for an arbitrary source (path/URI/stream)."""
    if isinstance(source, Path):
        return str(source)
    return str(source)


class MarkdownConverter:
    """Normalize heterogeneous files into Markdown via a pluggable backend.

    Parameters
    ----------
    backend:
        Any :class:`ConverterBackend` (e.g. :class:`MarkItDownBackend`,
        :class:`FakeConverterBackend`). Decouples conversion from any specific
        library so the core stays unit-testable. Defaults to a fresh
        :class:`MarkItDownBackend`.
    markitdown_kwargs:
        Convenience: if ``backend`` is omitted, forwarded to
        ``MarkItDownBackend(**markitdown_kwargs)`` (e.g. ``llm_client``,
        ``docintel_endpoint``, ``enable_plugins``).

    Examples
    --------
    >>> from sparksage import MarkdownConverter
    >>> conv = MarkdownConverter()                       # needs 'markitdown'
    >>> result = conv.convert("report.pdf")
    >>> result.markdown                                   # doctest: +SKIP

    Chain straight into block generation::

        blocks = IdeaBlockGenerator(client).generate(
            result.markdown, source=result.source_ref,
        )
    """

    def __init__(
        self,
        backend: ConverterBackend | None = None,
        **markitdown_kwargs: Any,
    ) -> None:
        self._backend: ConverterBackend = backend or MarkItDownBackend(
            **markitdown_kwargs
        )

    @property
    def backend(self) -> ConverterBackend:
        """The underlying :class:`ConverterBackend` (mainly for inspection)."""
        return self._backend

    def convert(self, source: Any) -> ConversionResult:
        """Convert a single file path / URI / stream to a :class:`ConversionResult`."""
        markdown, title = self._backend.convert(source)
        return ConversionResult(
            markdown=markdown,
            source=_source_descriptor(source),
            title=title,
        )

    def convert_to_markdown(self, source: Any) -> str:
        """Convenience: return just the Markdown text for ``source``."""
        return self.convert(source).markdown

    def convert_to_file(
        self,
        source: Any,
        dest_dir: str | Path,
        *,
        suffix: str = ".md",
    ) -> Path:
        """Convert ``source`` and write the Markdown to ``dest_dir/<name><suffix>``.

        Returns the path written. Parent directories are created as needed.
        """
        result = self.convert(source)
        return self._write(result, dest_dir, suffix=suffix)

    def convert_directory(
        self,
        src_dir: str | Path,
        dest_dir: str | Path | None = None,
        *,
        recursive: bool = True,
        extensions: Iterable[str] | None = DEFAULT_EXTENSIONS,
        suffix: str = ".md",
    ) -> list[ConversionResult]:
        """Batch-convert every matching file under ``src_dir``.

        Parameters
        ----------
        src_dir:
            Directory to scan for source files.
        dest_dir:
            If given, each result is also written to
            ``dest_dir/<stem><suffix>``. If omitted, results are only returned
            in memory.
        recursive:
            Recurse into subdirectories (default ``True``).
        extensions:
            Whitelist of extensions to convert. ``None`` means "all files".
            Defaults to :data:`DEFAULT_EXTENSIONS`.
        suffix:
            Output filename suffix when ``dest_dir`` is set.

        Returns
        -------
        list[ConversionResult]
            One result per converted file, in sorted path order. Files that fail
            to convert are logged and skipped (never abort the whole batch).
        """
        results: list[ConversionResult] = []
        for path in _iter_source_files(
            src_dir, recursive=recursive, extensions=extensions
        ):
            try:
                result = self.convert(path)
            except Exception as exc:  # noqa: BLE001 - batch must be resilient
                _logger.warning("skipping %s: %s", path, exc)
                continue
            if dest_dir is not None:
                self._write(result, dest_dir, source=path, suffix=suffix)
            results.append(result)
        return results

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _write(
        result: ConversionResult,
        dest_dir: str | Path,
        *,
        source: Any = None,
        suffix: str = ".md",
    ) -> Path:
        out_path = _output_path_for(
            source if source is not None else result.source, dest_dir, suffix=suffix
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.markdown, encoding="utf-8")
        return out_path
