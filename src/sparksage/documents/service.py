"""Framework-agnostic document-management orchestration: upload -> store.

:class:`DocumentService` is the knowledge-management counterpart to
:class:`~sparksage.api.SparkSageService`. It wires the reusable conversion +
cleaning stages together with the document parser, keyword extractor and store:

    uploaded bytes -> MarkdownConverter -> [TextCleaner] -> parse -> [auto-tag]
                                                                     -> DocumentStore

It is deliberately framework-agnostic (no FastAPI / HTTP imports here) so it is
fully unit-testable offline with :class:`FakeConverterBackend` and an
:class:`InMemoryDocumentStore`.

The only non-trivial concern it shares with :class:`SparkSageService` is
**temp-file management**: uploaded content arrives as raw ``bytes`` with an
original filename, while the converter backends detect format from the file
*extension*. The service writes the bytes to a short-lived temp file named with
the original extension, converts it, and swaps provenance back to the original
filename. (This helper is duplicated from ``api/pipeline`` on purpose: the
``documents`` package must not depend on the ``api`` layer -- ``api`` is the
outermost layer and depends on ``documents``, never the reverse.)
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from sparksage.clean.cleaner import TextCleaner
from sparksage.convert.converter import ConversionResult, MarkdownConverter
from sparksage.documents.keyword_extract import FrequencyKeywordExtractor, KeywordExtractor
from sparksage.documents.markdown_parser import parse_markdown
from sparksage.documents.schema import Document
from sparksage.documents.store import DocumentStore, InMemoryDocumentStore
from sparksage.schema.enums import TagSource
from sparksage.schema.source import SourceRef

_logger = logging.getLogger(__name__)

#: How many tags the auto-extractor should propose by default.
DEFAULT_AUTO_TAG_COUNT = 5


class DocumentServiceError(RuntimeError):
    """Base error for the document-management service."""


class DocumentNotFoundError(DocumentServiceError):
    """Raised when an operation targets a document id that does not exist."""


def _temp_suffix(filename: str | None) -> str:
    if not filename:
        return ""
    suffix = Path(filename).suffix
    return suffix if suffix else ""


@contextmanager
def _named_temp_file(data: bytes, filename: str | None) -> Iterator[Path]:
    suffix = _temp_suffix(filename)
    fd, name = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        yield Path(name)
    finally:
        try:
            os.unlink(name)
        except OSError:  # pragma: no cover - best-effort cleanup
            pass


class DocumentService:
    """Orchestrates convert -> clean -> parse -> tag -> store over uploads.

    Parameters
    ----------
    converter:
        The :class:`MarkdownConverter` used to turn uploaded bytes into Markdown.
        Always required -- conversion is the entry point.
    cleaner:
        The :class:`TextCleaner` applied between conversion and parsing. Defaults
        to a fresh cleaner with sensible normalization rules.
    extractor:
        The :class:`KeywordExtractor` used to auto-generate tags from content
        when the caller does not supply any. Defaults to
        :class:`FrequencyKeywordExtractor`.
    store:
        The :class:`DocumentStore` where records are persisted. Defaults to a
        fresh :class:`InMemoryDocumentStore`.

    Examples
    --------
    >>> from sparksage import (
    ...     FakeConverterBackend, MarkdownConverter, DocumentService,
    ... )
    >>> svc = DocumentService(converter=MarkdownConverter(backend=FakeConverterBackend()))
    >>> doc = svc.upload_text("# Title\\nSome body text.")   # doctest: +SKIP
    """

    def __init__(
        self,
        converter: MarkdownConverter,
        *,
        cleaner: TextCleaner | None = None,
        extractor: KeywordExtractor | None = None,
        store: DocumentStore | None = None,
    ) -> None:
        self._converter = converter
        self._cleaner = cleaner if cleaner is not None else TextCleaner()
        self._extractor = (
            extractor if extractor is not None else FrequencyKeywordExtractor()
        )
        self._store: DocumentStore = store if store is not None else InMemoryDocumentStore()

    @property
    def converter(self) -> MarkdownConverter:
        return self._converter

    @property
    def cleaner(self) -> TextCleaner:
        return self._cleaner

    @property
    def extractor(self) -> KeywordExtractor:
        return self._extractor

    @property
    def store(self) -> DocumentStore:
        return self._store

    # ------------------------------------------------------------------ #
    # create
    # ------------------------------------------------------------------ #
    def upload(
        self,
        data: bytes | str,
        filename: str | None = None,
        *,
        tags: list[str] | None = None,
        clean: bool = True,
        language: str = "en",
        auto_tag: bool = True,
        top_n: int = DEFAULT_AUTO_TAG_COUNT,
    ) -> Document:
        """Convert ``data`` to Markdown, parse it, tag it, and store it.

        Parameters
        ----------
        data, filename:
            Raw upload content and its original filename (used for
            extension-based format detection and provenance).
        tags:
            Caller-supplied tags. When empty/``None`` and ``auto_tag`` is true,
            tags are generated from content via :meth:`extract_tags`.
        clean:
            Run the text cleaner before parsing (recommended).
        language:
            BCP-47 code recorded on the document.
        auto_tag:
            When ``True`` (default) and no tags were supplied, auto-generate
            them. When ``False`` and no tags are supplied, the document is
            stored tag-less.
        top_n:
            Maximum number of auto-generated tags.
        """
        markdown, source_ref = self._to_markdown(data, filename, clean=clean)
        return self._ingest(
            markdown=markdown,
            source_ref=source_ref,
            tags=tags,
            language=language,
            auto_tag=auto_tag,
            top_n=top_n,
        )

    def upload_text(
        self,
        markdown: str,
        *,
        title: str | None = None,
        source_uri: str | None = None,
        tags: list[str] | None = None,
        language: str = "en",
        auto_tag: bool = True,
        top_n: int = DEFAULT_AUTO_TAG_COUNT,
    ) -> Document:
        """Ingest ``markdown`` text directly (no file conversion).

        Convenience for callers that already hold Markdown (e.g. an editor
        submit). ``title``/``source_uri`` override the auto-extracted values.
        """
        if markdown is None or not markdown.strip():
            raise ValueError("upload_text() requires non-empty markdown")

        parsed = parse_markdown(markdown)
        resolved_title = title or parsed.title
        source_ref: SourceRef | None = None
        if source_uri is not None:
            source_ref = SourceRef(uri=source_uri, title=resolved_title)

        tag_source: TagSource
        if tags:
            tag_source = TagSource.USER
            final_tags = self._normalize_tags(tags)
        elif auto_tag:
            final_tags = self._extractor.extract(parsed.content, top_n=top_n)
            tag_source = TagSource.AUTO
        else:
            final_tags = []
            tag_source = TagSource.USER

        document = Document(
            title=resolved_title,
            summary=parsed.summary,
            content=parsed.content,
            tags=final_tags,
            tag_source=tag_source,
            source=source_ref,
            language=language,
        )
        return self._store.save(document)

    # ------------------------------------------------------------------ #
    # read
    # ------------------------------------------------------------------ #
    def get(self, document_id: str | object) -> Document:
        """Return the document with ``document_id`` or raise."""
        doc = self._store.get(_to_uuid(document_id))
        if doc is None:
            raise DocumentNotFoundError(f"document not found: {document_id}")
        return doc

    def get_or_none(self, document_id: str | object) -> Document | None:
        """Like :meth:`get` but returns ``None`` when absent."""
        return self._store.get(_to_uuid(document_id))

    def list(
        self,
        *,
        tag: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Document]:
        """Return a page of documents, optionally filtered by ``tag``."""
        return self._store.list(tag=tag, limit=limit, offset=offset)

    def count(self, *, tag: str | None = None) -> int:
        return self._store.count(tag=tag)

    def tags(self) -> dict[str, int]:
        """Distinct tags across all documents mapped to document counts."""
        return self._store.tags()

    # ------------------------------------------------------------------ #
    # update
    # ------------------------------------------------------------------ #
    def update(
        self,
        document_id: str | object,
        *,
        title: str | None = None,
        summary: str | None = None,
        tags: list[str] | None = None,
    ) -> Document:
        """Patch a document's editable fields. Only provided fields change.

        Supplying ``tags`` replaces the tag set wholesale and marks it
        ``USER``-sourced (an explicit edit overrides auto-extraction).
        """
        doc = self.get(document_id)
        if title is not None:
            doc.title = title.strip() or None
        if summary is not None:
            doc.summary = summary.strip() or None
        if tags is not None:
            doc.replace_tags(self._normalize_tags(tags), source=TagSource.USER)
        else:
            doc.touch()
        return self._store.save(doc)

    def add_tags(self, document_id: str | object, tags: list[str]) -> Document:
        """Merge ``tags`` into a document's existing tag set."""
        if not tags:
            raise ValueError("add_tags() requires at least one tag")
        doc = self.get(document_id)
        doc.add_tags(self._normalize_tags(tags), source=TagSource.USER)
        return self._store.save(doc)

    def remove_tag(self, document_id: str | object, tag: str) -> Document:
        """Remove ``tag`` (case-insensitive) from a document."""
        doc = self.get(document_id)
        lowered = tag.strip().lower()
        if not lowered:
            raise ValueError("remove_tag() requires a non-empty tag")
        new_tags = [t for t in doc.tags if t.lower() != lowered]
        if len(new_tags) == len(doc.tags):
            raise DocumentNotFoundError(f"tag not present on document: {tag}")
        doc.replace_tags(new_tags, source=TagSource.USER)
        return self._store.save(doc)

    # ------------------------------------------------------------------ #
    # auto-tagging
    # ------------------------------------------------------------------ #
    def extract_tags(
        self,
        document_id: str | object,
        *,
        top_n: int = DEFAULT_AUTO_TAG_COUNT,
        replace_existing: bool = False,
    ) -> Document:
        """(Re)generate tags for a document from its content.

        When ``replace_existing`` is ``False`` (default), generated tags are
        merged onto the existing set and ``tag_source`` becomes ``MIXED`` if any
        user tags were already present. When ``True``, the tag set is replaced
        and ``tag_source`` is ``AUTO``.
        """
        doc = self.get(document_id)
        generated = self._extractor.extract(doc.content, top_n=top_n)
        if replace_existing:
            doc.replace_tags(generated, source=TagSource.AUTO)
        else:
            # The generated tags came from the extractor, so they are AUTO
            # regardless of what the document already held: merging AUTO tags
            # onto a USER-tagged document correctly yields MIXED.
            doc.add_tags(generated, source=TagSource.AUTO)
        return self._store.save(doc)

    # ------------------------------------------------------------------ #
    # delete
    # ------------------------------------------------------------------ #
    def delete(self, document_id: str | object) -> bool:
        """Delete a document. Returns ``True`` if something was removed."""
        return self._store.delete(_to_uuid(document_id))

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _to_markdown(
        self, data: bytes | str, filename: str | None, *, clean: bool
    ) -> tuple[str, SourceRef]:
        raw = data.encode("utf-8") if isinstance(data, str) else data
        result = self._to_conversion_result(raw, filename)
        if clean:
            cleaned = self._cleaner.clean_result(result)
            return cleaned.text, cleaned.source_ref
        return result.markdown, result.source_ref

    def _to_conversion_result(
        self, data: bytes, filename: str | None
    ) -> ConversionResult:
        with _named_temp_file(data, filename) as temp_path:
            result = self._converter.convert(temp_path)
        source = filename if filename else result.source
        return replace(result, source=source)

    def _ingest(
        self,
        *,
        markdown: str,
        source_ref: SourceRef,
        tags: list[str] | None,
        language: str,
        auto_tag: bool,
        top_n: int,
    ) -> Document:
        parsed = parse_markdown(markdown)
        title = parsed.title or source_ref.title
        source = source_ref.model_copy(update={"title": title}) if title else source_ref

        if tags:
            final_tags = self._normalize_tags(tags)
            tag_source = TagSource.USER
        elif auto_tag:
            final_tags = self._extractor.extract(parsed.content, top_n=top_n)
            tag_source = TagSource.AUTO
        else:
            final_tags = []
            tag_source = TagSource.USER

        document = Document(
            title=title,
            summary=parsed.summary,
            content=parsed.content,
            tags=final_tags,
            tag_source=tag_source,
            source=source,
            language=language,
        )
        return self._store.save(document)

    @staticmethod
    def _normalize_tags(tags: list[str]) -> list[str]:
        out: list[str] = []
        for tag in tags:
            tag = tag.strip()
            if tag and tag.lower() not in {t.lower() for t in out}:
                out.append(tag)
        return out


def _to_uuid(value: str | object) -> uuid.UUID:
    """Coerce ``value`` to a :class:`uuid.UUID`, passing :class:`UUID` through."""
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise DocumentNotFoundError(f"invalid document id: {value!r}") from exc
