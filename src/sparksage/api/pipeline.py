"""Framework-agnostic orchestration service: bytes -> Markdown / IdeaBlocks.

:class:`SparkSageService` is the thin glue that wires the three existing
pipeline stages together for an HTTP (or any other) caller:

    uploaded bytes -> MarkdownConverter -> [TextCleaner] -> IdeaBlockGenerator

It is deliberately framework-agnostic (no FastAPI / HTTP imports here) so it is
fully unit-testable offline with :class:`FakeConverterBackend` /
:class:`FakeLLMClient`. The only non-trivial concern it owns is **temp-file
management**: uploaded content arrives as raw ``bytes`` with an original
filename, while the converter backends (``markitdown``) detect format from the
file *extension*. The service writes the bytes to a short-lived temp file named
with the original extension, converts it, and swaps provenance back to the
original filename before handing the result downstream.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from sparksage.clean.cleaner import TextCleaner
from sparksage.convert.converter import ConversionResult, MarkdownConverter
from sparksage.documents.backends.memory import InMemoryDocumentStore
from sparksage.documents.models import DocumentRecord
from sparksage.documents.store import DocumentStore
from sparksage.documents.summarizer import Summarizer, default_summarizer
from sparksage.generator.generator import (
    GenerationStats,
    IdeaBlockGenerator,
)
from sparksage.schema.ideablock import IdeaBlock
from sparksage.schema.source import SourceRef
from sparksage.tags.extractor import KeywordExtractor, default_extractor

_logger = logging.getLogger(__name__)


class ServiceError(RuntimeError):
    """Base error for the orchestration service."""


class ConversionNotConfiguredError(ServiceError):
    """Raised when a conversion-capable converter is not available."""


class GenerationNotConfiguredError(ServiceError):
    """Raised when generation is requested but no generator is wired."""


@dataclass
class ConvertOutput:
    """Framework-agnostic result of a *convert* request."""

    markdown: str
    title: str | None
    source: SourceRef
    cleaned: bool


@dataclass
class GenerateOutput:
    """Framework-agnostic result of a *generate* request."""

    blocks: list[IdeaBlock]
    title: str | None
    source: SourceRef
    cleaned: bool
    stats: GenerationStats | None = None


def _temp_suffix(filename: str | None) -> str:
    """Return a dot-prefixed extension for the temp file, derived from ``filename``."""
    if not filename:
        return ""
    suffix = Path(filename).suffix
    return suffix if suffix else ""


@contextmanager
def _named_temp_file(data: bytes, filename: str | None) -> Iterator[Path]:
    """Write ``data`` to a temp file carrying ``filename``'s extension.

    The converter backends (``markitdown``) select the per-format handler from
    the file extension, so the temp file must keep the original extension.
    """
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


def _block_to_dict(block: IdeaBlock) -> dict[str, Any]:
    """Serialize an :class:`IdeaBlock` to a JSON-safe flat dict."""
    return block.model_dump(mode="json")


class SparkSageService:
    """Orchestrates convert -> clean -> generate over uploaded file bytes.

    Parameters
    ----------
    converter:
        The :class:`MarkdownConverter` used to turn bytes into Markdown. Always
        required -- conversion is the entry point of both operations.
    cleaner:
        The :class:`TextCleaner` applied between conversion and generation when
        ``clean=True``. Defaults to a fresh cleaner with sensible normalization
        rules.
    generator:
        The :class:`IdeaBlockGenerator` used for the generate operation. When
        ``None``, :meth:`generate` raises :class:`GenerationNotConfiguredError`
        so callers (e.g. the HTTP layer) can surface a clear ``503``.
    document_store:
        The :class:`~sparksage.documents.store.DocumentStore` used by the
        document-management operations (:meth:`ingest_document` /
        :meth:`list_documents` / ...). When ``None``, an
        :class:`~sparksage.documents.backends.memory.InMemoryDocumentStore` is
        created lazily on first use -- so document management works out of the
        box with no extra install. Pass a
        :class:`~sparksage.documents.backends.sqlite.SqliteDocumentStore` (or a
        future backend) for durable storage.
    keyword_extractor:
        The :class:`~sparksage.tags.extractor.KeywordExtractor` used to
        auto-tag documents that arrive without tags. Defaults lazily to a
        :class:`~sparksage.tags.extractor.RakeKeywordExtractor`.
    summarizer:
        The :class:`~sparksage.documents.summarizer.Summarizer` used to produce
        document-level summaries. Defaults lazily to an
        :class:`~sparksage.documents.summarizer.ExtractiveSummarizer`.

    Examples
    --------
    >>> from sparksage import (
    ...     FakeConverterBackend, FakeLLMClient, MarkdownConverter,
    ...     IdeaBlockGenerator, SparkSageService,
    ... )
    >>> fake_backend = FakeConverterBackend(markdown="# Hi\\nSome text.")
    >>> fake_llm = FakeLLMClient(responses=['{"blocks": []}'])
    >>> svc = SparkSageService(
    ...     converter=MarkdownConverter(backend=fake_backend),
    ...     generator=IdeaBlockGenerator(fake_llm),
    ... )
    >>> out = svc.convert(b"# Hi", "note.md")     # doctest: +SKIP
    """

    def __init__(
        self,
        converter: MarkdownConverter,
        cleaner: TextCleaner | None = None,
        generator: IdeaBlockGenerator | None = None,
        *,
        document_store: DocumentStore | None = None,
        keyword_extractor: KeywordExtractor | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        self._converter = converter
        self._cleaner = cleaner if cleaner is not None else TextCleaner()
        self._generator = generator
        self._document_store: DocumentStore | None = document_store
        self._keyword_extractor: KeywordExtractor | None = keyword_extractor
        self._summarizer: Summarizer | None = summarizer

    @property
    def converter(self) -> MarkdownConverter:
        return self._converter

    @property
    def cleaner(self) -> TextCleaner:
        return self._cleaner

    @property
    def generator(self) -> IdeaBlockGenerator | None:
        return self._generator

    @property
    def has_generator(self) -> bool:
        return self._generator is not None

    @property
    def document_store(self) -> DocumentStore:
        """The :class:`DocumentStore`, lazily defaulting to an in-memory one.

        Document management works with no explicit wiring: the first access
        materializes an :class:`~sparksage.documents.backends.memory.InMemoryDocumentStore`
        so :meth:`ingest_document` / :meth:`list_documents` / ... are always
        available. Pass a durable store in the constructor to override.
        """
        if self._document_store is None:
            self._document_store = InMemoryDocumentStore()
        return self._document_store

    @property
    def has_document_store(self) -> bool:
        return self._document_store is not None

    @property
    def keyword_extractor(self) -> KeywordExtractor:
        """The :class:`KeywordExtractor`, lazily defaulting to RAKE."""
        if self._keyword_extractor is None:
            self._keyword_extractor = default_extractor()
        return self._keyword_extractor

    @property
    def summarizer(self) -> Summarizer:
        """The :class:`Summarizer`, lazily defaulting to the extractive one."""
        if self._summarizer is None:
            self._summarizer = default_summarizer()
        return self._summarizer

    # ------------------------------------------------------------------ #
    # convert: bytes -> Markdown (+ optional cleaning)
    # ------------------------------------------------------------------ #
    def convert(
        self,
        data: bytes | str,
        filename: str | None = None,
        *,
        clean: bool = False,
    ) -> ConvertOutput:
        """Convert uploaded ``data`` to Markdown, optionally cleaning it.

        Parameters
        ----------
        data:
            Raw file content. ``str`` is accepted (encoded as UTF-8) for
            convenience but ``bytes`` is the expected HTTP-upload form.
        filename:
            Original filename -- used for extension-based format detection and
            as the provenance URI / cleaning-rules routing key.
        clean:
            When ``True``, run the result through :class:`TextCleaner` before
            returning.
        """
        raw = data.encode("utf-8") if isinstance(data, str) else data
        result = self._to_conversion_result(raw, filename)

        if clean:
            cleaned = self._cleaner.clean_result(result)
            return ConvertOutput(
                markdown=cleaned.text,
                title=cleaned.title,
                source=cleaned.source_ref,
                cleaned=True,
            )
        return ConvertOutput(
            markdown=result.markdown,
            title=result.title,
            source=result.source_ref,
            cleaned=False,
        )

    # ------------------------------------------------------------------ #
    # generate: bytes -> IdeaBlock list
    # ------------------------------------------------------------------ #
    def generate(
        self,
        data: bytes | str,
        filename: str | None = None,
        *,
        clean: bool = True,
        max_blocks: int | None = None,
        language: str | None = None,
        with_stats: bool = False,
    ) -> GenerateOutput:
        """Convert uploaded ``data`` to Markdown, then generate IdeaBlocks.

        Parameters
        ----------
        data, filename, clean:
            See :meth:`convert`. ``clean`` defaults to ``True`` here because raw
            converted text is rarely generation-ready.
        max_blocks, language:
            Forwarded to :meth:`IdeaBlockGenerator.generate`.
        with_stats:
            When ``True``, also runs :meth:`IdeaBlockGenerator.generate_with_stats`
            and attaches :class:`GenerationStats` to the output.

        Raises
        ------
        GenerationNotConfiguredError:
            If no generator was wired into the service.
        GenerationError:
            If the LLM pipeline fails outright.
        """
        if not self.has_generator:
            raise GenerationNotConfiguredError(
                "no IdeaBlockGenerator configured; cannot generate blocks."
            )

        raw = data.encode("utf-8") if isinstance(data, str) else data
        result = self._to_conversion_result(raw, filename)

        if clean:
            cleaned = self._cleaner.clean_result(result)
            text = cleaned.text
            source_ref = cleaned.source_ref
            title = cleaned.title
        else:
            text = result.markdown
            source_ref = result.source_ref
            title = result.title

        gen = self._generator
        assert gen is not None  # narrowed by has_generator

        if with_stats:
            blocks, stats = gen.generate_with_stats(
                text,
                source=source_ref,
                max_blocks=max_blocks,
                language=language,
            )
        else:
            blocks = gen.generate(
                text,
                source=source_ref,
                max_blocks=max_blocks,
                language=language,
            )
            stats = None

        return GenerateOutput(
            blocks=blocks,
            title=title,
            source=source_ref,
            cleaned=clean,
            stats=stats,
        )

    # ------------------------------------------------------------------ #
    # document management: ingest / list / get / update / delete / retag
    # ------------------------------------------------------------------ #
    def auto_tag(
        self, text: str, *, top_k: int = 8, max_tag_words: int = 3
    ) -> list[str]:
        """Extract ``top_k`` free-form tags from ``text`` via the keyword extractor.

        Returns the keyword strings (best first), dropping duplicates. Over-long
        phrases (more than ``max_tag_words`` space-separated tokens) are
        decomposed into their constituent single words so the resulting values
        stay tag-shaped (e.g. ``"machine learning models ranking"`` ->
        ``machine``, ``learning``, ``models``, ``ranking``). This is the no-LLM
        auto-tagging path mandated by the tag-management requirement.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if max_tag_words < 1:
            raise ValueError("max_tag_words must be >= 1")
        scores = self.keyword_extractor.extract(text, top_k=max(top_k * 2, top_k))
        tags: list[str] = []
        seen: set[str] = set()
        for ks in scores:
            words = [w for w in ks.keyword.split() if w]
            candidates: list[str]
            if len(words) <= max_tag_words:
                candidates = [ks.keyword.strip()]
            else:
                candidates = [w for w in words]
            for tag in candidates:
                t = tag.strip()
                if not t or t in seen:
                    continue
                seen.add(t)
                tags.append(t)
                if len(tags) >= top_k:
                    return tags
        return tags

    def summarize_text(self, text: str, *, max_sentences: int = 3) -> str:
        """Produce a document-level extractive summary of ``text``."""
        return self.summarizer.summarize(text, max_sentences=max_sentences)

    def ingest_document(
        self,
        data: bytes | str,
        filename: str | None = None,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
        auto_tag: bool = True,
        clean: bool = True,
        summarize: bool = True,
        max_summary_sentences: int = 3,
        top_k: int = 8,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentRecord:
        """Convert -> clean -> (auto-tag) -> (summarize) -> store a document.

        The end-to-end ingest of the document-management service. When ``tags``
        is empty and ``auto_tag`` is ``True`` (the default), tags are derived
        from the content via :meth:`auto_tag` (keyword-extraction algorithm, no
        LLM). When ``summarize`` is ``True`` and no ``summary`` is supplied, a
        document-level summary is produced by the configured
        :class:`Summarizer`.

        Parameters
        ----------
        data, filename, clean:
            See :meth:`convert`. ``clean`` defaults to ``True`` here because raw
            converted text is rarely tag-extraction-ready.
        title:
            Explicit title override. Falls back to the backend-extracted title.
        tags:
            Caller-supplied tags. When non-empty they win; otherwise
            ``auto_tag`` fills them.
        auto_tag:
            When ``True`` (default) and ``tags`` is empty, derive tags from the
            content.
        summarize:
            When ``True`` (default), produce a summary unless one is supplied
            via ``metadata['summary']`` (callers rarely need to override).
        top_k:
            Number of tags to extract when auto-tagging.
        doc_id:
            Optional explicit ``doc_id`` (otherwise a fresh UUID is generated).
        metadata:
            Free-form caller metadata stored on the record.

        Returns the stored :class:`DocumentRecord`.
        """
        conv = self.convert(data, filename, clean=clean)
        text = conv.markdown
        resolved_title = title if title is not None else conv.title

        final_tags = list(tags) if tags else []
        if not final_tags and auto_tag:
            final_tags = self.auto_tag(text, top_k=top_k)

        summary: str | None = None
        if summarize:
            summary = self.summarize_text(text, max_sentences=max_summary_sentences)

        record_kwargs: dict[str, Any] = {
            "title": resolved_title,
            "summary": summary,
            "body_markdown": text,
            "tags": final_tags,
            "source": SourceRef(uri=conv.source.uri, title=resolved_title),
            "metadata": dict(metadata) if metadata else {},
        }
        if doc_id is not None:
            record_kwargs["doc_id"] = doc_id
        record = DocumentRecord(**record_kwargs)
        return self.document_store.save(record)

    def list_documents(
        self,
        *,
        tag: str | None = None,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentRecord]:
        """List documents, optionally filtered by tag and/or a title/body query."""
        return self.document_store.list(tag=tag, q=q, limit=limit, offset=offset)

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        """Return the document for ``doc_id`` (or ``None`` if absent)."""
        return self.document_store.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete ``doc_id``. Return whether a record was removed."""
        return self.document_store.delete(doc_id)

    def count_documents(self, *, tag: str | None = None) -> int:
        """Number of stored documents, optionally restricted to a tag."""
        return self.document_store.count(tag=tag)

    def list_document_tags(self) -> list[str]:
        """Return the distinct tag vocabulary across all stored documents."""
        return self.document_store.list_tags()

    def update_document(
        self,
        doc_id: str,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentRecord:
        """Partially update a stored document and return the refreshed record.

        Only the fields supplied (non-``None``) are changed; ``updated_at`` is
        refreshed. ``tags`` replaces the full tag list when supplied.

        Raises :class:`KeyError` when ``doc_id`` is unknown.
        """
        existing = self.document_store.get(doc_id)
        if existing is None:
            raise KeyError(f"document not found: {doc_id}")
        changes: dict[str, Any] = {"updated_at": existing.updated_at}
        if title is not None:
            changes["title"] = title
        if tags is not None:
            changes["tags"] = list(tags)
        if summary is not None:
            changes["summary"] = summary
        if metadata is not None:
            changes["metadata"] = dict(metadata)
        updated = existing.model_copy(update=changes)
        return self.document_store.save(updated)

    def retag_document(
        self,
        doc_id: str,
        *,
        top_k: int = 8,
        replace: bool = True,
        extra_tags: list[str] | None = None,
    ) -> DocumentRecord:
        """Re-extract tags for ``doc_id`` from its body and return the record.

        Parameters
        ----------
        top_k:
            Number of tags to extract.
        replace:
            When ``True`` (default), the extracted tags replace the existing
            ones. When ``False``, they are appended (de-duplicated).
        extra_tags:
            Additional tags to merge in (always added, de-duplicated).

        Raises :class:`KeyError` when ``doc_id`` is unknown.
        """
        existing = self.document_store.get(doc_id)
        if existing is None:
            raise KeyError(f"document not found: {doc_id}")
        extracted = self.auto_tag(existing.body_markdown, top_k=top_k)
        merged: list[str] = []
        seen: set[str] = set()
        source_lists: list[list[str]] = []
        if replace:
            source_lists.append(extracted)
        else:
            source_lists.append(list(existing.tags))
            source_lists.append(extracted)
        if extra_tags:
            source_lists.append(list(extra_tags))
        for lst in source_lists:
            for tag in lst:
                t = str(tag).strip()
                if not t or t in seen:
                    continue
                seen.add(t)
                merged.append(t)
        return self.document_store.save(
            existing.model_copy(update={"tags": merged})
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _to_conversion_result(
        self, data: bytes, filename: str | None
    ) -> ConversionResult:
        """Convert raw bytes to a :class:`ConversionResult` with correct provenance.

        The temp file carries ``filename``'s extension so the backend picks the
        right per-format handler, while provenance (``source``) is set to the
        *original* filename -- not the throwaway temp path -- so cleaning rules
        routed on ``*.pdf`` / ``*.docx`` and IdeaBlock ``source.uri`` stay
        meaningful.
        """
        with _named_temp_file(data, filename) as temp_path:
            result = self._converter.convert(temp_path)
        source = filename if filename else result.source
        return replace(result, source=source)
