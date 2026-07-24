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
from sparksage.generator.generator import (
    GenerationStats,
    IdeaBlockGenerator,
)
from sparksage.schema.ideablock import IdeaBlock
from sparksage.schema.source import SourceRef

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
    ) -> None:
        self._converter = converter
        self._cleaner = cleaner if cleaner is not None else TextCleaner()
        self._generator = generator

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
