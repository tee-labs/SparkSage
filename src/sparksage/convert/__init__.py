"""Uniform file-to-Markdown conversion (any source format -> Markdown).

Inject any :class:`ConverterBackend` (a real :class:`MarkItDownBackend` backed by
Microsoft ``markitdown`` in production, or :class:`FakeConverterBackend` in tests)
into a :class:`MarkdownConverter` and call :meth:`~MarkdownConverter.convert` /
:meth:`~MarkdownConverter.convert_directory`.

The emitted :class:`ConversionResult` chains straight into block generation:
feed ``result.markdown`` as the text and ``result.source_ref`` as provenance to
:class:`~sparksage.generator.IdeaBlockGenerator`.
"""

from sparksage.convert.backend import (
    ConverterBackend,
    FakeConverterBackend,
    MarkItDownBackend,
)
from sparksage.convert.converter import (
    DEFAULT_EXTENSIONS,
    ConversionResult,
    MarkdownConverter,
)

__all__ = [
    "DEFAULT_EXTENSIONS",
    "ConversionResult",
    "ConverterBackend",
    "FakeConverterBackend",
    "MarkdownConverter",
    "MarkItDownBackend",
]
