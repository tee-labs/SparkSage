"""File-to-Markdown conversion backend abstraction.

The conversion core depends *only* on the :class:`ConverterBackend` protocol, so
it is fully unit-testable with the deterministic :class:`FakeConverterBackend`.
A concrete :class:`MarkItDownBackend` (backed by Microsoft's ``markitdown``
library) is provided for production use across dozens of file formats.

``markitdown`` is an *optional* dependency -- install it with
``pip install 'sparksage[convert]'``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ConverterBackend(Protocol):
    """Minimal file-to-Markdown interface the converter depends on.

    Any callable that turns a source (path, URI, stream, bytes) into a
    ``(markdown, title)`` pair implements this -- Microsoft ``markitdown`` in
    production, or a deterministic fake for tests.
    """

    def convert(self, source: Any, **kwargs: Any) -> tuple[str, str | None]:
        """Convert ``source`` and return ``(markdown, title)``.

        ``title`` is ``None`` when the backend cannot infer one.
        """
        ...


class MarkItDownBackend:
    """Converter backend backed by Microsoft ``markitdown``.

    Handles PDF, Word, PowerPoint, Excel, HTML, CSV/JSON/XML, images (EXIF +
    OCR), audio (transcription), EPub, ZIP archives and more -- all normalized
    to Markdown. See https://github.com/microsoft/markitdown for the full matrix.

    The ``markitdown`` package is an *optional* dependency -- install it with
    ``pip install 'sparksage[convert]'`` (or ``'markitdown[all]'`` for every
    format's binary deps).

    Parameters
    ----------
    markitdown_kwargs:
        Forwarded verbatim to ``MarkItDown(...)`` -- e.g. ``llm_client`` /
        ``llm_model`` for image description, ``docintel_endpoint`` /
        ``cu_endpoint`` for cloud extraction, ``enable_plugins=True`` for
        third-party converters.
    """

    def __init__(self, **markitdown_kwargs: Any) -> None:
        try:
            from markitdown import MarkItDown
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "MarkItDownBackend requires the 'markitdown' package. "
                "Install it with: pip install 'sparksage[convert]'"
            ) from exc
        self._markitdown = MarkItDown(**markitdown_kwargs)

    def convert(self, source: Any, **kwargs: Any) -> tuple[str, str | None]:
        result = self._markitdown.convert(source, **kwargs)
        return result.markdown, getattr(result, "title", None)


@dataclass
class FakeConverterBackend:
    """Deterministic, scriptable converter backend for tests and offline demos.

    By default returns the same ``markdown``/``title`` for every source. For
    directory-style tests, pass a ``by_source`` mapping keyed by the *string*
    form of the source (e.g. its path) to vary the output per file. Every source
    seen is recorded on ``calls`` so tests can assert on what was converted.
    """

    markdown: str = ""
    title: str | None = None
    by_source: dict[str, str] = field(default_factory=dict)
    calls: list[Any] = field(default_factory=list)

    def convert(self, source: Any, **kwargs: Any) -> tuple[str, str | None]:
        self.calls.append(source)
        text = self.by_source.get(str(source), self.markdown)
        return text, self.title
