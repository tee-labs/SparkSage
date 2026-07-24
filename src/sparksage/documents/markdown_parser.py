"""Best-effort extraction of a Markdown document's title and summary.

The knowledge-management ingest flow needs to populate :class:`Document.title`
and :class:`Document.summary` from uploaded Markdown without an LLM. This module
implements deterministic, dependency-free heuristics:

* **title** -- the first ``#`` (H1) heading; falls back to the first non-empty,
  non-frontmatter line; ``None`` when nothing plausible is found.
* **summary** -- the first paragraph that is not a heading, list item, code
  fence, table, HTML comment or frontmatter, collapsed to one line and
  truncated to a soft cap with an ellipsis.

These are heuristics, not a full Markdown AST: they are fast, predictable and
need no third-party parser, matching the project's pure-stdlib core convention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sparksage.documents.schema import SUMMARY_MAX

#: A Markdown ATX heading line (``#`` through ``######``), optionally indented.
#: Used to *skip* headings while summarizing (any level is non-prose).
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<text>\S.*?)\s*$")

#: An H1-only line (exactly one ``#``). The document title convention.
_H1_RE = re.compile(r"^\s{0,3}#\s+(?P<text>\S.*?)\s*$")

#: A YAML front-matter fence (``---``) at the very start of the document.
_FRONTMATTER_FENCE_RE = re.compile(r"^\s{0,3}---\s*$")

#: A fenced code block delimiter (``` or ~~~).
_CODE_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

#: Lines we never want to summarize from.
_SKIP_PREFIXES = ("<!--", "|", "<")

#: Characters removed entirely when collapsing a summary to one line.
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ParsedDocument:
    """The deterministic parse of a Markdown document.

    Attributes
    ----------
    title:
        Extracted title, or ``None``.
    summary:
        Extracted summary (already truncated), or ``None``.
    content:
        The Markdown body (the input stripped of leading whitespace).
    """

    title: str | None
    summary: str | None
    content: str


def _iter_content_lines(markdown: str):
    """Yield (index, raw_line) skipping YAML front-matter and fenced code.

    Shared by :func:`extract_title` and :func:`extract_summary` so both respect
    the same front-matter / code-fence boundaries.
    """
    in_frontmatter = False
    in_code = False
    fence_marker: str | None = None
    for i, raw in enumerate(markdown.splitlines()):
        if i == 0 and _FRONTMATTER_FENCE_RE.match(raw):
            in_frontmatter = True
            continue
        if in_frontmatter:
            if _FRONTMATTER_FENCE_RE.match(raw):
                in_frontmatter = False
            continue

        fence = _CODE_FENCE_RE.match(raw)
        if fence:
            marker = fence.group(1)[0]
            if in_code and fence_marker == marker:
                in_code = False
                fence_marker = None
            elif not in_code:
                in_code = True
                fence_marker = marker
            continue
        if in_code:
            continue
        yield i, raw


def extract_title(markdown: str) -> str | None:
    """Return the first H1 heading text, or ``None``.

    Only a single-``#`` H1 counts as the title (the Markdown convention). When
    no H1 exists anywhere, falls back to the first non-empty, non-skippable
    content line. Returns ``None`` for a document with no usable content.
    """
    fallback: str | None = None
    for _i, raw in _iter_content_lines(markdown):
        m = _H1_RE.match(raw)
        if m:
            return m.group("text").strip()
        line = raw.strip()
        if not line or line.startswith(_SKIP_PREFIXES):
            continue
        if _HEADING_RE.match(raw):
            continue
        if fallback is None:
            fallback = line
    return fallback


def extract_summary(markdown: str, *, max_chars: int = SUMMARY_MAX) -> str | None:
    """Return the first prose paragraph, collapsed and truncated.

    Skips headings, list items, code fences, tables, HTML comments and
    frontmatter. Returns ``None`` when the document has no usable prose.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    for _i, raw in _iter_content_lines(markdown):
        line = raw.strip()
        if not line:
            continue
        if _HEADING_RE.match(raw):
            continue
        if line.startswith(_SKIP_PREFIXES):
            continue
        if line.startswith(("-", "*", "+")) and len(line) > 1 and line[1] in " \t":
            continue
        if re.match(r"^\d+\.\s", line):
            continue
        if line.startswith(">"):
            line = line.lstrip(">").strip()
            if not line:
                continue

        return _truncate(line, max_chars)

    return None


def _truncate(text: str, max_chars: int) -> str:
    """Collapse ``text`` to one line and truncate to ``max_chars`` with an ellipsis.

    A paragraph that fits is returned verbatim (terminal punctuation kept);
    only an over-long paragraph is cut at a word boundary and terminated with
    an ellipsis so it is never broken mid-word.
    """
    collapsed = _WHITESPACE_RE.sub(" ", text).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    cut = collapsed[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def parse_markdown(markdown: str, *, summary_max: int = SUMMARY_MAX) -> ParsedDocument:
    """Parse ``markdown`` into a :class:`ParsedDocument`.

    The ``content`` is the stripped input; ``title`` / ``summary`` come from
    :func:`extract_title` / :func:`extract_summary`.
    """
    content = markdown.strip()
    title = extract_title(markdown)
    summary = extract_summary(markdown, max_chars=summary_max)
    return ParsedDocument(title=title, summary=summary, content=content)
