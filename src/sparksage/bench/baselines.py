"""Traditional chunking baselines for the benchmark suite (pure stdlib).

The benchmark compares SparkSage's question-aligned IdeaBlocks against the
*traditional* approach: fixed-size text slices from a
``RecursiveCharacterTextSplitter`` (the LangChain default everyone reaches for).
Rather than depend on LangChain just to get a baseline, this module ships a
faithful, dependency-free reimplementation of that splitter so the benchmark is
self-contained and reproducible without any extra install.

The splitter recursively walks a separator list (paragraph -> sentence ->
whitespace -> character), only breaking on a separator when a piece would
otherwise exceed ``chunk_size``. Adjacent chunks overlap by ``chunk_overlap``
characters -- the standard recipe that keeps a little context across the cut.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Default separator hierarchy, matching the LangChain recursive splitter. The
#: splitter tries each in turn (longest/most-semantic first) and only falls back
#: to the next when a piece still exceeds ``chunk_size``.
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
)

#: The empty-string separator means "split on characters" -- the final fallback.
_CHAR_SEPARATOR_SENTINEL: str = ""


def _default_separators() -> tuple[str, ...]:
    return DEFAULT_SEPARATORS


@dataclass
class Chunk:
    """A single fixed-size text slice produced by a baseline splitter.

    Attributes
    ----------
    id:
        Stable opaque id (``source#offset``) -- used as the vector-store key.
    text:
        The chunk text. This is what gets embedded for the baseline index.
    source:
        Stable source descriptor (file path / URI / document id).
    source_ref_id:
        Opaque id of the *origin unit* this chunk was derived from. When the
        benchmark chunks the IdeaBlock corpus, this is the originating block's
        id -- the ground-truth link that lets retrieval be scored automatically.
    start, end:
        Half-open ``[start, end)`` character offsets of the chunk within the
        source text (for diagnostics / overlap analysis).
    """

    id: str
    text: str
    source: str = ""
    source_ref_id: str | None = None
    start: int = 0
    end: int = 0


class RecursiveCharSplitter:
    """Faithful, dependency-free ``RecursiveCharacterTextSplitter``.

    Parameters
    ----------
    chunk_size:
        Maximum characters per chunk (default ``400``, a common "small chunk"
        choice that roughly matches an IdeaBlock ``embedding_text`` in size, so
        the token-efficiency comparison is apples-to-apples).
    chunk_overlap:
        Character overlap between adjacent chunks (default ``50``).
    separators:
        Ordered separator hierarchy (default :data:`DEFAULT_SEPARATORS`).

    Notes
    -----
    The algorithm is the standard recursive split:

    1. Pick the first separator that appears in the text.
    2. Split on it; if any piece still exceeds ``chunk_size``, recurse on that
       piece with the next separator.
    3. Merge consecutive pieces back together while they still fit in
       ``chunk_size``, keeping ``chunk_overlap`` between merged groups.

    Empty input yields no chunks.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        separators: tuple[str, ...] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self._chunk_size = int(chunk_size)
        self._chunk_overlap = int(chunk_overlap)
        self._separators = separators if separators is not None else _default_separators()

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    @property
    def separators(self) -> tuple[str, ...]:
        return self._separators

    def split_text(self, text: str) -> list[str]:
        """Split ``text`` into overlapping chunk strings."""
        if not text:
            return []
        return self._split(text, self._separators)

    def split(
        self,
        text: str,
        *,
        source: str = "",
        source_ref_id: str | None = None,
        id_prefix: str = "",
    ) -> list[Chunk]:
        """Split ``text`` into :class:`Chunk`s with provenance + offsets.

        Offsets are computed relative to the de-duplicated concatenation of the
        chunk strings (which, with overlap, sums to more than ``len(text)``);
        ``start``/``end`` therefore describe where each chunk sits in the
        *stream*, which is what overlap analysis needs.
        """
        pieces = self.split_text(text)
        chunks: list[Chunk] = []
        offset = 0
        for i, piece in enumerate(pieces):
            prefix = id_prefix or source or "chunk"
            chunk_id = f"{prefix}#{i}"
            if source_ref_id is not None:
                chunk_id = f"{source_ref_id}#{i}"
            start = offset
            end = offset + len(piece)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    source=source,
                    source_ref_id=source_ref_id,
                    start=start,
                    end=end,
                )
            )
            offset = end - self._chunk_overlap if self._chunk_overlap else end
        return chunks

    def split_blocks(
        self,
        blocks: list[object],
        *,
        text_attr: str = "embedding_text",
    ) -> list[Chunk]:
        """Split each block's ``text_attr`` into chunks tagged with the block id.

        Convenience for the benchmark: chunk the IdeaBlock corpus's
        ``embedding_text`` so every resulting :class:`Chunk` carries
        ``source_ref_id`` = the originating block's id. That is the ground-truth
        link the scorer uses (a query derived from block X is "answered" by any
        chunk whose ``source_ref_id`` == X).
        """
        out: list[Chunk] = []
        for block in blocks:
            text = getattr(block, text_attr, None)
            if not isinstance(text, str) or not text:
                continue
            block_id = str(getattr(block, "id", id(block)))
            source = ""
            src = getattr(block, "source", None)
            if src is not None and getattr(src, "uri", None):
                source = str(src.uri)
            out.extend(self.split(text, source=source, source_ref_id=block_id))
        return out

    # ------------------------------------------------------------------ #
    # internals: the recursive split
    # ------------------------------------------------------------------ #
    def _split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self._chunk_size:
            return [text] if text else []

        separator = self._pick_separator(text, separators)
        next_separators = self._next_separators(separators, separator)

        if separator == _CHAR_SEPARATOR_SENTINEL:
            return self._char_chunks(text)

        pieces = text.split(separator) if separator else [text]
        good: list[str] = []
        merged: list[str] = []

        for piece in pieces:
            if not piece:
                continue
            if len(piece) > self._chunk_size:
                if next_separators:
                    refined = self._split(piece, next_separators)
                else:
                    refined = self._char_chunks(piece)
                good.extend(refined)
            else:
                good.append(piece)

        for piece in good:
            if merged and len(merged[-1]) + len(separator) + len(piece) <= self._chunk_size:
                merged[-1] = merged[-1] + separator + piece
            elif merged and self._chunk_overlap:
                tail = merged[-1][-self._chunk_overlap :]
                if tail and len(tail) + len(separator) + len(piece) <= self._chunk_size:
                    candidate = tail + separator + piece
                    if len(candidate) <= self._chunk_size:
                        merged.append(candidate)
                        continue
                merged.append(piece)
            else:
                merged.append(piece)

        return [m for m in merged if m]

    def _pick_separator(self, text: str, separators: tuple[str, ...]) -> str:
        for sep in separators:
            if sep == _CHAR_SEPARATOR_SENTINEL:
                return sep
            if sep and sep in text:
                return sep
        return separators[-1] if separators else _CHAR_SEPARATOR_SENTINEL

    def _next_separators(self, separators: tuple[str, ...], used: str) -> tuple[str, ...]:
        if used not in separators:
            return separators
        idx = separators.index(used)
        return separators[idx + 1 :]

    def _char_chunks(self, text: str) -> list[str]:
        out: list[str] = []
        step = max(1, self._chunk_size - self._chunk_overlap)
        i = 0
        n = len(text)
        while i < n:
            piece = text[i : i + self._chunk_size]
            if piece:
                out.append(piece)
            if i + self._chunk_size >= n:
                break
            i += step
        return out


__all__ = [
    "Chunk",
    "DEFAULT_SEPARATORS",
    "RecursiveCharSplitter",
]
