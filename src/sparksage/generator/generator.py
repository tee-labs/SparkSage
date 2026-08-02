"""LLM-driven generation of IdeaBlocks from free text.

:class:`IdeaBlockGenerator` is the core "text -> many IdeaBlocks" feature. It:

1. Builds a prompt that teaches the model the IdeaBlock schema and the live
   controlled vocabularies (see :mod:`sparksage.generator.prompts`);
2. Calls an :class:`LLMClient` (pluggable -- inject a fake for tests);
3. Extracts JSON from the (possibly noisy) model response;
4. Coerces each raw block into a strict :class:`IdeaBlock`, mapping vocabularies
   and dropping or failing on bad data according to ``strict``.

The generation core never imports a concrete LLM SDK, so it is deterministic and
unit-testable without network access.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

from sparksage.generator.client import JSON_RESPONSE_FORMAT, LLMClient
from sparksage.generator.prompts import build_messages
from sparksage.generator.schema import (
    CoercionError,
    RawGenerationResult,
    coerce_block,
    parse_raw_result,
)
from sparksage.schema.ideablock import IdeaBlock
from sparksage.schema.source import SourceRef

_logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$", re.MULTILINE)

#: Default per-segment character budget for large inputs. Source text longer
#: than this is split into coherent segments (see :func:`_split_text_segments`)
#: and each segment is sent as its own generation request. This keeps the LLM's
#: JSON output small enough that it never gets truncated mid-stream by the
#: provider's output-token cap -- the cause of the ``Unprocessable Entity``
#: ingest failures on large uploads. Tune via ``max_input_chars=``.
DEFAULT_MAX_INPUT_CHARS = 8000

#: Separator hierarchy for :func:`_split_text_segments` (most semantic first).
#: The empty string is the final "split on characters" fallback so the size
#: bound always holds.
_SEGMENT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


class GenerationError(RuntimeError):
    """Base error for the generation pipeline."""


class EmptyResponseError(GenerationError):
    """The LLM returned no content."""


class ResponseParseError(GenerationError):
    """The model response could not be parsed as the expected JSON."""


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a possibly-noisy model response.

    Handles three common cases: plain JSON, JSON wrapped in ```json fences, and
    JSON embedded in surrounding prose (extracted via balanced brace matching).
    """
    cleaned = text.strip()
    if not cleaned:
        raise ResponseParseError("empty model response")

    cleaned = _FENCE_RE.sub("", cleaned).strip()

    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return cleaned


def _split_text_segments(text: str, max_chars: int) -> list[str]:
    """Split *text* into coherent segments each no longer than *max_chars*.

    Source text that would force a single huge LLM response (whose JSON output
    then gets truncated, failing parse) is broken into bounded segments that
    each yield a manageable number of blocks. Splits prefer semantic
    boundaries (blank lines, line breaks, sentence ends, spaces) and falls back
    to a hard character cut so the size bound always holds. Empty input yields
    no segments; trivially-small input yields a single segment unchanged.
    """
    if max_chars <= 0:
        return [text] if text else []
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= max_chars:
        return [stripped]
    pieces = _recursive_split(stripped, _SEGMENT_SEPARATORS, max_chars)
    merged: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if merged and len(merged[-1]) + 2 + len(piece) <= max_chars:
            merged[-1] = merged[-1] + "\n\n" + piece
        else:
            merged.append(piece)
    return merged


def _recursive_split(
    text: str, separators: tuple[str, ...], max_chars: int
) -> list[str]:
    """Recursively split *text* on the separator hierarchy until each piece fits."""
    if len(text) <= max_chars:
        return [text]
    sep, rest = _pick_separator(text, separators)
    if sep == "":
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    out: list[str] = []
    for part in text.split(sep):
        if not part:
            continue
        if len(part) > max_chars:
            out.extend(_recursive_split(part, rest, max_chars))
        else:
            out.append(part)
    return out


def _pick_separator(
    text: str, separators: tuple[str, ...]
) -> tuple[str, tuple[str, ...]]:
    """Return ``(separator, remaining_separators)`` for the first match in *text*."""
    for i, sep in enumerate(separators):
        if sep == "":
            return sep, separators[i + 1 :]
        if sep in text:
            return sep, separators[i + 1 :]
    return separators[-1], ()


@dataclass
class GenerationStats:
    """Diagnostic counters returned alongside generated blocks."""

    raw_block_count: int = 0
    emitted: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


class IdeaBlockGenerator:
    """Turn free text into a list of :class:`IdeaBlock` via an LLM.

    Parameters
    ----------
    client:
        Any :class:`LLMClient` (e.g. :class:`OpenAICompatibleClient`,
        :class:`FakeLLMClient`). Decouples generation from any specific SDK.
    model:
        Model name forwarded to the client (ignored by fakes).
    temperature:
        Sampling temperature. Low values give more faithful extraction.
    language:
        BCP-47 code written into every block (``IdeaBlock.language``).
    strict:
        If ``True``, the first malformed/invalid block aborts generation with a
        :class:`GenerationError`. If ``False`` (default), invalid blocks are
        skipped and recorded in :class:`GenerationStats.errors`.
    use_json_mode:
        Request JSON-mode structured output from the provider when supported.
    max_input_chars:
        Per-segment character budget for large inputs. Source text longer than
        this is split into coherent segments and each segment is sent as its own
        generation request, keeping the model's JSON output small enough that it
        is never truncated by the provider's output-token cap. ``0`` or negative
        disables splitting (one request regardless of size -- the legacy
        behaviour, prone to truncation on large uploads).
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        language: str = "en",
        strict: bool = False,
        use_json_mode: bool = True,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._language = language
        self._strict = strict
        self._use_json_mode = use_json_mode
        self._max_input_chars = int(max_input_chars)

    def generate(
        self,
        text: str,
        **kwargs: object,
    ) -> list[IdeaBlock]:
        """Generate IdeaBlocks from ``text``.

        A :class:`SourceRef` provenance is built from ``source`` (if given) or
        from the ``source_uri``/``source_title`` shortcut and attached to every
        emitted block. Returns the list of valid blocks; in non-strict mode this
        may be shorter than what the model produced. Large inputs are split
        into segments (see :func:`_split_text_segments`) so each generation
        request stays small enough to parse cleanly.
        """
        blocks, _stats = self.generate_with_stats(text, **kwargs)
        return blocks

    def generate_with_stats(
        self,
        text: str,
        **kwargs: object,
    ) -> tuple[list[IdeaBlock], GenerationStats]:
        """Like :meth:`generate` but also returns :class:`GenerationStats`."""
        if text is None or not str(text).strip():
            raise ValueError("generate() requires non-empty text")

        source = kwargs.pop("source", None)  # type: ignore[arg-type]
        source_uri = kwargs.pop("source_uri", None)  # type: ignore[arg-type]
        source_title = kwargs.pop("source_title", None)  # type: ignore[arg-type]
        if source is None and source_uri is not None:
            source = SourceRef(uri=source_uri, title=source_title)
        max_blocks = kwargs.pop("max_blocks", None)  # type: ignore[arg-type]
        language = kwargs.pop("language", None)  # type: ignore[arg-type]
        lang = language or self._language

        segments = _split_text_segments(str(text), self._max_input_chars)
        if not segments:
            raise ValueError("generate() requires non-empty text")

        # For a single segment, ``max_blocks`` is forwarded to the prompt as
        # before. When segmenting, each segment produces its natural blocks and
        # the global ``max_blocks`` is applied as a final cap -- so a huge doc
        # is still covered instead of stopping at the first segment.
        single = len(segments) == 1
        per_segment_max = max_blocks if single else None

        _logger.debug(
            "generating blocks: text_len=%d segments=%d max_blocks=%s lang=%s "
            "json_mode=%s",
            len(str(text)),
            len(segments),
            max_blocks,
            lang,
            self._use_json_mode,
        )

        all_blocks: list[IdeaBlock] = []
        stats = GenerationStats()
        for i, segment in enumerate(segments):
            try:
                seg_blocks, seg_stats = self._generate_one_segment(
                    segment,
                    source=source,
                    language=lang,
                    max_blocks=per_segment_max,
                )
            except (ResponseParseError, EmptyResponseError, GenerationError) as exc:
                # A single-segment run (or strict mode) surfaces the error to
                # preserve the generation contract. A multi-segment run in
                # non-strict mode is resilient: one bad segment is logged and
                # recorded rather than wasting the whole large-doc batch.
                if single or self._strict:
                    raise
                _logger.warning(
                    "segment %d/%d failed, skipping: %s",
                    i + 1,
                    len(segments),
                    exc,
                )
                stats.errors.append(f"segment {i + 1}/{len(segments)}: {exc}")
                continue
            all_blocks.extend(seg_blocks)
            stats.raw_block_count += seg_stats.raw_block_count
            stats.emitted += seg_stats.emitted
            stats.skipped += seg_stats.skipped
            stats.errors.extend(seg_stats.errors)

        if max_blocks is not None and len(all_blocks) > max_blocks:
            all_blocks = all_blocks[:max_blocks]

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "generated blocks detail: names=%s",
                [b.name[:40] for b in all_blocks],
            )
        _logger.info(
            "generated %d blocks (raw=%d, segments=%d)",
            len(all_blocks),
            stats.raw_block_count,
            len(segments),
        )
        return all_blocks, stats

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _generate_one_segment(
        self,
        text: str,
        *,
        source: SourceRef | None,
        language: str,
        max_blocks: int | None,
    ) -> tuple[list[IdeaBlock], GenerationStats]:
        """Run one LLM generation request for a single text segment."""
        messages = build_messages(
            text, source=source, max_blocks=max_blocks, language=language
        )
        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
        _logger.debug(
            "generating segment: text_len=%d max_blocks=%s prompt_chars=%d",
            len(text),
            max_blocks,
            prompt_chars,
        )
        t0 = time.perf_counter()
        response_text = self._client.complete(
            messages,
            model=self._model,
            temperature=self._temperature,
            response_format=JSON_RESPONSE_FORMAT if self._use_json_mode else None,
        )
        elapsed = time.perf_counter() - t0
        if not response_text or not response_text.strip():
            raise EmptyResponseError("the LLM returned an empty response")
        _logger.debug(
            "LLM response received: resp_len=%d elapsed=%.2fs",
            len(response_text),
            elapsed,
        )
        raw_result = self._parse(response_text)
        return self._coerce_all(raw_result, source=source, language=language)

    def _parse(self, response_text: str) -> RawGenerationResult:
        payload = _extract_json(response_text)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ResponseParseError(
                f"model response was not valid JSON: {exc.msg} "
                f"(response_len={len(response_text)}; a large/short response "
                "may have been truncated by the provider's output-token cap)"
            ) from exc
        try:
            return parse_raw_result(data)
        except CoercionError as exc:
            raise ResponseParseError(str(exc)) from exc

    def _coerce_all(
        self,
        raw_result: RawGenerationResult,
        *,
        source: SourceRef | None,
        language: str,
    ) -> tuple[list[IdeaBlock], GenerationStats]:
        stats = GenerationStats(raw_block_count=len(raw_result.blocks))
        blocks: list[IdeaBlock] = []
        for i, raw in enumerate(raw_result.blocks):
            try:
                block = coerce_block(
                    raw, strict=self._strict, source=source, language=language
                )
            except CoercionError as exc:
                stats.errors.append(f"block #{i}: {exc}")
                if self._strict:
                    raise GenerationError(f"block #{i}: {exc}") from exc
                _logger.warning("skipped invalid block #%d: %s", i, exc)
                stats.skipped += 1
                continue
            blocks.append(block)
            stats.emitted += 1
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "coerce blocks: raw=%d emitted=%d skipped=%d errors=%d",
                stats.raw_block_count,
                stats.emitted,
                stats.skipped,
                len(stats.errors),
            )
        return blocks, stats
