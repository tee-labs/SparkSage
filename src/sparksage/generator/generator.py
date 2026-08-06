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
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from sparksage.generator.client import JSON_RESPONSE_FORMAT, LLMClient
from sparksage.generator.prompts import build_messages
from sparksage.generator.schema import (
    CoercionError,
    RawGenerationResult,
    coerce_block,
    parse_raw_result,
)
from sparksage.llmutil import extract_json
from sparksage.schema.ideablock import IdeaBlock
from sparksage.schema.source import SourceRef

_logger = logging.getLogger(__name__)

#: Default per-segment character budget for large inputs. Source text longer
#: than this is split into coherent segments (see :func:`_split_text_segments`)
#: and each segment is sent as its own generation request. This keeps the LLM's
#: JSON output small enough that it never gets truncated mid-stream by the
#: provider's output-token cap -- the cause of the ``Unprocessable Entity``
#: ingest failures on large uploads. Modern long-context models (128k+) handle
#: this budget comfortably (Chinese 12000 chars ~= 6-8k tokens), so it is tuned
#: to minimize segmentation rather than to fit small-context legacy models.
#: Tune via ``max_input_chars=``.
DEFAULT_MAX_INPUT_CHARS = 12000

#: Default number of segments generated concurrently. ``1`` keeps the legacy
#: serial behaviour (deterministic, ordered response consumption for the
#: stateful :class:`FakeLLMClient` in tests). Production wiring raises this via
#: ``SPARKSAGE_GENERATE_MAX_WORKERS`` so a multi-segment large document is
#: chunked into parallel LLM calls instead of a single serial queue -- the
#: single biggest latency win on big uploads.
DEFAULT_SEGMENT_WORKERS = 1

#: Lower bound of the prompt's "one block per 300-500 characters" split rule.
#: Used to pre-compute a per-segment ``max_blocks`` cap when the caller did not
#: pass one, so the model cannot over-split (each extra block means extra output
#: tokens and a slower generation).
MIN_CHARS_PER_BLOCK = 800

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
    return extract_json(
        text,
        error_type=ResponseParseError,
        empty_msg="empty model response",
        lenient=True,
    )


def _repair_json(text: str) -> str:
    """Best-effort repair of truncated / loosely-formatted JSON.

    Removes trailing commas before closing brackets (a common artefact of a
    model emitting a dangling comma before its output was cut) and auto-closes
    any unclosed objects/arrays. String state is tracked so braces inside
    string literals are not counted. Used to recover from provider output-token
    truncation so a near-complete response is not wasted.
    """
    text = re.sub(r",\s*([}\]])", r"\1", text)
    stack: list[str] = []
    in_str = False
    escape = False
    for ch in text:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in "}]":
                if stack and stack[-1] == ch:
                    stack.pop()
    return text + "".join(reversed(stack))


def _salvage_blocks_json(text: str) -> str | None:
    """Recover complete blocks from a truncated ``{"blocks": [...]}`` response.

    When a provider's output-token cap cuts the response mid-block, the whole
    segment would otherwise be dropped (``segment i/N failed, skipping``). This
    scans the ``blocks`` array and keeps only the top-level ``{...}`` objects
    that fully closed before the cut, returning a well-formed
    ``{"blocks": [...]}`` payload. Returns ``None`` if the response is not
    blocks-shaped or no complete block survived.
    """
    key = '"blocks"'
    idx = text.find(key)
    if idx == -1:
        return None
    arr_start = text.find("[", idx)
    if arr_start == -1:
        return None
    blocks_text: list[str] = []
    depth = 0
    obj_start = -1
    in_str = False
    escape = False
    i = arr_start + 1
    n = len(text)
    while i < n:
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and obj_start != -1:
                    blocks_text.append(text[obj_start : i + 1])
                    obj_start = -1
            elif ch == "]" and depth == 0:
                break
        i += 1
    if not blocks_text:
        return None
    return '{"blocks": [' + ",".join(blocks_text) + "]}"


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
    max_workers:
        Number of segments generated concurrently. ``1`` (default) keeps the
        legacy serial behaviour so the stateful :class:`FakeLLMClient` used in
        tests consumes its scripted responses in order. Set higher (e.g. via
        ``SPARKSAGE_GENERATE_MAX_WORKERS`` in production wiring) so a multi-
        segment large document fans out into parallel LLM calls instead of a
        single serial queue -- the single biggest latency win on big uploads.
        Concurrency is only used when there is more than one segment; a single
        segment always runs inline.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        language: str = "en",
        strict: bool = False,
        use_json_mode: bool = True,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        max_workers: int = DEFAULT_SEGMENT_WORKERS,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._language = language
        self._strict = strict
        self._use_json_mode = use_json_mode
        self._max_input_chars = int(max_input_chars)
        self._max_workers = max(1, int(max_workers))

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
            "json_mode=%s max_workers=%d",
            len(str(text)),
            len(segments),
            max_blocks,
            lang,
            self._use_json_mode,
            self._max_workers,
        )

        all_blocks: list[IdeaBlock] = []
        stats = GenerationStats()
        segment_results = self._run_segments(
            segments,
            source=source,
            language=lang,
            per_segment_max=per_segment_max,
        )
        for i, result in enumerate(segment_results):
            if isinstance(result, BaseException):
                # A single-segment run (or strict mode) surfaces the error to
                # preserve the generation contract. A multi-segment run in
                # non-strict mode is resilient: one bad segment is logged and
                # recorded rather than wasting the whole large-doc batch.
                if single or self._strict:
                    raise result
                _logger.warning(
                    "segment %d/%d failed, skipping: %s",
                    i + 1,
                    len(segments),
                    result,
                )
                stats.errors.append(f"segment {i + 1}/{len(segments)}: {result}")
                continue
            seg_blocks, seg_stats = result
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
    def _run_segments(
        self,
        segments: list[str],
        *,
        source: SourceRef | None,
        language: str,
        per_segment_max: int | None,
    ) -> list:
        """Run one LLM generation per segment, returning results in order.

        Each element is either a ``(blocks, stats)`` tuple or the
        :class:`Exception` that segment raised (a
        :class:`ResponseParseError` / :class:`EmptyResponseError` /
        :class:`GenerationError`); the caller applies the strict / resilient
        policy uniformly. Single-segment or serial runs (``max_workers <= 1``)
        execute inline so the stateful :class:`FakeLLMClient` used in tests
        consumes its scripted responses in deterministic order. A multi-segment
        run with ``max_workers > 1`` fans the segments out across a thread pool
        -- the segments are independent HTTP requests, so running them
        concurrently collapses an N-segment serial queue into roughly one round
        (the single biggest latency win on large uploads). Results are gathered
        in submission order so emitted blocks stay ordered.
        """
        n = len(segments)
        parallel = n > 1 and self._max_workers > 1
        if not parallel:
            results = []
            for segment in segments:
                try:
                    results.append(
                        self._generate_one_segment(
                            segment,
                            source=source,
                            language=language,
                            max_blocks=per_segment_max,
                        )
                    )
                except (ResponseParseError, EmptyResponseError, GenerationError) as exc:
                    results.append(exc)
            return results
        workers = min(self._max_workers, n)
        results: list = [None] * n
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(
                    self._generate_one_segment,
                    segment,
                    source=source,
                    language=language,
                    max_blocks=per_segment_max,
                ): idx
                for idx, segment in enumerate(segments)
            }
            for future in future_to_idx:
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except (
                    ResponseParseError,
                    EmptyResponseError,
                    GenerationError,
                ) as exc:
                    results[idx] = exc
        return results

    def _generate_one_segment(
        self,
        text: str,
        *,
        source: SourceRef | None,
        language: str,
        max_blocks: int | None,
    ) -> tuple[list[IdeaBlock], GenerationStats]:
        """Run one LLM generation request for a single text segment."""
        if max_blocks is None:
            max_blocks = max(1, math.ceil(len(text) / MIN_CHARS_PER_BLOCK))
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
            data = self._recover_json(response_text, payload, exc)
        try:
            return parse_raw_result(data)
        except CoercionError as exc:
            raise ResponseParseError(str(exc)) from exc

    def _recover_json(
        self,
        response_text: str,
        payload: str,
        exc: json.JSONDecodeError,
    ) -> dict:
        """Recover parseable JSON from a truncated provider response.

        Two complementary paths address provider output-token truncation (the
        cause of the ``Expecting property name enclosed in double quotes``
        mid-key failures): first repair dangling commas / unclosed braces and
        retry; if that still fails, salvage the complete blocks before the cut
        so a near-complete segment is indexed rather than dropped entirely.
        Raises :class:`ResponseParseError` if neither path yields valid JSON.
        """
        repaired = _repair_json(payload)
        if repaired != payload:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        salvaged = _salvage_blocks_json(response_text)
        if salvaged is not None:
            try:
                data = json.loads(salvaged)
            except json.JSONDecodeError:
                data = None
            if data is not None:
                _logger.info(
                    "recovered %d complete blocks from truncated response "
                    "(response_len=%d)",
                    len(data.get("blocks", [])) if isinstance(data, dict) else 0,
                    len(response_text),
                )
                return data
        raise ResponseParseError(
            f"model response was not valid JSON: {exc.msg} "
            f"(response_len={len(response_text)}; a large/short response "
            "may have been truncated by the provider's output-token cap)"
        ) from exc

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
