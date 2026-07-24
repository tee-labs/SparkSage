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
import re
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

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$", re.MULTILINE)


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
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._language = language
        self._strict = strict
        self._use_json_mode = use_json_mode

    def generate(
        self,
        text: str,
        *,
        source: SourceRef | None = None,
        source_uri: str | None = None,
        source_title: str | None = None,
        max_blocks: int | None = None,
        language: str | None = None,
    ) -> list[IdeaBlock]:
        """Generate IdeaBlocks from ``text``.

        A :class:`SourceRef` provenance is built from ``source`` (if given) or
        from the ``source_uri``/``source_title`` shortcut and attached to every
        emitted block. Returns the list of valid blocks; in non-strict mode this
        may be shorter than what the model produced.
        """
        if text is None or not str(text).strip():
            raise ValueError("generate() requires non-empty text")

        if source is None and source_uri is not None:
            source = SourceRef(uri=source_uri, title=source_title)

        lang = language or self._language

        messages = build_messages(
            text,
            source=source,
            max_blocks=max_blocks,
            language=lang,
        )

        response_text = self._client.complete(
            messages,
            model=self._model,
            temperature=self._temperature,
            response_format=JSON_RESPONSE_FORMAT if self._use_json_mode else None,
        )

        if not response_text or not response_text.strip():
            raise EmptyResponseError("the LLM returned an empty response")

        raw_result = self._parse(response_text)
        blocks, _stats = self._coerce_all(raw_result, source=source, language=lang)
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

        messages = build_messages(
            text, source=source, max_blocks=max_blocks, language=lang
        )
        response_text = self._client.complete(
            messages,
            model=self._model,
            temperature=self._temperature,
            response_format=JSON_RESPONSE_FORMAT if self._use_json_mode else None,
        )
        if not response_text or not response_text.strip():
            raise EmptyResponseError("the LLM returned an empty response")

        raw_result = self._parse(response_text)
        return self._coerce_all(raw_result, source=source, language=lang)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _parse(self, response_text: str) -> RawGenerationResult:
        payload = _extract_json(response_text)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ResponseParseError(
                f"model response was not valid JSON: {exc.msg}"
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
                stats.skipped += 1
                continue
            blocks.append(block)
            stats.emitted += 1
        return blocks, stats
