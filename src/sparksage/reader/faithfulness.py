"""Faithfulness judging: detect hallucinations before surfacing an answer.

The reader's second stage. After the generator produces an answer, a
:class:`FaithfulnessJudge` scores how well that answer is *supported* by the
retrieved chunks (NLI / LLM-as-judge). A low score -> the reader abstains
("I don't know") rather than surface a hallucination. This is the symmetric
answer-side gate to :class:`~sparksage.query.processor.QueryProcessor`'s
query-side ``min_confidence`` floor.

The core depends only on the :class:`FaithfulnessJudge` protocol and reuses
the existing :class:`~sparksage.generator.LLMClient` (no new abstraction), so
it is fully unit-testable with :class:`~sparksage.generator.FakeLLMClient`.
A future NLI / cross-encoder backend implements the same protocol under an
optional extra.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from sparksage.generator.client import JSON_RESPONSE_FORMAT, LLMClient
from sparksage.reader.prompts import faithfulness_messages
from sparksage.reader.schema import (
    DEFAULT_FAITHFULNESS,
    CoercionError,
    FaithfulnessResult,
    coerce_faithfulness,
    parse_faithfulness_response,
)
from sparksage.retrieve.models import RetrievedChunk

_logger = logging.getLogger(__name__)


class FaithfulnessError(RuntimeError):
    """Base error for faithfulness judging."""


class FaithfulnessEmptyResponseError(FaithfulnessError):
    """The LLM returned no content."""


class FaithfulnessResponseParseError(FaithfulnessError):
    """The model response could not be parsed as the expected JSON."""


@runtime_checkable
class FaithfulnessJudge(Protocol):
    """Score how well an answer is supported by retrieved chunks."""

    def judge(
        self,
        query: str,
        answer_text: str,
        chunks: list[RetrievedChunk],
    ) -> FaithfulnessResult:
        """Return a :class:`FaithfulnessResult` for ``answer_text``."""
        ...


class LLMFaithfulnessJudge:
    """Faithfulness judge backed by an :class:`LLMClient`.

    The model is shown the generated answer plus the same chunks it was built
    from and asked to emit a JSON verdict (score + supported/unsupported claim
    counts). Lenient -> strict coercion clamps the score into ``[0, 1]`` and
    falls back to :data:`~sparksage.reader.schema.DEFAULT_FAITHFULNESS` on a
    missing value.

    On an empty / unparseable response the judge degrades to the default score
    rather than aborting -- so a flaky judge call never crashes a retrieval run.
    Set ``strict=True`` to raise instead.

    Parameters
    ----------
    client:
        Any :class:`LLMClient`. Reused verbatim from the generator.
    model:
        Model name forwarded to the client (ignored by fakes).
    temperature:
        Low (default ``0.0``) for stable verdicts.
    use_json_mode:
        Request JSON-mode structured output when supported.
    strict:
        If ``True``, raise on a bad response instead of falling back.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        use_json_mode: bool = True,
        strict: bool = False,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._use_json_mode = use_json_mode
        self._strict = strict

    @property
    def strict(self) -> bool:
        return self._strict

    def judge(
        self,
        query: str,
        answer_text: str,
        chunks: list[RetrievedChunk],
    ) -> FaithfulnessResult:
        if not str(answer_text).strip():
            return FaithfulnessResult(
                score=0.0,
                supported_claims=0,
                unsupported_claims=0,
                reasoning="empty answer",
            )

        messages = faithfulness_messages(query, answer_text, chunks)
        response_text = self._client.complete(
            messages,
            model=self._model,
            temperature=self._temperature,
            response_format=JSON_RESPONSE_FORMAT if self._use_json_mode else None,
        )
        if not response_text or not response_text.strip():
            if self._strict:
                raise FaithfulnessEmptyResponseError(
                    "the LLM returned an empty response"
                )
            _logger.warning("FaithfulnessJudge empty response; using default score")
            return FaithfulnessResult(
                score=DEFAULT_FAITHFULNESS, reasoning="empty judge response"
            )

        try:
            raw = parse_faithfulness_response(response_text)
            return coerce_faithfulness(raw, strict=self._strict)
        except CoercionError as exc:
            if self._strict:
                raise FaithfulnessResponseParseError(str(exc)) from exc
            _logger.warning("FaithfulnessJudge parse failure (%s); using default", exc)
            return FaithfulnessResult(
                score=DEFAULT_FAITHFULNESS, reasoning=f"parse error: {exc}"
            )


__all__ = [
    "FaithfulnessEmptyResponseError",
    "FaithfulnessError",
    "FaithfulnessJudge",
    "FaithfulnessResponseParseError",
    "LLMFaithfulnessJudge",
]
