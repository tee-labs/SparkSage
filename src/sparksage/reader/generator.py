"""Answer generation: retrieved chunks -> a grounded natural-language answer.

This is the biggest missing piece flagged by the project analysis: the
query-time pipeline ended at rewriting, with no reader to actually *answer*.
:class:`AnswerGenerator` is the reader-side counterpart of
:class:`~sparksage.query.rewriter.QueryRewriter`: it takes the retrieved
:class:`~sparksage.retrieve.models.RetrievedChunk` list and produces a
:class:`~sparksage.reader.schema.GeneratedAnswer` with grounded citations.

IdeaBlock's QA-aligned design is the payoff here: feeding the model whole
``critical_question`` + ``trusted_answer`` pairs (rather than naive text shards)
yields more focused, less drift-prone answers -- the real dividend of the data
layer, cashed in at the only stage where it shows.

The core depends only on the :class:`AnswerGenerator` protocol and reuses the
existing :class:`~sparksage.generator.LLMClient` (no new LLM abstraction), so
it is fully unit-testable with :class:`~sparksage.generator.FakeLLMClient`.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from sparksage.generator.client import JSON_RESPONSE_FORMAT, LLMClient
from sparksage.reader.prompts import answer_messages
from sparksage.reader.schema import (
    CoercionError,
    GeneratedAnswer,
    RawAnswer,
    coerce_answer,
    parse_answer_response,
)
from sparksage.retrieve.models import RetrievedChunk

_logger = logging.getLogger(__name__)


class AnswerError(RuntimeError):
    """Base error for answer generation."""


class AnswerEmptyResponseError(AnswerError):
    """The LLM returned no content."""


class AnswerResponseParseError(AnswerError):
    """The model response could not be parsed as the expected JSON."""


@runtime_checkable
class AnswerGenerator(Protocol):
    """Protocol: turn retrieved chunks into a grounded :class:`GeneratedAnswer`.

    Implementations should be deterministic for a given input. The reader
    orchestrator may run the generator and then a
    :class:`~sparksage.reader.faithfulness.FaithfulnessJudge`.
    """

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> GeneratedAnswer:
        """Return a :class:`GeneratedAnswer` for ``query`` over ``chunks``."""
        ...


class LLMAnswerGenerator:
    """Answer generator backed by an :class:`LLMClient`.

    The model is given the query plus each candidate's
    ``critical_question`` / ``trusted_answer`` and asked to emit a JSON answer
    with citations referencing candidate block ids. The lenient -> strict
    coercion binds those ids to the schema's ``source.uri`` / ``source.locator``
    provenance, so a grounded citation stays traceable to its source line.

    When ``chunks`` is empty the generator short-circuits to an abstention
    (no context -> no answer), avoiding a pointless LLM call.

    Parameters
    ----------
    client:
        Any :class:`LLMClient` (e.g. :class:`OpenAICompatibleClient`,
        :class:`FakeLLMClient`). Reused verbatim from the generator.
    model:
        Model name forwarded to the client (ignored by fakes).
    temperature:
        Low (default ``0.2``) for faithful, low-hallucination answers.
    use_json_mode:
        Request JSON-mode structured output when supported.
    strict:
        If ``False`` (default), an empty answer falls back to abstention rather
        than raising. If ``True``, raise on an empty answer.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.2,
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

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> GeneratedAnswer:
        if not chunks:
            return GeneratedAnswer(
                text="",
                citations=[],
                grounded_block_ids=[],
                confidence=0.0,
                abstained=True,
                abstention_reason="no retrieved context",
            )

        id_to_citation = {str(c.block.id): c.to_citation() for c in chunks}
        valid_ids = set(id_to_citation)

        messages = answer_messages(query, chunks)
        response_text = self._client.complete(
            messages,
            model=self._model,
            temperature=self._temperature,
            response_format=JSON_RESPONSE_FORMAT if self._use_json_mode else None,
        )
        if not response_text or not response_text.strip():
            raise AnswerEmptyResponseError("the LLM returned an empty response")

        try:
            raw: RawAnswer = parse_answer_response(response_text)
            return coerce_answer(
                raw, valid_ids, id_to_citation, strict=self._strict
            )
        except CoercionError as exc:
            raise AnswerResponseParseError(str(exc)) from exc


__all__ = [
    "AnswerEmptyResponseError",
    "AnswerError",
    "AnswerGenerator",
    "AnswerResponseParseError",
    "LLMAnswerGenerator",
]
