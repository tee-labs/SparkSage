"""Answer generation for RAG: retrieved chunks -> grounded answer (or abstain).

This is the reader -- the answer-side counterpart of :mod:`sparksage.query` and
the missing "right half" of the QA pipeline. It consumes
:class:`~sparksage.retrieve.models.RetrievedChunk` lists and produces a
:class:`~sparksage.reader.schema.GeneratedAnswer` with grounded citations, then
optionally judges faithfulness and abstains rather than hallucinate.

IdeaBlock's QA-aligned design pays off here: feeding whole
``critical_question`` + ``trusted_answer`` pairs to the model yields more
focused answers than naive text shards -- the real dividend of the data layer,
realized at the only stage where it shows.

Everything depends only on the :class:`AnswerGenerator` and
:class:`FaithfulnessJudge` protocols (both reusing the existing
:class:`~sparksage.generator.LLMClient`), so it runs fully offline under
:class:`~sparksage.generator.FakeLLMClient`. This is where the schema's
``source.locator`` provenance is finally *consumed* -- a grounded citation
carries ``uri`` / ``locator`` / ``title`` straight from the backing block.

Pipeline::

    retrieved chunks
        -> token-budget trim       [optional, Context-Cliff guard]
        -> AnswerGenerator       (grounded answer + citations)
        -> FaithfulnessJudge     [optional]
        -> abstention gate       (low faithfulness / confidence -> "I don't know")

Example
-------
::

    from sparksage import FakeLLMClient
    from sparksage.reader import LLMAnswerGenerator, LLMFaithfulnessJudge, Reader

    client = FakeLLMClient(responses=[...])
    reader = Reader(
        generator=LLMAnswerGenerator(client),
        faithfulness_judge=LLMFaithfulnessJudge(client),
    )
    result = reader.answer("how to deploy", chunks)
    if result.abstained:
        print(result.abstention_reason)
    else:
        print(result.answer.text, result.answer.citations)
"""

from sparksage.reader.budget import (
    DEFAULT_CHARS_PER_TOKEN,
    DEFAULT_KEEP_MIN,
    approx_tokens,
    trim_to_token_budget,
)
from sparksage.reader.faithfulness import (
    FaithfulnessEmptyResponseError,
    FaithfulnessError,
    FaithfulnessJudge,
    FaithfulnessResponseParseError,
    LLMFaithfulnessJudge,
)
from sparksage.reader.generator import (
    AnswerEmptyResponseError,
    AnswerError,
    AnswerGenerator,
    AnswerResponseParseError,
    LLMAnswerGenerator,
)
from sparksage.reader.orchestrator import (
    DEFAULT_ABSTENTION_REPLY,
    DEFAULT_MIN_ANSWER_CONFIDENCE,
    DEFAULT_MIN_FAITHFULNESS,
    AnswerResult,
    Reader,
)
from sparksage.reader.prompts import (
    answer_messages,
    answer_system_prompt,
    answer_user_prompt,
    faithfulness_messages,
    faithfulness_system_prompt,
    faithfulness_user_prompt,
)
from sparksage.reader.schema import (
    DEFAULT_ANSWER_CONFIDENCE,
    DEFAULT_FAITHFULNESS,
    CoercionError,
    FaithfulnessResult,
    GeneratedAnswer,
    RawAnswer,
    RawCitation,
    RawFaithfulness,
    coerce_answer,
    coerce_citations,
    coerce_faithfulness,
    extract_json,
    parse_answer_response,
    parse_faithfulness_response,
    parse_raw_answer,
    parse_raw_faithfulness,
)

__all__ = [
    "AnswerEmptyResponseError",
    "AnswerError",
    "AnswerGenerator",
    "AnswerResponseParseError",
    "AnswerResult",
    "CoercionError",
    "DEFAULT_ABSTENTION_REPLY",
    "DEFAULT_ANSWER_CONFIDENCE",
    "DEFAULT_CHARS_PER_TOKEN",
    "DEFAULT_FAITHFULNESS",
    "DEFAULT_KEEP_MIN",
    "DEFAULT_MIN_ANSWER_CONFIDENCE",
    "DEFAULT_MIN_FAITHFULNESS",
    "FaithfulnessEmptyResponseError",
    "FaithfulnessError",
    "FaithfulnessJudge",
    "FaithfulnessResponseParseError",
    "FaithfulnessResult",
    "GeneratedAnswer",
    "LLMAnswerGenerator",
    "LLMFaithfulnessJudge",
    "RawAnswer",
    "RawCitation",
    "RawFaithfulness",
    "Reader",
    "answer_messages",
    "answer_system_prompt",
    "answer_user_prompt",
    "approx_tokens",
    "coerce_answer",
    "coerce_citations",
    "coerce_faithfulness",
    "extract_json",
    "faithfulness_messages",
    "faithfulness_system_prompt",
    "faithfulness_user_prompt",
    "parse_answer_response",
    "parse_faithfulness_response",
    "parse_raw_answer",
    "parse_raw_faithfulness",
    "trim_to_token_budget",
]
