"""Retrieval grading: score retrieved-chunk relevance for self-corrective RAG.

This is the retrieval-side counterpart of
:class:`~sparksage.reader.faithfulness.FaithfulnessJudge`. After retrieval, a
:class:`RetrievalGrader` scores how *relevant* the retrieved chunks are to the
query. A low score lets the :class:`~sparksage.qa.QAEngine` refine the query and
re-retrieve (the self-reflective / iterative retrieval loop) -- the missing
middle gate of the three-stage policy:

    query-side gate (min_confidence) -> retrieval-side gate (min_relevance)
        -> answer-side gate (min_faithfulness)

It depends only on the :class:`RetrievalGrader` protocol and reuses the existing
:class:`~sparksage.generator.LLMClient` (no new abstraction), so it is fully
unit-testable with :class:`~sparksage.generator.FakeLLMClient`.

IdeaBlock's QA-alignment pays off here too: each candidate is graded by its
``critical_question`` + ``trusted_answer``, so the judge sees self-contained
units rather than naive text shards -- the same dividend the reader cashes in.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, ValidationError

from sparksage.generator.client import JSON_RESPONSE_FORMAT, LLMClient
from sparksage.retrieve.models import RetrievedChunk

_logger = logging.getLogger(__name__)

#: Relevance assumed when the grader omits a usable score.
DEFAULT_RELEVANCE = 0.5

#: Default cap on how many top chunks the grader inspects (cost control).
DEFAULT_GRADE_TOP_K = 5

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$", re.MULTILINE)

_SYSTEM_TEMPLATE = """\
You are a strict retrieval-relevance grader. Given a user question and the \
knowledge chunks retrieved for it, decide how RELEVANT the chunks are to \
answering the question.

- score = 1.0 means the chunks directly and fully address the question.
- score = 0.5 means the chunks are partially on-topic but miss key aspects.
- score = 0.0 means the chunks are irrelevant / off-topic.
- Judge relevance, NOT faithfulness or correctness.

Respond with ONLY a JSON object of the form:
{{"reasoning": "brief reasoning", "score": <float in [0, 1]>}}
No markdown, no commentary -- just the JSON object.
"""


class GraderError(RuntimeError):
    """Base error for retrieval grading."""


class GraderEmptyResponseError(GraderError):
    """The LLM returned no content."""


class GraderResponseParseError(GraderError):
    """The model response could not be parsed as the expected JSON."""


class CoercionError(ValueError):
    """Raised when raw relevance data cannot be coerced."""


class RawRelevance(BaseModel):
    """Lenient relevance verdict as emitted by an LLM."""

    model_config = ConfigDict(extra="ignore")

    reasoning: str = ""
    score: float = DEFAULT_RELEVANCE


@dataclass
class RelevanceResult:
    """Strict verdict on how relevant retrieved chunks are to a query.

    Attributes
    ----------
    score:
        Relevance in ``[0, 1]``. Below the QA engine's ``min_relevance`` the
        engine refines the query and re-retrieves (when a refiner is wired).
    reasoning:
        The grader's brief explanation (fed back to the query refiner).
    """

    score: float
    reasoning: str = ""


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise CoercionError("empty grader response")
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
    raise CoercionError("grader response was not valid JSON")


def _coerce_score(raw: float, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value != value or value in (float("inf"), float("-inf")):
        return default
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def parse_raw_relevance(data: object) -> RawRelevance:
    if not isinstance(data, dict):
        raise CoercionError(f"expected a JSON object, got {type(data).__name__}")
    try:
        return RawRelevance.model_validate(data)
    except ValidationError as exc:
        raise CoercionError(str(exc)) from exc


def parse_relevance_response(text: str) -> RawRelevance:
    return parse_raw_relevance(json.loads(_extract_json(text)))


def coerce_relevance(raw: RawRelevance) -> RelevanceResult:
    """Normalize a :class:`RawRelevance` into a strict :class:`RelevanceResult`."""
    if not isinstance(raw, RawRelevance):
        raise CoercionError("expected a RawRelevance")
    return RelevanceResult(
        score=_coerce_score(raw.score, DEFAULT_RELEVANCE),
        reasoning=(raw.reasoning or "").strip(),
    )


@runtime_checkable
class RetrievalGrader(Protocol):
    """Score how relevant retrieved chunks are to a query.

    Implementations should be deterministic for a given input. The QA engine
    calls :meth:`grade` after each retrieval in the self-reflective loop.
    """

    def grade(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> RelevanceResult:
        """Return a :class:`RelevanceResult` for ``chunks`` vs ``query``."""
        ...


class LLMRetrievalGrader:
    """Retrieval grader backed by an :class:`LLMClient`.

    The model is shown the query plus the top chunks (each rendered as its
    ``critical_question`` + ``trusted_answer``) and asked to emit a JSON
    relevance score. The output is coerced leniently and clamped into
    ``[0, 1]``.

    On an empty / unparseable response the grader degrades to the default score
    rather than aborting -- so a flaky grader call never crashes a retrieval run.
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
    top_k:
        Maximum number of top chunks to show the model (cost control). Defaults
        to :data:`DEFAULT_GRADE_TOP_K`.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        use_json_mode: bool = True,
        strict: bool = False,
        top_k: int = DEFAULT_GRADE_TOP_K,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._use_json_mode = use_json_mode
        self._strict = strict
        self._top_k = top_k if top_k and top_k > 0 else DEFAULT_GRADE_TOP_K

    @property
    def strict(self) -> bool:
        return self._strict

    @property
    def top_k(self) -> int:
        return self._top_k

    def grade(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> RelevanceResult:
        query = str(query)
        if not chunks:
            return RelevanceResult(score=0.0, reasoning="no retrieved context")

        messages = grade_messages(query, chunks[: self._top_k])
        response_text = self._client.complete(
            messages,
            model=self._model,
            temperature=self._temperature,
            response_format=JSON_RESPONSE_FORMAT if self._use_json_mode else None,
        )
        if not response_text or not response_text.strip():
            if self._strict:
                raise GraderEmptyResponseError("the LLM returned an empty response")
            _logger.warning("RetrievalGrader empty response; using default score")
            return RelevanceResult(
                score=DEFAULT_RELEVANCE, reasoning="empty grader response"
            )

        try:
            raw = parse_relevance_response(response_text)
            return coerce_relevance(raw)
        except CoercionError as exc:
            if self._strict:
                raise GraderResponseParseError(str(exc)) from exc
            _logger.warning("RetrievalGrader parse failure (%s); using default", exc)
            return RelevanceResult(
                score=DEFAULT_RELEVANCE, reasoning=f"parse error: {exc}"
            )

    def __repr__(self) -> str:
        return f"LLMRetrievalGrader(model={self._model!r}, top_k={self._top_k})"


def _render_chunks(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for c in chunks:
        bid = str(c.block.id)
        lines.append(f"[{bid}] {c.block.critical_question} || {c.block.trusted_answer}")
    return "\n".join(lines)


def grade_system_prompt() -> str:
    return _SYSTEM_TEMPLATE


def grade_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = _render_chunks(chunks)
    return (
        f"Question: {query.strip()}\n\n"
        f"Retrieved chunks:\n{context}\n\n"
        "Grade how relevant these chunks are to the question."
    )


def grade_messages(
    query: str, chunks: list[RetrievedChunk]
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": grade_system_prompt()},
        {"role": "user", "content": grade_user_prompt(query, chunks)},
    ]


__all__ = [
    "CoercionError",
    "DEFAULT_GRADE_TOP_K",
    "DEFAULT_RELEVANCE",
    "GraderEmptyResponseError",
    "GraderError",
    "GraderResponseParseError",
    "LLMRetrievalGrader",
    "RawRelevance",
    "RetrievalGrader",
    "RelevanceResult",
    "coerce_relevance",
    "grade_messages",
    "grade_system_prompt",
    "grade_user_prompt",
    "parse_raw_relevance",
    "parse_relevance_response",
]
