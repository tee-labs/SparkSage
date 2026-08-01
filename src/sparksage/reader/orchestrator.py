"""Reader orchestration: generate -> judge faithfulness -> answer or abstain.

:class:`Reader` is the answer-side counterpart of
:class:`~sparksage.query.processor.QueryProcessor`. It takes retrieved chunks
and produces a :class:`AnswerResult`: a generated answer with grounded
citations, a faithfulness verdict, and a dynamic abstention gate. This closes
the "no answer-generation stage" gap -- the largest single hole the project
analysis identified.

The abstention policy is symmetric with the query side:
:class:`~sparksage.query.processor.QueryProcessor` rejects low-confidence
*queries* before retrieval; the :class:`Reader` abstains on low-faithfulness
*answers* after retrieval -- so the system says "I don't know" rather than
hallucinate. The two thresholds (``min_confidence`` / ``min_faithfulness``)
are configuration, not hidden behaviour.

Everything depends only on the :class:`~sparksage.reader.generator.AnswerGenerator`
and :class:`~sparksage.reader.faithfulness.FaithfulnessJudge` protocols, so it
runs fully offline under :class:`~sparksage.generator.FakeLLMClient`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from sparksage.reader.budget import (
    DEFAULT_CHARS_PER_TOKEN,
    reorder_head_tail,
    trim_to_token_budget,
)
from sparksage.reader.faithfulness import FaithfulnessJudge
from sparksage.reader.generator import AnswerGenerator
from sparksage.reader.schema import FaithfulnessResult, GeneratedAnswer
from sparksage.retrieve.models import RetrievedChunk

_logger = logging.getLogger(__name__)

#: Default floor below which a generated answer is treated as unfaithful.
DEFAULT_MIN_FAITHFULNESS = 0.5

#: Default generator-confidence floor below which the reader abstains.
DEFAULT_MIN_ANSWER_CONFIDENCE = 0.2

#: Canned reply surfaced when the reader abstains.
DEFAULT_ABSTENTION_REPLY = (
    "I'm sorry, I don't have enough reliable information to answer that "
    "question based on the current knowledge base."
)


@dataclass
class AnswerResult:
    """The full outcome of reading retrieved chunks for one query.

    Attributes
    ----------
    query:
        The query that was answered (for provenance).
    answer:
        The :class:`~sparksage.reader.schema.GeneratedAnswer`. When
        ``abstained`` is ``True`` the ``text`` is the abstention reply and
        ``citations`` is empty.
    chunks:
        The retrieved chunks the answer was built from.
    faithfulness:
        The faithfulness verdict, or ``None`` when judging was disabled.
    abstained:
        Convenience flag mirroring ``answer.abstained``.
    abstention_reason:
        Why the reader abstained (``None`` when it answered).
    confidence:
        Effective confidence: the generator confidence modulated by the
        faithfulness score (``conf * faithfulness`` when both are known), so a
        confident-but-unfaithful answer reports low effective confidence.
    """

    query: str
    answer: GeneratedAnswer
    chunks: list[RetrievedChunk] = field(default_factory=list)
    faithfulness: FaithfulnessResult | None = None
    abstained: bool = False
    abstention_reason: str | None = None
    confidence: float = 0.0


class Reader:
    """Generate a grounded answer over retrieved chunks, or abstain.

    Parameters
    ----------
    generator:
        Any :class:`AnswerGenerator` (default :class:`LLMAnswerGenerator`).
    faithfulness_judge:
        Optional :class:`FaithfulnessJudge`. When ``None`` judging is skipped
        and only the generator's own confidence gates abstention.
    min_faithfulness:
        Score below which an answer is abstained (default ``0.5``).
    min_confidence:
        Generator confidence below which the reader abstains (default ``0.2``).
    abstention_reply:
        Canned text surfaced on :class:`AnswerResult` when abstaining.
    max_context_tokens:
        Optional soft ceiling on the total token cost of the chunks fed to the
        generator (and judge). When set, the best-first chunk list is trimmed to
        fit before generation -- the Context-Cliff guard. ``None`` (default)
        disables trimming (forward all chunks, the legacy behaviour).
    chars_per_token:
        Heuristic ratio for the built-in token estimator (default ``4.0``).
    token_counter:
        Optional exact tokenizer (``str -> int``) overriding the heuristic.
    context_keep_min:
        Minimum chunks to retain even when the first chunk exceeds the budget
        (default ``1``). See :func:`~sparksage.reader.budget.trim_to_token_budget`.
    reorder_context:
        When ``True`` and ``max_context_tokens`` is set, interleave the trimmed
        best-first list so the strongest chunks land at the head *and* tail --
        the lost-in-the-middle guard (see
        :func:`~sparksage.reader.budget.reorder_head_tail`). Default ``False``
        to preserve legacy head-only ordering. Ignored when trimming is off.

    Examples
    --------
    >>> from sparksage import FakeLLMClient                         # doctest: +SKIP
    >>> from sparksage.reader import Reader                          # doctest: +SKIP
    >>> reader = Reader(generator=..., faithfulness_judge=...)       # doctest: +SKIP
    >>> result = reader.answer("how to deploy", chunks)             # doctest: +SKIP
    >>> if result.abstained:                                        # doctest: +SKIP
    ...     show(result.abstention_reply)                           # doctest: +SKIP
    """

    def __init__(
        self,
        generator: AnswerGenerator,
        *,
        faithfulness_judge: FaithfulnessJudge | None = None,
        min_faithfulness: float = DEFAULT_MIN_FAITHFULNESS,
        min_confidence: float = DEFAULT_MIN_ANSWER_CONFIDENCE,
        abstention_reply: str = DEFAULT_ABSTENTION_REPLY,
        max_context_tokens: float | None = None,
        chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
        token_counter: Callable[[str], int] | None = None,
        context_keep_min: int = 1,
        reorder_context: bool = False,
    ) -> None:
        if not isinstance(generator, AnswerGenerator):
            raise TypeError("generator must implement the AnswerGenerator protocol")
        if faithfulness_judge is not None and not isinstance(
            faithfulness_judge, FaithfulnessJudge
        ):
            raise TypeError(
                "faithfulness_judge must implement the FaithfulnessJudge protocol"
            )
        if not 0.0 <= min_faithfulness <= 1.0:
            raise ValueError("min_faithfulness must be in [0.0, 1.0]")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0.0, 1.0]")
        if max_context_tokens is not None:
            if not isinstance(max_context_tokens, (int, float)) or isinstance(
                max_context_tokens, bool
            ):
                raise TypeError("max_context_tokens must be a number or None")
            if max_context_tokens < 0:
                raise ValueError("max_context_tokens must be >= 0")
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be > 0")
        if not isinstance(context_keep_min, int) or isinstance(context_keep_min, bool):
            raise TypeError("context_keep_min must be an int")
        if context_keep_min < 1:
            raise ValueError("context_keep_min must be >= 1")
        self._generator = generator
        self._judge = faithfulness_judge
        self._min_faithfulness = min_faithfulness
        self._min_confidence = min_confidence
        self._abstention_reply = abstention_reply
        self._max_context_tokens = max_context_tokens
        self._chars_per_token = chars_per_token
        self._token_counter = token_counter
        self._context_keep_min = context_keep_min
        self._reorder_context = reorder_context

    @property
    def generator(self) -> AnswerGenerator:
        return self._generator

    @property
    def faithfulness_judge(self) -> FaithfulnessJudge | None:
        return self._judge

    @property
    def min_faithfulness(self) -> float:
        return self._min_faithfulness

    @property
    def min_confidence(self) -> float:
        return self._min_confidence

    @property
    def max_context_tokens(self) -> float | None:
        """The Context-Cliff token ceiling, or ``None`` when trimming is off."""
        return self._max_context_tokens

    def answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> AnswerResult:
        """Generate -> (judge) -> gate. Returns an :class:`AnswerResult`."""
        query = str(query)
        if not chunks:
            return self._abstain(
                query, chunks, reason="no retrieved context", faithfulness=None
            )

        chunks = self._apply_budget(chunks)

        answer = self._generator.generate(query, chunks)

        if answer.abstained:
            return AnswerResult(
                query=query,
                answer=answer,
                chunks=chunks,
                faithfulness=None,
                abstained=True,
                abstention_reason=answer.abstention_reason or "generator abstained",
                confidence=0.0,
            )

        faithfulness: FaithfulnessResult | None = None
        if self._judge is not None:
            faithfulness = self._judge.judge(query, answer.text, chunks)
            answer.faithfulness = faithfulness.score
            if faithfulness.score < self._min_faithfulness:
                return self._abstain(
                    query,
                    chunks,
                    reason=(
                        f"faithfulness {faithfulness.score:.2f} below "
                        f"floor {self._min_faithfulness:.2f}"
                    ),
                    faithfulness=faithfulness,
                    generated=answer,
                )

        if answer.confidence < self._min_confidence:
            return self._abstain(
                query,
                chunks,
                reason=(
                    f"confidence {answer.confidence:.2f} below floor "
                    f"{self._min_confidence:.2f}"
                ),
                faithfulness=faithfulness,
                generated=answer,
            )

        eff_conf = answer.confidence
        if faithfulness is not None:
            eff_conf = answer.confidence * faithfulness.score
        return AnswerResult(
            query=query,
            answer=answer,
            chunks=chunks,
            faithfulness=faithfulness,
            abstained=False,
            abstention_reason=None,
            confidence=eff_conf,
        )

    def _apply_budget(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Trim ``chunks`` to the configured token budget (no-op when off).

        Applied once, before generation and judging, so the judge scores the
        answer against exactly the context the generator saw. When
        ``reorder_context`` is set the trimmed list is interleaved head/tail so
        the strongest chunks sit at both ends -- the lost-in-the-middle guard.
        """
        if self._max_context_tokens is None:
            return chunks
        trimmed = trim_to_token_budget(
            chunks,
            self._max_context_tokens,
            chars_per_token=self._chars_per_token,
            token_counter=self._token_counter,
            keep_min=self._context_keep_min,
        )
        if len(trimmed) < len(chunks):
            _logger.debug(
                "context trimmed %d -> %d chunks (max_context_tokens=%s)",
                len(chunks),
                len(trimmed),
                self._max_context_tokens,
            )
        if self._reorder_context and len(trimmed) > 1:
            trimmed = reorder_head_tail(trimmed)
        return trimmed

    def _abstain(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        reason: str,
        faithfulness: FaithfulnessResult | None,
        generated: GeneratedAnswer | None = None,
    ) -> AnswerResult:
        answer = GeneratedAnswer(
            text=self._abstention_reply,
            citations=[],
            grounded_block_ids=[],
            confidence=0.0,
            abstained=True,
            abstention_reason=reason,
        )
        return AnswerResult(
            query=query,
            answer=answer,
            chunks=chunks,
            faithfulness=faithfulness,
            abstained=True,
            abstention_reason=reason,
            confidence=0.0,
        )


__all__ = [
    "DEFAULT_ABSTENTION_REPLY",
    "DEFAULT_MIN_ANSWER_CONFIDENCE",
    "DEFAULT_MIN_FAITHFULNESS",
    "AnswerResult",
    "Reader",
]
