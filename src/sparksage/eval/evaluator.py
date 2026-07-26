"""End-to-end QA evaluation: run the QA engine over a test set, score quality.

This is the answer-correctness counterpart to :mod:`sparksage.bench` (which
scores *retrieval* alone). :class:`QAEvaluator` runs each :class:`QATestCase`
through a :class:`~sparksage.qa.QAEngine`, scores the surfaced answer for
*correctness*, and rolls the per-case outcomes into a :class:`QAEvalReport`
(mean correctness, abstention rate, retrieval hit@k reusing
:func:`~sparksage.bench.evaluate_retrieval`, mean faithfulness).

Correctness scoring is pluggable via the :class:`CorrectnessJudge` protocol:

* :class:`TokenOverlapJudge` (the default) -- dependency-free token-overlap F1
  between the generated and reference answers. Always available, fully offline.
* :class:`LLMCorrectnessJudge` -- an LLM-as-judge scoring semantic correctness,
  reusing the existing :class:`~sparksage.generator.LLMClient` (no new
  abstraction). Swap in for higher-fidelity scoring at the cost of an LLM call
  per case.

When a case has no reference answer, correctness falls back to a
retrieval-hit + faithfulness proxy (an abstention or a miss scores ``0.0``).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol, runtime_checkable

from sparksage.bench.metrics import evaluate_retrieval
from sparksage.embed.store import SearchHit
from sparksage.eval.models import QACaseResult, QAEvalReport, QATestCase
from sparksage.generator.client import JSON_RESPONSE_FORMAT, LLMClient
from sparksage.qa import QAEngine

_logger = logging.getLogger(__name__)

#: Default top-k for the retrieval-hit metric in QA evaluation.
DEFAULT_EVAL_K = 5

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CJK_RE = re.compile(
    "[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff"
    "\uac00-\ud7af]"
)


def _tokens(text: str) -> list[str]:
    if not text:
        return []
    toks = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    toks.extend(c for c in text if _CJK_RE.match(c))
    return toks


def token_f1(generated: str, reference: str) -> float:
    """Dependency-free token-overlap F1 between two answers.

    Used as the default correctness score (no LLM call). Handles mixed
    Latin + CJK text via the same word/char tokenization the lexical retrieever
    uses. Returns ``0.0`` when either side has no tokens.
    """
    gen = _tokens(generated)
    ref = _tokens(reference)
    if not gen or not ref:
        return 0.0
    gen_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for t in gen:
        gen_counts[t] = gen_counts.get(t, 0) + 1
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = sum(min(gen_counts[t], ref_counts.get(t, 0)) for t in gen_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(gen)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


@runtime_checkable
class CorrectnessJudge(Protocol):
    """Score answer correctness in ``[0, 1]``."""

    def score(self, query: str, generated: str, reference: str) -> float:
        ...


class TokenOverlapJudge:
    """Dependency-free F1 correctness judge (the default)."""

    def score(self, query: str, generated: str, reference: str) -> float:
        return token_f1(generated, reference)


class LLMCorrectnessJudge:
    """LLM-as-judge correctness scorer, reusing :class:`LLMClient`.

    The model is shown the query, the reference answer, and the generated
    answer, and asked for a JSON ``{"score": <0-1>}`` verdict. Lenient parsing
    falls back to :class:`TokenOverlapJudge` on an unparseable response, so a
    flaky judge call degrades to the offline score rather than crashing a run.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        use_json_mode: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._use_json_mode = use_json_mode
        self._fallback = TokenOverlapJudge()
        self.fallbacks = 0

    def score(self, query: str, generated: str, reference: str) -> float:
        if not str(generated).strip() or not str(reference).strip():
            return self._fallback.score(query, generated, reference)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an answer-correctness judge. Score how well the "
                    "generated answer matches the reference answer for the given "
                    "query, using ONLY semantic correctness (not wording). "
                    "Respond with ONLY a JSON object: {\"score\": <float 0-1>}. "
                    "1.0 = fully correct, 0.0 = wrong."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\nReference: {reference}\n\n"
                    f"Generated: {generated}\n\nReturn the JSON score."
                ),
            },
        ]
        try:
            raw = self._client.complete(
                messages,
                model=self._model,
                temperature=self._temperature,
                response_format=JSON_RESPONSE_FORMAT if self._use_json_mode else None,
            )
            payload = json.loads(raw.strip())
            score = float(payload.get("score", -1) if isinstance(payload, dict) else -1)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.fallbacks += 1
            _logger.warning("LLMCorrectnessJudge parse failure (%s); token-F1 fallback", exc)
            return self._fallback.score(query, generated, reference)
        if score != score or score < 0.0:
            self.fallbacks += 1
            return self._fallback.score(query, generated, reference)
        return max(0.0, min(1.0, score))


class QAEvaluator:
    """Run a :class:`QAEngine` over a test set and score the answers.

    Parameters
    ----------
    engine:
        The :class:`~sparksage.qa.QAEngine` under test.
    judge:
        :class:`CorrectnessJudge` for answer correctness. Defaults to
        :class:`TokenOverlapJudge` (offline). Pass :class:`LLMCorrectnessJudge`
        for semantic scoring.

    Examples
    --------
    >>> from sparksage.eval import QAEvaluator, QATestCase   # doctest: +SKIP
    >>> evaluator = QAEvaluator(engine=engine)               # doctest: +SKIP
    >>> report = evaluator.run(cases, k=5)                   # doctest: +SKIP
    >>> report.mean_correctness                              # doctest: +SKIP
    """

    def __init__(
        self,
        engine: QAEngine,
        *,
        judge: CorrectnessJudge | None = None,
    ) -> None:
        self._engine = engine
        self._judge: CorrectnessJudge = judge if judge is not None else TokenOverlapJudge()

    @property
    def engine(self) -> QAEngine:
        return self._engine

    @property
    def judge(self) -> CorrectnessJudge:
        return self._judge

    def run(
        self,
        cases: list[QATestCase],
        *,
        k: int = DEFAULT_EVAL_K,
        use_cache: bool = False,
    ) -> QAEvalReport:
        """Evaluate ``cases`` end-to-end and return a :class:`QAEvalReport`.

        ``use_cache`` defaults to ``False`` so each case exercises the full
        pipeline (caching would skew aggregate metrics toward repeat queries).
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        results: list[QACaseResult] = []
        rankings: list[list[SearchHit]] = []
        ground_truth: list[set[str]] = []

        for case in cases:
            qr = self._engine.ask(case.query, k=k, use_cache=use_cache)
            retrieved_ids: list[str] = []
            retrieved_hits: list[SearchHit] = []
            if qr.retrieval is not None:
                for chunk in qr.retrieval.chunks:
                    bid = str(chunk.block.id)
                    retrieved_ids.append(bid)
                    retrieved_hits.append(SearchHit(block_id=bid, score=chunk.score))
            relevant = {str(b) for b in case.relevant_block_ids}
            hit = bool(relevant) and any(bid in relevant for bid in retrieved_ids)

            generated = qr.text
            abstained = qr.abstained
            faithfulness = (
                qr.answer.faithfulness.score
                if qr.answer is not None and qr.answer.faithfulness is not None
                else None
            )
            confidence = qr.confidence if hasattr(qr, "confidence") else 0.0

            if case.reference_answer is not None:
                correctness = 0.0 if abstained else self._judge.score(
                    case.query, generated, case.reference_answer
                )
            else:
                # proxy: hit + faithfulness; abstention / miss -> 0
                if abstained or not hit:
                    correctness = 0.0
                else:
                    correctness = faithfulness if faithfulness is not None else 0.5

            results.append(
                QACaseResult(
                    query=case.query,
                    generated_text=generated,
                    abstained=abstained,
                    correctness=correctness,
                    faithfulness=faithfulness,
                    confidence=confidence,
                    retrieved_ids=retrieved_ids,
                    relevant_block_ids=relevant,
                    hit=hit,
                    tags=list(case.tags),
                )
            )
            rankings.append(retrieved_hits)
            ground_truth.append(relevant)

        n = len(results)
        if n == 0:
            return QAEvalReport(case_count=0)
        faith_values = [r.faithfulness for r in results if r.faithfulness is not None]
        retrieval = evaluate_retrieval(rankings, ground_truth, k_values=(1, 3, k))
        return QAEvalReport(
            case_count=n,
            mean_correctness=sum(r.correctness for r in results) / n,
            mean_faithfulness=(sum(faith_values) / len(faith_values)) if faith_values else 0.0,
            mean_confidence=sum(r.confidence for r in results) / n,
            abstention_rate=sum(1 for r in results if r.abstained) / n,
            retrieval=retrieval,
            results=results,
        )


__all__ = [
    "DEFAULT_EVAL_K",
    "CorrectnessJudge",
    "LLMCorrectnessJudge",
    "QAEvaluator",
    "TokenOverlapJudge",
    "token_f1",
]
