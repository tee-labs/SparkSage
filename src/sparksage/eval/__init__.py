"""End-to-end QA evaluation: answer correctness + retrieval + abstention.

This is the Phase-4 evaluation layer -- the answer-correctness counterpart to
:mod:`sparksage.bench` (which scores *retrieval* alone). :class:`QAEvaluator`
runs a :class:`~sparksage.qa.QAEngine` over a set of :class:`QATestCase`s and
rolls the per-case outcomes into a :class:`QAEvalReport`: mean answer
correctness, abstention rate, retrieval hit@k (reusing
:func:`~sparksage.bench.evaluate_retrieval` for comparability), and mean
faithfulness.

Correctness is a pluggable :class:`CorrectnessJudge`: the default
:class:`TokenOverlapJudge` is dependency-free token-F1 (fully offline);
:class:`LLMCorrectnessJudge` swaps in for semantic scoring, reusing the
existing :class:`~sparksage.generator.LLMClient`.

Together with :mod:`sparksage.feedback` (which closes the query->ingest loop),
this is the "long-term quality flywheel" the project roadmap specified: measure
end-to-end answer quality, capture user feedback, and feed self-healing signals
back to the corpus.

Example
-------
::

    from sparksage.eval import QAEvaluator, QATestCase

    evaluator = QAEvaluator(engine=engine)
    report = evaluator.run([
        QATestCase(query="how to deploy", reference_answer="pip install ...",
                   relevant_block_ids={block_id}),
    ])
    print(report.mean_correctness, report.abstention_rate)
"""

from sparksage.eval.evaluator import (
    DEFAULT_EVAL_K,
    CorrectnessJudge,
    LLMCorrectnessJudge,
    QAEvaluator,
    TokenOverlapJudge,
    token_f1,
)
from sparksage.eval.models import QACaseResult, QAEvalReport, QATestCase

__all__ = [
    "CorrectnessJudge",
    "DEFAULT_EVAL_K",
    "LLMCorrectnessJudge",
    "QACaseResult",
    "QAEvalReport",
    "QATestCase",
    "QAEvaluator",
    "TokenOverlapJudge",
    "token_f1",
]
