"""Data models for end-to-end QA evaluation.

Where :mod:`sparksage.bench` evaluates *retrieval* in isolation (does the right
block surface?), :mod:`sparksage.eval` evaluates the *whole* QA pipeline (is
the generated answer correct?). A :class:`QATestCase` pairs a query with a
reference answer and/or the set of relevant block ids; the evaluator runs the
:class:`~sparksage.qa.QAEngine` over the test set and rolls the per-case
outcomes into a :class:`QAEvalReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sparksage.bench.metrics import RetrievalMetrics


@dataclass
class QATestCase:
    """One end-to-end QA evaluation case.

    Attributes
    ----------
    query:
        The user query.
    reference_answer:
        Optional gold answer for correctness scoring (LLM-judge or token
        overlap). When omitted, correctness falls back to a retrieval-hit +
        faithfulness proxy.
    relevant_block_ids:
        The block ids a correct retrieval should surface (ground truth for the
        retrieval-hit metric).
    tags:
        Optional case tags for slicing the report (e.g. ``["comparison"]``).
    """

    query: str
    reference_answer: str | None = None
    relevant_block_ids: set[str] = field(default_factory=set)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.relevant_block_ids = {str(b) for b in self.relevant_block_ids}


@dataclass
class QACaseResult:
    """The scored outcome of running one :class:`QATestCase` through the engine.

    Attributes
    ----------
    query:
        The evaluated query.
    generated_text:
        The answer text the engine surfaced (or the abstention reply).
    abstained:
        Whether the reader abstained.
    correctness:
        Answer-correctness score in ``[0, 1]`` (LLM-judge or token-overlap
        against the reference; retrieval+faithfulness proxy when no reference).
    faithfulness:
        The reader's faithfulness score, when known; else ``None``.
    confidence:
        The effective confidence the reader reported.
    retrieved_ids:
        Block ids the engine retrieved (best first), capped at the eval ``k``.
    relevant_block_ids:
        The case's ground-truth ids.
    hit:
        Whether any relevant block appeared in the retrieved top-``k``.
    tags:
        Forwarded from the case.
    """

    query: str
    generated_text: str = ""
    abstained: bool = False
    correctness: float = 0.0
    faithfulness: float | None = None
    confidence: float = 0.0
    retrieved_ids: list[str] = field(default_factory=list)
    relevant_block_ids: set[str] = field(default_factory=set)
    hit: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class QAEvalReport:
    """Aggregate quality report over a QA test set.

    Attributes
    ----------
    case_count:
        Number of cases evaluated.
    mean_correctness:
        Mean per-case :attr:`QACaseResult.correctness`.
    mean_faithfulness:
        Mean per-case faithfulness (``0.0`` when none judged).
    mean_confidence:
        Mean per-case effective confidence.
    abstention_rate:
        Fraction of cases where the reader abstained.
    retrieval:
        :class:`~sparksage.bench.RetrievalMetrics` over the retrieved top-``k``
        vs the relevant-block ground truth (hit@k / MRR), reusing the same
        metric the retrieval-only benchmark uses for comparability.
    results:
        The per-case :class:`QACaseResult` list (for slicing / debugging).
    """

    case_count: int = 0
    mean_correctness: float = 0.0
    mean_faithfulness: float = 0.0
    mean_confidence: float = 0.0
    abstention_rate: float = 0.0
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    results: list[QACaseResult] = field(default_factory=list)

    @property
    def correctness(self) -> float:
        """Alias for :attr:`mean_correctness` (the headline QA metric)."""
        return self.mean_correctness


__all__ = [
    "QACaseResult",
    "QAEvalReport",
    "QATestCase",
]
