"""Measurable benchmark suite: IdeaBlock vs naive chunking on your own data.

The benchmark answers the adoption-blocking question "is the question-aligned
IdeaBlock design actually better than the traditional recursive-character
splitter everyone uses?" -- *measurably, on your own corpus*.

It reuses the existing :class:`~sparksage.embed.BlockEmbedder` and
:class:`~sparksage.embed.store.InMemoryVectorStore`, so it is pure stdlib + the
embedding client you already have (and runs offline with
:class:`~sparksage.embed.FakeEmbeddingClient`). The only thing it adds is:

* :class:`RecursiveCharSplitter` -- a faithful, dependency-free reimplementation
  of the LangChain recursive splitter, used as the baseline;
* :class:`BenchmarkRunner` -- builds two indexes over the *same* IdeaBlock
  corpus (one vector per block vs one vector per naive chunk), runs the *same*
  queries (each block's ``critical_question``, ground truth = the block itself)
  against both, and scores top-k retrieval (hit@k, MRR) + token efficiency;
* :class:`BenchmarkReport` -- side-by-side metrics + improvement factors, with a
  zero-dependency :meth:`~sparksage.bench.report.BenchmarkReport.to_html`
  renderer so the ROI can be shared as a single self-contained file.

The comparison is fair by construction: same embedder, same queries, same ground
truth -- only the chunking strategy differs.
"""

from sparksage.bench.baselines import (
    DEFAULT_SEPARATORS,
    Chunk,
    RecursiveCharSplitter,
)
from sparksage.bench.metrics import (
    RetrievalMetrics,
    TokenStats,
    approx_tokens,
    evaluate_retrieval,
    token_stats,
)
from sparksage.bench.report import BenchmarkReport, StrategyReport
from sparksage.bench.runner import (
    DEFAULT_CHARS_PER_TOKEN,
    DEFAULT_GROWTH_FACTOR,
    DEFAULT_K_VALUES,
    BenchmarkConfig,
    BenchmarkRunner,
)
from sparksage.bench.scaling import ScalingReport, TierResult

__all__ = [
    "BenchmarkConfig",
    "BenchmarkReport",
    "BenchmarkRunner",
    "Chunk",
    "DEFAULT_CHARS_PER_TOKEN",
    "DEFAULT_GROWTH_FACTOR",
    "DEFAULT_K_VALUES",
    "DEFAULT_SEPARATORS",
    "RecursiveCharSplitter",
    "RetrievalMetrics",
    "ScalingReport",
    "StrategyReport",
    "TierResult",
    "TokenStats",
    "approx_tokens",
    "evaluate_retrieval",
    "token_stats",
]
