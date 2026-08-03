"""Benchmark runner: IdeaBlock vs naive chunking, same queries, same embedder.

:class:`BenchmarkRunner` is the framework-agnostic core of the benchmark suite.
Given an :class:`~sparksage.embed.BlockEmbedder` and a corpus of IdeaBlocks, it
builds two retrieval indexes over the *same* underlying content --

1. an **IdeaBlock** index (one vector per block, embedding
   :attr:`~sparksage.schema.IdeaBlock.embedding_text`), and
2. a **naive baseline** index (the block corpus run through a
   :class:`~sparksage.bench.baselines.RecursiveCharSplitter`, one vector per
   chunk)

-- runs the *same* set of queries (each block's ``critical_question``, ground
truth = the block itself) against both, and returns a
:class:`~sparksage.bench.report.BenchmarkReport` with the side-by-side metrics.

This is the "prove the ROI on your own data" primitive: same embedder, same
queries, same ground truth -- only the chunking strategy differs, so the
comparison is fair and fully automatic (no human relevance judgments needed).

The runner depends only on the existing
:class:`~sparksage.embed.store.InMemoryVectorStore` and
:class:`~sparksage.embed.BlockEmbedder`, so it is pure stdlib + the embedding
client you already have, and runs offline with
:class:`~sparksage.embed.FakeEmbeddingClient`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sparksage.bench.baselines import Chunk, RecursiveCharSplitter
from sparksage.bench.metrics import (
    TokenStats,
    evaluate_retrieval,
    token_stats,
)
from sparksage.bench.report import BenchmarkReport, StrategyReport
from sparksage.bench.scaling import ScalingReport, TierResult
from sparksage.embed.indexer import BlockEmbedder
from sparksage.embed.store import InMemoryVectorStore, SearchHit
from sparksage.schema.ideablock import IdeaBlock

#: Default ``k`` cutoffs reported by the runner.
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5)

#: Default chars-per-token for the token-efficiency estimate.
DEFAULT_CHARS_PER_TOKEN: float = 4.0

#: Default per-tier growth factor for the scaling staircase (paper §3.3).
DEFAULT_GROWTH_FACTOR: float = 1.25


@dataclass
class BenchmarkConfig:
    """Snapshot of the runner's knobs, stored on the report for reproducibility.

    Attributes
    ----------
    chunk_size, chunk_overlap:
        Baseline splitter parameters.
    dimension:
        Embedding dimensionality used.
    k_values:
        ``k`` cutoffs evaluated.
    chars_per_token:
        Token-estimate heuristic ratio.
    splitter:
        Name of the baseline splitter class.
    """

    chunk_size: int = 0
    chunk_overlap: int = 0
    dimension: int = 0
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
    splitter: str = "RecursiveCharSplitter"

    def as_dict(self) -> dict[str, object]:
        return {
            "splitter": self.splitter,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_dimension": self.dimension,
            "k_values": list(self.k_values),
            "chars_per_token": self.chars_per_token,
        }


class BenchmarkRunner:
    """Compare IdeaBlock vs naive chunking retrieval on the same corpus.

    Parameters
    ----------
    embedder:
        A :class:`~sparksage.embed.BlockEmbedder` used to embed both the
        IdeaBlocks/chunks and the queries. Same client for both strategies is
        what keeps the comparison fair.
    splitter:
        Baseline chunker (default :class:`RecursiveCharSplitter` with
        ``chunk_size=400``).
    k_values:
        ``k`` cutoffs to report (default ``(1, 3, 5)``).
    search_k:
        How many hits to pull per query (default ``max(k_values)``). Larger
        values give a fuller ranking picture at a small search cost.
    chars_per_token:
        Heuristic ratio for the token estimate (default ``4.0``).
    token_counter:
        Optional ``(str) -> int`` for a real tokenizer. When given, it overrides
        ``chars_per_token`` for the IdeaBlock/baseline token stats. ``None``
        (default) uses the ``len / 4`` approximation.
    store_factory:
        Optional factory returning a fresh
        :class:`~sparksage.embed.store.InMemoryVectorStore` for a given
        dimension. Defaults to the brute-force store; override to benchmark a
        different :class:`~sparksage.embed.store.VectorStore` backend.

    Examples
    --------
    >>> from sparksage import BlockEmbedder, FakeEmbeddingClient
    >>> from sparksage.bench import BenchmarkRunner   # doctest: +SKIP
    >>> runner = BenchmarkRunner(                     # doctest: +SKIP
    ...     embedder=BlockEmbedder(FakeEmbeddingClient()),
    ... )
    >>> report = runner.run(blocks)                   # doctest: +SKIP
    >>> report.to_html()                              # doctest: +SKIP
    """

    def __init__(
        self,
        embedder: BlockEmbedder,
        *,
        splitter: RecursiveCharSplitter | None = None,
        k_values: tuple[int, ...] = DEFAULT_K_VALUES,
        search_k: int | None = None,
        chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
        token_counter: object | None = None,
        store_factory: object | None = None,
    ) -> None:
        if not k_values:
            raise ValueError("k_values must not be empty")
        self._embedder = embedder
        self._splitter = splitter if splitter is not None else RecursiveCharSplitter()
        self._k_values = tuple(k_values)
        self._search_k = int(search_k) if search_k is not None else max(self._k_values)
        if self._search_k < 1:
            raise ValueError("search_k must be >= 1")
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be > 0")
        self._chars_per_token = float(chars_per_token)
        self._token_counter = token_counter
        self._store_factory = store_factory

    @property
    def k_values(self) -> tuple[int, ...]:
        return self._k_values

    def run(self, blocks: list[IdeaBlock]) -> BenchmarkReport:
        """Run the benchmark and return a :class:`BenchmarkReport`.

        Builds both indexes, derives a query per block (its
        ``critical_question``), evaluates retrieval on both, and assembles the
        report. ``blocks`` must be non-empty.
        """
        if not blocks:
            raise ValueError("run() requires at least one IdeaBlock")

        dimension = self._embedder.dimension
        config = BenchmarkConfig(
            chunk_size=self._splitter.chunk_size,
            chunk_overlap=self._splitter.chunk_overlap,
            dimension=dimension,
            k_values=self._k_values,
            chars_per_token=self._chars_per_token,
            splitter=type(self._splitter).__name__,
        )

        ideablock_rankings, ideablock_ground_truth = self._evaluate_ideablocks(
            blocks, dimension
        )
        baseline_rankings, baseline_ground_truth, chunks = self._evaluate_baseline(
            blocks, dimension
        )

        ideablock_metrics = evaluate_retrieval(
            ideablock_rankings, ideablock_ground_truth, k_values=self._k_values
        )
        baseline_metrics = evaluate_retrieval(
            baseline_rankings, baseline_ground_truth, k_values=self._k_values
        )

        ideablock_texts = [b.embedding_text for b in blocks]
        baseline_texts = [c.text for c in chunks]

        ideablock_tokens = self._token_statistics(ideablock_texts)
        baseline_tokens = self._token_statistics(baseline_texts)

        report = BenchmarkReport(
            ideablock=StrategyReport(
                name="IdeaBlock", retrieval=ideablock_metrics, tokens=ideablock_tokens
            ),
            baseline=StrategyReport(
                name="Naive chunks", retrieval=baseline_metrics, tokens=baseline_tokens
            ),
            query_count=len(blocks),
            block_count=len(blocks),
            config=config.as_dict(),
        )
        return report

    def run_scaling(
        self,
        blocks: list[IdeaBlock],
        *,
        query_count: int | None = None,
        growth_factor: float = DEFAULT_GROWTH_FACTOR,
        min_tier_size: int | None = None,
        max_tiers: int | None = None,
    ) -> ScalingReport:
        """Run a nested-tier scaling staircase and return a :class:`ScalingReport`.

        This is the scaling counterpart to :meth:`run`: instead of a single
        A/B comparison it slices the *same* corpus into a nested staircase of
        increasingly large subsets (each tier grows by ``growth_factor``, the
        paper default of ``1.25``), keeps the **question set fixed** (the first
        ``query_count`` blocks' ``critical_question``), and runs the IdeaBlock
        vs naive-chunk comparison at every tier. This exposes *where* (at what
        corpus size) one strategy overtakes the other -- the scale-dependent
        crossover the single-tier benchmark cannot see.

        Parameters
        ----------
        blocks:
            The full corpus, ordered (the first ``query_count`` blocks become
            the fixed query set; tiers grow from the front).
        query_count:
            How many of the first blocks become the fixed query set. Defaults
            to a reasonable fraction of the corpus (capped so every tier has
            room to grow). Must be ``>= 1`` and ``<= len(blocks)``.
        growth_factor:
            Per-tier size multiplier (default ``1.25``, the paper §3.3 value).
            Must be ``> 1.0``.
        min_tier_size:
            Size of the smallest tier. Must be ``>= query_count`` (every tier
            must contain the full query set so ground truth is non-empty).
            Defaults to ``query_count``.
        max_tiers:
            Optional cap on the number of tiers evaluated.

        Examples
        --------
        >>> from sparksage import BlockEmbedder, FakeEmbeddingClient  # doctest: +SKIP
        >>> runner = BenchmarkRunner(                               # doctest: +SKIP
        ...     embedder=BlockEmbedder(FakeEmbeddingClient()),
        ... )
        >>> report = runner.run_scaling(blocks)                      # doctest: +SKIP
        >>> report.crossover_tier(metric="hit_at_1")                 # doctest: +SKIP
        """
        if not blocks:
            raise ValueError("run_scaling() requires at least one IdeaBlock")
        n = len(blocks)
        if growth_factor <= 1.0:
            raise ValueError("growth_factor must be > 1.0")
        if max_tiers is not None and max_tiers < 1:
            raise ValueError("max_tiers must be >= 1")

        if query_count is None:
            query_count = _default_query_count(n)
        if query_count < 1 or query_count > n:
            raise ValueError(
                f"query_count must be in [1, {n}], got {query_count}"
            )

        if min_tier_size is None:
            min_tier_size = query_count
        if min_tier_size < query_count:
            raise ValueError(
                f"min_tier_size ({min_tier_size}) must be >= query_count "
                f"({query_count}) so every tier contains the query set"
            )
        if min_tier_size > n:
            min_tier_size = n

        tier_sizes = _compute_tier_sizes(
            min_tier_size, n, growth_factor, max_tiers
        )
        query_blocks = blocks[:query_count]
        dimension = self._embedder.dimension

        config = BenchmarkConfig(
            chunk_size=self._splitter.chunk_size,
            chunk_overlap=self._splitter.chunk_overlap,
            dimension=dimension,
            k_values=self._k_values,
            chars_per_token=self._chars_per_token,
            splitter=type(self._splitter).__name__,
        )
        scaling_config = dict(config.as_dict())
        scaling_config["growth_factor"] = growth_factor
        scaling_config["min_tier_size"] = min_tier_size
        scaling_config["max_tiers"] = max_tiers

        tiers: list[TierResult] = []
        for i, size in enumerate(tier_sizes):
            corpus = blocks[:size]
            ib_rankings, ib_gt = self._evaluate_ideablocks(
                corpus, dimension, query_blocks=query_blocks
            )
            base_rankings, base_gt, chunks = self._evaluate_baseline(
                corpus, dimension, query_blocks=query_blocks
            )
            ib_metrics = evaluate_retrieval(
                ib_rankings, ib_gt, k_values=self._k_values
            )
            base_metrics = evaluate_retrieval(
                base_rankings, base_gt, k_values=self._k_values
            )
            ib_tokens = self._token_statistics(
                [b.embedding_text for b in corpus]
            )
            base_tokens = self._token_statistics([c.text for c in chunks])
            tiers.append(
                TierResult(
                    tier_index=i,
                    block_count=size,
                    ideablock=StrategyReport(
                        name="IdeaBlock", retrieval=ib_metrics, tokens=ib_tokens
                    ),
                    baseline=StrategyReport(
                        name="Naive chunks",
                        retrieval=base_metrics,
                        tokens=base_tokens,
                    ),
                )
            )

        return ScalingReport(
            tiers=tiers,
            query_count=query_count,
            growth_factor=growth_factor,
            config=scaling_config,
        )

    # ------------------------------------------------------------------ #
    # strategy evaluation
    # ------------------------------------------------------------------ #
    def _evaluate_ideablocks(
        self,
        blocks: list[IdeaBlock],
        dimension: int,
        *,
        query_blocks: list[IdeaBlock] | None = None,
    ) -> tuple[list[list[SearchHit]], list[set[str]]]:
        """Index one vector per corpus block; query the given query blocks.

        When ``query_blocks`` is omitted it defaults to ``blocks`` (every block
        queries itself), preserving the original single-tier behaviour. The
        scaling runner passes a fixed query subset so the same questions are
        asked against an increasingly large background corpus.
        """
        queries = query_blocks if query_blocks is not None else blocks
        store = self._new_store(dimension)
        vectors = self._embedder.vectors_for(blocks)
        store.add_many(vectors)

        query_vecs = self._embedder.embed_texts([b.critical_question for b in queries])
        rankings = [store.search(qv, k=self._search_k) for qv in query_vecs]
        ground_truth = [{str(b.id)} for b in queries]
        return rankings, ground_truth

    def _evaluate_baseline(
        self,
        blocks: list[IdeaBlock],
        dimension: int,
        *,
        query_blocks: list[IdeaBlock] | None = None,
    ) -> tuple[list[list[SearchHit]], list[set[str]], list[Chunk]]:
        """Index one vector per naive chunk; map each query to its origin chunks.

        When ``query_blocks`` is omitted it defaults to ``blocks``. The ground
        truth for each query block is the set of chunk ids derived from that
        block (looked up in the corpus-wide block->chunk map), so it stays
        non-empty as long as the query block is in the corpus.
        """
        queries = query_blocks if query_blocks is not None else blocks
        chunks = self._splitter.split_blocks(blocks)
        store = self._new_store(dimension)
        if chunks:
            chunk_vectors = self._embedder.embed_texts([c.text for c in chunks])
            for chunk, vector in zip(chunks, chunk_vectors, strict=True):
                store.add(chunk.id, list(vector))

        block_id_to_chunk_ids: dict[str, set[str]] = {str(b.id): set() for b in blocks}
        for chunk in chunks:
            if chunk.source_ref_id and chunk.source_ref_id in block_id_to_chunk_ids:
                block_id_to_chunk_ids[chunk.source_ref_id].add(chunk.id)

        query_texts = [b.critical_question for b in queries]
        query_vecs = self._embedder.embed_texts(query_texts) if query_texts else []
        rankings = [store.search(qv, k=self._search_k) for qv in query_vecs]
        ground_truth = [
            block_id_to_chunk_ids.get(str(b.id), set()) for b in queries
        ]
        return rankings, ground_truth, chunks

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _new_store(self, dimension: int) -> InMemoryVectorStore:
        if self._store_factory is not None:
            candidate = self._store_factory(dimension)  # type: ignore[misc]
            if not isinstance(candidate, InMemoryVectorStore):
                raise TypeError("store_factory must return an InMemoryVectorStore")
            return candidate
        return InMemoryVectorStore(dimension=dimension)

    def _token_statistics(self, texts: list[str]) -> TokenStats:
        if self._token_counter is None:
            return token_stats(texts, chars_per_token=self._chars_per_token)
        counts = [int(self._token_counter(t)) for t in texts]  # type: ignore[misc]
        if not counts:
            return TokenStats()
        n = len(counts)
        total = sum(counts)
        total_chars = sum(len(t) for t in texts)
        return TokenStats(
            unit_count=n,
            total_chars=total_chars,
            avg_chars=total_chars / n if n else 0.0,
            total_tokens=float(total),
            avg_tokens=total / n if n else 0.0,
            chars_per_token=self._chars_per_token,
        )


__all__ = [
    "BenchmarkConfig",
    "BenchmarkRunner",
    "DEFAULT_CHARS_PER_TOKEN",
    "DEFAULT_GROWTH_FACTOR",
    "DEFAULT_K_VALUES",
]


def _default_query_count(n: int) -> int:
    """Pick a sensible default fixed-query-set size for an ``n``-block corpus.

    Targets roughly a third of the corpus so the smallest tier still has room
    to grow across multiple staircase steps, floored at 1 and capped at ``n``.
    """
    return max(1, min(n, n // 3 if n >= 3 else 1))


def _compute_tier_sizes(
    min_size: int,
    max_size: int,
    growth_factor: float,
    max_tiers: int | None,
) -> list[int]:
    """Build the nested-staircase tier sizes.

    Starts at ``min_size`` and multiplies by ``growth_factor`` (rounded up) at
    each step until ``max_size`` is reached or ``max_tiers`` is hit. The full
    corpus (``max_size``) is always the final tier: when the cap is reached
    before the growth reaches ``max_size``, the last tier is replaced with
    ``max_size`` so the user never loses the full-scale data point.
    Duplicates are de-duplicated while preserving order.
    """
    if min_size >= max_size:
        return [max_size]
    sizes: list[int] = []
    size = min_size
    cap = max_tiers if max_tiers is not None else float("inf")
    while len(sizes) < cap:
        sizes.append(min(size, max_size))
        if size >= max_size:
            break
        nxt = math.ceil(size * growth_factor)
        if nxt <= size:
            nxt = size + 1
        size = nxt
    if not sizes:
        sizes = [max_size]
    if sizes[-1] != max_size:
        sizes[-1] = max_size
    seen: set[int] = set()
    out: list[int] = []
    for s in sizes:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out
