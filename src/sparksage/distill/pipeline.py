"""End-to-end Distill de-duplication pipeline: detect, cluster, merge, write back.

:class:`DistillPipeline` is the Distill counterpart of the ingest orchestration
in :class:`~sparksage.api.SparkSageService`. It wires the existing building
blocks together:

    blocks
        -> BlockEmbedder.vectors_for()          (reuse the embed core)
        -> ClusteringBackend.cluster()          (near-duplicate clusters)
        -> BlockMerger.merge_cluster()          (LLM fusion -> canonical block)
        -> lifecycle write-back                 (MERGED parents, ACTIVE canonical)

and drives them with **iterative threshold refinement**: start permissive, merge
the obvious duplicates, then tighten the threshold and repeat on the survivors.
This finds chains of near-duplicates that a single pass would miss, while never
merging anything below the tightened bar.

The pipeline depends only on three protocols -- :class:`~sparksage.embed.EmbeddingClient`
(via :class:`~sparksage.embed.BlockEmbedder`), :class:`~sparksage.generator.LLMClient`
(via :class:`~sparksage.distill.merge.BlockMerger`) and
:class:`~sparksage.distill.cluster.ClusteringBackend` -- so it is fully
unit-testable offline with the deterministic fakes. ``numpy`` / ``faiss`` /
``python-louvain`` belong to the optional ``[distill]`` extra and are imported
lazily inside :class:`~sparksage.distill.cluster.LouvainClusteringBackend`.

Lifecycle write-back uses the schema fields that already exist for this purpose:
merged-away blocks get ``status=MERGED``; each canonical block gets
``status=ACTIVE``, ``parents`` = the merged UUIDs, and ``confidence`` = the
cluster's mean pairwise similarity. Nothing leaves the IdeaBlock data model.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from sparksage.distill.cluster import (
    ClusteringBackend,
    ConnectedComponentsBackend,
    partition_by_strongest_edges,
)
from sparksage.distill.merge import BlockMerger
from sparksage.embed.indexer import BlockEmbedder
from sparksage.embed.similarity import SimilarityPair, find_similar_pairs
from sparksage.schema.enums import BlockStatus
from sparksage.schema.ideablock import IdeaBlock

_logger = logging.getLogger(__name__)

#: Default starting similarity threshold for the first Distill iteration.
DEFAULT_START_THRESHOLD: float = 0.55

#: Default per-iteration threshold increment (tightens each round).
DEFAULT_THRESHOLD_STEP: float = 0.01

#: Default ceiling on the tightening threshold across iterations.
DEFAULT_MAX_THRESHOLD: float = 0.98

#: Default maximum number of refinement iterations (~4 rounds is where the
#: tightened threshold stops finding new duplicates on typical corpora).
DEFAULT_MAX_ITERATIONS: int = 4

#: Default minimum cluster size worth merging. Clusters of 1 are singletons and
#: pass through untouched.
DEFAULT_MIN_CLUSTER_SIZE: int = 2

#: Default per-LLM-call cluster size cap. Clusters larger than this are
#: hierarchically partitioned (strongest-edge sub-clustering), merged bottom-up,
#: then the canonical blocks are merged again -- so even a 10k-block cluster
#: never exceeds one LLM context worth of members per call.
DEFAULT_MAX_CLUSTER_SIZE: int = 20


# --------------------------------------------------------------------------- #
# Result dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DistillIteration:
    """Diagnostic snapshot of one refinement iteration.

    Attributes
    ----------
    iteration:
        1-based iteration index.
    threshold:
        Similarity threshold applied this round.
    active_blocks:
        Number of blocks in the active set entering this round.
    candidate_pairs:
        Number of near-duplicate pairs found at ``threshold``.
    merge_clusters:
        Number of clusters with >= ``min_cluster_size`` members (actually merged).
    blocks_merged:
        Number of blocks folded into canonical blocks this round.
    canonical_emitted:
        Number of canonical blocks emitted this round.
    """

    iteration: int
    threshold: float
    active_blocks: int
    candidate_pairs: int
    merge_clusters: int
    blocks_merged: int
    canonical_emitted: int


@dataclass
class DistillStats:
    """Aggregate counters across the whole Distill run.

    Attributes
    ----------
    input_blocks:
        Size of the original block list handed to :meth:`DistillPipeline.run`.
    iterations:
        :class:`DistillIteration` snapshot per round actually executed.
    llm_merge_calls:
        Number of LLM merge calls issued (one per non-singleton cluster, plus
        the recursion overhead of hierarchical merge).
    fallbacks:
        Number of times a merge call fell back to promoting the first member
        (non-strict mode only). Zero in strict mode unless errors were swallowed.
    """

    input_blocks: int = 0
    iterations: list[DistillIteration] = field(default_factory=list)
    llm_merge_calls: int = 0
    fallbacks: int = 0


@dataclass
class DistillResult:
    """The full outcome of a Distill run.

    Attributes
    ----------
    survivors:
        The live, de-duplicated corpus -- canonical merged blocks plus every
        singleton that was never near-duplicate of anything. All carry
        ``status=ACTIVE``. This is what a retriever should index.
    merged_out:
        Blocks folded into a canonical parent, carrying ``status=MERGED``. Kept
        for audit/rollback; their ``id`` appears in some survivor's ``parents``.
    stats:
        :class:`DistillStats` with per-iteration diagnostics.
    reduction:
        Convenience metric: ``1 - len(survivors) / input_blocks`` (fraction of
        the corpus de-duplicated away). ``0.0`` means nothing merged.
    """

    survivors: list[IdeaBlock] = field(default_factory=list)
    merged_out: list[IdeaBlock] = field(default_factory=list)
    stats: DistillStats = field(default_factory=DistillStats)
    reduction: float = 0.0


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class DistillPipeline:
    """Detect, cluster, and merge near-duplicate IdeaBlocks end-to-end.

    The pipeline runs **iterative threshold refinement**: it starts at
    ``start_threshold``, clusters and merges the obvious duplicates, then
    tightens the threshold by ``threshold_step`` and re-runs on the survivors,
    up to ``max_iterations`` times (or until no cluster of >=
    ``min_cluster_size`` remains). The canonical block emitted by each merge is
    re-embedded so the next iteration sees the *fused* content, not the members
    -- this is what lets chains of near-duplicates collapse correctly.

    Parameters
    ----------
    embedder:
        A :class:`~sparksage.embed.BlockEmbedder` used to vectorize the active
        set each iteration (and to re-embed freshly merged canonical blocks).
    merger:
        A :class:`BlockMerger` used to fuse each cluster into one canonical block.
    clustering_backend:
        How near-duplicate pairs become clusters. Defaults to
        :class:`~sparksage.distill.cluster.ConnectedComponentsBackend` (pure
        stdlib union-find). Swap in
        :class:`~sparksage.distill.cluster.LouvainClusteringBackend` (or a future
        FAISS/LSH backend) for million-vector corpora.
    start_threshold, threshold_step, max_threshold:
        Iterative-refinement schedule. Defaults follow the Distill convention
        (``0.55`` -> ``+0.01``/round -> cap ``0.98``).
    max_iterations:
        Hard cap on refinement rounds (default ``4``).
    min_cluster_size:
        Clusters strictly below this are left as singletons (default ``2``).
    max_cluster_size:
        Per-LLM-call budget. Larger clusters are hierarchically partitioned
        (strongest-edge sub-clustering) and merged bottom-up (default ``20``).

    Examples
    --------
    >>> from sparksage import BlockEmbedder, FakeEmbeddingClient, FakeLLMClient
    >>> from sparksage.distill import DistillPipeline, BlockMerger
    >>> pipe = DistillPipeline(                            # doctest: +SKIP
    ...     embedder=BlockEmbedder(FakeEmbeddingClient()),
    ...     merger=BlockMerger(FakeLLMClient()),
    ... )
    >>> result = pipe.run(blocks)                          # doctest: +SKIP
    >>> len(result.survivors) < len(blocks)                # doctest: +SKIP
    True
    """

    def __init__(
        self,
        embedder: BlockEmbedder,
        merger: BlockMerger,
        *,
        clustering_backend: ClusteringBackend | None = None,
        start_threshold: float = DEFAULT_START_THRESHOLD,
        threshold_step: float = DEFAULT_THRESHOLD_STEP,
        max_threshold: float = DEFAULT_MAX_THRESHOLD,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        max_cluster_size: int = DEFAULT_MAX_CLUSTER_SIZE,
    ) -> None:
        self._embedder = embedder
        self._merger = merger
        self._backend: ClusteringBackend = (
            clustering_backend
            if clustering_backend is not None
            else ConnectedComponentsBackend()
        )
        self._start_threshold = self._validate_threshold(start_threshold, "start_threshold")
        self._threshold_step = self._validate_step(threshold_step)
        self._max_threshold = self._validate_threshold(max_threshold, "max_threshold")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self._max_iterations = int(max_iterations)
        if min_cluster_size < 2:
            raise ValueError("min_cluster_size must be >= 2 (singletons are never merged)")
        self._min_cluster_size = int(min_cluster_size)
        if max_cluster_size < 1:
            raise ValueError("max_cluster_size must be >= 1")
        self._max_cluster_size = int(max_cluster_size)

    @staticmethod
    def _validate_threshold(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a float")
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        return value

    @staticmethod
    def _validate_step(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("threshold_step must be a float")
        value = float(value)
        if value < 0.0:
            raise ValueError("threshold_step must be >= 0")
        return value

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def run(self, blocks: list[IdeaBlock]) -> DistillResult:
        """Run iterative Distill de-duplication on ``blocks``.

        Returns a :class:`DistillResult`. The input list is *not* mutated --
        merged-away copies carry ``status=MERGED`` in ``result.merged_out``, and
        canonical/singleton survivors (all ``status=ACTIVE``) are in
        ``result.survivors``.
        """
        stats = DistillStats(input_blocks=len(blocks))
        if not blocks:
            return DistillResult(stats=stats, reduction=0.0)

        active: list[IdeaBlock] = [
            b.model_copy(update={"status": BlockStatus.ACTIVE}) for b in blocks
        ]
        merged_out: list[IdeaBlock] = []

        threshold = self._start_threshold
        calls_before = self._merger.merge_calls

        for i in range(1, self._max_iterations + 1):
            id_to_block = {str(b.id): b for b in active}
            vectors = self._embedder.vectors_for(active)
            pairs = find_similar_pairs(vectors, threshold=threshold)
            clusters = self._backend.cluster(vectors, threshold=threshold)

            merge_clusters = [c for c in clusters if c.size >= self._min_cluster_size]
            if not merge_clusters:
                stats.iterations.append(
                    DistillIteration(
                        iteration=i,
                        threshold=threshold,
                        active_blocks=len(active),
                        candidate_pairs=len(pairs),
                        merge_clusters=0,
                        blocks_merged=0,
                        canonical_emitted=0,
                    )
                )
                _logger.debug(
                    "distill iter %d: no cluster >= %d at threshold %.3f; stopping",
                    i,
                    self._min_cluster_size,
                    threshold,
                )
                break

            new_active: list[IdeaBlock] = []
            round_merged = 0

            for cluster in merge_clusters:
                members = [id_to_block[mid] for mid in cluster.members if mid in id_to_block]
                if len(members) < self._min_cluster_size:
                    continue
                canonical = self._merge_with_hierarchy(members, vectors, threshold, pairs)
                canonical = canonical.model_copy(
                    update={
                        "status": BlockStatus.ACTIVE,
                        "confidence": cluster.confidence,
                    }
                )
                for member in members:
                    merged_out.append(
                        member.model_copy(update={"status": BlockStatus.MERGED})
                    )
                round_merged += len(members)
                new_active.append(canonical)

            singleton_ids = {
                mid
                for c in clusters
                if c.size < self._min_cluster_size
                for mid in c.members
            }
            singletons = [id_to_block[mid] for mid in singleton_ids if mid in id_to_block]
            active = singletons + new_active

            stats.iterations.append(
                DistillIteration(
                    iteration=i,
                    threshold=threshold,
                    active_blocks=len(id_to_block),
                    candidate_pairs=len(pairs),
                    merge_clusters=len(merge_clusters),
                    blocks_merged=round_merged,
                    canonical_emitted=len(new_active),
                )
            )
            _logger.info(
                "distill iter %d: threshold=%.3f pairs=%d clusters=%d merged=%d -> %d canonical",
                i,
                threshold,
                len(pairs),
                len(merge_clusters),
                round_merged,
                len(new_active),
            )

            threshold = min(self._max_threshold, threshold + self._threshold_step)

        stats.llm_merge_calls = self._merger.merge_calls - calls_before
        stats.fallbacks = self._merger.fallbacks

        reduction = 0.0
        if stats.input_blocks > 0:
            reduction = 1.0 - (len(active) / stats.input_blocks)
        return DistillResult(
            survivors=active,
            merged_out=merged_out,
            stats=stats,
            reduction=reduction,
        )

    # ------------------------------------------------------------------ #
    # internals: hierarchical merge
    # ------------------------------------------------------------------ #
    def _merge_with_hierarchy(
        self,
        members: list[IdeaBlock],
        vectors: dict[str, list[float]],
        threshold: float,
        pairs: list[SimilarityPair],
    ) -> IdeaBlock:
        """Merge ``members``, partitioning hierarchically if the cluster is large.

        Clusters larger than ``max_cluster_size`` are recursively split into
        ``~sqrt(N)*2`` sub-groups by their strongest intra-cluster edges, each
        sub-group merged bottom-up, then the resulting canonical blocks merged
        again. This bounds every LLM call to ``<= max_cluster_size`` members
        while still collapsing the whole cluster into one canonical block.
        """
        if len(members) <= self._max_cluster_size:
            return self._merger.merge_cluster(members, confidence=1.0)

        canonical = self._hierarchical_merge(members, threshold, pairs, depth=0)
        return canonical

    def _hierarchical_merge(
        self,
        members: list[IdeaBlock],
        threshold: float,
        pairs: list[SimilarityPair],
        *,
        depth: int,
    ) -> IdeaBlock:
        n = len(members)
        if n <= self._max_cluster_size:
            return self._merger.merge_cluster(members, confidence=1.0)

        n_groups = max(2, int(math.isqrt(n) * 2))
        member_ids = [str(b.id) for b in members]
        groups = partition_by_strongest_edges(member_ids, pairs, n_groups)
        member_by_id = {str(b.id): b for b in members}

        canonicals: list[IdeaBlock] = []
        for group_ids in groups:
            group_blocks = [member_by_id[bid] for bid in group_ids if bid in member_by_id]
            if len(group_blocks) <= 1:
                canonicals.extend(group_blocks)
            elif len(group_blocks) <= self._max_cluster_size:
                canonicals.append(self._merger.merge_cluster(group_blocks, confidence=1.0))
            else:
                sub_vectors = self._embedder.vectors_for(group_blocks)
                sub_pairs = find_similar_pairs(sub_vectors, threshold=threshold)
                canonicals.append(
                    self._hierarchical_merge(
                        group_blocks, threshold, sub_pairs, depth=depth + 1
                    )
                )

        if len(canonicals) == 1:
            return canonicals[0]
        if len(canonicals) <= self._max_cluster_size:
            return self._merger.merge_cluster(canonicals, confidence=1.0)
        canonical_vectors = self._embedder.vectors_for(canonicals)
        canonical_pairs = find_similar_pairs(canonical_vectors, threshold=threshold)
        return self._hierarchical_merge(
            canonicals, threshold, canonical_pairs, depth=depth + 1
        )


__all__ = [
    "DEFAULT_MAX_CLUSTER_SIZE",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_THRESHOLD",
    "DEFAULT_MIN_CLUSTER_SIZE",
    "DEFAULT_START_THRESHOLD",
    "DEFAULT_THRESHOLD_STEP",
    "DistillIteration",
    "DistillPipeline",
    "DistillResult",
    "DistillStats",
]
