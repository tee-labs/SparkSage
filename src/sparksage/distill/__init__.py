"""Distill: end-to-end near-duplicate detection, clustering, and LLM merge.

Distill collapses a corpus of near-duplicate IdeaBlocks into a smaller set of
canonical, more complete blocks. It is the deduplication stage promised in the
SparkSage roadmap, and it is built on the existing building blocks rather than
introducing new ones:

* candidate detection reuses
  :func:`~sparksage.embed.similarity.find_similar_pairs`;
* clustering is a protocol (:class:`~sparksage.distill.cluster.ClusteringBackend`)
  with a pure-stdlib default (:class:`ConnectedComponentsBackend`) and an
  optional :class:`LouvainClusteringBackend` under ``[distill]``;
* the merge step reuses the existing :class:`~sparksage.generator.LLMClient`
  protocol via :class:`~sparksage.distill.merge.BlockMerger`;
* lifecycle write-back uses the schema fields that already exist for this
  purpose -- ``status`` / ``parents`` / ``confidence``.

:class:`~sparksage.distill.pipeline.DistillPipeline` ties them together with
**iterative threshold refinement** (start permissive, tighten each round) and
**hierarchical merge** (large clusters are partitioned by their strongest edges,
merged bottom-up, so every LLM call stays within the per-call budget).

The whole pipeline is fully unit-testable offline with
:class:`~sparksage.embed.FakeEmbeddingClient` and
:class:`~sparksage.generator.FakeLLMClient`.
"""

from sparksage.distill.cluster import (
    LOUVAIN_THRESHOLD,
    Cluster,
    ClusteringBackend,
    ConnectedComponentsBackend,
    LouvainClusteringBackend,
    partition_by_strongest_edges,
    select_clustering_backend,
)
from sparksage.distill.merge import (
    BlockMerger,
    MergeEmptyResponseError,
    MergeError,
    MergeResponseParseError,
)
from sparksage.distill.pipeline import (
    DEFAULT_MAX_CLUSTER_SIZE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_THRESHOLD,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_START_THRESHOLD,
    DEFAULT_THRESHOLD_STEP,
    DistillIteration,
    DistillPipeline,
    DistillResult,
    DistillStats,
)
from sparksage.distill.prompts import merge_messages, merge_system_prompt
from sparksage.distill.schema import (
    MergeCoercionError,
    RawMergedBlock,
    coerce_merged_block,
    parse_raw_merged,
)

__all__ = [
    "DEFAULT_MAX_CLUSTER_SIZE",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_THRESHOLD",
    "DEFAULT_MIN_CLUSTER_SIZE",
    "DEFAULT_START_THRESHOLD",
    "DEFAULT_THRESHOLD_STEP",
    "LOUVAIN_THRESHOLD",
    "BlockMerger",
    "Cluster",
    "ClusteringBackend",
    "ConnectedComponentsBackend",
    "DistillIteration",
    "DistillPipeline",
    "DistillResult",
    "DistillStats",
    "LouvainClusteringBackend",
    "MergeCoercionError",
    "MergeEmptyResponseError",
    "MergeError",
    "MergeResponseParseError",
    "RawMergedBlock",
    "coerce_merged_block",
    "merge_messages",
    "merge_system_prompt",
    "parse_raw_merged",
    "partition_by_strongest_edges",
    "select_clustering_backend",
]
