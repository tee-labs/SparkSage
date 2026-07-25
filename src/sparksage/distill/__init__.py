"""Distill: end-to-end near-duplicate detection, clustering, and LLM merge.

Distill collapses a corpus of near-duplicate IdeaBlocks into a smaller set of
canonical, more complete blocks. It is the deduplication stage promised in the
SparkSage roadmap, and it is built on the existing building blocks rather than
introducing new ones:

* candidate detection reuses
  :func:`~sparksage.embed.similarity.find_similar_pairs` -- now with an
  optional :class:`~sparksage.embed.similarity.CandidateReducer`. For
  million-vector corpora the pure-stdlib :class:`LSHCandidateReducer` (random
  hyperplane LSH) replaces the ``O(n^2 * d)`` all-pairs scan with a cheap
  candidate-then-verify pass while keeping precision ``1.0``;
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

For long-running distill runs (minutes on 10k blocks, hours on a million),
:class:`DistillJob` wraps the pipeline in a pollable state machine
(``queued -> running -> success | failed | timeout | cancelled``) with
progress callbacks, and :class:`JobManager` is the in-process registry a
future ``/api/v1/distill`` route will wrap.

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
from sparksage.distill.job import (
    DEFAULT_JOB_TIMEOUT,
    DistillJob,
    JobManager,
    JobProgress,
    JobProgressCallback,
    JobSnapshot,
    JobStatus,
)
from sparksage.distill.lsh import (
    DEFAULT_NUM_HYPERPLANES,
    DEFAULT_NUM_TABLES,
    DEFAULT_SEED,
    LSH_ACTIVATION_THRESHOLD,
    LSHCandidateReducer,
    select_candidate_reducer,
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
    DistillProgress,
    DistillResult,
    DistillStats,
    ProgressCallback,
)
from sparksage.distill.prompts import merge_messages, merge_system_prompt
from sparksage.distill.schema import (
    MergeCoercionError,
    RawMergedBlock,
    coerce_merged_block,
    parse_raw_merged,
)

__all__ = [
    "DEFAULT_JOB_TIMEOUT",
    "DEFAULT_MAX_CLUSTER_SIZE",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_THRESHOLD",
    "DEFAULT_MIN_CLUSTER_SIZE",
    "DEFAULT_NUM_HYPERPLANES",
    "DEFAULT_NUM_TABLES",
    "DEFAULT_SEED",
    "DEFAULT_START_THRESHOLD",
    "DEFAULT_THRESHOLD_STEP",
    "DistillIteration",
    "DistillJob",
    "DistillPipeline",
    "DistillProgress",
    "DistillResult",
    "DistillStats",
    "JobManager",
    "JobProgress",
    "JobProgressCallback",
    "JobSnapshot",
    "JobStatus",
    "LSH_ACTIVATION_THRESHOLD",
    "LOUVAIN_THRESHOLD",
    "LSHCandidateReducer",
    "BlockMerger",
    "Cluster",
    "ClusteringBackend",
    "ConnectedComponentsBackend",
    "LouvainClusteringBackend",
    "MergeCoercionError",
    "MergeEmptyResponseError",
    "MergeError",
    "MergeResponseParseError",
    "ProgressCallback",
    "RawMergedBlock",
    "coerce_merged_block",
    "merge_messages",
    "merge_system_prompt",
    "parse_raw_merged",
    "partition_by_strongest_edges",
    "select_candidate_reducer",
    "select_clustering_backend",
]
