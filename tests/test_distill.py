"""Tests for the Distill de-duplication pipeline.

All tests run fully offline and dependency-free: clustering/pairing uses
hand-built unit vectors and the deterministic :class:`FakeEmbeddingClient`; the
LLM merge step uses :class:`FakeLLMClient` scripted with valid merge JSON. They
exercise the whole chain -- find_similar_pairs -> cluster -> merge -> lifecycle
write-back -- end-to-end.
"""

from __future__ import annotations

import json

import pytest

from sparksage import (
    BlockEmbedder,
    BlockStatus,
    FakeEmbeddingClient,
    FakeLLMClient,
    IdeaBlock,
)
from sparksage.distill import (
    BlockMerger,
    Cluster,
    ClusteringBackend,
    ConnectedComponentsBackend,
    DistillPipeline,
    MergeCoercionError,
    MergeEmptyResponseError,
    MergeError,
    partition_by_strongest_edges,
)
from sparksage.distill.cluster import _UnionFind
from sparksage.distill.prompts import merge_messages, merge_system_prompt
from sparksage.distill.schema import (
    RawMergedBlock,
    coerce_merged_block,
    parse_raw_merged,
)
from sparksage.schema.enums import Tag


# ---------------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------------- #
def _norm(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def _make_block(
    name: str = "Block",
    question: str = "What is this?",
    answer: str = "A short verified answer.",
) -> IdeaBlock:
    return IdeaBlock(name=name, critical_question=question, trusted_answer=answer)


def _merge_json(
    *,
    name: str = "Canonical",
    question: str = "What is the canonical answer?",
    answer: str = "A merged, concise, verified answer.",
    tags: list[str] | None = None,
    keywords: list[str] | None = None,
    reasoning: str = "merged duplicates",
) -> str:
    return json.dumps(
        {
            "name": name,
            "critical_question": question,
            "trusted_answer": answer,
            "tags": tags if tags is not None else ["IMPORTANT"],
            "entities": [
                {"entity_name": "SparkSage", "entity_type": "PRODUCT", "aliases": ["ss"]}
            ],
            "keywords": keywords if keywords is not None else ["merged", "dedup"],
            "reasoning": reasoning,
        }
    )


# ---------------------------------------------------------------------------- #
# UnionFind
# ---------------------------------------------------------------------------- #
class TestUnionFind:
    def test_singletons(self):
        uf = _UnionFind(["a", "b", "c"])
        assert uf.component_count() == 3
        assert sorted(uf.components()) == [["a"], ["b"], ["c"]]

    def test_union_merges(self):
        uf = _UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        assert uf.component_count() == 2
        comps = uf.components()
        assert ["a", "b"] in comps
        assert ["c"] in comps

    def test_union_transitive(self):
        uf = _UnionFind(["a", "b", "c", "d"])
        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("c", "d")
        assert uf.component_count() == 1
        assert uf.components() == [["a", "b", "c", "d"]]

    def test_union_idempotent(self):
        uf = _UnionFind(["a", "b"])
        uf.union("a", "b")
        before = uf.component_count()
        uf.union("a", "b")
        assert uf.component_count() == before


# ---------------------------------------------------------------------------- #
# ConnectedComponentsBackend
# ---------------------------------------------------------------------------- #
class TestConnectedComponentsBackend:
    def test_empty(self):
        backend = ConnectedComponentsBackend()
        assert backend.cluster({}, threshold=0.5) == []

    def test_singletons_when_no_pairs(self):
        backend = ConnectedComponentsBackend()
        vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}  # orthogonal
        clusters = backend.cluster(vectors, threshold=0.5)
        assert len(clusters) == 2
        assert all(c.size == 1 for c in clusters)

    def test_cluster_of_duplicates(self):
        backend = ConnectedComponentsBackend()
        vectors = {
            "a": [1.0, 0.0],
            "b": [1.0, 0.0],  # identical -> dot 1.0
            "c": [0.0, 1.0],  # orthogonal
        }
        clusters = backend.cluster(vectors, threshold=0.5)
        by_first = {c.members[0]: c for c in clusters}
        assert set(by_first["a"].members) == {"a", "b"}
        assert by_first["a"].confidence == pytest.approx(1.0)
        assert by_first["c"].size == 1
        assert by_first["c"].confidence == 1.0

    def test_transitive_chaining(self):
        backend = ConnectedComponentsBackend()
        # a~b, b~c, but a not directly ~ c at this threshold -> still one cluster
        vectors = {
            "a": _norm([1.0, 0.0, 0.0]),
            "b": _norm([0.9, 0.4, 0.0]),
            "c": _norm([0.0, 0.9, 0.4]),
        }
        clusters = backend.cluster(vectors, threshold=0.0)
        assert len(clusters) == 1
        assert set(clusters[0].members) == {"a", "b", "c"}

    def test_threshold_controls_clusters(self):
        backend = ConnectedComponentsBackend()
        vectors = {"a": [1.0, 0.0], "b": [0.7, 0.71]}  # dot ~ 0.7
        loose = backend.cluster(vectors, threshold=0.5)
        assert len(loose) == 1
        tight = backend.cluster(vectors, threshold=0.9)
        assert len(tight) == 2

    def test_min_cluster_size_validation(self):
        with pytest.raises(ValueError):
            ConnectedComponentsBackend(min_cluster_size=0)

    def test_implements_protocol(self):
        backend = ConnectedComponentsBackend()
        assert isinstance(backend, ClusteringBackend)


# ---------------------------------------------------------------------------- #
# partition_by_strongest_edges
# ---------------------------------------------------------------------------- #
class TestPartitionByStrongestEdges:
    def test_empty_members(self):
        assert partition_by_strongest_edges([], [], 3) == []

    def test_all_singletons_when_no_edges(self):
        members = ["a", "b", "c"]
        result = partition_by_strongest_edges(members, [], 2)
        assert sorted(members) == sorted([m for g in result for m in g])
        assert len(result) <= 2

    def test_caps_at_n_groups(self):
        from sparksage.embed.similarity import SimilarityPair

        members = ["a", "b", "c", "d", "e"]
        pairs = [
            SimilarityPair(a="a", b="b", score=0.9),
            SimilarityPair(a="b", b="c", score=0.8),
            SimilarityPair(a="c", b="d", score=0.7),
            SimilarityPair(a="d", b="e", score=0.6),
        ]
        result = partition_by_strongest_edges(members, pairs, 2)
        assert len(result) <= 2
        flat = sorted(m for g in result for m in g)
        assert flat == ["a", "b", "c", "d", "e"]

    def test_ignores_edges_outside_members(self):
        from sparksage.embed.similarity import SimilarityPair

        members = ["a", "b"]
        pairs = [
            SimilarityPair(a="a", b="b", score=0.9),
            SimilarityPair(a="c", b="d", score=0.99),  # outside members
        ]
        result = partition_by_strongest_edges(members, pairs, 1)
        assert sorted(m for g in result for m in g) == ["a", "b"]

    def test_n_groups_clamped(self):
        members = ["a", "b"]
        result = partition_by_strongest_edges(members, [], 10)
        assert len(result) <= 2


# ---------------------------------------------------------------------------- #
# schema: parse_raw_merged + coerce_merged_block
# ---------------------------------------------------------------------------- #
class TestSchemaParse:
    def test_parse_direct_object(self):
        raw = parse_raw_merged({"name": "X", "critical_question": "q?"})
        assert isinstance(raw, RawMergedBlock)
        assert raw.name == "X"

    def test_parse_block_envelope(self):
        raw = parse_raw_merged({"block": {"name": "Y", "critical_question": "q?"}})
        assert raw.name == "Y"

    def test_parse_merged_envelope(self):
        raw = parse_raw_merged({"merged": {"name": "Z", "critical_question": "q?"}})
        assert raw.name == "Z"

    def test_parse_rejects_non_object(self):
        with pytest.raises(MergeCoercionError):
            parse_raw_merged([1, 2, 3])  # type: ignore[arg-type]


class TestSchemaCoerce:
    def test_produces_active_block_with_parents(self):
        import uuid

        parents = [uuid.uuid4(), uuid.uuid4()]
        raw = RawMergedBlock(
            name="Canonical",
            critical_question="What?",
            trusted_answer="A concise merged answer.",
            tags=["IMPORTANT", "TECHNOLOGY"],
            keywords=["a", "b", "b"],
        )
        block = coerce_merged_block(raw, parents=parents, confidence=0.8)
        assert block.status == BlockStatus.ACTIVE
        assert block.parents == parents
        assert block.confidence == pytest.approx(0.8)
        assert Tag.IMPORTANT in block.tags
        assert Tag.TECHNOLOGY in block.tags

    def test_ensures_question_mark_non_strict(self):
        raw = RawMergedBlock(
            name="C", critical_question="what is it", trusted_answer="answer."
        )
        block = coerce_merged_block(raw, parents=[], confidence=1.0)
        assert block.critical_question.endswith("?")

    def test_drops_unknown_tags_non_strict(self):
        raw = RawMergedBlock(
            name="C",
            critical_question="q?",
            trusted_answer="a.",
            tags=["NOT_A_TAG", "IMPORTANT"],
        )
        block = coerce_merged_block(raw, parents=[], confidence=0.5)
        assert Tag.IMPORTANT in block.tags
        assert all(t != "NOT_A_TAG" for t in block.tags)

    def test_oversized_answer_raises(self):
        raw = RawMergedBlock(
            name="C",
            critical_question="q?",
            trusted_answer="x" * 600,
        )
        with pytest.raises(MergeCoercionError):
            coerce_merged_block(raw, parents=[], confidence=1.0)

    def test_confidence_clamped(self):
        raw = RawMergedBlock(name="C", critical_question="q?", trusted_answer="a.")
        block = coerce_merged_block(raw, parents=[], confidence=1.7)
        assert block.confidence == 1.0

    def test_empty_required_fields_raise(self):
        raw = RawMergedBlock(name="", critical_question="q?", trusted_answer="a.")
        with pytest.raises(MergeCoercionError):
            coerce_merged_block(raw, parents=[], confidence=1.0)


# ---------------------------------------------------------------------------- #
# prompts
# ---------------------------------------------------------------------------- #
class TestPrompts:
    def test_system_prompt_carries_vocab_and_limit(self):
        prompt = merge_system_prompt(3)
        assert "Tag" in prompt or "IMPORTANT" in prompt  # vocab injected
        assert "PRODUCT" in prompt  # entity types injected
        assert "500" in prompt  # answer cap injected

    def test_merge_messages_shape(self):
        blocks = [_make_block("A"), _make_block("B")]
        messages = merge_messages(blocks)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "A" in messages[1]["content"]
        assert "B" in messages[1]["content"]


# ---------------------------------------------------------------------------- #
# BlockMerger
# ---------------------------------------------------------------------------- #
class TestBlockMerger:
    def test_single_block_is_noop_promotion(self):
        merger = BlockMerger(FakeLLMClient(responses=[_merge_json()]))
        block = _make_block("Only")
        canonical = merger.merge_cluster([block], confidence=0.9)
        assert canonical.status == BlockStatus.ACTIVE
        assert canonical.parents == []
        assert canonical.confidence == pytest.approx(0.9)
        assert merger.merge_calls == 0  # no LLM call for a singleton

    def test_merges_cluster_into_canonical(self):
        merger = BlockMerger(FakeLLMClient(responses=[_merge_json()]))
        members = [
            _make_block("A", "q a?", "alpha answer."),
            _make_block("B", "q b?", "beta answer."),
        ]
        canonical = merger.merge_cluster(members, confidence=0.7)
        assert canonical.status == BlockStatus.ACTIVE
        assert canonical.parents == [m.id for m in members]
        assert canonical.confidence == pytest.approx(0.7)
        assert canonical.name == "Canonical"
        assert merger.merge_calls == 1

    def test_empty_cluster_rejected(self):
        merger = BlockMerger(FakeLLMClient())
        with pytest.raises(ValueError):
            merger.merge_cluster([])

    def test_empty_llm_response_falls_back_non_strict(self):
        merger = BlockMerger(FakeLLMClient(responses=[""]))
        members = [_make_block("A"), _make_block("B")]
        canonical = merger.merge_cluster(members, confidence=0.6)
        assert canonical.status == BlockStatus.ACTIVE
        assert canonical.parents == [members[1].id]  # fallback promotes first
        assert merger.fallbacks == 1

    def test_empty_llm_response_raises_strict(self):
        merger = BlockMerger(FakeLLMClient(responses=[""]), strict=True)
        with pytest.raises(MergeEmptyResponseError):
            merger.merge_cluster([_make_block("A"), _make_block("B")])

    def test_invalid_json_falls_back_non_strict(self):
        merger = BlockMerger(FakeLLMClient(responses=["not json at all"]))
        members = [_make_block("A"), _make_block("B")]
        canonical = merger.merge_cluster(members, confidence=0.5)
        assert canonical.status == BlockStatus.ACTIVE
        assert merger.fallbacks == 1

    def test_invalid_json_raises_strict(self):
        merger = BlockMerger(FakeLLMClient(responses=["not json"]), strict=True)
        with pytest.raises(MergeError):
            merger.merge_cluster([_make_block("A"), _make_block("B")])

    def test_oversized_merged_answer_falls_back_non_strict(self):
        oversized = _merge_json(answer="z" * 600)
        merger = BlockMerger(FakeLLMClient(responses=[oversized]))
        members = [_make_block("A"), _make_block("B")]
        canonical = merger.merge_cluster(members, confidence=0.5)
        assert canonical.status == BlockStatus.ACTIVE
        assert merger.fallbacks == 1

    def test_inherits_source_from_members(self):
        from sparksage.schema.source import SourceRef

        src = SourceRef(uri="file://doc.md", title="Doc")
        member = _make_block("A")
        member.source = src
        merger = BlockMerger(FakeLLMClient(responses=[_merge_json()]))
        canonical = merger.merge_cluster([member, _make_block("B")])
        assert canonical.source == src


# ---------------------------------------------------------------------------- #
# DistillPipeline: end-to-end
# ---------------------------------------------------------------------------- #
class TestDistillPipeline:
    def _build(
        self,
        llm_responses: list[str] | None = None,
        **kwargs: object,
    ) -> DistillPipeline:
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=128))
        merger = BlockMerger(FakeLLMClient(responses=llm_responses or [_merge_json()]))
        return DistillPipeline(embedder=embedder, merger=merger, **kwargs)  # type: ignore[arg-type]

    def test_empty_corpus(self):
        pipe = self._build()
        result = pipe.run([])
        assert result.survivors == []
        assert result.merged_out == []
        assert result.stats.input_blocks == 0
        assert result.reduction == 0.0

    def test_no_duplicates_passes_through(self):
        pipe = self._build(start_threshold=0.99)
        blocks = [
            _make_block("A", "what is alpha?", "alpha alpha alpha"),
            _make_block("B", "what is bravo?", "bravo bravo bravo"),
        ]
        result = pipe.run(blocks)
        assert len(result.survivors) == 2
        assert result.merged_out == []
        assert result.reduction == 0.0
        # all survivors ACTIVE
        assert all(b.status == BlockStatus.ACTIVE for b in result.survivors)

    def test_merges_obvious_duplicates(self):
        # two near-identical blocks + one disjoint
        pipe = self._build(start_threshold=0.5, max_iterations=1)
        blocks = [
            _make_block("Deploy1", "how to deploy?", "deploy sparksage locally fast"),
            _make_block("Deploy2", "how to run?", "deploy sparksage locally fast now"),
            _make_block("Cook", "how to bake?", "chocolate cake recipe sugar eggs"),
        ]
        result = pipe.run(blocks)
        assert len(result.survivors) == 2  # one canonical + the cook singleton
        assert len(result.merged_out) == 2
        assert all(b.status == BlockStatus.MERGED for b in result.merged_out)
        assert all(b.status == BlockStatus.ACTIVE for b in result.survivors)
        assert result.reduction > 0.0
        # the canonical survivor records the two deploy blocks as parents
        canonical = next(
            b for b in result.survivors if b.parents
        )
        assert len(canonical.parents) == 2
        merged_ids = {b.id for b in result.merged_out}
        assert set(canonical.parents) == merged_ids

    def test_does_not_mutate_input_blocks(self):
        pipe = self._build(start_threshold=0.5, max_iterations=1)
        block = _make_block("A", "q a?", "alpha alpha alpha")
        original_status = block.status
        pipe.run([block, _make_block("B", "q b?", "alpha alpha alpha beta")])
        assert block.status == original_status

    def test_input_blocks_not_mutated_on_merge(self):
        pipe = self._build(start_threshold=0.5, max_iterations=1)
        a = _make_block("A", "q a?", "alpha alpha alpha")
        b = _make_block("B", "q b?", "alpha alpha alpha")
        pipe.run([a, b])
        assert a.status != BlockStatus.MERGED
        assert b.status != BlockStatus.MERGED

    def test_iterative_threshold_records_snapshots(self):
        pipe = self._build(start_threshold=0.95, threshold_step=0.01, max_iterations=3)
        blocks = [
            _make_block("A", "q a?", "alpha alpha alpha"),
            _make_block("B", "q b?", "beta beta beta"),
        ]
        result = pipe.run(blocks)
        # at 0.95 these disjoint blocks never cluster -> iterations stop early
        assert len(result.stats.iterations) >= 1
        for snap in result.stats.iterations:
            assert snap.merge_clusters == 0
            assert snap.canonical_emitted == 0

    def test_stats_track_llm_calls_and_fallbacks(self):
        pipe = self._build(
            llm_responses=[_merge_json()],
            start_threshold=0.5,
            max_iterations=1,
        )
        blocks = [
            _make_block("A", "q a?", "alpha alpha alpha"),
            _make_block("B", "q b?", "alpha alpha alpha"),
        ]
        result = pipe.run(blocks)
        assert result.stats.llm_merge_calls >= 1
        assert result.stats.input_blocks == 2

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            self._build(start_threshold=1.5)
        with pytest.raises(ValueError):
            self._build(threshold_step=-0.1)
        with pytest.raises(ValueError):
            self._build(max_iterations=0)
        with pytest.raises(ValueError):
            self._build(min_cluster_size=1)
        with pytest.raises(TypeError):
            DistillPipeline(
                embedder=BlockEmbedder(FakeEmbeddingClient()),
                merger=BlockMerger(FakeLLMClient()),
                start_threshold=True,  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------- #
# DistillPipeline: hierarchical merge
# ---------------------------------------------------------------------------- #
class TestHierarchicalMerge:
    def test_large_cluster_collapses_to_one_canonical(self):
        # 25 near-identical blocks -> exceeds max_cluster_size, must still
        # collapse to exactly one canonical survivor via hierarchical merge.
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        # every merge call returns one valid canonical block
        responses = [_merge_json(name=f"C{i}") for i in range(50)]
        merger = BlockMerger(FakeLLMClient(responses=responses))
        pipe = DistillPipeline(
            embedder=embedder,
            merger=merger,
            start_threshold=0.3,
            max_iterations=1,
            max_cluster_size=5,
        )
        blocks = [
            _make_block(f"B{i}", f"q {i}?", "alpha alpha alpha beta gamma") for i in range(25)
        ]
        result = pipe.run(blocks)
        assert len(result.survivors) == 1
        assert all(b.status == BlockStatus.MERGED for b in result.merged_out)
        assert len(result.merged_out) == 25

    def test_mixed_corpus_large_cluster_plus_singletons(self):
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        responses = [_merge_json() for _ in range(20)]
        merger = BlockMerger(FakeLLMClient(responses=responses))
        pipe = DistillPipeline(
            embedder=embedder,
            merger=merger,
            start_threshold=0.3,
            max_iterations=1,
            max_cluster_size=4,
        )
        cluster_blocks = [
            _make_block(f"C{i}", f"q c{i}?", "alpha alpha alpha shared") for i in range(10)
        ]
        singleton_blocks = [
            _make_block("S1", "what is zeta?", "zeta zeta zeta"),
            _make_block("S2", "what is eta?", "eta eta eta"),
        ]
        result = pipe.run(cluster_blocks + singleton_blocks)
        # The large cluster collapses via hierarchical merge; survivors are
        # strictly fewer than inputs, every merged-out block is MERGED, and at
        # least one survivor is a canonical (carries parents).
        assert len(result.survivors) < len(cluster_blocks) + len(singleton_blocks)
        assert len(result.merged_out) > 0
        assert all(b.status == BlockStatus.MERGED for b in result.merged_out)
        assert all(b.status == BlockStatus.ACTIVE for b in result.survivors)
        assert any(b.parents for b in result.survivors)
        assert result.reduction > 0.0


# ---------------------------------------------------------------------------- #
# DistillPipeline: accepts injected ClusteringBackend
# ---------------------------------------------------------------------------- #
class TestCustomBackend:
    def test_injected_backend_is_used(self):
        calls: list[dict] = []

        class Spy:
            def cluster(
                self,
                vectors: dict[str, list[float]],
                *,
                threshold: float = 0.5,
            ) -> list[Cluster]:
                calls.append({"n": len(vectors), "threshold": threshold})
                return [Cluster(members=(bid,), confidence=1.0) for bid in vectors]

        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=32))
        merger = BlockMerger(FakeLLMClient(responses=[_merge_json()]))
        pipe = DistillPipeline(
            embedder=embedder,
            merger=merger,
            clustering_backend=Spy(),  # type: ignore[arg-type]
            start_threshold=0.5,
            max_iterations=1,
        )
        blocks = [_make_block("A"), _make_block("B")]
        result = pipe.run(blocks)
        assert len(result.survivors) == 2  # spy never clusters -> no merges
        assert len(calls) == 1
