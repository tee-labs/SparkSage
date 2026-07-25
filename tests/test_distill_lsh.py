"""Tests for the LSH candidate reducer and its pipeline integration.

All tests are fully offline and dependency-free: they build unit vectors by
hand, run the pure-stdlib :class:`LSHCandidateReducer`, and verify the two
contract guarantees -- **precision is always 1.0** (no false positives survive
the exact dot-product verification in :func:`find_similar_pairs`) and **recall
is high** for tight near-duplicates (the only thing LSH can lose is true
positives). The pipeline-integration tests confirm the end-to-end Distill
result is unchanged when LSH is plugged in for a corpus of obvious duplicates.
"""

from __future__ import annotations

import math
import random

import pytest

from sparksage import (
    BlockEmbedder,
    BlockMerger,
    CandidateReducer,
    ConnectedComponentsBackend,
    DistillPipeline,
    FakeEmbeddingClient,
    FakeLLMClient,
    LSHCandidateReducer,
    find_similar_pairs,
    select_candidate_reducer,
)
from sparksage.distill.lsh import (
    DEFAULT_NUM_HYPERPLANES,
    DEFAULT_NUM_TABLES,
    DEFAULT_SEED,
    LSH_ACTIVATION_THRESHOLD,
)


# ---------------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------------- #
def _norm(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    if n == 0.0:
        return vec
    return [x / n for x in vec]


def _perturb(anchor: list[float], noise: float, rng: random.Random) -> list[float]:
    """A near-duplicate of ``anchor`` with Gaussian noise added then re-normalized."""
    return _norm([a + rng.gauss(0.0, noise) for a in anchor])


def _build_corpus(
    dim: int = 48,
    *,
    n_groups: int = 3,
    group_size: int = 4,
    n_singletons: int = 6,
    seed: int = 7,
    noise: float = 0.05,
) -> tuple[dict[str, list[float]], set[tuple[str, str]], set[tuple[str, str]]]:
    """Build a labelled corpus for recall/precision tests.

    Returns ``(vectors, within_group_pairs, cross_group_pairs)``:

    * ``vectors``: ``{id: unit_vector}``.
    * ``within_group_pairs``: every pair of ids from the same group -- these are
      the true near-duplicates a reducer should find.
    * ``cross_group_pairs``: every pair spanning two groups (or a group and a
      singleton) -- these should *not* be returned at a reasonable threshold.
    """
    rng = random.Random(seed)
    vectors: dict[str, list[float]] = {}
    groups: list[list[str]] = []
    for g in range(n_groups):
        anchor = _norm([rng.gauss(0.0, 1.0) for _ in range(dim)])
        members: list[str] = []
        for m in range(group_size):
            bid = f"g{g}m{m}"
            vectors[bid] = _perturb(anchor, noise, rng) if m > 0 else list(anchor)
            members.append(bid)
        groups.append(members)
    for s in range(n_singletons):
        bid = f"s{s}"
        vectors[bid] = _norm([rng.gauss(0.0, 1.0) for _ in range(dim)])
        groups.append([bid])  # treat as its own "group" for cross-pair bookkeeping

    within: set[tuple[str, str]] = set()
    cross: set[tuple[str, str]] = set()
    all_ids = list(vectors.keys())
    for i, a in enumerate(all_ids):
        for b in all_ids[i + 1 :]:
            pair = (a, b) if a <= b else (b, a)
            same = any(a in grp and b in grp for grp in groups if len(grp) > 1)
            (within if same else cross).add(pair)
    return vectors, within, cross


# ---------------------------------------------------------------------------- #
# constructor & validation
# ---------------------------------------------------------------------------- #
class TestConstructor:
    def test_defaults_exposed(self):
        r = LSHCandidateReducer()
        assert r.num_hyperplanes == DEFAULT_NUM_HYPERPLANES
        assert r.num_tables == DEFAULT_NUM_TABLES
        assert r.seed == DEFAULT_SEED

    def test_rejects_zero_hyperplanes(self):
        with pytest.raises(ValueError):
            LSHCandidateReducer(num_hyperplanes=0)

    def test_rejects_zero_tables(self):
        with pytest.raises(ValueError):
            LSHCandidateReducer(num_tables=0)

    def test_rejects_bool_params(self):
        with pytest.raises(TypeError):
            LSHCandidateReducer(num_hyperplanes=True)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            LSHCandidateReducer(num_tables=False)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            LSHCandidateReducer(seed=True)  # type: ignore[arg-type]

    def test_implements_candidate_reducer_protocol(self):
        r = LSHCandidateReducer()
        assert isinstance(r, CandidateReducer)


# ---------------------------------------------------------------------------- #
# candidate_pairs: shape & determinism
# ---------------------------------------------------------------------------- #
class TestCandidatePairs:
    def test_empty_vectors(self):
        r = LSHCandidateReducer()
        assert list(r.candidate_pairs({})) == []

    def test_single_vector(self):
        r = LSHCandidateReducer()
        assert list(r.candidate_pairs({"a": [1.0, 0.0]})) == []

    def test_pairs_sorted_and_deduped(self):
        r = LSHCandidateReducer(num_hyperplanes=1, num_tables=4, seed=0)
        # Two identical vectors collide in every table -> one pair total.
        vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
        pairs = list(r.candidate_pairs(vectors))
        assert pairs == [("a", "b")]

    def test_pair_ordering_a_le_b(self):
        r = LSHCandidateReducer(num_hyperplanes=1, num_tables=4, seed=0)
        # Insert in reverse lexicographic order; output must still be ``a <= b``.
        vectors = {"z": [1.0, 0.0], "a": [1.0, 0.0]}
        pairs = list(r.candidate_pairs(vectors))
        assert pairs == [("a", "z")]

    def test_deterministic_across_instances_same_seed(self):
        vectors, _within, _cross = _build_corpus(seed=7)
        r1 = LSHCandidateReducer(seed=42)
        r2 = LSHCandidateReducer(seed=42)
        assert list(r1.candidate_pairs(vectors)) == list(r2.candidate_pairs(vectors))

    def test_different_seed_different_candidates(self):
        vectors, _within, _cross = _build_corpus(seed=7)
        r1 = LSHCandidateReducer(seed=1)
        r2 = LSHCandidateReducer(seed=2)
        assert list(r1.candidate_pairs(vectors)) != list(r2.candidate_pairs(vectors))


# ---------------------------------------------------------------------------- #
# theoretical_recall
# ---------------------------------------------------------------------------- #
class TestTheoreticalRecall:
    def test_perfect_at_similarity_one(self):
        r = LSHCandidateReducer()
        assert r.theoretical_recall(1.0) == pytest.approx(1.0)

    def test_monotonic_increasing_in_similarity(self):
        r = LSHCandidateReducer(num_hyperplanes=6, num_tables=20)
        prev = -1.0
        for s in (0.0, 0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.0):
            rcl = r.theoretical_recall(s)
            assert rcl >= prev - 1e-12
            prev = rcl

    def test_more_tables_higher_recall(self):
        r_lo = LSHCandidateReducer(num_tables=2)
        r_hi = LSHCandidateReducer(num_tables=50)
        s = 0.6
        assert r_hi.theoretical_recall(s) > r_lo.theoretical_recall(s)

    def test_rejects_out_of_range(self):
        r = LSHCandidateReducer()
        with pytest.raises(ValueError):
            r.theoretical_recall(1.5)
        with pytest.raises(ValueError):
            r.theoretical_recall(-1.5)

    def test_rejects_wrong_type(self):
        r = LSHCandidateReducer()
        with pytest.raises(TypeError):
            r.theoretical_recall("0.5")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------- #
# contract: precision == 1.0, recall is high for tight duplicates
# ---------------------------------------------------------------------------- #
class TestPrecisionAndRecall:
    def test_precision_always_one_no_false_positives(self):
        # Every pair returned via candidate_reducer must also appear in the exact
        # brute-force result at the same threshold -- LSH cannot invent pairs.
        vectors, _within, cross = _build_corpus(seed=11)
        reducer = LSHCandidateReducer(num_hyperplanes=4, num_tables=20, seed=42)
        threshold = 0.5
        exact = find_similar_pairs(vectors, threshold=threshold)
        reduced = find_similar_pairs(vectors, threshold=threshold, candidate_reducer=reducer)
        exact_set = {(p.a, p.b) for p in exact}
        reduced_set = {(p.a, p.b) for p in reduced}
        # precision: every reduced pair is a real above-threshold pair
        assert reduced_set <= exact_set
        # No cross-group pair should ever be returned (they're near-orthogonal).
        assert not (reduced_set & cross)

    def test_high_recall_on_tight_duplicates(self):
        # With generous LSH params and a corpus of tight near-duplicates, the
        # reducer should surface essentially every within-group pair.
        vectors, within, _cross = _build_corpus(noise=0.03, seed=3)
        reducer = LSHCandidateReducer(num_hyperplanes=3, num_tables=40, seed=42)
        threshold = 0.5
        reduced = {(p.a, p.b) for p in
                   find_similar_pairs(vectors, threshold=threshold, candidate_reducer=reducer)}
        found = reduced & within
        recall = len(found) / len(within) if within else 1.0
        # Generous floor: theoretical recall at the *worst* within-group cosine
        # is far above this, but we keep the bar modest to stay deterministic.
        assert recall >= 0.9, f"LSH recall too low: {recall:.3f}"

    def test_reduced_scores_match_exact(self):
        # Scores on the verified pairs must equal the exact dot products.
        vectors, _within, _cross = _build_corpus(seed=5)
        reducer = LSHCandidateReducer(num_hyperplanes=4, num_tables=12, seed=1)
        exact = {(p.a, p.b): p.score
                 for p in find_similar_pairs(vectors, threshold=0.3)}
        for p in find_similar_pairs(vectors, threshold=0.3, candidate_reducer=reducer):
            assert p.score == pytest.approx(exact[(p.a, p.b)])

    def test_compression_on_random_singletons(self):
        # On a corpus of mutually near-orthogonal random vectors, the LSH
        # candidate set must be smaller than the full ``n*(n-1)/2`` pair set.
        rng = random.Random(123)
        dim = 32
        vectors = {f"r{i}": _norm([rng.gauss(0.0, 1.0) for _ in range(dim)])
                   for i in range(60)}
        reducer = LSHCandidateReducer(num_hyperplanes=8, num_tables=6, seed=99)
        n_candidates = sum(1 for _ in reducer.candidate_pairs(vectors))
        n_all = len(vectors) * (len(vectors) - 1) // 2
        assert n_candidates < n_all


# ---------------------------------------------------------------------------- #
# select_candidate_reducer
# ---------------------------------------------------------------------------- #
class TestSelectCandidateReducer:
    def test_returns_none_below_threshold(self):
        assert select_candidate_reducer(LSH_ACTIVATION_THRESHOLD - 1) is None

    def test_returns_reducer_at_or_above_threshold(self):
        r = select_candidate_reducer(LSH_ACTIVATION_THRESHOLD)
        assert isinstance(r, LSHCandidateReducer)

    def test_prefer_lsh_false_returns_none(self):
        assert select_candidate_reducer(10**6, prefer_lsh=False) is None

    def test_forwards_kwargs(self):
        r = select_candidate_reducer(10**6, num_hyperplanes=3, seed=7)
        assert isinstance(r, LSHCandidateReducer)
        assert r.num_hyperplanes == 3
        assert r.seed == 7


# ---------------------------------------------------------------------------- #
# backend & pipeline integration
# ---------------------------------------------------------------------------- #
class TestBackendIntegration:
    def test_backend_forwards_reducer_to_find_similar_pairs(self):
        # With a reducer that proposes only a known subset of candidates, the
        # backend must only emit clusters over that subset.
        vectors, _within, _cross = _build_corpus(noise=0.02, seed=21)
        reducer = LSHCandidateReducer(num_hyperplanes=3, num_tables=30, seed=42)
        backend_exact = ConnectedComponentsBackend()
        backend_reduced = ConnectedComponentsBackend(candidate_reducer=reducer)
        clusters_exact = backend_exact.cluster(vectors, threshold=0.5)
        clusters_reduced = backend_reduced.cluster(vectors, threshold=0.5)
        # Every non-singleton cluster from the reduced run must also exist in
        # the exact run (precision); the reduced run may have *fewer* non-
        # singletons (recall < 1.0 is allowed for the reducer).
        exact_clusters = {
            frozenset(c.members) for c in clusters_exact if c.size >= 2
        }
        for c in clusters_reduced:
            if c.size >= 2:
                assert frozenset(c.members) in exact_clusters


def _merge_json() -> str:
    import json

    return json.dumps(
        {
            "name": "Canonical",
            "critical_question": "What is the canonical answer?",
            "trusted_answer": "A merged, concise, verified answer.",
            "tags": ["IMPORTANT"],
            "entities": [
                {"entity_name": "SparkSage", "entity_type": "PRODUCT", "aliases": ["ss"]}
            ],
            "keywords": ["merged", "dedup"],
            "reasoning": "merged duplicates",
        }
    )


def _make_block(name: str, answer: str):
    from sparksage import IdeaBlock

    return IdeaBlock(name=name, critical_question=f"What is {name}?", trusted_answer=answer)


class TestPipelineIntegration:
    def test_pipeline_with_reducer_merges_obvious_duplicates(self):
        # The end-to-end Distill outcome for a corpus of obvious duplicates must
        # be the same whether or not the LSH reducer is plugged in: the same
        # blocks merge, the same singletons survive.
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=128))
        merger = BlockMerger(FakeLLMClient(responses=[_merge_json()]))
        reducer = LSHCandidateReducer(num_hyperplanes=3, num_tables=30, seed=42)
        pipe = DistillPipeline(
            embedder=embedder,
            merger=merger,
            candidate_reducer=reducer,
            start_threshold=0.4,
            max_iterations=1,
        )
        blocks = [
            _make_block("Deploy1", "deploy sparksage locally fast now"),
            _make_block("Deploy2", "deploy sparksage locally fast quick"),
            _make_block("Cook", "chocolate cake recipe sugar eggs flour"),
        ]
        result = pipe.run(blocks)
        assert len(result.survivors) == 2  # one canonical + the cook singleton
        assert len(result.merged_out) == 2
        assert result.reduction > 0.0

    def test_pipeline_reducer_attribute_exposed(self):
        embedder = BlockEmbedder(FakeEmbeddingClient())
        merger = BlockMerger(FakeLLMClient())
        reducer = LSHCandidateReducer()
        pipe = DistillPipeline(embedder=embedder, merger=merger, candidate_reducer=reducer)
        assert pipe.candidate_reducer is reducer
        # Default-constructed pipeline has no reducer.
        pipe2 = DistillPipeline(embedder=embedder, merger=merger)
        assert pipe2.candidate_reducer is None

    def test_default_backend_inherits_pipeline_reducer(self):
        embedder = BlockEmbedder(FakeEmbeddingClient())
        merger = BlockMerger(FakeLLMClient())
        reducer = LSHCandidateReducer()
        pipe = DistillPipeline(
            embedder=embedder,
            merger=merger,
            candidate_reducer=reducer,
        )
        # When no backend is injected, the pipeline builds a
        # ConnectedComponentsBackend that shares the reducer.
        assert pipe._backend.candidate_reducer is reducer  # noqa: SLF001
