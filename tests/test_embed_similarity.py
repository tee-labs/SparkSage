"""Tests for the all-pairs similarity / near-duplicate layer.

All tests run fully offline and dependency-free: they use hand-built unit
vectors and the deterministic :class:`FakeEmbeddingClient` (via
:class:`BlockEmbedder`) to exercise the pair-detection contract that the future
Distill de-dup pipeline consumes.
"""

from __future__ import annotations

import pytest

from sparksage import (
    BlockEmbedder,
    FakeEmbeddingClient,
    SimilarityPair,
    find_similar_pairs,
)
from sparksage.schema import IdeaBlock


# ---------------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------------- #
def _norm(vec: list[float]) -> list[float]:
    """Scale ``vec`` to unit length (hand-built vectors in tests are normalized)."""
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def _make_block(
    name: str = "Block",
    question: str = "What is this?",
    answer: str = "A short verified answer.",
) -> IdeaBlock:
    return IdeaBlock(name=name, critical_question=question, trusted_answer=answer)


# ---------------------------------------------------------------------------- #
# SimilarityPair dataclass
# ---------------------------------------------------------------------------- #
class TestSimilarityPair:
    def test_is_frozen(self):
        pair = SimilarityPair(a="x", b="y", score=0.9)
        with pytest.raises(AttributeError):
            pair.score = 0.1  # type: ignore[misc]

    def test_fields_exposed(self):
        pair = SimilarityPair(a="x", b="y", score=0.9)
        assert pair.a == "x"
        assert pair.b == "y"
        assert pair.score == 0.9


# ---------------------------------------------------------------------------- #
# find_similar_pairs: shape & edge cases
# ---------------------------------------------------------------------------- #
class TestShape:
    def test_empty_dict(self):
        assert find_similar_pairs({}) == []

    def test_single_vector(self):
        assert find_similar_pairs({"a": [1.0, 0.0]}) == []

    def test_pair_below_threshold_excluded(self):
        # two orthogonal unit vectors -> dot product 0
        vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        assert find_similar_pairs(vectors, threshold=0.5) == []

    def test_pair_at_threshold_included(self):
        vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
        pairs = find_similar_pairs(vectors, threshold=1.0)
        assert len(pairs) == 1
        assert pairs[0].score == pytest.approx(1.0)

    def test_identical_vectors_score_one(self):
        vectors = {"a": _norm([3.0, 4.0]), "b": _norm([3.0, 4.0])}
        pairs = find_similar_pairs(vectors, threshold=0.5)
        assert len(pairs) == 1
        assert pairs[0].score == pytest.approx(1.0)
        assert pairs[0].a == "a"
        assert pairs[0].b == "b"


# ---------------------------------------------------------------------------- #
# normalization, dedup, ordering
# ---------------------------------------------------------------------------- #
class TestOrdering:
    def test_pair_ids_normalized_a_le_b(self):
        # insert in reverse lexicographic order; result must still be a <= b
        vectors = {"z": [1.0, 0.0], "a": [1.0, 0.0]}
        pairs = find_similar_pairs(vectors, threshold=0.5)
        assert len(pairs) == 1
        assert pairs[0].a == "a"
        assert pairs[0].b == "z"

    def test_each_unordered_pair_appears_once(self):
        vectors = {
            "a": [1.0, 0.0],
            "b": [0.9, 0.43589],  # ~normalized, similar to a
            "c": [0.0, 1.0],  # orthogonal to both
        }
        pairs = find_similar_pairs(vectors, threshold=0.0)
        # 3 choose 2 = 3 pairs, each unordered pair exactly once
        keys = {frozenset((p.a, p.b)) for p in pairs}
        assert keys == {frozenset(("a", "b")), frozenset(("a", "c")), frozenset(("b", "c"))}
        assert len(pairs) == 3

    def test_sorted_by_score_desc(self):
        vectors = {
            "low": [1.0, 0.0],
            "mid": [0.8, 0.6],  # dot with low = 0.8
            "high": [0.95, 0.05],  # dot with low ~ 0.95
        }
        pairs = find_similar_pairs(vectors, threshold=0.0)
        scores = [p.score for p in pairs]
        assert scores == sorted(scores, reverse=True)

    def test_self_pair_never_returned(self):
        vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
        pairs = find_similar_pairs(vectors, threshold=0.0)
        for p in pairs:
            assert p.a != p.b

    def test_deterministic_regardless_of_dict_order(self):
        v1 = {"a": [1.0, 0.0], "b": [0.9, 0.43589], "c": [0.0, 1.0]}
        v2 = {
            "c": [0.0, 1.0],
            "b": [0.9, 0.43589],
            "a": [1.0, 0.0],
        }
        assert find_similar_pairs(v1, threshold=0.0) == find_similar_pairs(v2, threshold=0.0)


# ---------------------------------------------------------------------------- #
# top_k
# ---------------------------------------------------------------------------- #
class TestTopK:
    def test_top_k_limits_results(self):
        # three vectors all similar to 'a'
        vectors = {
            "a": [1.0, 0.0],
            "b": [0.9, 0.01],
            "c": [0.8, 0.01],
            "d": [0.7, 0.01],
        }
        pairs = find_similar_pairs(vectors, threshold=0.0, top_k=2)
        assert len(pairs) == 2
        # still sorted by score desc
        scores = [p.score for p in pairs]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_none_returns_all(self):
        vectors = {"a": [1.0, 0.0], "b": [0.9, 0.43589], "c": [0.0, 1.0]}
        pairs = find_similar_pairs(vectors, threshold=0.0, top_k=None)
        assert len(pairs) == 3

    def test_top_k_larger_than_count_returns_all(self):
        vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
        pairs = find_similar_pairs(vectors, threshold=0.5, top_k=10)
        assert len(pairs) == 1


# ---------------------------------------------------------------------------- #
# validation
# ---------------------------------------------------------------------------- #
class TestValidation:
    def test_threshold_type_error(self):
        with pytest.raises(TypeError, match="threshold"):
            find_similar_pairs({}, threshold="0.5")  # type: ignore[arg-type]

    def test_threshold_bool_rejected(self):
        with pytest.raises(TypeError, match="threshold"):
            find_similar_pairs({}, threshold=True)  # type: ignore[arg-type]

    def test_threshold_below_zero(self):
        with pytest.raises(ValueError, match="threshold"):
            find_similar_pairs({}, threshold=-0.1)

    def test_threshold_above_one(self):
        with pytest.raises(ValueError, match="threshold"):
            find_similar_pairs({}, threshold=1.5)

    def test_threshold_zero_allowed(self):
        # boundary: threshold=0 is valid
        vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}  # orthogonal, dot=0
        assert len(find_similar_pairs(vectors, threshold=0.0)) == 1

    def test_threshold_one_allowed(self):
        vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
        assert len(find_similar_pairs(vectors, threshold=1.0)) == 1

    def test_top_k_type_error(self):
        with pytest.raises(TypeError, match="top_k"):
            find_similar_pairs({}, top_k=2.5)  # type: ignore[arg-type]

    def test_top_k_bool_rejected(self):
        with pytest.raises(TypeError, match="top_k"):
            find_similar_pairs({}, top_k=True)  # type: ignore[arg-type]

    def test_top_k_below_one(self):
        with pytest.raises(ValueError, match="top_k"):
            find_similar_pairs({}, top_k=0)

    def test_inconsistent_dimensions(self):
        vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0, 0.0]}
        with pytest.raises(ValueError, match="inconsistent vector dimensions"):
            find_similar_pairs(vectors, threshold=0.0)

    def test_vector_not_a_list(self):
        vectors = {"a": [1.0, 0.0], "b": "not-a-list"}  # type: ignore[dict-item]
        with pytest.raises(TypeError, match="must be a list"):
            find_similar_pairs(vectors, threshold=0.0)


# ---------------------------------------------------------------------------- #
# end-to-end: BlockEmbedder.vectors_for -> find_similar_pairs
# ---------------------------------------------------------------------------- #
class TestEndToEnd:
    def test_near_duplicates_detected(self):
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=256))
        blocks = [
            _make_block("Deploy", "How to deploy?", "how to deploy sparksage locally"),
            _make_block("Deploy2", "How to run?", "how to run sparksage locally fast"),
            _make_block("Cook", "How to bake?", "a recipe for chocolate cake"),
        ]
        vectors = embedder.vectors_for(blocks)

        deploy_ids = {str(blocks[0].id), str(blocks[1].id)}
        cake_id = str(blocks[2].id)

        # the two deploy blocks overlap heavily in the fake embedding, so their
        # pair must be the single highest-scoring one.
        pairs = find_similar_pairs(vectors, threshold=0.0)
        assert len(pairs) == 3  # all three unordered pairs
        top = pairs[0]
        assert {top.a, top.b} == deploy_ids
        assert top.score > pairs[1].score

        # there is a threshold that isolates the deploy~deploy pair from cake
        isolated = find_similar_pairs(vectors, threshold=top.score - 1e-6)
        assert len(isolated) == 1
        assert {isolated[0].a, isolated[0].b} == deploy_ids
        assert cake_id not in (isolated[0].a, isolated[0].b)

    def test_threshold_controls_recall(self):
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=128))
        blocks = [
            _make_block("A", "q a?", "alpha alpha alpha"),
            _make_block("B", "q b?", "alpha alpha alpha"),  # same answer as A
            _make_block("C", "q c?", "zzzzzzzzzzzz"),  # disjoint tokens
        ]
        vectors = embedder.vectors_for(blocks)
        a_id, b_id, c_id = (str(b.id) for b in blocks)

        all_pairs = {
            frozenset((p.a, p.b)): p.score
            for p in find_similar_pairs(vectors, threshold=0.0)
        }
        ab = all_pairs[frozenset((a_id, b_id))]
        ac = all_pairs[frozenset((a_id, c_id))]
        bc = all_pairs[frozenset((b_id, c_id))]

        # the identical-answer pair is clearly the most similar
        assert ab > ac
        assert ab > bc

        # tightening the threshold monotonically reduces recall
        n_loose = len(find_similar_pairs(vectors, threshold=0.0))
        n_mid = len(find_similar_pairs(vectors, threshold=0.5))
        n_hi = len(find_similar_pairs(vectors, threshold=0.9))
        assert n_loose >= n_mid >= n_hi

        # at a mid threshold only the near-identical A~B pair survives; C is gone
        mid = find_similar_pairs(vectors, threshold=0.5)
        assert len(mid) == 1
        assert {mid[0].a, mid[0].b} == {a_id, b_id}
        assert c_id not in (mid[0].a, mid[0].b)

    def test_consumes_store_vectors_snapshot(self):
        # the same dict contract is also produced by InMemoryVectorStore.vectors()
        from sparksage import InMemoryVectorStore

        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        blocks = [
            _make_block("A", "q a?", "alpha alpha alpha"),
            _make_block("B", "q b?", "alpha alpha alpha"),
        ]
        vectors = embedder.vectors_for(blocks)
        store = InMemoryVectorStore(dimension=64)
        store.add_many(vectors)

        from_store = find_similar_pairs(store.vectors(), threshold=0.5)
        direct = find_similar_pairs(vectors, threshold=0.5)
        assert from_store == direct
