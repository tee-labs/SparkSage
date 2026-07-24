"""Tests for the in-memory vector store (Phase 2).

All tests run fully offline and dependency-free: they use hand-built vectors
and the deterministic :class:`FakeEmbeddingClient` (via :class:`BlockEmbedder`)
to exercise the store + retrieval contract that the future Distill pipeline and
any RAG retriever consume.
"""

from __future__ import annotations

import pytest

from sparksage import (
    BlockEmbedder,
    FakeEmbeddingClient,
    InMemoryVectorStore,
    SearchHit,
    VectorStore,
)
from sparksage.schema import IdeaBlock


# ---------------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------------- #
def _make_block(
    name: str = "Block",
    question: str = "What is this?",
    answer: str = "A short verified answer.",
) -> IdeaBlock:
    return IdeaBlock(name=name, critical_question=question, trusted_answer=answer)


# ---------------------------------------------------------------------------- #
# construction & validation
# ---------------------------------------------------------------------------- #
class TestConstruction:
    def test_is_a_vector_store(self):
        assert isinstance(InMemoryVectorStore(dimension=4), VectorStore)

    def test_dimension_exposed(self):
        assert InMemoryVectorStore(dimension=128).dimension == 128

    def test_starts_empty(self):
        store = InMemoryVectorStore(dimension=4)
        assert len(store) == 0
        assert store.search([1.0, 0.0, 0.0, 0.0]) == []

    def test_dimension_must_be_positive(self):
        with pytest.raises(ValueError, match="dimension"):
            InMemoryVectorStore(dimension=0)

    def test_dimension_must_be_int(self):
        with pytest.raises(TypeError, match="dimension"):
            InMemoryVectorStore(dimension=3.5)  # type: ignore[arg-type]

    def test_dimension_bool_rejected(self):
        with pytest.raises(TypeError, match="dimension"):
            InMemoryVectorStore(dimension=True)  # type: ignore[arg-type]

    def test_repr(self):
        store = InMemoryVectorStore(dimension=8)
        store.add("x", [1.0] * 8)
        text = repr(store)
        assert "dimension=8" in text
        assert "count=1" in text


# ---------------------------------------------------------------------------- #
# add / get / remove / membership
# ---------------------------------------------------------------------------- #
class TestAddGetRemove:
    def test_add_then_contains(self):
        store = InMemoryVectorStore(dimension=3)
        store.add("a", [1.0, 0.0, 0.0])
        assert "a" in store
        assert len(store) == 1

    def test_add_coerces_key_to_str(self):
        store = InMemoryVectorStore(dimension=2)
        store.add(123, [1.0, 0.0])
        assert "123" in store
        assert 123 in store  # membership also coerces

    def test_add_overwrites(self):
        store = InMemoryVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        store.add("a", [0.0, 1.0])
        assert len(store) == 1
        assert store.get("a") == [0.0, 1.0]

    def test_add_rejects_wrong_dimension(self):
        store = InMemoryVectorStore(dimension=3)
        with pytest.raises(ValueError, match="dimension"):
            store.add("a", [1.0, 0.0])

    def test_add_copies_vector(self):
        store = InMemoryVectorStore(dimension=3)
        vec = [1.0, 0.0, 0.0]
        store.add("a", vec)
        vec[0] = 99.0
        assert store.get("a") == [1.0, 0.0, 0.0]

    def test_get_returns_copy(self):
        store = InMemoryVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        out = store.get("a")
        assert out is not None
        out[0] = 99.0
        assert store.get("a") == [1.0, 0.0]

    def test_get_missing_is_none(self):
        assert InMemoryVectorStore(dimension=2).get("nope") is None

    def test_remove(self):
        store = InMemoryVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        assert store.remove("a") is True
        assert "a" not in store
        assert len(store) == 0
        assert store.remove("a") is False

    def test_clear(self):
        store = InMemoryVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        store.add("b", [0.0, 1.0])
        store.clear()
        assert len(store) == 0
        assert store.dimension == 2  # dimension preserved

    def test_add_many(self):
        store = InMemoryVectorStore(dimension=2)
        store.add_many({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        assert len(store) == 2
        assert "a" in store and "b" in store

    def test_add_many_validates_before_mutating(self):
        store = InMemoryVectorStore(dimension=2)
        store.add("existing", [1.0, 0.0])
        with pytest.raises(ValueError, match="dimension"):
            store.add_many({"good": [0.0, 1.0], "bad": [1.0, 2.0, 3.0]})
        assert len(store) == 1  # nothing added
        assert "good" not in store

    def test_vectors_snapshot_is_a_copy(self):
        store = InMemoryVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        snap = store.vectors()
        snap["a"][0] = 99.0
        snap["b"] = [0.0, 0.0]
        assert store.get("a") == [1.0, 0.0]
        assert "b" not in store


# ---------------------------------------------------------------------------- #
# search
# ---------------------------------------------------------------------------- #
class TestSearch:
    def test_empty_store_returns_empty(self):
        store = InMemoryVectorStore(dimension=3)
        assert store.search([1.0, 0.0, 0.0]) == []

    def test_returns_searchhit_sorted_desc(self):
        store = InMemoryVectorStore(dimension=3)
        store.add("a", [1.0, 0.0, 0.0])
        store.add("b", [0.9, 0.1, 0.0])
        store.add("c", [0.0, 0.0, 1.0])
        hits = store.search([1.0, 0.0, 0.0], k=3)
        assert all(isinstance(h, SearchHit) for h in hits)
        assert [h.block_id for h in hits] == ["a", "b", "c"]
        assert hits[0].score == pytest.approx(1.0)
        assert hits[1].score > hits[2].score

    def test_top_k_limits_results(self):
        store = InMemoryVectorStore(dimension=3)
        store.add("a", [1.0, 0.0, 0.0])
        store.add("b", [0.9, 0.1, 0.0])
        store.add("c", [0.8, 0.2, 0.0])
        hits = store.search([1.0, 0.0, 0.0], k=2)
        assert [h.block_id for h in hits] == ["a", "b"]

    def test_k_larger_than_size_returns_all(self):
        store = InMemoryVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        hits = store.search([1.0, 0.0], k=10)
        assert len(hits) == 1

    def test_default_k(self):
        store = InMemoryVectorStore(dimension=2)
        for i in range(15):
            store.add(f"id-{i}", [float(i), 0.0])
        hits = store.search([100.0, 0.0])
        assert len(hits) == 10  # default k=10

    def test_k_must_be_positive(self):
        store = InMemoryVectorStore(dimension=2)
        with pytest.raises(ValueError, match="k"):
            store.search([1.0, 0.0], k=0)

    def test_k_must_be_int(self):
        store = InMemoryVectorStore(dimension=2)
        with pytest.raises(TypeError, match="k"):
            store.search([1.0, 0.0], k=2.5)  # type: ignore[arg-type]

    def test_query_dimension_validated(self):
        store = InMemoryVectorStore(dimension=3)
        store.add("a", [1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="query"):
            store.search([1.0, 0.0], k=1)

    def test_searchhit_is_frozen(self):
        store = InMemoryVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        hit = store.search([1.0, 0.0])[0]
        with pytest.raises(AttributeError):
            hit.block_id = "x"  # type: ignore[misc]

    def test_orthogonal_vectors_zero_score(self):
        store = InMemoryVectorStore(dimension=3)
        store.add("orth", [0.0, 1.0, 0.0])
        hit = store.search([1.0, 0.0, 0.0])[0]
        assert hit.score == pytest.approx(0.0)


# ---------------------------------------------------------------------------- #
# end-to-end: BlockEmbedder.vectors_for -> InMemoryVectorStore -> search
# ---------------------------------------------------------------------------- #
class TestEndToEndWithEmbedder:
    def test_store_consumes_vectors_for(self):
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=256))
        blocks = [
            _make_block("Deploy", "How to deploy?", "how to deploy sparksage locally"),
            _make_block("Cook", "How to bake?", "a recipe for chocolate cake"),
            _make_block("Deploy2", "How to run?", "how to run sparksage locally fast"),
        ]
        vectors = embedder.vectors_for(blocks)

        store = InMemoryVectorStore(dimension=256)
        store.add_many(vectors)

        # query reusing the deploy text -> the two deploy blocks should outrank
        # the cake block (FakeEmbeddingClient gives overlapping texts higher
        # dot-product similarity).
        query = embedder.embed_texts(["how to deploy sparksage locally"])[0]
        hits = store.search(query, k=3)
        assert len(hits) == 3
        ranked = [h.block_id for h in hits]
        cake_id = str(blocks[1].id)
        assert cake_id == ranked[-1]  # cake is least similar

    def test_query_from_real_block(self):
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=128))
        blocks = [
            _make_block("A", "q a?", "alpha alpha alpha"),
            _make_block("B", "q b?", "bravo bravo bravo"),
        ]
        vectors = embedder.vectors_for(blocks)
        store = InMemoryVectorStore(dimension=128)
        store.add_many(vectors)

        # searching with block A's own vector returns A first (score 1.0).
        hit = store.search(vectors[str(blocks[0].id)], k=1)[0]
        assert hit.block_id == str(blocks[0].id)
        assert hit.score == pytest.approx(1.0, abs=1e-6)

    def test_blocks_not_mutated_when_using_vectors_for(self):
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        block = _make_block("A", "q a?", "alpha alpha alpha")
        vectors = embedder.vectors_for([block])
        store = InMemoryVectorStore(dimension=64)
        store.add_many(vectors)
        assert block.embedding is None  # vectors_for never mutates


# ---------------------------------------------------------------------------- #
# protocol compliance of a duck type
# ---------------------------------------------------------------------------- #
class TestProtocolCompliance:
    def test_minimal_duck_type_matches_protocol(self):
        class _Min:
            @property
            def dimension(self) -> int:
                return 2

            def add(self, block_id, vector):  # noqa: ANN001
                self.v = {block_id: vector}
                return None

            def search(self, query, k=10):  # noqa: ANN001
                return []

            def __contains__(self, block_id):  # noqa: ANN001
                return False

            def __len__(self):
                return 0

        assert isinstance(_Min(), VectorStore)
