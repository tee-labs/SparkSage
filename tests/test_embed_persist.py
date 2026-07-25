"""Tests for vector-store persistence (Phase 3).

Verifies the zero-dependency JSON round-trip (``save_store`` -> ``load_store``),
the on-disk format shape, and that corrupt / foreign / future-version files fail
fast instead of silently producing a broken store.
"""

from __future__ import annotations

import json

import pytest

from sparksage import (
    BlockEmbedder,
    FakeEmbeddingClient,
    InMemoryVectorStore,
    load_store,
    save_store,
)
from sparksage.embed.persist import STORE_FORMAT, STORE_VERSION


# ---------------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------------- #
def _populated_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore(dimension=4)
    store.add("a", [1.0, 0.0, 0.0, 0.0])
    store.add("b", [0.0, 1.0, -1.0, 0.5])
    store.add("c", [0.25, -0.25, 0.5, 0.75])
    return store


# ---------------------------------------------------------------------------- #
# save -> load round-trip
# ---------------------------------------------------------------------------- #
class TestRoundTrip:
    def test_round_trip_preserves_dimension_and_vectors(self, tmp_path):
        store = _populated_store()
        path = save_store(store, tmp_path / "store.json")

        loaded = load_store(path)
        assert isinstance(loaded, InMemoryVectorStore)
        assert loaded.dimension == store.dimension
        assert len(loaded) == len(store)
        for bid in ("a", "b", "c"):
            assert loaded.get(bid) == pytest.approx(store.get(bid))

    def test_round_trip_preserves_search_ranking(self, tmp_path):
        store = _populated_store()
        path = save_store(store, tmp_path / "store.json")
        loaded = load_store(path)

        query = [1.0, 0.0, 0.0, 0.0]
        before = [h.block_id for h in store.search(query, k=3)]
        after = [h.block_id for h in loaded.search(query, k=3)]
        assert before == after

    def test_save_returns_path(self, tmp_path):
        store = InMemoryVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        returned = save_store(store, tmp_path / "out.json")
        assert returned.exists()
        assert returned.name == "out.json"

    def test_save_creates_parent_dirs(self, tmp_path):
        store = InMemoryVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        target = tmp_path / "nested" / "deep" / "store.json"
        save_store(store, target)
        assert target.exists()

    def test_save_accepts_string_path(self, tmp_path):
        store = InMemoryVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        save_store(store, str(tmp_path / "store.json"))
        assert (tmp_path / "store.json").exists()

    def test_empty_store_round_trips(self, tmp_path):
        store = InMemoryVectorStore(dimension=8)
        path = save_store(store, tmp_path / "empty.json")
        loaded = load_store(path)
        assert len(loaded) == 0
        assert loaded.dimension == 8

    def test_loaded_store_is_independent_of_original(self, tmp_path):
        store = _populated_store()
        path = save_store(store, tmp_path / "store.json")
        loaded = load_store(path)

        store.add("new", [1.0, 1.0, 1.0, 1.0])
        assert "new" in store
        assert "new" not in loaded  # snapshot, not a live view

    def test_high_precision_floats_preserved(self, tmp_path):
        store = InMemoryVectorStore(dimension=3)
        exact = [0.123456789012345, -0.987654321098765, 0.5555555555555555]
        store.add("x", exact)
        loaded = load_store(save_store(store, tmp_path / "p.json"))
        assert loaded.get("x") == pytest.approx(exact)


# ---------------------------------------------------------------------------- #
# on-disk format shape
# ---------------------------------------------------------------------------- #
class TestFormat:
    def test_format_marker_and_version(self, tmp_path):
        path = save_store(_populated_store(), tmp_path / "store.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["format"] == STORE_FORMAT
        assert payload["version"] == STORE_VERSION
        assert payload["dimension"] == 4
        assert isinstance(payload["vectors"], dict)
        assert set(payload["vectors"]) == {"a", "b", "c"}

    def test_constants(self):
        assert STORE_FORMAT == "sparksage-vector-store"
        assert STORE_VERSION == 1


# ---------------------------------------------------------------------------- #
# failure modes
# ---------------------------------------------------------------------------- #
class TestLoadFailures:
    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_store(tmp_path / "nope.json")

    def test_not_a_json_object(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a JSON object"):
            load_store(path)

    def test_wrong_format_marker(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps({"format": "someone-elses-store", "version": 1, "vectors": {}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="format marker"):
            load_store(path)

    def test_unknown_future_version(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps(
                {"format": STORE_FORMAT, "version": 999, "dimension": 2, "vectors": {}}
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unsupported vector-store version"):
            load_store(path)

    def test_invalid_dimension(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps(
                {"format": STORE_FORMAT, "version": 1, "dimension": 0, "vectors": {}}
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="invalid dimension"):
            load_store(path)

    def test_vectors_not_an_object(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps(
                {
                    "format": STORE_FORMAT,
                    "version": 1,
                    "dimension": 2,
                    "vectors": [1, 2, 3],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="invalid vectors"):
            load_store(path)

    def test_vector_entry_not_a_list(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps(
                {
                    "format": STORE_FORMAT,
                    "version": 1,
                    "dimension": 2,
                    "vectors": {"a": "not-a-list"},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="invalid vector"):
            load_store(path)

    def test_loaded_vector_dimension_mismatch(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps(
                {
                    "format": STORE_FORMAT,
                    "version": 1,
                    "dimension": 2,
                    "vectors": {"a": [1.0, 2.0, 3.0]},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="dimension"):
            load_store(path)


# ---------------------------------------------------------------------------- #
# end-to-end with the embedder
# ---------------------------------------------------------------------------- #
class TestEndToEnd:
    def test_embed_persist_reload_search(self, tmp_path):
        from sparksage.schema import IdeaBlock

        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=128))
        blocks = [
            IdeaBlock(
                name="Deploy",
                critical_question="How to deploy?",
                trusted_answer="how to deploy sparksage locally",
            ),
            IdeaBlock(
                name="Cook",
                critical_question="How to bake?",
                trusted_answer="a recipe for chocolate cake",
            ),
        ]
        vectors = embedder.vectors_for(blocks)

        store = InMemoryVectorStore(dimension=128)
        store.add_many(vectors)
        path = save_store(store, tmp_path / "corpus.json")

        # fresh process-equivalent: load back and search
        reloaded = load_store(path)
        query = embedder.embed_texts(["how to deploy sparksage locally"])[0]
        hits = reloaded.search(query, k=2)

        deploy_id = str(blocks[0].id)
        cake_id = str(blocks[1].id)
        assert hits[0].block_id == deploy_id
        assert hits[-1].block_id == cake_id
