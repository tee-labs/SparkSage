"""Tests for the concrete :class:`VectorStore` backends.

These exercise the real backend glue (add / add_many / search / overwrite /
remove / dimension validation / kNN ranking / protocol compliance) fully
offline: each optional SDK (``faiss``/``numpy``, ``chromadb``, ``psycopg``) is
mocked out via ``sys.modules`` -- the same pattern used by ``test_client.py``
for the ``openai`` SDK. That keeps the suite zero-dependency while still driving
the production code paths.

When a backend's real SDK is installed, an extra integration check
(guarded by ``importorskip``) runs against the genuine library end-to-end.
"""

from __future__ import annotations

import math
import sys
import types
from typing import Any

import pytest

from sparksage import InMemoryVectorStore, SearchHit, VectorStore
from sparksage.embed.backends import (
    ChromaVectorStore,
    FaissVectorStore,
    PgvectorVectorStore,
)


# ---------------------------------------------------------------------------- #
# shared helpers
# ---------------------------------------------------------------------------- #
def _norm(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n else list(vec)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _parse_vec(s: str) -> list[float]:
    inner = s.strip().lstrip("[").rstrip("]")
    return [float(x) for x in inner.split(",")] if inner else []


# ---------------------------------------------------------------------------- #
# fake numpy + faiss (minimal, exact dot-product)
# ---------------------------------------------------------------------------- #
class _Arr:
    """Minimal 1-D ndarray stand-in."""

    def __init__(self, data: Any) -> None:
        self._d = [float(x) for x in data]

    def reshape(self, *shape: int) -> _Arr2D:
        return _Arr2D([self._d])

    def __iter__(self):
        return iter(self._d)

    def __getitem__(self, i: int) -> float:
        return self._d[i]

    def __len__(self) -> int:
        return len(self._d)

    def __truediv__(self, scalar: float) -> _Arr:
        return _Arr([x / scalar for x in self._d])


class _Arr2D:
    """Minimal 2-D ndarray stand-in (rows of floats)."""

    def __init__(self, rows: Any) -> None:
        self._rows = [list(r) for r in rows]

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, i: int) -> list[float]:
        return self._rows[i]

    def __len__(self) -> int:
        return len(self._rows)


class _FakeNumpy:
    class _Linalg:
        @staticmethod
        def norm(arr: _Arr) -> float:
            return math.sqrt(sum(x * x for x in arr))

    def __init__(self) -> None:
        self.linalg = self._Linalg()

    def asarray(self, data: Any, dtype: Any = None) -> _Arr | _Arr2D:
        if isinstance(data, (list, tuple)) and data:
            first = data[0]
            if isinstance(first, _Arr):
                return _Arr2D([list(x._d) for x in data])
            if isinstance(first, (list, tuple)):
                return _Arr2D([list(r) for r in data])
        return _Arr(data)

    @staticmethod
    def isscalar(obj: Any) -> bool:
        # pytest.approx probes np.isscalar when it sees numpy in sys.modules;
        # expose it so the fake module doesn't trip the comparator.
        return not isinstance(obj, (list, tuple, _Arr, _Arr2D, dict))

    class _BoolStub:
        # pytest.approx.is_bool does isinstance(val, np.bool_); never matches
        # a real float, which is exactly what we want.
        pass

    bool_ = _BoolStub


class _FakeIndexFlatIP:
    def __init__(self, dim: int) -> None:
        self._dim = dim


class _IDSelectorBatch:
    def __init__(self, ids: list[int]) -> None:
        self._ids = list(ids)


class _FakeIDMap2:
    def __init__(self, base: _FakeIndexFlatIP) -> None:
        self._base = base
        self._store: dict[int, list[float]] = {}

    def add_with_ids(self, vectors: _Arr2D, ids: _Arr) -> None:
        for vec_row, fid in zip(vectors, ids, strict=False):
            self._store[int(fid)] = list(vec_row)

    def remove_ids(self, selector: _IDSelectorBatch) -> int:
        removed = 0
        for fid in selector._ids:
            if self._store.pop(int(fid), None) is not None:
                removed += 1
        return removed

    def search(self, query: _Arr2D, k: int) -> tuple[_Arr2D, _Arr2D]:
        q = list(query[0])
        scored = [
            (sum(a * b for a, b in zip(q, vec, strict=True)), fid)
            for fid, vec in self._store.items()
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:k]
        while len(top) < k:
            top.append((float("-inf"), -1))
        return _Arr2D([[s for s, _ in top]]), _Arr2D([[i for _, i in top]])


def _install_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    np_mod = types.ModuleType("numpy")
    fake_np = _FakeNumpy()
    np_mod.asarray = fake_np.asarray
    np_mod.linalg = fake_np.linalg  # type: ignore[attr-defined]
    np_mod.isscalar = fake_np.isscalar  # type: ignore[attr-defined]
    np_mod.bool_ = _FakeNumpy.bool_  # type: ignore[attr-defined]
    faiss_mod = types.ModuleType("faiss")
    faiss_mod.IndexFlatIP = _FakeIndexFlatIP  # type: ignore[attr-defined]
    faiss_mod.IndexIDMap2 = _FakeIDMap2  # type: ignore[attr-defined]
    faiss_mod.IDSelectorBatch = _IDSelectorBatch  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "numpy", np_mod)
    monkeypatch.setitem(sys.modules, "faiss", faiss_mod)


# ---------------------------------------------------------------------------- #
# fake chromadb (minimal, exact cosine)
# ---------------------------------------------------------------------------- #
class _FakeCollection:
    def __init__(self, name: str, metadata: dict[str, Any] | None) -> None:
        self.name = name
        self.metadata = metadata
        self._data: dict[str, list[float]] = {}

    def upsert(self, ids: list[str], embeddings: list[list[float]]) -> None:
        for bid, vec in zip(ids, embeddings, strict=False):
            self._data[str(bid)] = [float(x) for x in vec]

    def query(
        self, query_embeddings: list[list[float]], n_results: int
    ) -> dict[str, list[list[Any]]]:
        q = [float(x) for x in query_embeddings[0]]
        scored = [(1.0 - _cosine(q, vec), bid) for bid, vec in self._data.items()]
        scored.sort(key=lambda t: t[0])  # ascending distance
        top = scored[:n_results]
        return {
            "ids": [[bid for _, bid in top]],
            "distances": [[dist for dist, _ in top]],
        }

    def get(self, ids: list[str]) -> dict[str, list[str]]:
        return {"ids": [str(i) for i in ids if str(i) in self._data]}

    def delete(self, ids: list[str]) -> None:
        for bid in ids:
            self._data.pop(str(bid), None)

    def count(self) -> int:
        return len(self._data)


class _FakeChromaClient:
    def __init__(self) -> None:
        self._collections: dict[str, _FakeCollection] = {}

    def get_or_create_collection(
        self, name: str, metadata: dict[str, Any] | None = None
    ) -> _FakeCollection:
        if name not in self._collections:
            self._collections[name] = _FakeCollection(name, metadata)
        return self._collections[name]


def _install_chroma(monkeypatch: pytest.MonkeyPatch) -> _FakeChromaClient:
    client = _FakeChromaClient()
    mod = types.ModuleType("chromadb")
    mod.Client = lambda: client  # type: ignore[attr-defined]
    mod.PersistentClient = lambda path=None: client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chromadb", mod)
    return client


# ---------------------------------------------------------------------------- #
# fake psycopg (minimal, parses the SparkSage SQL surface)
# ---------------------------------------------------------------------------- #
class _FakePgCursor:
    def __init__(self, store: dict[str, list[float]]) -> None:
        self._store = store
        self._rows: list[tuple] = []

    def __enter__(self) -> _FakePgCursor:
        return self

    def __exit__(self, *a: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = ()) -> None:
        low = sql.strip().upper()
        if low.startswith("CREATE"):
            return
        params = tuple(params)
        if "INSERT INTO" in low and "ON CONFLICT" in low:
            bid, vec_str = params
            self._store[str(bid)] = _parse_vec(vec_str)
            return
        if low.startswith("SELECT 1 FROM") and "WHERE BLOCK_ID" in low:
            bid = params[0]
            self._rows = [(1,)] if str(bid) in self._store else []
            return
        if low.startswith("SELECT COUNT"):
            self._rows = [(len(self._store),)]
            return
        if low.startswith("SELECT BLOCK_ID") and "ORDER BY" in low:
            qvec = _parse_vec(params[0])
            k = int(params[2])
            scored = [(bid, _cosine(qvec, vec)) for bid, vec in self._store.items()]
            scored.sort(key=lambda t: t[1], reverse=True)
            self._rows = scored[:k]
            return
        if low.startswith("DELETE FROM"):
            self._store.pop(str(params[0]), None)
            return

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        for row in rows:
            self.execute(sql, row)

    def fetchone(self) -> tuple | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple]:
        return list(self._rows)


class _FakePgConnection:
    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}
        self.commits = 0
        self.closed = False

    def cursor(self) -> _FakePgCursor:
        return _FakePgCursor(self._store)

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True


def _install_psycopg(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("psycopg")
    mod.connect = lambda *a, **k: _FakePgConnection()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", mod)


# ============================================================================ #
# FAISS
# ============================================================================ #
class TestFaissVectorStore:
    def test_protocol_compliance(self, monkeypatch):
        _install_faiss(monkeypatch)
        store = FaissVectorStore(dimension=3)
        assert isinstance(store, VectorStore)

    def test_dimension_exposed(self, monkeypatch):
        _install_faiss(monkeypatch)
        assert FaissVectorStore(dimension=64).dimension == 64

    def test_dimension_must_be_positive(self, monkeypatch):
        _install_faiss(monkeypatch)
        with pytest.raises(ValueError, match="dimension"):
            FaissVectorStore(dimension=0)

    def test_dimension_must_be_int(self, monkeypatch):
        _install_faiss(monkeypatch)
        with pytest.raises(TypeError, match="dimension"):
            FaissVectorStore(dimension=2.5)  # type: ignore[arg-type]

    def test_add_contains_len(self, monkeypatch):
        _install_faiss(monkeypatch)
        store = FaissVectorStore(dimension=3)
        store.add("a", [1.0, 0.0, 0.0])
        assert "a" in store
        assert len(store) == 1

    def test_add_overwrites(self, monkeypatch):
        _install_faiss(monkeypatch)
        store = FaissVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        store.add("a", [0.0, 1.0])
        assert len(store) == 1
        hits = store.search([0.0, 1.0], k=1)
        assert hits[0].block_id == "a"
        assert hits[0].score == pytest.approx(1.0)

    def test_add_rejects_wrong_dimension(self, monkeypatch):
        _install_faiss(monkeypatch)
        store = FaissVectorStore(dimension=3)
        with pytest.raises(ValueError, match="dimension"):
            store.add("a", [1.0, 0.0])

    def test_add_many(self, monkeypatch):
        _install_faiss(monkeypatch)
        store = FaissVectorStore(dimension=2)
        store.add_many({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        assert len(store) == 2
        assert "a" in store and "b" in store

    def test_add_many_validates_before_mutating(self, monkeypatch):
        _install_faiss(monkeypatch)
        store = FaissVectorStore(dimension=2)
        with pytest.raises(ValueError, match="dimension"):
            store.add_many({"good": [0.0, 1.0], "bad": [1.0, 2.0, 3.0]})
        assert len(store) == 0

    def test_add_many_overwrites_existing(self, monkeypatch):
        _install_faiss(monkeypatch)
        store = FaissVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        store.add_many({"a": [0.0, 1.0], "b": [1.0, 0.0]})
        assert len(store) == 2
        hit = store.search([0.0, 1.0], k=1)[0]
        assert hit.block_id == "a"

    def test_search_sorted_best_first(self, monkeypatch):
        _install_faiss(monkeypatch)
        store = FaissVectorStore(dimension=3)
        store.add("a", _norm([1.0, 0.0, 0.0]))
        store.add("b", _norm([0.9, 0.1, 0.0]))
        store.add("c", [0.0, 0.0, 1.0])
        hits = store.search([1.0, 0.0, 0.0], k=3)
        assert all(isinstance(h, SearchHit) for h in hits)
        assert [h.block_id for h in hits] == ["a", "b", "c"]
        assert hits[0].score > hits[1].score > hits[2].score

    def test_search_top_k_limits(self, monkeypatch):
        _install_faiss(monkeypatch)
        store = FaissVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        store.add("b", [0.9, 0.1])
        hits = store.search([1.0, 0.0], k=1)
        assert [h.block_id for h in hits] == ["a"]

    def test_search_empty_store(self, monkeypatch):
        _install_faiss(monkeypatch)
        assert FaissVectorStore(dimension=2).search([1.0, 0.0]) == []

    def test_search_k_validation(self, monkeypatch):
        _install_faiss(monkeypatch)
        store = FaissVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        with pytest.raises(ValueError, match="k"):
            store.search([1.0, 0.0], k=0)

    def test_remove(self, monkeypatch):
        _install_faiss(monkeypatch)
        store = FaissVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        assert store.remove("a") is True
        assert "a" not in store
        assert store.remove("a") is False

    def test_normalize_path(self, monkeypatch):
        _install_faiss(monkeypatch)
        store = FaissVectorStore(dimension=3, normalize=True)
        store.add("a", [3.0, 0.0, 0.0])  # not unit length
        hit = store.search([1.0, 0.0, 0.0], k=1)[0]
        assert hit.score == pytest.approx(1.0, abs=1e-6)

    def test_repr(self, monkeypatch):
        _install_faiss(monkeypatch)
        store = FaissVectorStore(dimension=4)
        store.add("x", [1.0, 0.0, 0.0, 0.0])
        text = repr(store)
        assert "dimension=4" in text
        assert "count=1" in text


# ============================================================================ #
# Chroma
# ============================================================================ #
class TestChromaVectorStore:
    def test_protocol_compliance(self, monkeypatch):
        _install_chroma(monkeypatch)
        assert isinstance(ChromaVectorStore(dimension=3), VectorStore)

    def test_dimension_exposed(self, monkeypatch):
        _install_chroma(monkeypatch)
        assert ChromaVectorStore(dimension=16).dimension == 16

    def test_dimension_must_be_positive(self, monkeypatch):
        _install_chroma(monkeypatch)
        with pytest.raises(ValueError, match="dimension"):
            ChromaVectorStore(dimension=0)

    def test_add_contains_len(self, monkeypatch):
        _install_chroma(monkeypatch)
        store = ChromaVectorStore(dimension=3)
        store.add("a", [1.0, 0.0, 0.0])
        assert "a" in store
        assert len(store) == 1

    def test_add_overwrites(self, monkeypatch):
        _install_chroma(monkeypatch)
        store = ChromaVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        store.add("a", [0.0, 1.0])
        assert len(store) == 1
        hit = store.search([0.0, 1.0], k=1)[0]
        assert hit.block_id == "a"
        assert hit.score == pytest.approx(1.0, abs=1e-6)

    def test_add_rejects_wrong_dimension(self, monkeypatch):
        _install_chroma(monkeypatch)
        store = ChromaVectorStore(dimension=3)
        with pytest.raises(ValueError, match="dimension"):
            store.add("a", [1.0, 0.0])

    def test_add_many(self, monkeypatch):
        _install_chroma(monkeypatch)
        store = ChromaVectorStore(dimension=2)
        store.add_many({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        assert len(store) == 2

    def test_search_sorted_best_first(self, monkeypatch):
        _install_chroma(monkeypatch)
        store = ChromaVectorStore(dimension=3)
        store.add("a", [1.0, 0.0, 0.0])
        store.add("b", [0.9, 0.1, 0.0])
        store.add("c", [0.0, 0.0, 1.0])
        hits = store.search([1.0, 0.0, 0.0], k=3)
        assert [h.block_id for h in hits] == ["a", "b", "c"]

    def test_search_empty_store(self, monkeypatch):
        _install_chroma(monkeypatch)
        assert ChromaVectorStore(dimension=2).search([1.0, 0.0]) == []

    def test_search_k_validation(self, monkeypatch):
        _install_chroma(monkeypatch)
        store = ChromaVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        with pytest.raises(ValueError, match="k"):
            store.search([1.0, 0.0], k=0)

    def test_remove(self, monkeypatch):
        _install_chroma(monkeypatch)
        store = ChromaVectorStore(dimension=2)
        store.add("a", [1.0, 0.0])
        assert store.remove("a") is True
        assert "a" not in store
        assert store.remove("a") is False

    def test_score_is_similarity_not_distance(self, monkeypatch):
        _install_chroma(monkeypatch)
        store = ChromaVectorStore(dimension=3)
        store.add("a", [1.0, 0.0, 0.0])
        hit = store.search([1.0, 0.0, 0.0], k=1)[0]
        # cosine distance 0 -> similarity 1
        assert hit.score == pytest.approx(1.0, abs=1e-6)

    def test_uses_persistent_client_when_path_given(self, monkeypatch):
        calls: list[str] = []
        client = _FakeChromaClient()

        mod = types.ModuleType("chromadb")
        mod.Client = lambda: client  # type: ignore[attr-defined]

        def _persistent(path=None):
            calls.append(str(path))
            return client

        mod.PersistentClient = _persistent  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "chromadb", mod)

        ChromaVectorStore(dimension=2, path="/tmp/data")
        assert calls == ["/tmp/data"]

    def test_repr(self, monkeypatch):
        _install_chroma(monkeypatch)
        store = ChromaVectorStore(dimension=4)
        store.add("x", [1.0, 0.0, 0.0, 0.0])
        text = repr(store)
        assert "dimension=4" in text
        assert "count=1" in text


# ============================================================================ #
# pgvector
# ============================================================================ #
class TestPgvectorVectorStore:
    def test_protocol_compliance(self, monkeypatch):
        _install_psycopg(monkeypatch)
        conn = _FakePgConnection()
        store = PgvectorVectorStore(dimension=3, connection=conn)
        assert isinstance(store, VectorStore)

    def test_dimension_exposed(self, monkeypatch):
        _install_psycopg(monkeypatch)
        store = PgvectorVectorStore(dimension=8, connection=_FakePgConnection())
        assert store.dimension == 8

    def test_requires_dsn_or_connection(self, monkeypatch):
        _install_psycopg(monkeypatch)
        with pytest.raises(ValueError, match="dsn"):
            PgvectorVectorStore(dimension=3)

    def test_dimension_must_be_positive(self, monkeypatch):
        _install_psycopg(monkeypatch)
        with pytest.raises(ValueError, match="dimension"):
            PgvectorVectorStore(dimension=0, connection=_FakePgConnection())

    def test_invalid_table_name_rejected(self, monkeypatch):
        _install_psycopg(monkeypatch)
        with pytest.raises(ValueError, match="table"):
            PgvectorVectorStore(
                dimension=3, connection=_FakePgConnection(), table="bad name!"
            )

    def test_invalid_distance_rejected(self, monkeypatch):
        _install_psycopg(monkeypatch)
        with pytest.raises(ValueError, match="distance"):
            PgvectorVectorStore(
                dimension=3, connection=_FakePgConnection(), distance="hamming"
            )

    def test_add_contains_len(self, monkeypatch):
        _install_psycopg(monkeypatch)
        store = PgvectorVectorStore(dimension=3, connection=_FakePgConnection())
        store.add("a", [1.0, 0.0, 0.0])
        assert "a" in store
        assert len(store) == 1

    def test_add_overwrites(self, monkeypatch):
        _install_psycopg(monkeypatch)
        store = PgvectorVectorStore(dimension=2, connection=_FakePgConnection())
        store.add("a", [1.0, 0.0])
        store.add("a", [0.0, 1.0])
        assert len(store) == 1
        hit = store.search([0.0, 1.0], k=1)[0]
        assert hit.block_id == "a"

    def test_add_rejects_wrong_dimension(self, monkeypatch):
        _install_psycopg(monkeypatch)
        store = PgvectorVectorStore(dimension=3, connection=_FakePgConnection())
        with pytest.raises(ValueError, match="dimension"):
            store.add("a", [1.0, 0.0])

    def test_add_many(self, monkeypatch):
        _install_psycopg(monkeypatch)
        store = PgvectorVectorStore(dimension=2, connection=_FakePgConnection())
        store.add_many({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        assert len(store) == 2

    def test_add_many_validates_before_mutating(self, monkeypatch):
        _install_psycopg(monkeypatch)
        store = PgvectorVectorStore(dimension=2, connection=_FakePgConnection())
        with pytest.raises(ValueError, match="dimension"):
            store.add_many({"good": [0.0, 1.0], "bad": [1.0, 2.0, 3.0]})
        assert len(store) == 0

    def test_search_sorted_best_first(self, monkeypatch):
        _install_psycopg(monkeypatch)
        store = PgvectorVectorStore(dimension=3, connection=_FakePgConnection())
        store.add("a", [1.0, 0.0, 0.0])
        store.add("b", [0.9, 0.1, 0.0])
        store.add("c", [0.0, 0.0, 1.0])
        hits = store.search([1.0, 0.0, 0.0], k=3)
        assert [h.block_id for h in hits] == ["a", "b", "c"]

    def test_search_empty_store(self, monkeypatch):
        _install_psycopg(monkeypatch)
        store = PgvectorVectorStore(dimension=2, connection=_FakePgConnection())
        assert store.search([1.0, 0.0]) == []

    def test_search_k_validation(self, monkeypatch):
        _install_psycopg(monkeypatch)
        store = PgvectorVectorStore(dimension=2, connection=_FakePgConnection())
        store.add("a", [1.0, 0.0])
        with pytest.raises(ValueError, match="k"):
            store.search([1.0, 0.0], k=0)

    def test_remove(self, monkeypatch):
        _install_psycopg(monkeypatch)
        store = PgvectorVectorStore(dimension=2, connection=_FakePgConnection())
        store.add("a", [1.0, 0.0])
        assert store.remove("a") is True
        assert "a" not in store
        assert store.remove("a") is False

    def test_search_limits_to_k(self, monkeypatch):
        _install_psycopg(monkeypatch)
        store = PgvectorVectorStore(dimension=2, connection=_FakePgConnection())
        store.add("a", [1.0, 0.0])
        store.add("b", [0.9, 0.1])
        hits = store.search([1.0, 0.0], k=1)
        assert [h.block_id for h in hits] == ["a"]

    def test_repr(self, monkeypatch):
        _install_psycopg(monkeypatch)
        store = PgvectorVectorStore(dimension=4, connection=_FakePgConnection())
        store.add("x", [1.0, 0.0, 0.0, 0.0])
        text = repr(store)
        assert "dimension=4" in text
        assert "count=1" in text
        assert "table=" in text

    def test_close_only_when_owned(self, monkeypatch):
        _install_psycopg(monkeypatch)
        # injected connection is NOT owned -> close() is a no-op
        conn = _FakePgConnection()
        store = PgvectorVectorStore(dimension=2, connection=conn)
        store.close()
        assert conn.closed is False


# ============================================================================ #
# cross-backend ranking parity vs InMemoryVectorStore
# ============================================================================ #
@pytest.mark.parametrize(
    "backend",
    ["faiss", "chroma", "pgvector"],
)
def test_top1_matches_inmemory(backend, monkeypatch):
    vectors = {
        "a": _norm([1.0, 0.0, 0.0]),
        "b": _norm([0.0, 1.0, 0.0]),
        "c": _norm([0.7, 0.7, 0.0]),
    }
    query = [1.0, 0.0, 0.0]

    ref = InMemoryVectorStore(dimension=3)
    ref.add_many(vectors)
    ref_top1 = ref.search(query, k=1)[0].block_id

    if backend == "faiss":
        _install_faiss(monkeypatch)
        store = FaissVectorStore(dimension=3)
    elif backend == "chroma":
        _install_chroma(monkeypatch)
        store = ChromaVectorStore(dimension=3)
    else:
        _install_psycopg(monkeypatch)
        store = PgvectorVectorStore(dimension=3, connection=_FakePgConnection())

    store.add_many(vectors)
    hits = store.search(query, k=3)
    assert hits[0].block_id == ref_top1
    # full ranking order must match too
    ref_order = [h.block_id for h in ref.search(query, k=3)]
    assert [h.block_id for h in hits] == ref_order


# ============================================================================ #
# real-SDK integration (only when installed)
# ============================================================================ #
def test_real_faiss_end_to_end():
    pytest.importorskip("numpy")
    faiss = pytest.importorskip("faiss")  # noqa: F841
    store = FaissVectorStore(dimension=3)
    store.add("a", [1.0, 0.0, 0.0])
    store.add("b", [0.0, 1.0, 0.0])
    hits = store.search([1.0, 0.1, 0.0], k=2)
    assert hits[0].block_id == "a"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_real_chroma_end_to_end():
    chromadb = pytest.importorskip("chromadb")  # noqa: F841
    store = ChromaVectorStore(dimension=3)  # ephemeral in-process client
    store.add("a", [1.0, 0.0, 0.0])
    store.add("b", [0.0, 1.0, 0.0])
    hits = store.search([1.0, 0.1, 0.0], k=2)
    assert hits[0].block_id == "a"
