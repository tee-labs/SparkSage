"""FAISS-backed :class:`~sparksage.embed.store.VectorStore`.

FAISS earns its weight on million-vector corpora, where brute-force
:class:`~sparksage.embed.store.InMemoryVectorStore` kNN becomes the bottleneck.
This backend wraps an exact inner-product index
(``IndexFlatIP`` behind an ``IndexIDMap2`` so opaque string ``block_id`` keys map
to FAISS's ``int64`` ids) and implements the same :class:`VectorStore` protocol
the retrieval/Distill cores depend on -- swap it in anywhere an
:class:`InMemoryVectorStore` is used.

Vectors are assumed **L2-normalized** (every
:class:`~sparksage.embed.client.EmbeddingClient` normalizes by default), so
inner product == cosine similarity, matching the dot-product convention of
:class:`InMemoryVectorStore.search`. Store un-normalized vectors and the ranking
is still correct *as an inner product* -- the caller decides the metric by
pre-normalizing.

The ``faiss`` (and ``numpy``) packages are *optional* dependencies -- install
them with ``pip install 'sparksage[distill]'`` (the ``[distill]`` extra is where
the roadmap places the FAISS vector-store accelerator).
"""

from __future__ import annotations

from typing import Any

from sparksage.embed.store import SearchHit


class FaissVectorStore:
    """Exact inner-product vector store backed by FAISS ``IndexFlatIP``.

    Built for the million-vector scale where
    :class:`~sparksage.embed.store.InMemoryVectorStore` (brute-force Python kNN)
    is too slow. The index is an exact flat inner-product index wrapped in
    ``IndexIDMap2`` so opaque string ``block_id`` keys are preserved across
    adds/removes -- FAISS's native ``int64`` ids are an internal detail the
    caller never sees.

    Overwriting an existing ``block_id`` removes the old entry and re-inserts
    (``IndexFlatIP`` has no in-place update), so the latest vector always wins.
    Removes are supported via :meth:`remove_ids`.

    The ``faiss`` and ``numpy`` packages are *optional* dependencies -- install
    them with ``pip install 'sparksage[distill]'``.

    Parameters
    ----------
    dimension:
        The fixed length every stored vector (and every query) must have.
        Must be ``>= 1``.
    normalize:
        When ``True`` (default ``False``), L2-normalize every incoming vector
        before indexing. Leave ``False`` if your
        :class:`~sparksage.embed.client.EmbeddingClient` already normalizes
        (the default) -- matching :class:`InMemoryVectorStore` semantics.

    Examples
    --------
    >>> from sparksage.embed.backends import FaissVectorStore
    >>> store = FaissVectorStore(dimension=3)
    >>> store.add("a", [1.0, 0.0, 0.0])
    >>> store.add("b", [0.0, 1.0, 0.0])
    >>> hits = store.search([1.0, 0.1, 0.0], k=2)
    >>> hits[0].block_id
    'a'
    """

    def __init__(self, dimension: int, *, normalize: bool = False) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "FaissVectorStore requires the 'faiss' and 'numpy' packages. "
                "Install them with: pip install 'sparksage[distill]'"
            ) from exc
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise TypeError("dimension must be an int")
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self._dimension = dimension
        self._normalize = normalize
        self._faiss = faiss
        self._np = np
        base = faiss.IndexFlatIP(dimension)
        self._index = faiss.IndexIDMap2(base)
        self._id_to_faiss: dict[str, int] = {}
        self._faiss_to_id: dict[int, str] = {}
        self._next_id = 0

    @property
    def dimension(self) -> int:
        """The fixed length of every vector this store accepts."""
        return self._dimension

    def _prepare(self, vector: list[float]) -> Any:
        if len(vector) != self._dimension:
            raise ValueError(
                f"vector has dimension {len(vector)}; store expects {self._dimension}"
            )
        arr = self._np.asarray(vector, dtype="float32")
        if self._normalize:
            norm = self._np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
        return arr

    def _allocate(self, block_id: str) -> int:
        fid = self._next_id
        self._next_id += 1
        self._id_to_faiss[block_id] = fid
        self._faiss_to_id[fid] = block_id
        return fid

    def _drop_faiss_id(self, fid: int) -> None:
        selector = self._faiss.IDSelectorBatch([int(fid)])
        self._index.remove_ids(selector)
        self._faiss_to_id.pop(fid, None)

    def add(self, block_id: str, vector: list[float]) -> None:
        """Store ``vector`` under ``block_id``, overwriting any prior value."""
        bid = str(block_id)
        existing = self._id_to_faiss.get(bid)
        if existing is not None:
            self._drop_faiss_id(existing)
            self._id_to_faiss.pop(bid, None)
        arr = self._prepare(vector)
        fid = self._allocate(bid)
        self._index.add_with_ids(
            arr.reshape(1, -1),
            self._np.asarray([fid], dtype="int64"),
        )

    def add_many(self, vectors: dict[str, list[float]]) -> None:
        """Bulk-add a ``{block_id: vector}`` mapping (overwrites on key clash).

        All entries are validated before any mutation, mirroring
        :meth:`InMemoryVectorStore.add_many
        <sparksage.embed.store.InMemoryVectorStore.add_many>`.
        """
        items = [(str(k), v) for k, v in vectors.items()]
        prepared = [(bid, self._prepare(v)) for bid, v in items]
        # remove existing first so overwrites don't leave stale ids
        stale: list[int] = []
        for bid, _ in prepared:
            existing = self._id_to_faiss.get(bid)
            if existing is not None:
                stale.append(existing)
                self._id_to_faiss.pop(bid, None)
        if stale:
            selector = self._faiss.IDSelectorBatch([int(i) for i in stale])
            self._index.remove_ids(selector)
            for fid in stale:
                self._faiss_to_id.pop(fid, None)
        if not prepared:
            return
        matrix = self._np.asarray([arr for _, arr in prepared], dtype="float32")
        ids = self._np.asarray(
            [self._allocate(bid) for bid, _ in prepared], dtype="int64"
        )
        self._index.add_with_ids(matrix, ids)

    def search(self, query: list[float], k: int = 10) -> list[SearchHit]:
        """Return the ``k`` most similar stored vectors, best (highest score) first.

        Score is the inner product of ``query`` with each stored vector -- which
        equals cosine similarity when both are L2-normalized. Returns fewer than
        ``k`` (or ``[]``) when the store holds fewer vectors.
        """
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an int")
        if k < 1:
            raise ValueError("k must be >= 1")
        arr = self._prepare(query)
        if len(self._id_to_faiss) == 0:
            return []
        k_eff = min(k, len(self._id_to_faiss))
        scores, ids = self._index.search(arr.reshape(1, -1), k_eff)
        hits: list[SearchHit] = []
        for score, fid in zip(scores[0], ids[0], strict=True):
            if fid == -1:
                continue
            bid = self._faiss_to_id.get(int(fid))
            if bid is not None:
                hits.append(SearchHit(block_id=bid, score=float(score)))
        return hits

    def remove(self, block_id: str) -> bool:
        """Remove the vector for ``block_id``. Return whether it was present."""
        bid = str(block_id)
        fid = self._id_to_faiss.pop(bid, None)
        if fid is None:
            return False
        self._drop_faiss_id(fid)
        return True

    def __contains__(self, block_id: object) -> bool:
        return str(block_id) in self._id_to_faiss

    def __len__(self) -> int:
        return len(self._id_to_faiss)

    def __repr__(self) -> str:
        return (
            f"FaissVectorStore(dimension={self._dimension}, "
            f"count={len(self._id_to_faiss)})"
        )
