"""In-memory dense-vector store with k-nearest-neighbour similarity search.

This is the retrieval layer that consumes the vectors produced by
:class:`~sparksage.embed.indexer.BlockEmbedder`. Feed it the
``{block_id: vector}`` mapping returned by
:meth:`BlockEmbedder.vectors_for <sparksage.embed.indexer.BlockEmbedder.vectors_for>`
(or add vectors one at a time) and call :meth:`InMemoryVectorStore.search` to get
the ``k`` most similar block ids ranked by score.

Design mirrors the rest of SparkSage:

* The store depends *only* on the :class:`VectorStore` protocol, so it is
  decoupled from any embedding SDK or vector-DB backend. The brute-force
  :class:`InMemoryVectorStore` is pure Python (no ``numpy`` / ``faiss`` -- those
  belong to the future ``[distill]`` group) and fully unit-testable offline.
* Vectors are assumed **L2-normalized** (every :class:`~sparksage.embed.client.EmbeddingClient`
  normalizes by default), so cosine similarity reduces to a plain dot product
  and :meth:`InMemoryVectorStore.search` just computes dot products. Store
  un-normalized vectors and the ranking is still correct *as a dot product* --
  the caller decides the metric by pre-normalizing.
* The store is **text-agnostic**: it indexes vectors keyed by an opaque
  ``block_id`` string. Embedding a query string into a vector is the caller's
  job (one :meth:`~sparksage.embed.client.EmbeddingClient.embed_batch` call) --
  this keeps retrieval decoupled from embedding, exactly like the generator is
  decoupled from the LLM client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SearchHit:
    """A single similarity-search result.

    Attributes
    ----------
    block_id:
        The opaque key the vector was stored under (an :class:`~sparksage.schema.IdeaBlock`
        id as a string when fed from :meth:`~sparksage.embed.indexer.BlockEmbedder.vectors_for`).
    score:
        Similarity of the stored vector to the query. A dot product -- which
        equals cosine similarity when both vectors are L2-normalized.
    """

    block_id: str
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """Minimal keyed dense-vector store with kNN search.

    Any container that maps an opaque string key to a fixed-length vector and
    supports ranked similarity search implements this -- the brute-force
    :class:`InMemoryVectorStore` in-process, or a future FAISS / vector-DB
    backend in production. The Distill pipeline and any RAG retriever depend on
    this protocol, never on a concrete implementation.
    """

    @property
    def dimension(self) -> int:
        """The length of every vector this store accepts."""
        ...

    def add(self, block_id: str, vector: list[float]) -> None:
        """Store ``vector`` under ``block_id`` (overwriting any prior value)."""
        ...

    def search(self, query: list[float], k: int = 10) -> list[SearchHit]:
        """Return the ``k`` most similar stored vectors, best first.

        Results are :class:`SearchHit` instances sorted by descending score.
        Fewer than ``k`` may be returned when the store holds fewer vectors.
        """
        ...

    def __contains__(self, block_id: object) -> bool:
        """Whether ``block_id`` has a vector stored."""
        ...

    def __len__(self) -> int:
        """Number of vectors currently stored."""
        ...


def _dot(a: list[float], b: list[float]) -> float:
    """Pure-Python dot product (no numpy needed)."""
    return sum(x * y for x, y in zip(a, b, strict=True))


class InMemoryVectorStore:
    """Brute-force, dependency-free dense-vector store for retrieval.

    Searches are exact k-nearest-neighbour over a plain dict of vectors:
    compute the dot product of the query against every stored vector, then keep
    the top ``k``. This is ``O(n * d)`` per query -- fine for thousands of
    blocks; for millions, swap in a FAISS-backed :class:`VectorStore` (planned
    under the ``[distill]`` extra). The core never pays that cost until it needs
    to, and stays unit-testable with zero dependencies meanwhile.

    Vectors are stored **by value** (copied on add) so later mutation of the
    caller's list cannot corrupt the index. Keys are opaque strings; pass the
    ``str(block.id)`` from :meth:`~sparksage.embed.indexer.BlockEmbedder.vectors_for`.

    Parameters
    ----------
    dimension:
        The fixed length every stored vector (and every query) must have.
        Must be ``>= 1``.

    Examples
    --------
    >>> from sparksage import InMemoryVectorStore
    >>> store = InMemoryVectorStore(dimension=3)
    >>> store.add("a", [1.0, 0.0, 0.0])
    >>> store.add("b", [0.0, 1.0, 0.0])
    >>> hits = store.search([1.0, 0.1, 0.0], k=2)
    >>> hits[0].block_id
    'a'
    """

    def __init__(self, dimension: int) -> None:
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise TypeError("dimension must be an int")
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self._dimension = dimension
        self._vectors: dict[str, list[float]] = {}

    @property
    def dimension(self) -> int:
        """The fixed length of every vector this store accepts."""
        return self._dimension

    def _check_dim(self, vector: list[float], what: str) -> None:
        if len(vector) != self._dimension:
            raise ValueError(
                f"{what} has dimension {len(vector)}; store expects {self._dimension}"
            )

    def add(self, block_id: str, vector: list[float]) -> None:
        """Store ``vector`` under ``block_id``, overwriting any prior value.

        A defensive copy is taken so the caller's list can be reused or mutated
        without corrupting the index.
        """
        self._check_dim(vector, "vector")
        self._vectors[str(block_id)] = list(vector)

    def add_many(self, vectors: dict[str, list[float]]) -> None:
        """Bulk-add a ``{block_id: vector}`` mapping (e.g. straight from
        :meth:`~sparksage.embed.indexer.BlockEmbedder.vectors_for`).

        Each entry is validated individually, so a single bad vector fails fast
        without partially mutating the store before it (the whole call is
        validated, then applied).
        """
        items = list(vectors.items())
        for _bid, vec in items:
            self._check_dim(vec, "vector")
        for bid, vec in items:
            self._vectors[str(bid)] = list(vec)

    def search(self, query: list[float], k: int = 10) -> list[SearchHit]:
        """Return the ``k`` most similar stored vectors, best (highest score) first.

        Score is the dot product of ``query`` with each stored vector -- which
        equals cosine similarity when both are L2-normalized (the default for
        every :class:`~sparksage.embed.client.EmbeddingClient`). Returns fewer
        than ``k`` (or ``[]``) when the store holds fewer vectors.
        """
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an int")
        if k < 1:
            raise ValueError("k must be >= 1")
        self._check_dim(query, "query")
        if not self._vectors:
            return []
        scored = [
            (block_id, _dot(query, vec)) for block_id, vec in self._vectors.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        top = scored[:k]
        return [SearchHit(block_id=bid, score=score) for bid, score in top]

    def get(self, block_id: str) -> list[float] | None:
        """Return a copy of the vector stored under ``block_id``, or ``None``."""
        vec = self._vectors.get(str(block_id))
        return list(vec) if vec is not None else None

    def remove(self, block_id: str) -> bool:
        """Remove the vector for ``block_id``. Return whether it was present."""
        return self._vectors.pop(str(block_id), None) is not None

    def clear(self) -> None:
        """Drop every stored vector (dimension is kept)."""
        self._vectors.clear()

    def vectors(self) -> dict[str, list[float]]:
        """Return a shallow-copy snapshot ``{block_id: vector}`` of the store.

        Vectors themselves are copied, so mutating the result never touches the
        index. Used by :mod:`sparksage.embed.persist` to serialize the store.
        """
        return {bid: list(vec) for bid, vec in self._vectors.items()}

    def __contains__(self, block_id: object) -> bool:
        return str(block_id) in self._vectors

    def __len__(self) -> int:
        return len(self._vectors)

    def __repr__(self) -> str:
        return (
            f"InMemoryVectorStore(dimension={self._dimension}, "
            f"count={len(self._vectors)})"
        )
