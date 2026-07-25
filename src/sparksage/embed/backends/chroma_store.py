"""ChromaDB-backed :class:`~sparksage.embed.store.VectorStore`.

ChromaDB is the local-development-first vector database: it runs in-process
with no external service (an ephemeral in-memory client or a persistent
on-disk client), which makes it the easiest way to get IdeaBlock embeddings
out of a :class:`~sparksage.embed.store.InMemoryVectorStore` and into something
that survives restarts without standing up Postgres.

This backend wraps a single Chroma collection and implements the same
:class:`VectorStore` protocol the retrieval/Distill cores depend on -- swap it
in anywhere an :class:`InMemoryVectorStore` is used. The collection uses the
``cosine`` distance metric; since every
:class:`~sparksage.embed.client.EmbeddingClient` L2-normalizes by default,
Chroma's cosine distance ``d`` is reported back as similarity ``1 - d`` so the
returned :class:`~sparksage.embed.store.SearchHit` scores are directly
comparable to the dot products returned by
:class:`InMemoryVectorStore.search`.

The ``chromadb`` package is an *optional* dependency -- install it with
``pip install 'sparksage[chroma]'``.
"""

from __future__ import annotations

from typing import Any

from sparksage.embed.store import SearchHit

#: Default Chroma distance metric. Cosine matches the dot-product-on-normalized
#: convention used across SparkSage, so scores are comparable across backends.
DEFAULT_CHROMA_SPACE = "cosine"


class ChromaVectorStore:
    """Vector store backed by a ChromaDB collection.

    Works with any Chroma client (``chromadb.Client`` for an in-process
    ephemeral store in tests, ``chromadb.PersistentClient`` for on-disk
    persistence across restarts, or ``chromadb.HttpClient`` for a remote
    server). When neither ``client`` nor ``path`` is given, an ephemeral
    in-process client is created -- handy for quick experiments and tests.

    The block id is stored verbatim as the Chroma document id; vectors are
    stored as embeddings. Overwrites use Chroma's ``upsert`` (idempotent).

    The ``chromadb`` package is an *optional* dependency -- install it with
    ``pip install 'sparksage[chroma]'``.

    Parameters
    ----------
    dimension:
        The fixed length every stored vector (and every query) must have.
        Must be ``>= 1``. (Chroma itself is dimension-agnostic; this is enforced
        on the SparkSage side so a :class:`VectorStore` stays self-consistent.)
    collection_name:
        Chroma collection name (default ``"sparksage"``).
    path:
        On-disk directory for ``chromadb.PersistentClient``. When given, a
        persistent client is created and embeddings survive restarts. Ignored
        when ``client`` is supplied.
    client:
        A pre-built Chroma client (takes precedence over ``path``). Inject one
        in tests to avoid touching the filesystem.
    space:
        Chroma ``hnsw:space`` metric (default ``"cosine"``). With L2-normalized
        vectors cosine, dot product and inner product rank identically; cosine
        is the most broadly supported across Chroma versions.

    Examples
    --------
    >>> from sparksage.embed.backends import ChromaVectorStore
    >>> store = ChromaVectorStore(dimension=3)  # ephemeral in-process client
    >>> store.add("a", [1.0, 0.0, 0.0])
    >>> store.add("b", [0.0, 1.0, 0.0])
    >>> hits = store.search([1.0, 0.1, 0.0], k=2)
    >>> hits[0].block_id
    'a'
    """

    def __init__(
        self,
        dimension: int,
        *,
        collection_name: str = "sparksage",
        path: str | None = None,
        client: Any | None = None,
        space: str = DEFAULT_CHROMA_SPACE,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "ChromaVectorStore requires the 'chromadb' package. "
                "Install it with: pip install 'sparksage[chroma]'"
            ) from exc
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise TypeError("dimension must be an int")
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self._dimension = dimension
        self._space = space
        if client is None:
            if path is not None:
                client = chromadb.PersistentClient(path=str(path))
            else:
                client = chromadb.Client()
        self._client = client
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": space},
        )

    @property
    def dimension(self) -> int:
        """The fixed length of every vector this store accepts."""
        return self._dimension

    @property
    def collection(self) -> Any:
        """The underlying Chroma collection (for advanced/inspection use)."""
        return self._collection

    def _check_dim(self, vector: list[float], what: str = "vector") -> None:
        if len(vector) != self._dimension:
            raise ValueError(
                f"{what} has dimension {len(vector)}; store expects {self._dimension}"
            )

    def _score(self, distance: float) -> float:
        # cosine distance d in [0, 2] -> similarity 1 - d (== dot product on
        # L2-normalized vectors). For l2 the caller picked a different space,
        # so we still report 1 - d (larger = closer) which keeps "best first"
        # consistent; distance semantics are the caller's responsibility.
        return 1.0 - float(distance)

    def add(self, block_id: str, vector: list[float]) -> None:
        """Store ``vector`` under ``block_id``, overwriting any prior value."""
        self._check_dim(vector)
        self._collection.upsert(
            ids=[str(block_id)], embeddings=[list(map(float, vector))]
        )

    def add_many(self, vectors: dict[str, list[float]]) -> None:
        """Bulk-add a ``{block_id: vector}`` mapping (overwrites on key clash).

        All entries are validated before any mutation, mirroring
        :meth:`InMemoryVectorStore.add_many
        <sparksage.embed.store.InMemoryVectorStore.add_many>`.
        """
        items = [(str(k), v) for k, v in vectors.items()]
        for _, vec in items:
            self._check_dim(vec)
        if not items:
            return
        self._collection.upsert(
            ids=[bid for bid, _ in items],
            embeddings=[list(map(float, v)) for _, v in items],
        )

    def search(self, query: list[float], k: int = 10) -> list[SearchHit]:
        """Return the ``k`` most similar stored vectors, best (highest score) first.

        Score is ``1 - distance`` (cosine similarity with the default ``cosine``
        space), so it is directly comparable to the dot products returned by
        :meth:`InMemoryVectorStore.search
        <sparksage.embed.store.InMemoryVectorStore.search>`. Returns fewer than
        ``k`` (or ``[]``) when the store holds fewer vectors.
        """
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an int")
        if k < 1:
            raise ValueError("k must be >= 1")
        self._check_dim(query, "query")
        if len(self) == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(map(float, query))],
            n_results=min(k, len(self)),
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            SearchHit(block_id=bid, score=self._score(dist))
            for bid, dist in zip(ids, distances, strict=False)
        ]

    def remove(self, block_id: str) -> bool:
        """Remove the vector for ``block_id``. Return whether it was present."""
        bid = str(block_id)
        existed = bid in self
        self._collection.delete(ids=[bid])
        return existed

    def __contains__(self, block_id: object) -> bool:
        got = self._collection.get(ids=[str(block_id)])
        return len(got.get("ids", [])) > 0

    def __len__(self) -> int:
        return int(self._collection.count())

    def __repr__(self) -> str:
        return (
            f"ChromaVectorStore(dimension={self._dimension}, "
            f"count={len(self)}, space={self._space!r})"
        )
