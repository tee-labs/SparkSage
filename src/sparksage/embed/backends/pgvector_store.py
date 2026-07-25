"""pgvector-backed :class:`~sparksage.embed.store.VectorStore`.

pgvector is the vector similarity-search extension for PostgreSQL -- the
natural choice when IdeaBlock embeddings should live alongside the rest of an
application's data (Supabase, RDS, Cloud SQL, self-hosted Postgres). This
backend wraps a single ``vector(dimension)`` table behind a ``psycopg``
connection and implements the same :class:`VectorStore` protocol the
retrieval/Distill cores depend on -- swap it in anywhere an
:class:`InMemoryVectorStore` is used.

The table uses the pgvector **cosine** distance operator ``<=>`` (``1 -
cosine_similarity``); since every
:class:`~sparksage.embed.client.EmbeddingClient` L2-normalizes by default,
cosine == dot product, so the returned
:class:`~sparksage.embed.store.SearchHit` scores (``1 - distance``) are directly
comparable to the dot products returned by
:class:`InMemoryVectorStore.search`.

Vectors are sent to Postgres as the pgvector text input form ``[a,b,c]`` -- no
dependency on the separate ``pgvector`` python adapter is needed, only
``psycopg`` (v3) and a Postgres server with the ``vector`` extension enabled.

The ``psycopg`` package is an *optional* dependency -- install it with
``pip install 'sparksage[pgvector]'``.
"""

from __future__ import annotations

import re
from typing import Any

from sparksage.embed.store import SearchHit

#: Validates a SQL identifier (table name) to prevent injection -- table names
#: cannot be parameterized in SQL, so they are constrained to a safe charset.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PgvectorVectorStore:
    """Vector store backed by a Postgres + pgvector ``vector(d)`` table.

    Creates (``CREATE EXTENSION IF NOT EXISTS vector``) the ``vector``
    extension and a table ``(block_id TEXT PRIMARY KEY, embedding vector(d))``
    on construction, then serves the :class:`VectorStore` protocol over a
    single ``psycopg`` connection. Overwrites use
    ``INSERT ... ON CONFLICT DO UPDATE`` (idempotent upsert).

    The ``psycopg`` (v3) package is an *optional* dependency -- install it with
    ``pip install 'sparksage[pgvector]'``. The server must have the pgvector
    extension available (``CREATE EXTENSION vector``).

    Parameters
    ----------
    dimension:
        The fixed length every stored vector (and every query) must have.
        Must be ``>= 1``.
    dsn:
        libpq connection string (e.g. ``"postgresql://user:pass@host/db"``).
        Required when ``connection`` is not supplied; forwarded to
        ``psycopg.connect``.
    connection:
        A pre-built ``psycopg`` connection (takes precedence over ``dsn``).
        Inject one in tests to share a pool / transaction.
    table:
        Destination table name (default ``"sparksage_vectors"``). Constrained to
        ``[A-Za-z_][A-Za-z0-9_]*`` -- it cannot be SQL-parameterized.
    distance:
        pgvector distance operator name (default ``"cosine"``). One of
        ``"cosine"`` (``<=>``), ``"l2"`` (``<->``) or ``"ip"`` (``<#>``, inner
        product). With L2-normalized vectors all three rank identically; cosine
        keeps the returned score (``1 - distance``) directly comparable to the
        dot products from :meth:`InMemoryVectorStore.search
        <sparksage.embed.store.InMemoryVectorStore.search>`.
    **connect_kwargs:
        Extra keyword args forwarded to ``psycopg.connect`` (e.g. ``timeout``).

    Examples
    --------
    >>> from sparksage.embed.backends import PgvectorVectorStore
    >>> store = PgvectorVectorStore(dimension=3, dsn="postgresql:///mydb")
    >>> store.add("a", [1.0, 0.0, 0.0])
    >>> store.add("b", [0.0, 1.0, 0.0])
    >>> hits = store.search([1.0, 0.1, 0.0], k=2)
    >>> hits[0].block_id
    'a'
    """

    #: pgvector operator by distance name.
    _OPERATORS: dict[str, str] = {
        "cosine": "<=>",
        "l2": "<->",
        "ip": "<#>",
    }

    def __init__(
        self,
        dimension: int,
        *,
        dsn: str | None = None,
        connection: Any | None = None,
        table: str = "sparksage_vectors",
        distance: str = "cosine",
        **connect_kwargs: Any,
    ) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "PgvectorVectorStore requires the 'psycopg' package (v3). "
                "Install it with: pip install 'sparksage[pgvector]'"
            ) from exc
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise TypeError("dimension must be an int")
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        if not _IDENT_RE.match(table):
            raise ValueError(
                f"invalid table name {table!r}: must match [A-Za-z_][A-Za-z0-9_]*"
            )
        if distance not in self._OPERATORS:
            raise ValueError(
                f"distance must be one of {sorted(self._OPERATORS)}, got {distance!r}"
            )
        self._dimension = dimension
        self._table = table
        self._distance = distance
        self._operator = self._OPERATORS[distance]
        if connection is None:
            if dsn is None:
                raise ValueError("either 'dsn' or 'connection' must be provided")
            connection = psycopg.connect(dsn, **connect_kwargs)
            self._owns_conn = True
        else:
            self._owns_conn = False
        self._conn = connection
        self._ensure_schema()

    @property
    def dimension(self) -> int:
        """The fixed length of every vector this store accepts."""
        return self._dimension

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} "
                f"(block_id TEXT PRIMARY KEY, embedding vector({self._dimension}))"
            )
            # cosine index for the default (and most common) metric; harmless if
            # the caller picks another distance, and cheap to skip on conflict.
            if self._distance == "cosine":
                idx = f"{self._table}_embedding_cos_idx"
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx} "
                    f"ON {self._table} USING hnsw (embedding vector_cosine_ops)"
                )
        self._conn.commit()

    def _check_dim(self, vector: list[float], what: str = "vector") -> None:
        if len(vector) != self._dimension:
            raise ValueError(
                f"{what} has dimension {len(vector)}; store expects {self._dimension}"
            )

    @staticmethod
    def _to_pg_vector(vector: list[float]) -> str:
        # pgvector text input form: [a,b,c] -- accepted by an implicit text ->
        # vector cast, so no separate python adapter dependency is needed.
        return "[" + ",".join(repr(float(x)) for x in vector) + "]"

    def add(self, block_id: str, vector: list[float]) -> None:
        """Store ``vector`` under ``block_id``, overwriting any prior value."""
        self._check_dim(vector)
        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self._table} (block_id, embedding) VALUES (%s, %s) "
                f"ON CONFLICT (block_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                (str(block_id), self._to_pg_vector(vector)),
            )
        self._conn.commit()

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
        rows = [(bid, self._to_pg_vector(vec)) for bid, vec in items]
        with self._conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self._table} (block_id, embedding) VALUES (%s, %s) "
                f"ON CONFLICT (block_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                rows,
            )
        self._conn.commit()

    def search(self, query: list[float], k: int = 10) -> list[SearchHit]:
        """Return the ``k`` most similar stored vectors, best (highest score) first.

        Score is ``1 - distance`` (cosine similarity with the default ``cosine``
        operator), so it is directly comparable to the dot products returned by
        :meth:`InMemoryVectorStore.search
        <sparksage.embed.store.InMemoryVectorStore.search>`. Returns fewer than
        ``k`` (or ``[]``) when the store holds fewer vectors.
        """
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an int")
        if k < 1:
            raise ValueError("k must be >= 1")
        self._check_dim(query, "query")
        pg_query = self._to_pg_vector(query)
        op = self._operator
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT block_id, 1 - (embedding {op} %s::vector) AS score "
                f"FROM {self._table} "
                f"ORDER BY embedding {op} %s::vector LIMIT %s",
                (pg_query, pg_query, k),
            )
            rows = cur.fetchall()
        return [SearchHit(block_id=str(r[0]), score=float(r[1])) for r in rows]

    def remove(self, block_id: str) -> bool:
        """Remove the vector for ``block_id``. Return whether it was present."""
        bid = str(block_id)
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM {self._table} WHERE block_id = %s", (bid,)
            )
            existed = cur.fetchone() is not None
            if existed:
                cur.execute(f"DELETE FROM {self._table} WHERE block_id = %s", (bid,))
        self._conn.commit()
        return existed

    def close(self) -> None:
        """Close the underlying connection when this store owns it (built via
        ``dsn``). Injected connections are left to the caller."""
        if self._owns_conn:
            try:
                self._conn.close()
            finally:
                self._owns_conn = False

    def __contains__(self, block_id: object) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM {self._table} WHERE block_id = %s",
                (str(block_id),),
            )
            return cur.fetchone() is not None

    def __len__(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self._table}")
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def __repr__(self) -> str:
        return (
            f"PgvectorVectorStore(dimension={self._dimension}, "
            f"count={len(self)}, table={self._table!r}, distance={self._distance!r})"
        )
