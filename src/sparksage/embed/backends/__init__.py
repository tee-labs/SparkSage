"""Concrete :class:`~sparksage.embed.store.VectorStore` backends.

These are the production retrieval backends that complement the dependency-free
:class:`~sparksage.embed.store.InMemoryVectorStore`. Each implements the
:class:`~sparksage.embed.store.VectorStore` protocol and lazily imports its own
SDK, so the SparkSage core stays zero-dependency -- install only the backend you
need:

* :class:`FaissVectorStore` -- FAISS exact inner-product index for
  million-vector corpora (``pip install 'sparksage[distill]'``).
* :class:`ChromaVectorStore` -- ChromaDB collection, the local-dev-first
  vector database (``pip install 'sparksage[chroma]'``).
* :class:`PgvectorVectorStore` -- Postgres + pgvector ``vector(d)`` table for
  Supabase / managed Postgres (``pip install 'sparksage[pgvector]'``).

All three assume L2-normalized vectors (every
:class:`~sparksage.embed.client.EmbeddingClient` normalizes by default) and
report scores directly comparable to the dot products returned by
:meth:`InMemoryVectorStore.search
<sparksage.embed.store.InMemoryVectorStore.search>`.
"""

from __future__ import annotations

from sparksage.embed.backends.chroma_store import DEFAULT_CHROMA_SPACE, ChromaVectorStore
from sparksage.embed.backends.faiss_store import FaissVectorStore
from sparksage.embed.backends.pgvector_store import PgvectorVectorStore

__all__ = [
    "DEFAULT_CHROMA_SPACE",
    "ChromaVectorStore",
    "FaissVectorStore",
    "PgvectorVectorStore",
]
