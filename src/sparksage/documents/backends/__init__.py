"""Concrete :class:`~sparksage.documents.store.DocumentStore` backends.

Each backend implements the same
:class:`~sparksage.documents.store.DocumentStore` protocol and lazily imports any
SDK it needs, so the document core stays zero-dependency -- install only what
you use.

* :class:`InMemoryDocumentStore` -- pure-stdlib dict-backed store for tests,
  single-node demos and ephemeral runs. Defensive copies on read/write so
  callers cannot corrupt the store by mutating a returned record.
* :class:`SqliteDocumentStore` -- durable single-file persistence over a
  ``sqlite3`` connection (also stdlib, no extra install). Owns a ``documents``
  table plus a ``<table>_tags`` junction table for exact-match tag filtering.
"""

from __future__ import annotations

from sparksage.documents.backends.memory import InMemoryDocumentStore
from sparksage.documents.backends.sqlite import SqliteDocumentStore

__all__ = [
    "InMemoryDocumentStore",
    "SqliteDocumentStore",
]
