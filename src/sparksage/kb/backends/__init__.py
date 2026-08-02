"""Concrete durable backends for the knowledge-base layer.

* :class:`SqliteKnowledgeBaseStore` -- durable single-file persistence for
  :class:`~sparksage.kb.models.KnowledgeBaseInfo` metadata over a stdlib
  ``sqlite3`` connection (the multi-tenant counterpart of
  :class:`~sparksage.documents.backends.sqlite.SqliteDocumentStore`).
* :class:`SqliteKbStateStore` -- durable persistence for the live block
  registry + vector index + document<->block linkage, so a
  :class:`~sparksage.kb.KnowledgeBase` aggregate can be reconstructed after a
  restart without re-running the ingest pipeline.

Both are stdlib-only (no extra install, no server) and safe to share a single
SQLite database file -- each owns a different table prefix.
"""

from __future__ import annotations

from sparksage.kb.backends.sqlite import SqliteKnowledgeBaseStore
from sparksage.kb.backends.state import (
    KbStateSnapshot,
    KbStateStore,
    SqliteKbStateStore,
)

__all__ = [
    "KbStateSnapshot",
    "KbStateStore",
    "SqliteKbStateStore",
    "SqliteKnowledgeBaseStore",
]
