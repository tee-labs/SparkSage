"""Durable persistence for the live :class:`~sparksage.kb.KnowledgeBase` state.

The :class:`~sparksage.kb.store.KnowledgeBaseStore` persists only KB *metadata*.
A knowledge base also owns runtime state -- the block registry, the dense vector
index, and the document<->block linkage that makes cascade-delete work. That
state is what this module persists so a :class:`~sparksage.kb.KnowledgeBase`
aggregate can be reconstructed after a process restart without re-running the
expensive convert -> generate -> embed ingest pipeline.

Concretely, a :class:`KbStateStore` persists three things keyed by ``kb_id``:

* the :class:`~sparksage.schema.IdeaBlock` registry (each block serialized to
  JSON, **including its dense ``embedding``** so re-indexing after load never
  calls the embedding API again),
* the ``doc_id`` -> ``{block_id}`` linkage (so :meth:`remove_document` cascade
  and :meth:`blocks_for_document` work exactly as before after a restart), and
* the ``kb_id`` membership stamp on every block.

The default :class:`SqliteKbStateStore` is a single-file stdlib :mod:`sqlite3`
database (no extra install, no server), mirroring
:class:`~sparksage.documents.backends.sqlite.SqliteDocumentStore`. Two tables:

* ``{table}_blocks``  -- one row per block (``kb_id``, ``block_id``, ``doc_id``,
  ``block_json``). The full block JSON is the source of truth; columns
  ``kb_id`` / ``block_id`` / ``doc_id`` exist only for indexed delete/lookup.
* ``{table}_doc_links`` -- a (``kb_id``, ``doc_id``, ``block_id``) junction
  table powering cascade delete reconstruction.

A :class:`~sparksage.kb.KnowledgeBase` constructed with a ``state_store=``
writes through on every mutation (``add_blocks`` / ``remove_block`` /
``remove_document``) and loads its initial state on construction, so the
aggregate's consistency guarantee extends across restarts.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from sparksage.schema.ideablock import IdeaBlock

_logger = logging.getLogger(__name__)

#: A table name prefix must be a plain SQL identifier -- it cannot be passed as
#: a parameter, so it is regex-validated before being interpolated into SQL.
_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_IN_MEMORY_PATH = ":memory:"


@dataclass
class KbStateSnapshot:
    """A serializable snapshot of a knowledge base's runtime state.

    Attributes
    ----------
    blocks:
        Every :class:`~sparksage.schema.IdeaBlock` in the registry (each
        carrying its dense ``embedding`` when one was produced).
    doc_links:
        ``{doc_id: {block_id, ...}}`` so the document<->block linkage (and
        therefore cascade delete + ``blocks_for_document``) survives a restart.
    """

    blocks: list[IdeaBlock] = field(default_factory=list)
    doc_links: dict[str, set[str]] = field(default_factory=dict)


@runtime_checkable
class KbStateStore(Protocol):
    """Persistence for the live block registry + vector index + doc links."""

    def upsert_block(
        self, kb_id: str, block: IdeaBlock, doc_id: str | None
    ) -> None:
        """Insert or replace ``block`` under ``kb_id`` (and link it to ``doc_id``)."""
        ...

    def delete_block(self, kb_id: str, block_id: str) -> bool:
        """Remove ``block_id`` from ``kb_id``. Return whether it was present."""
        ...

    def unlink_doc(self, kb_id: str, doc_id: str) -> int:
        """Drop every doc-link row for ``doc_id``. Return how many were dropped."""
        ...

    def clear(self, kb_id: str) -> None:
        """Drop every block + doc-link row for ``kb_id``."""
        ...

    def load(self, kb_id: str) -> KbStateSnapshot:
        """Load the full snapshot for ``kb_id`` (empty when absent)."""
        ...


class SqliteKbStateStore:
    """Durable block + vector + doc-link store backed by a single SQLite file.

    Parameters
    ----------
    path:
        Database file path (``str`` / :class:`pathlib.Path`). Defaults to
        ``":memory:"`` (ephemeral, lost on close). Parent directories of a file
        path are created if missing. Sharing the same file as the
        :class:`~sparksage.kb.backends.sqlite.SqliteKnowledgeBaseStore` is fine:
        each owns a different table prefix.
    table:
        Name prefix for the two tables (default ``"kb_state"``). Two tables are
        created: ``{table}_blocks`` and ``{table}_doc_links``. Validated against
        ``^[A-Za-z_][A-Za-z0-9_]*$``.

    Examples
    --------
    >>> from sparksage import FakeEmbeddingClient, BlockEmbedder, IdeaBlock, Tag
    >>> from sparksage.kb.backends.state import SqliteKbStateStore
    >>> store = SqliteKbStateStore()           # in-memory
    >>> embedder = BlockEmbedder(FakeEmbeddingClient(dimension=8))
    >>> block = IdeaBlock(name="A", critical_question="what?",
    ...                    trusted_answer="answer", tags=[Tag.IMPORTANT])
    >>> embedder.embed_blocks([block])
    [IdeaBlock(...)]
    >>> store.upsert_block("kb1", block, "doc1")
    >>> snap = store.load("kb1")
    >>> len(snap.blocks)
    1
    >>> snap.doc_links
    {'doc1': {'...'}}
    """

    def __init__(
        self,
        path: str | Path = _IN_MEMORY_PATH,
        *,
        table: str = "kb_state",
    ) -> None:
        table_norm = str(table).strip()
        if not _TABLE_NAME_RE.match(table_norm):
            raise ValueError(
                f"invalid table name {table!r}: must match ^[A-Za-z_][A-Za-z0-9_]*$"
            )
        self._table = table_norm
        self._blocks_table = f"{table_norm}_blocks"
        self._links_table = f"{table_norm}_doc_links"

        path_str = str(path)
        if path_str != _IN_MEMORY_PATH:
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        self._path = path_str
        self._conn = self._connect(path_str)
        self._lock = threading.RLock()
        self._init_schema()

    @staticmethod
    def _connect(path: str) -> Any:
        import sqlite3

        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ #
    # schema
    # ------------------------------------------------------------------ #
    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{self._blocks_table}" (
                    kb_id       TEXT NOT NULL,
                    block_id    TEXT NOT NULL,
                    doc_id      TEXT,
                    block_json  TEXT NOT NULL,
                    PRIMARY KEY (kb_id, block_id)
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{self._links_table}" (
                    kb_id     TEXT NOT NULL,
                    doc_id    TEXT NOT NULL,
                    block_id  TEXT NOT NULL,
                    PRIMARY KEY (kb_id, doc_id, block_id)
                )
                """
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #
    # KbStateStore protocol
    # ------------------------------------------------------------------ #
    def upsert_block(
        self, kb_id: str, block: IdeaBlock, doc_id: str | None
    ) -> None:
        kb_id = str(kb_id)
        block_id = str(block.id)
        block_json = block.model_dump_json()
        doc_val = str(doc_id) if doc_id is not None else None
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                INSERT OR REPLACE INTO "{self._blocks_table}"
                    (kb_id, block_id, doc_id, block_json)
                VALUES (?, ?, ?, ?)
                """,
                (kb_id, block_id, doc_val, block_json),
            )
            if doc_val is not None:
                cur.execute(
                    f'INSERT OR IGNORE INTO "{self._links_table}" '
                    "(kb_id, doc_id, block_id) VALUES (?, ?, ?)",
                    (kb_id, doc_val, block_id),
                )
            self._conn.commit()

    def delete_block(self, kb_id: str, block_id: str) -> bool:
        kb_id = str(kb_id)
        block_id = str(block_id)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT 1 FROM "{self._blocks_table}" '
                "WHERE kb_id = ? AND block_id = ?",
                (kb_id, block_id),
            )
            existed = cur.fetchone() is not None
            cur.execute(
                f'DELETE FROM "{self._blocks_table}" '
                "WHERE kb_id = ? AND block_id = ?",
                (kb_id, block_id),
            )
            cur.execute(
                f'DELETE FROM "{self._links_table}" '
                "WHERE kb_id = ? AND block_id = ?",
                (kb_id, block_id),
            )
            self._conn.commit()
            return existed

    def unlink_doc(self, kb_id: str, doc_id: str) -> int:
        kb_id = str(kb_id)
        doc_id = str(doc_id)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'DELETE FROM "{self._links_table}" '
                "WHERE kb_id = ? AND doc_id = ?",
                (kb_id, doc_id),
            )
            n = cur.rowcount
            cur.execute(
                f'UPDATE "{self._blocks_table}" SET doc_id = NULL '
                "WHERE kb_id = ? AND doc_id = ?",
                (kb_id, doc_id),
            )
            self._conn.commit()
            return int(n) if n is not None else 0

    def clear(self, kb_id: str) -> None:
        kb_id = str(kb_id)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'DELETE FROM "{self._blocks_table}" WHERE kb_id = ?', (kb_id,)
            )
            cur.execute(
                f'DELETE FROM "{self._links_table}" WHERE kb_id = ?', (kb_id,)
            )
            self._conn.commit()

    def load(self, kb_id: str) -> KbStateSnapshot:
        kb_id = str(kb_id)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT block_json FROM "{self._blocks_table}" '
                "WHERE kb_id = ? ORDER BY block_id",
                (kb_id,),
            )
            rows = cur.fetchall()
            blocks = [IdeaBlock.model_validate_json(r["block_json"]) for r in rows]
            cur.execute(
                f'SELECT doc_id, block_id FROM "{self._links_table}" '
                "WHERE kb_id = ? ORDER BY doc_id, block_id",
                (kb_id,),
            )
            doc_links: dict[str, set[str]] = {}
            for r in cur.fetchall():
                doc_links.setdefault(r["doc_id"], set()).add(r["block_id"])
            return KbStateSnapshot(blocks=blocks, doc_links=doc_links)

    def __repr__(self) -> str:
        return f"SqliteKbStateStore(table={self._table!r}, path={self._path!r})"


__all__ = [
    "KbStateSnapshot",
    "KbStateStore",
    "SqliteKbStateStore",
]
