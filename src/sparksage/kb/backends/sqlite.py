"""SQLite-backed :class:`~sparksage.kb.store.KnowledgeBaseStore`.

Durable, single-file persistence for :class:`~sparksage.kb.models.KnowledgeBaseInfo`
metadata over a stdlib :mod:`sqlite3` connection -- no extra install, no server.
This is the multi-tenant counterpart of
:class:`~sparksage.documents.backends.sqlite.SqliteDocumentStore`: where that
stores documents, this stores knowledge-base metadata (id / name / language /
ACL / timestamps) keyed by ``kb_id``.

Only the *metadata* is persisted here. The live vector index + block registry
are runtime state owned by the :class:`~sparksage.kb.KnowledgeBase` aggregate;
they are persisted separately via a :class:`~sparksage.kb.backends.state.KbStateStore`
(blocks + vectors + doc-links) and rebuilt on load.

The table name is regex-validated (it cannot be SQL-parameterized), mirroring
the :class:`~sparksage.documents.backends.sqlite.SqliteDocumentStore` pattern.
The connection is opened with ``check_same_thread=False`` and every operation is
serialized by a :class:`threading.RLock`, so the store is safe to share across
the FastAPI threadpool / worker threads. Call :meth:`close` to release the
connection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sparksage._sqlite import SqliteMixin
from sparksage.kb.models import KnowledgeBaseInfo

_logger = logging.getLogger(__name__)

_IN_MEMORY_PATH = ":memory:"


class SqliteKnowledgeBaseStore(SqliteMixin):
    """Durable knowledge-base metadata store backed by a single SQLite file.

    Parameters
    ----------
    path:
        Database file path (``str`` / :class:`pathlib.Path`). Defaults to
        ``":memory:"`` (an ephemeral in-process DB -- useful for tests, lost on
        close). Parent directories of a file path are created if missing.
    table:
        Name of the main table (default ``"knowledge_bases"``). Validated
        against ``^[A-Za-z_][A-Za-z0-9_]*$``.

    Examples
    --------
    >>> from sparksage.kb import KnowledgeBaseInfo
    >>> from sparksage.kb.backends import SqliteKnowledgeBaseStore
    >>> store = SqliteKnowledgeBaseStore()           # in-memory
    >>> saved = store.save(KnowledgeBaseInfo(name="ops"))
    >>> store.get(saved.kb_id) is not None
    True
    """

    def __init__(
        self,
        path: str | Path = _IN_MEMORY_PATH,
        *,
        table: str = "knowledge_bases",
    ) -> None:
        self._open(path, table)

    # ------------------------------------------------------------------ #
    # schema
    # ------------------------------------------------------------------ #
    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{self._table}" (
                    kb_id       TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    info_json   TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_info(row: Any) -> KnowledgeBaseInfo:
        return KnowledgeBaseInfo.model_validate_json(row["info_json"])

    # ------------------------------------------------------------------ #
    # KnowledgeBaseStore protocol
    # ------------------------------------------------------------------ #
    def save(self, info: KnowledgeBaseInfo) -> KnowledgeBaseInfo:
        stored = info.model_copy(deep=True)
        info_json = stored.model_dump_json()
        created_iso = stored.created_at.isoformat()
        updated_iso = stored.updated_at.isoformat()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                INSERT OR REPLACE INTO "{self._table}"
                    (kb_id, name, created_at, updated_at, info_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(stored.kb_id),
                    stored.name,
                    created_iso,
                    updated_iso,
                    info_json,
                ),
            )
            self._conn.commit()
        return stored.model_copy(deep=True)

    def get(self, kb_id: str) -> KnowledgeBaseInfo | None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT * FROM "{self._table}" WHERE kb_id = ?',
                (str(kb_id),),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_info(row)

    def list(self, *, limit: int = 100, offset: int = 0) -> list[KnowledgeBaseInfo]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive int")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a non-negative int")
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT * FROM "{self._table}" '
                "ORDER BY created_at DESC, kb_id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = cur.fetchall()
            return [self._row_to_info(r) for r in rows]

    def delete(self, kb_id: str) -> bool:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT 1 FROM "{self._table}" WHERE kb_id = ?', (str(kb_id),)
            )
            existed = cur.fetchone() is not None
            cur.execute(
                f'DELETE FROM "{self._table}" WHERE kb_id = ?', (str(kb_id),)
            )
            self._conn.commit()
            return existed

    def __contains__(self, kb_id: object) -> bool:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT 1 FROM "{self._table}" WHERE kb_id = ?', (str(kb_id),)
            )
            return cur.fetchone() is not None

    def __len__(self) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(f'SELECT COUNT(*) AS c FROM "{self._table}"')
            return int(cur.fetchone()["c"])

    def __repr__(self) -> str:
        return f"SqliteKnowledgeBaseStore(table={self._table!r}, path={self._path!r})"


__all__ = ["SqliteKnowledgeBaseStore"]
