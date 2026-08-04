"""SQLite-backed :class:`~sparksage.documents.store.DocumentStore`.

Durable, single-file document persistence over a stdlib :mod:`sqlite3`
connection -- no extra install, no server. The store owns two tables:

* ``{table}`` -- one row per :class:`~sparksage.documents.models.DocumentRecord`,
  with explicit columns for the fields used in filtering / listing
  (``doc_id`` PK, ``title``, ``summary``, ``body_markdown``, timestamps,
  ``content_hash``) plus ``source_json`` / ``metadata_json`` for the structured
  nested objects.
* ``{table}_tags`` -- a (``doc_id``, ``tag``) junction table powering exact-match
  tag filtering and the distinct-tag vocabulary, without the substring pitfalls
  of a JSON column.

The table name is regex-validated (it cannot be SQL-parameterized), mirroring the
:class:`~sparksage.embed.backends.PgvectorVectorStore` pattern. The connection is
opened with ``check_same_thread=False`` and every operation is serialized by a
:class:`threading.Lock`, so the store is safe to share across the FastAPI
threadpool / worker threads. Call :meth:`close` to release the connection.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sparksage._sqlite import SqliteMixin
from sparksage.documents.models import DocumentRecord
from sparksage.documents.store import _normalize_tags, _validate_pagination
from sparksage.schema.source import SourceRef

_logger = logging.getLogger(__name__)

_IN_MEMORY_PATH = ":memory:"


class SqliteDocumentStore(SqliteMixin):
    """Durable document store backed by a single SQLite database file.

    Parameters
    ----------
    path:
        Database file path (``str`` / :class:`pathlib.Path`). Defaults to
        ``":memory:"`` (an ephemeral in-process DB -- useful for tests, lost on
        close). Parent directories of a file path are created if missing.
    table:
        Name of the main table (default ``"documents"``). Validated against
        ``^[A-Za-z_][A-Za-z0-9_]*$``. A ``{table}_tags`` junction table is
        created alongside.

    Examples
    --------
    >>> from sparksage.documents import DocumentRecord, SqliteDocumentStore
    >>> store = SqliteDocumentStore()           # in-memory
    >>> rec = DocumentRecord(body_markdown="body", source={"uri": "a.md"})
    >>> saved = store.save(rec)
    >>> store.get(saved.doc_id) is not None
    True
    """

    def __init__(
        self,
        path: str | Path = _IN_MEMORY_PATH,
        *,
        table: str = "documents",
    ) -> None:
        self._tags_table = f"{table.strip()}_tags"
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
                    doc_id        TEXT PRIMARY KEY,
                    external_key  TEXT,
                    title         TEXT,
                    summary       TEXT,
                    body_markdown TEXT NOT NULL,
                    source_json   TEXT NOT NULL,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL,
                    content_hash  TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{{}}'
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{self._tags_table}" (
                    doc_id TEXT NOT NULL,
                    tag    TEXT NOT NULL,
                    PRIMARY KEY (doc_id, tag)
                )
                """
            )
            cols = {r[1] for r in cur.execute(f'PRAGMA table_info("{self._table}")')}
            if "external_key" not in cols:
                cur.execute(f'ALTER TABLE "{self._table}" ADD COLUMN external_key TEXT')
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_record(
        row: Any, tags: list[str]
    ) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row["doc_id"],
            external_key=row["external_key"],
            title=row["title"],
            summary=row["summary"],
            body_markdown=row["body_markdown"],
            tags=tags,
            source=SourceRef.model_validate_json(row["source_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            content_hash=row["content_hash"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def _fetch_tags(self, cur: Any, doc_ids: list[str]) -> dict[str, list[str]]:
        if not doc_ids:
            return {}
        placeholders = ",".join(["?"] * len(doc_ids))
        cur.execute(
            f"""
            SELECT doc_id, tag FROM "{self._tags_table}"
            WHERE doc_id IN ({placeholders})
            ORDER BY doc_id, tag
            """,
            doc_ids,
        )
        out: dict[str, list[str]] = {d: [] for d in doc_ids}
        for doc_id, tag in cur.fetchall():
            out.setdefault(doc_id, []).append(tag)
        return out

    # ------------------------------------------------------------------ #
    # DocumentStore protocol
    # ------------------------------------------------------------------ #
    def save(self, record: DocumentRecord) -> DocumentRecord:
        stored = record.model_copy(deep=True)
        source_json = stored.source.model_dump_json()
        metadata_json = json.dumps(stored.metadata, ensure_ascii=False)
        created_iso = stored.created_at.isoformat()
        updated_iso = stored.updated_at.isoformat()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                INSERT OR REPLACE INTO "{self._table}"
                    (doc_id, title, summary, body_markdown, source_json,
                     created_at, updated_at, content_hash, metadata_json,
                     external_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(stored.doc_id),
                    stored.title,
                    stored.summary,
                    stored.body_markdown,
                    source_json,
                    created_iso,
                    updated_iso,
                    stored.content_hash,
                    metadata_json,
                    stored.external_key,
                ),
            )
            cur.execute(
                f'DELETE FROM "{self._tags_table}" WHERE doc_id = ?',
                (str(stored.doc_id),),
            )
            if stored.tags:
                cur.executemany(
                    f'INSERT INTO "{self._tags_table}" (doc_id, tag) VALUES (?, ?)',
                    [(str(stored.doc_id), t) for t in stored.tags],
                )
            self._conn.commit()
        return stored.model_copy(deep=True)

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT * FROM "{self._table}" WHERE doc_id = ?',
                (str(doc_id),),
            )
            row = cur.fetchone()
            if row is None:
                return None
            tags = self._fetch_tags(cur, [str(doc_id)]).get(str(doc_id), [])
            return self._row_to_record(row, tags)

    def list(
        self,
        *,
        tag: str | None = None,
        tags: list[str] | None = None,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentRecord]:
        _validate_pagination(limit, offset)
        required = _normalize_tags(tag, tags)
        query_norm = q.strip().lower() if q else None

        where: list[str] = []
        params: list[Any] = []
        if required:
            placeholders = ",".join(["?"] * len(required))
            where.append(
                f'doc_id IN (SELECT doc_id FROM "{self._tags_table}" '
                f"WHERE tag IN ({placeholders}))"
            )
            params.extend(sorted(required))
        if query_norm is not None:
            where.append("(LOWER(title) LIKE ? OR LOWER(body_markdown) LIKE ?)")
            like = f"%{query_norm}%"
            params.extend([like, like])

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        sql = (
            f'SELECT * FROM "{self._table}" {where_sql} '
            "ORDER BY created_at DESC, doc_id DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            doc_ids = [r["doc_id"] for r in rows]
            tags_by_id = self._fetch_tags(cur, doc_ids)
            return [
                self._row_to_record(r, tags_by_id.get(r["doc_id"], [])) for r in rows
            ]

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT 1 FROM "{self._table}" WHERE doc_id = ?', (str(doc_id),)
            )
            existed = cur.fetchone() is not None
            cur.execute(f'DELETE FROM "{self._table}" WHERE doc_id = ?', (str(doc_id),))
            cur.execute(
                f'DELETE FROM "{self._tags_table}" WHERE doc_id = ?', (str(doc_id),)
            )
            self._conn.commit()
            return existed

    def count(self, *, tag: str | None = None, tags: list[str] | None = None) -> int:
        required = _normalize_tags(tag, tags)
        if not required:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(f'SELECT COUNT(*) AS c FROM "{self._table}"')
                return int(cur.fetchone()["c"])
        placeholders = ",".join(["?"] * len(required))
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT COUNT(DISTINCT t.doc_id) AS c '
                f'FROM "{self._tags_table}" t WHERE t.tag IN ({placeholders})',
                sorted(required),
            )
            return int(cur.fetchone()["c"])

    def list_tags(self) -> list[str]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT DISTINCT tag FROM "{self._tags_table}" ORDER BY tag ASC'
            )
            return [r["tag"] for r in cur.fetchall()]

    def __contains__(self, doc_id: object) -> bool:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT 1 FROM "{self._table}" WHERE doc_id = ?', (str(doc_id),)
            )
            return cur.fetchone() is not None

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"SqliteDocumentStore(table={self._table!r}, path={self._path!r})"


__all__ = ["SqliteDocumentStore"]
