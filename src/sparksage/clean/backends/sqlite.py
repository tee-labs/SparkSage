"""SQLite-backed :class:`~sparksage.clean.store.CleaningRuleStore`.

Durable, single-file persistence for
:class:`~sparksage.clean.models.CleaningRuleRecord` over a stdlib
:mod:`sqlite3` connection -- no extra install, no server. The cleaning
counterpart of :class:`~sparksage.documents.backends.sqlite.SqliteDocumentStore`
/ :class:`~sparksage.feedback.backends.sqlite.SqliteFeedbackStore`: where those
store documents / feedback signals, this stores the custom cleaning-rule
definitions so they survive a restart and reload into the live
:class:`~sparksage.clean.cleaner.TextCleaner`.

The full record is serialized to JSON (the source of truth); an explicit
``order_index`` column preserves application order (the registry applies rules
in registration order, so a stable reload order matters). The table name is
regex-validated (it cannot be SQL-parameterized), mirroring the other SQLite
backends. The connection is opened with ``check_same_thread=False`` and every
operation is serialized by a :class:`threading.RLock`, so the store is safe to
share across the FastAPI threadpool / worker threads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sparksage._sqlite import SqliteMixin
from sparksage.clean.models import CleaningRuleRecord

_logger = logging.getLogger(__name__)

_IN_MEMORY_PATH = ":memory:"


class SqliteCleaningRuleStore(SqliteMixin):
    """Durable cleaning-rule store backed by a single SQLite database file.

    Parameters
    ----------
    path:
        Database file path (``str`` / :class:`pathlib.Path`). Defaults to
        ``":memory:"`` (an ephemeral in-process DB -- useful for tests, lost on
        close). Parent directories of a file path are created if missing.
    table:
        Name of the main table (default ``"cleaning_rules"``). Validated against
        ``^[A-Za-z_][A-Za-z0-9_]*$``.

    Examples
    --------
    >>> from sparksage.clean import CleaningRuleRecord
    >>> from sparksage.clean.backends import SqliteCleaningRuleStore
    >>> store = SqliteCleaningRuleStore()           # in-memory
    >>> rec = store.add(CleaningRuleRecord(
    ...     name="redact",
    ...     code="def clean(text, source=None):\n    return text.replace('X', 'Y')\n",
    ... ))
    >>> store.get(rec.rule_id) is not None
    True
    """

    def __init__(
        self,
        path: str | Path = _IN_MEMORY_PATH,
        *,
        table: str = "cleaning_rules",
    ) -> None:
        self._counter = 0
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
                    rule_id      TEXT PRIMARY KEY,
                    order_index  INTEGER NOT NULL,
                    created_at   TEXT NOT NULL,
                    record_json  TEXT NOT NULL
                )
                """
            )
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS "ix_{self._table}_order" '
                f'ON "{self._table}" (order_index)'
            )
            cur.execute("SELECT COALESCE(MAX(order_index), -1) AS m FROM "
                f'"{self._table}"')
            row = cur.fetchone()
            self._counter = int(row["m"]) + 1 if row is not None else 0
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_record(row: Any) -> CleaningRuleRecord:
        return CleaningRuleRecord.model_validate_json(row["record_json"])

    def _next_order(self) -> int:
        idx = self._counter
        self._counter += 1
        return idx

    # ------------------------------------------------------------------ #
    # CleaningRuleStore protocol
    # ------------------------------------------------------------------ #
    def add(self, record: CleaningRuleRecord) -> CleaningRuleRecord:
        stored = record.model_copy(deep=True)
        record_json = stored.model_dump_json()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                INSERT OR REPLACE INTO "{self._table}"
                    (rule_id, order_index, created_at, record_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(stored.rule_id),
                    self._next_order(),
                    stored.created_at.isoformat(),
                    record_json,
                ),
            )
            self._conn.commit()
        return stored.model_copy(deep=True)

    def get(self, rule_id: str) -> CleaningRuleRecord | None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT * FROM "{self._table}" WHERE rule_id = ?',
                (str(rule_id),),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def list(self, *, limit: int = 100, offset: int = 0) -> list[CleaningRuleRecord]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive int")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a non-negative int")
        sql = (
            f'SELECT * FROM "{self._table}" '
            "ORDER BY order_index ASC, created_at ASC LIMIT ? OFFSET ?"
        )
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(sql, (limit, offset))
            rows = cur.fetchall()
            return [self._row_to_record(r) for r in rows]

    def update(self, record: CleaningRuleRecord) -> CleaningRuleRecord:
        rid = record.rule_id
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT order_index FROM "{self._table}" WHERE rule_id = ?',
                (str(rid),),
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"cleaning rule not found: {rid}")
            order_index = int(row["order_index"])
            stored = record.model_copy(deep=True)
            cur.execute(
                f"""
                UPDATE "{self._table}"
                SET record_json = ?, created_at = ?
                WHERE rule_id = ?
                """,
                (
                    stored.model_dump_json(),
                    stored.created_at.isoformat(),
                    str(rid),
                ),
            )
            self._conn.commit()
            # keep order_index stable on update (no reorder)
            _ = order_index
        return stored.model_copy(deep=True)

    def delete(self, rule_id: str) -> bool:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT 1 FROM "{self._table}" WHERE rule_id = ?',
                (str(rule_id),),
            )
            existed = cur.fetchone() is not None
            cur.execute(
                f'DELETE FROM "{self._table}" WHERE rule_id = ?',
                (str(rule_id),),
            )
            self._conn.commit()
            return existed

    def count(self) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(f'SELECT COUNT(*) AS c FROM "{self._table}"')
            return int(cur.fetchone()["c"])

    def __contains__(self, rule_id: object) -> bool:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT 1 FROM "{self._table}" WHERE rule_id = ?',
                (str(rule_id),),
            )
            return cur.fetchone() is not None

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"SqliteCleaningRuleStore(table={self._table!r}, path={self._path!r})"


__all__ = ["SqliteCleaningRuleStore"]
