"""SQLite-backed :class:`~sparksage.feedback.store.FeedbackStore`.

Durable, single-file persistence for :class:`~sparksage.feedback.models.FeedbackRecord`
over a stdlib :mod:`sqlite3` connection -- no extra install, no server. The
feedback counterpart of
:class:`~sparksage.documents.backends.sqlite.SqliteDocumentStore`: where that
stores documents and
:class:`~sparksage.kb.backends.sqlite.SqliteKnowledgeBaseStore` stores KB
metadata, this stores the user-signal records that feed the quality flywheel --
so approval ratios / per-block breakdowns / healing signals survive a restart
instead of resetting to zero.

The full record is serialized to JSON (the source of truth); explicit columns
exist only for the fields used in filtering / listing
(``feedback_id`` PK, ``rating``, ``kb_id``, ``created_at``). The table name is
regex-validated (it cannot be SQL-parameterized), mirroring the other SQLite
backends. The connection is opened with ``check_same_thread=False`` and every
operation is serialized by a :class:`threading.RLock`, so the store is safe to
share across the FastAPI threadpool / worker threads.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from sparksage._sqlite import SqliteMixin
from sparksage.feedback.models import FeedbackRating, FeedbackRecord
from sparksage.feedback.store import FeedbackStats

_logger = logging.getLogger(__name__)

_IN_MEMORY_PATH = ":memory:"


class SqliteFeedbackStore(SqliteMixin):
    """Durable feedback store backed by a single SQLite database file.

    Parameters
    ----------
    path:
        Database file path (``str`` / :class:`pathlib.Path`). Defaults to
        ``":memory:"`` (an ephemeral in-process DB -- useful for tests, lost on
        close). Parent directories of a file path are created if missing.
    table:
        Name of the main table (default ``"feedback"``). Validated against
        ``^[A-Za-z_][A-Za-z0-9_]*$``.

    Examples
    --------
    >>> from sparksage.feedback import FeedbackRecord, FeedbackRating
    >>> from sparksage.feedback.backends import SqliteFeedbackStore
    >>> store = SqliteFeedbackStore()           # in-memory
    >>> rec = store.add(FeedbackRecord(query="how?", answer_text="...",
    ...                                rating=FeedbackRating.POSITIVE))
    >>> store.get(rec.feedback_id) is not None
    True
    """

    def __init__(
        self,
        path: str | Path = _IN_MEMORY_PATH,
        *,
        table: str = "feedback",
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
                    feedback_id  TEXT PRIMARY KEY,
                    rating       TEXT NOT NULL,
                    kb_id        TEXT,
                    created_at   TEXT NOT NULL,
                    record_json  TEXT NOT NULL
                )
                """
            )
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS "ix_{self._table}_kb_created" '
                f'ON "{self._table}" (kb_id, created_at DESC)'
            )
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_record(row: Any) -> FeedbackRecord:
        return FeedbackRecord.model_validate_json(row["record_json"])

    # ------------------------------------------------------------------ #
    # FeedbackStore protocol
    # ------------------------------------------------------------------ #
    def add(self, record: FeedbackRecord) -> FeedbackRecord:
        stored = record.model_copy(deep=True)
        record_json = stored.model_dump_json()
        created_iso = stored.created_at.isoformat()
        kb_val = str(stored.kb_id) if stored.kb_id is not None else None
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                INSERT OR REPLACE INTO "{self._table}"
                    (feedback_id, rating, kb_id, created_at, record_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(stored.feedback_id),
                    stored.rating.value,
                    kb_val,
                    created_iso,
                    record_json,
                ),
            )
            self._conn.commit()
        return stored.model_copy(deep=True)

    def get(self, feedback_id: str) -> FeedbackRecord | None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT * FROM "{self._table}" WHERE feedback_id = ?',
                (str(feedback_id),),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def list(
        self,
        *,
        rating: FeedbackRating | None = None,
        kb_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FeedbackRecord]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive int")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a non-negative int")

        where: list[str] = []
        params: list[Any] = []
        if rating is not None:
            where.append("rating = ?")
            params.append(rating.value)
        if kb_id is not None:
            where.append("kb_id = ?")
            params.append(str(kb_id))
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        sql = (
            f'SELECT * FROM "{self._table}" {where_sql} '
            "ORDER BY created_at DESC, feedback_id DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [self._row_to_record(r) for r in rows]

    def delete(self, feedback_id: str) -> bool:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT 1 FROM "{self._table}" WHERE feedback_id = ?',
                (str(feedback_id),),
            )
            existed = cur.fetchone() is not None
            cur.execute(
                f'DELETE FROM "{self._table}" WHERE feedback_id = ?',
                (str(feedback_id),),
            )
            self._conn.commit()
            return existed

    def count(self, *, kb_id: str | None = None) -> int:
        if kb_id is None:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(f'SELECT COUNT(*) AS c FROM "{self._table}"')
                return int(cur.fetchone()["c"])
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT COUNT(*) AS c FROM "{self._table}" WHERE kb_id = ?',
                (str(kb_id),),
            )
            return int(cur.fetchone()["c"])

    def stats(self, *, kb_id: str | None = None) -> FeedbackStats:
        """Aggregate :class:`FeedbackStats` over (a slice of) the store."""
        records = self.list(kb_id=kb_id, limit=10**9) if kb_id is not None else None
        if records is None:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    f'SELECT rating FROM "{self._table}"'
                )
                ratings = [r["rating"] for r in cur.fetchall()]
        else:
            ratings = [r.rating.value for r in records]
        counts = Counter(ratings)
        return FeedbackStats(
            total=len(ratings),
            positive=counts.get(FeedbackRating.POSITIVE.value, 0),
            negative=counts.get(FeedbackRating.NEGATIVE.value, 0),
            corrected=counts.get(FeedbackRating.CORRECTED.value, 0),
        )

    def block_breakdown(self) -> dict[str, FeedbackStats]:
        """Per-block :class:`FeedbackStats` (which blocks attract bad feedback)."""
        per_block: dict[str, list[FeedbackRating]] = {}
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(f'SELECT record_json FROM "{self._table}"')
            rows = cur.fetchall()
        for r in rows:
            rec = self._row_to_record(r)
            for bid in rec.block_ids:
                per_block.setdefault(bid, []).append(rec.rating)
        out: dict[str, FeedbackStats] = {}
        for bid, ratings in per_block.items():
            counts = Counter(ratings)
            out[bid] = FeedbackStats(
                total=len(ratings),
                positive=counts.get(FeedbackRating.POSITIVE, 0),
                negative=counts.get(FeedbackRating.NEGATIVE, 0),
                corrected=counts.get(FeedbackRating.CORRECTED, 0),
            )
        return out

    def __contains__(self, feedback_id: object) -> bool:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'SELECT 1 FROM "{self._table}" WHERE feedback_id = ?',
                (str(feedback_id),),
            )
            return cur.fetchone() is not None

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"SqliteFeedbackStore(table={self._table!r}, path={self._path!r})"


__all__ = ["SqliteFeedbackStore"]
