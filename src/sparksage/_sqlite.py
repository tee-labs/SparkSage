"""Shared prologue for the stdlib-sqlite persistence backends.

Four backends (:class:`~sparksage.documents.backends.sqlite.SqliteDocumentStore`,
:class:`~sparksage.kb.backends.sqlite.SqliteKnowledgeBaseStore`,
:class:`~sparksage.kb.backends.state.SqliteKbStateStore`,
:class:`~sparksage.feedback.backends.sqlite.SqliteFeedbackStore`) all open a
connection the same way: regex-validate the table name (it cannot be SQL-
parameterized), mkdir the parent of a file path, connect with
``check_same_thread=False`` + ``sqlite3.Row``, and serialize writes with a
:class:`threading.RLock`. This mixin is that shared prologue.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

#: A table name must be a plain SQL identifier -- it cannot be passed as a
#: parameter, so it is regex-validated before being interpolated into SQL.
_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_IN_MEMORY_PATH = ":memory:"


class SqliteMixin:
    """Open a validated SQLite connection with a write lock.

    Mixin users call :meth:`_open` (which sets ``self._table``, ``self._path``,
    ``self._conn``, ``self._lock``) and then build their schema in
    ``_init_schema`` (invoked by ``_open``). Subclasses may add extra table
    names before ``_init_schema`` runs -- the mixin only sets ``_table``.
    """

    _lock: threading.RLock
    _conn: Any

    def _open(self, path: str | Path, table: str) -> None:
        table_norm = str(table).strip()
        if not _TABLE_NAME_RE.match(table_norm):
            raise ValueError(
                f"invalid table name {table!r}: must match ^[A-Za-z_][A-Za-z0-9_]*$"
            )
        self._table = table_norm

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

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()
