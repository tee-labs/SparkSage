"""KnowledgeBase storage abstraction (:class:`KnowledgeBaseStore` protocol).

The multi-tenant counterpart of
:class:`~sparksage.documents.store.DocumentStore`: where that stores
*documents*, this stores *knowledge bases* -- each identified by its
``kb_id``. The default :class:`InMemoryKnowledgeBaseStore` is a plain dict;
the durable :class:`~sparksage.kb.backends.sqlite.SqliteKnowledgeBaseStore`
implements the same protocol over a single SQLite file so KB metadata
survives a process restart.

The store persists only the :class:`~sparksage.kb.models.KnowledgeBaseInfo`
metadata -- the live vector index + block registry are runtime state owned by
the :class:`~sparksage.kb.KnowledgeBase` aggregate, persisted separately via
a :class:`~sparksage.kb.backends.state.KbStateStore` (blocks + vectors +
doc-links) and rebuilt on load (exactly how the document store persists
records while the service holds runtime state).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sparksage.kb.models import KnowledgeBaseInfo


@runtime_checkable
class KnowledgeBaseStore(Protocol):
    """CRUD over :class:`KnowledgeBaseInfo` records (multi-tenant registry)."""

    def save(self, info: KnowledgeBaseInfo) -> KnowledgeBaseInfo:
        """Insert or replace ``info`` keyed by ``info.kb_id``."""
        ...

    def get(self, kb_id: str) -> KnowledgeBaseInfo | None:
        """Return the info for ``kb_id`` (a copy), or ``None`` if absent."""
        ...

    def list(self, *, limit: int = 100, offset: int = 0) -> list[KnowledgeBaseInfo]:
        """Return a slice of knowledge bases, newest-first by ``created_at``."""
        ...

    def delete(self, kb_id: str) -> bool:
        """Remove ``kb_id``. Return whether a record was actually deleted."""
        ...

    def __contains__(self, kb_id: object) -> bool:
        ...

    def __len__(self) -> int:
        ...


class InMemoryKnowledgeBaseStore:
    """Dict-backed :class:`KnowledgeBaseStore` (tests / single-node demos).

    Defensive copies are taken on save / get so caller mutation cannot corrupt
    the store, mirroring :class:`~sparksage.documents.backends.memory.InMemoryDocumentStore`.
    """

    def __init__(self) -> None:
        self._kbs: dict[str, KnowledgeBaseInfo] = {}

    def save(self, info: KnowledgeBaseInfo) -> KnowledgeBaseInfo:
        stored = info.model_copy()
        self._kbs[stored.kb_id] = stored
        return stored.model_copy()

    def get(self, kb_id: str) -> KnowledgeBaseInfo | None:
        info = self._kbs.get(str(kb_id))
        return info.model_copy() if info is not None else None

    def list(self, *, limit: int = 100, offset: int = 0) -> list[KnowledgeBaseInfo]:
        if limit < 1:
            raise ValueError("limit must be a positive int")
        if offset < 0:
            raise ValueError("offset must be a non-negative int")
        ordered = sorted(
            self._kbs.values(),
            key=lambda k: k.created_at,
            reverse=True,
        )
        return [k.model_copy() for k in ordered[offset : offset + limit]]

    def delete(self, kb_id: str) -> bool:
        return self._kbs.pop(str(kb_id), None) is not None

    def __contains__(self, kb_id: object) -> bool:
        return str(kb_id) in self._kbs

    def __len__(self) -> int:
        return len(self._kbs)


__all__ = [
    "InMemoryKnowledgeBaseStore",
    "KnowledgeBaseStore",
]
