"""Document storage abstraction for the knowledge-management service.

The service layer is fully decoupled from *where* documents live: it depends
only on the :class:`DocumentStore` :class:`~typing.Protocol`. A concrete
:class:`InMemoryDocumentStore` ships for demos, tests and single-process
deployments; production callers implement the Protocol against a database /
search engine (Postgres, SQLite, Elasticsearch, ...) without touching the
service or HTTP layers.

Following the project convention, the in-memory store is pure stdlib, returns
defensive copies (so callers cannot mutate internal state), and is the single
owner of identity assignment / timestamp bumping.
"""

from __future__ import annotations

import threading
import uuid
from typing import Protocol, runtime_checkable

from sparksage.documents.schema import Document


@runtime_checkable
class DocumentStore(Protocol):
    """Persistence interface for :class:`Document` records.

    Implementations own identity assignment (a fresh ``id`` on first save) and
    version/timestamp bumps. All read methods must return copies so callers
    cannot mutate stored state inadvertently.
    """

    def save(self, document: Document) -> Document:
        """Insert or update ``document``. Returns the persisted (copied) record."""
        ...

    def get(self, document_id: uuid.UUID) -> Document | None:
        """Return the document with ``document_id``, or ``None`` if absent."""
        ...

    def list(
        self,
        *,
        tag: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Document]:
        """Return a page of documents, optionally filtered by ``tag``."""
        ...

    def delete(self, document_id: uuid.UUID) -> bool:
        """Delete ``document_id``. Returns ``True`` if a record was removed."""
        ...

    def count(self, *, tag: str | None = None) -> int:
        """Number of stored documents, optionally restricted to ``tag``."""
        ...

    def tags(self) -> dict[str, int]:
        """Distinct tags mapped to how many documents carry them."""
        ...


def _matches_tag(document: Document, tag: str) -> bool:
    lowered = tag.lower()
    return any(t.lower() == lowered for t in document.tags)


class InMemoryDocumentStore:
    """Thread-safe, dict-backed :class:`DocumentStore`.

    Suitable for tests, examples and single-process services. All public read
    methods return deep copies via :meth:`Document.model_dump`, so mutating a
    returned record never affects stored state.
    """

    def __init__(self) -> None:
        self._docs: dict[uuid.UUID, Document] = {}
        self._lock = threading.Lock()

    def save(self, document: Document) -> Document:
        with self._lock:
            stored = document.model_copy(deep=True)
            self._docs[stored.id] = stored
            return stored.model_copy(deep=True)

    def get(self, document_id: uuid.UUID) -> Document | None:
        with self._lock:
            doc = self._docs.get(document_id)
            return doc.model_copy(deep=True) if doc is not None else None

    def list(
        self,
        *,
        tag: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Document]:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        with self._lock:
            docs = list(self._docs.values())
        if tag is not None:
            docs = [d for d in docs if _matches_tag(d, tag)]
        docs.sort(key=lambda d: d.created_at, reverse=True)
        page = docs[offset : offset + limit] if limit else docs[offset:]
        return [d.model_copy(deep=True) for d in page]

    def delete(self, document_id: uuid.UUID) -> bool:
        with self._lock:
            return self._docs.pop(document_id, None) is not None

    def count(self, *, tag: str | None = None) -> int:
        with self._lock:
            docs = list(self._docs.values())
        if tag is not None:
            docs = [d for d in docs if _matches_tag(d, tag)]
        return len(docs)

    def tags(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._lock:
            docs = list(self._docs.values())
        for doc in docs:
            for tag in doc.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
