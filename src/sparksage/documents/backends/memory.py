"""In-memory :class:`~sparksage.documents.store.DocumentStore`.

A pure-stdlib ``dict[doc_id, DocumentRecord]`` with full CRUD, tag filtering,
free-text substring search over title + body, and the distinct-tag vocabulary.
Mirrors :class:`~sparksage.embed.store.InMemoryVectorStore`: defensive copies on
save/get so a caller mutating a returned record cannot corrupt the store.

This is the dependency-free default -- fine for tests, single-node demos and
ephemeral API runs. For persistence across restarts, swap in a
:class:`~sparksage.documents.backends.sqlite.SqliteDocumentStore` (or a future
Postgres backend) without touching the service.
"""

from __future__ import annotations

from sparksage.documents.models import DocumentRecord
from sparksage.documents.store import _validate_pagination


class InMemoryDocumentStore:
    """Dict-backed document store with CRUD + tag / text filtering.

    Examples
    --------
    >>> from sparksage.documents import DocumentRecord, InMemoryDocumentStore
    >>> store = InMemoryDocumentStore()
    >>> rec = DocumentRecord(body_markdown="# Hi", source={"uri": "x"})
    >>> _ = store.save(rec)
    >>> rec.doc_id in store
    True
    >>> store.get(rec.doc_id).body_markdown
    '# Hi'
    """

    def __init__(self) -> None:
        self._records: dict[str, DocumentRecord] = {}

    def save(self, record: DocumentRecord) -> DocumentRecord:
        stored = record.model_copy(deep=True)
        self._records[str(record.doc_id)] = stored
        return stored.model_copy(deep=True)

    def get(self, doc_id: str) -> DocumentRecord | None:
        rec = self._records.get(str(doc_id))
        return rec.model_copy(deep=True) if rec is not None else None

    def list(
        self,
        *,
        tag: str | None = None,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentRecord]:
        _validate_pagination(limit, offset)
        tag_norm = tag.strip() if tag else None
        query_norm = q.strip().lower() if q else None
        records = sorted(
            self._records.values(),
            key=lambda r: (r.created_at, r.doc_id),
            reverse=True,
        )
        out: list[DocumentRecord] = []
        for rec in records:
            if tag_norm is not None and tag_norm not in rec.tags:
                continue
            if query_norm is not None:
                hay = (
                    (rec.title or "").lower() + "\n" + rec.body_markdown.lower()
                )
                if query_norm not in hay:
                    continue
            out.append(rec.model_copy(deep=True))
        return out[offset : offset + limit]

    def delete(self, doc_id: str) -> bool:
        return self._records.pop(str(doc_id), None) is not None

    def count(self, *, tag: str | None = None) -> int:
        if tag is None:
            return len(self._records)
        tag_norm = tag.strip()
        return sum(1 for r in self._records.values() if tag_norm in r.tags)

    def list_tags(self) -> list[str]:
        seen: set[str] = set()
        for rec in self._records.values():
            seen.update(rec.tags)
        return sorted(seen)

    def __contains__(self, doc_id: object) -> bool:
        return str(doc_id) in self._records

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"InMemoryDocumentStore(count={len(self._records)})"


__all__ = ["InMemoryDocumentStore"]
