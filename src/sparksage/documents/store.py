"""Document storage abstraction (:class:`DocumentStore` protocol).

The document-management layer depends *only* on this protocol, mirroring how the
rest of SparkSage decouples its cores from concrete backends
(:class:`~sparksage.embed.store.VectorStore`,
:class:`~sparksage.convert.backend.ConverterBackend`, ...). Swap an
:class:`~sparksage.documents.backends.memory.InMemoryDocumentStore` (tests /
single-node demos) for a :class:`~sparksage.documents.backends.sqlite.SqliteDocumentStore`
(durable single-file persistence) without touching the service.

A :class:`DocumentStore` is the CRUD + tag-filtering contract a document-
management service (and the ``/api/v1/documents`` route) consumes:

* :meth:`save` -- upsert a :class:`~sparksage.documents.models.DocumentRecord`
  by ``doc_id`` (insert or replace); returns the stored record.
* :meth:`get` / :meth:`delete` / :meth:`count` -- single-record read / remove /
  count.
* :meth:`list` -- paginated listing, optionally filtered by a tag and/or a free
  text query over title + body.
* :meth:`list_tags` -- the distinct tag vocabulary, for faceted browse.
* :meth:`__contains__` / :meth:`__len__` -- membership / size introspection.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sparksage.documents.models import DocumentRecord


@runtime_checkable
class DocumentStore(Protocol):
    """CRUD + tag/text querying over :class:`DocumentRecord` instances."""

    def save(self, record: DocumentRecord) -> DocumentRecord:
        """Insert or replace ``record`` keyed by ``record.doc_id``.

        Returns the stored record (a defensive copy, so later caller mutation
        cannot corrupt the store).
        """
        ...

    def get(self, doc_id: str) -> DocumentRecord | None:
        """Return the record for ``doc_id`` (a copy), or ``None`` if absent."""
        ...

    def list(
        self,
        *,
        tag: str | None = None,
        tags: list[str] | None = None,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentRecord]:
        """Return a slice of records, newest-first by ``created_at``.

        Parameters
        ----------
        tag:
            When set, only records carrying this exact tag are returned.
        tags:
            When set, only records carrying at least one of these tags are
            returned (any-match OR). Combined with ``tag`` when both are given.
        q:
            When set, only records whose ``title`` or ``body_markdown`` contain
            the substring (case-insensitive) are returned.
        limit, offset:
            Pagination. ``limit`` must be ``>= 1``; ``offset`` ``>= 0``.
        """
        ...

    def delete(self, doc_id: str) -> bool:
        """Remove ``doc_id``. Return whether a record was actually deleted."""
        ...

    def count(self, *, tag: str | None = None, tags: list[str] | None = None) -> int:
        """Number of records, optionally restricted to those carrying ``tag``.

        ``tags`` (any-match OR) is combined with ``tag`` when both are given.
        """
        ...

    def list_tags(self) -> list[str]:
        """Return the distinct tags across all records, sorted ascending."""
        ...

    def __contains__(self, doc_id: object) -> bool:
        """Whether ``doc_id`` is stored."""
        ...

    def __len__(self) -> int:
        """Number of records currently stored."""
        ...


def _validate_pagination(limit: int, offset: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive int")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative int")


def _normalize_tags(
    tag: str | None = None, tags: list[str] | None = None
) -> set[str]:
    """Collect the required-tag set from ``tag`` and/or ``tags``.

    Empty / blank entries are dropped. A record matches when it carries *any*
    of the returned tags (any-match OR). Returns an empty set for "no filter".
    """
    required: set[str] = set()
    if tag:
        norm = tag.strip()
        if norm:
            required.add(norm)
    if tags:
        for raw in tags:
            norm = raw.strip() if isinstance(raw, str) else None
            if norm:
                required.add(norm)
    return required


__all__ = [
    "DocumentStore",
]
