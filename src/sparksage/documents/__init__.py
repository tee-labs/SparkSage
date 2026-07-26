"""Document management: parsed documents + free-form tags + storage.

This package is the document-level counterpart of the chunk-level
:mod:`sparksage.schema`. It fills the gap surfaced by the enterprise
tag-management use case: there was no *document* object to attach a title,
summary, free-form tags or lifecycle to -- ingest jumped straight from a
Markdown blob to a list of :class:`~sparksage.schema.IdeaBlock` chunks.

* :class:`DocumentRecord` -- the Pydantic v2 document entity (title / summary /
  body / free-form ``tags: list[str]`` / provenance / timestamps). Intentionally
  *not* the closed :class:`~sparksage.schema.enums.Tag` enum.
* :class:`DocumentStore` -- the storage Protocol (save / get / list / delete /
  count / list_tags), with a pure-stdlib :class:`InMemoryDocumentStore` and a
  durable :class:`SqliteDocumentStore` (single-file, no server).
* :class:`ExtractiveSummarizer` -- dependency-free document-level summary.
* :class:`make_extractor` / the :class:`~sparksage.tags.KeywordExtractor`
  algorithms live in :mod:`sparksage.tags` and are wired into the ingest flow by
  :class:`~sparksage.api.SparkSageService`.

The ``/api/v1/documents`` route is a thin HTTP shell over the framework-agnostic
:class:`~sparksage.api.SparkSageService`, which owns convert -> clean ->
auto-tag -> summarize -> store.
"""

from sparksage.documents.backends import (
    InMemoryDocumentStore,
    SqliteDocumentStore,
)
from sparksage.documents.models import (
    DocumentRecord,
    content_hash_of,
    new_record,
)
from sparksage.documents.store import DocumentStore
from sparksage.documents.summarizer import (
    ExtractiveSummarizer,
    Summarizer,
    default_summarizer,
    split_sentences,
)

__all__ = [
    "DocumentRecord",
    "DocumentStore",
    "ExtractiveSummarizer",
    "InMemoryDocumentStore",
    "SqliteDocumentStore",
    "Summarizer",
    "content_hash_of",
    "default_summarizer",
    "new_record",
    "split_sentences",
]
