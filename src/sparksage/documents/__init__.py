"""SparkSage documents: enterprise knowledge-document tag management.

This package adds the *document*-granularity capabilities a knowledge-management
platform needs on top of SparkSage's chunk-oriented core:

* :class:`Document` -- a stored record with title, summary, body and tags.
* :mod:`markdown_parser` -- deterministic title / summary extraction.
* :mod:`keyword_extract` -- a pure-stdlib keyword-extraction algorithm
  (:class:`FrequencyKeywordExtractor`) behind a :class:`KeywordExtractor`
  protocol, used to auto-tag documents that ship without tags.
* :class:`DocumentStore` / :class:`InMemoryDocumentStore` -- pluggable
  persistence.
* :class:`DocumentService` -- framework-agnostic convert -> clean -> parse ->
  tag -> store orchestration.

The HTTP layer (:mod:`sparksage.api`) exposes these as RESTful CRUD routes.
"""

from sparksage.documents.keyword_extract import (
    DEFAULT_STOP_WORDS,
    FrequencyKeywordExtractor,
    KeywordExtractor,
)
from sparksage.documents.markdown_parser import (
    ParsedDocument,
    extract_summary,
    extract_title,
    parse_markdown,
)
from sparksage.documents.schema import Document
from sparksage.documents.service import (
    DEFAULT_AUTO_TAG_COUNT,
    DocumentNotFoundError,
    DocumentService,
    DocumentServiceError,
)
from sparksage.documents.store import DocumentStore, InMemoryDocumentStore

__all__ = [
    "DEFAULT_AUTO_TAG_COUNT",
    "DEFAULT_STOP_WORDS",
    "Document",
    "DocumentNotFoundError",
    "DocumentService",
    "DocumentServiceError",
    "DocumentStore",
    "FrequencyKeywordExtractor",
    "InMemoryDocumentStore",
    "KeywordExtractor",
    "ParsedDocument",
    "extract_summary",
    "extract_title",
    "parse_markdown",
]
