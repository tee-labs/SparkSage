"""SparkSage WEB API: expose convert / generate / documents / QA as HTTP.

The API layer is a thin shell over the framework-agnostic
:class:`SparkSageService` (ingest) and :class:`QAService` (end-to-end QA).
FastAPI is an *optional* dependency, imported lazily inside
:func:`create_app` / :func:`run` -- install it with
``pip install 'sparksage[api]'``.

Routes:

* ``POST /api/v1/convert`` -- uploaded file -> Markdown (optional cleaning).
* ``POST /api/v1/generate`` -- uploaded file -> list of IdeaBlocks.
* ``POST/GET /api/v1/documents``, ``GET/PATCH/DELETE /api/v1/documents/{id}``,
  ``POST /api/v1/documents/{id}/tags`` -- document management (upload + parse +
  auto-tag + summary + store, with CRUD + retag).
* ``GET /api/v1/tags`` -- distinct tag vocabulary across stored documents.
* ``POST /api/v1/knowledge_base/ingest`` -- upload knowledge: parse -> chunk ->
  embed -> index (makes it retrievable).
* ``POST /api/v1/query`` -- ask a question against the knowledge base.
* ``GET /api/v1/knowledge_base`` -- knowledge-base snapshot (counts).
* ``POST/GET /api/v1/feedback`` -- record / aggregate user verdicts.
"""

from sparksage.api.pipeline import (
    ConvertOutput,
    GenerateOutput,
    GenerationNotConfiguredError,
    ServiceError,
    SparkSageService,
)
from sparksage.api.qa_service import (
    IngestResult,
    QAService,
)
from sparksage.api.schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
    ConvertResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentSourceInfo,
    DocumentSummary,
    DocumentUpdateRequest,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatsResponse,
    GenerateResponse,
    GenerationStatsOut,
    HealthResponse,
    IngestAndIndexResponse,
    KnowledgeBaseResponse,
    RetagRequest,
    RetrievedChunkOut,
    SourceInfo,
    TagsResponse,
)

__all__ = [
    "AskRequest",
    "AskResponse",
    "CitationOut",
    "ConvertOutput",
    "ConvertResponse",
    "DocumentListResponse",
    "DocumentResponse",
    "DocumentSourceInfo",
    "DocumentSummary",
    "DocumentUpdateRequest",
    "FeedbackRequest",
    "FeedbackResponse",
    "FeedbackStatsResponse",
    "GenerateOutput",
    "GenerateResponse",
    "GenerationNotConfiguredError",
    "GenerationStatsOut",
    "HealthResponse",
    "IngestAndIndexResponse",
    "IngestResult",
    "KnowledgeBaseResponse",
    "QAService",
    "RetagRequest",
    "RetrievedChunkOut",
    "ServiceError",
    "SourceInfo",
    "SparkSageService",
    "TagsResponse",
]
