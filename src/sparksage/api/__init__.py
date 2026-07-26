"""SparkSage WEB API: expose convert / generate / documents as HTTP endpoints.

The API layer is a thin shell over the framework-agnostic
:class:`SparkSageService`. FastAPI is an *optional* dependency, imported lazily
inside :func:`create_app` / :func:`run` -- install it with
``pip install 'sparksage[api]'``.

Routes:

* ``POST /api/v1/convert`` -- uploaded file -> Markdown (optional cleaning).
* ``POST /api/v1/generate`` -- uploaded file -> list of IdeaBlocks.
* ``POST/GET /api/v1/documents``, ``GET/PATCH/DELETE /api/v1/documents/{id}``,
  ``POST /api/v1/documents/{id}/tags`` -- document management (upload + parse +
  auto-tag + summary + store, with CRUD + retag).
* ``GET /api/v1/tags`` -- distinct tag vocabulary across stored documents.
"""

from sparksage.api.pipeline import (
    ConvertOutput,
    GenerateOutput,
    GenerationNotConfiguredError,
    ServiceError,
    SparkSageService,
)
from sparksage.api.schemas import (
    ConvertResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentSourceInfo,
    DocumentSummary,
    DocumentUpdateRequest,
    GenerateResponse,
    GenerationStatsOut,
    HealthResponse,
    RetagRequest,
    SourceInfo,
    TagsResponse,
)

__all__ = [
    "ConvertOutput",
    "ConvertResponse",
    "DocumentListResponse",
    "DocumentResponse",
    "DocumentSourceInfo",
    "DocumentSummary",
    "DocumentUpdateRequest",
    "GenerateOutput",
    "GenerateResponse",
    "GenerationNotConfiguredError",
    "GenerationStatsOut",
    "HealthResponse",
    "RetagRequest",
    "ServiceError",
    "SourceInfo",
    "SparkSageService",
    "TagsResponse",
]
