"""SparkSage WEB API: expose convert / generate and document management as HTTP.

The API layer is a thin shell over the framework-agnostic services
(:class:`SparkSageService`, :class:`DocumentService`). FastAPI is an *optional*
dependency, imported lazily inside :func:`create_app` / :func:`run` -- install
it with ``pip install 'sparksage[api]'``.

Routes:

* ``POST /api/v1/convert``  -- uploaded file -> Markdown (optional cleaning).
* ``POST /api/v1/generate`` -- uploaded file -> list of IdeaBlocks.
* ``/api/v1/documents`` (+ ``/api/v1/tags``) -- knowledge-document CRUD and tag
  management (auto-tagging from content when no tags are supplied).
"""

from sparksage.api.pipeline import (
    ConvertOutput,
    GenerateOutput,
    GenerationNotConfiguredError,
    ServiceError,
    SparkSageService,
)
from sparksage.api.schemas import (
    AddTagsRequest,
    AutoTagRequest,
    ConvertResponse,
    DocumentCreateResponse,
    DocumentListResponse,
    DocumentOut,
    DocumentUpdateRequest,
    GenerateResponse,
    GenerationStatsOut,
    HealthResponse,
    SourceInfo,
    TagCount,
    TagsResponse,
)

__all__ = [
    "AddTagsRequest",
    "AutoTagRequest",
    "ConvertOutput",
    "ConvertResponse",
    "DocumentCreateResponse",
    "DocumentListResponse",
    "DocumentOut",
    "DocumentUpdateRequest",
    "GenerateOutput",
    "GenerateResponse",
    "GenerationNotConfiguredError",
    "GenerationStatsOut",
    "HealthResponse",
    "ServiceError",
    "SourceInfo",
    "SparkSageService",
    "TagCount",
    "TagsResponse",
]
