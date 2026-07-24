"""SparkSage WEB API: expose convert / generate as HTTP endpoints.

The API layer is a thin shell over the framework-agnostic
:class:`SparkSageService`. FastAPI is an *optional* dependency, imported lazily
inside :func:`create_app` / :func:`run` -- install it with
``pip install 'sparksage[api]'``.

Two routes:

* ``POST /api/v1/convert`` -- uploaded file -> Markdown (optional cleaning).
* ``POST /api/v1/generate`` -- uploaded file -> list of IdeaBlocks.
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
    GenerateResponse,
    GenerationStatsOut,
    HealthResponse,
    SourceInfo,
)

__all__ = [
    "ConvertOutput",
    "ConvertResponse",
    "GenerateOutput",
    "GenerateResponse",
    "GenerationNotConfiguredError",
    "GenerationStatsOut",
    "HealthResponse",
    "ServiceError",
    "SourceInfo",
    "SparkSageService",
]
