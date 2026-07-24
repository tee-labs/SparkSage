"""FastAPI application factory for the SparkSage WEB API.

The web layer is intentionally thin: it only does HTTP-shaped I/O (file upload,
form parsing, JSON serialization) and delegates every piece of real work to the
framework-agnostic :class:`SparkSageService`. FastAPI (and its multipart
dependency) is an *optional* dependency, imported lazily inside
:func:`create_app` / :func:`run`, so the rest of the library keeps working
without it.

Two routes cover the requested capabilities:

* ``POST /api/v1/convert`` -- uploaded file -> Markdown (optional cleaning).
* ``POST /api/v1/generate`` -- uploaded file -> list of IdeaBlocks.

Install with::

    pip install 'sparksage[api]'          # fastapi + uvicorn + python-multipart

Run with::

    uvicorn sparksage.api.app:create_app --factory
    # or
    python3 -m sparksage.api.app

Note: this module deliberately omits ``from __future__ import annotations``.
FastAPI resolves route parameter annotations via ``typing.get_type_hints``, which
looks at the *module* globals; since ``UploadFile`` / ``File`` / ``Form`` are
imported lazily inside :func:`create_app` (optional dependency), eager annotation
evaluation at function-definition time -- when those names are in the enclosing
scope -- is what lets FastAPI see them.
"""

import logging
import os
from typing import Annotated, Any

from sparksage.api.documents import register_document_routes
from sparksage.api.pipeline import (
    GenerationNotConfiguredError,
    SparkSageService,
)
from sparksage.clean.cleaner import TextCleaner
from sparksage.config import load_dotenv
from sparksage.convert.backend import MarkItDownBackend
from sparksage.convert.converter import MarkdownConverter
from sparksage.documents.keyword_extract import FrequencyKeywordExtractor
from sparksage.documents.service import DocumentService
from sparksage.documents.store import InMemoryDocumentStore
from sparksage.generator.client import OpenAICompatibleClient
from sparksage.generator.generator import GenerationError, IdeaBlockGenerator

_logger = logging.getLogger(__name__)

#: Environment variable names used by :func:`build_default_service`.
ENV_API_KEY = "SPARKSAGE_API_KEY"
ENV_BASE_URL = "SPARKSAGE_BASE_URL"
ENV_MODEL = "SPARKSAGE_MODEL"
ENV_STREAM = "SPARKSAGE_STREAM"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"

DEFAULT_MODEL = "gpt-4o-mini"
#: Streaming is on by default -- it is more robust for long generations.
DEFAULT_STREAM = True

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _env(name: str) -> str | None:
    val = os.environ.get(name)
    return val if val else None


def _env_bool(name: str, default: bool) -> bool:
    """Parse an environment variable as a boolean.

    Accepts the common truthy/falsy spellings (``1/0``, ``true/false``,
    ``yes/no``, ``on/off``); case-insensitive. Anything else (or unset) falls
    back to ``default``.
    """
    raw = _env(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    return default


def build_default_service() -> SparkSageService:
    """Wire a production :class:`SparkSageService` from configuration.

    Configuration is read from environment variables. Values may be supplied
    directly (container / CI / system env) **or** via a ``.env`` file in the
    current working directory -- :func:`load_dotenv` is called first, but real
    environment variables always take priority over the file (12-factor). See
    :mod:`sparksage.config` for the supported ``.env`` syntax and a template at
    ``.env.example`` in the repo root.

    * Converter: a :class:`MarkdownConverter` over :class:`MarkItDownBackend`
      (requires ``pip install 'sparksage[convert]'``).
    * Cleaner: a default :class:`TextCleaner`.
    * Generator: an :class:`IdeaBlockGenerator` over
      :class:`OpenAICompatibleClient` when an API key is present; ``None``
      otherwise (the ``/generate`` route returns ``503`` in that case).

    Recognized env vars (``SPARKSAGE_*`` take priority over ``OPENAI_*``):

    ============================  =========================================
    ``SPARKSAGE_API_KEY``         API key (falls back to ``OPENAI_API_KEY``)
    ``SPARKSAGE_BASE_URL``        Base URL (falls back to ``OPENAI_BASE_URL``)
    ``SPARKSAGE_MODEL``           Model id (default ``gpt-4o-mini``)
    ``SPARKSAGE_STREAM``          Stream the LLM response (default ``true``)
    ``SPARKSAGE_LANGUAGE``        Output language written into each block
    ============================  =========================================
    """
    load_dotenv()
    converter = MarkdownConverter(backend=MarkItDownBackend())
    cleaner = TextCleaner()

    generator: IdeaBlockGenerator | None = None
    api_key = _env(ENV_API_KEY) or _env(ENV_OPENAI_API_KEY)
    if api_key:
        base_url = _env(ENV_BASE_URL) or _env(ENV_OPENAI_BASE_URL)
        model = _env(ENV_MODEL) or DEFAULT_MODEL
        language = _env("SPARKSAGE_LANGUAGE") or "en"
        stream = _env_bool(ENV_STREAM, DEFAULT_STREAM)
        client = OpenAICompatibleClient(
            base_url=base_url, api_key=api_key, model=model, stream=stream
        )
        generator = IdeaBlockGenerator(client, language=language)
        _logger.info("generator configured with model=%s stream=%s", model, stream)
    else:
        _logger.warning(
            "no %s/%s set; the /generate route will return 503",
            ENV_API_KEY,
            ENV_OPENAI_API_KEY,
        )

    return SparkSageService(
        converter=converter, cleaner=cleaner, generator=generator
    )


def build_default_document_service(
    converter: MarkdownConverter | None = None,
    cleaner: TextCleaner | None = None,
) -> DocumentService:
    """Wire a production :class:`DocumentService` from configuration.

    Reuses the same conversion + cleaning stack as
    :func:`build_default_service`. Conversion is required (documents are parsed
    from Markdown); the keyword extractor and in-memory store need no API key,
    so document management works **LLM-free** even when ``/generate`` returns
    ``503``. Pass a shared ``converter`` / ``cleaner`` to avoid constructing a
    second ``markitdown`` instance.
    """
    converter = converter or MarkdownConverter(backend=MarkItDownBackend())
    cleaner = cleaner or TextCleaner()
    return DocumentService(
        converter=converter,
        cleaner=cleaner,
        extractor=FrequencyKeywordExtractor(),
        store=InMemoryDocumentStore(),
    )


def create_app(
    service: SparkSageService | None = None,
    document_service: DocumentService | None = None,
) -> Any:
    """Create and configure a FastAPI application.

    Parameters
    ----------
    service:
        A pre-built :class:`SparkSageService` for the convert/generate routes.
        When omitted, :func:`build_default_service` is used. Inject a custom
        service (e.g. with fakes) for testing.
    document_service:
        A pre-built :class:`DocumentService` for the knowledge-document CRUD +
        tag-management routes. When omitted,
        :func:`build_default_document_service` is used. The document routes work
        without any LLM API key.

    Raises
    ------
    ImportError
        If FastAPI / python-multipart are not installed.
    """
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "The SparkSage API requires 'fastapi' and 'python-multipart'. "
            "Install them with: pip install 'sparksage[api]'"
        ) from exc

    from sparksage import __version__
    from sparksage.api.schemas import (
        ConvertResponse,
        GenerateResponse,
        HealthResponse,
        to_convert_response,
        to_generate_response,
    )

    svc = service if service is not None else build_default_service()

    doc_svc = (
        document_service
        if document_service is not None
        else build_default_document_service()
    )

    app = FastAPI(
        title="SparkSage API",
        description=(
            "Turn any uploaded file into Markdown (optionally cleaned) or into a "
            "list of question-aligned IdeaBlocks, and manage a knowledge base of "
            "tagged documents over REST."
        ),
        version=__version__,
    )

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            generator_configured=svc.has_generator,
        )

    @app.post(
        "/api/v1/convert",
        response_model=ConvertResponse,
        summary="Convert an uploaded file to Markdown",
    )
    async def convert(
        file: Annotated[UploadFile, File(description="The source document to convert.")],
        clean: Annotated[
            bool, Form(description="Apply text cleaning before returning.")
        ] = False,
    ) -> ConvertResponse:
        data = await file.read()
        try:
            out = svc.convert(data, file.filename, clean=clean)
        except Exception as exc:  # noqa: BLE001 - surface as HTTP error
            raise HTTPException(status_code=422, detail=_detail(exc)) from exc
        return to_convert_response(out)

    @app.post(
        "/api/v1/generate",
        response_model=GenerateResponse,
        summary="Convert an uploaded file to a list of IdeaBlocks",
    )
    async def generate(
        file: Annotated[UploadFile, File(description="The source document to chunk.")],
        clean: Annotated[
            bool, Form(description="Apply text cleaning before generating.")
        ] = True,
        max_blocks: Annotated[
            int | None, Form(ge=1, description="Max number of blocks to emit.")
        ] = None,
        language: Annotated[
            str | None, Form(description="BCP-47 code written into every block.")
        ] = None,
        with_stats: Annotated[
            bool, Form(description="Include generation diagnostics.")
        ] = False,
    ) -> GenerateResponse:
        data = await file.read()
        try:
            out = svc.generate(
                data,
                file.filename,
                clean=clean,
                max_blocks=max_blocks,
                language=language,
                with_stats=with_stats,
            )
        except GenerationNotConfiguredError as exc:
            raise HTTPException(status_code=503, detail=_detail(exc)) from exc
        except GenerationError as exc:
            raise HTTPException(status_code=502, detail=_detail(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surface as HTTP error
            raise HTTPException(status_code=422, detail=_detail(exc)) from exc
        return to_generate_response(out)

    register_document_routes(app, doc_svc)

    app.state.service = svc
    app.state.document_service = doc_svc
    return app


def _detail(exc: BaseException) -> str:
    msg = str(exc)
    return msg or exc.__class__.__name__


def run(  # pragma: no cover - thin launcher
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Convenience launcher: ``python -m sparksage.api.app``."""
    import uvicorn

    uvicorn.run(
        "sparksage.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_STREAM",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_MODEL",
    "build_default_document_service",
    "ENV_STREAM",
    "build_default_service",
    "create_app",
    "run",
]


if __name__ == "__main__":  # pragma: no cover
    run()
