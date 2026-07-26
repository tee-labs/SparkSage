"""FastAPI application factory for the SparkSage WEB API.

The web layer is intentionally thin: it only does HTTP-shaped I/O (file upload,
form parsing, JSON serialization) and delegates every piece of real work to the
framework-agnostic :class:`SparkSageService`. FastAPI (and its multipart
dependency) is an *optional* dependency, imported lazily inside
:func:`create_app` / :func:`run`, so the rest of the library keeps working
without it.

Routes cover conversion, generation and document management:

* ``POST /api/v1/convert`` -- uploaded file -> Markdown (optional cleaning).
* ``POST /api/v1/generate`` -- uploaded file -> list of IdeaBlocks.
* ``POST /api/v1/documents`` -- uploaded file -> parsed, auto-tagged, summarized,
  stored :class:`~sparksage.documents.DocumentRecord`.
* ``GET /api/v1/documents`` -- paginated listing (``?tag=`` / ``?q=`` filters).
* ``GET /PATCH /DELETE /api/v1/documents/{doc_id}`` -- single-document CRUD.
* ``POST /api/v1/documents/{doc_id}/tags`` -- re-extract tags from the body.
* ``GET /api/v1/tags`` -- distinct tag vocabulary across stored documents.

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

from sparksage.api.pipeline import (
    GenerationNotConfiguredError,
    SparkSageService,
)
from sparksage.clean.cleaner import TextCleaner
from sparksage.config import load_dotenv
from sparksage.convert.backend import MarkItDownBackend
from sparksage.convert.converter import MarkdownConverter
from sparksage.documents.backends import SqliteDocumentStore
from sparksage.documents.backends.memory import InMemoryDocumentStore
from sparksage.generator.client import OpenAICompatibleClient
from sparksage.generator.generator import GenerationError, IdeaBlockGenerator
from sparksage.logging_config import ENV_LOG_LEVEL, configure_logging
from sparksage.tags.extractor import make_extractor
from sparksage.tags.tokenizer import AutoTokenizer

_logger = logging.getLogger(__name__)

#: Environment variable names used by :func:`build_default_service`.
ENV_API_KEY = "SPARKSAGE_API_KEY"
ENV_BASE_URL = "SPARKSAGE_BASE_URL"
ENV_MODEL = "SPARKSAGE_MODEL"
ENV_STREAM = "SPARKSAGE_STREAM"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"

# Document management
ENV_DOC_STORE = "SPARKSAGE_DOC_STORE"
ENV_DOC_STORE_TABLE = "SPARKSAGE_DOC_STORE_TABLE"
ENV_AUTO_TAG_EXTRACTOR = "SPARKSAGE_AUTO_TAG_EXTRACTOR"
ENV_TAGS_ZH = "SPARKSAGE_TAGS_ZH"

DEFAULT_MODEL = "gpt-4o-mini"
#: Streaming is on by default -- it is more robust for long generations.
DEFAULT_STREAM = True
#: Default SQLite table name for the durable document store.
DEFAULT_DOC_STORE_TABLE = "documents"
#: Default keyword-extraction algorithm used for auto-tagging.
DEFAULT_AUTO_TAG_EXTRACTOR = "rake"

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
    ``SPARKSAGE_LOG_LEVEL``       ``sparksage`` logger verbosity (default ``WARNING``)
    ``SPARKSAGE_DOC_STORE``       Path to a SQLite file for the document store
                                   (empty -> in-memory; the ``/documents`` routes work
                                   but do not persist across restarts)
    ``SPARKSAGE_DOC_STORE_TABLE`` SQLite table name (default ``documents``)
    ``SPARKSAGE_AUTO_TAG_EXTRACTOR``  Auto-tag algorithm: rake / tfidf / textrank
                                      (default ``rake``)
    ``SPARKSAGE_TAGS_ZH``         Use ``jieba`` for CJK segmentation when ``true``
                                   (requires ``pip install 'sparksage[tags-zh]'``)
    ============================  =========================================
    """
    load_dotenv()
    configure_logging()
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

    extractor_name = _env(ENV_AUTO_TAG_EXTRACTOR) or DEFAULT_AUTO_TAG_EXTRACTOR
    use_jieba = _env_bool(ENV_TAGS_ZH, False)
    tokenizer = AutoTokenizer(use_jieba=use_jieba)
    keyword_extractor = make_extractor(extractor_name, tokenizer=tokenizer)

    doc_store = _build_document_store()

    return SparkSageService(
        converter=converter,
        cleaner=cleaner,
        generator=generator,
        document_store=doc_store,
        keyword_extractor=keyword_extractor,
    )


def _build_document_store():
    """Resolve a :class:`DocumentStore` from the ``SPARKSAGE_DOC_STORE`` env var.

    A real path wires a durable :class:`SqliteDocumentStore`; unset / empty /
    ``":memory:"`` returns ``None`` so the service falls back to its lazy
    in-memory store (ephemeral, but the routes work with zero configuration).
    """
    path = _env(ENV_DOC_STORE)
    if not path:
        return None
    if path == ":memory:":
        return InMemoryDocumentStore()
    table = _env(ENV_DOC_STORE_TABLE) or DEFAULT_DOC_STORE_TABLE
    return SqliteDocumentStore(path, table=table)


def create_app(service: SparkSageService | None = None) -> Any:
    """Create and configure a FastAPI application.

    Parameters
    ----------
    service:
        A pre-built :class:`SparkSageService`. When omitted,
        :func:`build_default_service` is used (which reads env vars). Inject a
        custom service (e.g. with fakes) for testing.

    Raises
    ------
    ImportError
        If FastAPI / python-multipart are not installed.
    """
    try:
        from fastapi import (
            Body,
            FastAPI,
            File,
            Form,
            HTTPException,
            Path,
            Query,
            UploadFile,
        )
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "The SparkSage API requires 'fastapi' and 'python-multipart'. "
            "Install them with: pip install 'sparksage[api]'"
        ) from exc

    from sparksage import __version__
    from sparksage.api.schemas import (
        ConvertResponse,
        DocumentListResponse,
        DocumentResponse,
        DocumentUpdateRequest,
        GenerateResponse,
        HealthResponse,
        RetagRequest,
        TagsResponse,
        to_convert_response,
        to_document_list_response,
        to_document_response,
        to_generate_response,
    )

    svc = service if service is not None else build_default_service()

    app = FastAPI(
        title="SparkSage API",
        description=(
            "Turn any uploaded file into Markdown (optionally cleaned) or into a "
            "list of question-aligned IdeaBlocks."
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

    # ------------------------------------------------------------------ #
    # document management routes
    # ------------------------------------------------------------------ #
    def _parse_tags(raw: str | None) -> list[str] | None:
        if raw is None:
            return None
        parts = [p.strip() for p in raw.split(",")]
        return [p for p in parts if p]

    @app.post(
        "/api/v1/documents",
        response_model=DocumentResponse,
        summary="Upload, parse, tag and store a document",
    )
    async def create_document(
        file: Annotated[
            UploadFile, File(description="The source document to ingest.")
        ],
        title: Annotated[
            str | None, Form(description="Explicit title override.")
        ] = None,
        tags: Annotated[
            str | None,
            Form(description="Comma-separated tags. When empty, tags are auto-extracted."),
        ] = None,
        auto_tag: Annotated[
            bool,
            Form(description="Auto-extract tags from the content when none are given."),
        ] = True,
        clean: Annotated[
            bool, Form(description="Apply text cleaning before tagging/summarizing.")
        ] = True,
        summarize: Annotated[
            bool, Form(description="Produce a document-level summary.")
        ] = True,
        max_summary_sentences: Annotated[
            int, Form(ge=1, description="Max sentences in the summary.")
        ] = 3,
        top_k: Annotated[
            int, Form(ge=1, description="Number of tags to extract when auto-tagging.")
        ] = 8,
    ) -> DocumentResponse:
        data = await file.read()
        try:
            record = svc.ingest_document(
                data,
                file.filename,
                title=title,
                tags=_parse_tags(tags),
                auto_tag=auto_tag,
                clean=clean,
                summarize=summarize,
                max_summary_sentences=max_summary_sentences,
                top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001 - surface as HTTP error
            raise HTTPException(status_code=422, detail=_detail(exc)) from exc
        return to_document_response(record)

    @app.get(
        "/api/v1/documents",
        response_model=DocumentListResponse,
        summary="List stored documents (optionally filter by tag / text)",
    )
    async def list_documents(
        tag: Annotated[str | None, Query(description="Filter to documents with this tag.")] = None,
        q: Annotated[
            str | None, Query(description="Substring search over title + body.")
        ] = None,
        limit: Annotated[int, Query(ge=1, le=1000, description="Page size.")] = 100,
        offset: Annotated[int, Query(ge=0, description="Page offset.")] = 0,
    ) -> DocumentListResponse:
        items = svc.list_documents(tag=tag, q=q, limit=limit, offset=offset)
        total = svc.count_documents(tag=tag)
        return to_document_list_response(
            items, total=total, tag=tag, q=q, limit=limit, offset=offset
        )

    @app.get(
        "/api/v1/documents/{doc_id}",
        response_model=DocumentResponse,
        summary="Get a single document by id",
    )
    async def get_document(
        doc_id: Annotated[str, Path(description="The document id.")],
    ) -> DocumentResponse:
        record = svc.get_document(doc_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"document not found: {doc_id}")
        return to_document_response(record)

    @app.patch(
        "/api/v1/documents/{doc_id}",
        response_model=DocumentResponse,
        summary="Partially update a document (title / tags / summary / metadata)",
    )
    async def update_document(
        doc_id: Annotated[str, Path(description="The document id.")],
        body: Annotated[DocumentUpdateRequest, Body(description="Fields to update.")],
    ) -> DocumentResponse:
        try:
            record = svc.update_document(
                doc_id,
                title=body.title,
                tags=body.tags,
                summary=body.summary,
                metadata=body.metadata,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_detail(exc)) from exc
        return to_document_response(record)

    @app.delete(
        "/api/v1/documents/{doc_id}",
        summary="Delete a document",
    )
    async def delete_document(
        doc_id: Annotated[str, Path(description="The document id.")],
    ) -> dict[str, bool]:
        deleted = svc.delete_document(doc_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"document not found: {doc_id}")
        return {"deleted": True}

    @app.post(
        "/api/v1/documents/{doc_id}/tags",
        response_model=DocumentResponse,
        summary="Re-extract tags for a document from its body",
    )
    async def retag_document(
        doc_id: Annotated[str, Path(description="The document id.")],
        body: Annotated[RetagRequest | None, Body(description="Retag options.")] = None,
    ) -> DocumentResponse:
        opts = body if body is not None else RetagRequest()
        try:
            record = svc.retag_document(
                doc_id,
                top_k=opts.top_k,
                replace=opts.replace,
                extra_tags=opts.extra_tags,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_detail(exc)) from exc
        return to_document_response(record)

    @app.get(
        "/api/v1/tags",
        response_model=TagsResponse,
        summary="List the distinct tag vocabulary across stored documents",
    )
    async def list_tags() -> TagsResponse:
        return TagsResponse(tags=svc.list_document_tags())

    app.state.service = svc
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
    "ENV_LOG_LEVEL",
    "ENV_MODEL",
    "ENV_STREAM",
    "build_default_service",
    "configure_logging",
    "create_app",
    "run",
]


if __name__ == "__main__":  # pragma: no cover
    run()
