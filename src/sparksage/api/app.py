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
from pathlib import Path
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
from sparksage.documents.summarizer import LLMSummarizer
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

# End-to-end QA
ENV_EMBEDDING_API_KEY = "SPARKSAGE_EMBEDDING_API_KEY"
ENV_EMBEDDING_BASE_URL = "SPARKSAGE_EMBEDDING_BASE_URL"
ENV_EMBEDDING_MODEL = "SPARKSAGE_EMBEDDING_MODEL"
ENV_ENABLE_QA = "SPARKSAGE_ENABLE_QA"

DEFAULT_MODEL = "gpt-4o-mini"
#: Streaming is on by default -- it is more robust for long generations.
DEFAULT_STREAM = True
#: Default SQLite table name for the durable document store.
DEFAULT_DOC_STORE_TABLE = "documents"
#: Default keyword-extraction algorithm used for auto-tagging.
DEFAULT_AUTO_TAG_EXTRACTOR = "rake"
#: Default embedding model for the QA knowledge base.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

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
    * Summarizer: an :class:`LLMSummarizer` reusing the same client when an API
      key is present (it degrades to the extractive summarizer on any LLM
      failure); the pure-stdlib :class:`ExtractiveSummarizer` otherwise.

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
    summarizer = None
    api_key = _env(ENV_API_KEY) or _env(ENV_OPENAI_API_KEY)
    if api_key:
        base_url = _env(ENV_BASE_URL) or _env(ENV_OPENAI_BASE_URL)
        model = _env(ENV_MODEL) or DEFAULT_MODEL
        language = _env("SPARKSAGE_LANGUAGE") or "zh"
        stream = _env_bool(ENV_STREAM, DEFAULT_STREAM)
        client = OpenAICompatibleClient(
            base_url=base_url, api_key=api_key, model=model, stream=stream
        )
        generator = IdeaBlockGenerator(client, language=language)
        # Reuse the same client for high-quality LLM summaries (degrades to the
        # extractive summarizer on any failure, so ingest never loses a summary).
        summarizer = LLMSummarizer(
            client, model=model, language=language, use_json_mode=False
        )
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
        summarizer=summarizer,
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


def build_qa_service():
    """Wire a full end-to-end :class:`QAService` from configuration.

    Composes the ingest pipeline (:class:`SparkSageService`) with the retrieval
    + answer pipeline (:class:`QAEngine` over a :class:`KnowledgeBase`), sharing
    one LLM client across generation / query-processing / answering and one
    embedding client across indexing / retrieval. Reads the same env vars as
    :func:`build_default_service` plus:

    ============================  =============================================
    ``SPARKSAGE_EMBEDDING_API_KEY``  Embedding API key (falls back to the LLM key)
    ``SPARKSAGE_EMBEDDING_BASE_URL`` Embedding base URL (falls back to LLM base URL)
    ``SPARKSAGE_EMBEDDING_MODEL``    Embedding model (default ``text-embedding-3-small``)
    ============================  =============================================

    When no LLM key is set the QA routes return ``503`` with a clear message
    (the full pipeline needs an LLM for both generation and answering).
    """
    from sparksage.api.qa_service import QAService
    from sparksage.embed.indexer import BlockEmbedder
    from sparksage.reader.faithfulness import LLMFaithfulnessJudge
    from sparksage.reader.generator import LLMAnswerGenerator
    from sparksage.reader.orchestrator import Reader

    load_dotenv()
    configure_logging()

    api_key = _env(ENV_API_KEY) or _env(ENV_OPENAI_API_KEY)
    if not api_key:
        _logger.warning(
            "no %s/%s set; the QA routes will return 503",
            ENV_API_KEY,
            ENV_OPENAI_API_KEY,
        )

    base_url = _env(ENV_BASE_URL) or _env(ENV_OPENAI_BASE_URL)
    model = _env(ENV_MODEL) or DEFAULT_MODEL
    stream = _env_bool(ENV_STREAM, DEFAULT_STREAM)
    language = _env("SPARKSAGE_LANGUAGE") or "en"

    llm_client: OpenAICompatibleClient | None = None
    generator: IdeaBlockGenerator | None = None
    if api_key:
        llm_client = OpenAICompatibleClient(
            base_url=base_url, api_key=api_key, model=model, stream=stream
        )
        generator = IdeaBlockGenerator(llm_client, language=language)

    spark_service = SparkSageService(
        converter=MarkdownConverter(backend=MarkItDownBackend()),
        cleaner=TextCleaner(),
        generator=generator,
        document_store=_build_document_store(),
        keyword_extractor=make_extractor(
            _env(ENV_AUTO_TAG_EXTRACTOR) or DEFAULT_AUTO_TAG_EXTRACTOR,
            tokenizer=AutoTokenizer(use_jieba=_env_bool(ENV_TAGS_ZH, False)),
        ),
    )

    embed_api_key = _env(ENV_EMBEDDING_API_KEY) or api_key
    embed_base_url = _env(ENV_EMBEDDING_BASE_URL) or base_url
    embed_model = _env(ENV_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL
    embedder = BlockEmbedder(
        _build_embedding_client(embed_api_key, embed_base_url, embed_model)
    )

    reader: Reader
    if llm_client is not None:
        reader = Reader(
            generator=LLMAnswerGenerator(llm_client),
            faithfulness_judge=LLMFaithfulnessJudge(llm_client),
        )
    else:
        reader = Reader(generator=_DummyAnswerGenerator())

    return QAService(
        service=spark_service,
        embedder=embedder,
        reader=reader,
    )


def _build_embedding_client(api_key, base_url, model):
    """Resolve an :class:`EmbeddingClient` from configuration.

    Falls back to a :class:`FakeEmbeddingClient` when no API key is available so
    the QA service stays constructible (and testable) with zero configuration --
    though retrieval quality will be meaningless without a real embedder.
    """
    from sparksage.embed.client import FakeEmbeddingClient

    if not api_key:
        _logger.warning(
            "no embedding API key; using FakeEmbeddingClient (retrieval is "
            "non-functional until a real key is set)"
        )
        return FakeEmbeddingClient(dimension=128)
    from sparksage.embed.client import OpenAIEmbeddingClient

    return OpenAIEmbeddingClient(
        base_url=base_url, api_key=api_key, model=model
    )


class _DummyAnswerGenerator:
    """Stand-in answer generator used when no LLM is configured.

    Always abstains with a clear message so the QA route returns a helpful 503
    rather than crashing when the LLM key is missing.
    """

    def generate(self, query, chunks):  # noqa: ARG002
        from sparksage.reader.schema import GeneratedAnswer

        return GeneratedAnswer(
            text="",
            citations=[],
            grounded_block_ids=[],
            confidence=0.0,
            abstained=True,
            abstention_reason="no LLM configured; set SPARKSAGE_API_KEY to enable answering",
        )


def create_app(
    service: SparkSageService | None = None,
    *,
    qa_service: Any = None,
) -> Any:
    """Create and configure a FastAPI application.

    Parameters
    ----------
    service:
        A pre-built :class:`SparkSageService`. When omitted,
        :func:`build_default_service` is used (which reads env vars). Inject a
        custom service (e.g. with fakes) for testing.
    qa_service:
        An optional pre-built :class:`QAService`. When omitted, the end-to-end
        QA routes (``/api/v1/query``, ``/api/v1/knowledge_base/...``,
        ``/api/v1/feedback``) are *not* mounted. Pass an instance (or call
        :func:`build_qa_service`) to enable the full knowledge-QA loop. As a
        convenience, when neither ``service`` nor ``qa_service`` is provided and
        the ``SPARKSAGE_ENABLE_QA`` env var is truthy, :func:`build_qa_service`
        is used automatically -- this is how the Docker image exposes the full
        pipeline out of the box (``SPARKSAGE_ENABLE_QA=1``).

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
    from sparksage.api.config_manager import ConfigError, read_config, write_config
    from sparksage.api.schemas import (
        ConfigResponse,
        ConfigUpdateResponse,
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

    if service is not None:
        svc = service
    elif qa_service is not None:
        svc = qa_service.service
    elif _env_bool(ENV_ENABLE_QA, False):
        qa_service = build_qa_service()
        svc = qa_service.service
    else:
        svc = build_default_service()

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

    # ------------------------------------------------------------------ #
    # configuration management routes (edit the .env from the UI)
    # ------------------------------------------------------------------ #
    @app.get(
        "/api/v1/config",
        response_model=ConfigResponse,
        summary="Read the effective configuration (secrets masked)",
    )
    async def get_config() -> ConfigResponse:
        return ConfigResponse(variables=read_config())

    @app.post(
        "/api/v1/config",
        response_model=ConfigUpdateResponse,
        summary="Write a patch of configuration values to the .env file",
    )
    async def update_config(
        body: Annotated[
            dict[str, Any],
            Body(description="A {KEY: value} patch to apply to the .env file."),
        ],
    ) -> ConfigUpdateResponse:
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="expected a JSON object")
        str_updates = {str(k): ("" if v is None else str(v)) for k, v in body.items()}
        try:
            applied = write_config(str_updates)
        except ConfigError as exc:
            raise HTTPException(status_code=422, detail=_detail(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=_detail(exc)) from exc
        return ConfigUpdateResponse(
            applied=applied,
            restart_required=True,
            message="配置已保存。请手动重启服务使配置生效。",
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
        tag: Annotated[
            str | None,
            Query(description="Comma-separated tag filter (any-match OR)."),
        ] = None,
        q: Annotated[
            str | None, Query(description="Substring search over title + body.")
        ] = None,
        limit: Annotated[int, Query(ge=1, le=1000, description="Page size.")] = 100,
        offset: Annotated[int, Query(ge=0, description="Page offset.")] = 0,
    ) -> DocumentListResponse:
        tags = None
        if tag:
            tags = [p.strip() for p in tag.split(",") if p.strip()]
        items = svc.list_documents(tags=tags, q=q, limit=limit, offset=offset)
        total = svc.count_documents(tags=tags)
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

    # ------------------------------------------------------------------ #
    # end-to-end QA routes (mounted only when a QAService is provided)
    # ------------------------------------------------------------------ #
    if qa_service is not None:
        _mount_qa_routes(app, qa_service)

    # ------------------------------------------------------------------ #
    # optional static frontend (serve the built WEB UI from one origin)
    # ------------------------------------------------------------------ #
    _mount_static_frontend(app)

    app.state.service = svc
    if qa_service is not None:
        app.state.qa_service = qa_service
    return app


def _resolve_web_dist() -> Path | None:
    """Locate the built frontend (``web/dist``) if present.

    Resolution order: the ``SPARKSAGE_WEB_DIST`` env var (explicit override),
    then a ``web/dist`` directory relative to the CWD, then one relative to the
    package root. Returns ``None`` when no build exists (the API runs headless).
    """
    explicit = _env("SPARKSAGE_WEB_DIST")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(Path.cwd() / "web" / "dist")
    candidates.append(Path(__file__).resolve().parent.parent.parent.parent / "web" / "dist")
    for c in candidates:
        if c.is_dir() and (c / "index.html").is_file():
            return c
    return None


def _mount_static_frontend(app: Any) -> None:
    """Serve the built WEB UI (Vite ``dist``) behind a catch-all route.

    Mounted last so every ``/api/...`` route wins first. Non-API GET requests
    fall through to the SPA ``index.html``; any method on an unknown ``/api/``
    path returns a real ``404`` (not a ``405``) so API consumers get a clean
    "not found". A no-op when no build is found (the API runs headless).

    Uses ``include_in_schema=False`` so the catch-all does not clutter the
    OpenAPI docs.
    """
    dist = _resolve_web_dist()
    if dist is None:
        return

    from starlette.staticfiles import StaticFiles

    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="sparksage-assets")
    if (dist / "favicon.svg").is_file():
        app.mount(
            "/favicon.svg",
            StaticFiles(directory=str(dist), html=False),
            name="sparksage-favicon",
        )

    index_html = dist / "index.html"

    from fastapi import Request

    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
        include_in_schema=False,
    )
    async def spa(full_path: str, request: Request) -> Any:
        from fastapi import HTTPException
        from fastapi.responses import FileResponse

        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        if request.method not in ("GET", "HEAD"):
            raise HTTPException(status_code=405, detail="Method Not Allowed")
        candidate = dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(index_html))


def _mount_qa_routes(app: Any, qa_svc: Any) -> None:
    """Mount the end-to-end QA routes onto ``app`` using ``qa_svc``.

    Extracted from :func:`create_app` so the QA wiring stays readable and the
    base convert/generate/documents routes work unchanged when no QA service is
    provided.
    """
    from typing import Annotated

    from fastapi import Body, File, Form, HTTPException, Path, Query, UploadFile

    from sparksage.api.pipeline import GenerationNotConfiguredError
    from sparksage.api.qa_service import QAService
    from sparksage.api.schemas import (
        AskRequest,
        AskResponse,
        BlockListResponse,
        CreateKnowledgeBaseRequest,
        FeedbackListResponse,
        FeedbackRecordOut,
        FeedbackRequest,
        FeedbackResponse,
        FeedbackStatsResponse,
        IngestAndIndexResponse,
        KnowledgeBaseListResponse,
        KnowledgeBaseResponse,
        KnowledgeBaseSummary,
        QueryHistoryItem,
        QueryHistoryResponse,
        TagsResponse,
        _build_filter_from_request,
        _to_ask_response,
        _to_block_out,
        _to_ingest_response,
    )

    if not isinstance(qa_svc, QAService):
        raise TypeError("qa_service must be a QAService instance")

    @app.post(
        "/api/v1/knowledge_base/ingest",
        response_model=IngestAndIndexResponse,
        summary="Upload knowledge: parse -> chunk -> embed -> index",
    )
    async def kb_ingest(
        file: Annotated[
            UploadFile, File(description="The source document to ingest and index.")
        ],
        title: Annotated[
            str | None, Form(description="Explicit title override.")
        ] = None,
        tags: Annotated[
            str | None,
            Form(description="Comma-separated tags. When empty, tags are auto-extracted."),
        ] = None,
        auto_tag: Annotated[
            bool, Form(description="Auto-extract tags when none are given.")
        ] = True,
        clean: Annotated[
            bool, Form(description="Apply text cleaning before generation.")
        ] = True,
        summarize: Annotated[
            bool, Form(description="Produce a document-level summary.")
        ] = True,
        top_k: Annotated[
            int, Form(ge=1, description="Number of tags to extract when auto-tagging.")
        ] = 8,
        max_blocks: Annotated[
            int | None, Form(ge=1, description="Max IdeaBlocks to emit.")
        ] = None,
        language: Annotated[
            str | None, Form(description="BCP-47 code written into every block.")
        ] = None,
        kb_id: Annotated[
            str | None,
            Form(description="Target knowledge base id (defaults to the active KB)."),
        ] = None,
    ) -> IngestAndIndexResponse:
        data = await file.read()
        parsed_tags = None
        if tags is not None:
            parsed_tags = [p.strip() for p in tags.split(",") if p.strip()]
        try:
            result = qa_svc.ingest_and_index(
                data,
                file.filename,
                title=title,
                tags=parsed_tags,
                auto_tag=auto_tag,
                clean=clean,
                summarize=summarize,
                top_k=top_k,
                max_blocks=max_blocks,
                language=language,
                kb_id=kb_id,
            )
        except GenerationNotConfiguredError as exc:
            raise HTTPException(status_code=503, detail=_detail(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_detail(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=_detail(exc)) from exc
        return _to_ingest_response(result)

    @app.post(
        "/api/v1/query",
        response_model=AskResponse,
        summary="Ask a question against a knowledge base",
    )
    async def ask(
        body: Annotated[AskRequest, Body(description="The question + options.")],
    ) -> AskResponse:
        flt, context = _build_filter_from_request(body)
        try:
            result = qa_svc.ask(
                body.query,
                context=context,
                filter=flt,
                k=body.k,
                use_lexical=body.use_lexical,
                use_rerank=body.use_rerank,
                kb_id=body.kb_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_detail(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=_detail(exc)) from exc
        return _to_ask_response(result)

    @app.get(
        "/api/v1/knowledge_base",
        response_model=KnowledgeBaseResponse,
        summary="Active knowledge-base snapshot (block / document counts)",
    )
    async def kb_info(
        kb_id: Annotated[
            str | None,
            Query(description="Target KB (defaults to the active KB)."),
        ] = None,
    ) -> KnowledgeBaseResponse:
        try:
            info = qa_svc.knowledge_base_info(kb_id=kb_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_detail(exc)) from exc
        return KnowledgeBaseResponse(**info)

    # ------------------------------------------------------------------ #
    # multi-knowledge-base management (create / list / delete / activate)
    # ------------------------------------------------------------------ #
    @app.post(
        "/api/v1/knowledge_bases",
        response_model=KnowledgeBaseSummary,
        status_code=201,
        summary="Create a new knowledge base",
    )
    async def create_kb(
        body: Annotated[
            CreateKnowledgeBaseRequest, Body(description="The new KB metadata.")
        ],
    ) -> KnowledgeBaseSummary:
        try:
            info = qa_svc.create_knowledge_base(
                body.name,
                description=body.description,
                language=body.language,
                tags=body.tags,
                set_active=body.set_active,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=_detail(exc)) from exc
        snap = qa_svc.get_knowledge_base_info(info.kb_id)
        return _to_kb_summary(snap or {})

    @app.get(
        "/api/v1/knowledge_bases",
        response_model=KnowledgeBaseListResponse,
        summary="List all knowledge bases",
    )
    async def list_kbs(
        limit: Annotated[int, Query(ge=1, le=1000, description="Page size.")] = 100,
        offset: Annotated[int, Query(ge=0, description="Page offset.")] = 0,
    ) -> KnowledgeBaseListResponse:
        page, total = qa_svc.list_knowledge_bases(limit=limit, offset=offset)
        items = [_to_kb_summary(p) for p in page]
        return KnowledgeBaseListResponse(
            items=items, count=len(items), total=total, limit=limit, offset=offset
        )

    @app.get(
        "/api/v1/knowledge_bases/{kb_id}",
        response_model=KnowledgeBaseSummary,
        summary="Get a single knowledge base by id",
    )
    async def get_kb(
        kb_id: Annotated[str, Path(description="The knowledge base id.")],
    ) -> KnowledgeBaseSummary:
        snap = qa_svc.get_knowledge_base_info(kb_id)
        if snap is None:
            raise HTTPException(
                status_code=404, detail=f"knowledge base not found: {kb_id}"
            )
        return _to_kb_summary(snap)

    @app.delete(
        "/api/v1/knowledge_bases/{kb_id}",
        summary="Delete a knowledge base (metadata + live index)",
    )
    async def delete_kb(
        kb_id: Annotated[str, Path(description="The knowledge base id.")],
    ) -> dict[str, bool]:
        try:
            deleted = qa_svc.delete_knowledge_base(kb_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=_detail(exc)) from exc
        if not deleted:
            raise HTTPException(
                status_code=404, detail=f"knowledge base not found: {kb_id}"
            )
        return {"deleted": True}

    @app.post(
        "/api/v1/knowledge_bases/{kb_id}/activate",
        response_model=KnowledgeBaseSummary,
        summary="Set a knowledge base as the active routing target",
    )
    async def activate_kb(
        kb_id: Annotated[str, Path(description="The knowledge base id.")],
    ) -> KnowledgeBaseSummary:
        try:
            qa_svc.set_active_knowledge_base(kb_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_detail(exc)) from exc
        snap = qa_svc.get_knowledge_base_info(kb_id)
        return _to_kb_summary(snap or {})

    @app.delete(
        "/api/v1/knowledge_base/documents/{doc_id}",
        summary="Remove a document and cascade-remove its indexed blocks",
    )
    async def kb_remove_document(
        doc_id: Annotated[str, Path(description="The document id.")],
        kb_id: Annotated[
            str | None,
            Query(description="Target KB (defaults to the active KB)."),
        ] = None,
    ) -> dict[str, bool]:
        deleted = qa_svc.remove_document(doc_id, kb_id=kb_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"document not found: {doc_id}")
        return {"deleted": True}

    @app.post(
        "/api/v1/feedback",
        response_model=FeedbackResponse,
        summary="Record a user verdict on a surfaced answer",
    )
    async def record_feedback(
        body: Annotated[FeedbackRequest, Body(description="The feedback record.")],
    ) -> FeedbackResponse:
        valid = {"positive", "negative", "corrected"}
        if body.rating not in valid:
            raise HTTPException(
                status_code=422,
                detail=f"rating must be one of {sorted(valid)}",
            )
        record = qa_svc.add_feedback(
            body.query,
            body.answer_text,
            body.rating,
            correction=body.correction,
            block_ids=body.block_ids,
        )
        return FeedbackResponse(
            feedback_id=record.feedback_id,
            rating=record.rating.value,
            acknowledged=True,
        )

    @app.get(
        "/api/v1/feedback",
        response_model=FeedbackStatsResponse,
        summary="Aggregate feedback stats (approval ratio)",
    )
    async def feedback_stats() -> FeedbackStatsResponse:
        stats = qa_svc.feedback_stats()
        return FeedbackStatsResponse(
            total=stats.total,
            positive=stats.positive,
            negative=stats.negative,
            corrected=stats.corrected,
            approval=stats.approval,
        )

    @app.get(
        "/api/v1/knowledge_base/blocks",
        response_model=BlockListResponse,
        summary="List indexed IdeaBlocks (filter by tag / language / status)",
    )
    async def kb_list_blocks(
        tag: Annotated[
            str | None,
            Query(description="Comma-separated tag filter (any-match OR)."),
        ] = None,
        language: Annotated[
            str | None, Query(description="Restrict to this language code.")
        ] = None,
        status: Annotated[
            str | None,
            Query(description="Lifecycle status: ACTIVE / MERGED / DRAFT / ARCHIVED."),
        ] = None,
        kb_id: Annotated[
            str | None,
            Query(description="Target KB (defaults to the active KB)."),
        ] = None,
        limit: Annotated[int, Query(ge=1, le=1000, description="Page size.")] = 100,
        offset: Annotated[int, Query(ge=0, description="Page offset.")] = 0,
    ) -> BlockListResponse:
        tags = None
        if tag:
            tags = [p.strip() for p in tag.split(",") if p.strip()]
        try:
            page, total = qa_svc.list_blocks(
                tags=tags,
                language=language,
                status=status,
                limit=limit,
                offset=offset,
                kb_id=kb_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_detail(exc)) from exc
        items = [_to_block_out(b) for b in page]
        return BlockListResponse(
            items=items, count=len(items), total=total, limit=limit, offset=offset
        )

    @app.get(
        "/api/v1/knowledge_base/tags",
        response_model=TagsResponse,
        summary="Distinct tag vocabulary across indexed IdeaBlocks",
    )
    async def kb_list_tags(
        kb_id: Annotated[
            str | None,
            Query(description="Target KB (defaults to the active KB)."),
        ] = None,
    ) -> TagsResponse:
        try:
            return TagsResponse(tags=qa_svc.list_block_tags(kb_id=kb_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=_detail(exc)) from exc

    @app.get(
        "/api/v1/feedback/records",
        response_model=FeedbackListResponse,
        summary="List recent feedback records (newest-first)",
    )
    async def feedback_list(
        limit: Annotated[int, Query(ge=1, le=1000, description="Page size.")] = 50,
        offset: Annotated[int, Query(ge=0, description="Page offset.")] = 0,
    ) -> FeedbackListResponse:
        page, total = qa_svc.list_feedback(limit=limit, offset=offset)
        items = [
            FeedbackRecordOut(
                feedback_id=r.feedback_id,
                query=r.query,
                answer_text=r.answer_text,
                rating=r.rating.value,
                correction=r.correction,
                block_ids=list(r.block_ids),
                kb_id=r.kb_id,
                created_at=r.created_at,
                metadata=dict(r.metadata),
            )
            for r in page
        ]
        return FeedbackListResponse(
            items=items, count=len(items), total=total, limit=limit, offset=offset
        )

    @app.get(
        "/api/v1/query/history",
        response_model=QueryHistoryResponse,
        summary="List the persisted QA conversation history (newest-first)",
    )
    async def query_history(
        kb_id: Annotated[
            str | None,
            Query(description="Target KB (defaults to the active KB)."),
        ] = None,
        limit: Annotated[int, Query(ge=1, le=1000, description="Page size.")] = 100,
        offset: Annotated[int, Query(ge=0, description="Page offset.")] = 0,
    ) -> QueryHistoryResponse:
        page, total = qa_svc.list_history(limit=limit, offset=offset, kb_id=kb_id)
        items = [
            QueryHistoryItem(
                turn_id=t.turn_id,
                role=t.role.value,
                content=t.content,
                kb_id=t.kb_id,
                result=t.result,
                created_at=t.created_at,
            )
            for t in page
        ]
        return QueryHistoryResponse(
            items=items, count=len(items), total=total, limit=limit, offset=offset
        )

    @app.delete(
        "/api/v1/query/history",
        summary="Clear the persisted QA conversation history",
    )
    async def clear_history(
        kb_id: Annotated[
            str | None,
            Query(description="Target KB (defaults to the active KB)."),
        ] = None,
    ) -> dict[str, int]:
        removed = qa_svc.clear_history(kb_id=kb_id)
        return {"removed": removed}


def _detail(exc: BaseException) -> str:
    msg = str(exc)
    return msg or exc.__class__.__name__


def _to_kb_summary(snap: dict[str, Any]) -> Any:
    """Build a :class:`KnowledgeBaseSummary` from a QAService KB snapshot dict."""
    from sparksage.api.schemas import KnowledgeBaseSummary

    return KnowledgeBaseSummary(
        kb_id=snap["kb_id"],
        name=snap["name"],
        description=snap.get("description"),
        language=snap.get("language", "en"),
        tags=list(snap.get("tags", [])),
        block_count=int(snap.get("block_count", 0)),
        document_count=int(snap.get("document_count", 0)),
        active=bool(snap.get("active", False)),
        created_at=snap["created_at"],
        updated_at=snap["updated_at"],
    )


def run(  # pragma: no cover - thin launcher
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Convenience launcher: ``python -m sparksage.api.app``.

    Passes a unified ``log_config`` so uvicorn's startup / access logs share the
    same ``%(asctime)s %(levelname)s %(name)s:`` shape as the ``sparksage``
    application logger (see :func:`sparksage.logging_config.build_uvicorn_log_config`).
    """
    import uvicorn

    from sparksage.logging_config import build_uvicorn_log_config

    uvicorn.run(
        "sparksage.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        log_config=build_uvicorn_log_config(),
    )


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_STREAM",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_ENABLE_QA",
    "ENV_EMBEDDING_API_KEY",
    "ENV_EMBEDDING_BASE_URL",
    "ENV_EMBEDDING_MODEL",
    "ENV_LOG_LEVEL",
    "ENV_MODEL",
    "ENV_STREAM",
    "build_default_service",
    "build_qa_service",
    "configure_logging",
    "create_app",
    "run",
]


if __name__ == "__main__":  # pragma: no cover
    run()
