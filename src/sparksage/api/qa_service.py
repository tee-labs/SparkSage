"""Framework-agnostic end-to-end QA orchestration service.

:class:`QAService` is the glue that finally turns SparkSage into a *runnable*
knowledge-QA product over HTTP (or any other transport). It composes the three
existing layers -- :class:`SparkSageService` (ingest), :class:`KnowledgeBase`
(index + retrieval), :class:`QAEngine` (query -> retrieval -> answer) -- plus
the optional :class:`FeedbackStore` flywheel into one object a thin route layer
can call.

The complete product pipeline:

    INGEST (upload knowledge)
        bytes -> convert -> clean -> generate IdeaBlocks
             -> embed + index into KnowledgeBase
             -> store DocumentRecord (+ auto-tag + summary)

    QUERY (ask a question)
        user question
            -> QueryProcessor   (classify intent -> intercept -> rewrite)
            -> Retriever        (dense kNN + BM25 -> RRF fuse -> rerank -> filter)
            -> Reader           (generate grounded answer -> judge faithfulness)
            -> answer with citations (or a principled abstention)

    FEEDBACK (close the loop)
        user verdict on an answer
            -> FeedbackStore
            -> aggregate (approval ratio, per-block breakdown)
            -> healing signals (coverage gaps / split candidates)

It is deliberately framework-agnostic (no FastAPI / HTTP imports here) so it is
fully unit-testable offline with :class:`FakeConverterBackend` /
:class:`FakeLLMClient` / :class:`FakeEmbeddingClient`. The only concern it owns
beyond wiring is the *consistency* between the ingest path and the retrieval
path: blocks generated from an uploaded file are the same blocks that get
embedded, indexed, and ultimately retrieved -- all flowing through the shared
:class:`KnowledgeBase`.

Multi-tenant: the service holds a registry of :class:`KnowledgeBase` aggregates
keyed by ``kb_id`` (live index state) backed by a :class:`KnowledgeBaseStore`
(persistable metadata). Operations accept an optional ``kb_id`` to target a
specific KB (defaulting to the active one), so each tenant's documents, blocks
and indexes stay isolated while sharing one embedder / reader / query processor.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sparksage.agent.controller import AgentController
from sparksage.agent.engine import DEFAULT_MAX_ITERATIONS as DEFAULT_MAX_AGENT_ITERATIONS
from sparksage.agent.engine import AgenticQAEngine
from sparksage.agent.models import AgentResult
from sparksage.api.ingest_jobs import (
    IngestCancelled,
    IngestJob,
    IngestJobManager,
    ProgressCallback,
)
from sparksage.api.pipeline import (
    GenerationNotConfiguredError,
    SparkSageService,
)
from sparksage.documents.models import DocumentRecord, content_hash_of, new_record
from sparksage.embed.indexer import BlockEmbedder
from sparksage.feedback.models import FeedbackRating, FeedbackRecord
from sparksage.feedback.store import FeedbackStats, FeedbackStore, InMemoryFeedbackStore
from sparksage.kb.backends.state import KbStateStore
from sparksage.kb.knowledge_base import KnowledgeBase
from sparksage.kb.models import KnowledgeBaseInfo
from sparksage.kb.store import InMemoryKnowledgeBaseStore, KnowledgeBaseStore
from sparksage.qa.engine import QAEngine, QAResult
from sparksage.qa.history import (
    InMemoryQASessionStore,
    QASessionStore,
    QATurn,
    TurnRole,
)
from sparksage.query.context import ConversationContext
from sparksage.query.expander import QueryExpander
from sparksage.query.processor import QueryProcessor
from sparksage.query.refiner import QueryRefiner
from sparksage.reader.orchestrator import Reader
from sparksage.retrieve.grader import RetrievalGrader
from sparksage.retrieve.models import RetrievalFilter
from sparksage.retrieve.reranker import Reranker
from sparksage.schema.ideablock import IdeaBlock
from sparksage.schema.source import SourceRef

_logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Framework-agnostic result of an ingest-and-index request.

    Captures the doc id, the generated/indexed blocks, and the diagnostics a
    front-end needs to show "uploaded N, produced M chunks".
    """

    doc_id: str
    blocks: list[IdeaBlock]
    block_count: int
    title: str | None
    source: SourceRef
    tags: list[str]
    summary: str | None
    action: str = "created"


class QAService:
    """End-to-end knowledge-QA orchestrator: ingest -> index -> query -> feedback.

    Parameters
    ----------
    service:
        The :class:`SparkSageService` used for the *ingest* half (convert ->
        clean -> generate -> store). Owns the converter, cleaner, generator,
        keyword extractor, summarizer, and document store.
    embedder:
        The :class:`BlockEmbedder` used to vectorize blocks and queries. Shared
        between the :class:`KnowledgeBase` (index) and the :class:`QAEngine`
        (retrieval) so ingest and query use one embedding model.
    reader:
        The :class:`Reader` used for answer generation + faithfulness judging.
    query_processor:
        Optional :class:`QueryProcessor` (intent classification + rewrite). When
        ``None`` the raw query goes straight to retrieval.
    kb:
        The default :class:`KnowledgeBase` aggregate that owns documents +
        blocks + the dense + lexical indexes. Defaults to a fresh ``"default"``
        KB. Additional KBs can be created at runtime via
        :meth:`create_knowledge_base`.
    reranker:
        Optional :class:`Reranker` forwarded to every :class:`KnowledgeBase`
        aggregate's :class:`Retriever`. When ``None`` (default) retrieval runs
        without re-ranking regardless of ``use_rerank`` -- wire e.g. a
        :class:`~sparksage.retrieve.backends.CrossEncoderReranker` to make
        ``use_rerank=True`` actually re-order the candidate pool.
    kb_store:
        Optional :class:`KnowledgeBaseStore` registry for multi-tenant KB
        metadata. Defaults to an :class:`InMemoryKnowledgeBaseStore` so KB CRUD
        works out of the box. Pass a durable store (e.g.
        :class:`~sparksage.kb.SqliteKnowledgeBaseStore`) so KB metadata
        survives a restart.
    feedback_store:
        Optional :class:`FeedbackStore` for the quality flywheel. Defaults to an
        :class:`InMemoryFeedbackStore` so feedback works out of the box. Pass a
        durable store (e.g. :class:`~sparksage.feedback.SqliteFeedbackStore`)
        so approval ratios survive a restart.
    state_store:
        Optional :class:`~sparksage.kb.KbStateStore` for the live block
        registry + vector index + document<->block linkage. When set, every
        :class:`KnowledgeBase` aggregate (including ones created later via
        :meth:`create_knowledge_base`) writes its block/vector mutations
        through to it, and on construction the service reloads every persisted
        KB so a restart does not lose indexed knowledge or require re-embedding.
        Defaults to ``None`` (ephemeral in-memory indexes).
    history_store:
        Optional :class:`QASessionStore` for the persisted QA conversation log
        (the query/answer history the Q&A page restores on reload). Defaults to
        an :class:`InMemoryQASessionStore`.

    Examples
    --------
    >>> from sparksage import (
    ...     BlockEmbedder, FakeEmbeddingClient, FakeConverterBackend,
    ...     FakeLLMClient, IdeaBlockGenerator, MarkdownConverter, SparkSageService,
    ...     TextCleaner, LLMAnswerGenerator, Reader, BM25Retriever,
    ... )
    >>> from sparksage.api.qa_service import QAService  # doctest: +SKIP
    >>> svc = QAService(                                  # doctest: +SKIP
    ...     service=SparkSageService(
    ...         converter=MarkdownConverter(backend=FakeConverterBackend("# hi")),
    ...         generator=IdeaBlockGenerator(FakeLLMClient()),
    ...     ),
    ...     embedder=BlockEmbedder(FakeEmbeddingClient()),
    ...     reader=Reader(generator=LLMAnswerGenerator(FakeLLMClient())),
    ... )                                                  # doctest: +SKIP
    """

    def __init__(
        self,
        service: SparkSageService,
        embedder: BlockEmbedder,
        reader: Reader,
        *,
        query_processor: QueryProcessor | None = None,
        kb: KnowledgeBase | None = None,
        reranker: Reranker | None = None,
        kb_store: KnowledgeBaseStore | None = None,
        feedback_store: FeedbackStore | None = None,
        state_store: KbStateStore | None = None,
        history_store: QASessionStore | None = None,
        agent_controller: AgentController | None = None,
        agent_max_iterations: int = DEFAULT_MAX_AGENT_ITERATIONS,
        agent_retrieval_grader: RetrievalGrader | None = None,
        agent_query_refiner: QueryRefiner | None = None,
        agent_query_expander: QueryExpander | None = None,
        agent_step_min_relevance: float | None = None,
        agent_step_max_refine: int | None = None,
        agent_expander_n_variants: int | None = None,
        agent_max_stale_steps: int | None = None,
        ingest_jobs: IngestJobManager | None = None,
    ) -> None:
        self._service = service
        self._embedder = embedder
        self._reader = reader
        self._query_processor = query_processor
        self._reranker: Reranker | None = reranker
        self._kb_store: KnowledgeBaseStore = (
            kb_store if kb_store is not None else InMemoryKnowledgeBaseStore()
        )
        self._state_store: KbStateStore | None = state_store
        # Live KnowledgeBase aggregates keyed by kb_id (runtime index state).
        self._kbs: dict[str, KnowledgeBase] = {}
        # Lazily built QAEngine per KB (shares reader + query_processor).
        self._engines: dict[str, QAEngine] = {}
        # Lazily built AgenticQAEngine per KB (only when an agent controller is wired).
        self._agent_engines: dict[str, AgenticQAEngine] = {}
        self._agent_controller = agent_controller
        self._agent_max_iterations = agent_max_iterations
        # Per-KB agent engines inherit these optional reflection components
        # (the missing middle gate of the three-stage policy finally connected
        # to the agent loop): the grader + refiner drive the per-step refine +
        # re-retrieve cycle, the expander drives per-step multi-query RRF.
        self._agent_retrieval_grader = agent_retrieval_grader
        self._agent_query_refiner = agent_query_refiner
        self._agent_query_expander = agent_query_expander
        self._agent_step_min_relevance = agent_step_min_relevance
        self._agent_step_max_refine = agent_step_max_refine
        self._agent_expander_n_variants = agent_expander_n_variants
        self._agent_max_stale_steps = agent_max_stale_steps

        # Async ingest job registry. A long ingest (minutes on a large doc)
        # must not hold open an HTTP connection; ``submit_ingest`` returns a
        # job id immediately and ``GET /jobs/{id}`` polls the snapshot.
        # Defaults to a fresh in-process manager so async ingest works with
        # zero configuration.
        self._ingest_jobs: IngestJobManager = (
            ingest_jobs if ingest_jobs is not None else IngestJobManager()
        )

        if kb is not None:
            default_kb = kb
            self._register_kb(default_kb)
        else:
            # Reload every persisted KB from the durable store (if any), then
            # fall back to creating a fresh "default" KB when none exist. This
            # is what makes a Docker restart not lose indexed knowledge: the
            # block registry + vectors + doc-links come straight back off disk
            # via the state store, with no re-embedding.
            loaded = self._reload_persisted_kbs()
            if loaded:
                default_kb = self._kbs[loaded[0]]
            else:
                default_kb = KnowledgeBase(
                    info=KnowledgeBaseInfo(name="default"),
                    embedder=embedder,
                    document_store=self._service.document_store,
                    reranker=self._reranker,
                    state_store=self._state_store,
                )
                self._register_kb(default_kb)
        self._active_kb_id: str = default_kb.kb_id

        self._feedback_store: FeedbackStore = (
            feedback_store if feedback_store is not None else InMemoryFeedbackStore()
        )
        self._history_store: QASessionStore = (
            history_store if history_store is not None else InMemoryQASessionStore()
        )

    # ------------------------------------------------------------------ #
    # owned components
    # ------------------------------------------------------------------ #
    @property
    def service(self) -> SparkSageService:
        return self._service

    @property
    def embedder(self) -> BlockEmbedder:
        return self._embedder

    @property
    def reader(self) -> Reader:
        return self._reader

    @property
    def query_processor(self) -> QueryProcessor | None:
        return self._query_processor

    @property
    def knowledge_base(self) -> KnowledgeBase:
        """The currently *active* knowledge base (default routing target)."""
        return self._kbs[self._active_kb_id]

    @property
    def active_kb_id(self) -> str:
        """The ``kb_id`` operations target when none is supplied."""
        return self._active_kb_id

    @property
    def kb_store(self) -> KnowledgeBaseStore:
        return self._kb_store

    @property
    def state_store(self) -> KbStateStore | None:
        """The durable KB-state backend, or ``None`` when persistence is off."""
        return self._state_store

    @property
    def engine(self) -> QAEngine:
        """The QAEngine bound to the active knowledge base."""
        return self._engine_for(self._active_kb_id)

    @property
    def feedback_store(self) -> FeedbackStore:
        return self._feedback_store

    @property
    def history_store(self) -> QASessionStore:
        return self._history_store

    @property
    def has_generator(self) -> bool:
        """Whether block generation is available (ingest-and-index needs this)."""
        return self._service.has_generator

    @property
    def ingest_jobs(self) -> IngestJobManager:
        """The async ingest job registry (backs ``POST .../ingest/async``)."""
        return self._ingest_jobs

    # ------------------------------------------------------------------ #
    # multi-knowledge-base management
    # ------------------------------------------------------------------ #
    def _reload_persisted_kbs(self) -> list[str]:
        """Reconstruct every persisted KB aggregate from the durable stores.

        Reads the :class:`KnowledgeBaseInfo` metadata from ``kb_store`` and
        rebuilds each :class:`KnowledgeBase` with the shared embedder + document
        store + ``state_store`` (which hydrates the block registry + vectors +
        doc-links). Returns the list of loaded ``kb_id``s, newest-first, so the
        caller can pick the active target. A no-op when the store is empty (a
        fresh start) or when no ``state_store`` is wired (the metadata is still
        reloaded so KB CRUD state survives, but the indexes start empty).
        """
        if len(self._kb_store) == 0:
            return []
        infos = self._kb_store.list(limit=10**9)
        loaded: list[str] = []
        for info in infos:
            kb = KnowledgeBase(
                info=info,
                embedder=self._embedder,
                document_store=self._service.document_store,
                reranker=self._reranker,
                state_store=self._state_store,
            )
            self._kbs[kb.kb_id] = kb
            loaded.append(kb.kb_id)
        if loaded:
            _logger.info(
                "reloaded %d persisted knowledge base(s) (state_store=%s)",
                len(loaded),
                self._state_store is not None,
            )
        return loaded

    def _register_kb(self, kb: KnowledgeBase) -> KnowledgeBase:
        """Register a live aggregate + persist its metadata to the store."""
        self._kbs[kb.kb_id] = kb
        self._kb_store.save(kb.info)
        self._engines.pop(kb.kb_id, None)
        self._agent_engines.pop(kb.kb_id, None)
        return kb

    def _engine_for(self, kb_id: str) -> QAEngine:
        """Return (building lazily) the QAEngine bound to ``kb_id``'s retriever."""
        if kb_id not in self._engines:
            self._engines[kb_id] = QAEngine(
                retriever=self._kbs[kb_id].retriever,
                reader=self._reader,
                query_processor=self._query_processor,
            )
        return self._engines[kb_id]

    @property
    def agent_enabled(self) -> bool:
        """Whether an agentic QA controller is wired (``ask(mode="agent")``)."""
        return self._agent_controller is not None

    def _agent_engine_for(self, kb_id: str) -> AgenticQAEngine:
        """Return (building lazily) the AgenticQAEngine bound to ``kb_id``.

        Reuses the per-KB :class:`Retriever`, the shared :class:`Reader` and the
        shared :class:`QueryProcessor` (so the agent inherits the out-of-domain
        interception gate + rewrite-seeded first retrieval), and the wired
        :class:`AgentController`. Also wires the optional per-step reflection
        components (grader / refiner / expander) when supplied on this service.
        """
        if self._agent_controller is None:
            raise RuntimeError(
                "no agent_controller is wired on this QAService; pass one to "
                "QAService(..., agent_controller=...) to enable mode='agent'"
            )
        if kb_id not in self._agent_engines:
            kwargs: dict[str, Any] = {
                "max_iterations": self._agent_max_iterations,
            }
            if self._agent_retrieval_grader is not None:
                kwargs["retrieval_grader"] = self._agent_retrieval_grader
            if self._agent_query_refiner is not None:
                kwargs["query_refiner"] = self._agent_query_refiner
            if self._agent_query_expander is not None:
                kwargs["query_expander"] = self._agent_query_expander
            if self._agent_step_min_relevance is not None:
                kwargs["step_min_relevance"] = self._agent_step_min_relevance
            if self._agent_step_max_refine is not None:
                kwargs["step_max_refine"] = self._agent_step_max_refine
            if self._agent_expander_n_variants is not None:
                kwargs["expander_n_variants"] = self._agent_expander_n_variants
            if self._agent_max_stale_steps is not None:
                kwargs["max_stale_steps"] = self._agent_max_stale_steps
            self._agent_engines[kb_id] = AgenticQAEngine(
                controller=self._agent_controller,
                retriever=self._kbs[kb_id].retriever,
                reader=self._reader,
                query_processor=self._query_processor,
                **kwargs,
            )
        return self._agent_engines[kb_id]

    def _resolve_kb(
        self, kb_id: str | None, *, filter: RetrievalFilter | None = None
    ) -> KnowledgeBase:
        """Resolve the target KB: explicit id -> filter.kb_id -> active."""
        target = kb_id
        if target is None and filter is not None and filter.kb_id is not None:
            target = filter.kb_id
        if target is None:
            target = self._active_kb_id
        if target not in self._kbs:
            raise KeyError(f"knowledge base not found: {target}")
        return self._kbs[target]

    def create_knowledge_base(
        self,
        name: str,
        *,
        description: str | None = None,
        language: str = "zh",
        tags: list[str] | None = None,
        kb_id: str | None = None,
        set_active: bool = False,
    ) -> KnowledgeBaseInfo:
        """Create + register a new knowledge base aggregate.

        Returns the serializable :class:`KnowledgeBaseInfo` metadata. When
        ``set_active`` is true (or this is the first KB) the new KB becomes the
        active routing target.
        """
        info = KnowledgeBaseInfo(
            name=name,
            description=description,
            language=language,
            tags=list(tags) if tags else [],
            **({"kb_id": kb_id} if kb_id else {}),
        )
        kb = KnowledgeBase(
            info=info,
            embedder=self._embedder,
            document_store=self._service.document_store,
            reranker=self._reranker,
            state_store=self._state_store,
        )
        self._register_kb(kb)
        if set_active or len(self._kbs) == 1:
            self._active_kb_id = kb.kb_id
        _logger.info("created knowledge base %s (%s)", kb.kb_id, name)
        return kb.info

    def list_knowledge_bases(
        self, *, limit: int = 100, offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        """Return ``(page, total)`` of knowledge bases with live counts.

        Enriches the persisted :class:`KnowledgeBaseInfo` metadata with the
        live block / document counts from the aggregate registry so the
        listing reflects current state, not just the last persistence.
        """
        infos = self._kb_store.list(limit=limit, offset=offset)
        total = len(self._kb_store)
        page: list[dict[str, Any]] = []
        for info in infos:
            kb = self._kbs.get(info.kb_id)
            page.append(
                {
                    "kb_id": info.kb_id,
                    "name": info.name,
                    "description": info.description,
                    "language": info.language,
                    "tags": list(info.tags),
                    "block_count": kb.block_count() if kb is not None else 0,
                    "document_count": kb.document_count() if kb is not None else 0,
                    "created_at": info.created_at,
                    "updated_at": info.updated_at,
                    "active": info.kb_id == self._active_kb_id,
                }
            )
        return page, total

    def get_knowledge_base_info(self, kb_id: str | None = None) -> dict[str, Any] | None:
        """Return a single KB snapshot (live counts + metadata), or ``None``."""
        if kb_id is None:
            kb_id = self._active_kb_id
        info = self._kb_store.get(str(kb_id))
        if info is None:
            return None
        kb = self._kbs.get(info.kb_id)
        return {
            "kb_id": info.kb_id,
            "name": info.name,
            "block_count": kb.block_count() if kb is not None else 0,
            "document_count": kb.document_count() if kb is not None else 0,
            "language": info.language,
            "description": info.description,
            "tags": list(info.tags),
            "active": info.kb_id == self._active_kb_id,
            "created_at": info.created_at,
            "updated_at": info.updated_at,
        }

    def delete_knowledge_base(self, kb_id: str) -> bool:
        """Delete a knowledge base (metadata + live aggregate + engine).

        The last remaining KB cannot be deleted (there must always be one
        active target); raises :class:`ValueError` in that case.
        """
        kb_id = str(kb_id)
        if kb_id not in self._kbs:
            return False
        if len(self._kbs) == 1:
            raise ValueError("cannot delete the last knowledge base")
        del self._kbs[kb_id]
        self._engines.pop(kb_id, None)
        deleted = self._kb_store.delete(kb_id)
        if self._state_store is not None:
            self._state_store.clear(kb_id)
        if self._active_kb_id == kb_id:
            self._active_kb_id = next(iter(self._kbs))
        return deleted

    def set_active_knowledge_base(self, kb_id: str) -> None:
        """Set the default routing target to ``kb_id``."""
        kb_id = str(kb_id)
        if kb_id not in self._kbs:
            raise KeyError(f"knowledge base not found: {kb_id}")
        self._active_kb_id = kb_id

    # ------------------------------------------------------------------ #
    # ingest: bytes -> IdeaBlocks -> indexed in the knowledge base
    # ------------------------------------------------------------------ #
    def ingest_and_index(
        self,
        data: bytes | str,
        filename: str | None = None,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
        auto_tag: bool = True,
        clean: bool = True,
        summarize: bool = True,
        max_summary_sentences: int = 3,
        top_k: int = 8,
        max_blocks: int | None = None,
        language: str | None = None,
        kb_id: str | None = None,
        external_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_system: str | None = None,
        source_extra: dict[str, Any] | None = None,
        on_progress: ProgressCallback | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> IngestResult:
        """Convert -> clean -> generate IdeaBlocks -> embed + index -> store doc.

        The end-to-end ingest that wires the *generation* half (blocks) to the
        *retrieval* half (vector + lexical index). After this call the new
        knowledge is immediately retrievable via :meth:`ask`.

        Parameters
        ----------
        data, filename, clean:
            See :meth:`SparkSageService.convert`. ``clean`` defaults to ``True``.
        title:
            Explicit title override. Falls back to the converter-extracted title.
        tags:
            Caller-supplied tags. When empty and ``auto_tag`` is ``True``, tags
            are derived from the content via keyword extraction.
        max_blocks, language:
            Forwarded to the :class:`IdeaBlockGenerator`.
        kb_id:
            Target knowledge base. Defaults to the active KB.
        external_key:
            Deterministic external id (e.g. ``"wiki:123"``) stamped on the
            :class:`DocumentRecord` so a later :meth:`upsert_document` can
            address this document without knowing its opaque ``doc_id``.
        metadata:
            Free-form caller metadata attached to the document record (in
            addition to anything the pipeline derives).
        source_system, source_extra:
            Provenance enrichment carried into the stored ``SourceRef``
            (``system`` / ``extra``) so wiki/CRM-style sources can link
            citations back to their originating page.
        on_progress:
            Optional phase-progress callback (called with ``"converting"`` /
            ``"generating"`` / ``"indexing"`` before each phase). Used by
            :meth:`submit_ingest` so the async job's snapshot reflects live
            progress.
        is_cancelled:
            Optional cooperative-cancellation predicate. When it returns
            ``True`` at a phase boundary the ingest aborts with
            :class:`~sparksage.api.ingest_jobs.IngestCancelled` *before* the
            knowledge-base write -- so a cancelled async ingest leaves the KB
            untouched.

        Raises
        ------
        GenerationNotConfiguredError:
            If the underlying service has no generator wired (no LLM key).
        KeyError:
            If ``kb_id`` does not match a registered knowledge base.
        IngestCancelled:
            If ``is_cancelled`` fires at a phase boundary.
        """

        def _check_cancelled() -> None:
            if is_cancelled is not None and is_cancelled():
                raise IngestCancelled(
                    "ingest cancelled before the next phase (no data written)"
                )

        if not self._service.has_generator:
            raise GenerationNotConfiguredError(
                "no IdeaBlockGenerator configured; cannot ingest-and-index."
            )

        kb = self._resolve_kb(kb_id)
        _logger.debug(
            "ingest_and_index start: filename=%s kb=%s clean=%s max_blocks=%s lang=%s",
            filename,
            kb.kb_id,
            clean,
            max_blocks,
            language,
        )
        t_total = time.perf_counter()

        _check_cancelled()
        if on_progress is not None:
            on_progress("converting")
        t0 = time.perf_counter()
        conv = self._service.convert(data, filename, clean=clean)
        elapsed_conv = time.perf_counter() - t0
        text = conv.markdown
        source = conv.source
        resolved_title = title if title is not None else conv.title
        _logger.debug(
            "ingest convert done: markdown_len=%d title=%s elapsed=%.2fs",
            len(text),
            resolved_title,
            elapsed_conv,
        )

        gen = self._service.generator
        assert gen is not None

        final_tags = list(tags) if tags else []
        need_tags = not final_tags and auto_tag

        _check_cancelled()
        if on_progress is not None:
            on_progress("generating")
        t_parallel = time.perf_counter()
        blocks, final_tags, summary = self._parallel_generate(
            text,
            source,
            final_tags=final_tags,
            need_tags=need_tags,
            top_k=top_k,
            summarize=summarize,
            max_summary_sentences=max_summary_sentences,
            max_blocks=max_blocks,
            language=language,
        )
        elapsed_parallel = time.perf_counter() - t_parallel
        _logger.debug(
            "ingest generate/tag/summarize done: %d blocks, %d tags, "
            "%d summary chars elapsed=%.2fs",
            len(blocks),
            len(final_tags),
            len(summary or ""),
            elapsed_parallel,
        )

        record = new_record(
            title=resolved_title,
            summary=summary,
            body_markdown=text,
            tags=final_tags,
            source=SourceRef(
                uri=source.uri,
                title=resolved_title,
                system=source_system or source.system,
                extra=source_extra if source_extra is not None else dict(source.extra),
            ),
            metadata=metadata,
            external_key=external_key,
        )

        _check_cancelled()
        if on_progress is not None:
            on_progress("indexing")
        t0 = time.perf_counter()
        stored = kb.add_document(record, blocks=blocks)
        elapsed_index = time.perf_counter() - t0
        elapsed_total = time.perf_counter() - t_total
        _logger.info(
            "ingested %s: %d blocks indexed (kb=%s, doc=%s) "
            "elapsed=%.2fs (convert=%.2fs parallel=%.2fs index=%.2fs)",
            filename or source.uri,
            len(blocks),
            kb.kb_id,
            stored.doc_id,
            elapsed_total,
            elapsed_conv,
            elapsed_parallel,
            elapsed_index,
        )
        return IngestResult(
            doc_id=stored.doc_id,
            blocks=blocks,
            block_count=len(blocks),
            title=resolved_title,
            source=source,
            tags=list(final_tags),
            summary=summary,
        )

    def update_document_and_reindex(
        self,
        doc_id: str,
        data: bytes | str,
        filename: str | None = None,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
        auto_tag: bool = True,
        clean: bool = True,
        summarize: bool = True,
        max_summary_sentences: int = 3,
        top_k: int = 8,
        max_blocks: int | None = None,
        language: str | None = None,
        kb_id: str | None = None,
        external_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_system: str | None = None,
        source_extra: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Replace a document's content in a KB, re-indexing hash-aware.

        The knowledge-base content-update counterpart of :meth:`ingest_and_index`:
        convert -> (re)generate IdeaBlocks -> replace the document's blocks in the
        index while keeping ``doc_id`` stable. The write is hash-aware, mirroring
        :meth:`KnowledgeBase.update_document`: when the new body's ``content_hash``
        equals the stored one, generation + re-embedding are skipped entirely
        (only title/tags are patched), so re-uploading the same file is cheap.
        When the body changed, the old linked blocks are cascade-removed and the
        new ones indexed -- the consistency guarantee :meth:`remove_document`
        provides, without the delete-then-recreate dance.

        Returns an :class:`IngestResult` carrying the refreshed blocks (the
        existing live blocks on an unchanged body).

        Raises
        ------
        GenerationNotConfiguredError:
            If the body changed and the service has no generator wired (no LLM key).
        KeyError:
            If ``doc_id`` is not a document in the target KB.
        """
        kb = self._resolve_kb(kb_id)
        if not kb.contains_document(doc_id):
            raise KeyError(f"document not found: {doc_id}")
        existing = kb.document_store.get(doc_id)

        conv = self._service.convert(data, filename, clean=clean)
        text = conv.markdown
        source = conv.source
        body_changed = content_hash_of(text) != existing.content_hash
        resolved_title = title if title is not None else conv.title or existing.title

        if not body_changed:
            changes: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
            if title is not None:
                changes["title"] = title
            if tags is not None:
                changes["tags"] = list(tags)
            if metadata is not None:
                changes["metadata"] = dict(metadata)
            if external_key is not None:
                changes["external_key"] = external_key
            record = existing.model_copy(update=changes)
            stored = kb.update_document(doc_id, record=record, blocks=None)
            blocks = kb.blocks_for_document(doc_id)
            _logger.info(
                "updated document %s (unchanged body, metadata-only; kb=%s)",
                doc_id,
                kb.kb_id,
            )
            return IngestResult(
                doc_id=stored.doc_id,
                blocks=blocks,
                block_count=len(blocks),
                title=stored.title,
                source=stored.source,
                tags=list(stored.tags),
                summary=stored.summary,
                action="unchanged",
            )

        if not self._service.has_generator:
            raise GenerationNotConfiguredError(
                "no IdeaBlockGenerator configured; cannot re-generate blocks on update."
            )

        final_tags = list(tags) if tags else []
        need_tags = not final_tags and auto_tag
        blocks, final_tags, summary = self._parallel_generate(
            text,
            source,
            final_tags=final_tags,
            need_tags=need_tags,
            top_k=top_k,
            summarize=summarize,
            max_summary_sentences=max_summary_sentences,
            max_blocks=max_blocks,
            language=language,
        )
        record = new_record(
            doc_id=doc_id,
            title=resolved_title,
            summary=summary,
            body_markdown=text,
            tags=final_tags,
            source=SourceRef(
                uri=source.uri,
                title=resolved_title,
                system=source_system or source.system or existing.source.system,
                extra=(
                    source_extra
                    if source_extra is not None
                    else dict(existing.source.extra)
                ),
            ),
            metadata=metadata if metadata is not None else dict(existing.metadata),
            external_key=external_key if external_key is not None else existing.external_key,
        )
        stored = kb.update_document(doc_id, record=record, blocks=blocks)
        _logger.info(
            "updated document %s: %d blocks re-indexed (kb=%s)",
            doc_id,
            len(blocks),
            kb.kb_id,
        )
        return IngestResult(
            doc_id=stored.doc_id,
            blocks=blocks,
            block_count=len(blocks),
            title=resolved_title,
            source=source,
            tags=list(final_tags),
            summary=summary,
            action="updated",
        )

    def _parallel_generate(
        self,
        text: str,
        source: SourceRef,
        *,
        final_tags: list[str],
        need_tags: bool,
        top_k: int,
        summarize: bool,
        max_summary_sentences: int,
        max_blocks: int | None,
        language: str | None,
    ) -> tuple[list[IdeaBlock], list[str], str | None]:
        """Run block generation + auto-tagging + summarization in parallel."""
        gen = self._service.generator
        assert gen is not None
        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_blocks = pool.submit(
                gen.generate,
                text,
                source=source,
                max_blocks=max_blocks,
                language=language,
            )
            fut_tags = (
                pool.submit(self._service.auto_tag, text, top_k=top_k)
                if need_tags
                else None
            )
            fut_summary = (
                pool.submit(
                    self._service.summarize_text,
                    text,
                    max_sentences=max_summary_sentences,
                )
                if summarize
                else None
            )
            blocks = fut_blocks.result()
            if fut_tags is not None:
                final_tags = fut_tags.result()
            summary = fut_summary.result() if fut_summary is not None else None
        return blocks, final_tags, summary

    def submit_ingest(
        self,
        data: bytes | str,
        filename: str | None = None,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
        auto_tag: bool = True,
        clean: bool = True,
        summarize: bool = True,
        max_summary_sentences: int = 3,
        top_k: int = 8,
        max_blocks: int | None = None,
        language: str | None = None,
        kb_id: str | None = None,
        job_id: str | None = None,
    ) -> IngestJob:
        """Submit an async ingest job and return it immediately.

        This is the non-blocking counterpart of :meth:`ingest_and_index` for
        the "5-file upload, one timed out" failure mode: the HTTP request
        returns a job id at once (no axios timeout to race) and the heavy
        convert -> generate -> embed -> index work runs in a background
        thread. Poll ``job.snapshot()`` (or ``GET /api/v1/jobs/{job_id}``) for
        progress / completion.

        Validation (generator configured, ``kb_id`` resolvable) runs eagerly
        so a misconfigured request fails fast rather than producing a job that
        immediately errors. All ingest options mirror :meth:`ingest_and_index`.

        Cancellation is cooperative: ``job.cancel()`` flips the predicate the
        work polls at phase boundaries; a cancel that lands before the
        knowledge-base write aborts the ingest cleanly (no partial doc /
        blocks indexed), exactly inverting the old "client gone but server
        still wrote" bug.
        """
        if not self._service.has_generator:
            raise GenerationNotConfiguredError(
                "no IdeaBlockGenerator configured; cannot ingest-and-index."
            )
        # Validate kb_id eagerly so a bad target fails fast (KeyError) rather
        # than producing a job that errors on the worker thread.
        kb = self._resolve_kb(kb_id)
        target_kb_id = kb.kb_id

        kwargs = dict(
            title=title,
            tags=list(tags) if tags else None,
            auto_tag=auto_tag,
            clean=clean,
            summarize=summarize,
            max_summary_sentences=max_summary_sentences,
            top_k=top_k,
            max_blocks=max_blocks,
            language=language,
            kb_id=target_kb_id,
        )

        def _work(on_progress, is_cancelled):
            return self.ingest_and_index(
                data,
                filename,
                on_progress=on_progress,
                is_cancelled=is_cancelled,
                **kwargs,
            )

        return self._ingest_jobs.submit(
            _work, job_id=job_id, filename=filename
        )

    def remove_document(self, doc_id: str, *, kb_id: str | None = None) -> bool:
        """Remove a document *and* cascade-remove its indexed blocks."""
        kb = self._resolve_kb(kb_id)
        return kb.remove_document(doc_id)

    def find_by_external_key(
        self, external_key: str, *, kb_id: str | None = None
    ) -> DocumentRecord | None:
        """Return the :class:`DocumentRecord` owned by ``kb_id`` with ``external_key``.

        ``None`` when no owned document carries that external id. Scoped to the
        KB's own ``doc_id``s because the underlying document store may be shared
        across knowledge bases.
        """
        kb = self._resolve_kb(kb_id)
        for doc_id in kb.document_ids():
            record = kb.document_store.get(doc_id)
            if record is not None and record.external_key == external_key:
                return record
        return None

    def upsert_document(
        self,
        data: bytes | str,
        filename: str | None = None,
        *,
        external_key: str,
        title: str | None = None,
        tags: list[str] | None = None,
        auto_tag: bool = True,
        clean: bool = True,
        summarize: bool = True,
        max_summary_sentences: int = 3,
        top_k: int = 8,
        max_blocks: int | None = None,
        language: str | None = None,
        kb_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_system: str | None = None,
        source_extra: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Idempotent upsert keyed by ``external_key`` (the wiki-sync primitive).

        The external-id counterpart of :meth:`ingest_and_index` /
        :meth:`update_document_and_reindex` that makes a full wiki sync
        naturally incremental:

        * no document with ``external_key`` in the KB -> ingest (create),
        * a document with the *same* ``content_hash`` -> metadata-only patch,
          ``action="unchanged"`` (zero LLM cost),
        * a document whose body changed -> :meth:`update_document_and_reindex`
          (cascade re-index, ``doc_id`` kept stable), ``action="updated"``.

        ``external_key`` is required and deterministic (e.g. ``"wiki:123"``);
        pass it to the shared-document-store upserts so re-syncs can never
        duplicate a page. The returned :class:`IngestResult.action` reports
        which branch ran.

        Raises
        ------
        ValueError:
            If ``external_key`` is empty / ``None``.
        GenerationNotConfiguredError:
            If the body changed and the service has no generator wired.
        KeyError:
            If ``kb_id`` does not match a registered knowledge base.
        """
        if not external_key or not str(external_key).strip():
            raise ValueError("external_key is required for an idempotent upsert")
        kb = self._resolve_kb(kb_id)
        target_kb_id = kb.kb_id

        existing = self.find_by_external_key(external_key, kb_id=target_kb_id)
        if existing is None:
            result = self.ingest_and_index(
                data,
                filename,
                title=title,
                tags=tags,
                auto_tag=auto_tag,
                clean=clean,
                summarize=summarize,
                max_summary_sentences=max_summary_sentences,
                top_k=top_k,
                max_blocks=max_blocks,
                language=language,
                kb_id=target_kb_id,
                external_key=external_key,
                metadata=metadata,
                source_system=source_system,
                source_extra=source_extra,
            )
            result.action = "created"
            _logger.info(
                "upserted (created) external_key=%s -> doc=%s kb=%s",
                external_key,
                result.doc_id,
                target_kb_id,
            )
            return result

        conv = self._service.convert(data, filename, clean=clean)
        body_changed = content_hash_of(conv.markdown) != existing.content_hash
        if not body_changed:
            changes: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
            if title is not None:
                changes["title"] = title
            if tags is not None:
                changes["tags"] = list(tags)
            if metadata is not None:
                changes["metadata"] = dict(metadata)
            stored = kb.update_document(
                existing.doc_id,
                record=existing.model_copy(update=changes),
                blocks=None,
            )
            blocks = kb.blocks_for_document(existing.doc_id)
            _logger.info(
                "upserted (unchanged) external_key=%s -> doc=%s kb=%s",
                external_key,
                existing.doc_id,
                target_kb_id,
            )
            return IngestResult(
                doc_id=stored.doc_id,
                blocks=blocks,
                block_count=len(blocks),
                title=stored.title,
                source=stored.source,
                tags=list(stored.tags),
                summary=stored.summary,
                action="unchanged",
            )

        result = self.update_document_and_reindex(
            existing.doc_id,
            data,
            filename,
            title=title,
            tags=tags,
            auto_tag=auto_tag,
            clean=clean,
            summarize=summarize,
            max_summary_sentences=max_summary_sentences,
            top_k=top_k,
            max_blocks=max_blocks,
            language=language,
            kb_id=target_kb_id,
            external_key=external_key,
            metadata=metadata,
            source_system=source_system,
            source_extra=source_extra,
        )
        _logger.info(
            "upserted (updated) external_key=%s -> doc=%s kb=%s",
            external_key,
            result.doc_id,
            target_kb_id,
        )
        return result

    def list_documents(
        self,
        *,
        kb_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[DocumentRecord], int]:
        """Return a paginated slice of a knowledge base's own documents.

        Scoped to the target KB's ``doc_id``s (the document store may be
        shared across KBs). Returns ``(page, total)``; ``total`` is the KB's
        document count, not the whole shared store's. Powers the wiki-sync
        deletion-detection step: list ``external_key``s and diff against the
        latest page snapshot.
        """
        kb = self._resolve_kb(kb_id)
        doc_ids = sorted(kb.document_ids())
        total = len(doc_ids)
        page_ids = doc_ids[offset : offset + limit]
        page: list[DocumentRecord] = []
        for doc_id in page_ids:
            record = kb.document_store.get(doc_id)
            if record is not None:
                page.append(record)
        return page, total

    # ------------------------------------------------------------------ #
    # query: ask the knowledge base
    # ------------------------------------------------------------------ #
    def ask(
        self,
        query: str,
        *,
        context: ConversationContext | None = None,
        filter: RetrievalFilter | None = None,
        k: int | None = None,
        use_lexical: bool | None = None,
        use_rerank: bool | None = None,
        use_cache: bool = True,
        kb_id: str | None = None,
        mode: str = "default",
        on_progress: Callable[[Any], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> QAResult | AgentResult:
        """Answer ``query`` end-to-end.

        ``mode`` selects the QA strategy:

        * ``"default"`` (default): the single-shot :class:`QAEngine`
          (query -> retrieval -> answer). Fast, one retrieval.
        * ``"agent"``: the agentic :class:`AgenticQAEngine` -- an LLM-driven
          plan-act-observe-synthesize loop that decomposes multi-hop /
          comparative questions into a bounded sequence of focused retrievals
          (slower, but handles the problem classes one-shot RAG cannot). Requires
          an ``agent_controller`` to be wired on this service.

        Retrieval is scoped to the target knowledge base. The target resolves
        as: explicit ``kb_id`` -> ``filter.kb_id`` -> the active KB. Pass a
        :class:`RetrievalFilter` for tag / entity / language scoping; pass a
        :class:`ConversationContext` for multi-turn anaphora resolution.

        ``on_progress`` and ``is_cancelled`` only affect ``mode="agent"`` (the
        single-shot mode is synchronous); the SSE streaming route forwards them
        so a long agent run emits per-phase progress the client can render.
        """
        kb = self._resolve_kb(kb_id, filter=filter)
        if mode == "agent":
            return self._ask_agent(
                kb.kb_id,
                query,
                context=context,
                filter=filter,
                k=k,
                use_lexical=use_lexical,
                use_rerank=use_rerank,
                on_progress=on_progress,
                is_cancelled=is_cancelled,
            )
        return self._ask_default(
            kb.kb_id,
            query,
            context=context,
            filter=filter,
            k=k,
            use_lexical=use_lexical,
            use_rerank=use_rerank,
            use_cache=use_cache,
        )

    def _ask_default(
        self,
        kb_id: str,
        query: str,
        *,
        context,
        filter,
        k,
        use_lexical,
        use_rerank,
        use_cache,
    ) -> QAResult:
        engine = self._engine_for(kb_id)
        _logger.debug(
            "ask[default]: query=%r kb=%s k=%s lexical=%s rerank=%s",
            query[:80],
            kb_id,
            k,
            use_lexical,
            use_rerank,
        )
        t0 = time.perf_counter()
        result = engine.ask(
            query,
            context=context,
            filter=filter,
            k=k,
            use_lexical=use_lexical,
            use_rerank=use_rerank,
            use_cache=use_cache,
        )
        elapsed = time.perf_counter() - t0
        n_chunks = len(result.retrieval.chunks) if result.retrieval else 0
        intent_str = (
            result.query_result.intent.intent.value
            if result.query_result is not None
            else "n/a"
        )
        top_score = (
            result.retrieval.chunks[0].score
            if result.retrieval and result.retrieval.chunks
            else None
        )
        _logger.info(
            "answered %r: abstained=%s chunks=%d cached=%s intent=%s "
            "top_score=%s elapsed=%.2fs",
            query[:80],
            result.abstained,
            n_chunks,
            result.cached,
            intent_str,
            f"{top_score:.3f}" if top_score is not None else None,
            elapsed,
        )
        self._record_history(query, result, kb_id)
        return result

    def _ask_agent(
        self,
        kb_id: str,
        query: str,
        *,
        context,
        filter,
        k,
        use_lexical,
        use_rerank,
        on_progress=None,
        is_cancelled=None,
    ) -> AgentResult:
        engine = self._agent_engine_for(kb_id)
        _logger.debug(
            "ask[agent]: query=%r kb=%s k=%s lexical=%s rerank=%s",
            query[:80],
            kb_id,
            k,
            use_lexical,
            use_rerank,
        )
        t0 = time.perf_counter()
        result = engine.ask(
            query,
            context=context,
            filter=filter,
            k=k,
            use_lexical=use_lexical,
            use_rerank=use_rerank,
            on_progress=on_progress,
            is_cancelled=is_cancelled,
        )
        elapsed = time.perf_counter() - t0
        _logger.info(
            "answered[agent] %r: abstained=%s iterations=%d steps=%d evidence=%d "
            "aborted=%s elapsed=%.2fs",
            query[:80],
            result.abstained,
            result.iterations,
            len(result.steps),
            len(result.evidence),
            result.aborted,
            elapsed,
        )
        self._record_history(query, result, kb_id)
        return result

    def _record_history(
        self, query: str, result: QAResult | AgentResult, kb_id: str
    ) -> None:
        """Append the user question + assistant answer to the conversation log.

        The answer payload is the canonical HTTP ``AskResponse`` shape (via
        :func:`~sparksage.api.schemas._to_ask_response`) so the Q&A page can
        re-render citations / retrieved chunks / confidence on reload without
        re-running the pipeline.
        """
        from sparksage.api.schemas import _to_ask_response

        payload = _to_ask_response(result).model_dump(mode="json")
        try:
            self._history_store.add_turn(
                QATurn(role=TurnRole.USER, content=query, kb_id=kb_id)
            )
            self._history_store.add_turn(
                QATurn(
                    role=TurnRole.ASSISTANT,
                    content=result.text or "",
                    kb_id=kb_id,
                    result=payload,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            _logger.warning("history recording failed: %s", exc)

    # ------------------------------------------------------------------ #
    # knowledge-base stats
    # ------------------------------------------------------------------ #
    def knowledge_base_info(self, kb_id: str | None = None) -> dict[str, Any]:
        """Snapshot of a knowledge base: block / document counts + metadata.

        Defaults to the active KB when ``kb_id`` is omitted. Returns a shape
        compatible with :class:`~sparksage.api.schemas.KnowledgeBaseResponse`
        (no timestamps).
        """
        target = kb_id if kb_id is not None else self._active_kb_id
        kb = self._kbs.get(str(target))
        if kb is None:
            raise KeyError(f"knowledge base not found: {target}")
        info = kb.info
        return {
            "kb_id": kb.kb_id,
            "name": kb.name,
            "block_count": kb.block_count(),
            "document_count": kb.document_count(),
            "language": info.language,
            "description": info.description,
            "tags": list(info.tags),
            "active": kb.kb_id == self._active_kb_id,
        }

    # ------------------------------------------------------------------ #
    # feedback: the quality flywheel
    # ------------------------------------------------------------------ #
    def add_feedback(
        self,
        query: str,
        answer_text: str,
        rating: FeedbackRating | str,
        *,
        correction: str | None = None,
        block_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        kb_id: str | None = None,
    ) -> FeedbackRecord:
        """Record a user verdict on a surfaced answer.

        ``rating`` accepts a :class:`FeedbackRating` enum or its string value
        (``"positive"`` / ``"negative"`` / ``"corrected"``) for HTTP-friendliness.
        ``kb_id`` defaults to the active KB.
        """
        if isinstance(rating, str):
            rating = FeedbackRating(rating)
        target_kb_id = self._resolve_kb(kb_id).kb_id
        record = FeedbackRecord(
            query=query,
            answer_text=answer_text,
            rating=rating,
            correction=correction,
            block_ids=list(block_ids) if block_ids else [],
            kb_id=target_kb_id,
            metadata=dict(metadata) if metadata else {},
        )
        return self._feedback_store.add(record)

    def feedback_stats(self, kb_id: str | None = None) -> FeedbackStats:
        """Aggregate :class:`FeedbackStats` over the feedback store."""
        target_kb_id = self._resolve_kb(kb_id).kb_id
        if hasattr(self._feedback_store, "stats"):
            return self._feedback_store.stats(kb_id=target_kb_id)
        return FeedbackStats()

    def list_blocks(
        self,
        *,
        tags: list[str] | None = None,
        language: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
        kb_id: str | None = None,
    ) -> tuple[list[IdeaBlock], int]:
        """Return a filtered, paginated slice of a knowledge base's blocks.

        ``tags`` is an any-match OR over the block's coarse ``Tag`` values
        (unknown tag strings are ignored). ``status`` filters by lifecycle
        status value (``ACTIVE`` / ``MERGED`` / ...). Returns ``(page, total)``
        so the caller can render pagination. ``total`` is computed against the
        *filter*, not the whole registry.
        """
        from sparksage.schema.enums import BlockStatus, Tag

        kb = self._resolve_kb(kb_id)
        blocks = kb.blocks()
        norm_tags = set()
        if tags:
            for raw in tags:
                try:
                    norm_tags.add(Tag(raw))
                except ValueError:
                    continue

        status_enum: BlockStatus | None = None
        if status:
            try:
                status_enum = BlockStatus(status)
            except ValueError:
                status_enum = None

        def keep(b: IdeaBlock) -> bool:
            if norm_tags and not (set(b.tags) & norm_tags):
                return False
            if language and (b.language or "en") != language:
                return False
            if status_enum is not None and b.status != status_enum:
                return False
            return True

        filtered = [b for b in blocks if keep(b)]
        total = len(filtered)
        if limit < 1:
            limit = 1
        if offset < 0:
            offset = 0
        page = filtered[offset : offset + limit]
        return page, total

    def list_block_tags(self, kb_id: str | None = None) -> list[str]:
        """Return the distinct coarse ``Tag`` vocabulary across a KB's blocks.

        Sorted ascending. Powers the knowledge-base tag filter dropdown so its
        options do not shrink to the current page of blocks.
        """
        kb = self._resolve_kb(kb_id)
        seen: set[str] = set()
        for b in kb.blocks():
            for t in getattr(b, "tags", []) or []:
                seen.add(t.value if hasattr(t, "value") else str(t))
        return sorted(seen)

    def list_feedback(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        kb_id: str | None = None,
    ) -> tuple[list[FeedbackRecord], int]:
        """Return a newest-first paginated slice of feedback records.

        Scoped to the target knowledge base. Returns ``(page, total)``.
        """
        target_kb_id = self._resolve_kb(kb_id).kb_id
        total = self._feedback_store.count(kb_id=target_kb_id)
        if limit < 1:
            limit = 1
        if offset < 0:
            offset = 0
        page = self._feedback_store.list(
            kb_id=target_kb_id, limit=limit, offset=offset
        )
        return page, total

    # ------------------------------------------------------------------ #
    # conversation history: the persisted query log the Q&A page restores
    # ------------------------------------------------------------------ #
    def list_history(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        kb_id: str | None = None,
    ) -> tuple[list[QATurn], int]:
        """Return a newest-first paginated slice of the conversation log.

        Scoped to the target knowledge base (the active KB by default), exactly
        like :meth:`list_feedback`. Returns ``(page, total)``.
        """
        target_kb_id = self._resolve_kb(kb_id).kb_id
        total = self._history_store.count(kb_id=target_kb_id)
        if limit < 1:
            limit = 1
        if offset < 0:
            offset = 0
        page = self._history_store.list(
            kb_id=target_kb_id, limit=limit, offset=offset
        )
        return page, total

    def clear_history(self, kb_id: str | None = None) -> int:
        """Clear the conversation log (optionally scoped to a KB).

        Returns the number of turns removed. With no ``kb_id`` the whole log is
        wiped (mirrors the single-session demo UI).
        """
        target_kb_id = self._resolve_kb(kb_id).kb_id if kb_id is not None else None
        removed = self._history_store.count(kb_id=target_kb_id)
        self._history_store.clear(kb_id=target_kb_id)
        return removed


__all__ = [
    "IngestCancelled",
    "IngestResult",
    "QAService",
]
