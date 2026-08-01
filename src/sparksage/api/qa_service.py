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
from dataclasses import dataclass
from typing import Any

from sparksage.api.pipeline import (
    GenerationNotConfiguredError,
    SparkSageService,
)
from sparksage.documents.models import new_record
from sparksage.embed.indexer import BlockEmbedder
from sparksage.feedback.models import FeedbackRating, FeedbackRecord
from sparksage.feedback.store import FeedbackStats, FeedbackStore, InMemoryFeedbackStore
from sparksage.kb.knowledge_base import KnowledgeBase
from sparksage.kb.models import KnowledgeBaseInfo
from sparksage.kb.store import InMemoryKnowledgeBaseStore, KnowledgeBaseStore
from sparksage.qa.engine import QAEngine, QAResult
from sparksage.query.context import ConversationContext
from sparksage.query.processor import QueryProcessor
from sparksage.reader.orchestrator import Reader
from sparksage.retrieve.models import RetrievalFilter
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
    kb_store:
        Optional :class:`KnowledgeBaseStore` registry for multi-tenant KB
        metadata. Defaults to an :class:`InMemoryKnowledgeBaseStore` so KB CRUD
        works out of the box.
    feedback_store:
        Optional :class:`FeedbackStore` for the quality flywheel. Defaults to an
        :class:`InMemoryFeedbackStore` so feedback works out of the box.

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
        kb_store: KnowledgeBaseStore | None = None,
        feedback_store: FeedbackStore | None = None,
    ) -> None:
        self._service = service
        self._embedder = embedder
        self._reader = reader
        self._query_processor = query_processor
        self._kb_store: KnowledgeBaseStore = (
            kb_store if kb_store is not None else InMemoryKnowledgeBaseStore()
        )
        # Live KnowledgeBase aggregates keyed by kb_id (runtime index state).
        self._kbs: dict[str, KnowledgeBase] = {}
        # Lazily built QAEngine per KB (shares reader + query_processor).
        self._engines: dict[str, QAEngine] = {}

        default_kb = (
            kb
            if kb is not None
            else KnowledgeBase(
                info=KnowledgeBaseInfo(name="default"),
                embedder=embedder,
            )
        )
        self._register_kb(default_kb)
        self._active_kb_id: str = default_kb.kb_id

        self._feedback_store: FeedbackStore = (
            feedback_store if feedback_store is not None else InMemoryFeedbackStore()
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
    def engine(self) -> QAEngine:
        """The QAEngine bound to the active knowledge base."""
        return self._engine_for(self._active_kb_id)

    @property
    def feedback_store(self) -> FeedbackStore:
        return self._feedback_store

    @property
    def has_generator(self) -> bool:
        """Whether block generation is available (ingest-and-index needs this)."""
        return self._service.has_generator

    # ------------------------------------------------------------------ #
    # multi-knowledge-base management
    # ------------------------------------------------------------------ #
    def _register_kb(self, kb: KnowledgeBase) -> KnowledgeBase:
        """Register a live aggregate + persist its metadata to the store."""
        self._kbs[kb.kb_id] = kb
        self._kb_store.save(kb.info)
        self._engines.pop(kb.kb_id, None)
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
        language: str = "en",
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
        kb = KnowledgeBase(info=info, embedder=self._embedder)
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

        Raises
        ------
        GenerationNotConfiguredError:
            If the underlying service has no generator wired (no LLM key).
        KeyError:
            If ``kb_id`` does not match a registered knowledge base.
        """
        if not self._service.has_generator:
            raise GenerationNotConfiguredError(
                "no IdeaBlockGenerator configured; cannot ingest-and-index."
            )

        kb = self._resolve_kb(kb_id)
        conv = self._service.convert(data, filename, clean=clean)
        text = conv.markdown
        source = conv.source
        resolved_title = title if title is not None else conv.title

        gen = self._service.generator
        assert gen is not None
        blocks = gen.generate(
            text,
            source=source,
            max_blocks=max_blocks,
            language=language,
        )

        final_tags = list(tags) if tags else []
        if not final_tags and auto_tag:
            final_tags = self._service.auto_tag(text, top_k=top_k)

        summary: str | None = None
        if summarize:
            summary = self._service.summarize_text(
                text, max_sentences=max_summary_sentences
            )

        record = new_record(
            title=resolved_title,
            summary=summary,
            body_markdown=text,
            tags=final_tags,
            source=SourceRef(uri=source.uri, title=resolved_title),
        )

        stored = kb.add_document(record, blocks=blocks)
        _logger.info(
            "ingested %s: %d blocks indexed (kb=%s, doc=%s)",
            filename or source.uri,
            len(blocks),
            kb.kb_id,
            stored.doc_id,
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

    def remove_document(self, doc_id: str, *, kb_id: str | None = None) -> bool:
        """Remove a document *and* cascade-remove its indexed blocks."""
        kb = self._resolve_kb(kb_id)
        return kb.remove_document(doc_id)

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
    ) -> QAResult:
        """Answer ``query`` end-to-end (delegates to :class:`QAEngine`).

        Retrieval is scoped to the target knowledge base. The target resolves
        as: explicit ``kb_id`` -> ``filter.kb_id`` -> the active KB. Pass a
        :class:`RetrievalFilter` for tag / entity / language scoping; pass a
        :class:`ConversationContext` for multi-turn anaphora resolution.
        """
        kb = self._resolve_kb(kb_id, filter=filter)
        engine = self._engine_for(kb.kb_id)
        return engine.ask(
            query,
            context=context,
            filter=filter,
            k=k,
            use_lexical=use_lexical,
            use_rerank=use_rerank,
            use_cache=use_cache,
        )

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


__all__ = [
    "IngestResult",
    "QAService",
]
