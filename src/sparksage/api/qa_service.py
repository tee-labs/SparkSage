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
        The :class:`KnowledgeBase` aggregate that owns documents + blocks +
        the dense + lexical indexes. Defaults to a fresh ``"default"`` KB.
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
        feedback_store: FeedbackStore | None = None,
    ) -> None:
        self._service = service
        self._embedder = embedder
        self._reader = reader
        self._query_processor = query_processor
        self._kb: KnowledgeBase = (
            kb
            if kb is not None
            else KnowledgeBase(
                info=KnowledgeBaseInfo(name="default"),
                embedder=embedder,
            )
        )
        self._feedback_store: FeedbackStore = (
            feedback_store if feedback_store is not None else InMemoryFeedbackStore()
        )
        self._engine = QAEngine(
            retriever=self._kb.retriever,
            reader=reader,
            query_processor=query_processor,
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
        return self._kb

    @property
    def engine(self) -> QAEngine:
        return self._engine

    @property
    def feedback_store(self) -> FeedbackStore:
        return self._feedback_store

    @property
    def has_generator(self) -> bool:
        """Whether block generation is available (ingest-and-index needs it)."""
        return self._service.has_generator

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

        Raises
        ------
        GenerationNotConfiguredError:
            If the underlying service has no generator wired (no LLM key).
        """
        if not self._service.has_generator:
            raise GenerationNotConfiguredError(
                "no IdeaBlockGenerator configured; cannot ingest-and-index."
            )

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

        stored = self._kb.add_document(record, blocks=blocks)
        _logger.info(
            "ingested %s: %d blocks indexed (kb=%s, doc=%s)",
            filename or source.uri,
            len(blocks),
            self._kb.kb_id,
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

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document *and* cascade-remove its indexed blocks."""
        return self._kb.remove_document(doc_id)

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
    ) -> QAResult:
        """Answer ``query`` end-to-end (delegates to :class:`QAEngine`).

        Retrieval is scoped to this service's :class:`KnowledgeBase`. Pass a
        :class:`RetrievalFilter` for tag / entity / language scoping; pass a
        :class:`ConversationContext` for multi-turn anaphora resolution.
        """
        return self._engine.ask(
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
    def knowledge_base_info(self) -> dict[str, Any]:
        """Snapshot of the knowledge base: block / document counts + metadata."""
        kb = self._kb
        info = kb.info
        return {
            "kb_id": kb.kb_id,
            "name": kb.name,
            "block_count": kb.block_count(),
            "document_count": kb.document_count(),
            "language": info.language,
            "description": info.description,
            "tags": list(info.tags),
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
    ) -> FeedbackRecord:
        """Record a user verdict on a surfaced answer.

        ``rating`` accepts a :class:`FeedbackRating` enum or its string value
        (``"positive"`` / ``"negative"`` / ``"corrected"``) for HTTP-friendliness.
        """
        if isinstance(rating, str):
            rating = FeedbackRating(rating)
        record = FeedbackRecord(
            query=query,
            answer_text=answer_text,
            rating=rating,
            correction=correction,
            block_ids=list(block_ids) if block_ids else [],
            kb_id=self._kb.kb_id,
            metadata=dict(metadata) if metadata else {},
        )
        return self._feedback_store.add(record)

    def feedback_stats(self) -> FeedbackStats:
        """Aggregate :class:`FeedbackStats` over the feedback store."""
        if hasattr(self._feedback_store, "stats"):
            return self._feedback_store.stats(kb_id=self._kb.kb_id)
        return FeedbackStats()


__all__ = [
    "IngestResult",
    "QAService",
]
