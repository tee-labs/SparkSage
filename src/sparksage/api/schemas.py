"""HTTP request/response models for the SparkSage API.

These are plain Pydantic v2 models -- no FastAPI import -- so they are
reusable outside the web layer (e.g. by a CLI or test harness) and stay
consistent with the project's ``ConfigDict(extra="forbid")`` convention.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from sparksage.api.pipeline import GenerateOutput

if TYPE_CHECKING:
    from sparksage.query.context import ConversationContext
    from sparksage.retrieve.models import RetrievalFilter


class SourceInfo(BaseModel):
    """Provenance echoed back to the caller."""

    model_config = ConfigDict(extra="forbid")

    uri: str | None = Field(default=None, description="Source URI / filename.")
    title: str | None = Field(default=None, description="Document title, if known.")


class ConvertResponse(BaseModel):
    """Response body for ``POST /api/v1/convert``."""

    model_config = ConfigDict(extra="forbid")

    markdown: str = Field(description="The (optionally cleaned) Markdown text.")
    title: str | None = Field(default=None, description="Document title, if extracted.")
    source: SourceInfo = Field(description="Provenance of the converted document.")
    cleaned: bool = Field(
        default=False,
        description="Whether the text-cleaning pipeline was applied.",
    )


class GenerationStatsOut(BaseModel):
    """Diagnostic counters from block generation."""

    model_config = ConfigDict(extra="forbid")

    raw_block_count: int = Field(default=0, description="Blocks the LLM produced.")
    emitted: int = Field(default=0, description="Blocks that passed validation.")
    skipped: int = Field(default=0, description="Blocks dropped as invalid.")
    errors: list[str] = Field(
        default_factory=list,
        description="Per-block coercion errors (non-strict mode).",
    )


class GenerateResponse(BaseModel):
    """Response body for ``POST /api/v1/generate``."""

    model_config = ConfigDict(extra="forbid")

    blocks: list[dict[str, Any]] = Field(
        description=(
            "Generated IdeaBlocks serialized as flat JSON dicts "
            "(via ``IdeaBlock.model_dump(mode='json')``)."
        )
    )
    title: str | None = Field(default=None, description="Document title, if extracted.")
    source: SourceInfo = Field(description="Provenance of the source document.")
    cleaned: bool = Field(
        default=True,
        description="Whether the text-cleaning pipeline was applied.",
    )
    stats: GenerationStatsOut | None = Field(
        default=None,
        description="Generation diagnostics (present when ``with_stats`` was set).",
    )


class HealthResponse(BaseModel):
    """Response body for ``GET /api/v1/health``."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="ok")
    version: str = Field(description="SparkSage library version.")
    generator_configured: bool = Field(
        description="Whether block generation is available on this instance."
    )
    converter_configured: bool = Field(
        description="Whether file conversion is available on this instance.",
        default=True,
    )


def to_convert_response(out: object) -> ConvertResponse:
    """Build a :class:`ConvertResponse` from a service :class:`ConvertOutput`."""
    from sparksage.api.pipeline import ConvertOutput  # local to avoid cycle

    assert isinstance(out, ConvertOutput)
    return ConvertResponse(
        markdown=out.markdown,
        title=out.title,
        source=SourceInfo(uri=out.source.uri, title=out.source.title),
        cleaned=out.cleaned,
    )


def to_generate_response(out: GenerateOutput) -> GenerateResponse:
    """Build a :class:`GenerateResponse` from a service :class:`GenerateOutput`."""
    from sparksage.api.pipeline import _block_to_dict

    stats = None
    if out.stats is not None:
        stats = GenerationStatsOut(
            raw_block_count=out.stats.raw_block_count,
            emitted=out.stats.emitted,
            skipped=out.stats.skipped,
            errors=list(out.stats.errors),
        )
    return GenerateResponse(
        blocks=[_block_to_dict(b) for b in out.blocks],
        title=out.title,
        source=SourceInfo(uri=out.source.uri, title=out.source.title),
        cleaned=out.cleaned,
        stats=stats,
    )


# ---------------------------------------------------------------------------- #
# Document management
# ---------------------------------------------------------------------------- #
class DocumentSourceInfo(BaseModel):
    """Provenance of a stored document."""

    model_config = ConfigDict(extra="forbid")

    uri: str | None = Field(default=None, description="Source URI / filename.")
    title: str | None = Field(default=None, description="Document title, if known.")
    locator: str | None = Field(default=None, description="Position within source.")
    system: str | None = Field(default=None, description="Originating system.")


class DocumentResponse(BaseModel):
    """Full representation of a stored document (detail / create / update)."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(description="Stable unique id.")
    title: str | None = Field(default=None, description="Document title, if known.")
    summary: str | None = Field(default=None, description="Document-level summary.")
    body_markdown: str = Field(description="Full parsed Markdown body.")
    tags: list[str] = Field(default_factory=list, description="Free-form tags.")
    source: DocumentSourceInfo = Field(description="Provenance of the document.")
    created_at: datetime = Field(description="Creation time (UTC, ISO 8601).")
    updated_at: datetime = Field(description="Last write time (UTC, ISO 8601).")
    content_hash: str | None = Field(
        default=None, description="SHA-256 of the body for change detection."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Free-form caller metadata."
    )


class DocumentSummary(BaseModel):
    """Lightweight document row for list responses (no ``body_markdown``)."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(description="Stable unique id.")
    title: str | None = Field(default=None, description="Document title, if known.")
    summary: str | None = Field(default=None, description="Document-level summary.")
    tags: list[str] = Field(default_factory=list, description="Free-form tags.")
    source: DocumentSourceInfo = Field(description="Provenance of the document.")
    created_at: datetime = Field(description="Creation time (UTC, ISO 8601).")
    updated_at: datetime = Field(description="Last write time (UTC, ISO 8601).")
    content_hash: str | None = Field(
        default=None, description="SHA-256 of the body for change detection."
    )


class DocumentListResponse(BaseModel):
    """Paginated document listing."""

    model_config = ConfigDict(extra="forbid")

    items: list[DocumentSummary] = Field(description="The current page of documents.")
    count: int = Field(description="Number of items in this page.")
    total: int = Field(description="Total documents matching the filter.")
    tag: str | None = Field(default=None, description="Active tag filter, if any.")
    q: str | None = Field(default=None, description="Active text query, if any.")
    limit: int = Field(description="Page size used.")
    offset: int = Field(description="Offset used.")


class TagsResponse(BaseModel):
    """The distinct tag vocabulary across stored documents."""

    model_config = ConfigDict(extra="forbid")

    tags: list[str] = Field(description="Distinct tags, sorted ascending.")


class DocumentUpdateRequest(BaseModel):
    """JSON body for ``PATCH /api/v1/documents/{doc_id}``."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, description="New title.")
    tags: list[str] | None = Field(
        default=None, description="Replacement tag list (when supplied)."
    )
    summary: str | None = Field(default=None, description="New summary.")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Replacement metadata (when supplied)."
    )


class RetagRequest(BaseModel):
    """JSON body for ``POST /api/v1/documents/{doc_id}/tags``."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=8, ge=1, description="Number of tags to extract.")
    replace: bool = Field(
        default=True, description="Replace existing tags (True) or append (False)."
    )
    extra_tags: list[str] | None = Field(
        default=None, description="Additional tags to merge in."
    )


def _to_source_info(source: object) -> DocumentSourceInfo:
    return DocumentSourceInfo(
        uri=getattr(source, "uri", None),
        title=getattr(source, "title", None),
        locator=getattr(source, "locator", None),
        system=getattr(source, "system", None),
    )


def to_document_response(record: object) -> DocumentResponse:
    """Build a :class:`DocumentResponse` from a :class:`DocumentRecord`."""
    return DocumentResponse(
        doc_id=record.doc_id,
        title=record.title,
        summary=record.summary,
        body_markdown=record.body_markdown,
        tags=list(record.tags),
        source=_to_source_info(record.source),
        created_at=record.created_at,
        updated_at=record.updated_at,
        content_hash=record.content_hash,
        metadata=dict(record.metadata),
    )


def to_document_summary(record: object) -> DocumentSummary:
    """Build a :class:`DocumentSummary` from a :class:`DocumentRecord`."""
    return DocumentSummary(
        doc_id=record.doc_id,
        title=record.title,
        summary=record.summary,
        tags=list(record.tags),
        source=_to_source_info(record.source),
        created_at=record.created_at,
        updated_at=record.updated_at,
        content_hash=record.content_hash,
    )


def to_document_list_response(
    items: list[object],
    *,
    total: int,
    tag: str | None,
    q: str | None,
    limit: int,
    offset: int,
) -> DocumentListResponse:
    """Build a :class:`DocumentListResponse` from a page of records."""
    summaries = [to_document_summary(r) for r in items]
    return DocumentListResponse(
        items=summaries,
        count=len(summaries),
        total=total,
        tag=tag,
        q=q,
        limit=limit,
        offset=offset,
    )


# ---------------------------------------------------------------------------- #
# End-to-end QA (ingest-and-index + query + feedback)
# ---------------------------------------------------------------------------- #
class IngestAndIndexResponse(BaseModel):
    """Response body for ``POST /api/v1/knowledge_base/ingest``.

    Confirms that uploaded knowledge was parsed, chunked into IdeaBlocks, and
    indexed -- i.e. it is now retrievable via ``POST /api/v1/query``.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(description="Stored document id.")
    block_count: int = Field(description="Number of IdeaBlocks indexed.")
    blocks: list[dict[str, Any]] = Field(
        description=(
            "Generated IdeaBlocks serialized as flat JSON dicts "
            "(via ``IdeaBlock.model_dump(mode='json')``)."
        )
    )
    title: str | None = Field(default=None, description="Document title.")
    source: SourceInfo = Field(description="Provenance of the source document.")
    tags: list[str] = Field(default_factory=list, description="Document tags.")
    summary: str | None = Field(default=None, description="Document summary.")


class IngestJobSubmitResponse(BaseModel):
    """Response body for ``POST /api/v1/knowledge_base/ingest/async``.

    Returns a job id immediately -- the heavy convert -> generate -> embed ->
    index work runs in a background thread. Poll
    :class:`IngestJobSnapshotResponse` via ``GET /api/v1/jobs/{job_id}`` until
    ``status`` is terminal (``success`` / ``failed`` / ``cancelled``).
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(description="The opaque ingest job id.")
    status: str = Field(description="Initial status (``queued`` or ``running``).")
    filename: str | None = Field(
        default=None, description="The uploaded filename (for the per-file row)."
    )


class IngestJobSnapshotResponse(BaseModel):
    """Response body for ``GET /api/v1/jobs/{job_id}`` -- a pollable snapshot.

    Every field a progress UI needs: ``status`` + ``phase`` + ``percent`` for
    the bar, ``filename`` for the row, and ``doc_id`` / ``block_count`` /
    ``title`` once the work succeeds (``None`` / ``0`` until then). ``error``
    carries the failure message when ``status == failed``. ``result`` carries
    the full ingest payload (blocks + tags + summary) on success so the client
    gets the generated blocks in the same final poll that reports completion
    -- no second round-trip.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(description="The ingest job id.")
    status: str = Field(
        description="queued | running | success | failed | cancelled."
    )
    phase: str = Field(
        description=(
            "Coarse stage: queued | converting | generating | indexing | done."
        )
    )
    percent: float = Field(description="Completion fraction in [0, 1].")
    filename: str | None = Field(default=None, description="Uploaded filename.")
    title: str | None = Field(default=None, description="Document title on success.")
    block_count: int = Field(default=0, description="Indexed block count on success.")
    doc_id: str | None = Field(default=None, description="Stored document id on success.")
    error: str | None = Field(default=None, description="Failure message when failed.")
    duration: float | None = Field(
        default=None, description="Elapsed seconds (running or finished)."
    )
    result: IngestAndIndexResponse | None = Field(
        default=None,
        description=(
            "Full ingest payload (blocks / tags / summary) when "
            "``status == success``; ``None`` otherwise. Delivered on the "
            "terminal poll so the client renders the generated blocks without "
            "a second round-trip."
        ),
    )


class AskRequest(BaseModel):
    """JSON body for ``POST /api/v1/query``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="The natural-language question.")
    kb_id: str | None = Field(
        default=None,
        description=(
            "Target knowledge base id. Defaults to the active KB. "
            "Ignored when superseded by a tag/entity scoping that implies another KB."
        ),
    )
    k: int | None = Field(default=None, ge=1, description="Top-k chunks to retrieve.")
    use_lexical: bool | None = Field(
        default=None, description="Toggle the BM25 lexical leg of hybrid search."
    )
    use_rerank: bool | None = Field(
        default=None, description="Toggle re-ranking of the fused pool."
    )
    tags: list[str] | None = Field(
        default=None,
        description="Restrict to blocks carrying at least one of these Tag values.",
    )
    entities: list[str] | None = Field(
        default=None, description="Restrict to blocks referencing these entities."
    )
    languages: list[str] | None = Field(
        default=None, description="Restrict to blocks in these languages."
    )
    history: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "Prior conversation turns for multi-turn anaphora resolution. "
            'Each item is {"role": "user"|"assistant", "content": "..."}.'
        ),
    )
    mode: str = Field(
        default="default",
        description=(
            'QA strategy: "default" (single-shot QAEngine, one retrieval, fast) '
            'or "agent" (AgenticQAEngine -- an LLM-driven plan-act-observe-'
            "synthesize loop for multi-hop / comparative questions; slower). "
            '"agent" requires an agent controller to be configured server-side.'
        ),
    )
    stream: bool = Field(
        default=False,
        description=(
            "When ``True`` (and ``mode='agent'``) the response is an SSE stream "
            "of ``progress`` events (one per agent phase -- thinking / "
            "retrieving / synthesizing / done) terminated by a single "
            "``result`` event carrying the full :class:`AskResponse`. Ignored "
            "for ``mode='default'`` (single-shot is too fast to stream)."
        ),
    )


class CitationOut(BaseModel):
    """A single grounded citation backing part of a generated answer."""

    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(description="Backing IdeaBlock id.")
    quote: str = Field(default="", description="Supporting snippet.")
    uri: str | None = Field(default=None, description="``source.uri`` of the block.")
    locator: str | None = Field(
        default=None, description="``source.locator`` (page / line / anchor)."
    )
    title: str | None = Field(default=None, description="``source.title`` of the block.")


class RetrievedChunkOut(BaseModel):
    """One retrieved chunk surfaced to the caller (for transparency)."""

    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(description="The IdeaBlock id.")
    name: str = Field(description="Block name / short title.")
    critical_question: str = Field(description="The question this block answers.")
    trusted_answer: str = Field(description="The verified answer text.")
    score: float = Field(description="Final relevance score (higher is better).")
    rank: int = Field(default=0, description="0-indexed position in the ranked list.")


class AgentStepOut(BaseModel):
    """One executed retrieval step in the agent reasoning trajectory.

    Surfaced only for ``mode="agent"`` answers so a UI can render the ReAct
    plan-act-observe trace (the seed retrieval + each controller-decided
    sub-query). Mirrors :class:`~sparksage.agent.models.AgentStep`.
    """

    model_config = ConfigDict(extra="forbid")

    thought: str = Field(default="", description="The controller's reasoning (ReAct Thought).")
    query: str = Field(description="The sub-query that was actually retrieved.")
    retrieved_count: int = Field(
        description="How many chunks this single retrieval returned."
    )
    observation: str = Field(
        default="", description="Compact summary of what was found (fed back to the controller)."
    )
    relevance_score: float | None = Field(
        default=None,
        description=(
            "Per-step retrieval relevance score in [0, 1] (the ISREL gate). "
            "``None`` when no retrieval grader is wired on the agent."
        ),
    )
    relevance_reasoning: str | None = Field(
        default=None,
        description="The grader's brief explanation for ``relevance_score``.",
    )
    refined_query: str | None = Field(
        default=None,
        description=(
            "The refined query the step re-retrieved on a low relevance score "
            "(``None`` when no per-step refinement happened)."
        ),
    )
    created_at: datetime | None = Field(
        default=None, description="When the step ran (UTC, ISO 8601), for timeline rendering."
    )


class AskResponse(BaseModel):
    """Response body for ``POST /api/v1/query`` -- a grounded answer or abstention."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="The original question.")
    answer: str = Field(description="The surfaced answer (or abstention reply).")
    abstained: bool = Field(
        description="Whether the system chose not to answer (low faithfulness / recall)."
    )
    abstention_reason: str | None = Field(
        default=None, description="Why the system abstained (when ``abstained``)."
    )
    citations: list[CitationOut] = Field(
        default_factory=list,
        description="Grounded citations, each bound to ``source.locator``.",
    )
    retrieved: list[RetrievedChunkOut] = Field(
        default_factory=list,
        description="The retrieved chunks backing the answer (for transparency).",
    )
    cached: bool = Field(
        default=False, description="Whether the result was served from the cache."
    )
    confidence: float = Field(
        default=0.0, description="Effective confidence (generator * faithfulness)."
    )
    intent: str | None = Field(
        default=None, description="Classified query intent (when a processor is wired)."
    )
    mode: str = Field(
        default="default",
        description=(
            'QA strategy that produced this answer: "default" (single-shot) '
            'or "agent" (multi-hop plan-act-observe-synthesize loop).'
        ),
    )
    iterations: int | None = Field(
        default=None,
        description=(
            "Extra retrievals the agent controller requested beyond the seed "
            "(agent mode only; ``None`` for single-shot)."
        ),
    )
    aborted: bool | None = Field(
        default=None,
        description=(
            "Whether the agent loop hit ``max_iterations`` without synthesizing "
            "(agent mode only; ``None`` for single-shot)."
        ),
    )
    steps: list[AgentStepOut] = Field(
        default_factory=list,
        description=(
            "The agent reasoning trajectory (seed + each controller-decided "
            "retrieval). Empty for single-shot mode."
        ),
    )


class KnowledgeBaseResponse(BaseModel):
    """Snapshot of the knowledge base behind the QA service."""

    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(description="Stable unique id.")
    name: str = Field(description="Human-readable KB name.")
    block_count: int = Field(description="Number of indexed IdeaBlocks.")
    document_count: int = Field(description="Number of stored documents.")
    language: str = Field(default="zh", description="Default block language.")
    description: str | None = Field(default=None, description="Free-text description.")
    tags: list[str] = Field(default_factory=list, description="KB-level labels.")
    active: bool = Field(
        default=False,
        description="Whether this is the active routing target.",
    )


class KnowledgeBaseSummary(BaseModel):
    """One row in the multi-KB listing (lightweight, no block payload)."""

    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(description="Stable unique id.")
    name: str = Field(description="Human-readable KB name.")
    description: str | None = Field(default=None, description="Free-text description.")
    language: str = Field(default="zh", description="Default block language.")
    tags: list[str] = Field(default_factory=list, description="KB-level labels.")
    block_count: int = Field(default=0, description="Number of indexed IdeaBlocks.")
    document_count: int = Field(default=0, description="Number of stored documents.")
    active: bool = Field(
        default=False,
        description="Whether this is the active routing target.",
    )
    created_at: datetime = Field(description="Creation time (UTC, ISO 8601).")
    updated_at: datetime = Field(description="Last write time (UTC, ISO 8601).")


class KnowledgeBaseListResponse(BaseModel):
    """Paginated multi-KB listing."""

    model_config = ConfigDict(extra="forbid")

    items: list[KnowledgeBaseSummary] = Field(description="The current page of KBs.")
    count: int = Field(description="Number of items in this page.")
    total: int = Field(description="Total knowledge bases registered.")
    limit: int = Field(description="Page size used.")
    offset: int = Field(description="Offset used.")


class CreateKnowledgeBaseRequest(BaseModel):
    """JSON body for ``POST /api/v1/knowledge_bases``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Human-readable KB name.")
    description: str | None = Field(default=None, description="Free-text description.")
    language: str = Field(
        default="zh", min_length=2, max_length=16, description="Default block language."
    )
    tags: list[str] | None = Field(
        default=None, description="Free-form KB-level labels."
    )
    set_active: bool = Field(
        default=True,
        description="Make this KB the active routing target on creation.",
    )


class FeedbackRequest(BaseModel):
    """JSON body for ``POST /api/v1/feedback``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="The query that produced the answer.")
    answer_text: str = Field(default="", description="The surfaced answer text.")
    rating: str = Field(
        ...,
        description="One of: positive | negative | corrected.",
    )
    correction: str | None = Field(
        default=None, description="User-supplied corrected answer (for 'corrected')."
    )
    block_ids: list[str] | None = Field(
        default=None, description="Block ids the answer was built from."
    )


class FeedbackResponse(BaseModel):
    """Response body for ``POST /api/v1/feedback``."""

    model_config = ConfigDict(extra="forbid")

    feedback_id: str = Field(description="Stored feedback id.")
    rating: str = Field(description="The recorded rating.")
    acknowledged: bool = Field(default=True)


class FeedbackStatsResponse(BaseModel):
    """Response body for ``GET /api/v1/feedback``."""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(default=0, description="Total feedback records.")
    positive: int = Field(default=0, description="Positive count.")
    negative: int = Field(default=0, description="Negative count.")
    corrected: int = Field(default=0, description="Corrected count.")
    approval: float = Field(default=0, description="Approval ratio (positive / total).")


class ConfigResponse(BaseModel):
    """Response body for ``GET /api/v1/config``.

    Each known key is always present (empty string when unset). Secrets (keys
    ending in ``_API_KEY``) are returned as ``"****"`` rather than the real
    value so the response is safe to render in a browser.
    """

    model_config = ConfigDict(extra="forbid")

    variables: dict[str, str] = Field(
        description="Effective configuration values (secrets masked)."
    )


class ConfigUpdateResponse(BaseModel):
    """Response body for ``POST /api/v1/config``."""

    model_config = ConfigDict(extra="forbid")

    applied: list[str] = Field(
        description="Keys that were written (secrets left unchanged are excluded)."
    )
    restart_required: bool = Field(
        default=True,
        description="Always true -- a restart is needed for new values to take effect.",
    )
    message: str = Field(description="Human-readable status message.")


class BlockOut(BaseModel):
    """One IdeaBlock row for the knowledge-base listing.

    Serialized as a flat JSON dict (via ``IdeaBlock.model_dump(mode='json')``)
    so all schema fields (tags / entities / status / parents / ...) are
    available to the detail view.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable unique block id.")
    name: str = Field(description="Short title / headline.")
    critical_question: str = Field(description="The question this block answers.")
    trusted_answer: str = Field(description="The verified answer text.")
    tags: list[str] = Field(default_factory=list, description="Coarse semantic tags.")
    keywords: list[str] = Field(default_factory=list, description="Lexical keywords.")
    language: str = Field(default="en", description="Answer language code.")
    status: str = Field(default="DRAFT", description="Lifecycle status.")
    confidence: float | None = Field(default=None, description="Merge confidence.")
    parents: list[str] = Field(default_factory=list, description="Merged-in block ids.")
    source: SourceInfo | None = Field(default=None, description="Block provenance.")
    kb_id: str | None = Field(default=None, description="Owning knowledge-base id.")
    created_at: datetime | None = Field(default=None, description="Creation time.")
    updated_at: datetime | None = Field(default=None, description="Last update time.")


class BlockListResponse(BaseModel):
    """Paginated IdeaBlock listing for the knowledge-base browser."""

    model_config = ConfigDict(extra="forbid")

    items: list[BlockOut] = Field(description="The current page of blocks.")
    count: int = Field(description="Number of items in this page.")
    total: int = Field(description="Total blocks matching the filter.")
    limit: int = Field(description="Page size used.")
    offset: int = Field(description="Offset used.")


class FeedbackRecordOut(BaseModel):
    """One feedback record for the recent-feedback listing."""

    model_config = ConfigDict(extra="forbid")

    feedback_id: str = Field(description="Stable feedback id.")
    query: str = Field(description="The query that produced the answer.")
    answer_text: str = Field(default="", description="The surfaced answer text.")
    rating: str = Field(description="positive | negative | corrected.")
    correction: str | None = Field(default=None, description="User-supplied correction.")
    block_ids: list[str] = Field(default_factory=list, description="Backing block ids.")
    kb_id: str | None = Field(default=None, description="Source knowledge base.")
    created_at: datetime = Field(description="UTC timestamp.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Caller metadata.")


class FeedbackListResponse(BaseModel):
    """Paginated recent-feedback listing."""

    model_config = ConfigDict(extra="forbid")

    items: list[FeedbackRecordOut] = Field(description="Feedback records, newest-first.")
    count: int = Field(description="Number of items in this page.")
    total: int = Field(description="Total feedback records.")
    limit: int = Field(description="Page size used.")
    offset: int = Field(description="Offset used.")


class QueryHistoryItem(BaseModel):
    """One persisted QA turn for the conversation-history listing."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str = Field(description="Stable turn id.")
    role: str = Field(description="user | assistant.")
    content: str = Field(description="The user query or the surfaced answer text.")
    kb_id: str | None = Field(default=None, description="Source knowledge base.")
    result: AskResponse | None = Field(
        default=None,
        description="Full answer payload for assistant turns (re-renderable).",
    )
    created_at: datetime = Field(description="UTC timestamp.")


class QueryHistoryResponse(BaseModel):
    """Paginated QA conversation-history listing (newest-first)."""

    model_config = ConfigDict(extra="forbid")

    items: list[QueryHistoryItem] = Field(description="Turns, newest-first.")
    count: int = Field(description="Number of items in this page.")
    total: int = Field(description="Total turns.")
    limit: int = Field(description="Page size used.")
    offset: int = Field(description="Offset used.")


def _to_block_out(block: object) -> BlockOut:
    """Build a :class:`BlockOut` from an :class:`~sparksage.schema.IdeaBlock`."""
    tags = []
    for t in getattr(block, "tags", []) or []:
        tags.append(t.value if hasattr(t, "value") else str(t))
    source_obj = getattr(block, "source", None)
    source = (
        SourceInfo(
            uri=getattr(source_obj, "uri", None),
            title=getattr(source_obj, "title", None),
        )
        if source_obj is not None
        else None
    )
    parents = []
    for p in getattr(block, "parents", []) or []:
        parents.append(str(p))
    status_val = getattr(block, "status", None)
    status_str = status_val.value if hasattr(status_val, "value") else str(status_val)
    return BlockOut(
        id=str(block.id),
        name=getattr(block, "name", ""),
        critical_question=getattr(block, "critical_question", ""),
        trusted_answer=getattr(block, "trusted_answer", ""),
        tags=tags,
        keywords=list(getattr(block, "keywords", []) or []),
        language=getattr(block, "language", "en"),
        status=status_str,
        confidence=getattr(block, "confidence", None),
        parents=parents,
        source=source,
        kb_id=getattr(block, "kb_id", None),
        created_at=getattr(block, "created_at", None),
        updated_at=getattr(block, "updated_at", None),
    )


def _to_ingest_response(result: object) -> IngestAndIndexResponse:
    """Build an :class:`IngestAndIndexResponse` from an :class:`IngestResult`."""
    return IngestAndIndexResponse(
        doc_id=result.doc_id,
        block_count=result.block_count,
        blocks=[b.model_dump(mode="json") for b in result.blocks],
        title=result.title,
        source=SourceInfo(uri=result.source.uri, title=result.source.title),
        tags=list(result.tags),
        summary=result.summary,
    )


def _to_ingest_job_snapshot_response(
    snap: object,
    *,
    result: object | None = None,
) -> IngestJobSnapshotResponse:
    """Build an :class:`IngestJobSnapshotResponse` from an :class:`IngestJobSnapshot`.

    Accepts the frozen dataclass structurally so this stays free of an import
    cycle with :mod:`sparksage.api.ingest_jobs`. ``result`` is the live
    :class:`IngestResult` (read off the job); when non-None it is serialized
    into the ``result`` field so the client gets the generated blocks on the
    terminal poll.
    """
    status = getattr(snap, "status", None)
    status_val = status.value if hasattr(status, "value") else str(status)
    payload: IngestAndIndexResponse | None = None
    if result is not None and status_val == "success":
        payload = _to_ingest_response(result)
    return IngestJobSnapshotResponse(
        job_id=getattr(snap, "job_id", ""),
        status=status_val,
        phase=getattr(snap, "phase", "queued"),
        percent=float(getattr(snap, "percent", 0.0)),
        filename=getattr(snap, "filename", None),
        title=getattr(snap, "title", None),
        block_count=int(getattr(snap, "block_count", 0)),
        doc_id=getattr(snap, "doc_id", None),
        error=getattr(snap, "error", None),
        duration=getattr(snap, "duration", None),
        result=payload,
    )


def _build_filter_from_request(
    body: AskRequest,
) -> tuple[RetrievalFilter | None, ConversationContext | None]:
    """Translate an :class:`AskRequest` into the retrieval filter + context.

    ``body.kb_id`` is *not* folded into the :class:`RetrievalFilter` here -- the
    QAService resolves it to the right KB aggregate / engine directly. The
    filter still carries tag / entity / language scoping.
    """
    from sparksage.query.context import ConversationContext
    from sparksage.retrieve.models import RetrievalFilter
    from sparksage.schema.enums import Tag

    flt: RetrievalFilter | None = None
    if body.tags:
        parsed: set[Tag] = set()
        for raw in body.tags:
            try:
                parsed.add(Tag(raw))
            except ValueError:
                continue
        if parsed:
            flt = RetrievalFilter(tags=parsed)
    if body.entities:
        ent_set = set(body.entities)
        flt = RetrievalFilter(entities=ent_set) if flt is None else _merge(flt, entities=ent_set)
    if body.languages:
        lang_set = set(body.languages)
        flt = (
            RetrievalFilter(languages=lang_set)
            if flt is None
            else _merge(flt, languages=lang_set)
        )

    context: ConversationContext | None = None
    if body.history:
        ctx = ConversationContext()
        for turn in body.history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            ctx = ctx.with_turn(role, content)
        context = ctx

    return flt, context


def _merge(flt: RetrievalFilter, **kw: Any) -> RetrievalFilter:
    from dataclasses import replace

    return replace(flt, **kw)


def _to_ask_response(result: QAResultLike) -> AskResponse:
    """Build an :class:`AskResponse` from a :class:`~sparksage.qa.QAResult`.

    When ``result`` is an :class:`~sparksage.agent.AgentResult`, the agent
    trajectory (``mode`` / ``iterations`` / ``aborted`` / ``steps``) is folded
    in alongside the shared surface -- so a single-shot and an agentic answer
    stay interchangeable, but the agentic trace is no longer discarded (it
    flows through to the HTTP client, the persisted history, and the UI).
    The agent fields default to ``None`` / ``[]`` / ``"default"`` so a plain
    :class:`~sparksage.qa.QAResult` is serialized byte-for-byte as before.
    """
    citations = [
        CitationOut(
            block_id=str(getattr(c, "block_id", "")),
            quote=getattr(c, "quote", "") or "",
            uri=getattr(c, "uri", None),
            locator=getattr(c, "locator", None),
            title=getattr(c, "title", None),
        )
        for c in result.citations
    ]

    retrieved: list[RetrievedChunkOut] = []
    if result.retrieval is not None:
        for chunk in result.retrieval.chunks:
            retrieved.append(
                RetrievedChunkOut(
                    block_id=str(chunk.block.id),
                    name=chunk.block.name,
                    critical_question=chunk.block.critical_question,
                    trusted_answer=chunk.block.trusted_answer,
                    score=chunk.score,
                    rank=getattr(chunk, "rank", 0),
                )
            )

    intent: str | None = None
    if result.query_result is not None and result.query_result.intent is not None:
        intent = (
            result.query_result.intent.intent.value
            if hasattr(result.query_result.intent.intent, "value")
            else str(result.query_result.intent.intent)
        )

    # AgentResult-only trajectory. Detected by isinstance (local import keeps
    # the module import-cycle-free) so the single-shot path is untouched.
    from sparksage.agent.models import AgentResult

    if isinstance(result, AgentResult):
        steps = [
            AgentStepOut(
                thought=s.thought,
                query=s.query,
                retrieved_count=s.retrieved_count,
                observation=s.observation,
                relevance_score=(
                    s.relevance.score if getattr(s, "relevance", None) is not None else None
                ),
                relevance_reasoning=(
                    s.relevance.reasoning if getattr(s, "relevance", None) is not None else None
                ),
                refined_query=getattr(s, "refined_query", None),
                created_at=s.created_at,
            )
            for s in result.steps
        ]
        return AskResponse(
            query=result.query,
            answer=result.text,
            abstained=result.abstained,
            abstention_reason=(
                result.answer.abstention_reason
                if result.answer is not None
                else None
            ),
            citations=citations,
            retrieved=retrieved,
            cached=result.cached,
            confidence=result.answer.confidence if result.answer is not None else 0.0,
            intent=intent,
            mode="agent",
            iterations=result.iterations,
            aborted=result.aborted,
            steps=steps,
        )

    return AskResponse(
        query=result.query,
        answer=result.text,
        abstained=result.abstained,
        abstention_reason=(
            result.answer.abstention_reason
            if result.answer is not None
            else None
        ),
        citations=citations,
        retrieved=retrieved,
        cached=result.cached,
        confidence=result.answer.confidence if result.answer is not None else 0.0,
        intent=intent,
        mode="default",
    )


class QAResultLike:
    """Structural type hint for :func:`_to_ask_response` (avoids import cycle)."""

    query: str
    citations: list[Any]
    retrieval: Any
    query_result: Any
    answer: Any
    abstained: bool
    cached: bool
