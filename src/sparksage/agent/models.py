"""Data models for the agentic QA core.

These are the framework-agnostic contracts the agentic loop
(:mod:`sparksage.agent.engine`) produces and a web layer consumes. They mirror
the spirit of :mod:`sparksage.qa.engine.QAResult` -- an :class:`AgentResult`
exposes the same ``query`` / ``text`` / ``citations`` / ``abstained`` /
``answer`` / ``retrieval`` surface so a :class:`~sparksage.qa.QAEngine` and an
:class:`~sparksage.agent.engine.AgenticQAEngine` are interchangeable behind the
API's ``AskResponse`` serializer -- while adding the trajectory the agent loop
accumulates (the ``steps`` it reasoned through and the ``evidence`` it gathered).

The agent loop is deliberately a different *orchestrator* over the *same*
building blocks the single-shot :class:`~sparksage.qa.QAEngine` uses
(:class:`~sparksage.retrieve.Retriever`, :class:`~sparksage.reader.Reader`,
:class:`~sparksage.query.QueryProcessor`), so an :class:`AgentResult` can be
served by the exact same ``_to_ask_response`` path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from sparksage.retrieve.models import RetrievalFilter, RetrievalResult, RetrievedChunk

if TYPE_CHECKING:
    from sparksage.query.context import ConversationContext
    from sparksage.query.processor import QueryResult
    from sparksage.reader.orchestrator import AnswerResult


class ActionType(str, Enum):
    """The next move the agent controller has chosen.

    ``RETRIEVE`` runs another knowledge-base retrieval (a sub-question in a
    multi-hop / comparative plan); ``SYNTHESIZE`` stops the loop and hands the
    accumulated evidence to the :class:`~sparksage.reader.Reader`.
    """

    RETRIEVE = "retrieve"
    SYNTHESIZE = "synthesize"


@dataclass(frozen=True)
class AgentAction:
    """One controller decision: what to do next and why.

    Attributes
    ----------
    action:
        The :class:`ActionType` to perform.
    thought:
        The controller's brief reasoning (ReAct "Thought"). Surfaced for
        transparency and fed back into the next iteration's prompt.
    query:
        The sub-query to retrieve, present only for ``RETRIEVE``.
    k:
        Optional override of how many chunks this retrieval should return.
    filter:
        Optional per-step metadata scope. When ``None`` the call-level filter
        (kb_id / tags from self-query) is inherited, which is the common case.
    """

    action: ActionType
    thought: str = ""
    query: str | None = None
    k: int | None = None
    filter: RetrievalFilter | None = None


@dataclass(frozen=True)
class AgentStep:
    """One executed retrieval step in the agent trajectory.

    Attributes
    ----------
    thought:
        The controller's reasoning that produced this retrieval.
    query:
        The sub-query that was actually retrieved.
    retrieved_count:
        How many chunks this single retrieval returned.
    observation:
        Compact summary of what was found (truncated ``critical_question ||
        trusted_answer`` of the top hits) -- fed back to the controller so it
        does not re-ask the same thing.
    created_at:
        When the step ran, for timeline / progress rendering.
    """

    thought: str
    query: str
    retrieved_count: int
    observation: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentState:
    """The mutable working state the loop threads between iterations.

    Attributes
    ----------
    question:
        The original user question (the synthesis target).
    context:
        Optional multi-turn conversation context (anaphora resolution).
    steps:
        The retrieval steps executed so far (the reasoning trajectory).
    evidence:
        Accumulated :class:`RetrievedChunk` list, de-duplicated by block id
        (best score kept) -- what the reader will synthesize over.
    """

    question: str
    context: ConversationContext | None = None
    steps: list[AgentStep] = field(default_factory=list)
    evidence: list[RetrievedChunk] = field(default_factory=list)


#: Agent-loop phases reported through :class:`AgentProgress`.
PHASE_THINKING = "thinking"
PHASE_RETRIEVING = "retrieving"
PHASE_SYNTHESIZING = "synthesizing"
PHASE_DONE = "done"


@dataclass(frozen=True)
class AgentProgress:
    """One snapshot of the agent loop, emitted via the ``on_progress`` callback.

    Mirrors the ``{percent, phase, details}`` shape :class:`DistillProgress`
    exposes, so a polling REST client treats a long agent run exactly like a
    long Distill run.

    Attributes
    ----------
    iteration:
        The iteration about to start (1-indexed) or just finished.
    max_iterations:
        The cap on extra retrievals beyond the seed.
    phase:
        One of ``thinking`` / ``retrieving`` / ``synthesizing`` / ``done``.
    action:
        The :class:`ActionType` chosen this iteration (``None`` until the
        controller has answered).
    thought:
        The controller's reasoning for the current action.
    query:
        The sub-query being retrieved (``None`` while thinking / synthesizing).
    evidence_count:
        How many distinct chunks have been gathered so far.
    """

    iteration: int
    max_iterations: int
    phase: str = PHASE_THINKING
    action: ActionType | None = None
    thought: str = ""
    query: str | None = None
    evidence_count: int = 0

    @property
    def percent(self) -> float:
        """Coarse completion estimate in ``[0, 1]`` (snaps to 1.0 when done).

        Exact progress is unknowable for an open-ended loop (the controller
        decides when it is done), so this is an *upper-bound* estimate based on
        the iteration count -- the UI shows movement without over-promising.
        """
        if self.phase == PHASE_DONE or self.max_iterations <= 0:
            return 1.0
        ratio = self.iteration / max(self.max_iterations, 1)
        if ratio > 1.0:
            ratio = 1.0
        return ratio


@dataclass
class AgentResult:
    """The full outcome of one :meth:`AgenticQAEngine.ask` call.

    Exposes the same ``query`` / ``text`` / ``citations`` / ``abstained`` /
    ``answer`` surface as :class:`~sparksage.qa.QAResult`, so the HTTP
    ``AskResponse`` serializer (``_to_ask_response``) works unchanged, while
    adding the ``steps`` / ``evidence`` / ``iterations`` the agent accumulated.

    Attributes
    ----------
    query:
        The original user question.
    answer:
        The :class:`~sparksage.reader.AnswerResult` from the synthesis step, or
        ``None`` when the query was intercepted before retrieval.
    steps:
        The retrieval steps the loop executed (the reasoning trajectory).
    iterations:
        Number of *extra* retrievals the controller requested beyond the seed
        (``0`` for a single-pass / identity controller).
    evidence:
        The accumulated chunks the synthesis was built from.
    aborted:
        ``True`` when the loop hit ``max_iterations`` without the controller
        choosing to synthesize (the answer is a best-effort over the evidence
        gathered so far, never a hallucination -- the reader still abstains on
        empty / unfaithful evidence).
    query_result:
        The :class:`~sparksage.query.QueryResult` when a
        :class:`~sparksage.query.QueryProcessor` is wired (intent + rewrite),
        or ``None``.
    cached:
        Always ``False`` for an agent run (agent runs are not cached -- the
        cache lives one layer up on :class:`~sparksage.qa.QAEngine`). Present so
        the result is shape-compatible with :class:`~sparksage.qa.QAResult`.
    """

    query: str
    answer: AnswerResult | None = None
    steps: list[AgentStep] = field(default_factory=list)
    iterations: int = 0
    evidence: list[RetrievedChunk] = field(default_factory=list)
    aborted: bool = False
    query_result: QueryResult | None = None
    cached: bool = False

    @property
    def accepted(self) -> bool:
        """Whether the query passed interception (or no processor was wired)."""
        return self.query_result is None or self.query_result.accepted

    @property
    def abstained(self) -> bool:
        """Whether the reader abstained (or the query was rejected)."""
        if self.answer is not None:
            return self.answer.abstained
        return not self.accepted

    @property
    def text(self) -> str:
        """The surfaced answer text (generated, abstention reply, or reject)."""
        if self.answer is not None:
            return self.answer.answer.text
        if self.query_result is not None and self.query_result.default_reply:
            return self.query_result.default_reply
        return ""

    @property
    def citations(self) -> list[Any]:
        """The grounded citations on the surfaced answer (empty if none)."""
        if self.answer is not None:
            return list(self.answer.answer.citations)
        return []

    @property
    def retrieval(self) -> RetrievalResult:
        """A :class:`RetrievalResult` view over the accumulated evidence.

        Lets the HTTP ``AskResponse`` serializer surface the agent's gathered
        chunks as ``retrieved`` exactly as it does for a single-shot result,
        without the serializer needing an agent-aware branch.
        """
        return RetrievalResult(query=self.query, chunks=list(self.evidence))


__all__ = [
    "PHASE_DONE",
    "PHASE_RETRIEVING",
    "PHASE_SYNTHESIZING",
    "PHASE_THINKING",
    "ActionType",
    "AgentAction",
    "AgentProgress",
    "AgentResult",
    "AgentState",
    "AgentStep",
]
