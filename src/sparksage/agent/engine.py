"""The agentic QA engine: a plan-act-observe-synthesize loop over the QA core.

:class:`AgenticQAEngine` is a *different orchestrator* over the *same* building
blocks the single-shot :class:`~sparksage.qa.QAEngine` uses -- it reuses
:class:`~sparksage.retrieve.Retriever`, :class:`~sparksage.reader.Reader` and
:class:`~sparksage.query.QueryProcessor` unchanged, but replaces the fixed
retrieve-once pipeline with an LLM-driven control loop. That loop is what turns
SparkSage from a one-shot RAG into an *agentic* RAG (the planning-execution-
reflection cycle): for a complex / multi-hop / comparative question the
controller decomposes it into a sequence of focused retrievals, gathering
evidence until it judges it sufficient, then synthesizes one grounded answer.

The loop is shaped exactly like :class:`~sparksage.distill.DistillPipeline.run`:
a bounded ``max_iterations`` cap, an ``on_progress`` callback fired at each
phase boundary (thinking -> retrieving -> synthesizing -> done), and a
cooperative ``is_cancelled`` predicate polled between iterations. That is the
canonical long-running-job recipe in this codebase, so a future
``/api/v1/query/agent`` route can wrap a run in a pollable job (mirroring the
planned ``/api/v1/distill`` route) with no new concurrency primitives.

Crucially the agent never hallucinates: if the controller runs out of
iterations (``aborted=True``) or gathers no evidence, the final answer still
flows through the :class:`~sparksage.reader.Reader`'s abstention gate
(faithfulness / confidence floors), so a starved agent says "I don't know"
rather than invent an answer. An :class:`~sparksage.agent.models.AgentResult`
exposes the same ``query`` / ``text`` / ``citations`` / ``abstained`` /
``answer`` surface as :class:`~sparksage.qa.QAResult`, so the HTTP
``AskResponse`` serializer works unchanged.

Pipeline::

    question
        -> (QueryProcessor intercept)            [optional, out-of-domain gate]
        -> seed retrieval                        (always: guarantees evidence)
        -> loop (bounded by max_iterations):
              controller.next_action(state)      (ReAct: retrieve more | synthesize)
              if retrieve: search -> merge evidence -> record step
              if synthesize / cancelled / capped: break
        -> Reader.answer(question, evidence)     (generate -> judge -> abstain)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from sparksage.agent.controller import AgentController
from sparksage.agent.models import (
    PHASE_DONE,
    PHASE_RETRIEVING,
    PHASE_SYNTHESIZING,
    PHASE_THINKING,
    ActionType,
    AgentAction,
    AgentProgress,
    AgentResult,
    AgentState,
    AgentStep,
)
from sparksage.query.context import ConversationContext
from sparksage.query.processor import QueryProcessor, QueryResult
from sparksage.reader.orchestrator import Reader
from sparksage.retrieve.models import RetrievalFilter, RetrievedChunk
from sparksage.retrieve.orchestrator import RetrievalConfig, Retriever

_logger = logging.getLogger(__name__)

#: Default cap on *extra* retrievals beyond the seed retrieval.
DEFAULT_MAX_ITERATIONS = 4

#: Default ceiling on how many distinct chunks accumulate as evidence.
DEFAULT_MAX_EVIDENCE = 20

#: How many of a step's new chunks feed its observation summary.
DEFAULT_OBSERVATION_TOP_K = 5

#: Truncate each observation chunk's ``trusted_answer``.
DEFAULT_OBSERVATION_ANSWER_CHARS = 160

ProgressCallback = Callable[[AgentProgress], None]
CancelledPredicate = Callable[[], bool]


def _new_observation(
    new_chunks: list[RetrievedChunk],
    *,
    top_k: int,
    answer_chars: int,
) -> str:
    """Compact summary of a step's fresh hits, fed back to the controller."""
    if not new_chunks:
        return "(no new hits)"
    lines: list[str] = []
    for c in new_chunks[:top_k]:
        answer = c.block.trusted_answer
        if len(answer) > answer_chars:
            answer = answer[:answer_chars].rstrip() + "..."
        lines.append(f"[{c.block.id}] {c.block.critical_question} || {answer}")
    if len(new_chunks) > top_k:
        lines.append(f"(...and {len(new_chunks) - top_k} more)")
    return "\n".join(lines)


def _merge_evidence(
    evidence: list[RetrievedChunk],
    incoming: list[RetrievedChunk],
    *,
    max_evidence: int,
) -> list[RetrievedChunk]:
    """Merge new chunks into evidence, de-duplicating by block id (best score kept).

    Returns a new list capped at ``max_evidence`` (best-scored first). A block
    already present keeps whichever copy has the higher score, so a sharper
    re-retrieval upgrades its slot.
    """
    by_id: dict[str, RetrievedChunk] = {str(c.block.id): c for c in evidence}
    for c in incoming:
        bid = str(c.block.id)
        existing = by_id.get(bid)
        if existing is None or c.score > existing.score:
            by_id[bid] = c
    merged = sorted(by_id.values(), key=lambda c: c.score, reverse=True)
    if max_evidence > 0:
        merged = merged[:max_evidence]
    return merged


class AgenticQAEngine:
    """Agentic QA: a plan-act-observe-synthesize loop over the QA core.

    Parameters
    ----------
    controller:
        The :class:`AgentController` "brain". :class:`~sparksage.agent.IdentityController`
        collapses the loop to a single retrieval + synthesize (the single-shot
        baseline); :class:`~sparksage.agent.LLMAgentController` drives a ReAct
        loop for multi-hop / comparative questions.
    retriever:
        Any :class:`~sparksage.retrieve.Retriever` (reused per retrieval step).
    reader:
        Any :class:`~sparksage.reader.Reader` -- invoked once, at synthesis.
    query_processor:
        Optional :class:`~sparksage.query.QueryProcessor`. When wired, the query
        is classified first (out-of-domain interception) and the rewrite seeds
        the first retrieval -- the symmetric query-side gate to the reader's
        answer-side abstention.
    max_iterations:
        Maximum number of *extra* retrievals beyond the seed (default ``4``).
        ``0`` collapses to a single-shot run regardless of the controller.
    max_evidence:
        Ceiling on accumulated evidence chunks (best-scored kept). Bounds the
        synthesis prompt cost for long agent runs.
    config:
        :class:`~sparksage.retrieve.RetrievalConfig` defaults for each retrieval
        (``k`` / ``use_lexical`` / ``use_rerank`` / fusion weights ...).
    observation_top_k, observation_answer_chars:
        How much of each step's fresh hits is summarized back to the controller
        (bounds per-iteration prompt cost).

    Examples
    --------
    >>> from sparksage import FakeLLMClient                         # doctest: +SKIP
    >>> from sparksage.agent import AgenticQAEngine, LLMAgentController  # doctest: +SKIP
    >>> engine = AgenticQAEngine(                                   # doctest: +SKIP
    ...     controller=LLMAgentController(FakeLLMClient(responses=[...])),
    ...     retriever=retriever, reader=reader,
    ... )
    >>> result = engine.ask("compare A and B revenue")             # doctest: +SKIP
    >>> print(result.text, result.iterations)                      # doctest: +SKIP
    """

    def __init__(
        self,
        controller: AgentController,
        retriever: Retriever,
        reader: Reader,
        *,
        query_processor: QueryProcessor | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_evidence: int = DEFAULT_MAX_EVIDENCE,
        config: RetrievalConfig | None = None,
        observation_top_k: int = DEFAULT_OBSERVATION_TOP_K,
        observation_answer_chars: int = DEFAULT_OBSERVATION_ANSWER_CHARS,
    ) -> None:
        if not isinstance(controller, AgentController):
            raise TypeError(
                "controller must implement the AgentController protocol"
            )
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
            raise TypeError("max_iterations must be an int")
        if max_iterations < 0:
            raise ValueError("max_iterations must be >= 0")
        if not isinstance(max_evidence, int) or isinstance(max_evidence, bool):
            raise TypeError("max_evidence must be an int")
        if max_evidence < 1:
            raise ValueError("max_evidence must be >= 1")
        self._controller = controller
        self._retriever = retriever
        self._reader = reader
        self._query_processor = query_processor
        self._max_iterations = max_iterations
        self._max_evidence = max_evidence
        self._config = config if config is not None else RetrievalConfig()
        self._observation_top_k = observation_top_k
        self._observation_answer_chars = observation_answer_chars

    @property
    def controller(self) -> AgentController:
        return self._controller

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def reader(self) -> Reader:
        return self._reader

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def max_evidence(self) -> int:
        return self._max_evidence

    def ask(
        self,
        query: str,
        *,
        context: ConversationContext | None = None,
        filter: RetrievalFilter | None = None,
        k: int | None = None,
        use_lexical: bool | None = None,
        use_rerank: bool | None = None,
        on_progress: ProgressCallback | None = None,
        is_cancelled: CancelledPredicate | None = None,
    ) -> AgentResult:
        """Run the agentic loop for ``query`` and return an :class:`AgentResult`.

        Parameters mirror :meth:`sparksage.qa.QAEngine.ask` (so the two engines
        are drop-in swaps at the call site) and add two long-run hooks:
        ``on_progress`` receives an :class:`AgentProgress` snapshot at each phase
        boundary, ``is_cancelled`` is polled between iterations for cooperative
        cancellation (a future job wrapper flips it on timeout / cancel).
        """
        query = str(query)
        flt = filter if filter is not None else RetrievalFilter()

        query_result = self._intercept(query, context)
        if query_result is not None and not query_result.accepted:
            _logger.info(
                "agent query rejected: intent=%s confidence=%.2f",
                query_result.intent.intent.value,
                query_result.intent.confidence,
            )
            return AgentResult(query=query, query_result=query_result)

        seed_query = self._seed_query(query, query_result)
        state = AgentState(question=query, context=context)

        # Seed retrieval: always run so the controller is consulted with at
        # least one retrieval's evidence in hand (and an empty corpus abstains
        # cleanly through the reader rather than looping on nothing).
        self._retrieve_step(
            state,
            seed_query,
            thought="seed retrieval for the user question",
            flt=flt,
            k=k,
            use_lexical=use_lexical,
            use_rerank=use_rerank,
            on_progress=on_progress,
        )

        iterations = 0
        aborted = False
        while iterations < self._max_iterations:
            if is_cancelled is not None and is_cancelled():
                _logger.info("agent run cancelled at iteration %d", iterations)
                break
            self._emit(
                on_progress,
                AgentProgress(
                    iteration=iterations + 1,
                    max_iterations=self._max_iterations,
                    phase=PHASE_THINKING,
                    evidence_count=len(state.evidence),
                ),
            )
            action = self._controller.next_action(state)
            if action.action is ActionType.SYNTHESIZE:
                self._emit(
                    on_progress,
                    AgentProgress(
                        iteration=iterations + 1,
                        max_iterations=self._max_iterations,
                        phase=PHASE_THINKING,
                        action=action.action,
                        thought=action.thought,
                        evidence_count=len(state.evidence),
                    ),
                )
                break
            iterations += 1
            self._retrieve_step(
                state,
                action.query or seed_query,
                thought=action.thought or "controller-decided retrieval",
                flt=action.filter or flt,
                k=action.k if action.k is not None else k,
                use_lexical=use_lexical,
                use_rerank=use_rerank,
                on_progress=on_progress,
                progress_iteration=iterations,
                progress_action=action,
            )
        else:
            # Reached only when ``iterations >= max_iterations`` (the loop
            # condition went false without a break). ``max_iterations == 0`` is
            # the deliberate single-shot mode, not an abort.
            aborted = self._max_iterations > 0
            if aborted:
                _logger.info(
                    "agent hit max_iterations=%d; synthesizing best-effort",
                    self._max_iterations,
                )

        self._emit(
            on_progress,
            AgentProgress(
                iteration=max(iterations, 1),
                max_iterations=self._max_iterations,
                phase=PHASE_SYNTHESIZING,
                evidence_count=len(state.evidence),
            ),
        )
        t0 = time.perf_counter()
        answer = self._reader.answer(query, list(state.evidence))
        _logger.info(
            "agent synthesized: iterations=%d evidence=%d abstained=%s "
            "aborted=%s elapsed=%.2fs",
            iterations,
            len(state.evidence),
            answer.abstained,
            aborted,
            time.perf_counter() - t0,
        )
        self._emit(
            on_progress,
            AgentProgress(
                iteration=max(iterations, 1),
                max_iterations=self._max_iterations,
                phase=PHASE_DONE,
                action=ActionType.SYNTHESIZE,
                evidence_count=len(state.evidence),
            ),
        )
        return AgentResult(
            query=query,
            answer=answer,
            steps=list(state.steps),
            iterations=iterations,
            evidence=list(state.evidence),
            aborted=aborted,
            query_result=query_result,
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _intercept(
        self,
        query: str,
        context: ConversationContext | None,
    ) -> QueryResult | None:
        if self._query_processor is None:
            return None
        return self._query_processor.process(query, context)

    @staticmethod
    def _seed_query(query: str, query_result: QueryResult | None) -> str:
        if query_result is not None and query_result.rewrite:
            rewritten = (query_result.rewrite.rewritten_query or "").strip()
            if rewritten:
                return rewritten
        return query

    def _retrieve_step(
        self,
        state: AgentState,
        sub_query: str,
        *,
        thought: str,
        flt: RetrievalFilter,
        k: int | None,
        use_lexical: bool | None,
        use_rerank: bool | None,
        on_progress: ProgressCallback | None = None,
        progress_iteration: int = 0,
        progress_action: AgentAction | None = None,
    ) -> None:
        if not str(sub_query).strip():
            return
        effective_k = self._resolved(k, self._config.k)
        effective_lexical = self._resolved(use_lexical, self._config.use_lexical)
        effective_rerank = self._resolved(use_rerank, self._config.use_rerank)
        self._emit(
            on_progress,
            AgentProgress(
                iteration=max(progress_iteration, 1),
                max_iterations=self._max_iterations,
                phase=PHASE_RETRIEVING,
                action=progress_action.action if progress_action else None,
                thought=thought,
                query=sub_query,
                evidence_count=len(state.evidence),
            ),
        )
        retrieval = self._retriever.search(
            sub_query,
            k=effective_k,
            filter=flt,
            use_lexical=effective_lexical,
            use_rerank=effective_rerank,
        )
        new_chunks = retrieval.chunks
        before = len(state.evidence)
        state.evidence = _merge_evidence(
            state.evidence, new_chunks, max_evidence=self._max_evidence
        )
        observation = _new_observation(
            new_chunks,
            top_k=self._observation_top_k,
            answer_chars=self._observation_answer_chars,
        )
        state.steps.append(
            AgentStep(
                thought=thought,
                query=sub_query,
                retrieved_count=len(new_chunks),
                observation=observation,
            )
        )
        _logger.debug(
            "agent step: query=%r hits=%d evidence=%d->%d",
            sub_query[:80],
            len(new_chunks),
            before,
            len(state.evidence),
        )

    @staticmethod
    def _resolved(call_value: object, config_value: object) -> object:
        return call_value if call_value is not None else config_value

    @staticmethod
    def _emit(
        on_progress: ProgressCallback | None,
        progress: AgentProgress,
    ) -> None:
        if on_progress is None:
            return
        try:
            on_progress(progress)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.warning("on_progress callback failed: %s", exc)


__all__ = [
    "DEFAULT_MAX_EVIDENCE",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_OBSERVATION_ANSWER_CHARS",
    "DEFAULT_OBSERVATION_TOP_K",
    "AgenticQAEngine",
    "CancelledPredicate",
    "ProgressCallback",
]
