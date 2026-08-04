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
from collections import deque
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
from sparksage.query.expander import IdentityExpander, QueryExpander
from sparksage.query.processor import QueryProcessor, QueryResult
from sparksage.query.refiner import IdentityRefiner, QueryRefiner
from sparksage.reader.orchestrator import Reader
from sparksage.retrieve.grader import RetrievalGrader
from sparksage.retrieve.models import RetrievalFilter, RetrievalResult, RetrievedChunk
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

#: Default per-step relevance floor below which the agent refines + re-retrieves
#: (the missing middle gate of the three-stage policy -- mirrors
#: :data:`sparksage.qa.DEFAULT_MIN_RELEVANCE`).
DEFAULT_STEP_MIN_RELEVANCE = 0.5

#: Default cap on per-step refine + re-retrieve rounds (caps latency / cost).
DEFAULT_STEP_MAX_REFINE = 1

#: Default number of variants an optional :class:`QueryExpander` produces for
#: each per-step sub-query (RRF-fused recall boost).
DEFAULT_EXPANDER_N_VARIANTS = 3

#: Default cap on consecutive controller-decided steps that add **zero** new
#: evidence chunks. When the loop stalls (every retrieval returns blocks already
#: in the evidence pool), further iterations are futile -- the corpus simply
#: does not contain more relevant material. ``0`` disables the stall detector.
DEFAULT_MAX_STALE_STEPS = 2

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
    retrieval_grader:
        Optional :class:`~sparksage.retrieve.RetrievalGrader`. When wired, each
        retrieval step is graded for relevance; a low score triggers the
        per-step refine + re-retrieve loop (the missing middle gate of the
        three-stage policy -- finally connecting the existing grader to the
        agent loop).
    query_refiner:
        Optional :class:`~sparksage.query.QueryRefiner`. When wired alongside a
        grader, a low step relevance produces a refined query and the step
        re-retrieves (up to ``step_max_refine`` rounds), keeping the best-graded
        result so refinement can never lower quality.
    query_expander:
        Optional :class:`~sparksage.query.QueryExpander`. When wired, each
        retrieval step expands its sub-query into ``expander_n_variants``
        paraphrases and RRF-fuses the per-variant ranked lists -- the multi-query
        recall boost (HyDE is especially effective here, since IdeaBlock is
        embedded from ``trusted_answer`` and a hypothetical answer lands in the
        answer semantic space).
    step_min_relevance:
        Per-step relevance floor below which the grader-triggered refine loop
        fires (default :data:`DEFAULT_STEP_MIN_RELEVANCE`).
    step_max_refine:
        Cap on per-step refine + re-retrieve rounds (default
        :data:`DEFAULT_STEP_MAX_REFINE`). ``0`` disables refinement even when a
        grader is wired (steps are graded but never re-retrieved).
    expander_n_variants:
        How many variants the per-step expansion produces (default
        :data:`DEFAULT_EXPANDER_N_VARIANTS`).
    max_iterations:
        Maximum number of *extra* controller-decided steps (``PLAN`` or
        ``RETRIEVE``) beyond the seed (default ``4``). ``0`` collapses to a
        single-shot run regardless of the controller.
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
        retrieval_grader: RetrievalGrader | None = None,
        query_refiner: QueryRefiner | None = None,
        query_expander: QueryExpander | None = None,
        step_min_relevance: float = DEFAULT_STEP_MIN_RELEVANCE,
        step_max_refine: int = DEFAULT_STEP_MAX_REFINE,
        expander_n_variants: int = DEFAULT_EXPANDER_N_VARIANTS,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_evidence: int = DEFAULT_MAX_EVIDENCE,
        config: RetrievalConfig | None = None,
        observation_top_k: int = DEFAULT_OBSERVATION_TOP_K,
        observation_answer_chars: int = DEFAULT_OBSERVATION_ANSWER_CHARS,
        max_stale_steps: int = DEFAULT_MAX_STALE_STEPS,
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
        if retrieval_grader is not None and not isinstance(
            retrieval_grader, RetrievalGrader
        ):
            raise TypeError(
                "retrieval_grader must implement the RetrievalGrader protocol"
            )
        if query_refiner is not None and not isinstance(query_refiner, QueryRefiner):
            raise TypeError("query_refiner must implement the QueryRefiner protocol")
        if query_expander is not None and not isinstance(query_expander, QueryExpander):
            raise TypeError("query_expander must implement the QueryExpander protocol")
        if not 0.0 <= float(step_min_relevance) <= 1.0:
            raise ValueError("step_min_relevance must be in [0.0, 1.0]")
        if not isinstance(step_max_refine, int) or isinstance(step_max_refine, bool):
            raise TypeError("step_max_refine must be an int")
        if step_max_refine < 0:
            raise ValueError("step_max_refine must be >= 0")
        if not isinstance(expander_n_variants, int) or isinstance(expander_n_variants, bool):
            raise TypeError("expander_n_variants must be an int")
        if expander_n_variants < 1:
            raise ValueError("expander_n_variants must be >= 1")
        if not isinstance(max_stale_steps, int) or isinstance(max_stale_steps, bool):
            raise TypeError("max_stale_steps must be an int")
        if max_stale_steps < 0:
            raise ValueError("max_stale_steps must be >= 0")
        self._controller = controller
        self._retriever = retriever
        self._reader = reader
        self._query_processor = query_processor
        self._retrieval_grader = retrieval_grader
        self._query_refiner = query_refiner
        self._query_expander = query_expander
        self._step_min_relevance = float(step_min_relevance)
        self._step_max_refine = step_max_refine
        self._expander_n_variants = expander_n_variants
        self._max_iterations = max_iterations
        self._max_evidence = max_evidence
        self._config = config if config is not None else RetrievalConfig()
        self._observation_top_k = observation_top_k
        self._observation_answer_chars = observation_answer_chars
        self._max_stale_steps = max_stale_steps

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
    def retrieval_grader(self) -> RetrievalGrader | None:
        return self._retrieval_grader

    @property
    def query_refiner(self) -> QueryRefiner | None:
        return self._query_refiner

    @property
    def query_expander(self) -> QueryExpander | None:
        return self._query_expander

    @property
    def step_min_relevance(self) -> float:
        return self._step_min_relevance

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def max_evidence(self) -> int:
        return self._max_evidence

    @property
    def max_stale_steps(self) -> int:
        return self._max_stale_steps

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
        stale_steps = 0
        # Sub-queries enqueued by a PLAN action -- drained one per iteration
        # through the same retrieve path (graded / refined / expanded like any
        # step) WITHOUT consulting the controller between them, so a single
        # PLAN commits the engine to retrieving every queued sub-query (up to
        # ``max_iterations``). Once the queue empties the controller is
        # consulted again and may re-plan, retrieve more, or synthesize. This
        # is the Plan-and-Execute pattern: plan once -> execute -> re-plan.
        pending: deque[str] = deque()
        while iterations < self._max_iterations:
            if is_cancelled is not None and is_cancelled():
                _logger.info("agent run cancelled at iteration %d", iterations)
                break
            if (
                self._max_stale_steps > 0
                and stale_steps >= self._max_stale_steps
            ):
                _logger.info(
                    "agent stalled: %d consecutive steps added no evidence; "
                    "synthesizing best-effort",
                    stale_steps,
                )
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
            if pending:
                # Drain the next planned sub-query without consulting the
                # controller -- the plan already committed to it.  Planned
                # steps are excluded from the stall detector: a PLAN action
                # is a committed decomposition, not an open-ended probe, so
                # every queued sub-query deserves its retrieval even if the
                # seed already covered it.
                sub = pending.popleft()
                iterations += 1
                self._retrieve_step(
                    state,
                    sub,
                    thought="planned sub-query",
                    flt=flt,
                    k=k,
                    use_lexical=use_lexical,
                    use_rerank=use_rerank,
                    on_progress=on_progress,
                    progress_iteration=iterations,
                    progress_action=None,
                )
                continue
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
            if action.action is ActionType.PLAN:
                # Enqueue the decomposition; do not retrieve this iteration.
                # ``sub_queries`` is always populated here (the coercion rule
                # demotes a plan-without-sub_queries to synthesize).
                new_subs = [q for q in (action.sub_queries or []) if q]
                before = len(pending)
                pending.extend(new_subs)
                _logger.info(
                    "agent plan: enqueued %d sub-queries (queue %d->%d)",
                    len(new_subs),
                    before,
                    len(pending),
                )
                self._emit(
                    on_progress,
                    AgentProgress(
                        iteration=iterations,
                        max_iterations=self._max_iterations,
                        phase=PHASE_THINKING,
                        action=action.action,
                        thought=action.thought or "decomposed question into sub-queries",
                        evidence_count=len(state.evidence),
                    ),
                )
                continue
            # RETRIEVE (controller-decided, not plan-driven).
            ev_before = len(state.evidence)
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
            stale_steps = stale_steps + 1 if len(state.evidence) == ev_before else 0
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
        """One retrieval step: expand -> retrieve -> grade -> refine -> merge.

        Mirrors the per-step recipe of the single-shot QA engine's
        self-reflective loop, inlined so the agent loop finally consumes the
        existing reflection components:

        * **expand** (optional): when a :class:`QueryExpander` is wired, expand
          the sub-query into ``n`` variants and RRF-fuse them (the multi-query
          recall boost).
        * **retrieve**: one plain :meth:`Retriever.search`, or a fused
          multi-query retrieval when expansion produced variants.
        * **grade + refine** (optional): when a :class:`RetrievalGrader` is
          wired, score the retrieval; below ``step_min_relevance`` use a wired
          :class:`QueryRefiner` to rewrite and re-retrieve (up to
          ``step_max_refine`` rounds, keeping the best-graded result so
          refinement can never lower quality). The best result's relevance is
          recorded on the :class:`AgentStep` for trajectory transparency.
        """
        if not str(sub_query).strip():
            return
        effective_k = self._resolved_int(k, self._config.k)
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
        current_query = sub_query
        retrieval = self._retrieve_with_expand(
            current_query,
            k=effective_k,
            flt=flt,
            use_lexical=effective_lexical,
            use_rerank=effective_rerank,
        )
        relevance = self._maybe_grade(current_query, retrieval, None)

        # Per-step self-reflective refine + re-retrieve loop (CRAG / Self-RAG
        # ``ISREL`` gate, finally connected to the agent loop). Keeps the
        # best-graded retrieval so refinement can never lower quality.
        refined_query: str | None = None
        can_refine = (
            self._retrieval_grader is not None
            and self._query_refiner is not None
            and not isinstance(self._query_refiner, IdentityRefiner)
            and self._step_max_refine > 0
        )
        if can_refine and relevance is not None:
            rounds = 0
            best_retrieval, best_relevance = retrieval, relevance
            while (
                best_relevance.score < self._step_min_relevance
                and rounds < self._step_max_refine
            ):
                refined = self._query_refiner.refine(
                    current_query, best_relevance.score, best_relevance.reasoning
                )
                refined = (refined or "").strip()
                rounds += 1
                if not refined or refined.lower() == current_query.strip().lower():
                    break
                refined_query = refined
                cand_retrieval = self._retrieve_with_expand(
                    refined,
                    k=effective_k,
                    flt=flt,
                    use_lexical=effective_lexical,
                    use_rerank=effective_rerank,
                )
                cand_relevance = self._maybe_grade(refined, cand_retrieval, None)
                if cand_relevance is not None and cand_relevance.score > best_relevance.score:
                    best_retrieval, best_relevance = cand_retrieval, cand_relevance
                    current_query = refined
                    retrieval = best_retrieval
                    relevance = best_relevance
                if best_relevance.score >= self._step_min_relevance:
                    break

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
                relevance=relevance,
                refined_query=refined_query,
            )
        )
        if relevance is not None or refined_query is not None:
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
                    relevance=relevance,
                    refined_query=refined_query,
                ),
            )
        _logger.debug(
            "agent step: query=%r hits=%d evidence=%d->%d relevance=%s refined=%r",
            sub_query[:80],
            len(new_chunks),
            before,
            len(state.evidence),
            f"{relevance.score:.2f}" if relevance is not None else None,
            refined_query,
        )

    def _retrieve_with_expand(
        self,
        query: str,
        *,
        k: int,
        flt: RetrievalFilter,
        use_lexical: bool,
        use_rerank: bool,
    ) -> RetrievalResult:
        """Run one retrieval, optionally multi-query expanded + RRF-fused."""
        if self._query_expander is not None and not isinstance(
            self._query_expander, IdentityExpander
        ):
            from sparksage.retrieve.multi_query import multi_query_retrieve

            variants = self._query_expander.expand(query, n=self._expander_n_variants)
            if len(variants) > 1:
                extra = [v for v in variants[1:] if v and v != variants[0]]
                return multi_query_retrieve(
                    self._retriever,
                    variants[0],
                    extra,
                    k=k,
                    filter=flt,
                    use_lexical=use_lexical,
                    use_rerank=use_rerank,
                )
        return self._retriever.search(
            query,
            k=k,
            filter=flt,
            use_lexical=use_lexical,
            use_rerank=use_rerank,
        )

    def _maybe_grade(
        self,
        query: str,
        retrieval: RetrievalResult,
        existing: object,
    ) -> object:
        """Grade the retrieval when a grader is wired (no-op otherwise)."""
        if self._retrieval_grader is None:
            return existing
        try:
            return self._retrieval_grader.grade(query, retrieval.chunks)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.warning("RetrievalGrader raised %s; skipping grade", exc)
            return existing

    @staticmethod
    def _resolved(call_value: object, config_value: object) -> object:
        return call_value if call_value is not None else config_value

    @staticmethod
    def _resolved_int(call_value: int | None, config_value: int) -> int:
        if call_value is None:
            return config_value
        if not isinstance(call_value, int) or isinstance(call_value, bool):
            return config_value
        return call_value if call_value >= 1 else config_value

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
    "DEFAULT_EXPANDER_N_VARIANTS",
    "DEFAULT_MAX_EVIDENCE",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_STALE_STEPS",
    "DEFAULT_OBSERVATION_ANSWER_CHARS",
    "DEFAULT_OBSERVATION_TOP_K",
    "DEFAULT_STEP_MAX_REFINE",
    "DEFAULT_STEP_MIN_RELEVANCE",
    "AgenticQAEngine",
    "CancelledPredicate",
    "ProgressCallback",
]
