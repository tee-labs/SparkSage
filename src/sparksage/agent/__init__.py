"""Agentic QA: a plan-act-observe-synthesize loop over the QA core.

This is the agentic counterpart of :mod:`sparksage.qa`. Where
:class:`~sparksage.qa.QAEngine` runs a *fixed* retrieve-once pipeline (one-shot
RAG -- a static question->answer mapping), :class:`AgenticQAEngine` runs an
LLM-driven *control loop*: the :class:`AgentController` decomposes a complex /
multi-hop / comparative question into a sequence of focused retrievals,
gathering evidence until it judges it sufficient, then synthesizes one grounded
answer. That is the planning-execution-reflection cycle that defines Agentic
RAG -- the mode that handles the three problem classes one-shot RAG cannot:
multi-hop reasoning, conditional filtering, and comparative analysis.

The agent is deliberately a *different orchestrator* over the *same* building
blocks -- it reuses :class:`~sparksage.retrieve.Retriever`,
:class:`~sparksage.reader.Reader`, :class:`~sparksage.query.QueryProcessor` and
the existing :class:`~sparksage.generator.LLMClient` unchanged, and adds exactly
one new protocol (:class:`AgentController`). So the core stays zero-dependency
and fully unit-testable offline (under :class:`~sparksage.generator.FakeLLMClient`
/ :class:`~sparksage.embed.FakeEmbeddingClient`), exactly like the rest of the
right half. An :class:`AgentResult` exposes the same surface as
:class:`~sparksage.qa.QAResult`, so the two engines are interchangeable behind
the API's ``AskResponse`` serializer -- ``POST /api/v1/query`` selects the mode
via ``AskRequest.mode`` ("default" | "agent").

The loop reuses the canonical long-running-job shape
(``max_iterations`` / ``on_progress`` / ``is_cancelled``) from
:class:`~sparksage.distill.DistillPipeline`, so a future
``/api/v1/query/agent`` route can wrap a run in a pollable job (mirroring the
planned ``/api/v1/distill`` route) with no new concurrency primitives.

Pipeline::

    question
        -> QueryProcessor intercept            [optional, out-of-domain gate]
        -> seed retrieval                      (always: guarantees evidence)
        -> loop (bounded by max_iterations):
              AgentController.next_action      (ReAct: retrieve more | synthesize)
              if retrieve: Retriever.search -> merge evidence -> record step
              if synthesize / cancelled / capped: break
        -> Reader.answer(question, evidence)   (generate -> judge -> abstain)

Example
-------

::

    from sparksage import FakeLLMClient
    from sparksage.agent import AgenticQAEngine, LLMAgentController

    controller = LLMAgentController(FakeLLMClient(responses=[
        '{"thought":"need B too","action":"retrieve","query":"B revenue"}',
        '{"thought":"enough","action":"synthesize"}',
    ]))
    engine = AgenticQAEngine(controller, retriever=retriever, reader=reader)
    result = engine.ask("compare A and B revenue")
    print(result.text, "in", result.iterations, "extra steps")
"""

from sparksage.agent.controller import (
    ActionEmptyResponseError,
    ActionResponseParseError,
    AgentController,
    AgentError,
    IdentityController,
    LLMAgentController,
)
from sparksage.agent.engine import (
    DEFAULT_EXPANDER_N_VARIANTS,
    DEFAULT_MAX_EVIDENCE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_STEP_MAX_REFINE,
    DEFAULT_STEP_MIN_RELEVANCE,
    AgenticQAEngine,
)
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
from sparksage.agent.prompts import (
    DEFAULT_EVIDENCE_ANSWER_CHARS,
    DEFAULT_EVIDENCE_TOP_K,
    agent_messages,
    agent_system_prompt,
    agent_user_prompt,
)
from sparksage.agent.schema import (
    DEFAULT_ACTION,
    CoercionError,
    RawAgentAction,
    coerce_action,
    extract_json,
    parse_action_response,
    parse_raw_action,
)

__all__ = [
    "ActionEmptyResponseError",
    "ActionResponseParseError",
    "ActionType",
    "AgentAction",
    "AgentController",
    "AgentError",
    "AgentProgress",
    "AgentResult",
    "AgentState",
    "AgentStep",
    "AgenticQAEngine",
    "CoercionError",
    "DEFAULT_ACTION",
    "DEFAULT_EVIDENCE_ANSWER_CHARS",
    "DEFAULT_EVIDENCE_TOP_K",
    "DEFAULT_EXPANDER_N_VARIANTS",
    "DEFAULT_MAX_EVIDENCE",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_STEP_MAX_REFINE",
    "DEFAULT_STEP_MIN_RELEVANCE",
    "IdentityController",
    "LLMAgentController",
    "PHASE_DONE",
    "PHASE_RETRIEVING",
    "PHASE_SYNTHESIZING",
    "PHASE_THINKING",
    "RawAgentAction",
    "agent_messages",
    "agent_system_prompt",
    "agent_user_prompt",
    "coerce_action",
    "extract_json",
    "parse_action_response",
    "parse_raw_action",
]
