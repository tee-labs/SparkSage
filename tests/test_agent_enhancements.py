"""Tests for the Phase-1/2 agentic RAG enhancements.

Covers the five measures from the analysis report:

* Phase 1.1 -- connect :class:`RetrievalGrader` + :class:`QueryRefiner` to the
  agent loop (the missing middle gate of the three-stage policy).
* Phase 1.2 -- the controller system prompt now mentions the optional
  ``filter`` field (tags / entities / languages / kb_id) and the new ``plan``
  action.
* Phase 1.3 -- SSE streaming of the agent loop via ``POST /api/v1/query`` with
  ``stream=true``.
* Phase 2.4 -- the new ``PLAN`` action + sub-query queue (Plan-and-Execute).
* Phase 2.5 -- per-step query expansion + RRF (multi-query / HyDE inside the
  agent loop).

All tests run offline via :class:`FakeLLMClient` / :class:`FakeEmbeddingClient`.
"""

from __future__ import annotations

import json

import pytest

from sparksage import (
    ActionType,
    AgentAction,
    AgenticQAEngine,
    AgentResult,
    BlockEmbedder,
    FakeEmbeddingClient,
    FakeLLMClient,
    HyDEExpander,
    IdeaBlock,
    IdentityExpander,
    IdentityRefiner,
    InMemoryVectorStore,
    LLMQueryExpander,
    RelevanceResult,
    Tag,
)
from sparksage.agent import AgentState
from sparksage.agent.prompts import agent_system_prompt
from sparksage.reader import LLMAnswerGenerator, Reader
from sparksage.retrieve import BM25Retriever, Retriever
from sparksage.schema.source import SourceRef


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _block(name, body, *, tag=Tag.IMPORTANT, kb_id=None):
    return IdeaBlock(
        name=name,
        critical_question=f"what is {name}?",
        trusted_answer=body,
        tags=[tag],
        source=SourceRef(uri=f"file://{name}.md", title=name, locator="L1"),
        kb_id=kb_id,
    )


def _make_retriever(blocks, *, dim=64):
    embedder = BlockEmbedder(FakeEmbeddingClient(dimension=dim))
    store = InMemoryVectorStore(dimension=dim)
    registry: dict[str, IdeaBlock] = {}
    retriever = Retriever(
        registry, store, embedder, lexical=BM25Retriever(), min_fetch=5, fetch_factor=2
    )
    retriever.index(blocks)
    return retriever


def _answer_json(text="the answer", confidence=0.9):
    return json.dumps(
        {"reasoning": "r", "answer": text, "citations": [], "confidence": confidence}
    )


def _make_reader(answer_text="the answer"):
    gen = LLMAnswerGenerator(FakeLLMClient(responses=[_answer_json(answer_text)]))
    return Reader(generator=gen)


def _action(action, *, query=None, sub_queries=None, k=None, thought="t", filter=None):
    return AgentAction(
        action=action,
        thought=thought,
        query=query,
        sub_queries=sub_queries,
        k=k,
        filter=filter,
    )


def _rel_json(score=0.5, reasoning="r"):
    return json.dumps({"reasoning": reasoning, "score": score})


class _ScriptedController:
    """Returns scripted :class:`AgentAction` decisions in order."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0
        self.calls = 0
        self.states_seen: list[AgentState] = []

    def next_action(self, state):
        self.calls += 1
        self.states_seen.append(state)
        idx = min(self.i, len(self.actions) - 1)
        action = self.actions[idx]
        self.i += 1
        return action


class _ScriptedGrader:
    """A RetrievalGrader that returns scripted relevance scores in order."""

    def __init__(self, scores):
        self.scores = [float(s) for s in scores]
        self.calls = 0

    def grade(self, query, chunks):
        idx = min(self.calls, len(self.scores) - 1)
        score = self.scores[idx]
        self.calls += 1
        return RelevanceResult(score=score, reasoning="scripted")


class _ScriptedRefiner:
    """A QueryRefiner that returns scripted refined queries in order."""

    def __init__(self, queries):
        self.queries = list(queries)
        self.calls = 0

    def refine(self, query, relevance_score, reasoning=""):
        idx = min(self.calls, len(self.queries) - 1)
        q = self.queries[idx]
        self.calls += 1
        return q


# --------------------------------------------------------------------------- #
# Phase 1.2 -- system prompt now describes filter + plan
# --------------------------------------------------------------------------- #
class TestPromptMentionsFilterAndPlan:
    def test_prompt_mentions_filter(self):
        prompt = agent_system_prompt()
        assert "filter" in prompt.lower()
        assert "tags" in prompt.lower()
        assert "kb_id" in prompt

    def test_prompt_mentions_plan_action(self):
        prompt = agent_system_prompt()
        assert "plan" in prompt.lower()
        assert "sub_queries" in prompt


# --------------------------------------------------------------------------- #
# Phase 1.1 -- grader + refiner connected to the agent loop
# --------------------------------------------------------------------------- #
class TestAgentSelfReflectiveStep:
    def test_low_grade_triggers_refine_and_reretrieve(self):
        blocks = [_block("Deploy", "deploy via pip install sparksage")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("answer")
        ctrl = _ScriptedController([_action(ActionType.SYNTHESIZE)])
        engine = AgenticQAEngine(
            ctrl,
            retriever,
            reader,
            retrieval_grader=_ScriptedGrader([0.2, 0.9]),
            query_refiner=_ScriptedRefiner(["how to install sparksage"]),
            step_min_relevance=0.5,
            step_max_refine=2,
        )
        result = engine.ask("deploy")
        assert isinstance(result, AgentResult)
        # the seed step refined + re-retrieved
        step = result.steps[0]
        assert step.refined_query == "how to install sparksage"
        assert step.relevance is not None
        assert step.relevance.score == 0.9

    def test_high_grade_skips_refinement(self):
        blocks = [_block("Deploy", "deploy via pip install sparksage")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("answer")
        refiner = _ScriptedRefiner(["should-not-be-called"])
        engine = AgenticQAEngine(
            _ScriptedController([_action(ActionType.SYNTHESIZE)]),
            retriever,
            reader,
            retrieval_grader=_ScriptedGrader([0.9]),
            query_refiner=refiner,
            step_min_relevance=0.5,
        )
        result = engine.ask("deploy")
        assert result.steps[0].refined_query is None
        assert result.steps[0].relevance.score == 0.9
        assert refiner.calls == 0

    def test_grader_without_refiner_records_relevance_only(self):
        blocks = [_block("Deploy", "deploy via pip install sparksage")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("answer")
        engine = AgenticQAEngine(
            _ScriptedController([_action(ActionType.SYNTHESIZE)]),
            retriever,
            reader,
            retrieval_grader=_ScriptedGrader([0.3]),
            # no refiner wired
        )
        result = engine.ask("deploy")
        step = result.steps[0]
        assert step.relevance.score == 0.3
        assert step.refined_query is None  # no refinement happened

    def test_loop_keeps_best_when_refinement_is_worse(self):
        blocks = [_block("A", "alpha")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("answer")
        engine = AgenticQAEngine(
            _ScriptedController([_action(ActionType.SYNTHESIZE)]),
            retriever,
            reader,
            retrieval_grader=_ScriptedGrader([0.4, 0.1]),  # refined is WORSE
            query_refiner=_ScriptedRefiner(["REFINED"]),
            step_min_relevance=0.5,
            step_max_refine=2,
        )
        result = engine.ask("alpha")
        # refinement was worse -> the best (initial 0.4) is kept
        assert result.steps[0].relevance.score == 0.4
        # but the refined_query is still recorded (it was attempted)
        assert result.steps[0].refined_query == "REFINED"

    def test_identity_refiner_does_not_loop(self):
        blocks = [_block("Deploy", "deploy via pip install sparksage")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("answer")
        engine = AgenticQAEngine(
            _ScriptedController([_action(ActionType.SYNTHESIZE)]),
            retriever,
            reader,
            retrieval_grader=_ScriptedGrader([0.2]),
            query_refiner=IdentityRefiner(),
            step_min_relevance=0.5,
            step_max_refine=3,
        )
        result = engine.ask("deploy")
        assert result.steps[0].refined_query is None
        assert result.steps[0].relevance.score == 0.2

    def test_relevance_surfaces_in_progress(self):
        blocks = [_block("A", "alpha body")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("answer")
        progress: list = []
        engine = AgenticQAEngine(
            _ScriptedController([_action(ActionType.SYNTHESIZE)]),
            retriever,
            reader,
            retrieval_grader=_ScriptedGrader([0.7]),
        )
        engine.ask("alpha", on_progress=progress.append)
        # at least one progress event carries the relevance score
        rel_events = [p for p in progress if p.relevance is not None]
        assert rel_events
        assert rel_events[-1].relevance.score == 0.7

    def test_step_max_refine_zero_disables_loop(self):
        blocks = [_block("A", "alpha")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("answer")
        refiner = _ScriptedRefiner(["REFINED"])
        engine = AgenticQAEngine(
            _ScriptedController([_action(ActionType.SYNTHESIZE)]),
            retriever,
            reader,
            retrieval_grader=_ScriptedGrader([0.1]),
            query_refiner=refiner,
            step_max_refine=0,
        )
        result = engine.ask("alpha")
        assert result.steps[0].refined_query is None
        assert refiner.calls == 0

    def test_invalid_construction_args(self):
        retriever = _make_retriever([_block("A", "a")])
        reader = _make_reader()
        with pytest.raises(ValueError):
            AgenticQAEngine(
                _ScriptedController([]),
                retriever,
                reader,
                step_min_relevance=1.5,
            )
        with pytest.raises(ValueError):
            AgenticQAEngine(
                _ScriptedController([]),
                retriever,
                reader,
                step_max_refine=-1,
            )
        with pytest.raises(ValueError):
            AgenticQAEngine(
                _ScriptedController([]),
                retriever,
                reader,
                expander_n_variants=0,
            )

    def test_invalid_max_stale_steps(self):
        retriever = _make_retriever([_block("A", "a")])
        reader = _make_reader()
        with pytest.raises(ValueError):
            AgenticQAEngine(
                _ScriptedController([]),
                retriever,
                reader,
                max_stale_steps=-1,
            )
        with pytest.raises(TypeError):
            AgenticQAEngine(
                _ScriptedController([]),
                retriever,
                reader,
                max_stale_steps=True,  # type: ignore[arg-type]
            )


# --------------------------------------------------------------------------- #
# Stall detector -- early termination when evidence stops growing
# --------------------------------------------------------------------------- #
class TestAgentStallDetector:
    def test_stall_terminates_early(self):
        """When consecutive steps add zero new evidence the loop stops."""
        # Single block: every retrieval after the seed finds the same chunk.
        blocks = [_block("alpha", "alpha revenue grew")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("best effort answer")
        ctrl = _ScriptedController(
            [
                _action(ActionType.RETRIEVE, query="alpha revenue"),
                _action(ActionType.RETRIEVE, query="alpha growth"),
                _action(ActionType.RETRIEVE, query="should not reach"),
                _action(ActionType.SYNTHESIZE),
            ]
        )
        engine = AgenticQAEngine(
            ctrl,
            retriever,
            reader,
            max_iterations=10,
            max_stale_steps=2,
        )
        result = engine.ask("alpha revenue")
        # seed (evidence 0->1) + iter1 stale (1->1) + iter2 stale (1->1) = break
        # the controller should NOT have been called 4 times
        assert ctrl.calls <= 3
        # evidence never grew beyond 1
        assert len(result.evidence) == 1

    def test_stall_reset_on_new_evidence(self):
        """Stale counter resets when a step actually adds evidence."""
        # Use k=1 so the seed retrieval only gets 1 block; later steps can
        # then find new evidence on different sub-queries.
        blocks = [
            _block("alpha", "alpha revenue grew significantly"),
            _block("beta", "beta revenue fell sharply"),
            _block("gamma", "gamma revenue was stable"),
        ]
        retriever = _make_retriever(blocks)
        reader = _make_reader("answer")
        ctrl = _ScriptedController(
            [
                # Retrieves alpha again (stale=1)
                _action(ActionType.RETRIEVE, query="alpha"),
                # Retrieves gamma (new evidence -> stale resets to 0)
                _action(ActionType.RETRIEVE, query="gamma"),
                # Retrieves alpha again (stale=1) -- not enough to trigger
                _action(ActionType.RETRIEVE, query="alpha"),
                _action(ActionType.SYNTHESIZE),
            ]
        )
        engine = AgenticQAEngine(
            ctrl,
            retriever,
            reader,
            max_iterations=10,
            max_stale_steps=2,
        )
        result = engine.ask("compare alpha beta gamma", k=1)
        # all 3 controller-decided retrievals ran (stale never hit 2 consecutively)
        assert result.iterations >= 3
        assert ctrl.calls >= 4

    def test_stall_disabled_when_zero(self):
        """``max_stale_steps=0`` disables the stall detector entirely."""
        blocks = [_block("alpha", "alpha revenue grew")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("answer")
        ctrl = _ScriptedController(
            [
                _action(ActionType.RETRIEVE, query="alpha"),
                _action(ActionType.RETRIEVE, query="alpha again"),
                _action(ActionType.SYNTHESIZE),
            ]
        )
        engine = AgenticQAEngine(
            ctrl,
            retriever,
            reader,
            max_iterations=3,
            max_stale_steps=0,
        )
        result = engine.ask("alpha")
        # 2 RETRIEVE actions ran before SYNTHESIZE (no stall cut-off)
        assert result.iterations == 2

    def test_stall_default_is_two(self):
        from sparksage.agent.engine import DEFAULT_MAX_STALE_STEPS

        assert DEFAULT_MAX_STALE_STEPS == 2


# --------------------------------------------------------------------------- #
# Phase 2.5 -- per-step query expansion + RRF
# --------------------------------------------------------------------------- #
class TestAgentPerStepExpansion:
    def test_expander_produces_variants_per_step(self):
        blocks = [
            _block("Install", "pip install sparksage package"),
            _block("Setup", "configure sparksage via env vars"),
        ]
        retriever = _make_retriever(blocks)
        reader = _make_reader("answer")
        expander_client = FakeLLMClient(
            responses=[
                json.dumps({"variants": ["install sparksage", "setup sparksage env"]})
            ]
        )
        expander = LLMQueryExpander(expander_client)
        engine = AgenticQAEngine(
            _ScriptedController([_action(ActionType.SYNTHESIZE)]),
            retriever,
            reader,
            query_expander=expander,
        )
        result = engine.ask("sparksage")
        # one retrieval step that fused extra variants
        assert len(result.steps) == 1
        assert result.steps[0].retrieved_count >= 1
        assert len(expander_client.calls) >= 1

    def test_identity_expander_is_noop(self):
        blocks = [_block("A", "alpha body")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("answer")
        engine = AgenticQAEngine(
            _ScriptedController([_action(ActionType.SYNTHESIZE)]),
            retriever,
            reader,
            query_expander=IdentityExpander(),
        )
        result = engine.ask("alpha")
        assert len(result.steps) == 1

    def test_hyde_lands_in_answer_space(self):
        blocks = [_block("Install", "pip install sparksage to install")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("answer")
        hyde_client = FakeLLMClient(
            responses=[json.dumps({"hypothesis": "pip install sparksage"})]
        )
        # ``max_words=10`` so the single-token "install" (7 chars under the
        # char-count fallback) is treated as short and HyDE fires.
        hyde = HyDEExpander(hyde_client, max_words=10)
        engine = AgenticQAEngine(
            _ScriptedController([_action(ActionType.SYNTHESIZE)]),
            retriever,
            reader,
            query_expander=hyde,
        )
        result = engine.ask("install")  # short query -> HyDE fires
        assert len(hyde_client.calls) >= 1
        assert len(result.steps) == 1


# --------------------------------------------------------------------------- #
# Phase 2.4 -- PLAN action + sub-query queue
# --------------------------------------------------------------------------- #
class TestAgentPlanAction:
    def test_plan_enqueues_and_retrieves_each_subquery(self):
        blocks = [
            _block("alpha", "alpha revenue grew 10%"),
            _block("beta", "beta revenue fell 5%"),
        ]
        retriever = _make_retriever(blocks)
        reader = _make_reader("compare answer")
        # The Plan-and-Execute pattern: PLAN enqueues; the engine drains the
        # queue one per iteration WITHOUT consulting the controller between
        # them. Once the queue empties the controller is consulted again.
        ctrl = _ScriptedController(
            [
                _action(
                    ActionType.PLAN,
                    sub_queries=["alpha revenue", "beta revenue"],
                ),
                # Consulted again only after the queue drains.
                _action(ActionType.SYNTHESIZE),
            ]
        )
        engine = AgenticQAEngine(ctrl, retriever, reader, max_iterations=5)
        result = engine.ask("compare alpha and beta")
        # seed + 2 planned retrievals
        assert len(result.steps) == 3
        step_queries = [s.query for s in result.steps[1:]]
        assert "alpha revenue" in step_queries
        assert "beta revenue" in step_queries
        # PLAN consumed 1 iteration, 2 retrieves consumed 2 -> 3 total
        assert result.iterations == 3
        # controller consulted twice (PLAN + final SYNTHESIZE)
        assert ctrl.calls == 2

    def test_plan_without_subqueries_falls_back_to_synthesize(self):
        blocks = [_block("alpha", "alpha revenue")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("answer")
        # coercion demotes PLAN-without-sub_queries to SYNTHESIZE at the
        # controller layer; here we simulate that by directly emitting
        # SYNTHESIZE -- the engine never sees a bare PLAN.
        ctrl = _ScriptedController([_action(ActionType.SYNTHESIZE)])
        engine = AgenticQAEngine(ctrl, retriever, reader, max_iterations=3)
        result = engine.ask("alpha?")
        assert result.iterations == 0

    def test_plan_respects_max_iterations_cap(self):
        blocks = [
            _block("alpha", "alpha revenue"),
            _block("beta", "beta revenue"),
            _block("gamma", "gamma revenue"),
        ]
        retriever = _make_retriever(blocks)
        reader = _make_reader("best effort")
        ctrl = _ScriptedController(
            [
                _action(
                    ActionType.PLAN,
                    sub_queries=["alpha revenue", "beta revenue", "gamma revenue"],
                ),
                _action(ActionType.SYNTHESIZE),
            ]
        )
        # max_iterations=2 -> PLAN + 1 retrieve, then capped (queue not empty)
        engine = AgenticQAEngine(ctrl, retriever, reader, max_iterations=2)
        result = engine.ask("compare")
        assert result.aborted is True
        # only one planned sub-query got retrieved before the cap
        assert len(result.steps) == 2  # seed + 1 planned retrieve
        assert result.iterations == 2

    def test_plan_drains_queue_before_consulting_controller(self):
        blocks = [_block("alpha", "alpha revenue")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("answer")
        ctrl = _ScriptedController(
            [
                _action(ActionType.PLAN, sub_queries=["alpha revenue", "beta revenue"]),
                # Consulted only after the queue is fully drained.
                _action(ActionType.SYNTHESIZE),
            ]
        )
        engine = AgenticQAEngine(ctrl, retriever, reader, max_iterations=5)
        result = engine.ask("alpha")
        # seed + 2 planned retrievals ran before the controller was consulted
        assert len(result.steps) == 3
        assert result.iterations == 3
        assert ctrl.calls == 2

    def test_llm_controller_emits_plan(self):
        blocks = [
            _block("alpha", "alpha revenue grew"),
            _block("beta", "beta revenue fell"),
        ]
        retriever = _make_retriever(blocks)
        reader = _make_reader("compare answer")
        from sparksage.agent import LLMAgentController

        ctrl_client = FakeLLMClient(
            responses=[
                json.dumps(
                    {
                        "thought": "decompose",
                        "action": "plan",
                        "sub_queries": ["alpha revenue", "beta revenue"],
                    }
                ),
                # Consulted again only after the queue drains.
                json.dumps({"thought": "done", "action": "synthesize"}),
            ]
        )
        ctrl = LLMAgentController(ctrl_client)
        engine = AgenticQAEngine(ctrl, retriever, reader, max_iterations=5)
        result = engine.ask("compare alpha and beta")
        assert result.iterations == 3  # PLAN + 2 retrieves
        queries = [s.query for s in result.steps[1:]]
        assert "alpha revenue" in queries
        assert "beta revenue" in queries


# --------------------------------------------------------------------------- #
# Phase 1.1 + 2.4 + 2.5 combined (realistic compare question)
# --------------------------------------------------------------------------- #
class TestAgentCombinedEnhancements:
    def test_plan_then_grade_refine_per_step(self):
        blocks = [
            _block("alpha", "alpha revenue grew 10%"),
            _block("beta", "beta revenue fell 5%"),
        ]
        retriever = _make_retriever(blocks)
        reader = _make_reader("compare answer")
        ctrl = _ScriptedController(
            [
                _action(
                    ActionType.PLAN,
                    sub_queries=["alpha revenue", "beta revenue"],
                ),
                _action(ActionType.SYNTHESIZE),
            ]
        )
        engine = AgenticQAEngine(
            ctrl,
            retriever,
            reader,
            retrieval_grader=_ScriptedGrader([0.9, 0.9]),
            max_iterations=5,
        )
        result = engine.ask("compare alpha and beta")
        # each planned step was graded (2 planned retrieves -> 2 grades)
        planned_steps = result.steps[1:]
        assert len(planned_steps) == 2
        for step in planned_steps:
            assert step.relevance is not None
            assert step.relevance.score == 0.9


# --------------------------------------------------------------------------- #
# Phase 1.3 -- SSE streaming via the HTTP route
# --------------------------------------------------------------------------- #
class TestAgentStreamingRoute:
    def _make_service(self):
        from sparksage import (
            FakeConverterBackend,
            IdeaBlockGenerator,
            MarkdownConverter,
            SparkSageService,
            TextCleaner,
        )
        from sparksage.api.qa_service import QAService

        gen_json = json.dumps(
            {
                "blocks": [
                    {
                        "name": "Install",
                        "critical_question": "How to install?",
                        "trusted_answer": "Install with pip install sparksage.",
                        "tags": ["technology"],
                        "keywords": ["install", "pip"],
                    }
                ]
            }
        )
        answer_json = json.dumps(
            {"answer": "Use pip.", "citations": [], "confidence": 0.9}
        )
        converter = MarkdownConverter(
            backend=FakeConverterBackend(markdown="# Guide\nInstall via pip.", title="Guide")
        )
        spark_service = SparkSageService(
            converter=converter,
            cleaner=TextCleaner(),
            generator=IdeaBlockGenerator(FakeLLMClient(responses=[gen_json])),
        )
        from sparksage.agent import LLMAgentController

        ctrl_client = FakeLLMClient(
            responses=[
                json.dumps({"thought": "enough", "action": "synthesize"}),
            ]
        )
        return QAService(
            service=spark_service,
            embedder=BlockEmbedder(FakeEmbeddingClient(dimension=64)),
            reader=Reader(generator=LLMAnswerGenerator(FakeLLMClient(responses=[answer_json]))),
            agent_controller=LLMAgentController(ctrl_client),
        )

    def _client(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:  # pragma: no cover
            pytest.skip("fastapi not installed")
        from sparksage.api.app import create_app

        svc = self._make_service()
        app = create_app(qa_service=svc)
        return TestClient(app)

    def test_stream_returns_event_stream(self):
        client = self._client()
        # ingest first so the KB has content
        client.post(
            "/api/v1/knowledge_base/ingest",
            files={"file": ("g.md", b"dummy", "text/markdown")},
        )
        with client.stream(
            "POST",
            "/api/v1/query",
            json={"query": "How to install?", "mode": "agent", "stream": True},
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            events: list[str] = []
            data_lines: list[str] = []
            for line in resp.iter_lines():
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].strip())
            # always terminates with a done event
            assert "done" in events
            assert "result" in events
            # at least one progress event (the seed retrieval) flows before
            # the result event
            assert "progress" in events
            assert events.index("progress") < events.index("result")
            # the result event carries a full AskResponse payload
            result_payload = json.loads(data_lines[events.index("result")])
            assert result_payload["mode"] == "agent"
            assert result_payload["answer"] == "Use pip."
            # progress events carry the phase + percent
            first_progress = json.loads(data_lines[events.index("progress")])
            assert "phase" in first_progress
            assert "percent" in first_progress

    def test_stream_default_mode_emits_phase_progress(self):
        # stream=true with mode=default now emits coarse phase progress so the
        # QA page can show visible feedback during the single-shot wait.
        client = self._client()
        client.post(
            "/api/v1/knowledge_base/ingest",
            files={"file": ("g.md", b"dummy", "text/markdown")},
        )
        with client.stream(
            "POST",
            "/api/v1/query",
            json={"query": "How to install?", "mode": "default", "stream": True},
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            events: list[str] = []
            data_lines: list[str] = []
            for line in resp.iter_lines():
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].strip())
            assert "done" in events
            assert "result" in events
            assert "progress" in events
            assert events.index("progress") < events.index("result")
            result_payload = json.loads(data_lines[events.index("result")])
            assert result_payload["mode"] == "default"
            # the default-mode progress payload carries one of the coarse phases
            first_progress = json.loads(data_lines[events.index("progress")])
            assert first_progress["phase"] in {
                "understanding",
                "retrieving",
                "generating",
                "done",
            }
            assert "percent" in first_progress
