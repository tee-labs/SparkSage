"""Tests for the agentic QA core (:mod:`sparksage.agent`).

Covers:
* lenient -> strict action coercion (aliases, k clamping, fallbacks, strict).
* :class:`IdentityController` (degenerate single-step baseline).
* :class:`AgenticQAEngine` end-to-end: seed retrieval, multi-step loop,
  ``max_iterations`` abort, empty-corpus abstention, ``on_progress`` /
  ``is_cancelled`` hooks, query-processor interception.
* :class:`AgentResult` surface compatibility with the HTTP ``AskResponse``
  serializer (:func:`sparksage.api.schemas._to_ask_response`).
* :class:`LLMAgentController` driving a real multi-step loop via
  :class:`FakeLLMClient`.

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
    IdeaBlock,
    IdentityController,
    InMemoryVectorStore,
    LLMAgentController,
    QueryIntent,
    QueryProcessor,
    Tag,
)
from sparksage.agent import AgentState
from sparksage.agent.schema import (
    CoercionError,
    RawAgentAction,
    coerce_action,
    extract_json,
    parse_action_response,
)
from sparksage.reader import LLMAnswerGenerator, Reader
from sparksage.retrieve import BM25Retriever, Retriever
from sparksage.schema.source import SourceRef

try:
    from sparksage import LLMIntentClassifier, LLMQueryRewriter
except ImportError:  # pragma: no cover
    LLMIntentClassifier = None  # type: ignore[assignment]
    LLMQueryRewriter = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _block(name, body, *, tag=Tag.IMPORTANT):
    return IdeaBlock(
        name=name,
        critical_question=f"what is {name}?",
        trusted_answer=body,
        tags=[tag],
        source=SourceRef(uri=f"file://{name}.md", title=name, locator="L1"),
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


def _action(action, *, query=None, k=None, thought="t"):
    return AgentAction(action=action, thought=thought, query=query, k=k)


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


# --------------------------------------------------------------------------- #
# schema: lenient -> strict coercion
# --------------------------------------------------------------------------- #
class TestActionCoercion:
    def test_retrieve_alias_and_query(self):
        act = coerce_action(
            RawAgentAction(thought="x", action="search", query="sub q", k=5),
            strict=False,
        )
        assert act.action is ActionType.RETRIEVE
        assert act.query == "sub q"
        assert act.k == 5

    @pytest.mark.parametrize("alias", ["retrieve", "search", "lookup", "query"])
    def test_retrieve_aliases(self, alias):
        act = coerce_action(RawAgentAction(action=alias, query="q"), strict=False)
        assert act.action is ActionType.RETRIEVE

    @pytest.mark.parametrize(
        "alias", ["synthesize", "answer", "finish", "done", "final"]
    )
    def test_synthesize_aliases(self, alias):
        act = coerce_action(RawAgentAction(action=alias), strict=False)
        assert act.action is ActionType.SYNTHESIZE

    def test_unknown_action_falls_back_to_synthesize(self):
        act = coerce_action(RawAgentAction(action="ponder"), strict=False)
        assert act.action is ActionType.SYNTHESIZE

    def test_unknown_action_strict_raises(self):
        with pytest.raises(CoercionError):
            coerce_action(RawAgentAction(action="ponder"), strict=True)

    def test_retrieve_without_query_falls_back_to_synthesize(self):
        act = coerce_action(RawAgentAction(action="retrieve", query="   "), strict=False)
        assert act.action is ActionType.SYNTHESIZE
        assert act.query is None

    def test_retrieve_without_query_strict_raises(self):
        with pytest.raises(CoercionError):
            coerce_action(RawAgentAction(action="retrieve"), strict=True)

    def test_synthesize_drops_query_and_k(self):
        act = coerce_action(
            RawAgentAction(action="synthesize", query="ignored", k=9), strict=False
        )
        assert act.query is None
        assert act.k is None

    def test_plan_alias_and_sub_queries(self):
        act = coerce_action(
            RawAgentAction(
                thought="decompose",
                action="plan",
                sub_queries=["A revenue", "B revenue", "A revenue"],
            ),
            strict=False,
        )
        assert act.action is ActionType.PLAN
        # de-duplicated + stripped
        assert act.sub_queries == ["A revenue", "B revenue"]
        # PLAN drops query / k
        assert act.query is None
        assert act.k is None

    @pytest.mark.parametrize("alias", ["plan", "decompose", "break_down"])
    def test_plan_aliases(self, alias):
        act = coerce_action(
            RawAgentAction(action=alias, sub_queries=["q1"]), strict=False
        )
        assert act.action is ActionType.PLAN

    def test_plan_without_sub_queries_falls_back_to_synthesize(self):
        act = coerce_action(RawAgentAction(action="plan"), strict=False)
        assert act.action is ActionType.SYNTHESIZE
        assert act.sub_queries is None

    def test_plan_without_sub_queries_strict_raises(self):
        with pytest.raises(CoercionError):
            coerce_action(RawAgentAction(action="plan"), strict=True)

    def test_plan_with_empty_sub_queries_falls_back(self):
        act = coerce_action(
            RawAgentAction(action="plan", sub_queries=["", "  "]), strict=False
        )
        assert act.action is ActionType.SYNTHESIZE

    def test_plan_with_filter_is_coerced(self):
        act = coerce_action(
            RawAgentAction(
                action="retrieve",
                query="q",
                filter={"tags": ["important"], "kb_id": "kb-fin"},
            ),
            strict=False,
        )
        assert act.action is ActionType.RETRIEVE
        assert act.filter is not None
        assert act.filter.kb_id == "kb-fin"

    def test_filter_drops_unknown_tags(self):
        act = coerce_action(
            RawAgentAction(
                action="retrieve", query="q", filter={"tags": ["nonsense_tag"]}
            ),
            strict=False,
        )
        # no usable field survives -> filter is None
        assert act.filter is None

    def test_k_clamped_and_dropped(self):
        assert coerce_action(
            RawAgentAction(action="retrieve", query="q", k=0), strict=False
        ).k is None
        assert coerce_action(
            RawAgentAction(action="retrieve", query="q", k="3"), strict=False
        ).k == 3
        assert coerce_action(
            RawAgentAction(action="retrieve", query="q", k="bad"), strict=False
        ).k is None

    def test_parse_action_response_with_fence(self):
        text = '```json\n{"thought":"r","action":"retrieve","query":"q"}\n```'
        act = parse_action_response(text)
        assert act.action is ActionType.RETRIEVE
        assert act.query == "q"

    def test_parse_action_response_prose_wrapped(self):
        text = 'Here is my decision: {"action":"synthesize"} hope it helps'
        act = parse_action_response(text)
        assert act.action is ActionType.SYNTHESIZE

    def test_extract_json_empty_raises(self):
        with pytest.raises(CoercionError):
            extract_json("")

    def test_extract_json_not_json_raises(self):
        with pytest.raises(CoercionError):
            extract_json("no braces here")

    def test_parse_bad_json_raises(self):
        with pytest.raises(CoercionError):
            parse_action_response("totally not json")


# --------------------------------------------------------------------------- #
# IdentityController
# --------------------------------------------------------------------------- #
class TestIdentityController:
    def test_always_synthesize(self):
        ctrl = IdentityController()
        state = AgentState(question="q")
        action = ctrl.next_action(state)
        assert action.action is ActionType.SYNTHESIZE


# --------------------------------------------------------------------------- #
# AgenticQAEngine
# --------------------------------------------------------------------------- #
class TestAgenticQAEngine:
    def test_constructor_validation(self):
        retriever = _make_retriever([])
        reader = _make_reader()
        with pytest.raises(TypeError):
            AgenticQAEngine("not a controller", retriever, reader)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            AgenticQAEngine(IdentityController(), retriever, reader, max_iterations=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            AgenticQAEngine(IdentityController(), retriever, reader, max_iterations=-1)
        with pytest.raises(ValueError):
            AgenticQAEngine(IdentityController(), retriever, reader, max_evidence=0)

    def test_identity_collapses_to_single_shot(self):
        blocks = [_block("alpha", "alpha revenue grew 10%")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("alpha answer")
        engine = AgenticQAEngine(IdentityController(), retriever, reader)
        result = engine.ask("what about alpha?")
        assert isinstance(result, AgentResult)
        assert result.iterations == 0
        assert len(result.steps) == 1  # the seed retrieval only
        assert result.aborted is False
        assert result.abstained is False
        assert result.text == "alpha answer"
        assert result.citations == []
        assert len(result.evidence) >= 1

    def test_multi_step_loop(self):
        blocks = [
            _block("alpha", "alpha revenue grew 10%"),
            _block("beta", "beta revenue fell 5%"),
            _block("gamma", "gamma margins improved"),
        ]
        retriever = _make_retriever(blocks)
        reader = _make_reader("compare answer")
        ctrl = _ScriptedController(
            [
                _action(ActionType.RETRIEVE, query="beta revenue"),
                _action(ActionType.RETRIEVE, query="gamma margins"),
                _action(ActionType.SYNTHESIZE),
            ]
        )
        engine = AgenticQAEngine(ctrl, retriever, reader, max_iterations=5)
        result = engine.ask("compare alpha beta and gamma")
        assert result.iterations == 2
        assert len(result.steps) == 3  # seed + 2 controller retrievals
        assert ctrl.calls == 3
        assert result.aborted is False
        # the controller saw the accumulating evidence after the seed retrieval
        assert len(ctrl.states_seen[0].evidence) >= 1
        assert result.text == "compare answer"

    def test_max_iterations_aborts_and_synthesizes_best_effort(self):
        blocks = [_block("alpha", "alpha revenue grew")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("best effort")
        # controller never stops -> hit the cap
        ctrl = _ScriptedController([_action(ActionType.RETRIEVE, query="alpha")])
        engine = AgenticQAEngine(ctrl, retriever, reader, max_iterations=2)
        result = engine.ask("alpha?")
        assert result.aborted is True
        assert result.iterations == 2
        assert result.abstained is False
        assert result.text == "best effort"

    def test_max_iterations_zero_is_single_shot_not_aborted(self):
        blocks = [_block("alpha", "alpha revenue grew")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("single")
        # controller would retrieve more, but max_iterations=0 forbids it
        ctrl = _ScriptedController([_action(ActionType.RETRIEVE, query="alpha")])
        engine = AgenticQAEngine(ctrl, retriever, reader, max_iterations=0)
        result = engine.ask("alpha?")
        assert result.aborted is False  # deliberate single-shot, not an abort
        assert result.iterations == 0
        assert len(result.steps) == 1  # seed retrieval only
        assert ctrl.calls == 0  # controller never consulted

    def test_empty_corpus_abstains(self):
        retriever = _make_retriever([])
        reader = _make_reader()
        engine = AgenticQAEngine(IdentityController(), retriever, reader)
        result = engine.ask("anything?")
        assert result.abstained is True
        assert result.iterations == 0
        assert result.evidence == []
        assert result.answer is not None
        assert result.answer.abstained is True

    def test_per_step_k_override_used(self):
        block = _block("alpha", "alpha revenue grew 10%")
        retriever = _make_retriever([block])
        reader = _make_reader("a")
        ctrl = _ScriptedController(
            [_action(ActionType.RETRIEVE, query="alpha", k=1), _action(ActionType.SYNTHESIZE)]
        )
        engine = AgenticQAEngine(ctrl, retriever, reader, max_iterations=3)
        result = engine.ask("alpha")
        assert result.iterations == 1
        # the retrieve step recorded its k via a successful retrieval
        assert result.steps[-1].retrieved_count >= 1

    def test_on_progress_phases(self):
        blocks = [_block("alpha", "alpha revenue grew")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("ans")
        ctrl = _ScriptedController(
            [_action(ActionType.RETRIEVE, query="alpha"), _action(ActionType.SYNTHESIZE)]
        )
        progress: list = []
        engine = AgenticQAEngine(ctrl, retriever, reader, max_iterations=3)
        engine.ask("alpha", on_progress=progress.append)
        phases = [p.phase for p in progress]
        assert phases[0] == "retrieving"  # seed retrieval first
        assert "thinking" in phases
        assert "synthesizing" in phases
        assert phases[-1] == "done"
        # final progress reports completion
        assert progress[-1].percent == 1.0

    def test_is_cancelled_stops_loop_early(self):
        blocks = [_block("alpha", "alpha revenue grew")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("ans")
        ctrl = _ScriptedController([_action(ActionType.RETRIEVE, query="alpha")])
        cancelled = {"state": False}

        def is_cancelled():
            return cancelled["state"]

        engine = AgenticQAEngine(ctrl, retriever, reader, max_iterations=5)

        # flip cancellation after the first controller call
        original = ctrl.next_action

        def wrapped(state):
            result = original(state)
            cancelled["state"] = True
            return result

        ctrl.next_action = wrapped  # type: ignore[assignment]
        result = engine.ask("alpha", is_cancelled=is_cancelled)
        # loop stopped before hitting the cap
        assert result.aborted is False
        assert result.iterations <= 1

    def test_query_processor_interception(self):
        blocks = [_block("alpha", "alpha revenue")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("never used")
        # an out-of-domain intent short-circuits before any retrieval
        intent_json = json.dumps(
            {"reasoning": "r", "intent": "out_of_domain", "confidence": 0.95}
        )
        rewrite_json = json.dumps({"rewritten_query": "alpha", "sub_queries": []})
        client = FakeLLMClient(responses=[intent_json, rewrite_json])
        processor = QueryProcessor(
            classifier=LLMIntentClassifier(client),
            rewriter=LLMQueryRewriter(client),
        )
        engine = AgenticQAEngine(
            IdentityController(), retriever, reader, query_processor=processor
        )
        result = engine.ask("tell me a joke")
        assert result.accepted is False
        assert result.answer is None
        assert result.query_result is not None
        assert result.query_result.intent.intent is QueryIntent.OUT_OF_DOMAIN
        assert result.iterations == 0

    def test_result_surface_for_serializer(self):
        blocks = [_block("alpha", "alpha revenue grew 10%")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("surfaced text")
        engine = AgenticQAEngine(IdentityController(), retriever, reader)
        result = engine.ask("alpha?")
        # the duck-typed surface _to_ask_response relies on
        assert result.query == "alpha?"
        assert result.text == "surfaced text"
        assert result.abstained is False
        assert result.retrieval is not None
        assert result.retrieval.query == "alpha?"
        assert len(result.retrieval.chunks) == len(result.evidence)
        assert result.cached is False


# --------------------------------------------------------------------------- #
# LLMAgentController driving a real loop
# --------------------------------------------------------------------------- #
class TestLLMAgentController:
    def test_drives_multi_step_loop_then_synthesizes(self):
        blocks = [
            _block("alpha", "alpha revenue grew 10%"),
            _block("beta", "beta revenue fell 5%"),
        ]
        retriever = _make_retriever(blocks)
        reader = _make_reader("final answer")
        # controller: retrieve beta, then synthesize
        ctrl_client = FakeLLMClient(
            responses=[
                json.dumps(
                    {"thought": "need beta", "action": "retrieve", "query": "beta revenue"}
                ),
                json.dumps({"thought": "enough", "action": "synthesize"}),
            ]
        )
        ctrl = LLMAgentController(ctrl_client)
        engine = AgenticQAEngine(ctrl, retriever, reader, max_iterations=4)
        result = engine.ask("compare alpha and beta")
        assert result.iterations == 1
        assert ctrl.calls == 2
        assert ctrl.fallbacks == 0
        assert result.text == "final answer"

    def test_bad_response_falls_back_to_synthesize(self):
        blocks = [_block("alpha", "alpha revenue")]
        retriever = _make_retriever(blocks)
        reader = _make_reader("fallback answer")
        ctrl = LLMAgentController(FakeLLMClient(responses=["not json at all"]))
        engine = AgenticQAEngine(ctrl, retriever, reader, max_iterations=3)
        result = engine.ask("alpha?")
        # parse failure -> identity fallback (synthesize) ends the loop
        assert ctrl.fallbacks == 1
        assert result.iterations == 0
        assert result.text == "fallback answer"

    def test_empty_response_strict_raises(self):
        from sparksage.agent import ActionEmptyResponseError

        ctrl = LLMAgentController(FakeLLMClient(responses=[""]), strict=True)
        with pytest.raises(ActionEmptyResponseError):
            ctrl.next_action(AgentState(question="q"))

    def test_bad_response_strict_raises(self):
        from sparksage.agent import ActionResponseParseError

        ctrl = LLMAgentController(
            FakeLLMClient(responses=["totally not json"]), strict=True
        )
        with pytest.raises(ActionResponseParseError):
            ctrl.next_action(AgentState(question="q"))
