"""Tests for the Phase-5 self-corrective QA enhancements (measures 1-3):

* Measure 1 -- self-reflective retrieval loop:
  :class:`~sparksage.retrieve.RetrievalGrader` +
  :class:`~sparksage.query.QueryRefiner`, wired into :class:`~sparksage.qa.QAEngine`.
* Measure 2 -- HyDE: :class:`~sparksage.query.HyDEExpander`.
* Measure 3 -- intent -> KB routing: :class:`~sparksage.qa.IntentKBRouter`.

All tests run offline via :class:`FakeLLMClient` / :class:`FakeEmbeddingClient`.
"""

from __future__ import annotations

import json

import pytest

from sparksage import (
    BlockEmbedder,
    FakeEmbeddingClient,
    FakeLLMClient,
    HyDEExpander,
    IdeaBlock,
    IdentityRefiner,
    InMemoryVectorStore,
    IntentKBRouter,
    IntentResult,
    LLMIntentClassifier,
    LLMQueryRefiner,
    LLMQueryRewriter,
    LLMRetrievalGrader,
    QueryIntent,
    QueryProcessor,
    QueryRefiner,
    RelevanceResult,
    RetrievalFilter,
    RetrievalGrader,
    RetrievalResult,
    RetrievedChunk,
    Tag,
)
from sparksage.qa import QAEngine, QAResult
from sparksage.query.expander import _approx_word_count
from sparksage.reader import LLMAnswerGenerator, Reader
from sparksage.retrieve import (
    DEFAULT_RELEVANCE,
    BM25Retriever,
    GraderResponseParseError,
    RetrievalConfig,
    Retriever,
)
from sparksage.schema.source import SourceRef


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _block(name, body, *, kb_id=None):
    return IdeaBlock(
        name=name,
        critical_question=f"what is {name}?",
        trusted_answer=body,
        tags=[Tag.IMPORTANT],
        source=SourceRef(uri=f"file://{name}.md", title=name, locator="L1"),
        kb_id=kb_id,
    )


def _intent_json(intent="business_analysis", conf=0.9):
    return json.dumps({"reasoning": "r", "intent": intent, "confidence": conf})


def _rewrite_json(text="deploy sparksage"):
    return json.dumps({"rewritten_query": text, "sub_queries": []})


def _answer_json(text="Use pip install.", cid=None):
    return json.dumps(
        {
            "answer": text,
            "citations": [{"block_id": cid or "ID", "quote": "pip"}],
            "confidence": 0.9,
        }
    )


def _rel_json(score=0.5, reasoning="r"):
    return json.dumps({"reasoning": reasoning, "score": score})


def _make_retriever(blocks, *, dim=64):
    embedder = BlockEmbedder(FakeEmbeddingClient(dimension=dim))
    store = InMemoryVectorStore(dimension=dim)
    registry: dict[str, IdeaBlock] = {}
    retriever = Retriever(
        registry, store, embedder, lexical=BM25Retriever(), min_fetch=5, fetch_factor=2
    )
    retriever.index(blocks)
    return retriever


def _chunk(block, score=0.9, rank=0):
    return RetrievedChunk(block=block, score=score, rank=rank)


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
# Measure 2 -- HyDEExpander
# --------------------------------------------------------------------------- #
class TestHyDEExpander:
    def test_short_query_adds_hypothesis(self):
        client = FakeLLMClient(
            responses=[json.dumps({"hypothesis": "pip install is the way"})]
        )
        out = HyDEExpander(client).expand("deploy")
        assert out == ["deploy", "pip install is the way"]

    def test_long_query_is_skipped(self):
        client = FakeLLMClient(responses=[])
        exp = HyDEExpander(client, max_words=3)
        q = "how do I deploy sparksage on kubernetes cluster"
        assert exp.expand(q) == [q]
        assert client.calls == []  # no LLM call made

    def test_bad_json_falls_back(self):
        client = FakeLLMClient(responses=["nope"])
        exp = HyDEExpander(client)
        assert exp.expand("deploy") == ["deploy"]
        assert exp.fallbacks >= 1

    def test_empty_hypothesis_falls_back(self):
        client = FakeLLMClient(responses=[json.dumps({"hypothesis": "  "})])
        exp = HyDEExpander(client)
        assert exp.expand("deploy") == ["deploy"]
        assert exp.fallbacks >= 1

    def test_empty_query_and_n1(self):
        exp = HyDEExpander(FakeLLMClient(responses=[]))
        assert exp.expand("") == []
        assert exp.expand("deploy", n=1) == ["deploy"]

    def test_draft_is_truncated(self):
        long_draft = "answer " * 200
        client = FakeLLMClient(responses=[json.dumps({"hypothesis": long_draft})])
        exp = HyDEExpander(client, max_draft_chars=50)
        out = exp.expand("deploy")
        assert len(out) == 2
        assert len(out[1]) <= 50

    def test_word_count_cjk_and_latin(self):
        assert _approx_word_count("how to deploy") == 3
        # single whitespace token (CJK) falls back to char count
        assert _approx_word_count("营收怎么样") == 5
        assert _approx_word_count("") == 0

    def test_cjk_short_query_fires(self):
        client = FakeLLMClient(responses=[json.dumps({"hypothesis": "假设答案"})])
        out = HyDEExpander(client).expand("营收")  # 2 chars <= 6
        assert out == ["营收", "假设答案"]


# --------------------------------------------------------------------------- #
# Measure 1a -- RetrievalGrader
# --------------------------------------------------------------------------- #
class TestRetrievalGrader:
    def test_grades_relevant(self):
        client = FakeLLMClient(responses=[_rel_json(0.9, "on topic")])
        res = LLMRetrievalGrader(client).grade("q", [_chunk(_block("A", "body"))])
        assert res.score == 0.9
        assert res.reasoning == "on topic"

    def test_empty_chunks_scores_zero(self):
        res = LLMRetrievalGrader(FakeLLMClient(responses=[])).grade("q", [])
        assert res.score == 0.0
        assert "no retrieved" in res.reasoning

    def test_clamps_high_score(self):
        client = FakeLLMClient(responses=[_rel_json(1.7)])
        assert LLMRetrievalGrader(client).grade("q", [_chunk(_block("A", "b"))]).score == 1.0

    def test_clamps_low_score(self):
        client = FakeLLMClient(responses=[_rel_json(-0.4)])
        assert LLMRetrievalGrader(client).grade("q", [_chunk(_block("A", "b"))]).score == 0.0

    def test_bad_response_uses_default(self):
        client = FakeLLMClient(responses=["garbage"])
        res = LLMRetrievalGrader(client).grade("q", [_chunk(_block("A", "b"))])
        assert res.score == DEFAULT_RELEVANCE

    def test_strict_raises_on_bad_response(self):
        client = FakeLLMClient(responses=["garbage"])
        grader = LLMRetrievalGrader(client, strict=True)
        with pytest.raises(GraderResponseParseError):
            grader.grade("q", [_chunk(_block("A", "b"))])

    def test_top_k_truncates_context(self):
        chunks = [_chunk(_block(f"B{i}", f"body {i}")) for i in range(8)]
        client = FakeLLMClient(responses=[_rel_json(0.8)])
        LLMRetrievalGrader(client, top_k=3).grade("q", chunks)
        body = client.last_messages[1]["content"]
        # only the first 3 chunks are rendered (each carries "body N")
        assert "body 0" in body and "body 2" in body
        assert "body 3" not in body

    def test_protocol_satisfied(self):
        grader = LLMRetrievalGrader(FakeLLMClient(responses=[]))
        assert isinstance(grader, RetrievalGrader)
        assert isinstance(_ScriptedGrader([]), RetrievalGrader)


# --------------------------------------------------------------------------- #
# Measure 1b -- QueryRefiner
# --------------------------------------------------------------------------- #
class TestQueryRefiner:
    def test_identity(self):
        assert IdentityRefiner().refine("q", 0.1) == "q"
        assert IdentityRefiner().refine("", 0.1) == ""

    def test_llm_refines(self):
        client = FakeLLMClient(
            responses=[json.dumps({"refined_query": "revenue 2024 net profit"})]
        )
        out = LLMQueryRefiner(client).refine("营收", 0.2, "low relevance")
        assert out == "revenue 2024 net profit"

    def test_bad_json_falls_back(self):
        client = FakeLLMClient(responses=["nope"])
        ref = LLMQueryRefiner(client)
        assert ref.refine("q", 0.2) == "q"
        assert ref.fallbacks >= 1

    def test_empty_query_returns_empty(self):
        client = FakeLLMClient(responses=[json.dumps({"refined_query": "x"})])
        assert LLMQueryRefiner(client).refine("", 0.2) == ""

    def test_same_query_falls_back(self):
        client = FakeLLMClient(responses=[json.dumps({"refined_query": "q"})])
        ref = LLMQueryRefiner(client)
        assert ref.refine("q", 0.2) == "q"
        assert ref.fallbacks >= 1

    def test_protocol_satisfied(self):
        assert isinstance(LLMQueryRefiner(FakeLLMClient(responses=[])), QueryRefiner)
        assert isinstance(IdentityRefiner(), QueryRefiner)
        assert isinstance(_ScriptedRefiner([]), QueryRefiner)


# --------------------------------------------------------------------------- #
# Measure 3 -- IntentKBRouter
# --------------------------------------------------------------------------- #
class TestIntentKBRouter:
    def _ir(self, intent, conf=0.9):
        return IntentResult(intent=intent, confidence=conf)

    def test_routes_known_intent(self):
        r = IntentKBRouter(routing={QueryIntent.FINANCIAL_DATA: "kb-fin"})
        ir = self._ir(QueryIntent.FINANCIAL_DATA)
        assert r.route(ir) == "kb-fin"
        assert r(ir) == "kb-fin"  # callable -> usable as intent_router

    def test_default_when_unmapped(self):
        r = IntentKBRouter(routing={}, default="kb-default")
        assert r.route(self._ir(QueryIntent.TREND)) == "kb-default"

    def test_no_match_returns_none(self):
        r = IntentKBRouter(routing={})
        assert r.route(self._ir(QueryIntent.TREND)) is None

    def test_fallback_callable(self):
        r = IntentKBRouter(routing={}, fallback=lambda ir: "kb-" + ir.intent.value)
        assert r.route(self._ir(QueryIntent.TREND)) == "kb-trend"

    def test_explicit_mapping_beats_default(self):
        r = IntentKBRouter(
            routing={QueryIntent.FINANCIAL_DATA: "kb-fin"}, default="kb-default"
        )
        assert r.route(self._ir(QueryIntent.FINANCIAL_DATA)) == "kb-fin"
        assert r.route(self._ir(QueryIntent.TREND)) == "kb-default"


# --------------------------------------------------------------------------- #
# Measure 3 -- intent routing wired into QAEngine (end-to-end)
# --------------------------------------------------------------------------- #
class TestIntentRouting:
    def test_router_scopes_retrieval_to_kb(self):
        blocks = [
            _block("Revenue", "net profit revenue financial results", kb_id="kb-fin"),
            _block("Support", "how to contact customer support helpdesk", kb_id="kb-support"),
        ]
        retriever = _make_retriever(blocks)
        proc_client = FakeLLMClient(
            responses=[_intent_json("financial_data", 0.95), _rewrite_json("revenue")]
        )
        processor = QueryProcessor(
            classifier=LLMIntentClassifier(proc_client),
            rewriter=LLMQueryRewriter(proc_client),
        )
        gen_client = FakeLLMClient(responses=[_answer_json(cid=str(blocks[0].id))])
        engine = QAEngine(
            retriever=retriever,
            reader=Reader(generator=LLMAnswerGenerator(gen_client)),
            query_processor=processor,
            intent_router=IntentKBRouter(routing={QueryIntent.FINANCIAL_DATA: "kb-fin"}),
            config=RetrievalConfig(use_lexical=False),
        )
        result = engine.ask("revenue")
        assert result.retrieval is not None
        assert len(result.retrieval.chunks) == 1
        assert result.retrieval.chunks[0].block.kb_id == "kb-fin"

    def test_no_router_returns_all(self):
        blocks = [
            _block("Revenue", "net profit revenue financial results", kb_id="kb-fin"),
            _block("Support", "how to contact customer support helpdesk", kb_id="kb-support"),
        ]
        retriever = _make_retriever(blocks)
        proc_client = FakeLLMClient(
            responses=[_intent_json("financial_data", 0.95), _rewrite_json("revenue")]
        )
        processor = QueryProcessor(
            classifier=LLMIntentClassifier(proc_client),
            rewriter=LLMQueryRewriter(proc_client),
        )
        gen_client = FakeLLMClient(responses=[_answer_json(cid=str(blocks[0].id))])
        engine = QAEngine(
            retriever=retriever,
            reader=Reader(generator=LLMAnswerGenerator(gen_client)),
            query_processor=processor,
            config=RetrievalConfig(use_lexical=False),
        )
        result = engine.ask("revenue")
        assert len(result.retrieval.chunks) == 2  # no kb scoping

    def test_per_call_filter_kb_id_wins_over_router(self):
        blocks = [
            _block("Revenue", "net profit revenue financial results", kb_id="kb-fin"),
            _block("Support", "how to contact customer support helpdesk", kb_id="kb-support"),
        ]
        retriever = _make_retriever(blocks)
        proc_client = FakeLLMClient(
            responses=[_intent_json("financial_data", 0.95), _rewrite_json("revenue")]
        )
        processor = QueryProcessor(
            classifier=LLMIntentClassifier(proc_client),
            rewriter=LLMQueryRewriter(proc_client),
        )
        gen_client = FakeLLMClient(responses=[_answer_json(cid=str(blocks[1].id))])
        engine = QAEngine(
            retriever=retriever,
            reader=Reader(generator=LLMAnswerGenerator(gen_client)),
            query_processor=processor,
            intent_router=IntentKBRouter(routing={QueryIntent.FINANCIAL_DATA: "kb-fin"}),
            config=RetrievalConfig(use_lexical=False),
        )
        # caller explicitly asks for kb-support despite the router
        result = engine.ask("revenue", filter=RetrievalFilter(kb_id="kb-support"))
        assert len(result.retrieval.chunks) == 1
        assert result.retrieval.chunks[0].block.kb_id == "kb-support"

    def test_router_without_processor_is_noop(self):
        blocks = [_block("A", "alpha beta")]
        retriever = _make_retriever(blocks)
        gen_client = FakeLLMClient(responses=[_answer_json(cid=str(blocks[0].id))])
        engine = QAEngine(
            retriever=retriever,
            reader=Reader(generator=LLMAnswerGenerator(gen_client)),
            intent_router=IntentKBRouter(routing={QueryIntent.TREND: "kb-x"}),
            config=RetrievalConfig(use_lexical=False),
        )
        result = engine.ask("alpha", use_lexical=False)
        assert result.retrieval is not None
        assert len(result.retrieval.chunks) == 1  # unscoped, router never consulted


# --------------------------------------------------------------------------- #
# Measure 1 -- self-reflective loop wired into QAEngine
# --------------------------------------------------------------------------- #
class TestSelfReflectiveLoop:
    def test_low_grade_triggers_refine_and_reretrieve(self):
        blocks = [_block("Deploy", "deploy via pip install sparksage")]
        retriever = _make_retriever(blocks)
        grader_client = FakeLLMClient(responses=[_rel_json(0.2), _rel_json(0.9)])
        refiner_client = FakeLLMClient(
            responses=[json.dumps({"refined_query": "how to install sparksage package"})]
        )
        gen_client = FakeLLMClient(responses=[_answer_json(cid=str(blocks[0].id))])
        engine = QAEngine(
            retriever=retriever,
            reader=Reader(generator=LLMAnswerGenerator(gen_client)),
            retrieval_grader=LLMRetrievalGrader(grader_client),
            query_refiner=LLMQueryRefiner(refiner_client),
            min_relevance=0.5,
            max_iterations=2,
            config=RetrievalConfig(use_lexical=False),
        )
        result = engine.ask("deploy")
        assert result.iterations == 1
        assert result.refined_query == "how to install sparksage package"
        assert result.relevance is not None
        assert result.relevance.score == 0.9

    def test_high_initial_grade_skips_refinement(self):
        blocks = [_block("Deploy", "deploy via pip install sparksage")]
        retriever = _make_retriever(blocks)
        grader_client = FakeLLMClient(responses=[_rel_json(0.9)])
        refiner_client = FakeLLMClient(responses=[])  # must NOT be called
        gen_client = FakeLLMClient(responses=[_answer_json(cid=str(blocks[0].id))])
        engine = QAEngine(
            retriever=retriever,
            reader=Reader(generator=LLMAnswerGenerator(gen_client)),
            retrieval_grader=LLMRetrievalGrader(grader_client),
            query_refiner=LLMQueryRefiner(refiner_client),
            config=RetrievalConfig(use_lexical=False),
        )
        result = engine.ask("deploy")
        assert result.iterations == 0
        assert result.refined_query is None
        assert result.relevance.score == 0.9
        assert refiner_client.calls == []

    def test_grader_without_refiner_grades_once(self):
        blocks = [_block("Deploy", "deploy via pip install sparksage")]
        retriever = _make_retriever(blocks)
        grader_client = FakeLLMClient(responses=[_rel_json(0.3)])
        gen_client = FakeLLMClient(responses=[_answer_json(cid=str(blocks[0].id))])
        engine = QAEngine(
            retriever=retriever,
            reader=Reader(generator=LLMAnswerGenerator(gen_client)),
            retrieval_grader=LLMRetrievalGrader(grader_client),
            config=RetrievalConfig(use_lexical=False),
        )
        result = engine.ask("deploy")
        assert result.iterations == 0  # no refiner -> no loop
        assert result.refined_query is None
        assert result.relevance.score == 0.3
        assert len(grader_client.calls) == 1  # graded exactly once

    def test_loop_keeps_best_when_refinement_is_worse(self):
        blocks = [_block("A", "alpha"), _block("B", "beta")]
        retriever = _make_retriever(blocks)
        initial = RetrievalResult(query="SEARCH", chunks=[_chunk(blocks[0])])
        refined_ret = RetrievalResult(query="REFINED", chunks=[_chunk(blocks[1])])
        engine = QAEngine(
            retriever=retriever,
            reader=Reader(generator=LLMAnswerGenerator(FakeLLMClient(responses=[]))),
            retrieval_grader=_ScriptedGrader([0.4, 0.1]),  # refined is WORSE
            query_refiner=_ScriptedRefiner(["REFINED"]),
            min_relevance=0.5,
            max_iterations=1,
        )
        engine._retrieve = lambda q, **kw: refined_ret if q == "REFINED" else initial  # type: ignore[assignment]
        best_rel, best_ret, refined, rounds = engine._reflective_retrieve(
            "Q", "SEARCH", initial,
            context=None, filter=None, k=None, use_lexical=False, use_rerank=False,
        )
        assert best_ret is initial  # kept the better-graded initial
        assert best_rel.score == 0.4
        assert rounds == 1
        assert refined == "REFINED"

    def test_loop_reverts_to_best_across_two_rounds(self):
        blocks = [_block("A", "alpha")]
        retriever = _make_retriever(blocks)
        initial = RetrievalResult(query="SEARCH", chunks=[_chunk(blocks[0])])
        refined_ret = RetrievalResult(query="REFINED", chunks=[_chunk(blocks[0])])
        engine = QAEngine(
            retriever=retriever,
            reader=Reader(generator=LLMAnswerGenerator(FakeLLMClient(responses=[]))),
            retrieval_grader=_ScriptedGrader([0.2, 0.9]),  # 2nd round improves
            query_refiner=_ScriptedRefiner(["REFINED"]),
            min_relevance=0.5,
            max_iterations=2,
        )
        engine._retrieve = lambda q, **kw: refined_ret if q == "REFINED" else initial  # type: ignore[assignment]
        best_rel, best_ret, refined, rounds = engine._reflective_retrieve(
            "Q", "SEARCH", initial,
            context=None, filter=None, k=None, use_lexical=False, use_rerank=False,
        )
        assert best_rel.score == 0.9
        assert best_ret is refined_ret
        assert rounds == 1  # broke once relevance floor was met

    def test_no_grader_is_legacy_single_pass(self):
        blocks = [_block("Deploy", "deploy via pip install sparksage")]
        retriever = _make_retriever(blocks)
        gen_client = FakeLLMClient(responses=[_answer_json(cid=str(blocks[0].id))])
        engine = QAEngine(
            retriever=retriever,
            reader=Reader(generator=LLMAnswerGenerator(gen_client)),
            config=RetrievalConfig(use_lexical=False),
        )
        result = engine.ask("deploy")
        assert isinstance(result, QAResult)
        assert result.relevance is None
        assert result.iterations == 0
        assert not result.abstained

    def test_identity_refiner_does_not_loop(self):
        blocks = [_block("Deploy", "deploy via pip install sparksage")]
        retriever = _make_retriever(blocks)
        grader_client = FakeLLMClient(
            responses=[_rel_json(0.2), _rel_json(0.2), _rel_json(0.2)]
        )
        gen_client = FakeLLMClient(responses=[_answer_json(cid=str(blocks[0].id))])
        engine = QAEngine(
            retriever=retriever,
            reader=Reader(generator=LLMAnswerGenerator(gen_client)),
            retrieval_grader=LLMRetrievalGrader(grader_client),
            query_refiner=IdentityRefiner(),  # no-op -> loop should not run
            min_relevance=0.5,
            max_iterations=3,
            config=RetrievalConfig(use_lexical=False),
        )
        result = engine.ask("deploy")
        assert result.iterations == 0  # identity refiner -> no refinement rounds
        assert result.relevance.score == 0.2
        assert len(grader_client.calls) == 1  # only the initial grade

    def test_invalid_construction_args(self):
        retriever = _make_retriever([_block("A", "a")])
        reader = Reader(generator=LLMAnswerGenerator(FakeLLMClient(responses=[])))
        with pytest.raises(ValueError):
            QAEngine(retriever=retriever, reader=reader, min_relevance=1.5)
        with pytest.raises(ValueError):
            QAEngine(retriever=retriever, reader=reader, max_iterations=-1)
        with pytest.raises(TypeError):
            QAEngine(retriever=retriever, reader=reader, intent_router="not callable")  # type: ignore[arg-type]
