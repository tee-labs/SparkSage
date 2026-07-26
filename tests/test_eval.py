"""Tests for the end-to-end QA evaluator (Phase 4)."""

from __future__ import annotations

import json

import pytest

from sparksage import (
    BlockEmbedder,
    FakeEmbeddingClient,
    FakeLLMClient,
    IdeaBlock,
    InMemoryVectorStore,
    Tag,
)
from sparksage.eval import (
    LLMCorrectnessJudge,
    QAEvaluator,
    QATestCase,
    TokenOverlapJudge,
    token_f1,
)
from sparksage.qa import QAEngine
from sparksage.reader import LLMAnswerGenerator, Reader
from sparksage.retrieve import BM25Retriever, Retriever


def _block(name, body):
    return IdeaBlock(
        name=name,
        critical_question=f"what is {name}?",
        trusted_answer=body,
        tags=[Tag.IMPORTANT],
    )


def _engine(blocks, answer_text="pip install sparksage to deploy", faith_score=0.9):
    dim = 64
    embedder = BlockEmbedder(FakeEmbeddingClient(dimension=dim))
    retriever = Retriever(
        {}, InMemoryVectorStore(dimension=dim), embedder, lexical=BM25Retriever(), min_fetch=5
    )
    retriever.index(blocks)
    gen_client = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "answer": answer_text,
                    "citations": [{"block_id": str(blocks[0].id)}],
                    "confidence": 0.9,
                }
            ),
            json.dumps({"score": faith_score, "supported_claims": 1, "unsupported_claims": 0}),
        ]
    )
    reader = Reader(
        generator=LLMAnswerGenerator(gen_client),
        faithfulness_judge=None,  # skip faith for simplicity
    )
    return QAEngine(retriever=retriever, reader=reader, config=__import__(
        "sparksage.retrieve", fromlist=["RetrievalConfig"]).RetrievalConfig(use_lexical=False)
    )


class TestTokenF1:
    def test_identical(self):
        assert token_f1("pip install sparksage", "pip install sparksage") == 1.0

    def test_disjoint(self):
        assert token_f1("alpha beta", "gamma delta") == 0.0

    def test_partial(self):
        score = token_f1("install sparksage deploy", "install sparksage")
        assert 0.0 < score < 1.0

    def test_empty(self):
        assert token_f1("", "x") == 0.0


class TestJudges:
    def test_token_overlap_judge(self):
        assert TokenOverlapJudge().score("q", "a b c", "a b c") == 1.0

    def test_llm_judge_parses(self):
        client = FakeLLMClient(responses=[json.dumps({"score": 0.8})])
        assert LLMCorrectnessJudge(client).score("q", "gen", "ref") == 0.8

    def test_llm_judge_fallback(self):
        client = FakeLLMClient(responses=["garbage"])
        judge = LLMCorrectnessJudge(client)
        out = judge.score("q", "install sparksage", "install sparksage")
        assert out == 1.0  # fell back to token F1
        assert judge.fallbacks >= 1

    def test_llm_judge_clamps(self):
        client = FakeLLMClient(responses=[json.dumps({"score": 2.0})])
        assert LLMCorrectnessJudge(client).score("q", "g", "r") == 1.0


class TestEvaluator:
    def test_run_with_reference_answer(self):
        blocks = [_block("Deploy", "pip install sparksage to deploy")]
        engine = _engine(blocks)
        evaluator = QAEvaluator(engine=engine)
        cases = [
            QATestCase(
                query="how to deploy",
                reference_answer="pip install sparksage to deploy",
                relevant_block_ids={str(blocks[0].id)},
            )
        ]
        report = evaluator.run(cases, k=3)
        assert report.case_count == 1
        assert report.mean_correctness == 1.0
        assert report.abstention_rate == 0.0
        assert report.retrieval.hit_at_1 == 1.0

    def test_run_without_reference_uses_proxy(self):
        blocks = [_block("Deploy", "pip install sparksage to deploy")]
        engine = _engine(blocks)
        evaluator = QAEvaluator(engine=engine)
        cases = [QATestCase(query="how to deploy", relevant_block_ids={str(blocks[0].id)})]
        report = evaluator.run(cases, k=3)
        # hit but no faithfulness (judge off) -> proxy 0.5
        assert 0.0 < report.mean_correctness <= 1.0

    def test_abstention_scores_zero(self):
        blocks = [_block("Deploy", "pip install sparksage to deploy")]
        # generator abstains
        dim = 64
        retriever = Retriever(
            {}, InMemoryVectorStore(dimension=dim),
            BlockEmbedder(FakeEmbeddingClient(dimension=dim)),
            lexical=BM25Retriever(), min_fetch=5,
        )
        retriever.index(blocks)
        gen_client = FakeLLMClient(
            responses=[json.dumps({"answer": "", "citations": [], "confidence": 0.0})]
        )
        reader = Reader(generator=LLMAnswerGenerator(gen_client))
        engine = QAEngine(retriever=retriever, reader=reader)
        evaluator = QAEvaluator(engine=engine)
        cases = [QATestCase(query="how to deploy", reference_answer="something")]
        report = evaluator.run(cases, k=3)
        assert report.abstention_rate == 1.0
        assert report.mean_correctness == 0.0

    def test_empty_cases(self):
        blocks = [_block("Deploy", "deploy")]
        engine = _engine(blocks)
        report = QAEvaluator(engine=engine).run([], k=3)
        assert report.case_count == 0

    def test_bad_k(self):
        blocks = [_block("Deploy", "deploy")]
        engine = _engine(blocks)
        with pytest.raises(ValueError):
            QAEvaluator(engine=engine).run([QATestCase(query="q")], k=0)
