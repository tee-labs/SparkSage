"""Tests for the end-to-end QA engine: query -> retrieval -> answer.

All tests run offline via :class:`FakeLLMClient` / :class:`FakeEmbeddingClient`.
"""

from __future__ import annotations

import json

from sparksage import (
    BlockEmbedder,
    FakeEmbeddingClient,
    FakeLLMClient,
    IdeaBlock,
    InMemoryVectorStore,
    LLMIntentClassifier,
    LLMQueryRewriter,
    QueryProcessor,
    Tag,
)
from sparksage.qa import QAEngine, QAResult
from sparksage.reader import LLMAnswerGenerator, Reader
from sparksage.retrieve import BM25Retriever, RetrievalConfig, Retriever
from sparksage.schema.source import SourceRef


def _block(name, body):
    return IdeaBlock(
        name=name,
        critical_question=f"what is {name}?",
        trusted_answer=body,
        tags=[Tag.IMPORTANT],
        source=SourceRef(uri=f"file://{name}.md", title=name, locator="L1"),
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


def _faith_json(score=0.9):
    return json.dumps({"score": score, "supported_claims": 1, "unsupported_claims": 0})


def _make_engine(
    blocks,
    *,
    with_processor=False,
    reranker=None,
    faith=True,
):
    dim = 64
    embedder = BlockEmbedder(FakeEmbeddingClient(dimension=dim))
    store = InMemoryVectorStore(dimension=dim)
    registry: dict[str, IdeaBlock] = {}
    retriever = Retriever(
        registry, store, embedder, lexical=BM25Retriever(), reranker=reranker,
        min_fetch=5, fetch_factor=2,
    )
    retriever.index(blocks)

    # LLM responses depend on whether a processor is wired (it consumes calls first)
    responses = []
    processor = None
    if with_processor:
        responses.extend([_intent_json(), _rewrite_json()])
        client = FakeLLMClient(responses=responses)
        processor = QueryProcessor(
            classifier=LLMIntentClassifier(client),
            rewriter=LLMQueryRewriter(client),
        )
        gen_client = FakeLLMClient(responses=[_answer_json(cid=str(blocks[0].id)), _faith_json()])
    else:
        gen_client = FakeLLMClient(responses=[_answer_json(cid=str(blocks[0].id)), _faith_json()])

    judge = None
    if faith:
        judge = _make_judge(gen_client)
    reader = Reader(generator=LLMAnswerGenerator(gen_client), faithfulness_judge=judge)
    return QAEngine(retriever=retriever, reader=reader, query_processor=processor)


def _make_judge(client):
    from sparksage.reader import LLMFaithfulnessJudge

    return LLMFaithfulnessJudge(client)


class TestQAEngine:
    def test_end_to_end_answer(self):
        blocks = [_block("Deploy", "deploy via pip install sparksage")]
        engine = _make_engine(blocks)
        result = engine.ask("how to deploy", use_lexical=False)
        assert isinstance(result, QAResult)
        assert not result.abstained
        assert "pip" in result.text
        assert result.citations
        assert result.citations[0].uri == "file://Deploy.md"

    def test_with_query_processor(self):
        blocks = [_block("Deploy", "deploy via pip install sparksage")]
        engine = _make_engine(blocks, with_processor=True)
        result = engine.ask("how to deploy", use_lexical=False)
        assert result.query_result is not None
        assert result.query_result.accepted
        assert not result.abstained

    def test_rejected_query_short_circuits(self):
        blocks = [_block("Deploy", "deploy via pip install sparksage")]
        dim = 64
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=dim))
        retriever = Retriever(
            {}, InMemoryVectorStore(dimension=dim), embedder, lexical=BM25Retriever(), min_fetch=5
        )
        retriever.index(blocks)
        # classifier says out_of_domain -> rejected before retrieval
        client = FakeLLMClient(responses=[_intent_json(intent="out_of_domain", conf=0.99)])
        processor = QueryProcessor(
            classifier=LLMIntentClassifier(client),
            rewriter=LLMQueryRewriter(FakeLLMClient(responses=[_rewrite_json()])),
        )
        gen_client = FakeLLMClient(responses=[_answer_json(cid=str(blocks[0].id))])
        reader = Reader(generator=LLMAnswerGenerator(gen_client))
        engine = QAEngine(retriever=retriever, reader=reader, query_processor=processor)
        result = engine.ask("off topic", use_lexical=False)
        assert not result.query_result.accepted
        assert result.abstained
        assert result.retrieval is None
        assert result.answer is None

    def test_abstention_propagates(self):
        blocks = [_block("Deploy", "deploy via pip install sparksage")]
        dim = 64
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=dim))
        retriever = Retriever(
            {}, InMemoryVectorStore(dimension=dim), embedder, lexical=BM25Retriever(), min_fetch=5
        )
        retriever.index(blocks)
        # generator abstains (empty answer)
        gen_client = FakeLLMClient(
            responses=[json.dumps({"answer": "", "citations": [], "confidence": 0.0})]
        )
        reader = Reader(generator=LLMAnswerGenerator(gen_client))
        engine = QAEngine(retriever=retriever, reader=reader)
        result = engine.ask("how to deploy", use_lexical=False)
        assert result.abstained

    def test_subquery_fusion_path(self):
        blocks = [
            _block("A", "alpha beta gamma content"),
            _block("B", "delta epsilon zeta content"),
        ]
        dim = 64
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=dim))
        retriever = Retriever(
            {}, InMemoryVectorStore(dimension=dim), embedder, lexical=BM25Retriever(), min_fetch=5
        )
        retriever.index(blocks)
        # rewriter emits sub_queries -> engine takes the multi-retrieve path
        client = FakeLLMClient(
            responses=[
                _intent_json(),
                json.dumps({"rewritten_query": "alpha", "sub_queries": ["beta", "gamma"]}),
            ]
        )
        processor = QueryProcessor(
            classifier=LLMIntentClassifier(client), rewriter=LLMQueryRewriter(client)
        )
        gen_client = FakeLLMClient(responses=[_answer_json(cid=str(blocks[0].id))])
        reader = Reader(generator=LLMAnswerGenerator(gen_client))
        engine = QAEngine(retriever=retriever, reader=reader, query_processor=processor)
        result = engine.ask("compare", use_lexical=False)
        assert result.retrieval is not None
        assert result.retrieval.fused

    def test_cache_short_circuits(self):
        blocks = [_block("Deploy", "deploy via pip install sparksage")]
        engine = _make_engine(blocks, faith=False)

        class CountingCache:
            def __init__(self):
                self.store_map = {}
                self.lookups = 0

            def lookup(self, query):
                self.lookups += 1
                return self.store_map.get(query)

            def store(self, query, result):
                self.store_map[query] = result

        cache = CountingCache()
        engine._cache = cache  # type: ignore[attr-defined]
        engine._config = RetrievalConfig(use_lexical=False)  # type: ignore[attr-defined]

        r1 = engine.ask("how to deploy")
        first_calls = len(engine.reader.generator._client.calls)  # type: ignore[attr-defined]
        r2 = engine.ask("how to deploy")  # cache hit
        assert r2.cached
        assert not r1.cached
        # no new generator calls on the cache hit
        assert len(engine.reader.generator._client.calls) == first_calls  # type: ignore[attr-defined]

    def test_no_chunks_abstains(self):
        dim = 64
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=dim))
        retriever = Retriever(
            {}, InMemoryVectorStore(dimension=dim), embedder, lexical=BM25Retriever(), min_fetch=5
        )
        # index nothing
        reader = Reader(generator=LLMAnswerGenerator(FakeLLMClient(responses=[])))
        engine = QAEngine(retriever=retriever, reader=reader)
        result = engine.ask("anything", use_lexical=False)
        assert result.abstained
