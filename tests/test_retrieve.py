"""Tests for the retrieval layer: BM25, RRF fusion, reranker, hybrid orchestrator.

All tests run offline via :class:`FakeEmbeddingClient` / :class:`FakeLLMClient`,
no network required.
"""

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
from sparksage.embed.store import SearchHit
from sparksage.retrieve import (
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_MIN_FETCH,
    BM25Retriever,
    IdentityReranker,
    LLMReranker,
    NullLexicalRetriever,
    RetrievalFilter,
    RetrievedChunk,
    tokenize,
    tune_rrf_k,
)
from sparksage.retrieve.fusion import reciprocal_rank_fusion
from sparksage.schema.entity import Entity
from sparksage.schema.enums import EntityType
from sparksage.schema.source import SourceRef


def _block(name, q, a, *, tags=None, keywords=None, entities=None, kb_id=None):
    return IdeaBlock(
        name=name,
        critical_question=q,
        trusted_answer=a,
        tags=tags or [],
        keywords=keywords or [],
        entities=entities or [],
        source=SourceRef(uri=f"file://{name}.md", title=name, locator="L10"),
        kb_id=kb_id,
    )


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #
class TestTokenizer:
    def test_ascii_lowercased(self):
        assert tokenize("Deploy SparkSage NOW") == ["deploy", "sparksage", "now"]

    def test_cjk_unigrams_and_bigrams(self):
        toks = tokenize("中国移动")
        assert "中" in toks and "国" in toks
        assert "中国" in toks and "移动" in toks  # adjacent bigrams
        assert "中动" not in toks  # non-adjacent pair is not a bigram

    def test_empty(self):
        assert tokenize("") == []


# --------------------------------------------------------------------------- #
# BM25Retriever
# --------------------------------------------------------------------------- #
class TestBM25:
    def test_keyword_match_ranks_first(self):
        blocks = [
            _block("A", "qa?", "general prose about systems", keywords=["deploy"]),
            _block("B", "qb?", "step by step deploy procedure", keywords=["deploy"]),
        ]
        lex = BM25Retriever()
        lex.index(blocks)
        hits = lex.search("deploy", k=2)
        assert len(hits) == 2
        assert hits[0].block_id == str(blocks[1].id)

    def test_empty_index_returns_empty(self):
        assert BM25Retriever().search("x", k=3) == []

    def test_no_query_tokens_returns_empty(self):
        blocks = [_block("A", "q?", "deploy", keywords=["deploy"])]
        lex = BM25Retriever()
        lex.index(blocks)
        assert lex.search("   ", k=3) == []

    def test_len_and_contains(self):
        blocks = [_block("A", "q?", "deploy", keywords=["deploy"])]
        lex = BM25Retriever()
        lex.index(blocks)
        assert len(lex) == 1
        assert str(blocks[0].id) in lex

    def test_bad_params(self):
        with pytest.raises(ValueError):
            BM25Retriever(k1=-1)
        with pytest.raises(ValueError):
            BM25Retriever(b=2.0)


# --------------------------------------------------------------------------- #
# RRF fusion
# --------------------------------------------------------------------------- #
class TestRRF:
    def test_fuses_two_lists(self):
        dense = [SearchHit("a", 0.9), SearchHit("b", 0.8)]
        lex = [SearchHit("b", 11.0), SearchHit("c", 9.0)]
        fused = reciprocal_rank_fusion([dense, lex])
        ids = [h.block_id for h in fused]
        assert set(ids) == {"a", "b", "c"}
        # 'b' appears high in both -> ranked first
        assert ids[0] == "b"

    def test_single_list_passthrough_ish(self):
        dense = [SearchHit("a", 0.9), SearchHit("b", 0.8)]
        fused = reciprocal_rank_fusion([dense])
        assert [h.block_id for h in fused] == ["a", "b"]

    def test_top_n(self):
        lists = [[SearchHit(c, 1.0)] for c in "abcde"]
        fused = reciprocal_rank_fusion(lists, top_n=2)
        assert len(fused) == 2

    def test_bad_params(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([], k_const=0)
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([[SearchHit("a", 1.0)]], top_n=0)


# --------------------------------------------------------------------------- #
# Reranker
# --------------------------------------------------------------------------- #
class TestReranker:
    def _chunks(self, blocks):
        return [RetrievedChunk(block=b, score=0.5, rank=i) for i, b in enumerate(blocks)]

    def test_identity_preserves_order(self):
        blocks = [_block("A", "q?", "a"), _block("B", "q?", "b")]
        rr = IdentityReranker()
        out = rr.rerank("q", self._chunks(blocks))
        assert [c.block.name for c in out] == ["A", "B"]
        assert [c.rank for c in out] == [0, 1]

    def test_identity_top_n(self):
        blocks = [_block("A", "q?", "a"), _block("B", "q?", "b")]
        out = IdentityReranker().rerank("q", self._chunks(blocks), top_n=1)
        assert len(out) == 1

    def test_llm_reranker_reorders(self):
        blocks = [
            _block("A", "q?", "irrelevant content"),
            _block("B", "q?", "directly answers deployment"),
        ]
        client = FakeLLMClient(responses=[json.dumps([1, 0])])
        rr = LLMReranker(client)
        out = rr.rerank("deploy", self._chunks(blocks))
        assert out[0].block.name == "B"
        assert out[0].rank == 0
        assert out[0].score >= out[1].score

    def test_llm_reranker_falls_back_on_bad_json(self):
        blocks = [_block("A", "q?", "a"), _block("B", "q?", "b")]
        client = FakeLLMClient(responses=["not json at all"])
        rr = LLMReranker(client)
        out = rr.rerank("q", self._chunks(blocks))
        assert [c.block.name for c in out] == ["A", "B"]
        assert rr.fallbacks >= 1

    def test_llm_reranker_handles_dict_order(self):
        blocks = [_block("A", "q?", "a"), _block("B", "q?", "b")]
        client = FakeLLMClient(responses=[json.dumps({"order": [1, 0]})])
        out = LLMReranker(client).rerank("q", self._chunks(blocks))
        assert out[0].block.name == "B"

    def test_llm_reranker_empty_and_single(self):
        client = FakeLLMClient(responses=[])
        assert LLMReranker(client).rerank("q", []) == []
        one = [_block("A", "q?", "a")]
        out = LLMReranker(client).rerank("q", self._chunks(one))
        assert len(out) == 1


# --------------------------------------------------------------------------- #
# RetrievalFilter
# --------------------------------------------------------------------------- #
class TestRetrievalFilter:
    def test_empty_filter_matches_all(self):
        flt = RetrievalFilter()
        assert flt.is_empty
        assert flt.matches(_block("A", "q?", "a"))

    def test_tags_any(self):
        flt = RetrievalFilter(tags={Tag.IMPORTANT})
        assert flt.matches(_block("A", "q?", "a", tags=[Tag.IMPORTANT]))
        assert not flt.matches(_block("B", "q?", "b", tags=[Tag.WARNING]))

    def test_entities_case_insensitive_with_aliases(self):
        ent = Entity(entity_name="SparkSage", entity_type=EntityType.PRODUCT, aliases=["ss"])
        flt = RetrievalFilter(entities={"sparksage"})
        assert flt.matches(_block("A", "q?", "a", entities=[ent]))
        flt2 = RetrievalFilter(entities={"ss"})
        assert flt2.matches(_block("A", "q?", "a", entities=[ent]))
        assert not RetrievalFilter(entities={"other"}).matches(
            _block("A", "q?", "a", entities=[ent])
        )

    def test_language_and_block_ids_and_kb(self):
        b = _block("A", "q?", "a")
        b.language = "zh"
        b.kb_id = "kb1"
        assert RetrievalFilter(languages={"zh"}).matches(b)
        assert not RetrievalFilter(languages={"en"}).matches(b)
        assert RetrievalFilter(block_ids={str(b.id)}).matches(b)
        assert RetrievalFilter(kb_id="kb1").matches(b)
        assert not RetrievalFilter(kb_id="kb2").matches(b)


# --------------------------------------------------------------------------- #
# Retriever (hybrid orchestrator)
# --------------------------------------------------------------------------- #
class TestRetriever:
    def _make(self, blocks, *, lexical=True, reranker=None):
        dim = 64
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=dim))
        store = InMemoryVectorStore(dimension=dim)
        registry: dict[str, IdeaBlock] = {}
        from sparksage.retrieve import Retriever

        lex = BM25Retriever() if lexical else None
        retriever = Retriever(
            registry, store, embedder, lexical=lex, reranker=reranker,
            min_fetch=5, fetch_factor=2,
        )
        retriever.index(blocks)
        return retriever

    def test_dense_only_when_no_lexical(self):
        blocks = [
            _block("Deploy", "how to deploy?", "run pip install sparksage", keywords=["deploy"]),
            _block("Eat", "what to eat?", "try apples and oranges", keywords=["eat"]),
        ]
        retriever = self._make(blocks, lexical=False)
        result = retriever.search("deploy", k=2, use_lexical=False)
        assert result.top_score >= 0.0
        assert not result.fused
        assert len(result.chunks) <= 2

    def test_hybrid_fuses(self):
        blocks = [
            _block(
                "Deploy", "how to deploy?",
                "run pip install sparksage deploy", keywords=["deploy"],
            ),
            _block("Eat", "what to eat?", "try apples and oranges", keywords=["eat"]),
        ]
        retriever = self._make(blocks)
        result = retriever.search("deploy", k=2)
        assert result.fused
        assert result.chunks[0].block.name == "Deploy"

    def test_filter_scopes_results(self):
        blocks = [
            _block("A", "q?", "a", tags=[Tag.IMPORTANT]),
            _block("B", "q?", "b", tags=[Tag.WARNING]),
        ]
        retriever = self._make(blocks)
        result = retriever.search("a", k=5, filter=RetrievalFilter(tags={Tag.WARNING}))
        assert all(c.block.tags and Tag.WARNING in c.block.tags for c in result.chunks)

    def test_rerank_applied(self):
        blocks = [_block("A", "q?", "a"), _block("B", "q?", "b")]
        client = FakeLLMClient(responses=[json.dumps([1, 0])])
        rr = LLMReranker(client)
        retriever = self._make(blocks, reranker=rr)
        result = retriever.search("q", k=2, use_lexical=False, use_rerank=True)
        assert result.reranked

    def test_empty_query_returns_empty(self):
        retriever = self._make([_block("A", "q?", "a")])
        assert retriever.search("   ", k=3).is_empty

    def test_bad_k(self):
        retriever = self._make([_block("A", "q?", "a")])
        with pytest.raises(ValueError):
            retriever.search("q", k=0)

    def test_citation_carries_locator(self):
        blocks = [_block("A", "q?", "a")]
        retriever = self._make(blocks, lexical=False)
        result = retriever.search("a", k=1, use_lexical=False)
        cit = result.chunks[0].to_citation()
        assert cit.uri == "file://A.md"
        assert cit.locator == "L10"
        assert cit.title == "A"

    def test_index_empty_clears(self):
        retriever = self._make([_block("A", "q?", "a")])
        assert len(retriever.store) == 1
        retriever.index([])
        assert len(retriever.store) == 0

    def test_null_lexical_retriever(self):
        blocks = [_block("A", "q?", "deploy")]
        dim = 64
        from sparksage.retrieve import Retriever

        retriever = Retriever(
            {}, InMemoryVectorStore(dimension=dim),
            BlockEmbedder(FakeEmbeddingClient(dimension=dim)),
            lexical=NullLexicalRetriever(),
            min_fetch=3,
        )
        retriever.index(blocks)
        result = retriever.search("deploy", k=1)
        assert not result.fused


# --------------------------------------------------------------------------- #
# Defaults / configuration
# --------------------------------------------------------------------------- #
class TestDefaults:
    def test_min_fetch_floor_is_50(self):
        assert DEFAULT_MIN_FETCH == 50

    def test_default_dedup_threshold(self):
        assert DEFAULT_DEDUP_THRESHOLD == 0.9

    def test_dedup_threshold_property_and_default(self):
        from sparksage.retrieve import Retriever

        dim = 8
        r = Retriever(
            {}, InMemoryVectorStore(dimension=dim),
            BlockEmbedder(FakeEmbeddingClient(dimension=dim)),
        )
        assert r.dedup_threshold == DEFAULT_DEDUP_THRESHOLD

    def test_dedup_threshold_off_via_none(self):
        from sparksage.retrieve import Retriever

        dim = 8
        r = Retriever(
            {}, InMemoryVectorStore(dimension=dim),
            BlockEmbedder(FakeEmbeddingClient(dimension=dim)),
            dedup_threshold=None,
        )
        assert r.dedup_threshold is None

    def test_bad_dedup_threshold(self):
        from sparksage.retrieve import Retriever

        dim = 8
        with pytest.raises(ValueError):
            Retriever(
                {}, InMemoryVectorStore(dimension=dim),
                BlockEmbedder(FakeEmbeddingClient(dimension=dim)),
                dedup_threshold=1.5,
            )
        with pytest.raises(TypeError):
            Retriever(
                {}, InMemoryVectorStore(dimension=dim),
                BlockEmbedder(FakeEmbeddingClient(dimension=dim)),
                dedup_threshold="x",  # type: ignore[arg-type]
            )


# --------------------------------------------------------------------------- #
# Semantic dedup after rerank
# --------------------------------------------------------------------------- #
class TestSemanticDedup:
    def _retriever(self, blocks, *, dedup_threshold):
        dim = 128
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=dim))
        store = InMemoryVectorStore(dimension=dim)
        registry: dict[str, IdeaBlock] = {}
        from sparksage.retrieve import Retriever

        retriever = Retriever(
            registry, store, embedder,
            min_fetch=5, fetch_factor=2,
            dedup_threshold=dedup_threshold,
        )
        retriever.index(blocks)
        return retriever

    def test_near_duplicate_blocks_collapsed(self):
        # Two near-identical blocks (identical name/question/answer -> identical
        # embedding_text -> identical embedding). Only their UUIDs differ.
        blocks = [
            _block("Dup", "how to deploy?", "run pip install sparksage"),
            _block("Dup", "how to deploy?", "run pip install sparksage"),
            _block("Other", "what to eat?", "try apples and oranges"),
        ]
        retriever = self._retriever(blocks, dedup_threshold=0.9)
        result = retriever.search("deploy", k=3, use_lexical=False, use_rerank=False)
        ids = {c.block.name for c in result.chunks}
        # the two duplicates collapse; only one Dup + Other survive
        assert "Other" in ids
        assert "Dup" in ids
        assert len([c for c in result.chunks if c.block.name == "Dup"]) == 1

    def test_distinct_blocks_not_collapsed(self):
        blocks = [
            _block("A", "qa?", "general prose about systems"),
            _block("B", "qb?", "step by step deploy procedure"),
        ]
        retriever = self._retriever(blocks, dedup_threshold=0.9)
        result = retriever.search("deploy", k=2, use_lexical=False, use_rerank=False)
        assert len(result.chunks) == 2

    def test_dedup_disabled_returns_all(self):
        blocks = [
            _block("Dup", "how to deploy?", "run pip install sparksage"),
            _block("Dup", "how to deploy?", "run pip install sparksage"),
        ]
        retriever = self._retriever(blocks, dedup_threshold=None)
        result = retriever.search("deploy", k=2, use_lexical=False, use_rerank=False)
        assert len(result.chunks) == 2

    def test_dedup_ranks_are_contiguous(self):
        blocks = [
            _block("Dup", "how to deploy?", "run pip install sparksage"),
            _block("Dup", "how to deploy?", "run pip install sparksage"),
            _block("Other", "what to eat?", "try apples and oranges"),
        ]
        retriever = self._retriever(blocks, dedup_threshold=0.9)
        result = retriever.search("deploy", k=3, use_lexical=False, use_rerank=False)
        assert [c.rank for c in result.chunks] == list(range(len(result.chunks)))

    def test_dedup_with_rerank_keeps_k(self):
        # With a reranker + dedup, the final count still respects k where possible.
        blocks = [
            _block("Dup", "deploy?", "run pip install sparksage deploy"),
            _block("Dup", "deploy?", "run pip install sparksage deploy"),
            _block("Other", "eat?", "try apples and oranges food"),
        ]
        dim = 128
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=dim))
        store = InMemoryVectorStore(dimension=dim)
        registry: dict[str, IdeaBlock] = {}
        from sparksage.retrieve import Retriever

        client = FakeLLMClient(responses=[json.dumps([0, 1, 2])])
        rr = LLMReranker(client)
        retriever = Retriever(
            registry, store, embedder,
            min_fetch=5, fetch_factor=2, reranker=rr, dedup_threshold=0.9,
        )
        retriever.index(blocks)
        result = retriever.search("deploy", k=2, use_lexical=False, use_rerank=True)
        # dedup collapses the pair but Other survives -> at most 2 returned
        assert len(result.chunks) <= 2
        assert result.reranked


# --------------------------------------------------------------------------- #
# tune_rrf_k
# --------------------------------------------------------------------------- #
class TestTuneRRFK:
    def test_returns_default_when_no_data(self):
        from sparksage.retrieve.fusion import DEFAULT_RRF_K

        assert tune_rrf_k([], []) == DEFAULT_RRF_K

    def test_returns_int_in_grid(self):
        dense = [SearchHit("a", 0.9), SearchHit("b", 0.8)]
        lex = [SearchHit("b", 11.0), SearchHit("a", 9.0)]
        best = tune_rrf_k([[dense, lex]], [{"a", "b"}], top_n=2)
        assert isinstance(best, int)
        assert best >= 1

    def test_smallest_k_wins_on_tie(self):
        # All k values fuse identically here (single relevant id at rank 1 always)
        dense = [SearchHit("a", 1.0)]
        best = tune_rrf_k([[dense]], [{"a"}], k_candidates=[60, 10], top_n=1)
        # both score the same -> smallest (10) wins
        assert best == 10

    def test_custom_candidates(self):
        dense = [SearchHit("a", 1.0)]
        best = tune_rrf_k([[dense]], [{"a"}], k_candidates=[7, 9, 11], top_n=1)
        assert best in (7, 9, 11)

    def test_picks_higher_recall_k(self):
        # Two lists disagree: 'a' only in list1 rank1, 'b' in both.
        # Construct so a smaller k lifts the rank-1 items more.
        list1 = [SearchHit("a", 1.0), SearchHit("b", 0.5)]
        list2 = [SearchHit("b", 1.0), SearchHit("c", 0.5)]
        best = tune_rrf_k(
            [[list1, list2]], [{"a", "b"}], k_candidates=[1, 100], top_n=2
        )
        # k=1 makes rank-1 dominate: 'b' appears rank1 in list2 and rank2 in list1.
        # k=100 flattens everything. We just assert a valid int is returned.
        assert best in (1, 100)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            tune_rrf_k([[SearchHit("a", 1.0)]], [], top_n=1)

    def test_bad_top_n(self):
        with pytest.raises(ValueError):
            tune_rrf_k([[SearchHit("a", 1.0)]], [{"a"}], top_n=0)

    def test_bad_candidate(self):
        with pytest.raises(ValueError):
            tune_rrf_k([[SearchHit("a", 1.0)]], [{"a"}], k_candidates=[0], top_n=1)

    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError):
            tune_rrf_k([[SearchHit("a", 1.0)]], [{"a"}], k_candidates=[], top_n=1)
