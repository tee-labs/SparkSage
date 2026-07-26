"""Tests for query expansion + semantic cache (Phase 3)."""

from __future__ import annotations

import json

import pytest

from sparksage import FakeEmbeddingClient, FakeLLMClient
from sparksage.query import (
    DEFAULT_N_VARIANTS,
    IdentityExpander,
    InMemorySemanticCache,
    LLMQueryExpander,
    semantic_cache_stats,
)


class TestExpander:
    def test_identity(self):
        assert IdentityExpander().expand("how to deploy") == ["how to deploy"]
        assert IdentityExpander().expand("") == []

    def test_llm_expand_parses_variants(self):
        client = FakeLLMClient(
            responses=[json.dumps({"variants": ["install steps", "setup guide"]})]
        )
        out = LLMQueryExpander(client).expand("how to deploy", n=3)
        assert out == ["install steps", "setup guide"]

    def test_llm_expand_dedupes_and_caps(self):
        client = FakeLLMClient(
            responses=[json.dumps({"variants": ["a", "a", "b", "c", "d"]})]
        )
        out = LLMQueryExpander(client).expand("q", n=2)
        assert len(out) == 2
        assert len(set(out)) == 2

    def test_llm_expand_includes_original_context_dedup(self):
        # original is deduped out
        client = FakeLLMClient(responses=[json.dumps({"variants": ["q", "x"]})])
        out = LLMQueryExpander(client).expand("q", n=3)
        assert out == ["x"]

    def test_llm_expand_fallback_on_bad_json(self):
        client = FakeLLMClient(responses=["nope"])
        exp = LLMQueryExpander(client)
        out = exp.expand("how to deploy")
        assert out == ["how to deploy"]
        assert exp.fallbacks >= 1

    def test_llm_expand_empty_and_n1(self):
        client = FakeLLMClient(responses=[])
        exp = LLMQueryExpander(client)
        assert exp.expand("") == []
        assert exp.expand("q", n=1) == ["q"]

    def test_default_n(self):
        assert DEFAULT_N_VARIANTS == 3


class TestSemanticCache:
    def test_store_and_lookup_same(self):
        cache = InMemorySemanticCache(
            FakeEmbeddingClient(dimension=64), threshold=0.999
        )
        cache.store("how to deploy", "ans")
        assert cache.lookup("how to deploy") == "ans"
        stats = semantic_cache_stats(cache)
        assert stats.hits == 1

    def test_miss_on_different(self):
        cache = InMemorySemanticCache(
            FakeEmbeddingClient(dimension=128), threshold=0.99
        )
        cache.store("how to deploy", "ans")
        # very different tokens -> below threshold
        assert cache.lookup("zzz qqq xxx") is None
        assert semantic_cache_stats(cache).misses == 1

    def test_empty_query_miss(self):
        cache = InMemorySemanticCache(FakeEmbeddingClient(dimension=64))
        assert cache.lookup("   ") is None

    def test_max_entries_eviction(self):
        cache = InMemorySemanticCache(
            FakeEmbeddingClient(dimension=32), max_entries=2
        )
        cache.store("a", 1)
        cache.store("b", 2)
        cache.store("c", 3)
        assert len(cache) == 2

    def test_bad_threshold(self):
        with pytest.raises(ValueError):
            InMemorySemanticCache(FakeEmbeddingClient(dimension=64), threshold=2.0)

    def test_clear(self):
        cache = InMemorySemanticCache(FakeEmbeddingClient(dimension=64))
        cache.store("a", 1)
        cache.clear()
        assert len(cache) == 0

    def test_key_extractor(self):
        # store receives a value; key extracted from it
        cache = InMemorySemanticCache(
            FakeEmbeddingClient(dimension=64),
            threshold=0.999,
            key_extractor=lambda v: v["q"],
        )
        cache.store("ignored", {"q": "real query", "ans": "x"})
        assert cache.lookup("real query") == {"q": "real query", "ans": "x"}
