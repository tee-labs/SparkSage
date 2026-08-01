"""Tests for the embedding layer.

All tests run fully offline:

* :class:`FakeEmbeddingClient` is deterministic and dependency-free.
* :class:`OpenAIEmbeddingClient` is tested with the ``openai`` SDK mocked out
  via ``sys.modules`` (no network, no real key), asserting on the *request
  shape* (batching, model) and on vector assembly/ordering.
"""

from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from sparksage.embed import (
    BlockEmbedder,
    EmbeddingClient,
    FakeEmbeddingClient,
    OpenAIEmbeddingClient,
)
from sparksage.embed.client import _KNOWN_DIMS, _l2_normalize
from sparksage.schema import IdeaBlock


# ---------------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------------- #
def _make_block(
    name: str = "Block",
    question: str = "What is this?",
    answer: str = "A short verified answer.",
) -> IdeaBlock:
    return IdeaBlock(name=name, critical_question=question, trusted_answer=answer)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


# ---------------------------------------------------------------------------- #
# pure-Python helpers
# ---------------------------------------------------------------------------- #
class TestNormalize:
    def test_unit_length(self):
        out = _l2_normalize([3.0, 4.0])
        assert _norm(out) == pytest.approx(1.0)
        assert out == pytest.approx([0.6, 0.8])

    def test_zero_vector_unchanged(self):
        assert _l2_normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------- #
# FakeEmbeddingClient
# ---------------------------------------------------------------------------- #
class TestFakeEmbeddingClient:
    def test_is_an_embedding_client(self):
        assert isinstance(FakeEmbeddingClient(), EmbeddingClient)

    def test_default_dimension(self):
        assert FakeEmbeddingClient().dimension == 128

    def test_custom_dimension(self):
        client = FakeEmbeddingClient(dimension=8)
        assert client.dimension == 8
        vec = client.embed_batch(["hello"])[0]
        assert len(vec) == 8

    def test_deterministic(self):
        client = FakeEmbeddingClient(dimension=64)
        a = client.embed_batch(["the same text"])[0]
        b = client.embed_batch(["the same text"])[0]
        assert a == b

    def test_normalized(self):
        client = FakeEmbeddingClient(dimension=32)
        for vec in client.embed_batch(["alpha", "beta gamma delta", "z"]):
            assert _norm(vec) == pytest.approx(1.0, abs=1e-9)

    def test_empty_input(self):
        assert FakeEmbeddingClient().embed_batch([]) == []

    def test_count_matches_input(self):
        client = FakeEmbeddingClient(dimension=16)
        out = client.embed_batch(["one", "two", "three"])
        assert len(out) == 3
        assert all(len(v) == 16 for v in out)

    def test_semantic_overlap_higher_similarity(self):
        client = FakeEmbeddingClient(dimension=512)
        base = client.embed_batch(["how to deploy sparksage locally"])[0]
        similar = client.embed_batch(["how to deploy sparksage locally fast"])[0]
        dissimilar = client.embed_batch(["a recipe for chocolate cake"])[0]
        assert _dot(base, similar) > _dot(base, dissimilar)


# ---------------------------------------------------------------------------- #
# OpenAIEmbeddingClient (mocked openai SDK)
# ---------------------------------------------------------------------------- #
class _FakeEmbeddingsCreate:
    """Stand-in for ``client.embeddings.create``."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any):
        self.calls.append(kwargs)
        inp = kwargs["input"]
        data = [SimpleNamespace(embedding=self._vec(s)) for s in inp]
        return SimpleNamespace(data=data)

    def _vec(self, text: str) -> list[float]:
        # Deterministic, text-dependent vector (not normalized on purpose so we
        # can assert that the client normalizes itself).
        import hashlib

        digest = hashlib.sha256(text.encode()).digest()
        return [float((b / 255.0) - 0.5) for b in digest[: self.dim]]


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dim: int = 8,
):
    create = _FakeEmbeddingsCreate(dim=dim)
    module = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs = kwargs
            self.embeddings = SimpleNamespace(create=create)

    module.OpenAI = _OpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    return create


class TestOpenAIEmbeddingClientWiring:
    def test_base_url_and_api_key_forwarded(self, monkeypatch):
        _install_fake_openai(monkeypatch)
        client = OpenAIEmbeddingClient(
            base_url="https://my.host/v1", api_key="secret", model="m"
        )
        assert client._client.init_kwargs["base_url"] == "https://my.host/v1"
        assert client._client.init_kwargs["api_key"] == "secret"

    def test_known_dimension_without_probe(self, monkeypatch):
        _install_fake_openai(monkeypatch)
        client = OpenAIEmbeddingClient(model="text-embedding-3-small")
        assert client.dimension == _KNOWN_DIMS["text-embedding-3-small"]
        # never made an API call just to read the dimension
        assert client._dim is not None

    def test_explicit_dimension_overrides(self, monkeypatch):
        _install_fake_openai(monkeypatch)
        client = OpenAIEmbeddingClient(model="text-embedding-3-small", dimension=42)
        assert client.dimension == 42

    def test_unknown_dimension_probed_on_access(self, monkeypatch):
        _install_fake_openai(monkeypatch, dim=7)
        client = OpenAIEmbeddingClient(model="custom-model")
        assert client.dimension == 7

    def test_unknown_dimension_raises_when_probe_fails(self, monkeypatch):
        def _boom(**kwargs: Any):
            raise RuntimeError("endpoint unreachable")

        module = types.ModuleType("openai")

        class _OpenAI:
            def __init__(self, **kwargs: Any) -> None:
                self.embeddings = SimpleNamespace(create=_boom)

        module.OpenAI = _OpenAI
        monkeypatch.setitem(sys.modules, "openai", module)
        client = OpenAIEmbeddingClient(model="custom-model")
        with pytest.raises(RuntimeError, match="dimension"):
            _ = client.dimension


class TestOpenAIEmbeddingClientEmbed:
    def test_returns_one_vector_per_input(self, monkeypatch):
        _install_fake_openai(monkeypatch, dim=8)
        client = OpenAIEmbeddingClient(model="m", normalize=False)
        out = client.embed_batch(["a", "b", "c"])
        assert len(out) == 3
        assert all(len(v) == 8 for v in out)

    def test_empty_input_no_calls(self, monkeypatch):
        create = _install_fake_openai(monkeypatch)
        client = OpenAIEmbeddingClient(model="m")
        assert client.embed_batch([]) == []
        assert create.calls == []

    def test_model_forwarded(self, monkeypatch):
        create = _install_fake_openai(monkeypatch)
        client = OpenAIEmbeddingClient(model="my-embed-model")
        client.embed_batch(["x"])
        assert create.calls[-1]["model"] == "my-embed-model"

    def test_normalizes_by_default(self, monkeypatch):
        _install_fake_openai(monkeypatch, dim=8)
        client = OpenAIEmbeddingClient(model="m")  # normalize=True
        out = client.embed_batch(["a", "b"])
        for v in out:
            assert _norm(v) == pytest.approx(1.0, abs=1e-9)

    def test_normalize_off(self, monkeypatch):
        _install_fake_openai(monkeypatch, dim=8)
        client = OpenAIEmbeddingClient(model="m", normalize=False)
        raw = client.embed_batch(["a"])[0]
        # raw mock vectors are unlikely to be unit length
        assert abs(_norm(raw) - 1.0) > 1e-6

    def test_dimension_probed_after_first_call(self, monkeypatch):
        _install_fake_openai(monkeypatch, dim=12)
        client = OpenAIEmbeddingClient(model="custom-model")
        client.embed_batch(["x"])
        assert client.dimension == 12

    def test_single_batch_no_threadpool(self, monkeypatch):
        create = _install_fake_openai(monkeypatch, dim=4)
        client = OpenAIEmbeddingClient(model="m", normalize=False)
        client.embed_batch(["a", "b"])
        assert len(create.calls) == 1
        assert create.calls[0]["input"] == ["a", "b"]

    def test_batching_splits_at_batch_size(self, monkeypatch):
        create = _install_fake_openai(monkeypatch, dim=4)
        client = OpenAIEmbeddingClient(model="m", batch_size=2, normalize=False)
        texts = ["t0", "t1", "t2", "t3", "t4"]
        out = client.embed_batch(texts)
        # 5 texts / batch_size 2 -> 3 calls (2, 2, 1)
        assert len(create.calls) == 3
        assert len(out) == 5

    def test_batching_preserves_order(self, monkeypatch):
        _install_fake_openai(monkeypatch, dim=8)
        client = OpenAIEmbeddingClient(model="m", batch_size=2, normalize=False)
        texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
        out = client.embed_batch(texts)
        # Each output is text-dependent, so order == input order.
        expected = _FakeEmbeddingsCreate(dim=8)
        for text, vec in zip(texts, out, strict=True):
            assert vec == expected._vec(text)


class TestOpenAIEmbeddingClientValidation:
    def test_batch_size_zero_rejected(self, monkeypatch):
        _install_fake_openai(monkeypatch)
        with pytest.raises(ValueError, match="batch_size"):
            OpenAIEmbeddingClient(model="m", batch_size=0)

    def test_batch_size_over_api_limit_rejected(self, monkeypatch):
        _install_fake_openai(monkeypatch)
        with pytest.raises(ValueError, match="2048"):
            OpenAIEmbeddingClient(model="m", batch_size=2049)

    def test_max_workers_zero_rejected(self, monkeypatch):
        _install_fake_openai(monkeypatch)
        with pytest.raises(ValueError, match="max_workers"):
            OpenAIEmbeddingClient(model="m", max_workers=0)

    def test_missing_openai_raises_install_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises(ImportError, match="sparksage\\[embed\\]"):
            OpenAIEmbeddingClient(api_key="k")


# ---------------------------------------------------------------------------- #
# BlockEmbedder
# ---------------------------------------------------------------------------- #
class TestBlockEmbedder:
    def test_default_dimension_proxies_client(self):
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        assert embedder.dimension == 64

    def test_embed_blocks_fills_embedding(self):
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=32))
        blocks = [_make_block("A", "What is a?", "answer a"), _make_block("B", "q?", "v")]
        embedder.embed_blocks(blocks)
        for b in blocks:
            assert b.embedding is not None
            assert len(b.embedding) == 32
            assert _norm(b.embedding) == pytest.approx(1.0, abs=1e-9)

    def test_embed_blocks_empty_list(self):
        embedder = BlockEmbedder(FakeEmbeddingClient())
        assert embedder.embed_blocks([]) == []

    def test_embed_blocks_returns_same_list(self):
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=8))
        blocks = [_make_block()]
        assert embedder.embed_blocks(blocks) is blocks

    def test_embed_blocks_uses_embedding_text(self):
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=128))
        block = _make_block("MyName", "My question?", "My answer.")
        embedder.embed_blocks([block])
        # Manually compute the expected vector from embedding_text.
        client = FakeEmbeddingClient(dimension=128)
        expected = client._embed(block.embedding_text)
        assert block.embedding == pytest.approx(expected)
        # Sanity: embedding_text is the three-field concatenation.
        assert block.embedding_text == "MyName\nMy question?\nMy answer."

    def test_identical_blocks_get_identical_embeddings(self):
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=64))
        b1 = _make_block("Same", "Same question?", "Same answer.")
        b2 = _make_block("Same", "Same question?", "Same answer.")
        embedder.embed_blocks([b1, b2])
        assert b1.embedding == b2.embedding

    def test_vectors_for_does_not_mutate(self):
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=16))
        blocks = [_make_block("A", "q1?", "a1"), _make_block("B", "q2?", "a2")]
        vectors = embedder.vectors_for(blocks)
        assert set(vectors) == {str(b.id) for b in blocks}
        assert all(len(v) == 16 for v in vectors.values())
        # blocks untouched
        for b in blocks:
            assert b.embedding is None

    def test_vectors_for_empty(self):
        embedder = BlockEmbedder(FakeEmbeddingClient())
        assert embedder.vectors_for([]) == {}

    def test_embed_texts(self):
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=8))
        out = embedder.embed_texts(["hello", "world"])
        assert len(out) == 2
        assert all(len(v) == 8 for v in out)

    def test_embed_texts_empty(self):
        embedder = BlockEmbedder(FakeEmbeddingClient())
        assert embedder.embed_texts([]) == []

    def test_accepts_protocol_compliant_duck_type(self):
        class _Min:
            @property
            def dimension(self) -> int:
                return 4

            def embed_batch(self, texts):
                return [[0.1] * 4 for _ in texts]

        embedder = BlockEmbedder(_Min())  # type: ignore[arg-type]
        assert embedder.dimension == 4
        blocks = [_make_block()]
        embedder.embed_blocks(blocks)
        assert blocks[0].embedding == [0.1, 0.1, 0.1, 0.1]


# ---------------------------------------------------------------------------- #
# BlockEmbedder -- Contextual Retrieval context_prefix
# ---------------------------------------------------------------------------- #
class TestContextPrefix:
    def test_constructor_prefix_default_none(self):
        assert BlockEmbedder(FakeEmbeddingClient()).context_prefix is None

    def test_constructor_prefix_stored(self):
        e = BlockEmbedder(FakeEmbeddingClient(), context_prefix="doc summary")
        assert e.context_prefix == "doc summary"

    def test_prefix_changes_embedded_vector_embed_blocks(self):
        block = _make_block("A", "What is a?", "answer a")
        client = FakeEmbeddingClient(dimension=128)
        plain = BlockEmbedder(client).embed_blocks([block])[0].embedding
        block2 = _make_block("A", "What is a?", "answer a")
        prefixed = BlockEmbedder(client, context_prefix="CONTEXT").embed_blocks([block2])[0]
        assert prefixed.embedding is not None
        assert prefixed.embedding != plain

    def test_prefix_prepended_text_matches_manual(self):
        block = _make_block("MyName", "My question?", "My answer.")
        client = FakeEmbeddingClient(dimension=128)
        BlockEmbedder(client, context_prefix="PRE").embed_blocks([block])
        expected = client._embed(f"PRE\n{block.embedding_text}")
        assert block.embedding == pytest.approx(expected)

    def test_prefix_does_not_mutate_embedding_text_property(self):
        block = _make_block("A", "q?", "a")
        before = block.embedding_text
        BlockEmbedder(FakeEmbeddingClient(dimension=64), context_prefix="CTX").embed_blocks([block])
        assert block.embedding_text == before  # property unchanged

    def test_prefix_applied_to_vectors_for(self):
        block = _make_block("A", "q?", "a")
        client = FakeEmbeddingClient(dimension=128)
        plain = BlockEmbedder(client).vectors_for([block])[str(block.id)]
        block2 = _make_block("A", "q?", "a")
        prefixed = BlockEmbedder(client, context_prefix="CTX").vectors_for([block2])[str(block2.id)]
        assert prefixed != plain
        # and not stored on the block
        assert block2.embedding is None

    def test_per_call_override_takes_precedence(self):
        block = _make_block("A", "q?", "a")
        client = FakeEmbeddingClient(dimension=128)
        e = BlockEmbedder(client, context_prefix="CTOR")
        via_ctor = e.embed_blocks([block])[0].embedding
        block2 = _make_block("A", "q?", "a")
        via_override = e.embed_blocks([block2], context_prefix="OVERRIDE")[0].embedding
        assert via_override != via_ctor

    def test_per_call_none_disables_constructor_prefix(self):
        block = _make_block("A", "q?", "a")
        client = FakeEmbeddingClient(dimension=128)
        plain = BlockEmbedder(client).embed_blocks([block])[0].embedding
        block2 = _make_block("A", "q?", "a")
        e = BlockEmbedder(client, context_prefix="CTOR")
        via_none = e.embed_blocks([block2], context_prefix=None)[0].embedding
        assert via_none == plain  # no prefix -> same as plain

    def test_embed_texts_ignores_prefix(self):
        e = BlockEmbedder(FakeEmbeddingClient(dimension=128), context_prefix="CTX")
        plain_e = BlockEmbedder(FakeEmbeddingClient(dimension=128))
        # query side must NOT carry the prefix
        assert e.embed_texts(["how to deploy"]) == plain_e.embed_texts(["how to deploy"])

    def test_empty_blocks_with_prefix_returns_empty(self):
        e = BlockEmbedder(FakeEmbeddingClient(), context_prefix="CTX")
        assert e.embed_blocks([], context_prefix="x") == []
        assert e.vectors_for([], context_prefix="x") == {}

    def test_bad_prefix_type_raises(self):
        e = BlockEmbedder(FakeEmbeddingClient())
        with pytest.raises(TypeError):
            e.embed_blocks([_make_block()], context_prefix=123)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            e.vectors_for([_make_block()], context_prefix=123)  # type: ignore[arg-type]

    def test_empty_prefix_string_is_no_prefix(self):
        block = _make_block("A", "q?", "a")
        client = FakeEmbeddingClient(dimension=128)
        plain = BlockEmbedder(client).embed_blocks([block])[0].embedding
        block2 = _make_block("A", "q?", "a")
        empty_pref = BlockEmbedder(client, context_prefix="").embed_blocks([block2])[0].embedding
        assert empty_pref == plain


# ---------------------------------------------------------------------------- #
# TechnicalBlock embedding_text is respected
# ---------------------------------------------------------------------------- #
class TestTechnicalBlockEmbedding:
    def test_technical_block_uses_steps_text(self):
        from sparksage.schema.enums import SentenceRole
        from sparksage.schema.technical import AnnotatedSentence, TechnicalBlock

        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=128))
        block = TechnicalBlock(
            name="Deploy",
            critical_question="How to deploy?",
            trusted_answer="Run pip install.",
            steps=[
                AnnotatedSentence(text="install python", role=SentenceRole.COMMAND),
            ],
        )
        embedder.embed_blocks([block])
        client = FakeEmbeddingClient(dimension=128)
        expected = client._embed(block.embedding_text)
        assert block.embedding == pytest.approx(expected)
