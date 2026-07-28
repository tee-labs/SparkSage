"""Tests for the concrete :class:`Reranker` backends.

These exercise the production glue (lazy SDK import, pair scoring, ordering,
sigmoid normalization, ``top_n`` truncation, protocol compliance) fully offline:
the optional ``sentence_transformers`` SDK is mocked out via ``sys.modules`` --
the same pattern used by ``test_embed_backends.py`` for the vector-store SDKs.
That keeps the suite zero-dependency while still driving the production paths.
"""

from __future__ import annotations

import sys
import types

import pytest

from sparksage import IdeaBlock
from sparksage.retrieve import RetrievedChunk
from sparksage.retrieve.backends import (
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderReranker,
)
from sparksage.retrieve.reranker import Reranker
from sparksage.schema.source import SourceRef


def _block(name, answer):
    return IdeaBlock(
        name=name,
        critical_question="q?",
        trusted_answer=answer,
        source=SourceRef(uri=f"file://{name}.md", title=name, locator="L1"),
    )


def _chunks(*blocks):
    return [RetrievedChunk(block=b, score=0.5, rank=i) for i, b in enumerate(blocks)]


# ---------------------------------------------------------------------------- #
# fake sentence_transformers.CrossEncoder (deterministic pair scoring)
# ---------------------------------------------------------------------------- #
class _FakeCrossEncoder:
    """Records the model_name and scores pairs by a scripted callable."""

    instances: list[_FakeCrossEncoder] = []

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = dict(kwargs)
        self.predict_calls: list[list[tuple[str, str]]] = []
        _FakeCrossEncoder.instances.append(self)

    def predict(self, pairs):
        self.predict_calls.append([tuple(p) for p in pairs])
        # Score each pair by summed term-frequency overlap: for every query
        # token, add how many times it appears in the document. Higher overlap
        # -> higher score -> ranked first. Deterministic & offline.
        from collections import Counter

        scores = []
        for query, doc in pairs:
            q_tokens = (query or "").lower().split()
            if not q_tokens:
                scores.append(0.0)
                continue
            d_counts = Counter((doc or "").lower().split())
            scores.append(float(sum(d_counts.get(qt, 0) for qt in q_tokens)))
        return scores


@pytest.fixture(autouse=True)
def _install_fake_st(monkeypatch):
    _FakeCrossEncoder.instances.clear()
    mod = types.ModuleType("sentence_transformers")
    mod.CrossEncoder = _FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    yield
    _FakeCrossEncoder.instances.clear()


# ---------------------------------------------------------------------------- #
# construction / lazy import
# ---------------------------------------------------------------------------- #
class TestConstruction:
    def test_import_error_when_sdk_missing(self, monkeypatch):
        # simulate the SDK being genuinely unavailable
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(ImportError) as exc:
            CrossEncoderReranker()
        assert "sparksage[rerank]" in str(exc.value)

    def test_default_model_name(self):
        rr = CrossEncoderReranker()
        assert rr.model_name == DEFAULT_CROSS_ENCODER_MODEL
        assert rr.apply_sigmoid is True
        assert isinstance(rr, Reranker)

    def test_custom_model_and_kwargs_forwarded(self):
        CrossEncoderReranker(
            "BAAI/bge-reranker-v2-m3",
            max_length=512,
            device="cpu",
            trust_remote_code=True,
        )
        inst = _FakeCrossEncoder.instances[-1]
        assert inst.model_name == "BAAI/bge-reranker-v2-m3"
        assert inst.kwargs == {"max_length": 512, "device": "cpu",
                               "trust_remote_code": True}

    def test_bad_model_name(self):
        with pytest.raises(TypeError):
            CrossEncoderReranker(model_name="")  # type: ignore[arg-type]

    def test_bad_apply_sigmoid(self):
        with pytest.raises(TypeError):
            CrossEncoderReranker(apply_sigmoid="yes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------- #
# rerank behaviour
# ---------------------------------------------------------------------------- #
class TestRerank:
    def test_empty_and_single(self):
        rr = CrossEncoderReranker()
        assert rr.rerank("deploy", []) == []
        one = _chunks(_block("A", "deploy procedure"))
        out = rr.rerank("deploy", one)
        assert len(out) == 1
        assert out[0].rank == 0

    def test_reorders_best_first(self):
        rr = CrossEncoderReranker()
        chunks = _chunks(
            _block("A", "cooking recipe apples"),   # 0 overlap with "deploy"
            _block("B", "deploy procedure steps"),  # 1 overlap
        )
        out = rr.rerank("deploy", chunks)
        assert [c.block.name for c in out] == ["B", "A"]
        assert out[0].rank == 0
        assert out[0].score >= out[1].score

    def test_predict_receives_pairs(self):
        rr = CrossEncoderReranker()
        chunks = _chunks(_block("A", "deploy"), _block("B", "other"))
        rr.rerank("deploy spark", chunks)
        inst = _FakeCrossEncoder.instances[-1]
        pairs = inst.predict_calls[-1]
        assert pairs[0][0] == "deploy spark"
        # candidate text is name + question + answer
        assert "deploy" in pairs[0][1]

    def test_sigmoid_normalizes_to_unit_interval(self):
        # Force large raw scores by using many overlapping tokens.
        rr = CrossEncoderReranker(apply_sigmoid=True)
        chunks = _chunks(_block("A", "deploy deploy deploy deploy"))
        out = rr.rerank("deploy", chunks)
        assert 0.0 < out[0].score < 1.0

    def test_raw_logits_when_sigmoid_off(self):
        rr = CrossEncoderReranker(apply_sigmoid=False)
        chunks = _chunks(
            _block("A", "deploy deploy"),
            _block("B", "unrelated"),
        )
        out = rr.rerank("deploy", chunks)
        # raw score == overlap count (2.0 for A)
        assert out[0].block.name == "A"
        assert out[0].score == 2.0

    def test_top_n_truncates(self):
        rr = CrossEncoderReranker()
        chunks = _chunks(
            _block("A", "deploy"),
            _block("B", "deploy deploy"),
            _block("C", "deploy deploy deploy"),
        )
        out = rr.rerank("deploy", chunks, top_n=2)
        assert len(out) == 2
        assert [c.rank for c in out] == [0, 1]

    def test_bad_top_n(self):
        rr = CrossEncoderReranker()
        chunks = _chunks(_block("A", "deploy"))
        with pytest.raises(ValueError):
            rr.rerank("deploy", chunks, top_n=-1)
        with pytest.raises(TypeError):
            rr.rerank("deploy", chunks, top_n=1.0)  # type: ignore[arg-type]

    def test_scores_and_ranks_reassigned(self):
        rr = CrossEncoderReranker()  # sigmoid on by default
        chunks = _chunks(
            _block("A", "nope"),
            _block("B", "deploy deploy"),
        )
        out = rr.rerank("deploy", chunks)
        assert [c.rank for c in out] == [0, 1]
        # B is the best (raw 2.0 -> sigmoid(2.0) in (0.5, 1)); A is raw 0.0 ->
        # sigmoid(0.0) == 0.5. The reranker's own scores supersede the input 0.5.
        assert out[0].block.name == "B"
        assert out[0].score > 0.5
        assert out[0].score > out[1].score

    def test_protocol_compatibility_with_retriever(self):
        # The Retriever accepts any Reranker; ensure the backend slots in.
        from sparksage import BlockEmbedder, FakeEmbeddingClient, InMemoryVectorStore
        from sparksage.retrieve import Retriever

        blocks = [
            _block("A", "deploy procedure"),
            _block("B", "cooking apples"),
        ]
        dim = 64
        embedder = BlockEmbedder(FakeEmbeddingClient(dimension=dim))
        store = InMemoryVectorStore(dimension=dim)
        registry: dict[str, IdeaBlock] = {}
        retriever = Retriever(
            registry, store, embedder,
            reranker=CrossEncoderReranker(),
            min_fetch=5,
        )
        retriever.index(blocks)
        result = retriever.search("deploy", k=2, use_rerank=True)
        assert result.reranked
        assert len(result.chunks) <= 2


# ---------------------------------------------------------------------------- #
# repr
# ---------------------------------------------------------------------------- #
class TestRepr:
    def test_repr(self):
        rr = CrossEncoderReranker()
        assert "CrossEncoderReranker" in repr(rr)
        assert DEFAULT_CROSS_ENCODER_MODEL in repr(rr)


# ---------------------------------------------------------------------------- #
# contrast: LLMReranker still works (sanity for the protocol contract)
# ---------------------------------------------------------------------------- #
class TestProtocolContract:
    def test_cross_encoder_is_a_reranker(self):
        assert isinstance(CrossEncoderReranker(), Reranker)
