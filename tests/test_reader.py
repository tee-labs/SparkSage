"""Tests for the reader layer: answer generation + faithfulness + abstention.

All tests run offline via :class:`FakeLLMClient`.
"""

from __future__ import annotations

import json

import pytest

from sparksage import FakeLLMClient, IdeaBlock, Tag
from sparksage.reader import (
    CoercionError,
    LLMAnswerGenerator,
    LLMFaithfulnessJudge,
    RawAnswer,
    RawCitation,
    RawFaithfulness,
    Reader,
    coerce_answer,
    coerce_citations,
    coerce_faithfulness,
    extract_json,
)
from sparksage.retrieve.models import RetrievedChunk
from sparksage.schema.source import SourceRef


def _block(name="A", loc="L1"):
    return IdeaBlock(
        name=name,
        critical_question="q?",
        trusted_answer="SparkSage deploys via pip install sparksage.",
        tags=[Tag.IMPORTANT],
        source=SourceRef(uri=f"file://{name}.md", title=name, locator=loc),
    )


def _chunk(block=None):
    return RetrievedChunk(block=block or _block(), score=0.9, rank=0)


def _answer_json(text="Use pip install sparksage.", cid=None, conf=0.9):
    if cid is None:
        cid = "ID"
    return json.dumps(
        {
            "reasoning": "r",
            "answer": text,
            "citations": [{"block_id": cid, "quote": "pip install sparksage"}],
            "confidence": conf,
        }
    )


class TestExtraction:
    def test_plain_json(self):
        assert json.loads(extract_json('{"a": 1}')) == {"a": 1}

    def test_fenced(self):
        assert json.loads(extract_json("```json\n{\"a\": 1}\n```")) == {"a": 1}

    def test_embedded(self):
        assert json.loads(extract_json('prose {"a": 1} tail')) == {"a": 1}

    def test_empty_raises(self):
        with pytest.raises(CoercionError):
            extract_json("   ")


class TestCoercion:
    def test_coerce_citations_drops_unknown(self):
        b1, b2 = _block("A"), _block("B")
        valid = {str(b1.id), str(b2.id)}
        raw = [RawCitation(block_id=str(b1.id), quote="q"), RawCitation(block_id="bogus")]
        out = coerce_citations(raw, valid)
        assert len(out) == 1
        assert out[0].block_id == str(b1.id)

    def test_coerce_answer_attaches_provenance(self):
        b = _block("A", "L42")
        id_to_cit = {str(b.id): _chunk(b).to_citation()}
        raw = RawAnswer(
            answer="x", citations=[RawCitation(block_id=str(b.id), quote="q")], confidence=2.0
        )
        ans = coerce_answer(raw, {str(b.id)}, id_to_cit, strict=False)
        assert ans.text == "x"
        assert ans.confidence == 1.0  # clamped
        assert ans.citations[0].locator == "L42"
        assert ans.citations[0].uri == "file://A.md"
        assert ans.grounded_block_ids == [str(b.id)]

    def test_coerce_answer_empty_abstains(self):
        raw = RawAnswer(answer="   ")
        ans = coerce_answer(raw, set(), {}, strict=False)
        assert ans.abstained
        assert ans.abstention_reason == "no answer produced"

    def test_coerce_answer_strict_empty_raises(self):
        raw = RawAnswer(answer="   ")
        with pytest.raises(CoercionError):
            coerce_answer(raw, set(), {}, strict=True)

    def test_coerce_faithfulness_clamps(self):
        raw = RawFaithfulness(score=5.0, supported_claims=3, unsupported_claims=1)
        out = coerce_faithfulness(raw, strict=False)
        assert out.score == 1.0
        assert out.supported_claims == 3


class TestGenerator:
    def test_generate_grounds_citations(self):
        b = _block("A", "L5")
        client = FakeLLMClient(responses=[_answer_json(cid=str(b.id))])
        gen = LLMAnswerGenerator(client)
        ans = gen.generate("how to deploy", [_chunk(b)])
        assert ans.text.startswith("Use pip")
        assert ans.citations[0].block_id == str(b.id)
        assert ans.citations[0].locator == "L5"
        assert ans.confidence == 0.9

    def test_generate_no_chunks_abstains(self):
        gen = LLMAnswerGenerator(FakeLLMClient(responses=[]))
        ans = gen.generate("q", [])
        assert ans.abstained
        assert gen._client.calls == []

    def test_generate_empty_response_raises(self):
        from sparksage.reader import AnswerEmptyResponseError

        client = FakeLLMClient(responses=["  "])
        with pytest.raises(AnswerEmptyResponseError):
            LLMAnswerGenerator(client).generate("q", [_chunk()])

    def test_generate_bad_json_raises(self):
        from sparksage.reader import AnswerResponseParseError

        client = FakeLLMClient(responses=["nope"])
        with pytest.raises(AnswerResponseParseError):
            LLMAnswerGenerator(client).generate("q", [_chunk()])


class TestFaithfulness:
    def test_judge_parses(self):
        client = FakeLLMClient(
            responses=[json.dumps({"reasoning": "r", "score": 0.9,
                                    "supported_claims": 2, "unsupported_claims": 0})]
        )
        out = LLMFaithfulnessJudge(client).judge("q", "ans", [_chunk()])
        assert out.score == 0.9
        assert out.supported_claims == 2

    def test_judge_empty_answer_zero(self):
        out = LLMFaithfulnessJudge(FakeLLMClient(responses=[])).judge("q", "  ", [_chunk()])
        assert out.score == 0.0

    def test_judge_fallback_on_bad_response(self):
        client = FakeLLMClient(responses=["garbage"])
        out = LLMFaithfulnessJudge(client).judge("q", "ans", [_chunk()])
        assert 0.0 <= out.score <= 1.0  # default, not crash

    def test_judge_strict_raises_on_empty(self):
        from sparksage.reader import FaithfulnessEmptyResponseError

        client = FakeLLMClient(responses=[""])
        with pytest.raises(FaithfulnessEmptyResponseError):
            LLMFaithfulnessJudge(client, strict=True).judge("q", "ans", [_chunk()])


class TestReader:
    def test_answer_when_faithful(self):
        b = _block()
        # generator then judge
        client = FakeLLMClient(
            responses=[
                _answer_json(cid=str(b.id)),
                json.dumps({"score": 0.9, "supported_claims": 1, "unsupported_claims": 0}),
            ]
        )
        reader = Reader(
            generator=LLMAnswerGenerator(client),
            faithfulness_judge=LLMFaithfulnessJudge(client),
        )
        result = reader.answer("how to deploy", [_chunk(b)])
        assert not result.abstained
        assert result.confidence == pytest.approx(0.9 * 0.9, rel=1e-3)
        assert result.faithfulness is not None

    def test_abstain_on_low_faithfulness(self):
        b = _block()
        client = FakeLLMClient(
            responses=[
                _answer_json(cid=str(b.id)),
                json.dumps({"score": 0.1, "supported_claims": 0, "unsupported_claims": 2}),
            ]
        )
        reader = Reader(
            generator=LLMAnswerGenerator(client),
            faithfulness_judge=LLMFaithfulnessJudge(client),
            min_faithfulness=0.5,
        )
        result = reader.answer("q", [_chunk(b)])
        assert result.abstained
        assert "faithfulness" in (result.abstention_reason or "")

    def test_abstain_on_no_chunks(self):
        reader = Reader(generator=LLMAnswerGenerator(FakeLLMClient(responses=[])))
        result = reader.answer("q", [])
        assert result.abstained

    def test_abstain_on_low_confidence(self):
        b = _block()
        client = FakeLLMClient(responses=[_answer_json(cid=str(b.id), conf=0.1)])
        reader = Reader(
            generator=LLMAnswerGenerator(client),
            min_confidence=0.5,
        )
        result = reader.answer("q", [_chunk(b)])
        assert result.abstained
        assert "confidence" in (result.abstention_reason or "")

    def test_no_judge_skips_faithfulness_gate(self):
        b = _block()
        client = FakeLLMClient(responses=[_answer_json(cid=str(b.id), conf=0.9)])
        reader = Reader(generator=LLMAnswerGenerator(client))
        result = reader.answer("q", [_chunk(b)])
        assert not result.abstained
        assert result.faithfulness is None

    def test_bad_generator_type(self):
        with pytest.raises(TypeError):
            Reader(generator="not a generator")  # type: ignore[arg-type]

    def test_bad_thresholds(self):
        with pytest.raises(ValueError):
            Reader(generator=LLMAnswerGenerator(FakeLLMClient([])), min_faithfulness=2.0)
        with pytest.raises(ValueError):
            Reader(generator=LLMAnswerGenerator(FakeLLMClient([])), min_confidence=-0.1)
