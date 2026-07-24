"""Tests for LLM-driven IdeaBlock generation.

All tests run offline via :class:`FakeLLMClient`, so no network or API key is
required. The fake returns canned JSON, exercising the real prompt-building,
JSON-extraction, enum-mapping and strict-coercion logic.
"""

from __future__ import annotations

import json

import pytest

from sparksage.generator import (
    CoercionError,
    EmptyResponseError,
    FakeLLMClient,
    GenerationError,
    IdeaBlockGenerator,
    ResponseParseError,
    coerce_block,
)
from sparksage.generator.generator import _extract_json
from sparksage.generator.prompts import build_messages, system_prompt
from sparksage.generator.schema import RawEntity, RawIdeaBlock
from sparksage.schema import IdeaBlock, Tag
from sparksage.schema.enums import BlockStatus, EntityType
from sparksage.schema.source import SourceRef

SAMPLE_TEXT = (
    "SparkSage replaces naive text slicing with question-aligned chunks. "
    "Each IdeaBlock carries a critical question and a concise trusted answer. "
    "Only the trusted_answer field is embedded, which avoids mid-sentence cuts."
)

TWO_BLOCK_JSON = json.dumps(
    {
        "blocks": [
            {
                "name": "What SparkSage does",
                "critical_question": "What problem does SparkSage solve?",
                "trusted_answer": (
                    "SparkSage turns documents into question-aligned knowledge units "
                    "so retrieval returns whole answers instead of text shards."
                ),
                "tags": ["IMPORTANT", "TECHNOLOGY"],
                "entities": [
                    {"entity_name": "SparkSage", "entity_type": "PRODUCT"}
                ],
                "keywords": ["rag", "chunking"],
            },
            {
                "name": "Embedding strategy",
                "critical_question": "Which field is embedded?",
                "trusted_answer": (
                    "Only the trusted_answer field is embedded, avoiding "
                    "mid-sentence cuts that wreck naive chunking."
                ),
                "tags": ["ARCHITECTURE"],
                "keywords": ["embedding"],
            },
        ]
    }
)


class TestHappyPath:
    def test_generates_multiple_valid_blocks(self):
        gen = IdeaBlockGenerator(FakeLLMClient(responses=[TWO_BLOCK_JSON]))
        blocks = gen.generate(SAMPLE_TEXT)

        assert len(blocks) == 2
        assert all(isinstance(b, IdeaBlock) for b in blocks)
        assert blocks[0].critical_question.endswith("?")
        assert blocks[0].trusted_answer[0].isalpha()
        assert Tag.IMPORTANT in blocks[0].tags
        assert Tag.TECHNOLOGY in blocks[0].tags
        assert blocks[0].entities[0].entity_type == EntityType.PRODUCT
        assert blocks[1].entities == []

    def test_blocks_are_draft_and_round_trip(self):
        gen = IdeaBlockGenerator(FakeLLMClient(responses=[TWO_BLOCK_JSON]))
        blocks = gen.generate(SAMPLE_TEXT)

        for b in blocks:
            assert b.status == BlockStatus.DRAFT
            restored = IdeaBlock.model_validate(b.model_dump())
            assert restored == b

    def test_generated_blocks_embeddable(self):
        gen = IdeaBlockGenerator(FakeLLMClient(responses=[TWO_BLOCK_JSON]))
        blocks = gen.generate(SAMPLE_TEXT)

        for b in blocks:
            et = b.embedding_text
            assert b.name in et
            assert b.critical_question in et
            assert b.trusted_answer in et

    def test_generate_with_stats_reports_counts(self):
        gen = IdeaBlockGenerator(FakeLLMClient(responses=[TWO_BLOCK_JSON]))
        blocks, stats = gen.generate_with_stats(SAMPLE_TEXT)

        assert stats.raw_block_count == 2
        assert stats.emitted == 2
        assert stats.skipped == 0
        assert stats.errors == []
        assert len(blocks) == 2


class TestProvenance:
    def test_source_attached_to_every_block(self):
        gen = IdeaBlockGenerator(FakeLLMClient(responses=[TWO_BLOCK_JSON]))
        src = SourceRef(uri="file://docs/overview.md", title="Overview", locator="§1")
        blocks = gen.generate(SAMPLE_TEXT, source=src)

        for b in blocks:
            assert b.source is not None
            assert b.source.uri == "file://docs/overview.md"
            assert b.source.title == "Overview"

    def test_source_uri_shortcut_builds_source_ref(self):
        gen = IdeaBlockGenerator(FakeLLMClient(responses=[TWO_BLOCK_JSON]))
        blocks = gen.generate(
            SAMPLE_TEXT, source_uri="https://example.com/doc", source_title="Doc"
        )
        assert blocks[0].source == SourceRef(
            uri="https://example.com/doc", title="Doc"
        )

    def test_language_propagated_to_blocks(self):
        gen = IdeaBlockGenerator(
            FakeLLMClient(responses=[TWO_BLOCK_JSON]), language="zh"
        )
        blocks = gen.generate(SAMPLE_TEXT)
        assert all(b.language == "zh" for b in blocks)


class TestPromptBuilding:
    def test_prompt_contains_live_vocabularies(self):
        sysp = system_prompt()
        # Vocabulary is read from the enum, so it never drifts from code.
        assert Tag.IMPORTANT.value in sysp
        assert EntityType.PRODUCT.value in sysp
        assert "500" in sysp  # RECOMMENDED_ANSWER_MAX

    def test_build_messages_structure(self):
        msgs = build_messages(SAMPLE_TEXT, max_blocks=3)
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert SAMPLE_TEXT in msgs[1]["content"]
        assert "AT MOST 3" in msgs[1]["content"]

    def test_generator_passes_messages_to_client(self):
        fake = FakeLLMClient(responses=[TWO_BLOCK_JSON])
        gen = IdeaBlockGenerator(fake)
        gen.generate(SAMPLE_TEXT)
        assert fake.last_messages is not None
        assert len(fake.last_messages) == 2
        assert fake.last_messages[0]["role"] == "system"
        assert SAMPLE_TEXT in fake.last_messages[1]["content"]


class TestRobustness:
    def test_strips_markdown_json_fence(self):
        gen = IdeaBlockGenerator(
            FakeLLMClient(responses=["```json\n" + TWO_BLOCK_JSON + "\n```"])
        )
        assert len(gen.generate(SAMPLE_TEXT)) == 2

    def test_extracts_json_from_surrounding_prose(self):
        wrapped = "Sure! Here are the blocks:\n" + TWO_BLOCK_JSON + "\nHope this helps!"
        gen = IdeaBlockGenerator(FakeLLMClient(responses=[wrapped]))
        assert len(gen.generate(SAMPLE_TEXT)) == 2

    def test_accepts_bare_top_level_list(self):
        bare = json.dumps(json.loads(TWO_BLOCK_JSON)["blocks"])
        gen = IdeaBlockGenerator(FakeLLMClient(responses=[bare]))
        assert len(gen.generate(SAMPLE_TEXT)) == 2

    def test_missing_question_mark_is_repaired_in_non_strict(self):
        payload = json.dumps(
            {
                "blocks": [
                    {
                        "name": "x",
                        "critical_question": "what is this",  # no '?'
                        "trusted_answer": "an answer.",
                    }
                ]
            }
        )
        gen = IdeaBlockGenerator(FakeLLMClient(responses=[payload]))
        blocks = gen.generate(SAMPLE_TEXT)
        assert len(blocks) == 1
        assert blocks[0].critical_question.endswith("?")

    def test_empty_response_raises(self):
        gen = IdeaBlockGenerator(FakeLLMClient(responses=["   "]))
        with pytest.raises(EmptyResponseError):
            gen.generate(SAMPLE_TEXT)

    def test_invalid_json_raises_parse_error(self):
        gen = IdeaBlockGenerator(FakeLLMClient(responses=["not json at all {{{" ]))
        with pytest.raises(ResponseParseError):
            gen.generate(SAMPLE_TEXT)

    def test_oversized_answer_block_skipped_in_non_strict(self):
        payload = json.dumps(
            {
                "blocks": [
                    {
                        "name": "too long",
                        "critical_question": "q?",
                        "trusted_answer": "x" * 600,  # > 500
                    },
                    {
                        "name": "ok",
                        "critical_question": "q?",
                        "trusted_answer": "fine answer.",
                    },
                ]
            }
        )
        gen = IdeaBlockGenerator(FakeLLMClient(responses=[payload]))
        blocks, stats = gen.generate_with_stats(SAMPLE_TEXT)
        assert len(blocks) == 1
        assert blocks[0].name == "ok"
        assert stats.skipped == 1
        assert stats.emitted == 1
        assert len(stats.errors) == 1

    def test_empty_blocks_list_returns_empty(self):
        gen = IdeaBlockGenerator(FakeLLMClient(responses=['{"blocks": []}']))
        assert gen.generate(SAMPLE_TEXT) == []


class TestEnumMapping:
    def test_unknown_tag_dropped_in_non_strict(self):
        raw = RawIdeaBlock(
            name="x",
            critical_question="q?",
            trusted_answer="a.",
            tags=["IMPORTANT", "BOGUS_TAG", "warning"],  # warning maps case-insensitively
        )
        block = coerce_block(raw, strict=False)
        assert block.tags == [Tag.IMPORTANT, Tag.WARNING]

    def test_unknown_entity_type_falls_back_to_concept(self):
        raw = RawIdeaBlock(
            name="x",
            critical_question="q?",
            trusted_answer="a.",
            entities=[
                RawEntity(entity_name="Foo", entity_type="NONSENSE"),
                RawEntity(entity_name="Bar", entity_type="person"),
            ],
        )
        block = coerce_block(raw, strict=False)
        assert block.entities[0].entity_type == EntityType.CONCEPT
        assert block.entities[1].entity_type == EntityType.PERSON

    def test_missing_required_field_raises_coercion_error(self):
        raw = RawIdeaBlock(
            name="x", critical_question="q?", trusted_answer=""
        )
        with pytest.raises(CoercionError):
            coerce_block(raw, strict=False)


class TestStrictMode:
    def test_strict_unknown_tag_raises_generation_error(self):
        payload = json.dumps(
            {
                "blocks": [
                    {
                        "name": "x",
                        "critical_question": "q?",
                        "trusted_answer": "a.",
                        "tags": ["BOGUS"],
                    }
                ]
            }
        )
        gen = IdeaBlockGenerator(FakeLLMClient(responses=[payload]), strict=True)
        with pytest.raises(GenerationError):
            gen.generate(SAMPLE_TEXT)

    def test_strict_oversized_answer_raises(self):
        payload = json.dumps(
            {
                "blocks": [
                    {
                        "name": "x",
                        "critical_question": "q?",
                        "trusted_answer": "x" * 600,
                    }
                ]
            }
        )
        gen = IdeaBlockGenerator(FakeLLMClient(responses=[payload]), strict=True)
        with pytest.raises(GenerationError):
            gen.generate(SAMPLE_TEXT)


class TestExtractJsonHelper:
    def test_plain_json(self):
        assert json.loads(_extract_json('{"a": 1}')) == {"a": 1}

    def test_fenced_json(self):
        text = "```json\n{\"a\": 1}\n```"
        assert json.loads(_extract_json(text)) == {"a": 1}

    def test_embedded_json(self):
        text = "prefix {\"a\": 1} suffix"
        assert json.loads(_extract_json(text)) == {"a": 1}

    def test_empty_raises(self):
        with pytest.raises(ResponseParseError):
            _extract_json("")


class TestInputValidation:
    def test_empty_text_rejected(self):
        gen = IdeaBlockGenerator(FakeLLMClient(responses=[TWO_BLOCK_JSON]))
        with pytest.raises(ValueError):
            gen.generate("   ")

    def test_replays_last_response_for_repeated_calls(self):
        fake = FakeLLMClient(responses=[TWO_BLOCK_JSON])
        gen = IdeaBlockGenerator(fake)
        first = gen.generate(SAMPLE_TEXT)
        second = gen.generate(SAMPLE_TEXT)
        assert len(first) == len(second) == 2
