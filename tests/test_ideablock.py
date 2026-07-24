"""Tests for the core IdeaBlock chunk schema."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from sparksage.schema import BlockStatus, Entity, EntityType, IdeaBlock, Tag


class TestIdeaBlockConstruction:
    def test_minimal_block_defaults(self):
        block = IdeaBlock(
            name="Title",
            critical_question="What is this?",
            trusted_answer="A short, verified answer.",
        )
        assert isinstance(block.id, uuid.UUID)
        assert block.status == BlockStatus.DRAFT
        assert block.tags == []
        assert block.entities == []
        assert block.keywords == []
        assert block.version == 1
        assert block.is_live is False  # DRAFT is not live

    def test_full_block_round_trips_to_dict(self, sample_ideablock: IdeaBlock):
        d = sample_ideablock.model_dump()
        assert d["name"] == "What SparkSage does"
        assert d["critical_question"].endswith("?")
        assert d["tags"][0] == Tag.IMPORTANT
        restored = IdeaBlock.model_validate(d)
        assert restored == sample_ideablock


class TestIdeaBlockValidation:
    def test_empty_required_text_rejected(self):
        with pytest.raises(ValidationError):
            IdeaBlock(name="   ", critical_question="q?", trusted_answer="a")

    def test_non_question_rejected(self):
        with pytest.raises(ValidationError):
            IdeaBlock(
                name="t", critical_question="this is a statement", trusted_answer="a"
            )

    def test_chinese_question_mark_accepted(self):
        block = IdeaBlock(
            name="标题",
            critical_question="这是什么？",
            trusted_answer="这是一个测试。",
        )
        assert block.critical_question.endswith("？")

    def test_oversized_answer_rejected(self):
        with pytest.raises(ValidationError, match="concise"):
            IdeaBlock(
                name="t",
                critical_question="q?",
                trusted_answer="x" * 501,
            )

    def test_keywords_dedup_and_normalize_case_insensitive(self):
        block = IdeaBlock(
            name="t",
            critical_question="q?",
            trusted_answer="a",
            keywords=["RAG", "rag", " Chunking ", ""],
        )
        assert block.keywords == ["RAG", "Chunking"]

    def test_tags_dedup_preserves_order(self):
        block = IdeaBlock(
            name="t",
            critical_question="q?",
            trusted_answer="a",
            tags=[Tag.IMPORTANT, Tag.TECHNOLOGY, Tag.IMPORTANT],
        )
        assert block.tags == [Tag.IMPORTANT, Tag.TECHNOLOGY]

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            IdeaBlock(
                name="t",
                critical_question="q?",
                trusted_answer="a",
                bogus=42,  # type: ignore[arg-type]
            )


class TestEmbeddingText:
    def test_embedding_text_concatenates_three_fields(self, sample_ideablock: IdeaBlock):
        et = sample_ideablock.embedding_text
        assert sample_ideablock.name in et
        assert sample_ideablock.critical_question in et
        assert sample_ideablock.trusted_answer in et
        assert et.count("\n") == 2


class TestSerialization:
    def test_to_xml_contains_canonical_tags(self, sample_ideablock: IdeaBlock):
        xml = sample_ideablock.to_xml()
        assert "<ideablock>" in xml and "</ideablock>" in xml
        assert "<name>" in xml
        assert "<critical_question>" in xml
        assert "<trusted_answer>" in xml
        assert "<tags>" in xml
        assert "<keywords>" in xml
        assert "<entity>" in xml
        assert "<entity_type>PRODUCT</entity_type>" in xml

    def test_xml_escapes_special_characters(self):
        block = IdeaBlock(
            name="a < b & c",
            critical_question="is a < b?",
            trusted_answer="yes & no <tag>",
        )
        xml = block.to_xml()
        assert "&lt;" in xml
        assert "&amp;" in xml
        assert "<tag>" not in xml.split("<trusted_answer>")[1]

    def test_searchable_dict_is_flat(self, sample_ideablock: IdeaBlock):
        d = sample_ideablock.to_searchable_dict()
        assert isinstance(d["id"], str)
        assert isinstance(d["tags"], list) and d["tags"][0] == "IMPORTANT"
        assert d["entity_types"] == ["PRODUCT"]
        assert "embedding" not in d
        assert d["status"] == "DRAFT"


class TestLifecycle:
    def test_touch_bumps_version_and_time(self, sample_ideablock: IdeaBlock):
        before = sample_ideablock.updated_at
        sample_ideablock.touch()
        assert sample_ideablock.version == 2
        assert sample_ideablock.updated_at >= before

    def test_active_is_live(self):
        block = IdeaBlock(
            name="t",
            critical_question="q?",
            trusted_answer="a",
            status=BlockStatus.ACTIVE,
        )
        assert block.is_live is True


class TestEntity:
    def test_entity_strips_and_dedupes_aliases(self):
        e = Entity(
            entity_name=" SparkSage ",
            entity_type=EntityType.PRODUCT,
            aliases=["SparkSage", " spark ", "", "spark"],
        )
        assert e.entity_name == "SparkSage"
        assert e.aliases == ["SparkSage", "spark"]

    def test_entity_requires_name(self):
        with pytest.raises(ValidationError):
            Entity(entity_name="   ", entity_type=EntityType.PRODUCT)
