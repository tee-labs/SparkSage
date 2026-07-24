"""Tests for the order-sensitive TechnicalBlock variant."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sparksage.schema import SentenceRole, TechnicalBlock
from sparksage.schema.technical import AnnotatedSentence


class TestTechnicalBlock:
    def test_inherits_ideablock_core(self, sample_technical_block: TechnicalBlock):
        assert sample_technical_block.trusted_answer
        assert sample_technical_block.critical_question.endswith("?")
        assert len(sample_technical_block.steps) == 4

    def test_requires_at_least_one_step(self):
        with pytest.raises(ValidationError):
            TechnicalBlock(
                name="t",
                critical_question="q?",
                trusted_answer="a",
                steps=[],
            )

    def test_commands_and_warnings_filters(self, sample_technical_block: TechnicalBlock):
        assert len(sample_technical_block.commands) == 1
        assert sample_technical_block.commands[0].role == SentenceRole.COMMAND
        assert len(sample_technical_block.warnings) == 1

    def test_consecutive_duplicate_steps_removed(self):
        block = TechnicalBlock(
            name="t",
            critical_question="q?",
            trusted_answer="a",
            steps=[
                AnnotatedSentence(text="run setup", role=SentenceRole.COMMAND),
                AnnotatedSentence(text="run setup", role=SentenceRole.COMMAND),
                AnnotatedSentence(text="verify", role=SentenceRole.RESULT),
            ],
        )
        assert [s.text for s in block.steps] == ["run setup", "verify"]

    def test_embedding_text_enumerates_roles(self, sample_technical_block: TechnicalBlock):
        et = sample_technical_block.embedding_text
        assert "[COMMAND]" in et
        assert "[WARNING]" in et
        assert "1. " in et and "4. " in et

    def test_context_window_stored(self, sample_technical_block: TechnicalBlock):
        assert sample_technical_block.context.primary == "Installation"
        assert sample_technical_block.context.following == "# Verification"

    def test_searchable_dict_includes_steps(self, sample_technical_block: TechnicalBlock):
        d = sample_technical_block.to_searchable_dict()
        assert isinstance(d["steps"], list)
        assert d["steps"][0]["role"] == "PREREQUISITE"
        assert d["context_primary"] == "Installation"
