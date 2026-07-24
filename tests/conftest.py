"""Shared fixtures for schema tests."""

from __future__ import annotations

import pytest

from sparksage.schema import Entity, IdeaBlock, Tag, TechnicalBlock
from sparksage.schema.enums import EntityType, SentenceRole
from sparksage.schema.source import SourceRef


@pytest.fixture
def sample_entity() -> Entity:
    return Entity(
        entity_name="SparkSage",
        entity_type=EntityType.PRODUCT,
        aliases=["sparksage"],
    )


@pytest.fixture
def sample_ideablock(sample_entity: Entity) -> IdeaBlock:
    return IdeaBlock(
        name="What SparkSage does",
        critical_question="What problem does SparkSage solve?",
        trusted_answer=(
            "SparkSage turns documents into question-aligned knowledge units so "
            "retrieval hits whole, self-contained answers instead of text shards."
        ),
        tags=[Tag.IMPORTANT, Tag.TECHNOLOGY],
        entities=[sample_entity],
        keywords=["rag", "chunking", "retrieval"],
        source=SourceRef(uri="file://docs/overview.md", title="Overview"),
    )


@pytest.fixture
def sample_technical_block(sample_entity: Entity) -> TechnicalBlock:
    from sparksage.schema.technical import AnnotatedSentence, _ContextWindow

    return TechnicalBlock(
        name="Deploying SparkSage",
        critical_question="How do I deploy SparkSage locally?",
        trusted_answer="Run the install command after confirming Python 3.10+.",
        tags=[Tag.PROCESS, Tag.IMPORTANT],
        entities=[sample_entity],
        keywords=["deploy", "install"],
        context=_ContextWindow(
            primary="Installation",
            proceeding="# Setup",
            following="# Verification",
        ),
        steps=[
            AnnotatedSentence(
                text="Python 3.10 or newer must be installed.",
                role=SentenceRole.PREREQUISITE,
            ),
            AnnotatedSentence(text="pip install sparksage", role=SentenceRole.COMMAND),
            AnnotatedSentence(text="Do not run as root.", role=SentenceRole.WARNING),
            AnnotatedSentence(text="The CLI is now on PATH.", role=SentenceRole.RESULT),
        ],
    )
