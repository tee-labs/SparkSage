"""End-to-end demo of the SparkSage chunk schema.

Run with:  PYTHONPATH=src python3 examples/build_chunks.py
"""

from __future__ import annotations

import json

from sparksage.schema import (
    BlockStatus,
    Entity,
    EntityType,
    IdeaBlock,
    SentenceRole,
    Tag,
    TechnicalBlock,
)
from sparksage.schema.source import SourceRef
from sparksage.schema.technical import AnnotatedSentence, _ContextWindow


def build_ideablock() -> IdeaBlock:
    return IdeaBlock(
        name="Embedding strategy",
        critical_question="Why does SparkSage embed only the trusted_answer?",
        tags=[Tag.IMPORTANT, Tag.ARCHITECTURE],
        entities=[
            Entity(
                entity_name="IdeaBlock",
                entity_type=EntityType.CONCEPT,
                aliases=["idea block"],
            )
        ],
        keywords=["embedding", "single-field", "chunking"],
        trusted_answer=(
            "Embedding a single self-contained field avoids the 'sentence cut in "
            "half' problem of naive chunking, so dense vectors cluster around whole "
            "answers instead of arbitrary text boundaries."
        ),
        source=SourceRef(uri="file://docs/architecture.md", locator="§2.1"),
        status=BlockStatus.ACTIVE,
    )


def build_technical_block() -> TechnicalBlock:
    return TechnicalBlock(
        name="Rebuild the index",
        critical_question="How do I rebuild the SparkSage index?",
        tags=[Tag.PROCESS, Tag.IMPORTANT],
        entities=[Entity(entity_name="SparkSage", entity_type=EntityType.PRODUCT)],
        keywords=["index", "rebuild"],
        trusted_answer="Stop the service, run the rebuild command, then verify.",
        context=_ContextWindow(
            primary="Maintenance",
            proceeding="## Pre-flight",
            following="## Health check",
        ),
        steps=[
            AnnotatedSentence(text="Service must be stopped first.", role=SentenceRole.PREREQUISITE),
            AnnotatedSentence(text="sparksage index rebuild", role=SentenceRole.COMMAND),
            AnnotatedSentence(text="Do not interrupt the rebuild.", role=SentenceRole.WARNING),
            AnnotatedSentence(text="Index now contains the latest blocks.", role=SentenceRole.RESULT),
        ],
        status=BlockStatus.ACTIVE,
    )


def main() -> None:
    ib = build_ideablock()
    tb = build_technical_block()

    print("=== IdeaBlock XML ===")
    print(ib.to_xml())
    print("\n=== IdeaBlock embedding text ===")
    print(ib.embedding_text)
    print("\n=== TechnicalBlock embedding text (role-tagged) ===")
    print(tb.embedding_text)
    print("\n=== TechnicalBlock searchable record ===")
    print(json.dumps(tb.to_searchable_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
