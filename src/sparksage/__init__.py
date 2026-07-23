"""SparkSage: structured, question-aligned knowledge chunks for RAG."""

from sparksage.generator import (
    FakeLLMClient,
    IdeaBlockGenerator,
    LLMClient,
    OpenAICompatibleClient,
)
from sparksage.schema.enums import (
    BlockStatus,
    EntityRelation,
    EntityType,
    SentenceRole,
    Tag,
)
from sparksage.schema.ideablock import IdeaBlock
from sparksage.schema.technical import TechnicalBlock

__all__ = [
    "BlockStatus",
    "EntityRelation",
    "EntityType",
    "IdeaBlock",
    "IdeaBlockGenerator",
    "FakeLLMClient",
    "LLMClient",
    "OpenAICompatibleClient",
    "SentenceRole",
    "Tag",
    "TechnicalBlock",
]

__version__ = "0.1.0"
