"""SparkSage: structured, question-aligned knowledge chunks for RAG."""

from sparksage.convert import (
    DEFAULT_EXTENSIONS,
    ConversionResult,
    ConverterBackend,
    FakeConverterBackend,
    MarkdownConverter,
    MarkItDownBackend,
)
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
    "DEFAULT_EXTENSIONS",
    "BlockStatus",
    "ConversionResult",
    "ConverterBackend",
    "EntityRelation",
    "EntityType",
    "FakeConverterBackend",
    "FakeLLMClient",
    "IdeaBlock",
    "IdeaBlockGenerator",
    "LLMClient",
    "MarkdownConverter",
    "MarkItDownBackend",
    "OpenAICompatibleClient",
    "SentenceRole",
    "Tag",
    "TechnicalBlock",
]

__version__ = "0.1.0"
