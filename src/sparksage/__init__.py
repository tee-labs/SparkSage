"""SparkSage: structured, question-aligned knowledge chunks for RAG."""

from sparksage.clean import (
    DEFAULT_RULES,
    CallableRule,
    CleaningRegistry,
    CleaningResult,
    CleaningRule,
    RegexReplaceRule,
    TextCleaner,
)
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
    "DEFAULT_RULES",
    "BlockStatus",
    "CallableRule",
    "CleaningRegistry",
    "CleaningResult",
    "CleaningRule",
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
    "RegexReplaceRule",
    "SentenceRole",
    "Tag",
    "TechnicalBlock",
    "TextCleaner",
]

__version__ = "0.1.0"
