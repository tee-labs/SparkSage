"""SparkSage: structured, question-aligned knowledge chunks for RAG."""

from sparksage.api import (
    ConvertOutput,
    ConvertResponse,
    GenerateOutput,
    GenerateResponse,
    GenerationNotConfiguredError,
    GenerationStatsOut,
    HealthResponse,
    ServiceError,
    SourceInfo,
    SparkSageService,
)
from sparksage.clean import (
    DEFAULT_RULES,
    CallableRule,
    CleaningRegistry,
    CleaningResult,
    CleaningRule,
    RegexReplaceRule,
    TextCleaner,
)
from sparksage.config import EnvParseError, load_dotenv, parse_env_file
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
    "ConvertOutput",
    "ConvertResponse",
    "ConversionResult",
    "ConverterBackend",
    "EntityRelation",
    "EntityType",
    "EnvParseError",
    "FakeConverterBackend",
    "FakeLLMClient",
    "GenerateOutput",
    "GenerateResponse",
    "GenerationNotConfiguredError",
    "GenerationStatsOut",
    "HealthResponse",
    "IdeaBlock",
    "IdeaBlockGenerator",
    "LLMClient",
    "MarkdownConverter",
    "MarkItDownBackend",
    "OpenAICompatibleClient",
    "RegexReplaceRule",
    "SentenceRole",
    "ServiceError",
    "SourceInfo",
    "SparkSageService",
    "Tag",
    "TechnicalBlock",
    "TextCleaner",
    "load_dotenv",
    "parse_env_file",
]

__version__ = "0.1.0"
