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
from sparksage.embed import (
    BlockEmbedder,
    EmbeddingClient,
    FakeEmbeddingClient,
    InMemoryVectorStore,
    OpenAIEmbeddingClient,
    SearchHit,
    VectorStore,
    load_store,
    save_store,
)
from sparksage.generator import (
    FakeLLMClient,
    IdeaBlockGenerator,
    LLMClient,
    OpenAICompatibleClient,
)
from sparksage.logging_config import (
    DEFAULT_LOG_LEVEL,
    ENV_LOG_LEVEL,
    LogLevelError,
    configure_logging,
    parse_level,
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
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_RULES",
    "BlockEmbedder",
    "BlockStatus",
    "CallableRule",
    "CleaningRegistry",
    "CleaningResult",
    "CleaningRule",
    "ConvertOutput",
    "ConvertResponse",
    "ConversionResult",
    "ConverterBackend",
    "EmbeddingClient",
    "ENV_LOG_LEVEL",
    "EntityRelation",
    "EntityType",
    "EnvParseError",
    "FakeConverterBackend",
    "FakeEmbeddingClient",
    "FakeLLMClient",
    "GenerateOutput",
    "GenerateResponse",
    "GenerationNotConfiguredError",
    "GenerationStatsOut",
    "HealthResponse",
    "IdeaBlock",
    "IdeaBlockGenerator",
    "InMemoryVectorStore",
    "LLMClient",
    "LogLevelError",
    "MarkdownConverter",
    "MarkItDownBackend",
    "OpenAICompatibleClient",
    "OpenAIEmbeddingClient",
    "RegexReplaceRule",
    "SearchHit",
    "SentenceRole",
    "ServiceError",
    "SourceInfo",
    "SparkSageService",
    "Tag",
    "TechnicalBlock",
    "TextCleaner",
    "VectorStore",
    "configure_logging",
    "load_dotenv",
    "load_store",
    "parse_env_file",
    "parse_level",
    "save_store",
]

__version__ = "0.1.0"
