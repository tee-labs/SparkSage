"""LLM-driven generation of IdeaBlocks from free text.

Inject any :class:`LLMClient` (a real :class:`OpenAICompatibleClient` in
production, or :class:`FakeLLMClient` in tests) into an
:class:`IdeaBlockGenerator` and call :meth:`~IdeaBlockGenerator.generate`.
"""

from sparksage.generator.client import (
    JSON_RESPONSE_FORMAT,
    FakeLLMClient,
    LLMClient,
    OpenAICompatibleClient,
)
from sparksage.generator.generator import (
    EmptyResponseError,
    GenerationError,
    GenerationStats,
    IdeaBlockGenerator,
    ResponseParseError,
)
from sparksage.generator.prompts import build_messages, system_prompt, user_prompt
from sparksage.generator.schema import (
    CoercionError,
    RawEntity,
    RawGenerationResult,
    RawIdeaBlock,
    coerce_block,
    parse_raw_result,
)

__all__ = [
    "CoercionError",
    "EmptyResponseError",
    "FakeLLMClient",
    "GenerationError",
    "GenerationStats",
    "IdeaBlockGenerator",
    "JSON_RESPONSE_FORMAT",
    "LLMClient",
    "OpenAICompatibleClient",
    "RawEntity",
    "RawGenerationResult",
    "RawIdeaBlock",
    "ResponseParseError",
    "build_messages",
    "coerce_block",
    "parse_raw_result",
    "system_prompt",
    "user_prompt",
]
