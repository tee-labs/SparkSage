"""Query-time intent recognition and rewriting for RAG.

This is the query-side counterpart to the ingest pipeline. It turns a user's
natural-language question into (a) an :class:`QueryIntent` label, used to
intercept out-of-domain / low-confidence queries, and (b) a search-ready
:class:`RewriteResult` (resolved anaphora, completed context, optional
sub-queries). Everything depends only on the :class:`LLMClient` protocol
already used by the generator, so it runs fully offline under a
:class:`FakeLLMClient` in tests.

The two stages are independent, swappable protocols:

- :class:`IntentClassifier`  -- default :class:`LLMIntentClassifier`, or
  :class:`RuleIntentClassifier` for cost-free keyword routing.
- :class:`QueryRewriter`     -- default :class:`LLMQueryRewriter`, or
  :class:`RuleQueryRewriter` for template/regex rules.

:class:`QueryProcessor` wires them together with an interception policy.

Example
-------
::

    from sparksage import OpenAICompatibleClient
    from sparksage.query import (
        QueryProcessor, LLMIntentClassifier, LLMQueryRewriter,
    )

    client = OpenAICompatibleClient(api_key=...)
    proc = QueryProcessor(
        classifier=LLMIntentClassifier(client),
        rewriter=LLMQueryRewriter(client),
    )
    result = proc.process("那联通呢")
    if result.accepted:
        search_with(result.rewrite.rewritten_query)
    else:
        show(result.default_reply)
"""

from sparksage.query.cache import (
    DEFAULT_CACHE_THRESHOLD,
    DEFAULT_MAX_ENTRIES,
    CacheStats,
    InMemorySemanticCache,
    semantic_cache_stats,
)
from sparksage.query.classifier import (
    IntentClassifier,
    IntentEmptyResponseError,
    IntentError,
    IntentResponseParseError,
    IntentResult,
    IntentRule,
    KeywordIntentRule,
    LLMIntentClassifier,
    RuleIntentClassifier,
)
from sparksage.query.context import (
    ASSISTANT,
    USER,
    ConversationContext,
    ConversationTurn,
)
from sparksage.query.expander import (
    DEFAULT_HYDE_MAX_WORDS,
    DEFAULT_N_VARIANTS,
    HyDEExpander,
    IdentityExpander,
    LLMQueryExpander,
    QueryExpander,
)
from sparksage.query.processor import (
    DEFAULT_DEFAULT_REPLY,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_REJECTED_INTENTS,
    QueryProcessor,
    QueryResult,
)
from sparksage.query.prompts import (
    intent_messages,
    intent_system_prompt,
    intent_user_prompt,
    rewrite_messages,
    rewrite_system_prompt,
    rewrite_user_prompt,
)
from sparksage.query.refiner import (
    DEFAULT_MAX_REFINED_CHARS,
    IdentityRefiner,
    LLMQueryRefiner,
    QueryRefiner,
)
from sparksage.query.rewriter import (
    CallableRewriteRule,
    LLMQueryRewriter,
    QueryRewriter,
    RegexRewriteRule,
    RewriteEmptyResponseError,
    RewriteError,
    RewriteResponseParseError,
    RewriteResult,
    RewriteRule,
    RuleQueryRewriter,
)
from sparksage.query.schema import (
    DEFAULT_CONFIDENCE,
    DEFAULT_INTENT,
    CoercionError,
    RawIntent,
    RawRewrite,
    coerce_intent,
    coerce_rewrite,
    extract_json,
    parse_intent_response,
    parse_raw_intent,
    parse_raw_rewrite,
    parse_rewrite_response,
)
from sparksage.query.self_query import (
    IdentitySelfQueryParser,
    LLMSelfQueryParser,
    RawSelfQuery,
    SelfQueryEmptyResponseError,
    SelfQueryError,
    SelfQueryParser,
    SelfQueryResponseParseError,
    SelfQueryResult,
    coerce_self_query,
    parse_raw_self_query,
    parse_self_query_response,
)

__all__ = [
    "ASSISTANT",
    "CacheStats",
    "DEFAULT_CACHE_THRESHOLD",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_DEFAULT_REPLY",
    "DEFAULT_HYDE_MAX_WORDS",
    "DEFAULT_INTENT",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MAX_REFINED_CHARS",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_N_VARIANTS",
    "DEFAULT_REJECTED_INTENTS",
    "CallableRewriteRule",
    "CoercionError",
    "ConversationContext",
    "ConversationTurn",
    "HyDEExpander",
    "IdentityExpander",
    "IdentityRefiner",
    "IdentitySelfQueryParser",
    "InMemorySemanticCache",
    "IntentClassifier",
    "IntentEmptyResponseError",
    "IntentError",
    "IntentResponseParseError",
    "IntentResult",
    "IntentRule",
    "KeywordIntentRule",
    "LLMIntentClassifier",
    "LLMQueryExpander",
    "LLMQueryRefiner",
    "LLMQueryRewriter",
    "LLMSelfQueryParser",
    "QueryExpander",
    "QueryProcessor",
    "QueryRefiner",
    "QueryResult",
    "QueryRewriter",
    "RawIntent",
    "RawRewrite",
    "RawSelfQuery",
    "RegexRewriteRule",
    "RewriteEmptyResponseError",
    "RewriteError",
    "RewriteResponseParseError",
    "RewriteResult",
    "RewriteRule",
    "RuleIntentClassifier",
    "RuleQueryRewriter",
    "SelfQueryEmptyResponseError",
    "SelfQueryError",
    "SelfQueryParser",
    "SelfQueryResponseParseError",
    "SelfQueryResult",
    "USER",
    "coerce_intent",
    "coerce_rewrite",
    "coerce_self_query",
    "extract_json",
    "intent_messages",
    "intent_system_prompt",
    "intent_user_prompt",
    "parse_intent_response",
    "parse_raw_intent",
    "parse_raw_rewrite",
    "parse_raw_self_query",
    "parse_rewrite_response",
    "parse_self_query_response",
    "rewrite_messages",
    "rewrite_system_prompt",
    "rewrite_user_prompt",
    "semantic_cache_stats",
]
