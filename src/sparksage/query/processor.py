"""Query processing orchestration: intent recognition + interception + rewrite.

:class:`QueryProcessor` is the query-time counterpart of the ingest pipeline --
it sits in front of retrieval and wires together an
:class:`~sparksage.query.classifier.IntentClassifier` and a
:class:`~sparksage.query.rewriter.QueryRewriter`:

    user query
        -> IntentClassifier   (what kind of question is this?)
        -> intercept          (out-of-domain / low-confidence? stop here)
        -> QueryRewriter      (turn human words into search words)
        -> retrieval (downstream, not built yet)

The processor depends only on the two protocols, so it is fully unit-testable
with deterministic fakes. Interception policy (which intents to reject, the
confidence floor, the canned reply) is configuration, not behaviour hidden in a
client.

Design note: this is the framework-agnostic core. A future ``/api/v1/query``
route will be a thin FastAPI wrapper around it, exactly as
:class:`~sparksage.api.SparkSageService` wraps the ingest pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from sparksage.query.classifier import IntentClassifier, IntentResult
from sparksage.query.context import ConversationContext
from sparksage.query.rewriter import QueryRewriter, RewriteResult
from sparksage.schema.enums import QueryIntent

_logger = logging.getLogger(__name__)

#: Default intents that short-circuit the pipeline (skip the rewrite entirely).
DEFAULT_REJECTED_INTENTS: frozenset[QueryIntent] = frozenset(
    {QueryIntent.OUT_OF_DOMAIN}
)

#: Default floor below which a query is treated as unanswerable.
DEFAULT_MIN_CONFIDENCE: float = 0.4

#: Default canned reply when a query is intercepted.
DEFAULT_DEFAULT_REPLY: str = (
    "Sorry, I can only answer questions within the knowledge domain "
    "this system serves."
)


@dataclass
class QueryResult:
    """The full outcome of processing one query.

    Attributes
    ----------
    intent:
        The classified intent (always populated, even when rejected).
    rewrite:
        The rewrite result. Populated only when ``accepted``; for rejected
        queries it is an identity rewrite of the original query so callers
        always have a usable ``rewritten_query`` if they want one.
    accepted:
        ``True`` when the query passed interception and was rewritten (eligible
        for retrieval). ``False`` when it was rejected -- see ``default_reply``.
    default_reply:
        Canned reply to surface when ``accepted`` is ``False``; ``None`` when
        the query was accepted.
    original_query:
        The raw query as received, for provenance.
    """

    intent: IntentResult
    rewrite: RewriteResult
    accepted: bool
    default_reply: str | None = None
    original_query: str = ""


@dataclass
class QueryProcessor:
    """Two-stage query pipeline: classify -> intercept -> rewrite.

    Parameters
    ----------
    classifier:
        Any :class:`IntentClassifier` (LLM, rule-based, or a composition).
    rewriter:
        Any :class:`QueryRewriter` (LLM, rule-based, or a composition).
    min_confidence:
        Queries classified below this confidence are rejected (default ``0.4``).
    rejected_intents:
        Intents that short-circuit the pipeline regardless of confidence
        (default ``{OUT_OF_DOMAIN}``).
    default_reply:
        Canned reply surfaced on :class:`QueryResult` when a query is rejected.
    rewrite_on_low_confidence:
        When ``False`` (default) a low-confidence query is rejected just like an
        out-of-domain one. When ``True`` low-confidence is allowed through to
        rewriting (useful when the classifier is noisy but rewriting is cheap).

    Examples
    --------
    >>> from sparksage.query import (
    ...     QueryProcessor, LLMIntentClassifier, LLMQueryRewriter,
    ... )
    >>> proc = QueryProcessor(                       # doctest: +SKIP
    ...     classifier=LLMIntentClassifier(client),
    ...     rewriter=LLMQueryRewriter(client),
    ... )
    >>> result = proc.process("中国移动2024年净利润怎么样")  # doctest: +SKIP
    >>> result.accepted                                     # doctest: +SKIP
    True
    """

    classifier: IntentClassifier
    rewriter: QueryRewriter
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    rejected_intents: frozenset[QueryIntent] = field(
        default_factory=lambda: frozenset(DEFAULT_REJECTED_INTENTS)
    )
    default_reply: str = DEFAULT_DEFAULT_REPLY
    rewrite_on_low_confidence: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.classifier, IntentClassifier):
            raise TypeError("classifier must implement the IntentClassifier protocol")
        if not isinstance(self.rewriter, QueryRewriter):
            raise TypeError("rewriter must implement the QueryRewriter protocol")
        self.rejected_intents = frozenset(self.rejected_intents)
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0.0, 1.0]")

    def process(
        self,
        query: str,
        context: ConversationContext | None = None,
    ) -> QueryResult:
        """Classify, intercept, and (if accepted) rewrite ``query``."""
        original = str(query)
        _logger.debug(
            "query process start: query=%r has_context=%s",
            original[:80],
            context is not None,
        )
        intent = self.classifier.classify(query, context)
        _logger.debug(
            "query classified: intent=%s confidence=%.2f",
            intent.intent.value,
            intent.confidence,
        )

        if self._should_reject(intent):
            _logger.debug(
                "query rejected: intent=%s (in rejected=%s below_floor=%s)",
                intent.intent.value,
                intent.intent in self.rejected_intents,
                intent.confidence < self.min_confidence,
            )
            return QueryResult(
                intent=intent,
                rewrite=_identity_rewrite(original),
                accepted=False,
                default_reply=self.default_reply,
                original_query=original,
            )

        rewrite = self.rewriter.rewrite(query, context=context, intent=intent)
        _logger.debug(
            "query rewritten: %r -> %r sub_queries=%d",
            original[:60],
            rewrite.rewritten_query[:80],
            len(rewrite.sub_queries or []),
        )
        return QueryResult(
            intent=intent,
            rewrite=rewrite,
            accepted=True,
            default_reply=None,
            original_query=original,
        )

    def _should_reject(self, intent: IntentResult) -> bool:
        if intent.intent in self.rejected_intents:
            return True
        if intent.confidence < self.min_confidence:
            return not self.rewrite_on_low_confidence
        return False

    def add_rejected_intent(self, intent: QueryIntent | Iterable[QueryIntent]) -> QueryProcessor:
        """Add ``intent`` (or an iterable of intents) to the reject set.

        Returns ``self`` for fluent chaining.
        """
        if isinstance(intent, QueryIntent):
            intents = (intent,)
        else:
            intents = tuple(intent)
        self.rejected_intents = self.rejected_intents | frozenset(intents)
        return self


def _identity_rewrite(query: str) -> RewriteResult:
    """A rewrite that just echoes the query (used for rejected queries)."""
    return RewriteResult(
        rewritten_query=query, reasoning="rejected; identity rewrite"
    )
