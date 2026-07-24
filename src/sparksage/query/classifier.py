"""Intent classification: what kind of question is this?

Classification runs *before* retrieval so the pipeline can intercept
out-of-domain or low-confidence queries early (saving a rewrite + retrieval
LLM call). The core depends only on the :class:`IntentClassifier` protocol, so
it is fully unit-testable with a deterministic fake -- the same
:class:`~sparksage.generator.LLMClient` protocol the generator already uses.

Two implementations ship out of the box:

- :class:`LLMIntentClassifier`: the default, chain-of-thought + JSON + the live
  :class:`QueryIntent` vocabulary (see :mod:`sparksage.query.prompts`).
- :class:`RuleIntentClassifier`: keyword/regex rules with no LLM call, for the
  cost-control "high-frequency patterns hit a rule first" pattern.

Chain them (rules first to short-circuit, LLM as fallback) via
:class:`~sparksage.query.processor.QueryProcessor` or by composing classifiers
yourself.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sparksage.generator.client import JSON_RESPONSE_FORMAT, LLMClient
from sparksage.query.context import ConversationContext
from sparksage.query.prompts import intent_messages
from sparksage.query.schema import (
    CoercionError,
    IntentResult,
    coerce_intent,
    parse_intent_response,
)
from sparksage.schema.enums import QueryIntent


class IntentError(RuntimeError):
    """Base error for intent classification."""


class IntentEmptyResponseError(IntentError):
    """The LLM returned no content."""


class IntentResponseParseError(IntentError):
    """The model response could not be parsed as the expected JSON."""


@runtime_checkable
class IntentClassifier(Protocol):
    """Classify a query into an intent + confidence.

    Implementations should be deterministic for a given input (LLM clients are
    "deterministic enough" with a low temperature). ``context`` lets contextual
    classifiers reason about multi-turn dialogue, but a classifier may ignore it.
    """

    def classify(
        self,
        query: str,
        context: ConversationContext | None = None,
    ) -> IntentResult:
        """Return the :class:`IntentResult` for ``query``."""
        ...


# --------------------------------------------------------------------------- #
# Rule-based classification (no LLM call)
# --------------------------------------------------------------------------- #
@runtime_checkable
class IntentRule(Protocol):
    """A single, side-effect-free intent rule.

    Returns an :class:`IntentResult` when the rule matches, or ``None`` to let
    the next rule try. Rules are tried in registration order; first match wins.
    """

    def match(
        self,
        query: str,
        context: ConversationContext | None = None,
    ) -> IntentResult | None:
        ...


@dataclass
class KeywordIntentRule:
    """Match when the query contains any of ``keywords`` (case-insensitive).

    The cheapest cost-control tool: pin high-frequency, unambiguous patterns
    ("营收"/"revenue" -> ``FINANCIAL_DATA``, greetings/chitchat ->
    ``OUT_OF_DOMAIN``) to an intent without spending an LLM call.
    """

    keywords: tuple[str, ...]
    intent: QueryIntent
    confidence: float = 1.0
    reasoning: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.keywords, str):
            self.keywords = (self.keywords,)
        self.keywords = tuple(self.keywords)

    def match(
        self,
        query: str,
        context: ConversationContext | None = None,
    ) -> IntentResult | None:
        lowered = query.lower()
        for kw in self.keywords:
            if kw and kw.lower() in lowered:
                reason = self.reasoning or f"keyword match: {kw!r}"
                return IntentResult(
                    intent=self.intent,
                    confidence=self.confidence,
                    reasoning=reason,
                )
        return None


@dataclass
class RuleIntentClassifier:
    """Intent classifier composed of :class:`IntentRule`s; no LLM call.

    Rules are tried in registration order; the first match wins. If nothing
    matches, ``default`` is returned with ``default_confidence`` -- this lets a
    rule set be "optimistic" (assume in-domain, let the LLM decide later) or
    "pessimistic" (assume out-of-domain unless a rule fires).

    The classifier never touches the network, which makes it ideal as a cheap
    pre-filter in front of :class:`LLMIntentClassifier`.
    """

    rules: list[IntentRule] = field(default_factory=list)
    default: QueryIntent = QueryIntent.BUSINESS_ANALYSIS
    default_confidence: float = 0.5

    def add(self, rule: IntentRule) -> RuleIntentClassifier:
        """Append a rule. Returns ``self`` for fluent chaining."""
        self.rules.append(rule)
        return self

    def add_keyword(
        self,
        keywords: Iterable[str],
        intent: QueryIntent,
        *,
        confidence: float = 1.0,
        reasoning: str = "",
    ) -> RuleIntentClassifier:
        """Convenience wrapper to append a :class:`KeywordIntentRule`."""
        return self.add(
            KeywordIntentRule(
                keywords, intent, confidence=confidence, reasoning=reasoning
            )
        )

    def classify(
        self,
        query: str,
        context: ConversationContext | None = None,
    ) -> IntentResult:
        for rule in self.rules:
            result = rule.match(query, context)
            if result is not None:
                return result
        return IntentResult(
            intent=self.default,
            confidence=self.default_confidence,
            reasoning="no rule matched; using default intent",
        )


# --------------------------------------------------------------------------- #
# LLM-based classification (the default)
# --------------------------------------------------------------------------- #
class LLMIntentClassifier:
    """Intent classifier backed by an :class:`LLMClient`.

    Parameters
    ----------
    client:
        Any :class:`LLMClient` (e.g. :class:`OpenAICompatibleClient`,
        :class:`FakeLLMClient`). Decouples classification from any SDK.
    model:
        Model name forwarded to the client (ignored by fakes). Rewriting needs
        only a lightweight model, so this is where cost is controlled.
    temperature:
        Sampling temperature. Low (default ``0.0``) for stable labels.
    use_json_mode:
        Request JSON-mode structured output from the provider when supported.
    strict:
        If ``False`` (default), an unrecognised intent string falls back to
        :data:`~sparksage.query.schema.DEFAULT_INTENT` rather than raising --
        query-time should be forgiving. If ``True``, raise on bad labels.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        use_json_mode: bool = True,
        strict: bool = False,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._use_json_mode = use_json_mode
        self._strict = strict

    def classify(
        self,
        query: str,
        context: ConversationContext | None = None,
    ) -> IntentResult:
        if not str(query).strip():
            return IntentResult(
                intent=QueryIntent.OUT_OF_DOMAIN,
                confidence=1.0,
                reasoning="empty query",
            )

        messages = intent_messages(query)
        response_text = self._client.complete(
            messages,
            model=self._model,
            temperature=self._temperature,
            response_format=JSON_RESPONSE_FORMAT if self._use_json_mode else None,
        )
        if not response_text or not response_text.strip():
            raise IntentEmptyResponseError("the LLM returned an empty response")

        try:
            raw = parse_intent_response(response_text)
            return coerce_intent(raw, strict=self._strict)
        except CoercionError as exc:
            raise IntentResponseParseError(str(exc)) from exc

    @property
    def strict(self) -> bool:
        return self._strict
