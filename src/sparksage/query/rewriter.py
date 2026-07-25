"""Query rewriting: turn users' natural language into search-engine language.

Users ask with omissions, anaphora, and ambiguity ("那联通呢" / "the same metric
for last year"); vector retrieval needs precise, complete, specific queries.
A :class:`QueryRewriter` closes that gap. The core depends only on the
:class:`QueryRewriter` protocol, reusing the existing
:class:`~sparksage.generator.LLMClient` -- no new LLM abstraction.

Two implementations ship out of the box:

- :class:`LLMQueryRewriter`: the default -- chain-of-thought + JSON + a hard
  "do not over-expand" constraint, with conversation context baked in for
  multi-turn anaphora resolution (see :mod:`sparksage.query.prompts`).
- :class:`RuleQueryRewriter`: composable regex/callable rules for the
  cost-control "high-frequency patterns hit a rule first" pattern.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sparksage.generator.client import JSON_RESPONSE_FORMAT, LLMClient
from sparksage.query.classifier import IntentResult
from sparksage.query.context import ConversationContext
from sparksage.query.prompts import rewrite_messages
from sparksage.query.schema import (
    CoercionError,
    RewriteResult,
    coerce_rewrite,
    parse_rewrite_response,
)


class RewriteError(RuntimeError):
    """Base error for query rewriting."""


class RewriteEmptyResponseError(RewriteError):
    """The LLM returned no content."""


class RewriteResponseParseError(RewriteError):
    """The model response could not be parsed as the expected JSON."""


@runtime_checkable
class QueryRewriter(Protocol):
    """Rewrite one query into a search-ready :class:`RewriteResult`.

    ``context`` carries conversation history for anaphora resolution and
    ``intent`` lets an intent-aware rewriter branch (e.g. split comparisons).
    Either may be ``None`` / ignored.
    """

    def rewrite(
        self,
        query: str,
        context: ConversationContext | None = None,
        intent: IntentResult | None = None,
    ) -> RewriteResult:
        """Return the :class:`RewriteResult` for ``query``."""
        ...


# --------------------------------------------------------------------------- #
# Rule-based rewriting (no LLM call)
# --------------------------------------------------------------------------- #
@runtime_checkable
class RewriteRule(Protocol):
    """A single, side-effect-free rewrite rule.

    Returns a :class:`RewriteResult` when the rule applies, or ``None`` to let
    the next rule try. Rules are tried in registration order; first hit wins.
    """

    def rewrite(
        self,
        query: str,
        context: ConversationContext | None = None,
        intent: IntentResult | None = None,
    ) -> RewriteResult | None:
        ...


@dataclass
class RegexRewriteRule:
    """Replace every match of ``pattern`` with ``replacement``.

    A lightweight, dependency-free escape hatch: normalise terminology, fix
    common typos, expand a known short-form, etc. When the substitution leaves
    the query unchanged the rule reports "no match" (``None``).
    """

    pattern: re.Pattern[str]
    replacement: str

    def __init__(
        self,
        pattern: str | re.Pattern[str],
        replacement: str = "",
        *,
        flags: int = 0,
    ) -> None:
        self.pattern = (
            re.compile(pattern, flags) if isinstance(pattern, str) else pattern
        )
        self.replacement = replacement

    def rewrite(
        self,
        query: str,
        context: ConversationContext | None = None,
        intent: IntentResult | None = None,
    ) -> RewriteResult | None:
        new = self.pattern.sub(self.replacement, query)
        if new == query:
            return None
        return RewriteResult(
            rewritten_query=new, reasoning="regex rewrite"
        )


class CallableRewriteRule:
    """Wrap a plain function as a rewrite rule.

    The fastest way to add one-off business logic (e.g. a template that fills a
    company name from context) without subclassing::

        rewriter.add(CallableRewriteRule(lambda q, c, i: my_template(q, c)))

    The wrapped callable may accept ``(query)``, ``(query, context)`` or
    ``(query, context, intent)``. Returning ``None`` means "not applicable";
    returning a :class:`RewriteResult` (or a plain ``str``) applies the rule.
    """

    def __init__(self, fn: Callable[..., object]) -> None:
        self._fn = fn
        try:
            nparams = len(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            nparams = 1
        self._nparams = nparams

    def rewrite(
        self,
        query: str,
        context: ConversationContext | None = None,
        intent: IntentResult | None = None,
    ) -> RewriteResult | None:
        if self._nparams >= 3:
            result = self._fn(query, context, intent)
        elif self._nparams == 2:
            result = self._fn(query, context)
        else:
            result = self._fn(query)
        if result is None:
            return None
        if isinstance(result, RewriteResult):
            return result
        return RewriteResult(rewritten_query=str(result), reasoning="callable rewrite")


@dataclass
class RuleQueryRewriter:
    """Query rewriter composed of :class:`RewriteRule`s; no LLM call.

    Rules are tried in registration order; the first one that returns a result
    wins. If nothing applies, the query is returned unchanged (identity), so the
    pipeline always yields something searchable.
    """

    rules: list[RewriteRule] = field(default_factory=list)

    def add(self, rule: RewriteRule) -> RuleQueryRewriter:
        """Append a rule. Returns ``self`` for fluent chaining."""
        self.rules.append(rule)
        return self

    def add_regex(
        self,
        pattern: str | re.Pattern[str],
        replacement: str = "",
        *,
        flags: int = 0,
    ) -> RuleQueryRewriter:
        """Convenience wrapper to append a :class:`RegexRewriteRule`."""
        return self.add(RegexRewriteRule(pattern, replacement, flags=flags))

    def rewrite(
        self,
        query: str,
        context: ConversationContext | None = None,
        intent: IntentResult | None = None,
    ) -> RewriteResult:
        for rule in self.rules:
            result = rule.rewrite(query, context, intent)
            if result is not None:
                return result
        return RewriteResult(
            rewritten_query=query, reasoning="no rule applied; identity rewrite"
        )


# --------------------------------------------------------------------------- #
# LLM-based rewriting (the default)
# --------------------------------------------------------------------------- #
class LLMQueryRewriter:
    """Query rewriter backed by an :class:`LLMClient`.

    Parameters
    ----------
    client:
        Any :class:`LLMClient` (e.g. :class:`OpenAICompatibleClient`,
        :class:`FakeLLMClient`). Decouples rewriting from any SDK.
    model:
        Model name forwarded to the client (ignored by fakes). Rewriting needs
        only a lightweight model -- set this to control cost.
    temperature:
        Sampling temperature. Low (default ``0.2``) for faithful rewrites.
    use_json_mode:
        Request JSON-mode structured output from the provider when supported.
    strict:
        If ``False`` (default), an empty ``rewritten_query`` falls back to the
        original query. If ``True``, raise on an empty rewrite.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        use_json_mode: bool = True,
        strict: bool = False,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._use_json_mode = use_json_mode
        self._strict = strict

    def rewrite(
        self,
        query: str,
        context: ConversationContext | None = None,
        intent: IntentResult | None = None,
    ) -> RewriteResult:
        original = str(query).strip()
        if not original:
            return RewriteResult(
                rewritten_query="", reasoning="empty query; nothing to rewrite"
            )

        messages = rewrite_messages(query, context=context)
        response_text = self._client.complete(
            messages,
            model=self._model,
            temperature=self._temperature,
            response_format=JSON_RESPONSE_FORMAT if self._use_json_mode else None,
        )
        if not response_text or not response_text.strip():
            raise RewriteEmptyResponseError("the LLM returned an empty response")

        try:
            raw = parse_rewrite_response(response_text)
            return coerce_rewrite(raw, original, strict=self._strict)
        except CoercionError as exc:
            raise RewriteResponseParseError(str(exc)) from exc

    @property
    def strict(self) -> bool:
        return self._strict
