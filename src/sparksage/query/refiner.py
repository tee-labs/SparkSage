"""Query refinement: rewrite a query using retrieval-feedback signals.

This is the self-corrective-retrieval companion to
:class:`~sparksage.query.rewriter.QueryRewriter`. When the
:class:`~sparksage.retrieve.grader.RetrievalGrader` scores a retrieval as weakly
relevant, a :class:`QueryRefiner` rewrites the query to be more specific /
searchable and the :class:`~sparksage.qa.QAEngine` re-retrieves -- the
"relevance score -> query refinement -> re-retrieve" iteration of the
self-reflective retrieval loop.

The refiner depends only on the relevance *score* and *reasoning* (not on the
retrieved chunks themselves), so the :mod:`sparksage.query` package stays free of
any dependency on :mod:`sparksage.retrieve` -- the same clean layering the
rewriter already observes. It reuses the existing
:class:`~sparksage.generator.LLMClient` (no new abstraction), so it is fully
unit-testable with :class:`~sparksage.generator.FakeLLMClient`.

Two implementations ship:

- :class:`LLMQueryRefiner`: the default -- feeds the query and the low-relevance
  feedback to the model and asks for one more specific, search-ready query
  (lenient -> strict coercion, robust to messy JSON).
- :class:`IdentityRefiner`: returns the query unchanged, so refinement is always
  configurable as "off" without branching (the self-reflective loop then grades
  but never re-retrieves).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol, runtime_checkable

from sparksage.generator.client import JSON_RESPONSE_FORMAT, LLMClient

_logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$", re.MULTILINE)

#: Maximum length of a refined query (keeps retrieval prompts bounded).
DEFAULT_MAX_REFINED_CHARS = 400


@runtime_checkable
class QueryRefiner(Protocol):
    """Refine a query given that its retrieval scored poorly on relevance.

    Implementations return a single, more specific / search-ready query string.
    An :class:`IdentityRefiner` returns the query unchanged.
    """

    def refine(
        self,
        query: str,
        relevance_score: float,
        reasoning: str = "",
    ) -> str:
        """Return a refined query string for the next retrieval round."""
        ...


class IdentityRefiner:
    """A no-op refiner that returns ``query`` unchanged.

    Lets callers treat "refinement disabled" uniformly as a
    :class:`QueryRefiner` rather than branching on ``None``. With this wired the
    self-reflective loop still grades each retrieval but never re-retrieves.
    """

    def refine(
        self,
        query: str,
        relevance_score: float,
        reasoning: str = "",
    ) -> str:
        return str(query) if str(query).strip() else ""

    def __repr__(self) -> str:
        return "IdentityRefiner()"


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty refine response")
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    raise ValueError("refine response was not valid JSON")


class LLMQueryRefiner:
    """Query refiner backed by an :class:`LLMClient`.

    The model is told the previous query retrieved weakly relevant chunks
    (with the grader's score + reasoning) and asked to emit one *more specific,
    search-ready* query that keeps the same intent. The output is coerced
    leniently and *always* falls back to the original query on an unparseable
    response -- so a bad model call degrades to no refinement, never aborts the
    self-reflective loop.

    Parameters
    ----------
    client:
        Any :class:`LLMClient`. Reused verbatim from the generator.
    model:
        Model name forwarded to the client (ignored by fakes).
    temperature:
        Low (default ``0.3``) for faithful, controlled refinements.
    use_json_mode:
        Request JSON-mode structured output when supported.
    max_chars:
        Cap on the refined query length.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.3,
        use_json_mode: bool = True,
        max_chars: int = DEFAULT_MAX_REFINED_CHARS,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._use_json_mode = use_json_mode
        self._max_chars = max_chars if max_chars and max_chars > 0 else DEFAULT_MAX_REFINED_CHARS
        self.fallbacks = 0

    @property
    def model(self) -> str | None:
        return self._model

    def refine(
        self,
        query: str,
        relevance_score: float,
        reasoning: str = "",
    ) -> str:
        original = str(query).strip()
        if not original:
            return ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query-refinement module for semantic retrieval. A "
                    "previous search retrieved chunks that scored LOW on relevance "
                    "to the user's question. Rewrite the question into ONE more "
                    "specific, search-ready query that keeps the SAME intent but is "
                    "more likely to retrieve the right content (clarify entities, "
                    "add discriminating terms, remove ambiguity). Do NOT change the "
                    "meaning or invent unrelated entities. Respond with ONLY a JSON "
                    'object: {"refined_query": "..."}. No commentary.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {original}\n"
                    f"Retrieval relevance score (0-1, lower is worse): "
                    f"{float(relevance_score):.2f}\n"
                    f"Grader feedback: {reasoning.strip() or 'low relevance'}\n\n"
                    "Produce one refined query."
                ),
            },
        ]
        try:
            raw = self._client.complete(
                messages,
                model=self._model,
                temperature=self._temperature,
                response_format=JSON_RESPONSE_FORMAT if self._use_json_mode else None,
            )
            payload = json.loads(_extract_json(raw))
            refined = ""
            if isinstance(payload, dict):
                refined = str(payload.get("refined_query", "")).strip()
        except (ValueError, json.JSONDecodeError, AttributeError, TypeError) as exc:
            self.fallbacks += 1
            _logger.warning("LLMQueryRefiner parse failure (%s); identity fallback", exc)
            return original

        if not refined or refined.lower() == original.lower():
            self.fallbacks += 1
            return original
        if len(refined) > self._max_chars:
            refined = refined[: self._max_chars].rstrip()
        return refined

    def __repr__(self) -> str:
        return f"LLMQueryRefiner(model={self._model!r})"


__all__ = [
    "DEFAULT_MAX_REFINED_CHARS",
    "IdentityRefiner",
    "LLMQueryRefiner",
    "QueryRefiner",
]
