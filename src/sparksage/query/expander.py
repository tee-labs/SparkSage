"""Query expansion: multi-query / HyDE-style variants for fused recall.

A single dense query can miss relevant blocks whose wording differs from the
query even when the *meaning* matches. Multi-query expansion closes that gap:
produce ``n`` paraphrase variants of the query, retrieve for each, and fuse the
ranked lists via RRF (the :class:`~sparksage.qa.QAEngine` does this when an
expander is wired). This is the classic multi-query / RAG-Fusion retrieval
boost -- orthogonal to the rewriter (which produces *one* improved query) and
to sub-query decomposition (which splits a *compound* question).

The core depends only on the :class:`QueryExpander` protocol and reuses the
existing :class:`~sparksage.generator.LLMClient`, so it is fully unit-testable
with :class:`~sparksage.generator.FakeLLMClient`.

Two implementations ship:

- :class:`LLMQueryExpander`: the default -- asks the model for ``n`` search-
  engine-style paraphrases (lenient -> strict coercion, robust to messy JSON).
- :class:`IdentityExpander`: returns ``[query]`` unchanged, so expansion is
  always configurable as "off" without branching.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol, runtime_checkable

from sparksage.generator.client import JSON_RESPONSE_FORMAT, LLMClient

_logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$", re.MULTILINE)

#: Default number of variants an expander produces.
DEFAULT_N_VARIANTS = 3


@runtime_checkable
class QueryExpander(Protocol):
    """Expand one query into ``n`` search-ready variants.

    Implementations return a de-duplicated list of variant strings. An
    :class:`IdentityExpander` returns ``[query]``. The variants are retrieved
    independently and their ranked lists fused (RRF) by the QA engine.
    """

    def expand(self, query: str, *, n: int = DEFAULT_N_VARIANTS) -> list[str]:
        """Return up to ``n`` variant phrasings of ``query`` (best first)."""
        ...


class IdentityExpander:
    """A no-op expander that returns ``[query]``.

    Lets callers treat "expansion disabled" uniformly as a
    :class:`QueryExpander` rather than branching on ``None``.
    """

    def expand(self, query: str, *, n: int = DEFAULT_N_VARIANTS) -> list[str]:
        return [str(query)] if str(query).strip() else []

    def __repr__(self) -> str:
        return "IdentityExpander()"


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty expand response")
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
    raise ValueError("expand response was not valid JSON")


def _dedupe_variants(items: list[str], original: str, n: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = {original.strip().lower()} if original.strip() else set()
    for raw in items:
        v = (raw or "").strip()
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
        if len(out) >= n:
            break
    return out


class LLMQueryExpander:
    """Multi-query expansion backed by an :class:`LLMClient`.

    The model is asked for ``n`` search-engine-style paraphrases of the query
    -- same intent, different wording -- so that RRF-fused recall catches
    blocks phrased differently from the query. The output is coerced leniently
    (handles bare ``[...]``, ``{"variants": [...]}``, prose-wrapped JSON) and
    *always* falls back to ``[query]`` on an unparseable response -- so a bad
    model call degrades to no expansion, never aborts the pipeline.

    Parameters
    ----------
    client:
        Any :class:`LLMClient`. Reused verbatim from the generator.
    model:
        Model name forwarded to the client (ignored by fakes).
    temperature:
        Slightly higher (default ``0.4``) to encourage variant diversity.
    use_json_mode:
        Request JSON-mode structured output when supported.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.4,
        use_json_mode: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._use_json_mode = use_json_mode
        self.fallbacks = 0

    @property
    def model(self) -> str | None:
        return self._model

    def expand(self, query: str, *, n: int = DEFAULT_N_VARIANTS) -> list[str]:
        original = str(query).strip()
        if not original:
            return []
        if n < 1:
            return []
        if n == 1:
            return [original]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query-expansion module for semantic retrieval. Given a "
                    "user query, produce search-engine-style paraphrases with the SAME "
                    "intent but DIFFERENT wording / terminology, to improve recall. Do "
                    "NOT change the meaning or add new entities. Respond with ONLY a "
                    "JSON object: {\"variants\": [\"...\", \"...\"]}. No commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {original}\n\nProduce up to {n} paraphrased variants."
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
            items = payload.get("variants") if isinstance(payload, dict) else payload
            if not isinstance(items, list):
                raise ValueError("expected a variants list")
        except (ValueError, json.JSONDecodeError) as exc:
            self.fallbacks += 1
            _logger.warning("LLMQueryExpander parse failure (%s); identity fallback", exc)
            return [original]

        variants = _dedupe_variants([str(x) for x in items], original, n)
        if not variants:
            self.fallbacks += 1
            return [original]
        return variants

    def __repr__(self) -> str:
        return f"LLMQueryExpander(model={self._model!r})"


__all__ = [
    "DEFAULT_N_VARIANTS",
    "IdentityExpander",
    "LLMQueryExpander",
    "QueryExpander",
]
