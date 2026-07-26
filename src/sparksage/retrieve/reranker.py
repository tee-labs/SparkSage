"""Re-ranking of retrieved candidates (Protocol + LLM + identity defaults).

Coarse two-tower recall (cheap dense + lexical) returns a generous pool; a
*cross-attention* re-ranker then re-scores that pool against the query and
keeps the top few. This is, after chunking strategy, the largest single-point
lever on RAG answer quality. The core depends only on the
:class:`Reranker` protocol and reuses the existing :class:`~sparksage.generator.LLMClient`
-- no new LLM abstraction -- so the default :class:`LLMReranker` asks the model
to order the candidate IdeaBlocks, and an optional future cross-encoder SDK
backend implements the same protocol under an optional extra.

Two implementations ship:

- :class:`LLMReranker`: the default -- presents each candidate's
  ``name`` / ``critical_question`` / ``trusted_answer`` to the model and asks
  for a relevance-ordered index list (lenient -> strict coercion, robust to
  messy JSON). Falls back to identity ordering on a bad response.
- :class:`IdentityReranker`: a no-op that preserves input order, so reranking
  is always configurable as "off".
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, runtime_checkable

from sparksage.generator.client import JSON_RESPONSE_FORMAT, LLMClient
from sparksage.retrieve.models import RetrievedChunk

_logger = logging.getLogger(__name__)

#: Candidate id marker rendered into the rerank prompt.
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$", re.MULTILINE)


@runtime_checkable
class Reranker(Protocol):
    """Re-order / re-score a pool of retrieved candidates for a query.

    Implementations receive the candidate :class:`RetrievedChunk` list (already
    fused) and return a re-ordered, re-scored slice (best first), typically
    shorter than the input (``top_n``). Scores in the returned chunks are the
    reranker's own relevance scores.
    """

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        ...


class IdentityReranker:
    """A no-op reranker that preserves the input order.

    Returns the first ``top_n`` chunks unchanged (scores untouched), so the
    retrieval orchestrator can treat "reranking disabled" as just another
    :class:`Reranker` rather than branching on ``None``.
    """

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        out = chunks if top_n is None else chunks[:top_n]
        return [_with_rank(c, i) for i, c in enumerate(out)]

    def __repr__(self) -> str:
        return "IdentityReranker()"


def _with_rank(chunk: RetrievedChunk, rank: int) -> RetrievedChunk:
    from dataclasses import replace

    return replace(chunk, rank=rank)


def _render_candidates(chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
    """Render candidates as ``[i] name | question | answer`` and return ids."""
    lines: list[str] = []
    ids: list[str] = []
    for i, c in enumerate(chunks):
        ids.append(str(c.block.id))
        ans = c.block.trusted_answer
        if len(ans) > 220:
            ans = ans[:219] + "…"
        lines.append(
            f"[{i}] {c.block.name} || {c.block.critical_question} || {ans}"
        )
    return "\n".join(lines), ids


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty rerank response")
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
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
    raise ValueError("rerank response was not valid JSON")


def _parse_order(payload: str, n: int) -> list[int] | None:
    """Parse the model's ordering into a list of candidate indices."""
    data = json.loads(payload)
    order: list[int] = []

    def _coerce_int(v: Any) -> int | None:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return None
        return iv if 0 <= iv < n else None

    if isinstance(data, list):
        for v in data:
            iv = _coerce_int(v)
            if iv is not None:
                order.append(iv)
    elif isinstance(data, dict):
        raw = data.get("order") or data.get("ranking") or data.get("ranked_ids")
        if isinstance(raw, list):
            for v in raw:
                iv = _coerce_int(v)
                if iv is not None:
                    order.append(iv)
        ranked = data.get("ranked")
        if isinstance(ranked, list) and not order:
            for v in ranked:
                iv = _coerce_int(v)
                if iv is not None:
                    order.append(iv)
    seen: set[int] = set()
    deduped: list[int] = []
    for iv in order:
        if iv not in seen:
            seen.add(iv)
            deduped.append(iv)
    for i in range(n):
        if i not in seen:
            deduped.append(i)
    return deduped if deduped else None


class LLMReranker:
    """Re-rank candidate blocks by asking an :class:`LLMClient` to order them.

    The model is given the query plus each candidate's
    ``name`` / ``critical_question`` / ``trusted_answer`` and asked to emit a
    JSON array of candidate indices in descending relevance order. The output is
    coerced leniently (handles ``{"order": [...]}``, bare ``[...]``, prose-wrapped
    JSON, out-of-range / duplicate indices) and *always* falls back to the input
    order on an unparseable response -- so a bad model call degrades to
    identity, never aborts retrieval.

    Parameters
    ----------
    client:
        Any :class:`LLMClient` (e.g. :class:`OpenAICompatibleClient`,
        :class:`FakeLLMClient`). Reused verbatim from the generator -- no new
        LLM abstraction.
    model:
        Model name forwarded to the client (ignored by fakes).
    temperature:
        Low (default ``0.0``) for stable rankings.
    use_json_mode:
        Request JSON-mode structured output when supported.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.0,
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

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        from dataclasses import replace

        if not chunks:
            return []
        if len(chunks) == 1:
            return [_with_rank(chunks[0], 0)]

        rendered, _ids = _render_candidates(chunks)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a relevance re-ranker. Given a query and candidate "
                    "knowledge chunks, order them by descending relevance to the "
                    "query. Respond with ONLY a JSON array of the candidate "
                    "indices in relevance order, best first -- e.g. [2, 0, 1]. "
                    "No commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query.strip()}\n\nCandidates:\n{rendered}\n\n"
                    "Return the JSON array of indices, best first."
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
            order = _parse_order(_extract_json(raw), len(chunks))
        except (ValueError, json.JSONDecodeError) as exc:
            self.fallbacks += 1
            _logger.warning("LLMReranker parse failure (%s); falling back to input order", exc)
            order = None

        if order is None:
            self.fallbacks += 1
            order = list(range(len(chunks)))

        n = len(order)
        # Descending-relevance score: normalized to (0, 1].
        reranked: list[RetrievedChunk] = []
        for rank, idx in enumerate(order):
            if idx < 0 or idx >= len(chunks):
                continue
            score = (n - rank) / n if n else 0.0
            chunk = chunks[idx]
            reranked.append(
                replace(chunk, score=score, dense_score=chunk.dense_score, rank=rank)
            )
        if top_n is not None:
            reranked = reranked[:top_n]
        return reranked

    def __repr__(self) -> str:
        return f"LLMReranker(model={self._model!r})"


__all__ = [
    "IdentityReranker",
    "LLMReranker",
    "Reranker",
]
