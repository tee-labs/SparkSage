"""Document summarization (extractive pure stdlib + optional LLM).

A document-level summary is one of the parsed artifacts an enterprise knowledge
service extracts alongside the title/body/tags. This module ships two
implementations of the same :class:`Summarizer` protocol:

* :class:`ExtractiveSummarizer` -- the cheap, offline, explainable default. It
  cleans Markdown noise (code fences / tables / images / link URLs / blockquotes
  / horizontal rules), splits into sentences (ASCII + CJK terminators), scores
  each sentence by a position-aware normalized word-frequency signal, then
  greedily selects the top sentences via Maximal Marginal Relevance (MMR) so the
  summary is not a list of near-duplicate sentences.
* :class:`LLMSummarizer` -- the high-quality, budget-allowing alternative. It
  reuses the existing :class:`~sparksage.generator.LLMClient` (no new LLM
  abstraction), asks the model for a single concise summary, and degrades
  gracefully to an :class:`ExtractiveSummarizer` on any failure (empty / parse
  error / exception) so a bad LLM call never costs the document its summary.

Both depend only on stdlib plus the :class:`~sparksage.tags.tokenizer.Tokenizer`
protocol and the stop-word sets in :mod:`sparksage.tags.stoplist` (the LLM
variant additionally depends on :class:`LLMClient`, exactly like the generator /
reader / query cores).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sparksage.generator.client import JSON_RESPONSE_FORMAT, LLMClient
from sparksage.tags.stoplist import DEFAULT_STOPWORDS
from sparksage.tags.tokenizer import AutoTokenizer, Tokenizer

_logger = logging.getLogger(__name__)

#: Sentence splitter: keep sentence content, discard the trailing punctuation.
#: Handles ASCII and CJK sentence terminators (incl. ``;`` / ``；`` / ``：``)
#: plus newlines.
_SENTENCE_SPLIT_RE = re.compile(r"[^.!?;:。！？；：\n\r]+")

#: Leading Markdown heading markers, stripped from summary sentences.
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*")

#: Markdown emphasis / list markers, stripped from the start of summary sentences.
_LEADING_NOISE_RE = re.compile(r"^\s*[*_>\-\u2022]+\s*")

#: Fenced code blocks (```...``` or ~~~...~~~ with optional language tag).
_CODE_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,}).*?$.*?^\s{0,3}\1\s*$", re.MULTILINE | re.DOTALL)

#: Standalone HTML-ish / Markdown image tags: ``![alt](url)``.
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

#: Inline links ``[text](url)`` -> keep just the text.
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")

#: Markdown table rows (``| a | b |``) and the delimiter row (``|---|---|``).
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)

#: Horizontal rules (``---`` / ``***`` / ``___`` on their own line).
_HR_RE = re.compile(r"^\s{0,3}([-*_])\1{2,}\s*$", re.MULTILINE)

#: Minimum sentence length (characters) to be eligible as a summary sentence.
_MIN_SENTENCE_LEN = 8

#: Position/lead bonus: the first sentences get a multiplicative boost so a
#: summary favours the document's opening (where the gist usually lives in
#: business / report / manual prose). Decays linearly across the first
#: ``_LEAD_WINDOW`` sentences.
_LEAD_WINDOW = 5
_LEAD_BONUS = 0.35

#: MMR lambda: trade-off between sentence relevance and novelty. Higher = more
#: relevance, lower = more diversity. ``0.6`` favours relevance while still
#: suppressing near-duplicate picks.
_MMR_LAMBDA = 0.6

#: Hard redundancy cap: a candidate whose token overlap (Jaccard) with any
#: already-picked sentence exceeds this is dropped entirely, regardless of its
#: relevance score. Guarantees a summary never restates the same point; only
#: triggers on near-restatements (``> 0.65`` shared topical tokens).
_REDUNDANCY_CAP = 0.65


@runtime_checkable
class Summarizer(Protocol):
    """Produce a short document-level summary from raw text."""

    def summarize(self, text: str, *, max_sentences: int = 3) -> str:
        """Return a summary of at most ``max_sentences`` sentences."""
        ...


@dataclass(frozen=True)
class _ScoredSentence:
    index: int
    text: str
    score: float
    tokens: frozenset[str]


def _clean_sentence(raw: str) -> str:
    """Strip Markdown heading / emphasis / list markers from a sentence."""
    s = _HEADING_RE.sub("", raw)
    s = _LEADING_NOISE_RE.sub("", s)
    return s.strip()


def _strip_markdown_noise(text: str) -> str:
    """Remove Markdown constructs that should never feed sentence scoring.

    Drops fenced code blocks, image tags, table rows, horizontal rules and the
    URL part of inline links (keeping the anchor text so it can still score).
    Done once, before sentence splitting, so the splitter never sees ``|`` row
    fragments or ``` fences as "sentences".
    """
    s = _CODE_FENCE_RE.sub("", text)
    s = _IMAGE_RE.sub("", s)
    s = _LINK_RE.sub(r"\1", s)
    s = _TABLE_ROW_RE.sub("", s)
    s = _HR_RE.sub("", s)
    return s


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into trimmed, non-empty sentences.

    Handles ASCII (``.!?;:``) and CJK (``。！？；：``) terminators and newlines,
    and strips leading Markdown heading / emphasis / list markers so a summary
    never starts with ``# `` or ``* ``.
    """
    cleaned_text = _strip_markdown_noise(text)
    out: list[str] = []
    for piece in _SENTENCE_SPLIT_RE.findall(cleaned_text):
        cleaned = _clean_sentence(piece)
        if cleaned:
            out.append(cleaned)
    return out


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Token-overlap similarity in ``[0, 1]`` (0 for two empty sets)."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


class ExtractiveSummarizer:
    """Frequency-based extractive summarizer with MMR de-duplication.

    Each sentence scores the sum of its non-stopword tokens' frequencies,
    normalized by the square root of its token count (so a long, dense sentence
    is not over-penalized the way a linear ``/len`` would). A small *lead
    bonus* multiplies the score of the first few sentences, matching the
    well-known prior that the document's opening carries the gist. The top
    ``max_sentences`` are then chosen greedily via Maximal Marginal Relevance:
    each pick maximizes ``lambda * relevance - (1 - lambda) * max_similarity``
    against the already-picked sentences, so the summary is not a list of
    near-duplicates. Selected sentences are returned in their *original*
    document order.

    Parameters
    ----------
    tokenizer:
        Any :class:`~sparksage.tags.tokenizer.Tokenizer`. Defaults to a fresh
        :class:`~sparksage.tags.tokenizer.AutoTokenizer` (CJK-aware).
    stopwords:
        Stop-word set. Defaults to
        :data:`~sparksage.tags.stoplist.DEFAULT_STOPWORDS`.

    Examples
    --------
    >>> text = (
    ...     "Revenue grew 12 percent. The growth was driven by APAC expansion. "
    ...     "Next year we enter Europe. Thank you for reading."
    ... )
    >>> s = ExtractiveSummarizer().summarize(text, max_sentences=1)
    >>> isinstance(s, str) and len(s) > 0
    True
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer | None = None,
        stopwords: frozenset[str] | None = None,
    ) -> None:
        self._tokenizer: Tokenizer = tokenizer if tokenizer is not None else AutoTokenizer()
        self._stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS

    def summarize(self, text: str, *, max_sentences: int = 3) -> str:
        if not isinstance(max_sentences, int) or isinstance(max_sentences, bool):
            raise TypeError("max_sentences must be an int")
        if max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")
        sentences = [s for s in split_sentences(text) if len(s) >= _MIN_SENTENCE_LEN]
        if not sentences:
            stripped = text.strip()
            return stripped

        freq: dict[str, int] = {}
        tokenized: list[list[str]] = []
        for sentence in sentences:
            toks = [
                t.lower()
                for t in self._tokenizer.tokenize(sentence)
                if t.lower() not in self._stopwords and any(c.isalnum() for c in t)
            ]
            tokenized.append(toks)
            for t in toks:
                freq[t] = freq.get(t, 0) + 1

        if not any(tokenized):
            # No topical tokens -- fall back to the first sentences verbatim.
            return " ".join(sentences[:max_sentences])

        max_freq = max(freq.values())
        scored: list[_ScoredSentence] = []
        for i, (sentence, toks) in enumerate(zip(sentences, tokenized, strict=True)):
            if not toks:
                base = 0.0
            else:
                total = sum(freq[t] for t in toks)
                # sqrt normalization: rewards informative density without the
                # harsh penalty that linear /len imposes on long sentences.
                base = (total / max_freq) / (len(toks) ** 0.5)
            # Linearly decaying lead bonus over the first _LEAD_WINDOW sentences.
            if i < _LEAD_WINDOW:
                base *= 1.0 + _LEAD_BONUS * (1.0 - i / _LEAD_WINDOW)
            scored.append(
                _ScoredSentence(
                    index=i, text=sentence, score=base, tokens=frozenset(toks)
                )
            )

        # Maximal Marginal Relevance greedy selection.
        remaining = list(scored)
        picked: list[_ScoredSentence] = []
        while remaining and len(picked) < max_sentences:
            best: _ScoredSentence | None = None
            best_key = -1.0
            for cand in remaining:
                if picked:
                    sim = max(_jaccard(cand.tokens, p.tokens) for p in picked)
                else:
                    sim = 0.0
                # Hard redundancy cap: never restates a picked sentence.
                if sim > _REDUNDANCY_CAP:
                    continue
                mmr = _MMR_LAMBDA * cand.score - (1.0 - _MMR_LAMBDA) * sim
                if mmr > best_key:
                    best_key = mmr
                    best = cand
            if best is None:
                break
            picked.append(best)
            remaining.remove(best)

        picked.sort(key=lambda s: s.index)
        return " ".join(s.text for s in picked)


class LLMSummarizer:
    """LLM-backed summarizer implementing the :class:`Summarizer` protocol.

    Asks the model for a single concise summary of the document (free-form text,
    *not* a re-statement of the input) and returns it verbatim. On any failure
    -- empty response, JSON parse error, or a client exception -- it degrades
    gracefully to the configured fallback (by default a fresh
    :class:`ExtractiveSummarizer`) so a document is never left without a summary
    because of a flaky LLM call.

    The model is asked to emit ``{"summary": "..."}`` JSON so JSON-mode-capable
    providers can be constrained, but a non-JSON response is also accepted (the
    raw text is used) so this works against providers without JSON mode.

    Parameters
    ----------
    client:
        Any :class:`LLMClient` (e.g. :class:`OpenAICompatibleClient`,
        :class:`FakeLLMClient`). Reused verbatim from the generator.
    model:
        Model name forwarded to the client (ignored by fakes).
    temperature:
        Low (default ``0.3``) for faithful, low-hallucination summaries.
    max_chars:
        Soft cap on summary length, advertised in the prompt (default ``300``).
    language:
        BCP-47 code; the model is asked to write the summary in this language.
    use_json_mode:
        Request JSON-mode structured output when supported.
    fallback:
        :class:`Summarizer` used on any failure. Defaults to a fresh
        :class:`ExtractiveSummarizer`.

    Examples
    --------
    >>> from sparksage.generator import FakeLLMClient
    >>> client = FakeLLMClient(responses=['{"summary": "A short doc summary."}'])
    >>> s = LLMSummarizer(client).summarize("Some long document body ...")
    >>> isinstance(s, str) and s
    True
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_chars: int = 300,
        language: str = "en",
        use_json_mode: bool = True,
        fallback: Summarizer | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_chars = max_chars
        self._language = language
        self._use_json_mode = use_json_mode
        self._fallback: Summarizer = fallback if fallback is not None else ExtractiveSummarizer()

    def summarize(self, text: str, *, max_sentences: int = 3) -> str:
        if not isinstance(max_sentences, int) or isinstance(max_sentences, bool):
            raise TypeError("max_sentences must be an int")
        if max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")

        body = text.strip()
        if not body:
            return text.strip()

        messages = self._build_messages(body)
        try:
            response_text = self._client.complete(
                messages,
                model=self._model,
                temperature=self._temperature,
                response_format=JSON_RESPONSE_FORMAT if self._use_json_mode else None,
            )
        except Exception as exc:  # pragma: no cover - defensive, depends on SDK
            _logger.warning("LLM summarizer call failed: %s; using fallback", exc)
            return self._fallback.summarize(text, max_sentences=max_sentences)

        summary = self._parse_response(response_text)
        if not summary or not summary.strip():
            _logger.warning("LLM summarizer returned empty response; using fallback")
            return self._fallback.summarize(text, max_sentences=max_sentences)
        return summary.strip()

    def _build_messages(self, body: str) -> list[dict[str, str]]:
        lang_clause = (
            f" Write the summary in language '{self._language}'."
            if self._language
            else ""
        )
        system = (
            "You are a concise document summarizer. Produce a faithful summary "
            f"of at most {self._max_chars} characters that captures the document's "
            f"main point and key supporting facts. Do not invent information not "
            f"present in the document.{lang_clause} Respond as JSON with a single "
            "'summary' key holding the summary text."
        )
        # Keep the prompt bounded: very large bodies are truncated so the request
        # stays cheap; the lead + tail together usually carry the gist.
        cap = 12000
        if len(body) > cap:
            half = cap // 2
            body = body[:half] + "\n…[truncated]…\n" + body[-half:]
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": body},
        ]

    @staticmethod
    def _parse_response(response_text: str) -> str:
        """Extract the summary string from a possibly-noisy model response.

        Accepts ``{"summary": "..."}`` JSON (with or without ```json fences),
        and falls back to the raw text when the response is not JSON.
        """
        if not response_text:
            return ""
        cleaned = response_text.strip()
        # Try fenced JSON first.
        fence = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if fence:
            try:
                obj = json.loads(fence.group(1))
                if isinstance(obj, dict) and isinstance(obj.get("summary"), str):
                    return obj["summary"]
            except json.JSONDecodeError:
                pass
        # Try a bare JSON object.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and isinstance(obj.get("summary"), str):
                    return obj["summary"]
            except json.JSONDecodeError:
                pass
        # Fall back to the raw text (provider without JSON mode / free-form).
        return cleaned


def default_summarizer() -> Summarizer:
    """Return the conventional default :class:`ExtractiveSummarizer`."""
    return ExtractiveSummarizer()


__all__ = [
    "ExtractiveSummarizer",
    "LLMSummarizer",
    "Summarizer",
    "default_summarizer",
    "split_sentences",
]
