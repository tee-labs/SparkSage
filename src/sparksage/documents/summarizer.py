"""Extractive document summarization (pure stdlib).

A document-level summary is one of the parsed artifacts an enterprise knowledge
service extracts alongside the title/body/tags. This module ships a dependency-
free :class:`ExtractiveSummarizer`: score each sentence by the sum of its words'
normalized frequencies (excluding stop words), then return the top-``N`` highest-
scoring sentences in their original order. It is the cheap, offline, explainable
default; an LLM-generated summary can replace it where a budget allows.

Depends only on the :class:`~sparksage.tags.tokenizer.Tokenizer` protocol and the
stop-word sets in :mod:`sparksage.tags.stoplist` -- no NLTK / spaCy / HuggingFace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sparksage.tags.stoplist import DEFAULT_STOPWORDS
from sparksage.tags.tokenizer import AutoTokenizer, Tokenizer

#: Sentence splitter: keep sentence content, discard the trailing punctuation.
#: Handles ASCII and CJK sentence terminators plus newlines.
_SENTENCE_SPLIT_RE = re.compile(r"[^.!?。！？\n\r]+")

#: Leading Markdown heading markers, stripped from summary sentences.
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*")

#: Markdown emphasis / list markers, stripped from the start of summary sentences.
_LEADING_NOISE_RE = re.compile(r"^\s*[*_>\-\u2022]+\s*")

#: Minimum sentence length (characters) to be eligible as a summary sentence.
_MIN_SENTENCE_LEN = 8


@runtime_checkable
class Summarizer(Protocol):
    """Produce a short document-level summary from raw text."""

    def summarize(self, text: str, *, max_sentences: int = 3) -> str:
        """Return an extractive summary of at most ``max_sentences`` sentences."""
        ...


@dataclass(frozen=True)
class _ScoredSentence:
    index: int
    text: str
    score: float


def _clean_sentence(raw: str) -> str:
    """Strip Markdown heading / emphasis / list markers from a sentence."""
    s = _HEADING_RE.sub("", raw)
    s = _LEADING_NOISE_RE.sub("", s)
    return s.strip()


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into trimmed, non-empty sentences.

    Handles ASCII (``.!?``) and CJK (``。！？``) terminators and newlines, and
    strips leading Markdown heading / emphasis / list markers so a summary never
    starts with ``# `` or ``* ``.
    """
    out: list[str] = []
    for piece in _SENTENCE_SPLIT_RE.findall(text):
        cleaned = _clean_sentence(piece)
        if cleaned:
            out.append(cleaned)
    return out


class ExtractiveSummarizer:
    """Frequency-based extractive summarizer.

    Each sentence scores the sum of its non-stopword tokens' frequencies,
    normalized by sentence length (so a long list of common words does not beat
    a tight topical sentence). The top ``max_sentences`` are returned in their
    *original* document order, joined with single spaces.

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

        if not freq:
            # No topical tokens -- fall back to the first sentences verbatim.
            return " ".join(sentences[:max_sentences])

        max_freq = max(freq.values())
        scored: list[_ScoredSentence] = []
        for i, (sentence, toks) in enumerate(zip(sentences, tokenized, strict=True)):
            if not toks:
                score = 0.0
            else:
                total = sum(freq[t] for t in toks)
                score = (total / max_freq) / max(len(toks), 1)
            scored.append(_ScoredSentence(index=i, text=sentence, score=score))

        top = sorted(scored, key=lambda s: (-s.score, s.index))[:max_sentences]
        top.sort(key=lambda s: s.index)
        return " ".join(s.text for s in top)


def default_summarizer() -> Summarizer:
    """Return the conventional default :class:`ExtractiveSummarizer`."""
    return ExtractiveSummarizer()


__all__ = [
    "ExtractiveSummarizer",
    "Summarizer",
    "default_summarizer",
    "split_sentences",
]
