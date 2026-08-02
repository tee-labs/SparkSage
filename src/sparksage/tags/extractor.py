"""Keyword / tag extraction algorithms (pure stdlib).

This is the dependency-free answer to "when the document has no tags, generate
them from the content". Each :class:`KeywordExtractor` turns a single text into
a ranked list of :class:`KeywordScore` (``keyword`` + ``score``) that the
document-management layer (:mod:`sparksage.documents`) stores as free-form tags.

Three classic, language-agnostic algorithms ship out of the box:

* :class:`TfidfKeywordExtractor` -- term frequency * inverse document frequency,
  where the "corpus" is the document's own sentences/paragraphs. Surfaces words
  that are frequent in this document but spread across many of its sections.
* :class:`RakeKeywordExtractor` -- Rapid Automatic Keyword Extraction (Rose et
  al. 2010): build candidate phrases by splitting on stop words + punctuation,
  score each word by degree/frequency, rank phrases by the sum of member scores.
* :class:`TextRankKeywordExtractor` -- Mihalcea & Tarau's graph-based ranker:
  build a word co-occurrence graph within a sliding window, iterate a PageRank-
  style fixed point, then merge adjacent top-ranked words into multi-word
  keywords.

All three depend only on the :class:`~sparksage.tags.tokenizer.Tokenizer`
protocol and the stop-word sets in :mod:`sparksage.tags.stoplist`. No third-party
NLP library is imported here -- ``jieba`` lives behind
:class:`~sparksage.tags.tokenizer.JiebaTokenizer` and is the user's choice via
``tokenizer=``.

Selection helper: :func:`make_extractor` / :func:`default_extractor`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sparksage.tags.cohesion import DEFAULT_MIN_COHESION, cjk_bigram_gate
from sparksage.tags.stoplist import DEFAULT_STOPWORDS
from sparksage.tags.tokenizer import (
    AutoTokenizer,
    Tokenizer,
    is_cjk_char,
)

#: Maximum iterations for the TextRank fixed-point iteration. The iteration
#: converges quickly for vocabulary sizes typical of a single document; the cap
#: is a safety bound, not a tuning knob.
_TEXTRANK_MAX_ITER = 50

#: TextRank damping factor (the classic ``d = 0.85`` from PageRank).
_TEXTRANK_DAMPING = 0.85

#: TextRank co-occurrence window size, in tokens.
_TEXTRANK_WINDOW = 4

#: Maximum tokens in a TextRank merged keyword. Adjacent top-ranked words beyond
#: this length are flushed as separate keywords, avoiding run-on phrases that
#: span a whole sentence.
_TEXTRANK_MAX_MERGE_LEN = 4

#: Token length below which a token is dropped as too short to be a keyword
#: (covers stray punctuation survivors and single Latin letters). CJK bigrams
#: always pass.
_MIN_TOKEN_LEN = 2

#: Sentence / phrase boundary characters shared by all extractors.
_BOUNDARY_RE = re.compile(r"[.!?,;:\n\r\t()\"'“”‘’。！？，；：、（）【】《》\[\]{}]")


@dataclass(frozen=True)
class KeywordScore:
    """A single ranked keyword with its (algorithm-specific) score.

    Attributes
    ----------
    keyword:
        The candidate tag (a single token or a multi-word phrase, lower-cased).
    score:
        Algorithm-specific weight. Higher = more salient. Comparable within one
        extractor run; not meaningful across algorithms.
    """

    keyword: str
    score: float


@runtime_checkable
class KeywordExtractor(Protocol):
    """Turn a text into a ranked list of keyword candidates.

    Any callable with an ``extract(text, top_k)`` method returning
    ``list[KeywordScore]`` implements this -- the document layer depends on the
    protocol, never on a concrete algorithm.
    """

    def extract(self, text: str, top_k: int = 10) -> list[KeywordScore]:
        """Return the ``top_k`` highest-scoring keywords for ``text`` (best first).

        Fewer than ``top_k`` may be returned when the text yields fewer
        candidates.
        """
        ...


# ---------------------------------------------------------------------------- #
# shared helpers
# ---------------------------------------------------------------------------- #
def _split_sentences(text: str) -> list[str]:
    """Split ``text`` on sentence-ending punctuation / newlines.

    Returns non-empty, stripped sentences. Used as the "corpus" of sections by
    TF-IDF and as the unit of co-occurrence by TextRank.
    """
    parts = _BOUNDARY_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _is_cjk_bigram(token: str) -> bool:
    """True when ``token`` is exactly two CJK / kana / Hangul characters."""
    return len(token) == 2 and is_cjk_char(token[0]) and is_cjk_char(token[1])


def _tokenize_filtered(
    text: str,
    tokenizer: Tokenizer,
    stopwords: frozenset[str],
    *,
    keep_case: bool = False,
    cjk_allow: frozenset[str] | None = None,
) -> list[str]:
    """Tokenize ``text`` and drop stop words / very short / noisy tokens.

    ``cjk_allow`` is the blessed-bigram set from
    :func:`~sparksage.tags.cohesion.cjk_bigram_gate`; when present, 2-char CJK
    tokens that are *not* blessed (cross-boundary noise such as ``晨一``) are
    dropped before they can enter a phrase, a TF-IDF bag, or a TextRank graph --
    so the cohesion filter cleans scoring *and* tag output at once.
    """
    raw = tokenizer.tokenize(text)
    out: list[str] = []
    for tok in raw:
        t = tok if keep_case else tok
        if t in stopwords:
            continue
        # CJK tokens (bigrams / single chars) are always long enough; for Latin
        # require >= _MIN_TOKEN_LEN chars and at least one alphanumeric.
        if not any(c.isalnum() for c in t):
            continue
        is_cjk = any(ord(c) >= 0x3000 for c in t)
        if not is_cjk and len(t) < _MIN_TOKEN_LEN:
            continue
        if cjk_allow is not None and _is_cjk_bigram(t) and t not in cjk_allow:
            continue
        out.append(t)
    return out


def _normalize_phrase(tokens: list[str]) -> str:
    """Join a token list into a single keyword string (space-separated)."""
    return " ".join(tokens)


# ---------------------------------------------------------------------------- #
# RAKE
# ---------------------------------------------------------------------------- #
class RakeKeywordExtractor:
    """Rapid Automatic Keyword Extraction.

    Candidate phrases are maximal runs of non-stopword tokens delimited by stop
    words or punctuation. Each token scores ``degree / frequency`` (degree =
    co-occurrence breadth: a token that appears in long phrases ranks higher),
    and each phrase scores the sum of its member token scores.

    Language-agnostic: pass a CJK-capable tokenizer (the default
    :class:`~sparksage.tags.tokenizer.AutoTokenizer` auto-selects one) to
    extract Mandarin / Japanese keywords without ``jieba``.

    Parameters
    ----------
    tokenizer:
        Any :class:`~sparksage.tags.tokenizer.Tokenizer`. Defaults to a fresh
        :class:`~sparksage.tags.tokenizer.AutoTokenizer`.
    stopwords:
        Stop-word set used to split phrases. Defaults to
        :data:`~sparksage.tags.stoplist.DEFAULT_STOPWORDS`.
    min_phrase_len:
        Minimum number of tokens for a phrase to be kept (default ``1`` -- single
        tokens are valid keywords).
    max_phrase_len:
        Maximum number of tokens for a phrase (default ``5``) -- longer phrases
        are usually sentence fragments, not keywords.

    Examples
    --------
    >>> kw = RakeKeywordExtractor().extract("SparkSage turns documents into "
    ...                                      "knowledge chunks for RAG.")
    >>> any("sparksage" == k.keyword or "sparksage" in k.keyword for k in kw)
    True
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer | None = None,
        stopwords: frozenset[str] | None = None,
        min_phrase_len: int = 1,
        max_phrase_len: int = 5,
        min_cohesion: float | None = DEFAULT_MIN_COHESION,
    ) -> None:
        self._tokenizer: Tokenizer = tokenizer if tokenizer is not None else AutoTokenizer()
        self._stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self._min_phrase_len = max(1, int(min_phrase_len))
        self._max_phrase_len = max(self._min_phrase_len, int(max_phrase_len))
        self._min_cohesion = min_cohesion

    def extract(self, text: str, top_k: int = 10) -> list[KeywordScore]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        cjk_allow = cjk_bigram_gate(text, min_cohesion=self._min_cohesion)
        phrases = self._candidate_phrases(text, cjk_allow)
        if not phrases:
            return []

        freq: Counter[str] = Counter()
        degree: Counter[str] = Counter()
        for phrase in phrases:
            n = len(phrase)
            for tok in set(phrase):
                freq[tok] += phrase.count(tok)
                degree[tok] += n - 1

        word_score: dict[str, float] = {}
        for tok, f in freq.items():
            f = max(f, 1)
            word_score[tok] = (degree[tok] + f) / f

        scored: list[KeywordScore] = []
        for phrase in phrases:
            key = _normalize_phrase(phrase)
            if not key:
                continue
            score = sum(word_score.get(t, 0.0) for t in phrase)
            # weight slightly by phrase length so multi-word concepts win ties
            score *= 1.0 + 0.1 * (len(phrase) - 1)
            scored.append(KeywordScore(keyword=key, score=score))

        # Deduplicate keeping the highest-scoring instance of each keyword.
        best: dict[str, KeywordScore] = {}
        for ks in scored:
            cur = best.get(ks.keyword)
            if cur is None or ks.score > cur.score:
                best[ks.keyword] = ks
        ranked = sorted(
            best.values(), key=lambda k: (-k.score, k.keyword)
        )
        return ranked[:top_k]

    def _candidate_phrases(
        self, text: str, cjk_allow: frozenset[str] | None = None
    ) -> list[list[str]]:
        phrases: list[list[str]] = []
        for sentence in _split_sentences(text):
            current: list[str] = []
            for tok in self._tokenizer.tokenize(sentence):
                t = tok.lower()
                if t in self._stopwords or not any(c.isalnum() for c in t):
                    if current:
                        self._append(current, phrases)
                        current = []
                    continue
                is_cjk = any(ord(c) >= 0x3000 for c in t)
                if not is_cjk and len(t) < _MIN_TOKEN_LEN:
                    if current:
                        self._append(current, phrases)
                        current = []
                    continue
                # A cross-boundary CJK bigram (not blessed by the cohesion
                # filter) is a word boundary, not a keyword -- flush the
                # current phrase so the noise never lands inside a tag.
                if (
                    cjk_allow is not None
                    and _is_cjk_bigram(t)
                    and t not in cjk_allow
                ):
                    if current:
                        self._append(current, phrases)
                        current = []
                    continue
                current.append(t)
            if current:
                self._append(current, phrases)
        return phrases

    def _append(self, phrase: list[str], phrases: list[list[str]]) -> None:
        if self._min_phrase_len <= len(phrase) <= self._max_phrase_len:
            phrases.append(list(phrase))


# ---------------------------------------------------------------------------- #
# TF-IDF
# ---------------------------------------------------------------------------- #
class TfidfKeywordExtractor:
    """Term-frequency * inverse-document-frequency keyword extraction.

    Treats the document's own sentences as the "corpus": a term that is frequent
    overall but spread across many sentences is more topical than one bunched in
    a single sentence. Score is ``tf * idf`` with a sub-linear ``tf`` (``1 +
    log(tf)``) and ``idf = log((1 + N) / (1 + df)) + 1`` (smoothed, so a term in
    every section still scores > 0).

    Single tokens are scored and ranked; adjacent high-scoring tokens that share
    a sentence are *also* surfaced as phrases when they co-occur tightly, but the
    default output is single-token keywords -- the cleanest, most predictable
    behaviour.

    Parameters
    ----------
    tokenizer, stopwords:
        See :class:`RakeKeywordExtractor`.
    min_token_len:
        Minimum token length for Latin tokens (CJK always passes).
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer | None = None,
        stopwords: frozenset[str] | None = None,
        min_token_len: int = _MIN_TOKEN_LEN,
        min_cohesion: float | None = DEFAULT_MIN_COHESION,
    ) -> None:
        self._tokenizer: Tokenizer = tokenizer if tokenizer is not None else AutoTokenizer()
        self._stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self._min_token_len = max(1, int(min_token_len))
        self._min_cohesion = min_cohesion

    def extract(self, text: str, top_k: int = 10) -> list[KeywordScore]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        sentences = _split_sentences(text)
        if not sentences:
            return []

        cjk_allow = cjk_bigram_gate(text, min_cohesion=self._min_cohesion)
        doc_freq: Counter[str] = Counter()
        term_freq: Counter[str] = Counter()
        section_tokens: list[list[str]] = []
        for sentence in sentences:
            toks = self._tokens(sentence, cjk_allow)
            section_tokens.append(toks)
            for t in set(toks):
                doc_freq[t] += 1
            for t in toks:
                term_freq[t] += 1

        n_sections = len(sentences)
        scored: list[KeywordScore] = []
        for term, tf in term_freq.items():
            df = doc_freq.get(term, 0)
            idf = math.log((1 + n_sections) / (1 + df)) + 1.0
            sublinear_tf = 1.0 + math.log(tf) if tf > 0 else 0.0
            scored.append(KeywordScore(keyword=term, score=sublinear_tf * idf))

        ranked = sorted(scored, key=lambda k: (-k.score, k.keyword))
        return ranked[:top_k]

    def _tokens(
        self, text: str, cjk_allow: frozenset[str] | None = None
    ) -> list[str]:
        out: list[str] = []
        for tok in self._tokenizer.tokenize(text):
            t = tok.lower()
            if t in self._stopwords:
                continue
            if not any(c.isalnum() for c in t):
                continue
            is_cjk = any(ord(c) >= 0x3000 for c in t)
            if not is_cjk and len(t) < self._min_token_len:
                continue
            if cjk_allow is not None and _is_cjk_bigram(t) and t not in cjk_allow:
                continue
            out.append(t)
        return out


# ---------------------------------------------------------------------------- #
# TextRank
# ---------------------------------------------------------------------------- #
class TextRankKeywordExtractor:
    """Graph-based keyword extraction (Mihalcea & Tarau, 2004).

    Build an undirected word co-occurrence graph (edge between two non-stopword
    tokens appearing within a ``window``-token span of the same sentence), then
    iterate the TextRank fixed point::

        WS(V_i) = (1 - d) + d * sum_{j in In(i)} w_ji / out_strength(j) * WS(V_j)

    until convergence (or :data:`_TEXTRANK_MAX_ITER` iterations). Adjacent
    top-ranked tokens that co-occur in the text are then merged into multi-word
    keywords, mirroring the original paper.

    Language-agnostic: works on Latin tokens and CJK bigrams alike.

    Parameters
    ----------
    tokenizer, stopwords:
        See :class:`RakeKeywordExtractor`.
    window:
        Co-occurrence window in tokens (default :data:`_TEXTRANK_WINDOW`).
    damping:
        PageRank damping factor (default :data:`_TEXTRANK_DAMPING`).
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer | None = None,
        stopwords: frozenset[str] | None = None,
        window: int = _TEXTRANK_WINDOW,
        damping: float = _TEXTRANK_DAMPING,
        min_cohesion: float | None = DEFAULT_MIN_COHESION,
    ) -> None:
        self._tokenizer: Tokenizer = tokenizer if tokenizer is not None else AutoTokenizer()
        self._stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self._window = max(1, int(window))
        self._damping = float(damping)
        self._min_cohesion = min_cohesion

    def extract(self, text: str, top_k: int = 10) -> list[KeywordScore]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        sentences = _split_sentences(text)
        if not sentences:
            return []

        cjk_allow = cjk_bigram_gate(text, min_cohesion=self._min_cohesion)
        cooc: Counter[tuple[str, str]] = Counter()
        nodes: set[str] = set()
        for sentence in sentences:
            toks = _tokenize_filtered(
                sentence, self._tokenizer, self._stopwords, cjk_allow=cjk_allow
            )
            nodes.update(toks)
            w = self._window
            for i, a in enumerate(toks):
                for j in range(i + 1, min(i + w + 1, len(toks))):
                    b = toks[j]
                    if a == b:
                        continue
                    if a < b:
                        cooc[(a, b)] += 1
                    else:
                        cooc[(b, a)] += 1

        if not nodes:
            return []

        adj: dict[str, dict[str, float]] = {n: {} for n in nodes}
        for (a, b), weight in cooc.items():
            adj[a][b] = adj[a].get(b, 0.0) + weight
            adj[b][a] = adj[b].get(a, 0.0) + weight

        scores = self._iterate(adj)
        merged = self._merge_adjacent(text, scores, top_k, cjk_allow)
        return merged

    def _iterate(
        self, adj: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        nodes = list(adj.keys())
        n = len(nodes)
        if n == 0:
            return {}
        scores = {v: 1.0 for v in nodes}
        out_strength = {v: sum(adj[v].values()) for v in nodes}
        d = self._damping
        for _ in range(_TEXTRANK_MAX_ITER):
            new_scores: dict[str, float] = {}
            diff = 0.0
            for v in nodes:
                incoming = 0.0
                for u, w in adj[v].items():
                    s_out = out_strength.get(u, 0.0)
                    if s_out > 0:
                        incoming += w * (scores[u] / s_out)
                new_scores[v] = (1.0 - d) + d * incoming
                diff += abs(new_scores[v] - scores[v])
            scores = new_scores
            if diff < 1e-6:
                break
        return scores

    def _merge_adjacent(
        self,
        text: str,
        scores: dict[str, float],
        top_k: int,
        cjk_allow: frozenset[str] | None = None,
    ) -> list[KeywordScore]:
        if not scores:
            return []
        # Top candidates by score.
        ranked_words = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        keep_n = min(max(top_k * 3, top_k), len(ranked_words))
        top_words = {w for w, _ in ranked_words[:keep_n]}

        phrases: list[KeywordScore] = []
        seen_phrases: set[str] = set()
        for sentence in _split_sentences(text):
            toks = _tokenize_filtered(
                sentence, self._tokenizer, self._stopwords, cjk_allow=cjk_allow
            )
            current: list[str] = []
            current_score: list[float] = []
            for tok in toks:
                if tok in top_words:
                    if len(current) >= _TEXTRANK_MAX_MERGE_LEN:
                        self._flush_phrase(current, current_score, phrases, seen_phrases)
                        current, current_score = [], []
                    current.append(tok)
                    current_score.append(scores.get(tok, 0.0))
                else:
                    self._flush_phrase(current, current_score, phrases, seen_phrases)
                    current, current_score = [], []
            self._flush_phrase(current, current_score, phrases, seen_phrases)

        # Single high-scoring words that never co-occurred adjacent to another
        # top word still need to be surfaced.
        covered: set[str] = set()
        for ks in phrases:
            covered.update(ks.keyword.split())
        for word, score in ranked_words:
            if word in covered:
                continue
            if word not in seen_phrases:
                phrases.append(KeywordScore(keyword=word, score=score))
                seen_phrases.add(word)

        phrases.sort(key=lambda k: (-k.score, k.keyword))
        return phrases[:top_k]

    @staticmethod
    def _flush_phrase(
        current: list[str],
        current_score: list[float],
        phrases: list[KeywordScore],
        seen: set[str],
    ) -> None:
        if not current:
            return
        key = _normalize_phrase(current)
        if not key or key in seen:
            return
        score = sum(current_score) / max(len(current_score), 1)
        phrases.append(KeywordScore(keyword=key, score=score))
        seen.add(key)


# ---------------------------------------------------------------------------- #
# selection helpers
# ---------------------------------------------------------------------------- #
#: Names recognised by :func:`make_extractor` (lower-cased).
EXTRACTOR_NAMES = ("rake", "tfidf", "textrank")

_DEFAULT_NAME = "rake"


def make_extractor(
    name: str,
    *,
    tokenizer: Tokenizer | None = None,
    stopwords: frozenset[str] | None = None,
    min_cohesion: float | None = DEFAULT_MIN_COHESION,
) -> KeywordExtractor:
    """Build a :class:`KeywordExtractor` by short name.

    Recognized names (case-insensitive): ``"rake"`` (default), ``"tfidf"``,
    ``"textrank"``. Unknown names raise :class:`ValueError` so config typos fail
    fast -- matching the project's ``extra="forbid"`` stance.

    ``min_cohesion`` controls the CJK bigram cohesion filter
    (:data:`~sparksage.tags.cohesion.DEFAULT_MIN_COHESION` by default; ``None``
    disables it).
    """
    key = (name or "").strip().lower()
    if key in ("rake",):
        return RakeKeywordExtractor(
            tokenizer=tokenizer, stopwords=stopwords, min_cohesion=min_cohesion
        )
    if key in ("tfidf", "tf-idf"):
        return TfidfKeywordExtractor(
            tokenizer=tokenizer, stopwords=stopwords, min_cohesion=min_cohesion
        )
    if key in ("textrank", "text-rank"):
        return TextRankKeywordExtractor(
            tokenizer=tokenizer, stopwords=stopwords, min_cohesion=min_cohesion
        )
    raise ValueError(
        f"unknown extractor {name!r}; choose one of {EXTRACTOR_NAMES}"
    )


def default_extractor() -> KeywordExtractor:
    """Return the conventional default extractor (:class:`RakeKeywordExtractor`)."""
    return RakeKeywordExtractor()


__all__ = [
    "EXTRACTOR_NAMES",
    "KeywordExtractor",
    "KeywordScore",
    "RakeKeywordExtractor",
    "TextRankKeywordExtractor",
    "TfidfKeywordExtractor",
    "default_extractor",
    "make_extractor",
]
