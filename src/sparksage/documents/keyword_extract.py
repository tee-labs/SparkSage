"""Keyword extraction for auto-tagging documents.

The knowledge-management requirement is explicit: *when a document ships
without tags, the system must auto-generate them from content using a keyword
extraction algorithm*. That is a deterministic, content-driven step -- not an
LLM call -- so it works offline, reproducibly, and without an API key.

This module provides:

* a small :class:`KeywordExtractor` :class:`~typing.Protocol` so any algorithm
  (TextRank, YAKE, jieba, ...) can be plugged in;
* :class:`FrequencyKeywordExtractor`, a solid pure-stdlib default that ranks
  terms by **position-weighted term frequency**: tokens appearing in Markdown
  headings (especially the title) outrank body tokens, stop words are filtered,
  and both Latin words/phrases and CJK unigrams/bigrams are scored. Sub-terms
  already covered by a higher-ranked phrase are dropped to avoid redundancy.

The extractor is pure Python with no third-party dependency, matching the rest
of the framework-agnostic core.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Protocol, runtime_checkable

from sparksage.documents.schema import TAGS_MAX

#: CJK Unified Ideographs + Extension A + Compatibility ideographs.
_CJK_RANGE = r"\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF"

_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
_CJK_RUN_RE = re.compile(f"[{_CJK_RANGE}]+")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<text>\S.*?)\s*$")
_FRONTMATTER_FENCE_RE = re.compile(r"^\s{0,3}---\s*$")
_CODE_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_DIGIT_RE = re.compile(r"^\d+$")

#: Conservative stop-word list for English + common Chinese function chars.
#: Kept small on purpose: domain jargon (the stuff that makes good tags) is
#: rarely in any stop list, so over-pruning hurts precision more than it helps.
DEFAULT_STOP_WORDS: frozenset[str] = frozenset(
    {
        # English articles / conjunctions / pronouns / auxiliaries / common verbs
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "for",
        "of", "to", "in", "on", "at", "by", "with", "from", "as", "is", "are",
        "was", "were", "be", "been", "being", "this", "that", "these", "those",
        "it", "its", "they", "them", "their", "we", "you", "your", "our", "us",
        "i", "he", "she", "his", "her", "can", "will", "would", "should",
        "could", "may", "might", "must", "shall", "do", "does", "did", "has",
        "have", "had", "not", "no", "so", "than", "too", "very", "just", "also",
        "about", "into", "over", "under", "more", "most", "such", "each",
        "which", "who", "whom", "what", "when", "where", "why", "how", "all",
        "any", "both", "few", "some", "there", "here", "out", "up", "down",
        "off", "via", "per", "etc", "ie", "eg", "using", "use", "used",
        # Chinese function characters (single-char stop set)
        "的", "了", "和", "是", "在", "与", "及", "或", "也", "都", "而",
        "为", "对", "由", "从", "到", "向", "把", "被", "让", "使", "给", "跟",
        "这", "那", "其", "之", "所", "以", "于", "可", "能", "会", "将",
        "已", "正", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
        "个", "们", "中", "上", "下", "里", "外", "前", "后", "并", "但", "因",
        "如", "若", "则", "即", "该", "此", "它", "他", "她", "我", "你",
    }
)


@runtime_checkable
class KeywordExtractor(Protocol):
    """Turn free text into a ranked list of tag strings.

    Implementations must be pure (no I/O, no side effects) so they are fully
    deterministic and unit-testable offline.
    """

    def extract(self, text: str, *, top_n: int = 5) -> list[str]:
        """Return up to ``top_n`` keywords, most salient first."""
        ...


def _line_terms(
    text: str, heading_weight: int
) -> tuple[Counter[str, int], list[tuple[str, str, str]]]:
    """Score candidate terms and collect (left, phrase, right) bigram parts.

    Returns a :class:`~collections.Counter` mapping each candidate term to its
    position-weighted score, plus a list of bigram component pairs used later to
    prune redundant unigrams.
    """
    scores: Counter[str, int] = Counter()
    bigram_parts: list[tuple[str, str, str]] = []

    in_frontmatter = False
    in_code = False
    fence_marker: str | None = None

    for i, raw in enumerate(text.split("\n")):
        if i == 0 and _FRONTMATTER_FENCE_RE.match(raw):
            in_frontmatter = True
            continue
        if in_frontmatter:
            if _FRONTMATTER_FENCE_RE.match(raw):
                in_frontmatter = False
            continue

        fence = _CODE_FENCE_RE.match(raw)
        if fence:
            marker = fence.group(1)[0]
            if in_code and fence_marker == marker:
                in_code = False
                fence_marker = None
            elif not in_code:
                in_code = True
                fence_marker = marker
            continue
        if in_code:
            continue

        m = _HEADING_RE.match(raw)
        in_heading = m is not None
        weight = heading_weight if in_heading else 1
        content = m.group("text") if m else raw

        latin = [w.lower() for w in _LATIN_WORD_RE.findall(content)]
        for w in latin:
            scores[w] += weight
        for a, b in zip(latin, latin[1:], strict=False):
            phrase = f"{a} {b}"
            scores[phrase] += weight
            bigram_parts.append((a, phrase, b))

        for run in _CJK_RUN_RE.findall(content):
            chars = list(run)
            for c in chars:
                scores[c] += weight
            for a, b in zip(chars, chars[1:], strict=False):
                phrase = a + b
                scores[phrase] += weight
                bigram_parts.append((a, phrase, b))

    return scores, bigram_parts


def _is_noise(term: str, stop_words: frozenset[str], min_len: int) -> bool:
    """True if ``term`` should be filtered out before ranking."""
    if len(term) < min_len:
        return True
    if _DIGIT_RE.match(term):
        return True
    if " " in term:
        parts = term.split(" ")
        if any(p in stop_words for p in parts):
            return True
        return False
    return term in stop_words


def _covered_unigrams(
    ranked: list[tuple[str, int]], bigram_parts: list[tuple[str, str, str]]
) -> set[str]:
    """Unigrams to drop because a higher-ranked bigram already contains them."""
    covered: set[str] = set()
    rank_of = {term: i for i, (term, _) in enumerate(ranked)}
    for left, phrase, right in bigram_parts:
        if phrase not in rank_of:
            continue
        p_rank = rank_of[phrase]
        for part in (left, right):
            if part in rank_of and rank_of[part] > p_rank:
                covered.add(part)
    return covered


class FrequencyKeywordExtractor:
    """Position-weighted term-frequency keyword extractor (pure stdlib).

    Parameters
    ----------
    stop_words:
        Terms excluded from the candidate set. Defaults to
        :data:`DEFAULT_STOP_WORDS` (English + Chinese function words).
    heading_weight:
        Score multiplier for tokens found inside Markdown headings. Headings
        (especially the H1 title) are strong signals of what a document is
        *about*, so they are weighted higher than body prose. Default ``4``.
    min_term_length:
        Minimum character length for a candidate term. Default ``2`` (drops
        single Latin letters; CJK single chars survive length-wise but are
        pruned via stop words / bigram coverage).
    default_top_n:
        Default value of ``top_n`` when :meth:`extract` is called without it.

    Notes
    -----
    The algorithm is intentionally transparent and reproducible:

    1. Tokenize each non-code, non-frontmatter line into Latin words and CJK
       runs; form adjacent bigrams of both.
    2. Score every candidate by summing a per-occurrence weight (``1`` in body,
       ``heading_weight`` in headings).
    3. Drop stop words, pure digits, and too-short terms.
    4. Drop a unigram when a higher-ranked bigram containing it already won a
       slot (removes the redundant "learning" next to "machine learning").
    5. Return the ``top_n`` highest-scoring terms, tie-broken by length then
       lexicographic order for deterministic output.
    """

    def __init__(
        self,
        *,
        stop_words: frozenset[str] | None = None,
        heading_weight: int = 4,
        min_term_length: int = 2,
        default_top_n: int = 5,
    ) -> None:
        if heading_weight < 1:
            raise ValueError("heading_weight must be >= 1")
        if min_term_length < 1:
            raise ValueError("min_term_length must be >= 1")
        if default_top_n < 1:
            raise ValueError("default_top_n must be >= 1")
        self._stop_words = stop_words if stop_words is not None else DEFAULT_STOP_WORDS
        self._heading_weight = heading_weight
        self._min_term_length = min_term_length
        self._default_top_n = default_top_n

    @property
    def stop_words(self) -> frozenset[str]:
        return self._stop_words

    def extract(self, text: str, *, top_n: int | None = None) -> list[str]:
        """Return up to ``top_n`` keywords for ``text``, most salient first."""
        if text is None or not str(text).strip():
            return []
        n = top_n if top_n is not None else self._default_top_n
        if n <= 0:
            return []
        n = min(n, TAGS_MAX)

        scores, bigram_parts = _line_terms(text, self._heading_weight)

        kept = [
            (term, score)
            for term, score in scores.items()
            if not _is_noise(term, self._stop_words, self._min_term_length)
        ]
        if not kept:
            return []

        kept.sort(key=lambda item: (-item[1], -len(item[0]), item[0]))
        covered = _covered_unigrams(kept, bigram_parts)

        result: list[str] = []
        for term, _score in kept:
            if term in covered:
                continue
            result.append(term)
            if len(result) >= n:
                break
        return result
