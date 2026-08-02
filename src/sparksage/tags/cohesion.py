"""CJK bigram cohesion filtering for keyword extraction (pure stdlib).

The :class:`~sparksage.tags.tokenizer.CharBigramTokenizer` emits *every*
overlapping character bigram of a CJK run as a dictionary-free scoring signal --
but those bigrams are scoring features, not words. Cross-boundary bigrams
(``凌晨一点执行`` -> ``晨一`` / ``点执``) are statistical accidents and become
noise when surfaced as tags, which is the root cause of "scattered" auto-tags
on Chinese documents.

This module turns bigram scoring features back into tag-shaped words *without a
dictionary*, using two complementary signals:

* **Cohesion** -- ``f(ab) / max(f(a), f(b))`` (a bidirectional conditional
  probability): a real word's two characters stick together far more often than
  chance, while a cross-boundary pair's characters are "promiscuous" (``晨``
  mostly lives in ``凌晨`` / ``早晨``, so ``晨一`` is rare relative to that).
* **Non-overlap** -- within one CJK run overlapping bigrams compete; a
  maximum-weight non-overlapping cover (weighted by cohesion, selected via
  dynamic programming) keeps the densest real-word segmentation and drops the
  cross-boundary losers even when a phrase is repeated verbatim (the case pure
  cohesion cannot disambiguate on its own).

:func:`blessed_cjk_bigrams` returns the set of 2-char CJK strings that survive
both filters; the extractors pass it to their token filtering so low-quality
bigrams never reach :class:`~sparksage.tags.extractor.KeywordScore` (and never
participate in TF-IDF / TextRank scoring either). For word-perfect Mandarin
segmentation install ``jieba`` (``pip install 'sparksage[tags-zh]'``); this is
the dependency-free fallback that removes the scatter while staying
unit-testable offline like every other SparkSage core.
"""

from __future__ import annotations

from collections import Counter

from sparksage.tags.tokenizer import is_cjk_char

#: Default cohesion floor. A bigram is a *segmentation candidate* only when its
#: two characters co-occur in this configuration at least this fraction of the
#: time. ``0.34`` keeps real words that share a common character (e.g. ``占用``
#: where ``用`` also appears in ``使用`` / ``作用``) while dropping cross-boundary
#: accidents. Tunable on every extractor via ``min_cohesion=`` (``None``
#: disables the filter entirely).
DEFAULT_MIN_COHESION = 0.34


def _cjk_runs(text: str) -> list[str]:
    """Return the maximal runs of CJK characters in ``text``."""
    runs: list[str] = []
    current: list[str] = []
    for ch in text:
        if is_cjk_char(ch):
            current.append(ch)
        elif current:
            runs.append("".join(current))
            current = []
    if current:
        runs.append("".join(current))
    return runs


def _char_and_bigram_freqs(
    text: str,
) -> tuple[Counter[str], Counter[str]]:
    """Character and adjacent-bigram frequencies over the CJK runs of ``text``."""
    char_freq: Counter[str] = Counter()
    bigram_freq: Counter[str] = Counter()
    for run in _cjk_runs(text):
        for ch in run:
            char_freq[ch] += 1
        for i in range(len(run) - 1):
            bigram_freq[run[i : i + 2]] += 1
    return char_freq, bigram_freq


def _cohesion(
    bigram: str,
    char_freq: Counter[str],
    bigram_freq: Counter[str],
) -> float:
    """Bidirectional conditional probability of ``bigram`` sticking together.

    ``f(ab) / max(f(a), f(b))`` == ``min(P(b|a), P(a|b))``: of all the places
    ``a`` / ``b`` appear, how often are they found *together* as ``ab``. Robust
    on short documents (unlike PMI, which is undefined when every character is
    unique and identical across a fresh run).
    """
    a, b = bigram[0], bigram[1]
    denom = max(char_freq.get(a, 0), char_freq.get(b, 0))
    if denom <= 0:
        return 0.0
    return bigram_freq.get(bigram, 0) / denom


def _select_run_bigrams(run: str, weights: dict[int, float]) -> list[int]:
    """Max-weight non-overlapping bigram start-indices for one CJK ``run``.

    ``weights`` maps a start-index ``i`` to the cohesion of ``run[i:i+2]`` for
    the *eligible* bigrams (cohesion >= floor). Two bigrams overlap when their
    start-indices differ by less than 2; the DP selects a subset maximising the
    total cohesion, preferring to *take* a bigram on ties so the densest
    (word-like) cover wins -- which is exactly the segmentation a dictionary
    would produce for 2-char-word-dominated Chinese.
    """
    length = len(run)
    if length < 2 or not weights:
        return []
    best = [0.0] * (length + 1)
    take = [False] * (length + 1)
    for i in range(2, length + 1):
        skip = best[i - 1]
        weight = weights.get(i - 2)
        if weight is None or best[i - 2] + weight < skip:
            best[i] = skip
            take[i] = False
        else:
            best[i] = best[i - 2] + weight
            take[i] = True
    selected: list[int] = []
    i = length
    while i >= 2:
        if take[i]:
            selected.append(i - 2)
            i -= 2
        else:
            i -= 1
    selected.reverse()
    return selected


def blessed_cjk_bigrams(
    text: str,
    *,
    min_cohesion: float = DEFAULT_MIN_COHESION,
) -> frozenset[str]:
    """Return the set of 2-char CJK bigrams fit to be tags.

    Combines a cohesion floor (drop promiscuous pairs) with a per-run
    maximum-weight non-overlap selection (drop cross-boundary pairs that lose
    to the real-word segmentation). The result is the "blessed" set the
    extractors keep; every other CJK bigram is treated as a scoring-only
    feature and dropped before keywords are built.
    """
    char_freq, bigram_freq = _char_and_bigram_freqs(text)
    if not bigram_freq:
        return frozenset()
    blessed: set[str] = set()
    for run in _cjk_runs(text):
        if len(run) < 2:
            continue
        weights: dict[int, float] = {}
        for i in range(len(run) - 1):
            bigram = run[i : i + 2]
            score = _cohesion(bigram, char_freq, bigram_freq)
            if score >= min_cohesion:
                weights[i] = score
        for i in _select_run_bigrams(run, weights):
            blessed.add(run[i : i + 2])
    return frozenset(blessed)


def cjk_bigram_gate(
    text: str,
    *,
    min_cohesion: float | None = DEFAULT_MIN_COHESION,
) -> frozenset[str] | None:
    """Return the blessed bigram set, or ``None`` to disable filtering.

    ``min_cohesion=None`` disables the filter (returns ``None``); otherwise
    returns :func:`blessed_cjk_bigrams`. The extractors pass the result to
    their token-filtering helpers as ``cjk_allow``.
    """
    if min_cohesion is None:
        return None
    return blessed_cjk_bigrams(text, min_cohesion=min_cohesion)


__all__ = [
    "DEFAULT_MIN_COHESION",
    "blessed_cjk_bigrams",
    "cjk_bigram_gate",
]
