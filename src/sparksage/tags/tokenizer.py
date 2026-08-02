"""Tokenizers for keyword extraction.

A :class:`Tokenizer` turns raw text into a flat list of normalized tokens. The
extractors in :mod:`sparksage.tags.extractor` depend *only* on this protocol and
on the stop-word sets in :mod:`sparksage.tags.stoplist`, so the whole tag-extraction
core is pure stdlib and unit-testable offline -- exactly like the rest of
SparkSage.

Three implementations cover the practical language matrix:

* :class:`WhitespaceTokenizer` -- Latin / Cyrillic / Greek alphabets: lower-case
  then split on non-letter/digit runs. No segmentation library needed.
* :class:`CharBigramTokenizer` -- CJK without segmentation: emit overlapping
  character bigrams (``知识管理`` -> ``["知识", "识管", "管理"]``). Bigrams carry
  far more topical signal than single characters and need *no* dictionary, so
  keyword extraction works on Chinese / Japanese / Korean out of the box.
* :class:`JiebaTokenizer` -- word-level Mandarin segmentation via ``jieba``
  (optional dependency under the ``[tags-zh]`` extra), imported lazily inside
  ``__init__`` like every other optional SDK in SparkSage.

:class:`AutoTokenizer` inspects the text and picks the right one: CJK content
goes to the bigram tokenizer (or ``jieba`` when explicitly requested), Latin
content to the whitespace tokenizer. It is the default tokenizer of every
extractor.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

#: Unicode ranges covering CJK unified ideographs (common across zh/ja/ko/繁/简).
_CJK_RANGES = (
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # Extension B
    (0x2A700, 0x2B73F),  # Extension C
    (0x2B740, 0x2B81F),  # Extension D
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    (0xAC00, 0xD7AF),    # Hangul Syllables
)

_CJK_CHAR_RE = re.compile(
    "[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _CJK_RANGES) + "]"
)

#: Latin / digit token: a maximal run of ASCII letters / digits.
_LATIN_TOKEN_RE = re.compile(r"[0-9a-z]+")


def _latin_tokens(lowered: str) -> list[str]:
    """Tokenize a lower-cased Latin run, flattening apostrophes first.

    ``don't`` -> ``dont``, ``it's`` -> ``its``: keeps contractions / possessives
    as one stable token so minor punctuation differences do not split keywords.
    """
    return _LATIN_TOKEN_RE.findall(lowered.replace("'", ""))


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_CHAR_RE.search(text))


def is_cjk_char(ch: str) -> bool:
    """Return ``True`` when ``ch`` is a single CJK / kana / Hangul character."""
    return len(ch) == 1 and bool(_CJK_CHAR_RE.fullmatch(ch))


@runtime_checkable
class Tokenizer(Protocol):
    """Turn raw text into a flat list of lower-cased tokens."""

    def tokenize(self, text: str) -> list[str]:
        """Return the ordered list of tokens extracted from ``text``."""
        ...


class WhitespaceTokenizer:
    """Lower-case + split on non letter/digit boundaries.

    Good for Latin-alphabet languages (English, French, German, Spanish, ...).
    Apostrophes are flattened so ``don't`` -> ``dont`` and ``it's`` -> ``its``,
    keeping tokens stable against minor punctuation differences.

    Examples
    --------
    >>> WhitespaceTokenizer().tokenize("SparkSage turns docs into Markdown!")
    ['sparksage', 'turns', 'docs', 'into', 'markdown']
    """

    def tokenize(self, text: str) -> list[str]:
        return _latin_tokens(text.lower())


class CharBigramTokenizer:
    """CJK tokenizer that needs no segmentation library.

    Latin / digit runs are kept whole (lower-cased). For each maximal run of CJK
    characters the overlapping character bigrams are emitted -- bigrams are a
    strong, dictionary-free approximation of Chinese words and work well for
    frequency / graph based keyword scoring. A long CJK run yields
    ``len(run) - 1`` bigrams; runs of a single CJK character fall back to that
    one character so it is not silently dropped.

    Examples
    --------
    >>> CharBigramTokenizer().tokenize("企业知识管理平台")
    ['企业', '业知', '知识', '识管', '管理', '理平', '平台']
    """

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for run in self._runs(text):
            if _CJK_CHAR_RE.match(run[0]):
                if len(run) == 1:
                    tokens.append(run)
                else:
                    tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
            else:
                tokens.extend(_latin_tokens(run.lower()))
        return tokens

    @staticmethod
    def _runs(text: str) -> list[str]:
        """Split ``text`` into maximal CJK / non-CJK runs."""
        runs: list[str] = []
        if not text:
            return runs
        current = text[0]
        current_is_cjk = bool(_CJK_CHAR_RE.match(current))
        for ch in text[1:]:
            ch_is_cjk = bool(_CJK_CHAR_RE.match(ch))
            if ch_is_cjk == current_is_cjk:
                current += ch
            else:
                runs.append(current)
                current = ch
                current_is_cjk = ch_is_cjk
        runs.append(current)
        # Keep only runs that contain letters/digits/CJK; drop pure whitespace/punct
        return [r for r in runs if any(c.isalnum() or _CJK_CHAR_RE.match(c) for c in r)]


class JiebaTokenizer:
    """Word-level Mandarin tokenizer backed by ``jieba``.

    ``jieba`` is an *optional* dependency -- install it with
    ``pip install 'sparksage[tags-zh]'``. It is imported lazily inside
    :meth:`__init__`, mirroring how every optional SDK is handled in SparkSage,
    so the tag-extraction core stays zero-dependency.

    Parameters
    ----------
    hmm:
        Enable HMM new-word discovery (default ``True``). Forwarded to
        :func:`jieba.cut`.
    """

    def __init__(self, *, hmm: bool = True) -> None:
        try:
            import jieba  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "JiebaTokenizer requires the 'jieba' package. "
                "Install it with: pip install 'sparksage[tags-zh]'"
            ) from exc
        self._jieba = jieba
        self._hmm = hmm

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for raw in self._jieba.cut(text, HMM=self._hmm):
            tok = raw.strip().lower()
            if tok and any(c.isalnum() or _CJK_CHAR_RE.match(c) for c in tok):
                out.append(tok)
        return out


class AutoTokenizer:
    """Pick a tokenizer from the text itself.

    CJK content routes to :class:`CharBigramTokenizer` (dependency-free). Pass
    ``use_jieba=True`` to prefer :class:`JiebaTokenizer` when the ``[tags-zh]``
    extra is installed -- the bigram tokenizer remains the fallback if ``jieba``
    is unavailable, so behaviour is always defined.

    This is the default tokenizer of every
    :class:`~sparksage.tags.extractor.KeywordExtractor`.
    """

    def __init__(self, *, use_jieba: bool = False) -> None:
        self._whitespace = WhitespaceTokenizer()
        self._bigram = CharBigramTokenizer()
        self._jieba: JiebaTokenizer | None = None
        if use_jieba:
            try:
                self._jieba = JiebaTokenizer()
            except ImportError:  # pragma: no cover - soft fallback
                self._jieba = None

    def tokenize(self, text: str) -> list[str]:
        if _contains_cjk(text):
            if self._jieba is not None:
                return self._jieba.tokenize(text)
            return self._bigram.tokenize(text)
        return self._whitespace.tokenize(text)


def default_tokenizer(*, use_jieba: bool = False) -> Tokenizer:
    """Return a fresh :class:`AutoTokenizer` (the conventional default)."""
    return AutoTokenizer(use_jieba=use_jieba)


__all__ = [
    "AutoTokenizer",
    "CharBigramTokenizer",
    "JiebaTokenizer",
    "Tokenizer",
    "WhitespaceTokenizer",
    "default_tokenizer",
    "is_cjk_char",
]
