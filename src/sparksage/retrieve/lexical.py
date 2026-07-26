"""Lexical (BM25) retrieval over IdeaBlocks (pure stdlib).

This is the sparse-retrieval half of hybrid search. IdeaBlock's ``keywords``
field is documented as *"for BM25 / lexical recall boosting"* but, until now,
nothing consumed it. :class:`BM25Retriever` finally does: each block becomes a
BM25 "document" whose token bag is the ``keywords`` (weighted, since they are
curated retrieval terms) plus the ``trusted_answer`` / ``critical_question`` /
``name`` text. A keyword-only query then gets strong lexical recall even when
the dense embedding is misled by paraphrase.

The implementation is classic Robertson-Spärck-Jones BM25 with ``k1`` / ``b``
defaults (``1.5`` / ``0.75``), pure Python -- no ``rank_bm25`` dependency, so
the retrieval core stays unit-testable offline like the rest of SparkSage.

Tokenization is deliberately simple and multilingual-aware: runs of ASCII
letters/digits become word tokens, and CJK characters are emitted as both
unigrams and overlapping bigrams (the same dictionary-free trick the
:class:`~sparksage.tags.CharBigramTokenizer` uses), so Mandarin / Japanese /
Korean content retrieves lexically without a segmentation dependency.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sparksage.embed.store import SearchHit
from sparksage.schema.ideablock import IdeaBlock

#: BM25 term-frequency saturation parameter (typical default ``1.5``).
DEFAULT_K1 = 1.5
#: BM25 length-normalization parameter (typical default ``0.75``).
DEFAULT_B = 0.75

#: Multiplier applied to curated ``keywords`` tokens so they outweigh prose
#: tokens in the BM25 document bag (they exist *to* be matched).
KEYWORD_WEIGHT = 3

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CJK_RE = re.compile(
    "[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff"
    "\uac00-\ud7af]"
)


def tokenize(text: str) -> list[str]:
    """Tokenize ``text`` for BM25 indexing.

    ASCII word-runs become lowercased word tokens. Every CJK character becomes
    a unigram token, and every adjacent CJK pair becomes a bigram token -- the
    dictionary-free scheme that gives Mandarin / Japanese strong lexical
    signal without a segmenter. Non-word, non-CJK characters are separators.
    """
    if not text:
        return []
    tokens: list[str] = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    chars = [c for c in text if _CJK_RE.match(c)]
    tokens.extend(chars)
    for i in range(len(chars) - 1):
        tokens.append(chars[i] + chars[i + 1])
    return tokens


def _block_tokens(block: IdeaBlock) -> list[str]:
    """Token bag for one block, with ``keywords`` weighted up."""
    base = tokenize(block.trusted_answer)
    base.extend(tokenize(block.critical_question))
    base.extend(tokenize(block.name))
    for kw in block.keywords:
        base.extend([t for t in tokenize(kw) for _ in range(KEYWORD_WEIGHT)])
    return base


@runtime_checkable
class LexicalRetriever(Protocol):
    """Sparse-retrieval counterpart to the dense :class:`VectorStore`.

    Implementations index a corpus of blocks and return BM25-style ranked
    :class:`~sparksage.embed.store.SearchHit` lists for a query. The score is
    *not* comparable to a cosine -- it is a raw BM25 score; the retrieval
    orchestrator combines the two via rank fusion (RRF) rather than score
    arithmetic, precisely because the scales differ.
    """

    def index(self, blocks: list[IdeaBlock]) -> None:
        """(Re)build the lexical index over ``blocks``."""
        ...

    def search(self, query: str, k: int = 10) -> list[SearchHit]:
        """Return the top-``k`` lexical hits, best first."""
        ...

    def __len__(self) -> int:
        ...


class BM25Retriever:
    """Dependency-free BM25 retriever over :class:`IdeaBlock` corpora.

    Token bags weight curated ``keywords`` higher than prose so the field the
    schema designates for lexical boosting actually moves the needle. Replaces
    any previously indexed corpus on :meth:`index`.

    Parameters
    ----------
    k1, b:
        Standard BM25 parameters. Defaults (``1.5`` / ``0.75``) are the
        literature-standard starting point.

    Examples
    --------
    >>> from sparksage.retrieve import BM25Retriever
    >>> lex = BM25Retriever()
    >>> lex.index(blocks)              # doctest: +SKIP
    >>> hits = lex.search("deploy spark", k=3)   # doctest: +SKIP
    """

    def __init__(self, *, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        if k1 < 0.0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self._k1 = float(k1)
        self._b = float(b)
        self._ids: list[str] = []
        self._docs: list[dict[str, int]] = []
        self._lengths: list[int] = []
        self._avgdl: float = 0.0
        self._df: dict[str, int] = {}
        self._n: int = 0

    @property
    def k1(self) -> float:
        return self._k1

    @property
    def b(self) -> float:
        return self._b

    def index(self, blocks: list[IdeaBlock]) -> None:
        """(Re)build the BM25 index, replacing any prior corpus."""
        self._ids = []
        self._docs = []
        self._lengths = []
        self._df = {}
        for block in blocks:
            tokens = _block_tokens(block)
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            bid = str(block.id)
            self._ids.append(bid)
            self._docs.append(tf)
            self._lengths.append(len(tokens))
            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1
        self._n = len(self._ids)
        total = sum(self._lengths)
        self._avgdl = (total / self._n) if self._n else 0.0

    def __len__(self) -> int:
        return self._n

    def __contains__(self, block_id: object) -> bool:
        return str(block_id) in self._ids

    def search(self, query: str, k: int = 10) -> list[SearchHit]:
        """Return the top-``k`` BM25 hits for ``query``, best first.

        Fewer than ``k`` (or ``[]``) when the index holds fewer blocks.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if self._n == 0:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []
        q_tf: dict[str, int] = {}
        for term in q_terms:
            q_tf[term] = q_tf.get(term, 0) + 1

        scored: list[tuple[str, float]] = []
        for i, bid in enumerate(self._ids):
            score = self._score_one(i, q_tf)
            if score > 0.0:
                scored.append((bid, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [SearchHit(block_id=bid, score=s) for bid, s in scored[:k]]

    def _score_one(self, i: int, q_tf: dict[str, int]) -> float:
        doc = self._docs[i]
        dl = self._lengths[i]
        denom_dl = self._avgdl if self._avgdl > 0 else 1.0
        score = 0.0
        for term, _qf in q_tf.items():
            f = doc.get(term)
            if not f:
                continue
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
            tf_norm = (f * (self._k1 + 1.0)) / (
                f + self._k1 * (1.0 - self._b + self._b * dl / denom_dl)
            )
            score += idf * tf_norm
        return score


@dataclass
class NullLexicalRetriever:
    """A no-op lexical retriever that indexes nothing and never hits.

    Lets the hybrid orchestrator treat "lexical disabled" uniformly as a
    :class:`LexicalRetriever` rather than branching on ``None``.
    """

    _n: int = field(default=0, repr=False, compare=False)

    def index(self, blocks: list[IdeaBlock]) -> None:
        self._n = len(blocks)

    def search(self, query: str, k: int = 10) -> list[SearchHit]:
        return []

    def __len__(self) -> int:
        return 0


__all__ = [
    "BM25Retriever",
    "DEFAULT_B",
    "DEFAULT_K1",
    "KEYWORD_WEIGHT",
    "LexicalRetriever",
    "NullLexicalRetriever",
    "tokenize",
]
