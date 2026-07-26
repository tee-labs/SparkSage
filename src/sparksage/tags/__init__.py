"""Keyword / tag extraction for auto-tagging documents.

When a document arrives without tags, a :class:`KeywordExtractor` derives them
from the content using classic, dependency-free algorithms (RAKE / TF-IDF /
TextRank). The core depends only on the :class:`~sparksage.tags.tokenizer.Tokenizer`
protocol and the stop-word sets in :mod:`sparksage.tags.stoplist`, so it is pure
stdlib and unit-testable offline -- matching how every other SparkSage core is
built.

Mandarin / Japanese / Korean work out of the box via the
:class:`~sparksage.tags.tokenizer.CharBigramTokenizer` (no segmentation library
needed); word-level segmentation is available behind the optional
:class:`~sparksage.tags.tokenizer.JiebaTokenizer` (``pip install
'sparksage[tags-zh]'``).

The emitted :class:`KeywordScore` list is consumed by the document-management
layer (:mod:`sparksage.documents`) as free-form tags -- intentionally *not* the
closed :class:`~sparksage.schema.enums.Tag` enum, which keeps its coarse-grained
semantic-filtering role.
"""

from sparksage.tags.extractor import (
    EXTRACTOR_NAMES,
    KeywordExtractor,
    KeywordScore,
    RakeKeywordExtractor,
    TextRankKeywordExtractor,
    TfidfKeywordExtractor,
    default_extractor,
    make_extractor,
)
from sparksage.tags.stoplist import (
    CJK_STOPWORDS,
    DEFAULT_STOPWORDS,
    ENGLISH_STOPWORDS,
    is_stopword,
)
from sparksage.tags.tokenizer import (
    AutoTokenizer,
    CharBigramTokenizer,
    JiebaTokenizer,
    Tokenizer,
    WhitespaceTokenizer,
    default_tokenizer,
)

__all__ = [
    "CJK_STOPWORDS",
    "CharBigramTokenizer",
    "DEFAULT_STOPWORDS",
    "ENGLISH_STOPWORDS",
    "EXTRACTOR_NAMES",
    "AutoTokenizer",
    "JiebaTokenizer",
    "KeywordExtractor",
    "KeywordScore",
    "RakeKeywordExtractor",
    "TextRankKeywordExtractor",
    "TfidfKeywordExtractor",
    "Tokenizer",
    "WhitespaceTokenizer",
    "default_extractor",
    "default_tokenizer",
    "is_stopword",
    "make_extractor",
]
