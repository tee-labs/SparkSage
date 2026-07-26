"""Stop-word lists for keyword extraction.

Stop words are the function words (articles, conjunctions, pronouns, auxiliaries,
...) that carry little topical signal and would dominate any frequency-based
scorer if left in. The :mod:`sparksage.tags` extractors filter them out before
building candidate phrases / graphs.

The lists are pure data (no logic), kept small and curated rather than exhaustive,
so they are easy to audit. English is the default; a compact CJK set covers the
common Mandarin function words so the dependency-free
:class:`~sparksage.tags.tokenizer.CharBigramTokenizer` does not surface ``的 / 是
/ 了`` as "keywords". Domain vocabularies (legal / medical / ...) are deliberately
*not* included -- drop in a custom set via the ``stopwords=`` argument of any
:class:`~sparksage.tags.extractor.KeywordExtractor`.
"""

from __future__ import annotations

#: A compact, audited English stop-word set. Covers articles, pronouns,
#: auxiliary / modal verbs, common prepositions, conjunctions and quantifiers.
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an",
        "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
        "before", "being", "below", "between", "both", "but", "by", "can",
        "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
        "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
        "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
        "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
        "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
        "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
        "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
        "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
        "own",         "same", "shan't", "she", "she'd", "she'll", "she's", "should",
        "shouldn't", "so", "some", "such", "that", "that's", "the",
        "their", "theirs", "them", "themselves", "then", "there", "there's",
        "these", "they", "they'd", "they'll", "they're", "they've", "this",
        "those", "through", "to", "too", "under", "until", "up", "very", "was",
        "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
        "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't",
        "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
        "your", "yours", "yourself", "yourselves",
        "also", "just", "may", "might", "will", "shall", "now",
        "via", "per", "etc", "upon", "within", "without",
    }
)

#: Compact Mandarin stop-word / function-word set so character-bigram keyword
#: extraction over CJK text (without ``jieba``) is not flooded by ``的 / 是``.
#: Mixed with :data:`ENGLISH_STOPWORDS` in :data:`DEFAULT_STOPWORDS`.
CJK_STOPWORDS: frozenset[str] = frozenset(
    {
        "的", "了", "是", "在", "和", "与", "或", "也", "都", "就", "还", "又",
        "而", "但", "则", "及", "以", "于", "对", "为", "把", "被", "让", "使",
        "给", "由", "从", "到", "向", "上", "下", "中", "里", "外", "前", "后",
        "这", "那", "这个", "那个", "这些", "那些", "其", "之", "着", "过", "地",
        "得", "吗", "呢", "吧", "啊", "呀", "哦", "嗯", "你", "我", "他", "她",
        "它", "们", "你们", "我们", "他们", "一个", "一种", "一些", "可以",
        "已经", "应该", "可能", "如果", "因为", "所以", "但是", "虽然", "然而",
        "进行", "通过", "根据", "按照", "以及", "并且", "或者", "不", "没有",
        "不是", "非常", "很", "太", "更", "最", "只", "才", "会", "能", "要",
        "想", "做", "说", "看", "知道", "使用", "需要",
    }
)

#: Default combined stop-word set used by every extractor unless overridden.
#: Combine the Latin and CJK lists so mixed-language documents "just work".
DEFAULT_STOPWORDS: frozenset[str] = ENGLISH_STOPWORDS | CJK_STOPWORDS


def is_stopword(token: str, stopwords: frozenset[str] | None = None) -> bool:
    """Return whether ``token`` (already lower-cased for Latin) is a stop word.

    Falls back to :data:`DEFAULT_STOPWORDS` when ``stopwords`` is ``None``.
    """
    table = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    return token in table


__all__ = [
    "CJK_STOPWORDS",
    "DEFAULT_STOPWORDS",
    "ENGLISH_STOPWORDS",
    "is_stopword",
]
