"""Tests for the keyword-extraction (tag) core.

Everything runs fully offline and dependency-free. The CJK segmentation path via
:class:`JiebaTokenizer` is exercised behind ``importorskip`` so the suite stays
green without the ``[tags-zh]`` extra installed.
"""

from __future__ import annotations

import pytest

from sparksage import (
    DEFAULT_MIN_COHESION,
    DEFAULT_STOPWORDS,
    ENGLISH_STOPWORDS,
    EXTRACTOR_NAMES,
    AutoTokenizer,
    CharBigramTokenizer,
    KeywordExtractor,
    KeywordScore,
    RakeKeywordExtractor,
    TextRankKeywordExtractor,
    TfidfKeywordExtractor,
    WhitespaceTokenizer,
    blessed_cjk_bigrams,
    cjk_bigram_gate,
    default_extractor,
    is_cjk_char,
    is_stopword,
    make_extractor,
)
from sparksage.tags.tokenizer import JiebaTokenizer

# ---------------------------------------------------------------------------- #
# sample documents
# ---------------------------------------------------------------------------- #
ENGLISH_TEXT = (
    "SparkSage turns documents into question-aligned knowledge chunks for "
    "high-quality RAG. The schema layer is the foundational part of the "
    "library. Chunks improve retrieval and ranking. Retrieval quality matters "
    "for RAG applications. Document chunking is the core feature of the system."
)

CJK_TEXT = (
    "企业知识管理平台提供统一的文档管理能力。系统支持 Markdown 文档上传与解析，"
    "自动提取标题、摘要、正文及标签。知识管理提升企业文档分类的规范性。"
    "标签管理提高检索效率。文档管理与知识检索共同构成企业知识库的核心。"
)


# ---------------------------------------------------------------------------- #
# tokenizers
# ---------------------------------------------------------------------------- #
class TestWhitespaceTokenizer:
    def test_lowercases_and_splits(self):
        toks = WhitespaceTokenizer().tokenize("SparkSage turns docs into Markdown!")
        assert toks == ["sparksage", "turns", "docs", "into", "markdown"]

    def test_handles_apostrophes(self):
        toks = WhitespaceTokenizer().tokenize("it's don't we're")
        assert toks == ["its", "dont", "were"]

    def test_empty(self):
        assert WhitespaceTokenizer().tokenize("") == []

    def test_digits_kept(self):
        assert WhitespaceTokenizer().tokenize("Q3 2024 revenue") == [
            "q3",
            "2024",
            "revenue",
        ]


class TestCharBigramTokenizer:
    def test_bigrams_for_cjk(self):
        toks = CharBigramTokenizer().tokenize("企业知识")
        assert toks == ["企业", "业知", "知识"]

    def test_keeps_latin_runs(self):
        toks = CharBigramTokenizer().tokenize("使用 SparkSage 进行检索")
        assert "sparksage" in toks
        assert "检索" in toks or "行检" in toks

    def test_single_cjk_char_falls_back_to_char(self):
        assert CharBigramTokenizer().tokenize("龙") == ["龙"]

    def test_punctuation_dropped(self):
        toks = CharBigramTokenizer().tokenize("知识，管理。")
        for t in toks:
            assert t.isalnum() or any(ord(c) >= 0x3000 for c in t)


class TestAutoTokenizer:
    def test_routes_cjk_to_bigram(self):
        toks = AutoTokenizer().tokenize("知识管理")
        assert "知识" in toks and "管理" in toks

    def test_routes_latin_to_whitespace(self):
        toks = AutoTokenizer().tokenize("Hello World")
        assert toks == ["hello", "world"]

    def test_use_jieba_falls_back_when_unavailable(self):
        tok = AutoTokenizer(use_jieba=True)
        # If jieba is absent, must still return bigrams (never raise at tokenize).
        out = tok.tokenize("知识管理")
        assert "知识" in out


class TestJiebaTokenizer:
    def test_import_guard(self, monkeypatch):
        pytest.importorskip("jieba")
        tok = JiebaTokenizer()
        out = tok.tokenize("企业知识管理平台")
        assert isinstance(out, list)
        assert len(out) > 0


# ---------------------------------------------------------------------------- #
# stoplist
# ---------------------------------------------------------------------------- #
class TestStoplist:
    def test_english_stopwords_present(self):
        assert "the" in ENGLISH_STOPWORDS
        assert "is" in ENGLISH_STOPWORDS

    def test_is_stopword_default(self):
        assert is_stopword("the") is True
        assert is_stopword("sparksage") is False

    def test_is_stopword_custom_set(self):
        custom = frozenset({"alpha", "beta"})
        assert is_stopword("alpha", custom) is True
        assert is_stopword("gamma", custom) is False

    def test_default_stopwords_union(self):
        assert DEFAULT_STOPWORDS >= ENGLISH_STOPWORDS
        assert "的" in DEFAULT_STOPWORDS


# ---------------------------------------------------------------------------- #
# KeywordScore + protocol compliance
# ---------------------------------------------------------------------------- #
class TestProtocolAndTypes:
    def test_keywordscore_is_frozen(self):
        ks = KeywordScore(keyword="rag", score=1.5)
        with pytest.raises(AttributeError):
            ks.score = 2.0  # type: ignore[misc]

    def test_each_extractor_implements_protocol(self):
        for ex in (
            RakeKeywordExtractor(),
            TfidfKeywordExtractor(),
            TextRankKeywordExtractor(),
        ):
            assert isinstance(ex, KeywordExtractor)

    def test_extract_returns_keywordscore(self):
        out = RakeKeywordExtractor().extract(ENGLISH_TEXT, top_k=3)
        assert all(isinstance(k, KeywordScore) for k in out)


# ---------------------------------------------------------------------------- #
# shared behaviour
# ---------------------------------------------------------------------------- #
class TestCommonBehaviour:
    @pytest.mark.parametrize(
        "extractor",
        [RakeKeywordExtractor(), TfidfKeywordExtractor(), TextRankKeywordExtractor()],
    )
    def test_empty_text_returns_empty(self, extractor):
        assert extractor.extract("", top_k=5) == []

    @pytest.mark.parametrize(
        "extractor",
        [RakeKeywordExtractor(), TfidfKeywordExtractor(), TextRankKeywordExtractor()],
    )
    def test_top_k_cap(self, extractor):
        out = extractor.extract(ENGLISH_TEXT, top_k=3)
        assert len(out) <= 3

    @pytest.mark.parametrize(
        "extractor",
        [RakeKeywordExtractor(), TfidfKeywordExtractor(), TextRankKeywordExtractor()],
    )
    def test_top_k_validation(self, extractor):
        with pytest.raises(ValueError, match="top_k"):
            extractor.extract(ENGLISH_TEXT, top_k=0)

    @pytest.mark.parametrize(
        "extractor",
        [RakeKeywordExtractor(), TfidfKeywordExtractor(), TextRankKeywordExtractor()],
    )
    def test_deterministic(self, extractor):
        a = extractor.extract(ENGLISH_TEXT, top_k=5)
        b = extractor.extract(ENGLISH_TEXT, top_k=5)
        assert [k.keyword for k in a] == [k.keyword for k in b]

    @pytest.mark.parametrize(
        "extractor",
        [RakeKeywordExtractor(), TfidfKeywordExtractor(), TextRankKeywordExtractor()],
    )
    def test_cjk_works_dependency_free(self, extractor):
        out = extractor.extract(CJK_TEXT, top_k=5)
        assert len(out) >= 1
        for k in out:
            assert k.keyword


# ---------------------------------------------------------------------------- #
# RAKE specifics
# ---------------------------------------------------------------------------- #
class TestRake:
    def test_surfaces_topical_terms(self):
        out = [k.keyword for k in RakeKeywordExtractor().extract(ENGLISH_TEXT, top_k=5)]
        joined = " ".join(out)
        assert "retrieval" in joined or "chunks" in joined or "sparksage" in joined

    def test_phrases_respect_max_len(self):
        ex = RakeKeywordExtractor(max_phrase_len=2)
        out = ex.extract(ENGLISH_TEXT, top_k=8)
        for k in out:
            assert len(k.keyword.split()) <= 2

    def test_user_stopwords_drive_splitting(self):
        ex = RakeKeywordExtractor(stopwords=frozenset({"chunks"}))
        out = [k.keyword for k in ex.extract(ENGLISH_TEXT, top_k=5)]
        # "chunks" is now a delimiter -> it never appears inside any phrase
        for phrase in out:
            assert "chunks" not in phrase.split()

    def test_scores_descending(self):
        out = RakeKeywordExtractor().extract(ENGLISH_TEXT, top_k=5)
        scores = [k.score for k in out]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------- #
# TF-IDF specifics
# ---------------------------------------------------------------------------- #
class TestTfidf:
    def test_returns_single_tokens(self):
        out = TfidfKeywordExtractor().extract(ENGLISH_TEXT, top_k=5)
        for k in out:
            assert " " not in k.keyword

    def test_scores_descending(self):
        out = TfidfKeywordExtractor().extract(ENGLISH_TEXT, top_k=5)
        scores = [k.score for k in out]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------- #
# TextRank specifics
# ---------------------------------------------------------------------------- #
class TestTextRank:
    def test_merged_phrase_capped(self):
        out = TextRankKeywordExtractor().extract(ENGLISH_TEXT, top_k=8)
        for k in out:
            assert len(k.keyword.split()) <= 4

    def test_scores_descending(self):
        out = TextRankKeywordExtractor().extract(ENGLISH_TEXT, top_k=5)
        scores = [k.score for k in out]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------- #
# make_extractor / defaults
# ---------------------------------------------------------------------------- #
class TestMakeExtractor:
    @pytest.mark.parametrize("name", list(EXTRACTOR_NAMES) + ["tf-idf", "text-rank"])
    def test_known_names(self, name):
        ex = make_extractor(name)
        assert isinstance(ex, KeywordExtractor)
        assert len(ex.extract(ENGLISH_TEXT, top_k=3)) >= 1

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown extractor"):
            make_extractor("nonsense")

    def test_default_extractor_is_rake(self):
        assert isinstance(default_extractor(), RakeKeywordExtractor)


# ---------------------------------------------------------------------------- #
# is_cjk_char
# ---------------------------------------------------------------------------- #
class TestIsCjkChar:
    def test_cjk_chars_are_cjk(self):
        assert is_cjk_char("知")
        assert is_cjk_char("あ")  # Hiragana
        assert is_cjk_char("ア")  # Katakana
        assert is_cjk_char("가")  # Hangul

    def test_non_cjk_are_not_cjk(self):
        assert not is_cjk_char("a")
        assert not is_cjk_char("0")
        assert not is_cjk_char(" ")
        assert not is_cjk_char("，")

    def test_rejects_multichar(self):
        assert not is_cjk_char("知识")


# ---------------------------------------------------------------------------- #
# CJK bigram cohesion filter
# ---------------------------------------------------------------------------- #
# Reproduces the reported scatter: a doc whose tags came out as
# "凌晨、晨一、一点、点执、执行" -- the cross-boundary bigrams (晨一 / 点执) are
# pure noise from the dictionary-free CharBigramTokenizer.
SCATTER_DOC = (
    "凌晨一点执行数据库备份任务。该任务占用较多资源。"
    "备份在凌晨一点开始，执行时间约两小时。为减少占用，建议调整执行时间。"
)


class TestCohesion:
    def test_default_threshold_in_range(self):
        assert 0.0 < DEFAULT_MIN_COHESION < 1.0

    def test_empty_and_non_cjk_return_empty(self):
        assert blessed_cjk_bigrams("") == frozenset()
        assert blessed_cjk_bigrams("english only, no cjk here") == frozenset()

    def test_drops_cross_boundary_keeps_real_words(self):
        # "知识管理" repeated verbatim -> 知识 / 管理 blessed, 识管 (cross-boundary) dropped.
        text = "知识管理很重要。知识管理提升效率。"
        blessed = blessed_cjk_bigrams(text)
        assert "知识" in blessed
        assert "管理" in blessed
        assert "识管" not in blessed

    def test_reported_scatter_noise_eliminated(self):
        blessed = blessed_cjk_bigrams(SCATTER_DOC)
        # cross-boundary accidents never survive
        assert "晨一" not in blessed
        assert "点执" not in blessed
        # the real words do
        assert "凌晨" in blessed
        assert "一点" in blessed
        assert "执行" in blessed
        assert "占用" in blessed

    def test_gate_disabled_returns_none(self):
        assert cjk_bigram_gate(SCATTER_DOC, min_cohesion=None) is None

    def test_high_threshold_filters_more(self):
        # a very high floor drops even repeated real words whose chars are promiscuous
        text = "占用资源。使用资源。作用资源。"
        loose = blessed_cjk_bigrams(text, min_cohesion=0.0)
        strict = blessed_cjk_bigrams(text, min_cohesion=0.99)
        assert loose  # something survives with no floor
        assert len(strict) <= len(loose)


class TestExtractorCohesionIntegration:
    @pytest.mark.parametrize(
        "extractor",
        [RakeKeywordExtractor(), TfidfKeywordExtractor(), TextRankKeywordExtractor()],
    )
    def test_no_cross_boundary_bigram_tags(self, extractor):
        out = [k.keyword for k in extractor.extract(SCATTER_DOC, top_k=12)]
        assert out, "extractor should still produce tags"
        joined = " ".join(out)
        assert "晨一" not in joined
        assert "点执" not in joined
        # and the real words still surface somewhere in the keyword set
        assert "执行" in joined

    @pytest.mark.parametrize(
        "extractor",
        [RakeKeywordExtractor(), TfidfKeywordExtractor(), TextRankKeywordExtractor()],
    )
    def test_min_cohesion_none_disables_filter(self, extractor):
        # With the filter off, cross-boundary bigrams CAN appear again (old behaviour).
        disabled = extractor.__class__(min_cohesion=None)
        # Re-derive a fresh instance with the filter on for contrast.
        enabled = extractor.__class__()
        on_out = " ".join(k.keyword for k in enabled.extract(SCATTER_DOC, top_k=12))
        off_out = " ".join(k.keyword for k in disabled.extract(SCATTER_DOC, top_k=12))
        assert "晨一" not in on_out
        # The disabled run must be noisier: it surfaces at least one bigram
        # that the filter removes. Find any blessed bigram's overlapping neighbour.
        # (We assert the filter actually changed the token stream, not a fixed token,
        #  to keep this robust against DP tie-breaks.)
        from sparksage.tags.cohesion import blessed_cjk_bigrams

        blessed = blessed_cjk_bigrams(SCATTER_DOC)
        # collect every 2-char CJK token the disabled extractor emits that the
        # filter would drop; there must be at least one (that is the whole bug).
        def _cjk2(s):
            return len(s) == 2 and is_cjk_char(s[0]) and is_cjk_char(s[1])

        dropped = {
            t
            for kw in off_out.split()
            for t in kw.split()
            if _cjk2(t) and t not in blessed
        }
        assert dropped, "with min_cohesion=None the noisy bigrams must reappear"

    @pytest.mark.parametrize(
        "extractor",
        [RakeKeywordExtractor(), TfidfKeywordExtractor(), TextRankKeywordExtractor()],
    )
    def test_english_unaffected_by_filter(self, extractor):
        out = [k.keyword for k in extractor.extract(ENGLISH_TEXT, top_k=5)]
        joined = " ".join(out)
        assert "retrieval" in joined or "chunks" in joined or "sparksage" in joined

    def test_make_extractor_forwards_min_cohesion(self):
        ex = make_extractor("rake", min_cohesion=None)
        assert ex._min_cohesion is None
        ex2 = make_extractor("tfidf", min_cohesion=0.9)
        assert ex2._min_cohesion == 0.9
