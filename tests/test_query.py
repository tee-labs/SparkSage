"""Tests for the query-time pipeline: intent recognition + rewriting.

All tests run offline via :class:`FakeLLMClient`, so no network or API key is
required. They exercise the real prompt-building, JSON-extraction, enum-coercion
and interception logic end-to-end.
"""

from __future__ import annotations

import json

import pytest

from sparksage.generator import FakeLLMClient
from sparksage.query import (
    CoercionError,
    ConversationContext,
    IntentEmptyResponseError,
    IntentResponseParseError,
    IntentResult,
    KeywordIntentRule,
    LLMIntentClassifier,
    LLMQueryRewriter,
    QueryProcessor,
    RewriteEmptyResponseError,
    RewriteResponseParseError,
    RewriteResult,
    RuleIntentClassifier,
    RuleQueryRewriter,
)
from sparksage.query.prompts import (
    intent_messages,
    intent_system_prompt,
    rewrite_messages,
)
from sparksage.query.rewriter import CallableRewriteRule, RegexRewriteRule
from sparksage.query.schema import (
    DEFAULT_INTENT,
    RawIntent,
    RawRewrite,
    coerce_intent,
    coerce_rewrite,
    extract_json,
    parse_intent_response,
    parse_raw_intent,
    parse_raw_rewrite,
    parse_rewrite_response,
)
from sparksage.schema.enums import QueryIntent


# --------------------------------------------------------------------------- #
# Fixtures / canned LLM payloads
# --------------------------------------------------------------------------- #
def _intent_json(intent: str, confidence: float = 0.9, reasoning: str = "r") -> str:
    return json.dumps(
        {"reasoning": reasoning, "intent": intent, "confidence": confidence}
    )


def _rewrite_json(
    rewritten: str = "中国移动 2024年净利润",
    *,
    sub_queries: list[str] | None = None,
    companies: list[str] | None = None,
    years: list[str] | None = None,
    reasoning: str = "resolved anaphora",
) -> str:
    return json.dumps(
        {
            "reasoning": reasoning,
            "rewritten_query": rewritten,
            "sub_queries": sub_queries or [],
            "extracted_companies": companies or [],
            "extracted_years": years or [],
        }
    )


# --------------------------------------------------------------------------- #
# ConversationContext
# --------------------------------------------------------------------------- #
class TestConversationContext:
    def test_empty_context(self):
        ctx = ConversationContext()
        assert ctx.is_empty()
        assert ctx.as_text() == ""

    def test_with_turn_chains_immutable(self):
        ctx = ConversationContext().with_turn("user", "hi")
        ctx2 = ctx.with_turn("assistant", "hello")
        assert len(ctx.turns) == 1
        assert len(ctx2.turns) == 2
        assert ctx.is_empty() is False

    def test_from_pairs(self):
        ctx = ConversationContext.from_pairs(
            [("user", "q1"), ("assistant", "a1"), ("user", "q2")]
        )
        text = ctx.as_text()
        assert text == "user: q1\nassistant: a1\nuser: q2"

    def test_invalid_role(self):
        with pytest.raises(ValueError):
            ConversationContext().with_turn("system", "x")

    def test_max_turns_keeps_recent(self):
        ctx = ConversationContext.from_pairs(
            [("user", f"q{i}") for i in range(5)]
        )
        assert ctx.as_text(max_turns=2) == "user: q3\nuser: q4"


# --------------------------------------------------------------------------- #
# Schema: JSON extraction + parsing + coercion
# --------------------------------------------------------------------------- #
class TestExtraction:
    def test_plain_json(self):
        out = extract_json('{"a": 1}')
        assert json.loads(out) == {"a": 1}

    def test_fenced_json(self):
        out = extract_json('```json\n{"a": 1}\n```')
        assert json.loads(out) == {"a": 1}

    def test_json_in_prose(self):
        out = extract_json('Here you go: {"intent": "x", "confidence": 1} done')
        assert json.loads(out)["intent"] == "x"

    def test_empty_raises(self):
        with pytest.raises(CoercionError):
            extract_json("   ")

    def test_garbage_raises(self):
        with pytest.raises(CoercionError):
            extract_json("not json at all")


class TestCoercion:
    def test_coerce_intent_maps_known(self):
        raw = RawIntent(intent="FINANCIAL_DATA", confidence="0.8")
        result = coerce_intent(raw, strict=False)
        assert result.intent is QueryIntent.FINANCIAL_DATA
        assert result.confidence == pytest.approx(0.8)

    def test_coerce_intent_case_insensitive_value(self):
        raw = RawIntent(intent="Out_Of_Domain", confidence=0.1)
        assert coerce_intent(raw, strict=False).intent is QueryIntent.OUT_OF_DOMAIN

    def test_coerce_intent_unknown_falls_back(self):
        raw = RawIntent(intent="bogus", confidence=0.5)
        result = coerce_intent(raw, strict=False)
        assert result.intent is DEFAULT_INTENT

    def test_coerce_intent_unknown_strict_raises(self):
        raw = RawIntent(intent="bogus", confidence=0.5)
        with pytest.raises(CoercionError):
            coerce_intent(raw, strict=True)

    @pytest.mark.parametrize("value", [-0.5, 1.5])
    def test_confidence_clamped(self, value: float):
        raw = RawIntent(intent="trend", confidence=value)
        result = coerce_intent(raw, strict=False)
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_bad_value_defaults(self):
        raw = RawIntent(intent="trend", confidence=float("nan"))
        result = coerce_intent(raw, strict=False)
        assert 0.0 <= result.confidence <= 1.0

    def test_coerce_rewrite_falls_back_to_original(self):
        raw = RawRewrite(rewritten_query="   ")
        result = coerce_rewrite(raw, "original", strict=False)
        assert result.rewritten_query == "original"

    def test_coerce_rewrite_empty_strict_raises(self):
        raw = RawRewrite(rewritten_query="")
        with pytest.raises(CoercionError):
            coerce_rewrite(raw, "original", strict=True)

    def test_coerce_rewrite_dedup_lists(self):
        raw = RawRewrite(
            rewritten_query="x",
            sub_queries=["a", " a ", "b", ""],
            extracted_companies=["移动", "移动"],
            extracted_years=["2024", "2024", "2023"],
        )
        result = coerce_rewrite(raw, "x", strict=False)
        assert result.sub_queries == ["a", "b"]
        assert result.extracted_companies == ["移动"]
        assert result.extracted_years == ["2024", "2023"]


class TestParsing:
    def test_parse_intent_response(self):
        raw = parse_intent_response(_intent_json("comparison", 0.77))
        assert raw.intent == "comparison"
        assert raw.confidence == pytest.approx(0.77)

    def test_parse_rewrite_response(self):
        raw = parse_rewrite_response(
            _rewrite_json(companies=["中国移动"], years=["2024"])
        )
        assert raw.rewritten_query == "中国移动 2024年净利润"
        assert raw.extracted_companies == ["中国移动"]

    def test_parse_raw_intent_rejects_non_dict(self):
        with pytest.raises(CoercionError):
            parse_raw_intent([1, 2, 3])

    def test_parse_raw_rewrite_rejects_non_dict(self):
        with pytest.raises(CoercionError):
            parse_raw_rewrite("nope")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
class TestPrompts:
    def test_intent_prompt_contains_all_vocab(self):
        prompt = intent_system_prompt()
        for member in QueryIntent:
            assert member.value in prompt

    def test_intent_messages_shape(self):
        msgs = intent_messages("how much revenue?")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "how much revenue?" in msgs[1]["content"]

    def test_rewrite_prompt_has_no_overexpand_constraint(self):
        ctx = ConversationContext.from_pairs([("user", "old")])
        msgs = rewrite_messages("a unique current query", context=ctx)
        system = msgs[0]["content"]
        user = msgs[1]["content"]
        assert "Do NOT over-expand" in system
        assert "中芯国际集成电路制造有限公司" in system  # the concrete anti-example
        # context is baked into the system prompt; the query into the user prompt
        assert "user: old" in system
        assert "a unique current query" in user
        assert "a unique current query" not in system

    def test_rewrite_prompt_no_context_marker(self):
        msgs = rewrite_messages("hello", context=None)
        assert "(no conversation context)" in msgs[0]["content"]


# --------------------------------------------------------------------------- #
# IntentClassifier
# --------------------------------------------------------------------------- #
class TestLLMIntentClassifier:
    def test_classifies_known_intent(self):
        clf = LLMIntentClassifier(FakeLLMClient(responses=[_intent_json("trend")]))
        result = clf.classify("未来三年趋势如何")
        assert result.intent is QueryIntent.TREND
        assert isinstance(result, IntentResult)

    def test_classify_unknown_intent_falls_back(self):
        clf = LLMIntentClassifier(
            FakeLLMClient(responses=[_intent_json("nonsense")])
        )
        result = clf.classify("q")
        assert result.intent is DEFAULT_INTENT

    def test_classify_unknown_intent_strict_raises(self):
        clf = LLMIntentClassifier(
            FakeLLMClient(responses=[_intent_json("nonsense")]), strict=True
        )
        with pytest.raises(IntentResponseParseError):
            clf.classify("q")

    def test_empty_response_raises(self):
        clf = LLMIntentClassifier(FakeLLMClient(responses=[""]))
        with pytest.raises(IntentEmptyResponseError):
            clf.classify("q")

    def test_malformed_json_raises(self):
        clf = LLMIntentClassifier(FakeLLMClient(responses=["not json"]))
        with pytest.raises(IntentResponseParseError):
            clf.classify("q")

    def test_empty_query_is_out_of_domain(self):
        clf = LLMIntentClassifier(FakeLLMClient(responses=[]))
        result = clf.classify("   ")
        assert result.intent is QueryIntent.OUT_OF_DOMAIN
        assert result.confidence == 1.0


class TestRuleIntentClassifier:
    def test_keyword_match(self):
        clf = RuleIntentClassifier().add_keyword(
            ["营收", "revenue", "profit"], QueryIntent.FINANCIAL_DATA
        )
        assert clf.classify("去年营收").intent is QueryIntent.FINANCIAL_DATA

    def test_no_match_uses_default(self):
        clf = RuleIntentClassifier(default=QueryIntent.BUSINESS_ANALYSIS)
        result = clf.classify("anything")
        assert result.intent is QueryIntent.BUSINESS_ANALYSIS
        assert result.reasoning  # default carries a reason

    def test_keyword_rule_directly(self):
        rule = KeywordIntentRule(["天气", "weather"], QueryIntent.OUT_OF_DOMAIN)
        assert rule.match("今天天气怎么样") is not None
        assert rule.match("营收多少") is None


# --------------------------------------------------------------------------- #
# QueryRewriter
# --------------------------------------------------------------------------- #
class TestLLMQueryRewriter:
    def test_rewrite_happy_path(self):
        rw = LLMQueryRewriter(
            FakeLLMClient(responses=[_rewrite_json(companies=["中国移动"])]),
        )
        result = rw.rewrite("那净利润呢")
        assert result.rewritten_query == "中国移动 2024年净利润"
        assert result.extracted_companies == ["中国移动"]
        assert isinstance(result, RewriteResult)

    def test_rewrite_passes_context_into_messages(self):
        fake = FakeLLMClient(responses=[_rewrite_json()])
        ctx = ConversationContext.from_pairs(
            [("user", "中国移动 2024年营收")]
        )
        rw = LLMQueryRewriter(fake)
        rw.rewrite("那净利润呢", context=ctx)
        sent = fake.last_messages
        assert sent is not None
        assert "中国移动 2024年营收" in sent[0]["content"]

    def test_rewrite_empty_falls_back_to_original(self):
        rw = LLMQueryRewriter(
            FakeLLMClient(responses=[_rewrite_json(rewritten="   ")]),
        )
        result = rw.rewrite("original query")
        assert result.rewritten_query == "original query"

    def test_rewrite_empty_strict_raises(self):
        rw = LLMQueryRewriter(
            FakeLLMClient(responses=[_rewrite_json(rewritten="")]), strict=True
        )
        with pytest.raises(RewriteResponseParseError):
            rw.rewrite("original query")

    def test_rewrite_empty_response_raises(self):
        rw = LLMQueryRewriter(FakeLLMClient(responses=[""]))
        with pytest.raises(RewriteEmptyResponseError):
            rw.rewrite("q")

    def test_rewrite_malformed_json_raises(self):
        rw = LLMQueryRewriter(FakeLLMClient(responses=["oops"]))
        with pytest.raises(RewriteResponseParseError):
            rw.rewrite("q")

    def test_rewrite_empty_query_identity(self):
        rw = LLMQueryRewriter(FakeLLMClient(responses=[]))
        result = rw.rewrite("")
        assert result.rewritten_query == ""


class TestRuleQueryRewriter:
    def test_regex_rule_applies(self):
        rw = RuleQueryRewriter().add_regex(r"foo", "bar")
        result = rw.rewrite("foo baz")
        assert result.rewritten_query == "bar baz"

    def test_regex_no_match_returns_identity(self):
        rw = RuleQueryRewriter().add_regex(r"zzz", "q")
        result = rw.rewrite("foo baz")
        assert result.rewritten_query == "foo baz"

    def test_regex_rule_directly_no_match_is_none(self):
        rule = RegexRewriteRule("xyz", "abc")
        assert rule.rewrite("no match here") is None

    def test_callable_rule_one_arg(self):
        rw = RuleQueryRewriter()
        rw.add(CallableRewriteRule(lambda q: q.upper()))
        assert rw.rewrite("hi").rewritten_query == "HI"

    def test_callable_rule_two_args_uses_context(self):
        def fill(query: str, ctx):
            if ctx and not ctx.is_empty():
                return f"{query} [with context]"
            return None

        rw = RuleQueryRewriter().add(CallableRewriteRule(fill))
        assert rw.rewrite("hi").rewritten_query == "hi"  # no context -> identity
        ctx = ConversationContext.from_pairs([("user", "x")])
        assert rw.rewrite("hi", context=ctx).rewritten_query == "hi [with context]"

    def test_callable_rule_three_args(self):
        def with_intent(query, ctx, intent):
            return RewriteResult(rewritten_query=f"{query}|{intent.intent.value}")

        rw = RuleQueryRewriter().add(CallableRewriteRule(with_intent))
        intent = IntentResult(intent=QueryIntent.TREND, confidence=0.9)
        assert rw.rewrite("q", intent=intent).rewritten_query == "q|trend"

    def test_callable_returning_none_skips(self):
        rw = RuleQueryRewriter().add(CallableRewriteRule(lambda q: None))
        assert rw.rewrite("hi").rewritten_query == "hi"

    def test_first_rule_wins(self):
        rw = RuleQueryRewriter()
        rw.add_regex("a", "1").add_regex("a", "2")
        assert rw.rewrite("a b").rewritten_query == "1 b"


# --------------------------------------------------------------------------- #
# QueryProcessor orchestration
# --------------------------------------------------------------------------- #
class TestQueryProcessor:
    @staticmethod
    def _fake_intent(intent: str) -> FakeLLMClient:
        return FakeLLMClient(responses=[_intent_json(intent)])

    def test_accepted_path_classifies_and_rewrites(self):
        # intent call then rewrite call -> two responses
        fake = FakeLLMClient(
            responses=[
                _intent_json("financial_data", 0.95),
                _rewrite_json("中国移动 2024年净利润"),
            ]
        )
        proc = QueryProcessor(
            classifier=LLMIntentClassifier(fake),
            rewriter=LLMQueryRewriter(fake),
        )
        result = proc.process("利润怎么样")
        assert result.accepted is True
        assert result.default_reply is None
        assert result.intent.intent is QueryIntent.FINANCIAL_DATA
        assert result.rewrite.rewritten_query == "中国移动 2024年净利润"
        assert result.original_query == "利润怎么样"
        assert len(fake.calls) == 2  # exactly two LLM calls

    def test_out_of_domain_is_intercepted_skips_rewrite(self):
        fake = FakeLLMClient(
            responses=[
                _intent_json("out_of_domain", 0.99),
                _rewrite_json("should not be used"),
            ]
        )
        proc = QueryProcessor(
            classifier=LLMIntentClassifier(fake),
            rewriter=LLMQueryRewriter(fake),
        )
        result = proc.process("今天天气怎么样")
        assert result.accepted is False
        assert result.default_reply  # canned reply present
        assert result.rewrite.rewritten_query == "今天天气怎么样"
        assert len(fake.calls) == 1  # rewrite NOT called

    def test_low_confidence_is_rejected(self):
        fake = FakeLLMClient(responses=[_intent_json("trend", 0.2)])
        proc = QueryProcessor(
            classifier=LLMIntentClassifier(fake),
            rewriter=LLMQueryRewriter(FakeLLMClient(responses=[])),
            min_confidence=0.4,
        )
        result = proc.process("hmm")
        assert result.accepted is False
        assert result.default_reply is not None

    def test_low_confidence_allowed_when_flag_set(self):
        fake = FakeLLMClient(
            responses=[
                _intent_json("trend", 0.2),
                _rewrite_json("rewritten"),
            ]
        )
        proc = QueryProcessor(
            classifier=LLMIntentClassifier(fake),
            rewriter=LLMQueryRewriter(fake),
            min_confidence=0.4,
            rewrite_on_low_confidence=True,
        )
        result = proc.process("hmm")
        assert result.accepted is True

    def test_custom_default_reply(self):
        fake = FakeLLMClient(responses=[_intent_json("out_of_domain")])
        proc = QueryProcessor(
            classifier=LLMIntentClassifier(fake),
            rewriter=LLMQueryRewriter(FakeLLMClient(responses=[])),
            default_reply="sorry!",
        )
        result = proc.process("off topic")
        assert result.default_reply == "sorry!"

    def test_custom_rejected_intents(self):
        fake = FakeLLMClient(responses=[_intent_json("comparison")])
        proc = QueryProcessor(
            classifier=LLMIntentClassifier(fake),
            rewriter=LLMQueryRewriter(FakeLLMClient(responses=[])),
            rejected_intents=frozenset({QueryIntent.COMPARISON}),
        )
        result = proc.process("A vs B")
        assert result.accepted is False

    def test_add_rejected_intent_fluent(self):
        fake = FakeLLMClient(responses=[_intent_json("trend")])
        proc = QueryProcessor(
            classifier=LLMIntentClassifier(fake),
            rewriter=LLMQueryRewriter(FakeLLMClient(responses=[])),
        )
        proc.add_rejected_intent(QueryIntent.TREND)
        assert proc.process("growth").accepted is False

    def test_rule_based_processor_never_calls_llm(self):
        clf = RuleIntentClassifier().add_keyword(
            ["营收"], QueryIntent.FINANCIAL_DATA, confidence=0.9
        )
        rw = RuleQueryRewriter().add_regex("营收", "营业收入")
        proc = QueryProcessor(classifier=clf, rewriter=rw)
        result = proc.process("去年营收")
        assert result.accepted is True
        assert result.rewrite.rewritten_query == "去年营业收入"

    def test_invalid_min_confidence_raises(self):
        with pytest.raises(ValueError):
            QueryProcessor(
                classifier=RuleIntentClassifier(),
                rewriter=RuleQueryRewriter(),
                min_confidence=1.5,
            )

    def test_non_protocol_objects_rejected(self):
        with pytest.raises(TypeError):
            QueryProcessor(classifier="not a classifier", rewriter=RuleQueryRewriter())  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            QueryProcessor(classifier=RuleIntentClassifier(), rewriter="nope")  # type: ignore[arg-type]
