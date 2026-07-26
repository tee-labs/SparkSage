"""Demo: query-time intent recognition + rewriting.

Runs fully offline with a :class:`FakeLLMClient` (no API key needed). A
rule-based classifier intercepts out-of-domain queries without an LLM call;
in-domain queries are rewritten by the (scripted) LLM rewriter, including a
multi-turn follow-up whose anaphora ("那联通呢") is resolved against history.

To use a real model, swap in :class:`OpenAICompatibleClient`:

    pip install 'sparksage[llm]'

    from sparksage import OpenAICompatibleClient
    client = OpenAICompatibleClient(api_key=..., model="gpt-4o-mini")
    proc = QueryProcessor(
        classifier=LLMIntentClassifier(client),
        rewriter=LLMQueryRewriter(client),
    )

Run with:  PYTHONPATH=src python3 examples/process_query.py
"""

from __future__ import annotations

import json

from sparksage import FakeLLMClient, QueryIntent
from sparksage.query import (
    ConversationContext,
    KeywordIntentRule,
    LLMQueryRewriter,
    QueryProcessor,
    RuleIntentClassifier,
)

# The rewriter asks the model for JSON; the fake replays these in order.
REWRITE_RESPONSES = [
    json.dumps({"rewritten_query": "China Mobile net profit 2024"}),
    json.dumps({"rewritten_query": "China Unicom net profit 2024"}),
]


def build_processor() -> QueryProcessor:
    classifier = RuleIntentClassifier([
        KeywordIntentRule(("天气", "weather", "笑话"), QueryIntent.OUT_OF_DOMAIN),
    ])
    rewriter = LLMQueryRewriter(FakeLLMClient(responses=REWRITE_RESPONSES))
    return QueryProcessor(classifier=classifier, rewriter=rewriter)


def report(label: str, query: str, result) -> None:
    print(f"\n=== {label} ===")
    print(f"query:     {query!r}")
    print(f"intent:    {result.intent.intent.value}  (conf={result.intent.confidence:.2f})")
    print(f"accepted:  {result.accepted}")
    if result.accepted:
        print(f"rewrite:   {result.rewrite.rewritten_query!r}")
    else:
        print(f"reply:     {result.default_reply!r}")


def main() -> None:
    proc = build_processor()

    # 1) out-of-domain -> intercepted, no LLM call
    report("rejected", "今天天气怎么样", proc.process("今天天气怎么样"))

    # 2) in-domain -> rewritten
    first = proc.process("中国移动2024年净利润怎么样")
    report("first turn", "中国移动2024年净利润怎么样", first)

    # 3) multi-turn follow-up: anaphora resolved against the conversation history
    ctx = ConversationContext.from_pairs([
        ("user", "中国移动2024年净利润怎么样"),
        ("assistant", first.rewrite.rewritten_query),
    ])
    followup = proc.process("那联通呢", context=ctx)
    report("follow-up", "那联通呢", followup)


if __name__ == "__main__":
    main()
