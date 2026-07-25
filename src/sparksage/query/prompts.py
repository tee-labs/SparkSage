"""Prompt construction for query-time intent recognition and rewriting.

The controlled vocabulary (:class:`QueryIntent`) is read straight from the enum
definition so the prompt can never drift from the code -- exactly as
:mod:`sparksage.generator.prompts` does for block generation. Add a new
``QueryIntent`` member and the model is automatically allowed to emit it.
"""

from __future__ import annotations

from sparksage.query.context import ConversationContext
from sparksage.schema.enums import QueryIntent

_INTENT_SYSTEM_TEMPLATE = """\
You are SparkSage's intent-recognition module. Classify the user's query into \
exactly ONE of these intents:
{intent_descriptions}

Respond with ONLY a JSON object of the form:
{{"reasoning": "your brief reasoning", "intent": "<one of: {intents}>", \
"confidence": <float between 0.0 and 1.0>}}
No markdown, no commentary -- just the JSON object.
"""

_INTENT_DESCRIPTIONS = {
    QueryIntent.FINANCIAL_DATA: (
        "- financial_data: queries about concrete numbers, metrics, revenue, "
        "profit, KPIs, or other measurable figures."
    ),
    QueryIntent.BUSINESS_ANALYSIS: (
        "- business_analysis: qualitative questions about products, strategy, "
        "operations, segments, or business performance."
    ),
    QueryIntent.COMPARISON: (
        "- comparison: questions contrasting two or more entities, periods, "
        "or metrics against each other."
    ),
    QueryIntent.TREND: (
        "- trend: questions about changes, growth, decline, or evolution over "
        "time."
    ),
    QueryIntent.OUT_OF_DOMAIN: (
        "- out_of_domain: questions unrelated to the knowledge domain the "
        "system serves."
    ),
}

_REWRITE_SYSTEM_TEMPLATE = """\
You are a professional query-rewriting expert for semantic retrieval.

Rewriting principles:
- Keep the query concise. Do NOT over-expand (e.g. never expand a company's \
short name into its full legal name -- "中芯国际" stays "中芯国际", not \
"中芯国际集成电路制造有限公司").
- Only fill in MISSING key information (implied company names, years, metric \
names). Do not add redundant decoration.
- The rewritten query should be more precise than the original but must NOT be \
longer than twice the original.
- For short follow-ups (e.g. "那联通呢" / "what about Unicom?" / "利润呢"), you \
MUST resolve anaphora and inherit company / year / metric from the conversation \
context, producing a complete, self-contained query.
- If the query is a compound question, decompose it into independent \
sub-queries in "sub_queries".

{context_section}

Reason step by step:
1. Identify the company name(s) mentioned; keep the short form. If none are \
explicit but the context implies one, fill it in.
2. Identify the year / time range; fill it in from context if implied.
3. Decide whether any implied information needs completing.
4. Decide whether the query is compound and split it into sub-questions.
5. Produce the final rewritten query -- explicit, complete, concise, and \
search-friendly.

Respond with ONLY a JSON object of the form:
{{"reasoning": "your reasoning", "rewritten_query": "the rewritten query", \
"sub_queries": ["sub-question 1", ...], "extracted_companies": ["..."], \
"extracted_years": ["..."]}}
No markdown, no commentary -- just the JSON object.
"""


def _enum_list(members: type) -> str:
    return ", ".join(m.value for m in members)


def _intent_descriptions() -> str:
    """Render the intent vocabulary (one bullet per member).

    Reads live from :class:`QueryIntent`, so an extended enum is picked up
    automatically -- built-in members get a description, any unknown member
    falls back to a generic ``- <value>:`` line so the prompt never breaks.
    """
    lines = []
    for member in QueryIntent:
        lines.append(
            _INTENT_DESCRIPTIONS.get(
                member, f"- {member.value}: {member.name.lower()}"
            )
        )
    return "\n".join(lines)


def _context_section(context: ConversationContext | None) -> str:
    """The conversation-history block injected into the rewrite prompt."""
    rendered = (context or ConversationContext()).as_text()
    if rendered:
        return (
            "The following is the conversation context:\n"
            f"{rendered}\n\n"
            "Rewrite the current query using this context, filling in implied "
            "company names, years, metrics, and resolving anaphora."
        )
    return "(no conversation context)"


def intent_system_prompt() -> str:
    """Build the intent-classification system prompt from the live vocabulary."""
    return _INTENT_SYSTEM_TEMPLATE.format(
        intent_descriptions=_intent_descriptions(),
        intents=_enum_list(QueryIntent),
    )


def intent_user_prompt(query: str) -> str:
    """Build the intent-classification user prompt carrying the raw query."""
    return f"Classify the intent of this query:\n\n{query.strip()}"


def intent_messages(
    query: str,
) -> list[dict[str, str]]:
    """Assemble the full chat message list for intent classification."""
    return [
        {"role": "system", "content": intent_system_prompt()},
        {"role": "user", "content": intent_user_prompt(query)},
    ]


def rewrite_system_prompt(context: ConversationContext | None = None) -> str:
    """Build the rewrite system prompt, injecting the conversation context."""
    return _REWRITE_SYSTEM_TEMPLATE.format(
        context_section=_context_section(context)
    )


def rewrite_user_prompt(query: str) -> str:
    """Build the rewrite user prompt carrying the raw query."""
    return f"Rewrite this query for semantic retrieval:\n\n{query.strip()}"


def rewrite_messages(
    query: str,
    *,
    context: ConversationContext | None = None,
) -> list[dict[str, str]]:
    """Assemble the full chat message list for query rewriting.

    The conversation context is baked into the system prompt (so it stays
    authoritative) rather than the user prompt.
    """
    return [
        {"role": "system", "content": rewrite_system_prompt(context)},
        {"role": "user", "content": rewrite_user_prompt(query)},
    ]
