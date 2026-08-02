"""Prompt construction for the agentic QA controller (ReAct-style).

The controller is the "brain" of the agentic loop: given the question, the
multi-turn context, the trajectory it has reasoned through so far, and the
evidence it has already gathered, it decides the next action -- retrieve another
sub-query (multi-hop / comparative decomposition) or synthesize a final answer.

IdeaBlock's question-aligned design pays off doubly here: each gathered chunk is
rendered back to the controller as ``[id] critical_question || trusted_answer``
(the same compact, self-contained rendering the reader / grader / reranker use),
so the controller can judge *whether it has enough* without re-reading raw
shards. The ``trusted_answer`` is truncated to keep the prompt token cost bounded
as evidence accumulates across iterations.
"""

from __future__ import annotations

from sparksage.agent.models import AgentState
from sparksage.retrieve.models import RetrievedChunk

#: Truncate each evidence chunk's ``trusted_answer`` in the prompt to bound cost.
DEFAULT_EVIDENCE_ANSWER_CHARS = 240

#: Cap how many evidence chunks are rendered back to the controller per turn.
DEFAULT_EVIDENCE_TOP_K = 8

_AGENT_SYSTEM_TEMPLATE = """\
You are SparkSage's agentic QA controller. Your job is to decide the NEXT single \
action that moves the question closer to a faithful, fully-supported answer.

You have three actions:
- "retrieve": run ONE more knowledge-base search for a sub-question you still \
need. Use this to fill a gap in the evidence, or to drill into one prong of a \
multi-hop / comparative question.
- "plan": decompose the question into a list of focused, self-contained \
sub-queries you want retrieved in sequence. Use this *once* at the start of a \
complex / comparative question ("compare A and B revenue and margins" -> \
["A revenue", "B revenue", "A margins", "B margins"]). The engine enqueues the \
sub-queries and retrieves each one through the same retrieve path; you will \
still be consulted between them and may "synthesize" early once you have enough.
- "synthesize": stop searching and let the answer module write the final answer \
from the evidence gathered so far.

Decision rules:
- Prefer "plan" up front for multi-hop / comparative questions; otherwise use \
"retrieve" for a single gap.
- Prefer "synthesize" as soon as the current evidence is sufficient to answer \
the question faithfully. Do NOT over-search.
- Never repeat a sub-question you already retrieved (listed under "Steps"). \
Reformulate or move on.
- Each "retrieve" must carry a focused, self-contained "query" (resolve any \
anaphora against the conversation / earlier steps). "k" is optional.
- When the knowledge base is partitioned by metadata (tags / entities / \
language / knowledge base), you MAY scope a "retrieve" or "plan" by setting \
"filter" to an object with any of: "tags" (list of coarse Tag values like \
"important" / "technology"), "entities" (list of entity strings), "languages" \
(list of language codes), "kb_id" (a knowledge-base id). Omit "filter" to \
inherit the call-level scope (the common case).
- If the evidence cannot answer the question and no new retrieval would help, \
still choose "synthesize" -- the answer module will abstain rather than \
hallucinate.

Respond with ONLY a JSON object of the form:
{{"thought": "one-sentence reasoning", "action": "retrieve" | "plan" | \
"synthesize", "query": "<sub-question when retrieve>", "sub_queries": \
["<list of sub-questions when plan>"], "k": <int or null>, "filter": \
{{"tags": [...], "entities": [...], "languages": [...], "kb_id": "..."} | null}}
No markdown, no commentary -- just the JSON object.
"""


def _render_evidence(
    chunks: list[RetrievedChunk],
    *,
    top_k: int,
    answer_chars: int,
) -> str:
    """Render gathered evidence as ``[id] question || (truncated) answer``."""
    if not chunks:
        return "(no evidence gathered yet)"
    lines: list[str] = []
    for c in chunks[:top_k]:
        bid = str(c.block.id)
        answer = c.block.trusted_answer
        if len(answer) > answer_chars:
            answer = answer[:answer_chars].rstrip() + "..."
        lines.append(f"[{bid}] {c.block.critical_question} || {answer}")
    if len(chunks) > top_k:
        lines.append(f"(...and {len(chunks) - top_k} more)")
    return "\n".join(lines)


def _render_trajectory(state: AgentState) -> str:
    """Render the steps already executed so the controller avoids repeating them."""
    if not state.steps:
        return "(none yet -- this is the first decision after the seed retrieval)"
    lines: list[str] = []
    for i, step in enumerate(state.steps, start=1):
        obs = step.observation if step.observation else "(no hits)"
        lines.append(
            f"{i}. thought: {step.thought}\n"
            f"   retrieved: {step.query} -> {step.retrieved_count} chunk(s)\n"
            f"   observation: {obs}"
        )
    return "\n".join(lines)


def agent_system_prompt() -> str:
    return _AGENT_SYSTEM_TEMPLATE


def agent_user_prompt(
    state: AgentState,
    *,
    evidence_top_k: int = DEFAULT_EVIDENCE_TOP_K,
    evidence_answer_chars: int = DEFAULT_EVIDENCE_ANSWER_CHARS,
) -> str:
    context_text = ""
    if state.context is not None:
        rendered = state.context.as_text(max_turns=4)
        if rendered:
            context_text = f"Conversation so far:\n{rendered}\n\n"
    evidence = _render_evidence(
        state.evidence,
        top_k=evidence_top_k,
        answer_chars=evidence_answer_chars,
    )
    return (
        f"Question: {state.question.strip()}\n\n"
        f"{context_text}"
        f"Steps already taken:\n{_render_trajectory(state)}\n\n"
        f"Current evidence ({len(state.evidence)} chunk(s)):\n{evidence}\n\n"
        "Decide the next action."
    )


def agent_messages(
    state: AgentState,
    *,
    evidence_top_k: int = DEFAULT_EVIDENCE_TOP_K,
    evidence_answer_chars: int = DEFAULT_EVIDENCE_ANSWER_CHARS,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": agent_system_prompt()},
        {
            "role": "user",
            "content": agent_user_prompt(
                state,
                evidence_top_k=evidence_top_k,
                evidence_answer_chars=evidence_answer_chars,
            ),
        },
    ]


__all__ = [
    "DEFAULT_EVIDENCE_ANSWER_CHARS",
    "DEFAULT_EVIDENCE_TOP_K",
    "agent_messages",
    "agent_system_prompt",
    "agent_user_prompt",
]
