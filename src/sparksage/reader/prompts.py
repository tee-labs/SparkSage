"""Prompt construction for answer generation and faithfulness judging.

IdeaBlock's question-aligned design (``critical_question`` + ``trusted_answer``)
makes a far better reader context than naive text shards: the model gets whole,
self-contained answers, so it stays focused and grounded instead of stitching
fragments together. The prompts below lean on that -- each candidate is
rendered with its id, question and trusted answer so the model can emit
grounded citations referencing the ids (which the strict layer then binds to
the schema's ``source.uri`` / ``source.locator`` provenance).
"""

from __future__ import annotations

from sparksage.retrieve.models import RetrievedChunk

_ANSWER_SYSTEM_TEMPLATE = """\
You are SparkSage's answer-generation module. Using ONLY the provided knowledge \
chunks, answer the user's question concisely and faithfully.

Rules:
- Ground every claim in the provided chunks. Do NOT use outside knowledge.
- If the chunks do not contain the answer, say you do not know.
- Cite the chunks you used: list their ids in "citations", each with the short \
verbatim quote from that chunk's trusted_answer that supports the claim.
- Keep the answer focused and brief (2-4 sentences unless the question needs more).
- Report a "confidence" in [0, 1] for how well the chunks answer the question.

Respond with ONLY a JSON object of the form:
{{"reasoning": "brief reasoning", "answer": "the answer text", "citations": \
[{{"block_id": "<id>", "quote": "<verbatim substring>"}}], "confidence": \
<float>}}
No markdown, no commentary -- just the JSON object.
"""

_FAITHFULNESS_SYSTEM_TEMPLATE = """\
You are a strict faithfulness judge. Given a generated answer and the knowledge \
chunks it was built from, decide how well the answer is SUPPORTED by those \
chunks (no outside knowledge allowed).

- score = 1.0 means every claim in the answer is directly supported by the chunks.
- score = 0.0 means the answer is unsupported / hallucinated.
- Count the supported vs unsupported claims for transparency.

Respond with ONLY a JSON object of the form:
{{"reasoning": "brief reasoning", "score": <float in [0, 1]>, \
"supported_claims": <int>, "unsupported_claims": <int>}}
No markdown, no commentary -- just the JSON object.
"""


def _render_context(chunks: list[RetrievedChunk]) -> tuple[str, dict[str, str]]:
    """Render candidate chunks as ``[id] question || answer`` and return id->id."""
    lines: list[str] = []
    id_map: dict[str, str] = {}
    for c in chunks:
        bid = str(c.block.id)
        id_map[bid] = bid
        lines.append(
            f"[{bid}] {c.block.critical_question} || {c.block.trusted_answer}"
        )
    return "\n".join(lines), id_map


def answer_system_prompt() -> str:
    return _ANSWER_SYSTEM_TEMPLATE


def answer_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context, _ = _render_context(chunks)
    return (
        f"Question: {query.strip()}\n\n"
        f"Knowledge chunks:\n{context}\n\n"
        "Answer the question using only these chunks."
    )


def answer_messages(
    query: str, chunks: list[RetrievedChunk]
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": answer_system_prompt()},
        {"role": "user", "content": answer_user_prompt(query, chunks)},
    ]


def faithfulness_system_prompt() -> str:
    return _FAITHFULNESS_SYSTEM_TEMPLATE


def faithfulness_user_prompt(
    query: str, answer_text: str, chunks: list[RetrievedChunk]
) -> str:
    context, _ = _render_context(chunks)
    return (
        f"Question: {query.strip()}\n\n"
        f"Generated answer:\n{answer_text.strip()}\n\n"
        f"Knowledge chunks:\n{context}\n\n"
        "Judge how faithfully the answer is supported by the chunks."
    )


def faithfulness_messages(
    query: str, answer_text: str, chunks: list[RetrievedChunk]
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": faithfulness_system_prompt()},
        {
            "role": "user",
            "content": faithfulness_user_prompt(query, answer_text, chunks),
        },
    ]


__all__ = [
    "answer_messages",
    "answer_system_prompt",
    "answer_user_prompt",
    "faithfulness_messages",
    "faithfulness_system_prompt",
    "faithfulness_user_prompt",
]
