"""Prompt construction for the Distill merge step.

The merge LLM is shown a cluster of near-duplicate IdeaBlocks and asked to fuse
them into ONE canonical block that is more complete and more concise than any
individual member -- reconciling wording, deduplicating entities/keywords, and
keeping the trusted answer within the brevity cap. The controlled vocabularies
(``Tag``, ``EntityType``) are read straight from the enum definitions so the
prompt can never drift from the code, exactly like the generator prompt.
"""

from __future__ import annotations

import textwrap

from sparksage.schema.enums import EntityType, Tag
from sparksage.schema.ideablock import RECOMMENDED_ANSWER_MAX, IdeaBlock

_SYSTEM_TEMPLATE = """\
You are SparkSage-Distill, an expert knowledge engineer that merges
near-duplicate IdeaBlocks into one canonical block.

You will be given a CLUSTER of {n} IdeaBlocks that a similarity search found to \
be near-duplicates (they answer essentially the same question in different \
wording, possibly with partial overlap). Fuse them into ONE canonical IdeaBlock \
that is:

- more COMPLETE than any member (reconcile complementary facts);
- more CONCISE than the members combined (drop pure repetition);
- faithful to the members -- do NOT invent facts none of them state.

The canonical block MUST have, as JSON:
- "name": a short title (<=200 chars).
- "critical_question": the single question this cluster answers, ending with "?".
- "trusted_answer": a verified, self-consistent answer of 2-3 sentences and NO
  MORE THAN {answer_max} characters. Compress aggressively; if two members
  contradict, prefer the more specific one and keep it short.
- "tags": 0..N tags drawn ONLY from this controlled vocabulary: {tags}.
  Union the members' tags, deduplicated.
- "entities": 0..N named things. Each is
  {{"entity_name": str, "entity_type": <one of: {entity_types}>, "aliases": \
[str]}}. Merge entities that refer to the same thing (fold aliases).
- "keywords": 0..N short keywords. Union and deduplicate the members' keywords.
- "reasoning": one sentence on how you reconciled the cluster (diagnostics only).

Rules:
1. Emit exactly ONE block. Never split the cluster into several.
2. "critical_question" MUST end with "?".
3. Keep "trusted_answer" <= {answer_max} chars -- compress, do not truncate.
4. Use only the tag / entity_type strings listed above.
5. Respond with ONLY the JSON object. No markdown, no commentary.
"""

_USER_HEADER = """\
Merge the following cluster of {n} near-duplicate IdeaBlocks into ONE canonical \
block. Members (name / question / answer / tags / keywords):
"""


def merge_system_prompt(n: int) -> str:
    """Build the merge system prompt for a cluster of ``n`` blocks."""
    return _SYSTEM_TEMPLATE.format(
        n=n,
        answer_max=RECOMMENDED_ANSWER_MAX,
        tags=", ".join(t.value for t in Tag),
        entity_types=", ".join(e.value for e in EntityType),
    )


def _render_block(block: IdeaBlock, index: int) -> str:
    tags = ", ".join(t.value for t in block.tags) or "-"
    keywords = ", ".join(block.keywords) or "-"
    answer = textwrap.shorten(block.trusted_answer, width=320, placeholder="…")
    return (
        f"  [{index}] {block.name}\n"
        f"      question: {block.critical_question}\n"
        f"      answer:   {answer}\n"
        f"      tags:     {tags}\n"
        f"      keywords: {keywords}"
    )


def merge_messages(blocks: list[IdeaBlock]) -> list[dict[str, str]]:
    """Assemble the chat message list for merging ``blocks`` into one canonical.

    The block payloads are rendered as a compact, read-only digest (name /
    question / shortened answer / tags / keywords); the full lifecycle and
    embedding fields are intentionally omitted -- the merge decision is a content
    decision, not a metadata one.
    """
    n = len(blocks)
    rendered = "\n".join(_render_block(b, i) for i, b in enumerate(blocks, 1))
    user_prompt = _USER_HEADER.format(n=n) + rendered
    return [
        {"role": "system", "content": merge_system_prompt(n)},
        {"role": "user", "content": user_prompt},
    ]


__all__ = ["merge_messages", "merge_system_prompt"]
