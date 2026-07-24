"""Prompt construction for LLM-driven IdeaBlock generation.

The controlled vocabularies (``Tag``, ``EntityType``) are read straight from the
enum definitions so the prompt can never drift from the code -- add a new
``Tag`` member and the model is automatically allowed to emit it.
"""

from __future__ import annotations

import textwrap

from sparksage.schema.enums import EntityType, Tag
from sparksage.schema.ideablock import QUESTION_MAX, RECOMMENDED_ANSWER_MAX
from sparksage.schema.source import SourceRef

_SYSTEM_TEMPLATE = """\
You are SparkSage, an expert knowledge engineer that decomposes source text into
question-aligned knowledge chunks called "IdeaBlocks".

An IdeaBlock is a small, self-contained unit aligned to ONE question a user
might ask. Every block MUST have:
- "name": a short title (<=200 chars) for the topic of the block.
- "critical_question": a single real question, ending with "?", that this block
  answers. Never a statement or heading.
- "trusted_answer": a verified, self-consistent answer of 2-3 sentences and NO
  MORE THAN {answer_max} characters. If the answer is long, split it across
  MULTIPLE blocks (each with its own question) instead of writing one huge
  answer. Do not pad, do not speculate beyond the source text.
- "tags": 0..N tags drawn ONLY from this controlled vocabulary:
  {tags}.
- "entities": 0..N named things the block references. Each entity is an object
  {{"entity_name": str, "entity_type": <one of: {entity_types}>, "aliases": [str]}}.
- "keywords": 0..N short keywords useful for lexical (BM25) recall.

Rules:
1. Cover the source text faithfully. Do NOT invent facts not supported by it.
2. One IdeaBlock per question; coarse-grained is fine -- prefer fewer, complete
   blocks over many tiny ones.
3. "critical_question" MUST end with "?" and be phrased as a real question.
4. Keep "trusted_answer" <= {answer_max} chars; split long topics into several
   blocks each answering its own question.
5. Only use the tag / entity_type strings listed above. If unsure, omit the tag.
6. Respond with ONLY a JSON object of the form:
   {{"blocks": [ {{...one IdeaBlock...}}, ... ]}}
   No markdown, no commentary -- just the JSON object.
"""

_SOURCE_HINT = """Source provenance attached to every block:
- uri: {uri}{title_part}{locator_part}
Reflect this source in your answers; do not fabricate a different source."""


def _enum_list(members: type) -> str:
    return ", ".join(m.value for m in members)


def system_prompt() -> str:
    """Build the system prompt, injecting the live vocabularies + limits."""
    return _SYSTEM_TEMPLATE.format(
        answer_max=RECOMMENDED_ANSWER_MAX,
        question_max=QUESTION_MAX,
        tags=_enum_list(Tag),
        entity_types=_enum_list(EntityType),
    )


def user_prompt(
    text: str,
    *,
    source: SourceRef | None = None,
    max_blocks: int | None = None,
    language: str = "en",
) -> str:
    """Build the user prompt carrying the source text to decompose."""
    parts: list[str] = []
    if source is not None:
        title_part = f", title: {source.title}" if source.title else ""
        locator_part = f", locator: {source.locator}" if source.locator else ""
        parts.append(
            _SOURCE_HINT.format(
                uri=source.uri,
                title_part=title_part,
                locator_part=locator_part,
            )
        )
    if max_blocks is not None:
        parts.append(f"Produce AT MOST {max_blocks} block(s).")
    if language and language.lower() not in ("en", "en-us"):
        parts.append(f"Write names, questions, and answers in language: {language}.")

    source_text = textwrap.dedent(text).strip()
    parts.append("Decompose the following text into IdeaBlocks:\n\n" + source_text)
    return "\n\n".join(parts)


def build_messages(
    text: str,
    *,
    source: SourceRef | None = None,
    max_blocks: int | None = None,
    language: str = "en",
) -> list[dict[str, str]]:
    """Assemble the full chat message list for the LLM."""
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(
            text, source=source, max_blocks=max_blocks, language=language
        )},
    ]
