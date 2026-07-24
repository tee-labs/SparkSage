"""Demo: generate IdeaBlocks from free text via an LLM.

Runs offline with a :class:`FakeLLMClient` (no API key needed). To use a real
model, swap in :class:`OpenAICompatibleClient`:

    pip install 'sparksage[llm]'

    from sparksage import OpenAICompatibleClient
    client = OpenAICompatibleClient(api_key=..., model="gpt-4o-mini")

Run with:  PYTHONPATH=src python3 examples/generate_blocks.py
"""

from __future__ import annotations

from sparksage import FakeLLMClient, IdeaBlockGenerator
from sparksage.schema.source import SourceRef

SAMPLE_TEXT = """
SparkSage replaces naive fixed-size text slicing with the IdeaBlock, a small,
self-contained knowledge unit aligned to a single question. Every IdeaBlock
carries a critical question and a concise, verified trusted answer. Only the
trusted_answer field is embedded, which avoids the mid-sentence cuts that wreck
naive chunking. Rich metadata (tags, entities, keywords) powers filtering and
hybrid retrieval.
"""

FAKE_RESPONSE = """
Here are the IdeaBlocks:

{
  "blocks": [
    {
      "name": "What SparkSage does",
      "critical_question": "What problem does SparkSage solve?",
      "trusted_answer": "SparkSage turns documents into question-aligned units.",
      "tags": ["IMPORTANT", "TECHNOLOGY"],
      "entities": [{"entity_name": "SparkSage", "entity_type": "PRODUCT"}],
      "keywords": ["rag", "chunking"]
    },
    {
      "name": "Embedding strategy",
      "critical_question": "Which field should be embedded?",
      "trusted_answer": "Only the trusted_answer field is embedded, avoiding mid-sentence cuts.",
      "tags": ["ARCHITECTURE"],
      "keywords": ["embedding", "single-field"]
    }
  ]
}
"""


def main() -> None:
    client = FakeLLMClient(responses=[FAKE_RESPONSE])
    gen = IdeaBlockGenerator(client)

    blocks, stats = gen.generate_with_stats(
        SAMPLE_TEXT,
        source=SourceRef(uri="file://docs/overview.md", title="Overview"),
    )

    print(f"Generated {len(blocks)} block(s) "
          f"(emitted={stats.emitted}, skipped={stats.skipped}):\n")
    for i, b in enumerate(blocks, 1):
        print(f"--- Block {i} ---")
        print(b.to_xml())
        print(f"embedding_text: {b.embedding_text}\n")


if __name__ == "__main__":
    main()
