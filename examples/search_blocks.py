"""Demo: embed IdeaBlocks, store them, search, and persist to disk.

Runs fully offline with :class:`FakeEmbeddingClient` (no API key needed). It
exercises the whole embedding retrieval stack end-to-end:

    blocks --BlockEmbedder--> vectors --InMemoryVectorStore--> search
                                          |
                            save_store / load_store (JSON)

To use a real model, swap in :class:`OpenAIEmbeddingClient`:

    pip install 'sparksage[embed]'

    from sparksage import OpenAIEmbeddingClient
    client = OpenAIEmbeddingClient(api_key=..., model="text-embedding-3-small")

Run with:  PYTHONPATH=src python3 examples/search_blocks.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from sparksage import (
    BlockEmbedder,
    FakeEmbeddingClient,
    IdeaBlock,
    InMemoryVectorStore,
    load_store,
    save_store,
)

SAMPLE_BLOCKS = [
    IdeaBlock(
        name="Deploy SparkSage",
        critical_question="How do I deploy SparkSage locally?",
        trusted_answer="Install python then run uvicorn to serve the api locally.",
    ),
    IdeaBlock(
        name="Run the server",
        critical_question="How do I start the server?",
        trusted_answer="Start uvicorn on port 8000 to serve the sparksage api.",
    ),
    IdeaBlock(
        name="Bake a cake",
        critical_question="How do I bake a chocolate cake?",
        trusted_answer="Mix flour sugar cocoa and eggs then bake for thirty minutes.",
    ),
]

QUERY_TEXT = "how to deploy and run the sparksage api locally"


def main() -> None:
    dim = 256
    embedder = BlockEmbedder(FakeEmbeddingClient(dimension=dim))

    # 1) embed blocks WITHOUT mutating them (vectors_for returns a dict).
    vectors = embedder.vectors_for(SAMPLE_BLOCKS)

    # 2) index into an in-memory store and search.
    store = InMemoryVectorStore(dimension=dim)
    store.add_many(vectors)
    print(f"Indexed {len(store)} blocks (dimension={store.dimension}).\n")

    query_vec = embedder.embed_texts([QUERY_TEXT])[0]
    print(f'Query: "{QUERY_TEXT}"\n')
    print("Top-k similar blocks:")
    for hit in store.search(query_vec, k=3):
        block = next(b for b in SAMPLE_BLOCKS if str(b.id) == hit.block_id)
        print(f"  {hit.score:+.4f}  {block.name}: {block.critical_question}")

    # 3) persist to disk and reload (would survive a process restart).
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "corpus.json"
        save_store(store, path)
        print(f"\nSaved store to {path} ({path.stat().st_size} bytes).")

        reloaded = load_store(path)
        print(f"Reloaded {len(reloaded)} vectors, dimension={reloaded.dimension}.")
        top_after = reloaded.search(query_vec, k=1)[0]
        print(
            "Top hit after reload: "
            f"{top_after.score:+.4f} -> block {top_after.block_id}\n"
        )


if __name__ == "__main__":
    main()
