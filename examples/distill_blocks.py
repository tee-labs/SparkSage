"""Demo: de-duplicate a corpus of IdeaBlocks with the Distill pipeline.

Runs fully offline with :class:`FakeEmbeddingClient` and a scripted
:class:`FakeLLMClient` (no API key needed). It exercises the whole Distill chain
end-to-end:

    blocks
        -> BlockEmbedder.vectors_for()          (reuse the embed core)
        -> find_similar_pairs / cluster()       (near-duplicate clusters)
        -> BlockMerger.merge_cluster()          (LLM fusion -> canonical block)
        -> lifecycle write-back                 (MERGED parents, ACTIVE canonical)

To use a real model, swap in the real clients:

    pip install 'sparksage[llm,embed]'

    from sparksage import OpenAICompatibleClient, OpenAIEmbeddingClient
    llm = OpenAICompatibleClient(api_key=..., model="gpt-4o-mini")
    embedder = BlockEmbedder(OpenAIEmbeddingClient(api_key=...))

For very large corpora (>= ~1000 blocks), install the optional acceleration deps
and let the pipeline auto-select a Louvain backend:

    pip install 'sparksage[distill]'

Run with:  PYTHONPATH=src python3 examples/distill_blocks.py
"""

from __future__ import annotations

import json

from sparksage import (
    BlockEmbedder,
    BlockStatus,
    FakeEmbeddingClient,
    FakeLLMClient,
    IdeaBlock,
)
from sparksage.distill import BlockMerger, DistillPipeline

SAMPLE_BLOCKS = [
    IdeaBlock(
        name="Deploy locally",
        critical_question="How do I deploy SparkSage locally?",
        trusted_answer="Install python then run uvicorn to serve the api locally.",
    ),
    IdeaBlock(
        name="Run the server",
        critical_question="How do I start the server?",
        trusted_answer="Start uvicorn on a local port to serve the sparksage api.",
    ),
    IdeaBlock(
        name="Bake a cake",
        critical_question="How do I bake a chocolate cake?",
        trusted_answer="Mix flour sugar cocoa and eggs then bake for thirty minutes.",
    ),
    IdeaBlock(
        name="Configure env",
        critical_question="How do I configure SparkSage?",
        trusted_answer="Set environment variables or use a dotenv file for configuration.",
    ),
]


def _merge_response() -> str:
    return json.dumps(
        {
            "name": "Canonical",
            "critical_question": "How do I run SparkSage locally?",
            "trusted_answer": (
                "Install python, then run uvicorn to serve the api on a local port."
            ),
            "tags": ["PROCESS"],
            "entities": [
                {"entity_name": "SparkSage", "entity_type": "PRODUCT", "aliases": ["ss"]}
            ],
            "keywords": ["deploy", "uvicorn", "local"],
            "reasoning": "fused two near-duplicate deploy blocks",
        }
    )


def main() -> None:
    embedder = BlockEmbedder(FakeEmbeddingClient(dimension=128))
    merger = BlockMerger(FakeLLMClient(responses=[_merge_response()]))

    pipe = DistillPipeline(
        embedder=embedder,
        merger=merger,
        start_threshold=0.5,
        max_iterations=2,
    )

    print(f"Distilling {len(SAMPLE_BLOCKS)} blocks...\n")
    result = pipe.run(SAMPLE_BLOCKS)

    print(f"Input blocks:    {result.stats.input_blocks}")
    print(f"Survivors:       {len(result.survivors)}")
    print(f"Merged out:      {len(result.merged_out)}")
    print(f"Reduction:       {result.reduction:.1%}")
    print(f"LLM merge calls: {result.stats.llm_merge_calls}\n")

    print("--- survivors (status=ACTIVE) ---")
    for block in result.survivors:
        parents = len(block.parents)
        conf = f"{block.confidence:.3f}" if block.confidence is not None else "n/a"
        print(f"  [{block.status.value}] {block.name}  (parents={parents}, conf={conf})")

    print("\n--- merged out (status=MERGED) ---")
    for block in result.merged_out:
        print(f"  [{block.status.value}] {block.name}")

    print("\n--- per-iteration diagnostics ---")
    for snap in result.stats.iterations:
        print(
            f"  iter {snap.iteration}: threshold={snap.threshold:.2f} "
            f"pairs={snap.candidate_pairs} clusters={snap.merge_clusters} "
            f"merged={snap.blocks_merged} -> {snap.canonical_emitted} canonical"
        )

    assert len(result.survivors) < len(SAMPLE_BLOCKS), "expected some de-duplication"
    assert all(b.status == BlockStatus.ACTIVE for b in result.survivors)
    assert all(b.status == BlockStatus.MERGED for b in result.merged_out)
    print("\nDistill demo OK.")


if __name__ == "__main__":
    main()
