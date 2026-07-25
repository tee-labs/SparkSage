"""Demo: benchmark IdeaBlock vs naive chunking on your own corpus.

Runs fully offline with :class:`FakeEmbeddingClient` (no API key needed). It
exercises the whole benchmark stack end-to-end:

    blocks
        -> BenchmarkRunner.run()
            -> IdeaBlock index   (one vector per block)
            -> baseline index    (RecursiveCharSplitter -> one vector per chunk)
            -> same queries run against both (each block's critical_question)
            -> hit@k / MRR / token-efficiency scored automatically
        -> BenchmarkReport.to_html()  (self-contained HTML report)

To benchmark on your real corpus with a real embedder:

    pip install 'sparksage[embed]'

    from sparksage import BlockEmbedder, OpenAIEmbeddingClient
    embedder = BlockEmbedder(OpenAIEmbeddingClient(api_key=...))

    from sparksage.bench import BenchmarkRunner
    runner = BenchmarkRunner(embedder=embedder)
    report = runner.run(my_blocks)
    open("benchmark.html", "w").write(report.to_html())

Run with:  PYTHONPATH=src python3 examples/run_benchmark.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from sparksage import BlockEmbedder, FakeEmbeddingClient, IdeaBlock
from sparksage.bench import BenchmarkRunner, RecursiveCharSplitter

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
    IdeaBlock(
        name="Configure env",
        critical_question="How do I configure SparkSage?",
        trusted_answer="Set environment variables or use a dotenv file for configuration.",
    ),
    IdeaBlock(
        name="Persist embeddings",
        critical_question="How do I save embeddings to disk?",
        trusted_answer="Use save_store to write the vector index to a json file.",
    ),
]


def main() -> None:
    embedder = BlockEmbedder(FakeEmbeddingClient(dimension=128))
    # Default recursive splitter (chunk_size=400). On already-concise IdeaBlock
    # content each block is a single chunk, so token efficiency is ~neutral --
    # the token win shows up on *long* source documents where naive chunking
    # slices prose into 400-char fragments while IdeaBlocks stay compressed.
    # Plug your own corpus + a real embedder to see the gap.
    splitter = RecursiveCharSplitter()
    runner = BenchmarkRunner(embedder=embedder, splitter=splitter, k_values=(1, 3, 5))

    print(f"Benchmarking IdeaBlock vs naive chunking over {len(SAMPLE_BLOCKS)} blocks...")
    print(f"(naive splitter: chunk_size={splitter.chunk_size}, "
          f"overlap={splitter.chunk_overlap})\n")
    report = runner.run(SAMPLE_BLOCKS)

    print(report.summary())
    print()
    ib = report.ideablock
    base = report.baseline
    print(f"  IdeaBlock:   hit@1={ib.retrieval.hit_at_1:.1%}  "
          f"MRR={ib.retrieval.mrr:.3f}  avg tokens/unit={ib.tokens.avg_tokens:.0f}")
    print(f"  Naive chunks: hit@1={base.retrieval.hit_at_1:.1%}  "
          f"MRR={base.retrieval.mrr:.3f}  avg tokens/unit={base.tokens.avg_tokens:.0f}")
    print(f"  hit@1 improvement: {report.hit_at_1_improvement:.2f}x")
    print(f"  token efficiency:   {report.token_efficiency:.2f}x")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "benchmark.html"
        out.write_text(report.to_html(), encoding="utf-8")
        print(f"\nHTML report written to {out} ({out.stat().st_size} bytes).")

    assert report.query_count == len(SAMPLE_BLOCKS)
    print("\nBenchmark demo OK.")


if __name__ == "__main__":
    main()
