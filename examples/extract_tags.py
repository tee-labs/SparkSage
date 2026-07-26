"""Demo: auto-tag a document with RAKE / TF-IDF / TextRank.

Runs fully offline (no API key, no optional dependencies). The three extractors
are pure stdlib; CJK (Chinese / Japanese / Korean) works out of the box via the
dictionary-free CharBigramTokenizer.

For word-level Mandarin segmentation, install jieba and use the JiebaTokenizer:

    pip install 'sparksage[tags-zh]'

    from sparksage import AutoTokenizer, make_extractor
    tok = AutoTokenizer(use_jieba=True)
    extractor = make_extractor("rake", tokenizer=tok)

Run with:  PYTHONPATH=src python3 examples/extract_tags.py
"""

from __future__ import annotations

from sparksage import EXTRACTOR_NAMES, make_extractor

SAMPLE_TEXT = """
SparkSage replaces naive fixed-size text slicing with the IdeaBlock, a small,
self-contained knowledge unit aligned to a single question. Every IdeaBlock
carries a critical question and a concise, verified trusted answer. Only the
trusted_answer field is embedded, which avoids the mid-sentence cuts that wreck
naive chunking. Rich metadata, including tags, entities and keywords, powers
filtering and hybrid retrieval. The tags power coarse semantic filtering across
the corpus, while keywords support lexical recall.
"""


def main() -> None:
    print("=== sample text ===")
    print(SAMPLE_TEXT.strip())

    for name in EXTRACTOR_NAMES:
        extractor = make_extractor(name)
        print(f"\n=== {name} (top 6) ===")
        for ks in extractor.extract(SAMPLE_TEXT, top_k=6):
            print(f"  {ks.score:.3f}  {ks.keyword}")


if __name__ == "__main__":
    main()
