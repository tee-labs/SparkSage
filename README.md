# SparkSage

**Structured, question-aligned knowledge chunks for high-quality RAG.**

SparkSage replaces naive fixed-size text slicing with the **IdeaBlock** — a
small, self-contained *knowledge unit* that is aligned to how users ask
questions. Instead of embedding arbitrary text fragments (which get cut
mid-sentence and retrieve poorly), SparkSage embeds whole, verified answers.

> Status: **Pre-Alpha**. SparkSage is a **knowledge *preprocessing* library for
> RAG** — not an end-to-end question-answering system. It owns everything from
> raw bytes to a retrievable, de-duplicated corpus and hands clean structures
> off to *your* retriever / generator. Implemented today:
>
> - the **chunk schema** (IdeaBlock + TechnicalBlock);
> - **LLM-driven generation** (free text → many IdeaBlocks);
> - **file-to-Markdown conversion** (any format via a pluggable backend);
> - **customizable, source-aware text cleaning**;
> - **dense-vector embedding & retrieval** (`EmbeddingClient` Protocol + an
>   in-memory kNN store, JSON persistence, all-pairs near-duplicate detection,
>   and production backends for FAISS / Chroma / pgvector);
> - the **Distill de-duplication pipeline** (detect → cluster → LLM-merge →
>   lifecycle write-back), with an async job layer and LSH acceleration for
>   million-vector corpora;
> - a **document-management service** (parsed documents + free-form tags +
>   summaries + durable SQLite storage);
> - a **dependency-free auto-tagging engine** (RAKE / TF-IDF / TextRank, CJK
>   out of the box);
> - **query-time intent recognition + rewriting** (multi-turn, with rule-based
>   and LLM implementations);
> - a **measurable benchmark suite** (IdeaBlock vs naive chunking on your own
>   data); and
> - a **WEB API** exposing convert / generate / documents / tags over HTTP.

---

## Why question-aligned chunks?

Traditional `RecursiveCharacterTextSplitter` chunks:

- get cut mid-sentence → semantic breakage,
- carry no notion of *what question they answer* → sparse vector clusters,
- lack queryable metadata → weak filtering / hybrid retrieval.

An IdeaBlock fixes all three at the data layer:

| Problem | IdeaBlock answer |
| --- | --- |
| Sentence breakage | Single-field embedding of a concise `trusted_answer` |
| No query alignment | Every block carries its `critical_question` |
| Poor metadata | `tags` / `entities` / `keywords` / provenance |

---

## The IdeaBlock schema

```xml
<ideablock>
  <name>标题 / short title</name>
  <critical_question>the single question this block answers?</critical_question>
  <trusted_answer>verified, self-consistent answer (2–3 sentences, ≤500 chars)</trusted_answer>
  <tags>IMPORTANT, TECHNOLOGY, ...</tags>
  <entity><entity_name>..</entity_name><entity_type>PRODUCT|..</entity_type></entity>
  <keywords>keywords for BM25 / lexical recall</keywords>
</ideablock>
```

Core model: [`src/sparksage/schema/ideablock.py`](src/sparksage/schema/ideablock.py).

### Design principles

- **Question–answer alignment** — `critical_question` + `trusted_answer` align
  the chunk to the query manifold, so dense vectors cluster tightly around
  user intent.
- **Single-field embedding** — only `trusted_answer` is embedded by default,
  killing the "splitter cut my sentence in half" problem. See
  [`embedding_text`](src/sparksage/schema/ideablock.py).
- **Rich, queryable metadata** — `tags` / `entities` / `keywords` power
  filtering, permission scoping and hybrid (BM25 + dense) retrieval.
- **Provenance & lifecycle** — every block knows its source
  ([`SourceRef`](src/sparksage/schema/source.py)) and dedup state
  (`status` / `parents`), so the corpus is auditable and the Distill pipeline
  can merge safely.

### TechnicalBlock (ordered content variant)

For manuals / SOPs / runbooks where *sequence is meaning*, the
[`TechnicalBlock`](src/sparksage/schema/technical.py) layers in:

- **ordered, role-tagged sentences** (`INFO` / `COMMAND` / `WARNING` /
  `PREREQUISITE` / `REFERENCE` / `RESULT`), and
- **Primary / Proceeding / Following** context windows.

It inherits the full IdeaBlock core, so it interoperates with the same
retrieval stack.

---

## Quick start

```bash
pip install -e ".[dev]"

python3 - <<'PY'
from sparksage.schema import IdeaBlock, Tag, Entity, EntityType, BlockStatus

block = IdeaBlock(
    name="What SparkSage does",
    critical_question="What problem does SparkSage solve?",
    trusted_answer=(
        "SparkSage turns documents into question-aligned knowledge units so "
        "retrieval hits whole, self-contained answers instead of text shards."
    ),
    tags=[Tag.IMPORTANT, Tag.TECHNOLOGY],
    entities=[Entity(entity_name="SparkSage", entity_type=EntityType.PRODUCT)],
    keywords=["rag", "chunking"],
    status=BlockStatus.ACTIVE,
)
print(block.embedding_text)
print(block.to_xml())
PY
```

A fuller end-to-end demo:

```bash
PYTHONPATH=src python3 examples/build_chunks.py
```

---

## Generate IdeaBlocks from text

SparkSage decomposes a passage of free text into several question-aligned
IdeaBlocks via an LLM. The generation core depends on a small
[`LLMClient`](src/sparksage/generator/client.py) protocol, so it works with any
OpenAI-compatible endpoint (OpenAI, Azure, vLLM, Ollama, GLM, ...) and is fully
testable offline with a deterministic fake.

```bash
pip install 'sparksage[llm]'   # pulls the optional 'openai' SDK
```

```python
from sparksage import IdeaBlockGenerator, OpenAICompatibleClient

client = OpenAICompatibleClient(api_key="...", model="gpt-4o-mini")
gen = IdeaBlockGenerator(client)

blocks = gen.generate(
    "SparkSage replaces naive text slicing with question-aligned chunks ...",
    source_uri="file://docs/overview.md",
)
for b in blocks:
    print(b.critical_question, "->", b.trusted_answer)
```

How it stays robust and schema-safe:

- The prompt teaches the model the IdeaBlock format and the **live controlled
  vocabularies** (`Tag` / `EntityType`) read straight from the enum definitions,
  so it can never drift from the code.
- Model output is parsed into [lenient intermediate
  models](src/sparksage/generator/schema.py), then **coerced** through the
  vocabularies into strict `IdeaBlock`s. Unknown tags are dropped; the
  `critical_question` is repaired to end with `?`; oversized answers are skipped
  (split into more blocks instead of truncating).
- `strict=True` fails fast on the first malformed block; the default skips bad
  blocks and reports them via `generate_with_stats()`.
- Provenance (`source_uri`) is attached to every emitted block.

Offline demo (no API key):

```bash
PYTHONPATH=src python3 examples/generate_blocks.py
```

---

## Convert any file to Markdown

Before chunking, source documents come in many formats. SparkSage normalizes them
all to Markdown (the lingua franca downstream generation expects) via a pluggable
backend built on Microsoft
[`markitdown`](https://github.com/microsoft/markitdown) — PDF, Word, PowerPoint,
Excel, HTML, CSV/JSON/XML, images (EXIF + OCR), audio (transcription), EPub, ZIP
archives and more.

```bash
pip install 'sparksage[convert]'   # pulls markitdown[all]
```

```python
from sparksage import MarkdownConverter

conv = MarkdownConverter()

# single file -> Markdown
result = conv.convert("report.pdf")
print(result.markdown)

# whole directory tree -> .md files
conv.convert_directory("docs/", dest_dir="docs_md/")
```

The returned [`ConversionResult`](src/sparksage/convert/converter.py) chains
straight into generation:

```python
blocks = IdeaBlockGenerator(client).generate(
    result.markdown, source=result.source_ref,
)
```

How it stays robust and dependency-light:

- The conversion core depends only on a small
  [`ConverterBackend`](src/sparksage/convert/backend.py) protocol, so it is fully
  unit-testable offline with a deterministic fake — `markitdown` is imported
  lazily and only when no backend is injected.
- Batch conversion is **resilient**: a single bad file is logged and skipped
  rather than aborting the whole run.
- `convert_directory` filters by a sensible
  [`DEFAULT_EXTENSIONS`](src/sparksage/convert/converter.py) set (overridable)
  and recurses by default; `convert_to_file` writes `<name>.md` for each source.

Offline demo (no `markitdown` needed):

```bash
PYTHONPATH=src python3 examples/convert_files.py
```

---

## Clean document text

Conversion yields *raw* Markdown faithful to the source bytes — but that text is
seldom generation-ready: BOMs, mixed line endings, leaked control characters,
page headers/footers, watermarks, boilerplate, PII. **Which of those are noise
depends on your business**, so cleaning is built to be customized.

[`TextCleaner`](src/sparksage/clean/cleaner.py) applies a pipeline of tiny,
composable rules. Rules can be **global** (every document) or
**source/filename-specific** (PDF footers only, Confluence macros only, ...):

```python
from sparksage import TextCleaner, RegexReplaceRule

cleaner = TextCleaner()                                     # sensible defaults
cleaner.add(RegexReplaceRule(r"CONFIDENTIAL", ""))          # every document
cleaner.add(RegexReplaceRule(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED]"))  # PII
cleaner.add_for("*.pdf", RegexReplaceRule(r"Page \d+ of \d+", ""))     # PDF footers only

cleaned = cleaner.clean(raw_text, source="docs/report.pdf")
# or chain straight off a ConversionResult:
cleaned = cleaner.clean_result(conv_result)

blocks = IdeaBlockGenerator(client).generate(
    cleaned.text, source=cleaned.source_ref,
)
```

Built-in rules cover the normalization that helps almost every document
(`RemoveBomRule`, `NormalizeLineEndingsRule`, `RemoveControlCharsRule`,
`StripTrailingWhitespaceRule`, `CollapseBlankLinesRule`, `RemoveHtmlCommentsRule`).
Two escape hatches cover business-specific surgery without writing a class:

- [`RegexReplaceRule`](src/sparksage/clean/rules.py) — pattern-based
  remove/replace (watermarks, footers, redaction, terminology normalization).
- [`CallableRule`](src/sparksage/clean/rules.py) — wrap any
  `(text, source) -> text` function.

For full control, implement the
[`CleaningRule`](src/sparksage/clean/rules.py) protocol (a single `clean` method)
and register it. Source routing lives in the
[`CleaningRegistry`](src/sparksage/clean/registry.py), which matches by glob
(against both path and basename) or regex.

How it stays robust:

- The cleaning core depends only on the `CleaningRule` protocol and the
  `CleaningRegistry` dispatcher — pure Python, no external dependencies, fully
  unit-testable offline.
- `DEFAULT_RULES` run first (normalize bytes before business logic); custom rules
  layer on top in registration order. Pass `use_defaults=False` for total control.

Offline demo (convert -> clean -> generate, no API key, no `markitdown`):

```bash
PYTHONPATH=src python3 examples/clean_text.py
```

---

## Embed & retrieve IdeaBlocks

Once you have IdeaBlocks, vectorize them and search by similarity. Only
[`embedding_text`](src/sparksage/schema/ideablock.py) (name + question + answer)
is ever embedded, so a whole, verified answer is what gets matched.

```bash
pip install 'sparksage[embed]'   # pulls the optional 'openai' SDK
```

```python
from sparksage import (
    BlockEmbedder,
    InMemoryVectorStore,
    OpenAIEmbeddingClient,
)

client = OpenAIEmbeddingClient(api_key="...", model="text-embedding-3-small")
embedder = BlockEmbedder(client)

# vectors_for() returns {block_id: vector} WITHOUT mutating the blocks
# (keeps large corpora memory-light and vectors off the objects).
vectors = embedder.vectors_for(blocks)

store = InMemoryVectorStore(dimension=client.dimension)
store.add_many(vectors)

# embed a free-text query, then kNN-search the store
query_vec = embedder.embed_texts(["how do I deploy sparksage?"])[0]
for hit in store.search(query_vec, k=5):
    print(hit.score, hit.block_id)
```

The store is **text-agnostic**: it indexes vectors keyed by an opaque
`block_id` string and computes pure dot products (cosine, since every client
L2-normalizes). Embedding a query is one `embed_texts` call — retrieval stays
decoupled from embedding, exactly like the generator is decoupled from the LLM
client.

How it stays dependency-light and pluggable:

- The retrieval core depends only on the small
  [`VectorStore`](src/sparksage/embed/store.py) Protocol (`dimension` / `add` /
  `search`), so it is fully unit-testable offline with
  [`FakeEmbeddingClient`](src/sparksage/embed/client.py). The brute-force
  [`InMemoryVectorStore`](src/sparksage/embed/store.py) is pure stdlib (no
  `numpy` / `faiss`). Production backends in
  [`embed/backends/`](src/sparksage/embed/backends/) each lazily import their own
  SDK so the core stays zero-dependency — install only what you need:
  [`FaissVectorStore`](src/sparksage/embed/backends/faiss_store.py) (exact
  inner-product index, `[distill]` extra),
  [`ChromaVectorStore`](src/sparksage/embed/backends/chroma_store.py) (`[chroma]`)
  and [`PgvectorVectorStore`](src/sparksage/embed/backends/pgvector_store.py)
  (`[pgvector]`, Supabase/Postgres). All three assume L2-normalized vectors and
  return scores directly comparable to `InMemoryVectorStore.search`.
- Vectors are stored by value (copied on add), so callers can't corrupt the
  index. `search` returns [`SearchHit`](src/sparksage/embed/store.py)s sorted
  best-first.
- [`save_store`](src/sparksage/embed/persist.py) / [`load_store`](src/sparksage/embed/persist.py)
  persist a store to a **zero-dependency JSON** file so embeddings survive
  restarts; the loader validates the format marker + version and fails fast on
  corrupt / foreign files rather than guessing.

```python
from sparksage import save_store, load_store

save_store(store, "corpus.json")     # embeddings to disk
store = load_store("corpus.json")    # reload next run (same VectorStore)
```

### Find near-duplicate blocks

The store answers "what's most similar to *this query*?" — but Distill also
needs "which blocks are duplicates of *each other*?". That is the all-pairs
counterpart:

```python
from sparksage import find_similar_pairs

# vectors is the same {block_id: vector} dict from vectors_for() above
for pair in find_similar_pairs(vectors, threshold=0.6):
    print(f"{pair.score:.3f}  {pair.a} ~ {pair.b}")
```

`find_similar_pairs` is pure stdlib (`O(n²·d)`, fine for thousands of blocks),
returns each unordered pair once (`a <= b`), sorted by score then by id for
determinism. It is the first dependency-free step of the Distill de-dup
pipeline; approximate LSH + FAISS candidate reduction takes over at
million-vector scale under the `[distill]` extra.

Offline demo (embed -> index -> search -> persist -> reload, no API key):

```bash
PYTHONPATH=src python3 examples/search_blocks.py
```

---

## De-duplicate with Distill

Once a corpus is embedded, near-duplicates creep in -- the same fact re-stated
across documents, slight rewordings of one answer, a copy-pasted procedure.
**Distill** collapses them into a smaller set of canonical, more complete
IdeaBlocks. It is built on the existing building blocks rather than new ones:

- candidate detection reuses `find_similar_pairs`;
- clustering is a protocol with a pure-stdlib default (union-find connected
  components) and an optional Louvain backend under `[distill]`;
- the merge step reuses the existing `LLMClient` protocol;
- lifecycle write-back uses the schema fields that already exist for this
  purpose -- `status` / `parents` / `confidence`.

```python
from sparksage import BlockEmbedder, OpenAIEmbeddingClient, OpenAICompatibleClient
from sparksage.distill import DistillPipeline, BlockMerger

embedder = BlockEmbedder(OpenAIEmbeddingClient(api_key="..."))
merger = BlockMerger(OpenAICompatibleClient(api_key="...", model="gpt-4o-mini"))
pipe = DistillPipeline(embedder=embedder, merger=merger)

result = pipe.run(blocks)
print(f"{len(blocks)} -> {len(result.survivors)} blocks ({result.reduction:.1%} dedup)")

# result.survivors   -> canonical merged + untouched singletons, all ACTIVE
# result.merged_out  -> folded blocks, status=MERGED (kept for audit/rollback)
# result.stats       -> per-iteration diagnostics (threshold, pairs, clusters)
```

The pipeline runs **iterative threshold refinement**: start permissive (default
`0.55`), merge the obvious duplicates, re-embed the canonical blocks, then
tighten by `+0.01`/round (cap `0.98`, ~4 rounds). This collapses *chains* of
near-duplicates a single pass would miss, while never merging below the
tightened bar. Clusters larger than the per-call budget (default 20) are
**hierarchically partitioned** by their strongest intra-cluster edges, merged
bottom-up, and merged again -- so even a 10k-block cluster never exceeds one LLM
context per call.

How it stays dependency-light and pluggable:

- The pipeline depends only on three protocols -- `EmbeddingClient` (via
  `BlockEmbedder`), `LLMClient` (via `BlockMerger`), and `ClusteringBackend` --
  so it is fully unit-testable offline with `FakeEmbeddingClient` /
  `FakeLLMClient`. `numpy` / `networkx` / `python-louvain` belong to the optional
  `[distill]` extra and are imported lazily inside `LouvainClusteringBackend`.
- Merged-away blocks get `status=MERGED`; canonical blocks get `status=ACTIVE`,
  `parents` = the merged UUIDs, and `confidence` = the cluster's mean pairwise
  similarity. Nothing leaves the IdeaBlock data model.
- The merge step is **resilient**: in non-strict mode (default), one bad LLM
  output falls back to promoting a member rather than aborting a 100k-block run.

For very large corpora (≥ ~1000 blocks), install the acceleration deps and let
the pipeline auto-select a Louvain backend:

```bash
pip install 'sparksage[distill]'   # numpy + networkx + python-louvain
```

Offline demo (no API key; scripted FakeLLMClient does a real merge):

```bash
PYTHONPATH=src python3 examples/distill_blocks.py
```

---

## Auto-tag documents

A corpus is only as filterable as its metadata. When a document arrives
*without* tags, SparkSage derives them from the content using classic,
**dependency-free** algorithms — no LLM, no NLTK / spaCy / jieba in the core.
The `tags/` package depends only on a small
[`Tokenizer`](src/sparksage/tags/tokenizer.py) Protocol and the stop-word sets
in [`stoplist.py`](src/sparksage/tags/stoplist.py):

```python
from sparksage import make_extractor

extractor = make_extractor("rake")          # or "tfidf" / "textrank"
for ks in extractor.extract(my_text, top_k=8):
    print(f"{ks.score:.3f}  {ks.keyword}")
```

Three extractors ship out of the box, each pure stdlib:

- [`RakeKeywordExtractor`](src/sparksage/tags/extractor.py) — phrase co-occurrence
  scoring (the default; fast, good on English).
- [`TfidfKeywordExtractor`](src/sparksage/tags/extractor.py) — term frequency ×
  inverse document frequency over the document's own sentences.
- [`TextRankKeywordExtractor`](src/sparksage/tags/extractor.py) — token
  co-occurrence graph + PageRank.

CJK (Chinese / Japanese / Korean) works out of the box via the dictionary-free
[`CharBigramTokenizer`](src/sparksage/tags/tokenizer.py) (overlapping character
bigrams carry strong topical signal). Word-level Mandarin segmentation is the
optional [`JiebaTokenizer`](src/sparksage/tags/tokenizer.py) under `[tags-zh]`,
imported lazily like every other optional SDK. [`AutoTokenizer`](src/sparksage/tags/tokenizer.py)
inspects the text and routes CJK → bigrams, Latin → whitespace, and is the
default tokenizer of every extractor.

Tags are **free-form** (`KeywordScore.keyword → list[str]` on the document) —
intentionally *not* the closed [`Tag`](src/sparksage/schema/enums.py) enum,
which keeps its coarse-grained semantic-filtering role. `make_extractor(name)`
is the config-driven factory (unknown names fail fast).

Offline demo (no dependencies, exercises all three algorithms):

```bash
PYTHONPATH=src python3 examples/extract_tags.py
```

---

## Manage documents

There was no *document* object — only chunk-level IdeaBlocks. The
`documents/` package fills that gap with a
[`DocumentRecord`](src/sparksage/documents/models.py): a Pydantic v2 entity
(`extra="forbid"`, like every schema model) carrying `title` / `summary` /
`body_markdown` / **free-form** `tags: list[str]` /
[`SourceRef`](src/sparksage/schema/source.py) provenance / timestamps / a
`content_hash` for cheap change detection.

```python
from sparksage import (
    InMemoryDocumentStore, SqliteDocumentStore, new_record,
)

# ephemeral (great for tests / a single process)
store = InMemoryDocumentStore()

# or durable (single-file SQLite, no server)
store = SqliteDocumentStore("./docs.db")

record = store.save(new_record(
    title="Annual Report",
    body_markdown="# Annual Report\nRevenue grew 12% ...",
    tags=["revenue", "annual"],
    source="file://docs/annual_report.md",
))
for r in store.list(tag="revenue", limit=10):
    print(r.doc_id, r.title, r.tags)
```

The storage layer depends only on the
[`DocumentStore`](src/sparksage/documents/store.py) Protocol
(`save`/`get`/`list`/`delete`/`count`/`list_tags`/`__contains__`/`__len__`) —
never on `sqlite3`-specific SQL in the core.
[`InMemoryDocumentStore`](src/sparksage/documents/backends/memory.py) is pure
stdlib; the durable
[`SqliteDocumentStore`](src/sparksage/documents/backends/sqlite.py) owns a
`documents` table plus a `<table>_tags` junction table for exact-match tag
filtering, and is thread-safe across the FastAPI threadpool.

[`ExtractiveSummarizer`](src/sparksage/documents/summarizer.py) produces the
document-level summary: frequency-scored sentences returned in original order,
Markdown heading / emphasis markers stripped — no LLM needed.

Offline demo (in-memory store → CRUD → tag vocabulary; no API key, no SQLite file kept):

```bash
PYTHONPATH=src python3 examples/manage_documents.py
```

---

## Process queries (intent + rewrite)

Query-time is the counterpart of the ingest pipeline. Before a question ever
hits retrieval, SparkSage classifies its intent, intercepts out-of-domain /
low-confidence queries, and rewrites the (possibly multi-turn, anaphora-heavy)
phrasing into search-ready text. The `query/` package reuses the existing
[`LLMClient`](src/sparksage/generator/client.py) Protocol — never a concrete LLM
SDK — so it runs fully offline under a [`FakeLLMClient`](src/sparksage/generator/client.py).

```python
from sparksage import (
    OpenAICompatibleClient, QueryIntent,
)
from sparksage.query import (
    QueryProcessor, LLMIntentClassifier, LLMQueryRewriter,
    RuleIntentClassifier, KeywordIntentRule,
    ConversationContext,
)

client = OpenAICompatibleClient(api_key="...", model="gpt-4o-mini")
proc = QueryProcessor(
    classifier=RuleIntentClassifier([
        KeywordIntentRule(("weather", "笑话"), QueryIntent.OUT_OF_DOMAIN),
    ]),
    rewriter=LLMQueryRewriter(client),
)

# first turn
result = proc.process("中国移动2024年净利润怎么样")
if result.accepted:
    retrieve(result.rewrite.rewritten_query)   # e.g. "China Mobile 2024 net profit"
else:
    show(result.default_reply)

# follow-up turn — anaphora ("那联通呢") resolved against history
ctx = ConversationContext().with_turn("user", "...the China Mobile answer...")
result = proc.process("那联通呢", context=ctx)
```

The two stages are independent, swappable protocols, each with an LLM default
and a no-LLM rule-based alternative for the cost-control "high-frequency
patterns hit a rule first" pattern:

- [`IntentClassifier`](src/sparksage/query/classifier.py) — default
  [`LLMIntentClassifier`](src/sparksage/query/classifier.py) (chain-of-thought +
  JSON over the **live** [`QueryIntent`](src/sparksage/schema/enums.py)
  vocabulary), or [`RuleIntentClassifier`](src/sparksage/query/classifier.py)
  for keyword/regex routing.
- [`QueryRewriter`](src/sparksage/query/rewriter.py) — default
  [`LLMQueryRewriter`](src/sparksage/query/rewriter.py), or
  [`RuleQueryRewriter`](src/sparksage/query/rewriter.py) for template rules.

[`QueryProcessor`](src/sparksage/query/processor.py) wires them together with an
**interception policy**: which intents to reject (`OUT_OF_DOMAIN` by default), a
`min_confidence` floor (`0.4`), and a canned reply — all configuration, not
hidden behaviour. The lenient→strict two-stage pattern is reused verbatim from
`generator/`: raw LLM output is parsed into lenient `RawIntent` / `RawRewrite`,
then coerced through the `QueryIntent` enum into strict `IntentResult` /
`RewriteResult`. [`ConversationContext`](src/sparksage/query/context.py) is a
first-class, immutable value object baked into the rewrite prompt, so multi-turn
anaphora resolution ("那", "it", "the same") is supported from day one.

> **Note:** this is the framework-agnostic core. A future `/api/v1/query` route
> will be a thin wrapper, mirroring how
> [`SparkSageService`](src/sparksage/api/pipeline.py) wraps the ingest pipeline.

Offline demo (rule classifier + scripted FakeLLMClient rewriter; no API key):

```bash
PYTHONPATH=src python3 examples/process_query.py
```

---

## Benchmark IdeaBlock vs naive chunking

The adoption-blocking question -- *"is the question-aligned IdeaBlock design
actually better than the recursive-character splitter everyone uses?"* -- is
answered **measurably, on your own corpus**. The benchmark reuses the existing
`BlockEmbedder` and `InMemoryVectorStore` and adds only a dependency-free
reimplementation of the LangChain recursive splitter as the baseline:

```python
from sparksage import BlockEmbedder, OpenAIEmbeddingClient
from sparksage.bench import BenchmarkRunner

runner = BenchmarkRunner(
    embedder=BlockEmbedder(OpenAIEmbeddingClient(api_key="...")),
)
report = runner.run(my_blocks)

print(report.summary())
open("benchmark.html", "w").write(report.to_html())   # self-contained report
```

The runner builds **two indexes over the same corpus** -- one vector per
IdeaBlock vs one vector per naive chunk -- runs the **same queries** (each
block's `critical_question`, ground truth = the block itself) against both, and
scores top-k retrieval (**hit@k**, **MRR**, mean top score) + token efficiency.
The comparison is fair by construction: same embedder, same queries, same ground
truth -- only the chunking strategy differs.

`BenchmarkReport.to_html()` renders a self-contained HTML page (no external
CSS/JS, no template engine) with the side-by-side metrics, improvement factors,
and configuration snapshot -- the "prove the ROI on your own data" artifact,
shareable as a single file. Plug a real tokenizer via
`BenchmarkRunner(token_counter=...)` for absolute token numbers.

How it stays dependency-light:

- The runner is pure stdlib + the embedding client you already have -- no
  LangChain, no metric library, no template engine. It runs offline with
  `FakeEmbeddingClient`.
- `RecursiveCharSplitter` is a faithful reimplementation of the
  `RecursiveCharacterTextSplitter` (the LangChain default), so the baseline is
  reproducible without any extra install.

Offline demo (no API key):

```bash
PYTHONPATH=src python3 examples/run_benchmark.py
```

---

## Serve the WEB API

SparkSage exposes its capabilities over a small HTTP API:

* `POST /api/v1/convert` — upload a file, get back Markdown (optionally cleaned).
* `POST /api/v1/generate` — upload a file, get back a list of IdeaBlocks.
* `POST /api/v1/documents` — upload a file → parse → clean → (auto-tag) →
  (summarize) → store a [`DocumentRecord`](src/sparksage/documents/models.py).
* `GET /api/v1/documents` — paginated listing (`?tag=` / `?q=` filters).
* `GET / PATCH / DELETE /api/v1/documents/{doc_id}` — single-document CRUD.
* `POST /api/v1/documents/{doc_id}/tags` — re-extract tags from the body.
* `GET /api/v1/tags` — distinct tag vocabulary across stored documents.
* `GET /api/v1/health` — liveness probe; reports the version and whether
  generation is configured. Used by the Docker `HEALTHCHECK`.

The API layer is a thin shell over a framework-agnostic
[`SparkSageService`](src/sparksage/api/pipeline.py) that wires convert → clean →
generate (and, for documents, → auto-tag → summarize → store) together. FastAPI
is an *optional* dependency.

```bash
pip install 'sparksage[api]'          # fastapi + uvicorn + python-multipart
pip install 'sparksage[convert]'      # markitdown for real file conversion
pip install 'sparksage[llm]'          # openai SDK for real generation
```

#### Vector-store backends

The default `InMemoryVectorStore` needs no dependencies. For production-scale
retrieval, install the backend you need:

```bash
pip install 'sparksage[distill]'      # FaissVectorStore (faiss-cpu + numpy)
pip install 'sparksage[chroma]'       # ChromaVectorStore (ChromaDB, local-dev-first)
pip install 'sparksage[pgvector]'     # PgvectorVectorStore (psycopg v3 + Postgres)
```

All three implement the same `VectorStore` Protocol, so you swap them in
wherever an `InMemoryVectorStore` is used.

### Run the server

```bash
export SPARKSAGE_API_KEY=sk-...                    # or OPENAI_API_KEY
export SPARKSAGE_MODEL=gpt-4o-mini                 # optional
export SPARKSAGE_BASE_URL=https://your-endpoint/v1 # custom endpoint (optional)
export SPARKSAGE_STREAM=true                       # stream mode (default)
uvicorn sparksage.api.app:create_app --factory --port 8000
```

Prefer a `.env` file? See [Configuration](#configuration) — a built-in loader
(`cp .env.example .env`) means no `python-dotenv` dependency.

Interactive docs are auto-generated at `http://localhost:8000/docs`.

### Call the endpoints

```bash
# 1) file -> Markdown (optional cleaning)
curl -F "file=@report.pdf" -F "clean=true" \
     http://localhost:8000/api/v1/convert

# 2) file -> IdeaBlock list
curl -F "file=@report.pdf" -F "with_stats=true" \
     http://localhost:8000/api/v1/generate

# 3) file -> stored document (auto-tagged + summarized)
curl -F "file=@report.pdf" -F "top_k=8" \
     http://localhost:8000/api/v1/documents

# 4) list / filter documents (optional ?tag= and ?q=)
curl "http://localhost:8000/api/v1/documents?tag=revenue&limit=10"

# 5) distinct tag vocabulary across stored documents
curl http://localhost:8000/api/v1/tags
```

`POST /api/v1/convert` returns:

```json
{
  "markdown": "# Report\n\nRevenue grew 12% ...",
  "title": "Annual Report",
  "source": {"uri": "report.pdf", "title": "Annual Report"},
  "cleaned": true
}
```

`POST /api/v1/generate` returns:

```json
{
  "blocks": [
    {
      "name": "Revenue growth",
      "critical_question": "How did revenue change?",
      "trusted_answer": "Revenue grew 12% year over year.",
      "tags": ["IMPORTANT"],
      "keywords": ["revenue"],
      "source": {"uri": "report.pdf"},
      "status": "draft",
      "language": "en"
    }
  ],
  "source": {"uri": "report.pdf", "title": "Annual Report"},
  "cleaned": true,
  "stats": {"raw_block_count": 1, "emitted": 1, "skipped": 0, "errors": []}
}
```

`POST /api/v1/documents` returns the stored record (tags auto-extracted when
none are supplied; summary produced by the extractive summarizer):

```json
{
  "doc_id": "b8e1...c4",
  "title": "Annual Report",
  "summary": "Revenue grew 12% year over year.",
  "body_markdown": "# Annual Report\nRevenue grew 12% ...",
  "tags": ["revenue", "annual", "growth"],
  "source": {"uri": "report.pdf", "title": "Annual Report"},
  "content_hash": "0bd7d516...",
  "created_at": "2026-07-26T07:30:00Z",
  "updated_at": "2026-07-26T07:30:00Z"
}
```

How it stays testable and pluggable:

- The orchestration lives entirely in
  [`SparkSageService`](src/sparksage/api/pipeline.py) — no HTTP imports — so it
  is fully unit-testable offline with fakes. The FastAPI layer only does
  upload/serialization.
- `create_app(service=...)` accepts an injected service (for tests); when
  omitted it builds one from env vars (`SPARKSAGE_API_KEY` / `OPENAI_API_KEY`,
  plus `SPARKSAGE_DOC_STORE` → a durable `SqliteDocumentStore`,
  `SPARKSAGE_AUTO_TAG_EXTRACTOR` = rake|tfidf|textrank, `SPARKSAGE_TAGS_ZH` →
  jieba).
- If no API key is set, `/generate` returns a clear `503` instead of crashing;
  `/convert`, `/documents`, and `/tags` work independently of any LLM.
- Uploaded bytes are written to a short-lived temp file carrying the original
  extension (so the backend picks the right format handler), and provenance is
  set back to the *original* filename — keeping cleaning-rule routing and
  `source.uri` meaningful.

Offline demo (no API key, no `markitdown`, exercises `/convert` + `/generate`
via TestClient):

```bash
PYTHONPATH=src python3 examples/serve_api.py
```

---

## Deploy with Docker

The repository ships a production-ready [`Dockerfile`](Dockerfile) that builds
the library and serves the WEB API over uvicorn. The image is built as a
non-root, multi-stage image with the `convert` + `llm` extras pre-installed, so
both `/convert` and `/generate` work out of the box — only secrets need to be
provided at run time.

### Build & run

```bash
# build the image (Python 3.11 by default; override with a build arg)
docker build -t sparksage:latest .
docker build --build-arg PYTHON_VERSION=3.12 -t sparksage:latest .

# run it, mounting secrets via an env-file (recommended):
docker run --rm -p 8000:8000 --env-file .env sparksage:latest

# or pass individual variables:
docker run --rm -p 8000:8000 \
  -e SPARKSAGE_API_KEY=sk-... \
  -e SPARKSAGE_BASE_URL=https://your-endpoint/v1 \
  -e SPARKSAGE_MODEL=gpt-4o-mini \
  -e SPARKSAGE_STREAM=true \
  sparksage:latest
```

The API is then live at `http://localhost:8000` (interactive docs at `/docs`),
and [`/api/v1/health`](#serve-the-web-api) reports whether generation is
configured.

> No key? The container still starts and serves `/convert` and `/health`;
> `/generate` returns a clear `503` instead of crashing.

### Docker Compose

For local stacks or self-hosting, drop this in `docker-compose.yml`:

```yaml
services:
  sparksage:
    build: .
    image: sparksage:latest
    container_name: sparksage
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped
```

```bash
docker compose up --build -d
docker compose logs -f sparksage
```

### What the image gives you

- **Multi-stage build** — a `builder` stage wheels the package; a slim
  `runtime` stage installs only the wheel + extras. Source code stays out of
  the final layer and the image stays small.
- **Non-root runtime** — the server runs as a dedicated `sparksage` user
  (uid/gid `1001`), so a process breakout never starts as root.
- **Pre-bundled extras** — `sparksage[api,convert,llm]` is installed, so PDF /
  DOCX / PPTX / … conversion and OpenAI-compatible generation work without any
  extra `pip install` inside the container.
- **Built-in health check** — `HEALTHCHECK` polls `/api/v1/health` every 30s,
  so orchestrators (Docker Compose, Swarm, Kubernetes) get liveness for free.
- **Secrets never baked in** — [`.dockerignore`](.dockerignore) excludes
  `.env` / `.env.*` from the build context; secrets are injected at run time,
  matching the [12-factor](https://12factor.net/config) convention (see
  [Configuration](#configuration)).
- **Run as a library, not a server** — the default `CMD` launches the API, but
  you can override the entrypoint to use the image as a CLI tool, e.g.
  `docker run --rm sparksage:latest python -c "import sparksage; ..."`.

---

## Configuration

SparkSage reads settings from **environment variables**. You can set them the
usual way (`export ...`, container env, CI secrets), **or** drop them in a
`.env` file in the working directory — a zero-dependency `.env` loader is built
in (no `python-dotenv` required).

### Priority (highest first)

1. Real environment variables already set in the process (container / CI /
   system). These **always win**.
2. Values from the `.env` file (only fill in variables that are *not* already
   set).

This is the [12-factor](https://12factor.net/config) convention: deploy-time
secrets override the local file, so the same `.env` is safe to commit-ish
defaults while production injects real credentials.

### Quick start with `.env`

```bash
cp .env.example .env       # template is committed; .env itself is git-ignored
# edit .env:  SPARKSAGE_API_KEY=sk-...
uvicorn sparksage.api.app:create_app --factory --port 8000
```

The server calls `load_dotenv()` once on startup, so any `.env` in the CWD is
picked up automatically. You can also load it explicitly from Python:

```python
from sparksage import load_dotenv

load_dotenv()                       # reads ./.env, env vars take priority
load_dotenv("/etc/sparksage.env")   # explicit path
load_dotenv(override=True)          # let the file clobber real env vars
```

### Recognized variables

`SPARKSAGE_*` take priority over the `OPENAI_*` fallbacks.

| Variable              | Purpose                                              |
| --------------------- | ---------------------------------------------------- |
| `SPARKSAGE_API_KEY`   | API key (falls back to `OPENAI_API_KEY`)             |
| `SPARKSAGE_BASE_URL`  | OpenAI-compatible base URL (Azure/vLLM/Ollama/GLM…)  |
| `SPARKSAGE_MODEL`     | Model id (default `gpt-4o-mini`)                     |
| `SPARKSAGE_STREAM`    | Stream the LLM response (default `true`)             |
| `SPARKSAGE_LANGUAGE`  | BCP-47 code written into each block (e.g. `en`, `zh`)|
| `SPARKSAGE_LOG_LEVEL` | `sparksage` logger verbosity (default `WARNING`; `INFO`/`DEBUG` for analysis) |
| `SPARKSAGE_DOC_STORE` | Path to a SQLite file for the document store (empty → in-memory; the `/documents` routes work but do not persist across restarts) |
| `SPARKSAGE_DOC_STORE_TABLE` | SQLite table name (default `documents`)         |
| `SPARKSAGE_AUTO_TAG_EXTRACTOR` | Auto-tag algorithm: `rake` \| `tfidf` \| `textrank` (default `rake`) |
| `SPARKSAGE_TAGS_ZH`   | Use `jieba` for CJK segmentation when `true` (requires `pip install 'sparksage[tags-zh]'`) |

### Supported `.env` syntax

The built-in parser implements the well-defined subset of `.env` syntax —
`KEY=VALUE`, single/double quotes, `export` prefix, and `#` comments (a `#` is
only a comment when preceded by whitespace, so URLs like
`https://host/#anchor` stay intact). Shell expansion (`$VAR`, `$(...)`,
backticks) and multi-line values are **not** interpreted on purpose — that
avoids the quoting/injection bugs a real shell parser would introduce. See
[`sparksage.config`](src/sparksage/config.py) for details.

> Keep secrets out of git: `.env` is git-ignored. Commit `.env.example` as a
> template only.

---

## Project layout

```
src/sparksage/
├── config.py          # .env loader (stdlib; env vars take priority over file)
├── logging_config.py  # SPARKSAGE_LOG_LEVEL -> sparksage logger (stdlib; idempotent)
├── schema/
│   ├── enums.py        # controlled vocabularies (Tag, EntityType, QueryIntent, ...)
│   ├── entity.py       # named things a block references
│   ├── source.py       # provenance (where a block came from)
│   ├── ideablock.py    # the core question-aligned chunk  ★
│   └── technical.py    # order-sensitive variant for SOPs/manuals
├── generator/
│   ├── client.py       # LLMClient protocol + OpenAI-compatible + Fake client
│   ├── prompts.py      # prompt builder (reads enums -> never drifts)
│   ├── schema.py       # lenient raw models + enum coercion
│   └── generator.py    # text -> list[IdeaBlock]  ★
├── convert/
│   ├── backend.py      # ConverterBackend protocol + MarkItDown + Fake backend
│   └── converter.py    # any-file -> Markdown (single + batch)  ★
├── clean/
│   ├── rules.py        # CleaningRule protocol + built-in & configurable rules
│   ├── registry.py     # source/filename-aware rule routing (glob/regex)
│   └── cleaner.py      # raw text -> final document text  ★
├── embed/
│   ├── client.py       # EmbeddingClient protocol + OpenAI + Fake client
│   ├── indexer.py      # BlockEmbedder: blocks -> vectors (fills .embedding)  ★
│   ├── store.py        # VectorStore protocol + InMemoryVectorStore kNN  ★
│   ├── similarity.py   # find_similar_pairs: all-pairs near-duplicate detection  ★
│   ├── persist.py      # save_store / load_store (zero-dep JSON)
│   └── backends/       # FaissVectorStore / ChromaVectorStore / PgvectorVectorStore
├── distill/
│   ├── cluster.py      # ClusteringBackend protocol + union-find + Louvain  ★
│   ├── merge.py        # BlockMerger: cluster -> one canonical block (LLMClient)  ★
│   ├── prompts.py      # merge prompt (reads enums -> never drifts)
│   ├── schema.py       # lenient raw merge model + coercion
│   ├── lsh.py          # LSHCandidateReducer: random-hyperplane LSH (stdlib)  ★
│   ├── pipeline.py     # DistillPipeline: iterative refine + hierarchical merge  ★
│   └── job.py          # DistillJob / JobManager: async pollable state machine  ★
├── tags/
│   ├── tokenizer.py    # Tokenizer protocol + Auto/Whitespace/CharBigram/Jieba  ★
│   ├── stoplist.py     # English + CJK stop-word sets
│   └── extractor.py    # RAKE / TF-IDF / TextRank keyword extractors  ★
├── documents/
│   ├── models.py       # DocumentRecord (title/summary/body/tags/provenance)  ★
│   ├── store.py        # DocumentStore protocol (save/get/list/delete/count/...)
│   ├── summarizer.py   # ExtractiveSummarizer (stdlib, no LLM)  ★
│   └── backends/       # InMemoryDocumentStore + SqliteDocumentStore
├── query/
│   ├── classifier.py   # IntentClassifier protocol + LLM + rule-based  ★
│   ├── rewriter.py     # QueryRewriter protocol + LLM + rule-based  ★
│   ├── context.py      # ConversationContext (multi-turn anaphora carrier)
│   ├── prompts.py      # intent/rewrite prompts (read QueryIntent live)
│   ├── schema.py       # lenient raw models + QueryIntent coercion
│   └── processor.py    # QueryProcessor: classify -> intercept -> rewrite  ★
├── bench/
│   ├── baselines.py    # RecursiveCharSplitter (LangChain-default reimpl.)  ★
│   ├── metrics.py      # hit@k / MRR / token-efficiency (pure stdlib)
│   ├── report.py       # BenchmarkReport + zero-dep HTML renderer  ★
│   └── runner.py       # BenchmarkRunner: IdeaBlock vs naive on your corpus  ★
├── api/
│   ├── pipeline.py     # SparkSageService: convert→clean→generate→tag→store  ★
│   ├── schemas.py      # request/response Pydantic models (no fastapi)
│   └── app.py          # FastAPI app factory + routes (lazy fastapi import)
tests/                  # schema + generation + conversion + cleaning + api + config
examples/               # runnable demos
```

## Development

```bash
PYTHONPATH=src python3 -m pytest -q          # tests
ruff check src tests                          # lint
```

## Roadmap

Implemented:

- [x] Chunk schema (IdeaBlock + TechnicalBlock) — *first release*
- [x] LLM-driven generation (text -> many IdeaBlocks via pluggable LLM client)
- [x] Uniform file-to-Markdown conversion (any format -> Markdown via markitdown)
- [x] Customizable text cleaning (business-specific rules, source-aware routing)
- [x] `.env` configuration (built-in loader, env vars override file)
- [x] Dense-vector embedding & retrieval (pluggable `EmbeddingClient` +
      in-memory kNN `VectorStore` + all-pairs near-duplicate detection +
      JSON persistence, pure stdlib core)
- [x] Production vector-store backends — FAISS (`[distill]`), Chroma (`[chroma]`),
      pgvector (`[pgvector]`); each lazily imports its own SDK
- [x] Distill de-duplication pipeline (iterative threshold refinement +
      union-find/Louvain clustering + hierarchical LLM merge + lifecycle
      write-back, pure stdlib core + optional `[distill]` acceleration)
- [x] Distill async job layer (`DistillJob` / `JobManager`: pollable state
      machine + progress callbacks + cooperative cancellation) and LSH
      candidate reduction (`LSHCandidateReducer`, auto-enabled ≥ 5000 vectors)
- [x] Reproducible benchmark suite (IdeaBlock vs naive recursive splitter,
      hit@k/MRR + token efficiency, zero-dependency HTML report)
- [x] Dependency-free auto-tagging engine (`tags/`: RAKE / TF-IDF / TextRank
      over the `Tokenizer` Protocol, CJK out of the box, optional jieba)
- [x] Document-management service (`documents/`: `DocumentRecord` + free-form
      tags + extractive summary + `DocumentStore` Protocol with in-memory and
      durable SQLite backends)
- [x] Query-time intent recognition + rewriting (`query/`: LLM + rule-based
      `IntentClassifier` / `QueryRewriter`, multi-turn `ConversationContext`,
      lenient→strict `QueryIntent` coercion, `QueryProcessor` interception)
- [x] WEB API (FastAPI: `/convert`, `/generate`, `/documents` CRUD, `/tags`)

Planned next:

- [ ] `/api/v1/query` route wrapping `QueryProcessor`
- [ ] `/api/v1/distill` route wrapping `JobManager` (submit / poll / cancel)
- [ ] OpenAI-compatible ingest / distill API
- [ ] End-to-end retrieval→generation→answer-correctness benchmark (extends
      `bench/`, which today scores retrieval only)

## License

Apache-2.0.
