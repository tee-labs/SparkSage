# AGENTS.md

Guidance for AI agents working in this repository.

## Project

SparkSage is a Python library providing structured, question-aligned knowledge
chunks (the "IdeaBlock") for high-quality RAG. The schema layer is the
foundational, most important part.

## Tech stack

- Language: **Python >= 3.10** (src layout under `src/sparksage/`)
- Data validation: **Pydantic v2** (`>=2.5,<3`)
- Tests: **pytest**
- Lint: **ruff**

## Commands

```bash
# Run the test suite (src layout -> set PYTHONPATH)
PYTHONPATH=src python3 -m pytest -q

# Run a single test file
PYTHONPATH=src python3 -m pytest tests/test_ideablock.py -v

# Lint
ruff check src tests

# Editable install (pulls pydantic + dev deps)
pip install -e ".[dev]"

# Run the demo
PYTHONPATH=src python3 examples/build_chunks.py
```

## Conventions

- Package code lives under `src/sparksage/`; tests under `tests/`.
- All schema models use Pydantic v2 with `ConfigDict(extra="forbid")` to fail
  fast on typos.
- Enums are the single source of truth for controlled vocabularies
  (`schema/enums.py`). Do not inline magic strings.
- `IdeaBlock.embedding_text` is the *only* text that should be embedded.
- Do not add comments unless asked.
- Keep `trusted_answer` concise (≤ 500 chars) — split into more blocks instead.
- The generation core (`generator/generator.py`) depends only on the
  `LLMClient` Protocol — never import a concrete LLM SDK there. New clients
  implement the Protocol; raw model output is coerced through the enums before
  building strict `IdeaBlock`s (see `generator/schema.py`).
- The conversion core (`convert/converter.py`) depends only on the
  `ConverterBackend` Protocol — never import `markitdown` there. It is an
  optional dependency (`pip install 'sparksage[convert]'`), imported lazily only
  inside `MarkItDownBackend`. `MarkdownConverter` returns a `ConversionResult`
  whose `.markdown` feeds `IdeaBlockGenerator` and whose `.source_ref` provides
  provenance.
- The cleaning core (`clean/cleaner.py`) depends only on the `CleaningRule`
  Protocol and the `CleaningRegistry` dispatcher — never import a third-party
  cleaning library there. It is pure stdlib and needs no optional dependency.
  `TextCleaner` sits between conversion and generation: feed the
  `ConversionResult.markdown` (via `clean_result`) and emit a `CleaningResult`
  whose `.text` feeds `IdeaBlockGenerator` and whose `.source_ref` provides
  provenance. `source` is both provenance *and* the key for source/filename-aware
  rule routing (`add_for`), since cleaning is strongly business-dependent.
  Built-in rules are normalization only; business logic goes in custom rules
  registered on a `TextCleaner` instance.
- The embedding core (`embed/`) depends only on the `EmbeddingClient` Protocol
  — never import `openai` or `numpy` there. `OpenAIEmbeddingClient` is an
  optional dependency (`pip install 'sparksage[embed]'`), imported lazily only
  inside itself (it batches at 1000 inputs/request over a `ThreadPoolExecutor`
  and L2-normalizes by default so cosine = dot product). The core
  (`BlockEmbedder` in `embed/indexer.py`) is pure stdlib and unit-testable with
  the deterministic `FakeEmbeddingClient` (signed feature-hashing over n-grams,
  zero dependencies, gives non-trivial similarity for overlapping texts).
  Embeddings are produced from `IdeaBlock.embedding_text` only (the three-field
  `name + "\n" + critical_question + "\n" + trusted_answer` concatenation).
  `embed_blocks` fills the `.embedding` field in place; `vectors_for` returns a
  `{block_id: vector}` dict without mutating blocks — the contract a vector store
  / the future Distill pipeline consumes (vectors stay off the objects so large
  corpora stay memory-light). Vector dimension is fixed per model and probed
  lazily on the first `embed_batch` when not known statically.
- The retrieval/persistence core (`embed/store.py` + `embed/persist.py`) depends
  only on the `VectorStore` Protocol (`dimension` / `add` / `search` /
  `__contains__` / `__len__`) — never import `numpy` or `faiss` there (those are
  deferred to the future `[distill]` group, where an approximate FAISS-backed
  `VectorStore` earns its weight on million-vector corpora). `InMemoryVectorStore`
  is pure stdlib: brute-force exact kNN, score = dot product (cosine, since every
  `EmbeddingClient` L2-normalizes by default), vectors stored by value (copied on
  add so callers can't corrupt the index). It is **text-agnostic** — index vectors
  keyed by an opaque `block_id` string, and embed the query text yourself via one
  `embed_batch` call; this keeps retrieval decoupled from embedding exactly as the
  generator is decoupled from the LLM client. Feed it `BlockEmbedder.vectors_for`
  output via `add_many`; `search` returns `SearchHit`s sorted best-first.
  `save_store` / `load_store` persist a store to a **zero-dependency JSON** file
  (`{format, version, dimension, vectors}`) so embeddings survive restarts; the
  loader validates the format marker + version and fails fast on foreign/corrupt/
  future-version files rather than guessing.
- The de-dup bridge (`embed/similarity.py`) depends only on the same
  `{block_id: vector}` contract — never import `numpy` or `faiss` there.
  `find_similar_pairs` is the all-pairs counterpart to the store's one-to-many
  `search`: brute-force exact `O(n²·d)` dot products over `vectors_for` /
  `InMemoryVectorStore.vectors` output, returning `SimilarityPair`s (ids
  normalized so `a <= b`; each unordered pair once; sorted by score desc then
  `(a, b)` for determinism) at or above a `[0, 1]` threshold. For million-vector
  corpora pass a `CandidateReducer` (`candidate_reducer=` kwarg): it cheaply
  proposes a small set of *candidate* pairs (e.g. via LSH bucketing) and
  `find_similar_pairs` still does the exact dot-product verification, so
  **precision stays 1.0** — a reducer can only drop true duplicates (lowering
  recall), never invent false positives. This is the dependency-free step of the
  Distill pipeline; clustering / Louvain / LLM-merge live in the `distill/`
  package, where the pure-stdlib `LSHCandidateReducer` (random-hyperplane LSH)
  accelerates the scan at million-vector scale under the `[distill]` extra.
- The Distill de-dup core (`distill/`) depends only on three protocols — the
  existing `EmbeddingClient` (via `BlockEmbedder`), the existing `LLMClient`
  (via `BlockMerger`), and a new `ClusteringBackend` (`cluster.py`) — never
  import `numpy`, `networkx`, or `python-louvain` in the core. The default
  `ConnectedComponentsBackend` is pure stdlib (union-find over
  `find_similar_pairs` output); `LouvainClusteringBackend` is an optional
  dependency (`pip install 'sparksage[distill]'`), imported lazily only inside
  itself and auto-selected via `select_clustering_backend` only for corpora ≥
  `LOUVAIN_THRESHOLD` (1000). Both backends accept a `candidate_reducer=`
  (forwarded to `find_similar_pairs`) so a million-vector corpus can skip the
  `O(n²·d)` all-pairs scan. `partition_by_strongest_edges` powers the
  hierarchical merge: a cluster larger than the per-call budget is recursively
  split by its strongest intra-cluster edges (union-find until `~sqrt(N)*2`
  groups remain, with an even-slice fallback so the group count is *always* ≤
  target — guaranteeing strict reduction and no infinite recursion). The merge
  step (`merge.py`) reuses the lenient→strict pattern from `generator/schema`
  (`RawMergedBlock` → `coerce_merged_block`) and reuses the generator's tag /
  entity-type mapping helpers so the controlled vocabularies stay the single
  source of truth. `DistillPipeline` (`pipeline.py`) is the framework-agnostic
  orchestrator: iterative threshold refinement (`0.55` start, `+0.01`/round,
  cap `0.98`, ~4 rounds), re-embedding canonical blocks between rounds so
  near-duplicate *chains* collapse, with lifecycle write-back through the
  schema's existing fields (merged-away → `status=MERGED`; canonical →
  `status=ACTIVE`, `parents` = merged UUIDs, `confidence` = cluster mean
  similarity). It accepts an optional `candidate_reducer=` (plumbed into both
  the default backend and the hierarchical sub-cluster pair scan) and emits
  `DistillProgress` snapshots via an `on_progress=` callback (called inline from
  the worker thread at each iteration boundary; also polls an `is_cancelled=`
  predicate for cooperative cancellation). `BlockMerger.merge_calls` /
  `.fallbacks` counters feed `DistillStats`; non-strict mode (default) falls
  back to promoting a member on a bad LLM output rather than aborting a large
  run.
- The LSH candidate reducer (`distill/lsh.py`) is the `[distill]` accelerator
  for million-vector dedup. `LSHCandidateReducer` implements the
  `embed.similarity.CandidateReducer` protocol with random-hyperplane LSH:
  pure stdlib (Gaussian hyperplanes via `random.Random`, sign-of-dot-product
  hashing, no `numpy`), so it stays fully unit-testable offline like the rest
  of the core. Defaults (`num_hyperplanes=6`, `num_tables=20`, `seed=42`) give
  ~89% recall at cosine 0.55 (the Distill start) and 97%+ across the tightened
  regime; `theoretical_recall(s)` is the closed-form recall curve for tuning.
  `select_candidate_reducer` auto-enables it only for corpora ≥
  `LSH_ACTIVATION_THRESHOLD` (5000) — below that the exact brute force wins.
  Precision is always 1.0 because `find_similar_pairs` exact-verifies every
  candidate via dot product.
- The async Distill job layer (`distill/job.py`) wraps `DistillPipeline` in a
  pollable state machine (`DistillJob`: `queued → running → success | failed |
  timeout | cancelled`) for long-running dedup runs (minutes on 10k blocks,
  hours on a million). It depends only on the existing pipeline plus stdlib
  (`threading`, `asyncio`) — never import a job queue or task framework there.
  `run_sync()` drives the pipeline in the caller thread; `start()` runs it in a
  worker thread via `asyncio.to_thread` (the blocking LLM/embedding I/O stays
  off the event loop), with optional `timeout=`. The pipeline's `on_progress`
  callback (invoked from the worker) updates a lock-protected `JobSnapshot`
  (percent / phase / iteration / threshold / active_blocks / candidate_pairs /
  ...); `cancel()` flips a cooperative predicate the pipeline polls at iteration
  boundaries, so cancelled/timed-out runs wind down promptly (Python cannot kill
  the worker thread, so cancellation is cooperative — the partial result is
  retained on the snapshot). `JobManager` is the in-process registry a future
  `/api/v1/distill` route will wrap: `submit()` returns immediately (autostarts
  in a background thread / loop task), `snapshot(id)` backs `GET /jobs/{id}`,
  `wait_for(id)` / `gather(ids)` back long-poll / batch flush. Intentionally
  in-process so the layer stays unit-testable with the deterministic fakes.
- The benchmark core (`bench/`) depends only on the existing `BlockEmbedder`
  and `InMemoryVectorStore` — never import LangChain, a metric library, or a
  template engine there. `RecursiveCharSplitter` (`baselines.py`) is a
  faithful, dependency-free reimplementation of the `RecursiveCharacterTextSplitter`
  (the LangChain default); it is the baseline everyone compares against, shipped
  so the benchmark is reproducible without an extra install. `BenchmarkRunner`
  (`runner.py`) builds two indexes over the *same* corpus (one vector per
  IdeaBlock vs one vector per naive chunk), runs the *same* queries (each
  block's `critical_question`, ground truth = the block's own id / the chunks
  derived from it) against both, and scores top-k retrieval (`evaluate_retrieval`
  in `metrics.py`: hit@k, MRR, mean top score) + token efficiency
  (`token_stats` / `approx_tokens` with a `len/4` heuristic, overridable via
  `token_counter=`). `BenchmarkReport.to_html()` (`report.py`) renders a
  self-contained HTML page (inlined CSS, no externals, no template engine) —
  the "prove the ROI on your own data" artifact. The comparison is fair by
  construction: same embedder, same queries, same ground truth — only the
  chunking strategy differs.
- The API orchestration core (`api/pipeline.py` → `SparkSageService`) is
  framework-agnostic — never import FastAPI or any web framework there. It wires
  the existing `MarkdownConverter` / `TextCleaner` / `IdeaBlockGenerator` together
  and owns only temp-file management for uploaded bytes (the converter backends
  detect format from the file *extension*, so the temp file must carry the
  original extension; provenance is swapped back to the original filename via
  `dataclasses.replace`). FastAPI is an optional dependency (`pip install
  'sparksage[api]'`), imported lazily only inside `api/app.py:create_app`.
  `create_app(service=...)` accepts an injected service (for tests); when omitted
  it builds one from env vars (`SPARKSAGE_API_KEY` / `OPENAI_API_KEY`). If no API
  key is set, `/generate` returns `503` while `/convert` works LLM-free. Note:
  `app.py` deliberately omits `from __future__ import annotations` so FastAPI can
  resolve the lazily-imported route-parameter types (`UploadFile`/`File`/`Form`)
  via eager annotation evaluation.
- Configuration (`config.py`) is pure stdlib — never import `python-dotenv` or
  any env-loading library. `load_dotenv()` is called once at the top of
  `build_default_service()`; it reads `.env` from the CWD but **real env vars
  always win** (`override=False`), matching 12-factor. The parser supports only
  the well-defined `.env` subset (`KEY=VALUE`, quotes, `export`, `#` comments)
  and deliberately does NOT do shell expansion (`$VAR`/`$(...)`/backticks) or
  multi-line values. `.env` is git-ignored; commit `.env.example` as a template
  only.
- Logging (`logging_config.py`) is pure stdlib and reads exactly one env var,
  `SPARKSAGE_LOG_LEVEL` (default `WARNING`), via `configure_logging()`.
  `build_default_service()` calls it right after `load_dotenv()` so `.env` and
  real env vars both feed it (12-factor). It sets the level on the `sparksage`
  logger only (never the root logger) and installs a single `StreamHandler`
  **only when nobody else has configured logging** (`root.hasHandlers()` is
  false) — so under uvicorn/gunicorn records propagate to the host's handlers
  with no duplicate output, while in a plain script `SPARKSAGE_LOG_LEVEL=DEBUG`
  Just Works. It is idempotent (never stacks handlers) and is never called on
  import (libraries must not mutate global logging state as a side effect).
  New sub-packages should `logging.getLogger(__name__)` under the `sparksage`
  namespace so they are covered by the single level.
- The query-processing core (`query/`) is the query-time counterpart of the
  ingest pipeline. It depends only on two protocols — `IntentClassifier` and
  `QueryRewriter` — and **reuses the existing `LLMClient` protocol** (never
  import a concrete LLM SDK or invent a new client abstraction there). Both
  protocols ship an LLM default (`LLMIntentClassifier` / `LLMQueryRewriter`) and
  a no-LLM rule-based alternative (`RuleIntentClassifier` / `RuleQueryRewriter`)
  for the cost-control "high-frequency patterns hit a rule first" pattern.
  `QueryProcessor` (`processor.py`) is the framework-agnostic orchestrator:
  classify → intercept (out-of-domain / below `min_confidence`) → rewrite, with
  the reject set / confidence floor / canned reply as *configuration*. The
  lenient→strict two-stage pattern is reused verbatim from `generator/`: raw LLM
  output is parsed into lenient `RawIntent` / `RawRewrite` (`extra="ignore"`),
  then coerced through the `QueryIntent` enum (`schema/enums.py` — the single
  source of truth) into strict `IntentResult` / `RewriteResult`
  (`extra="forbid"`). The prompts (`prompts.py`) read the `QueryIntent`
  vocabulary live from the enum, so extending it widens what the model may emit
  with no prompt edit. Multi-turn anaphora resolution is first-class:
  `ConversationContext` (`context.py`) is baked into the rewrite system prompt.
  This is **not** wired to the web layer yet — a future `/api/v1/query` route
  will be a thin wrapper, mirroring how `SparkSageService` wraps the ingest
  pipeline.

## Roadmap context

Implemented now: chunk schema (IdeaBlock + TechnicalBlock), LLM-driven
generation (`generator/`: prompt building, JSON extraction, enum coercion),
uniform file-to-Markdown conversion (`convert/`: pluggable backend built on
`markitdown`, single-file + resilient batch directory mode), customizable
text cleaning (`clean/`: composable `CleaningRule`s, source/filename-aware
routing via `CleaningRegistry`, sits between conversion and generation),
dense-vector embedding (`embed/`: pluggable `EmbeddingClient` Protocol,
`BlockEmbedder` fills `IdeaBlock.embedding` from `embedding_text`,
deterministic `FakeEmbeddingClient` for tests, `OpenAIEmbeddingClient` with
batching + concurrency as an optional dep; in-memory retrieval via a
`VectorStore` Protocol + brute-force `InMemoryVectorStore` kNN (pure stdlib,
consumes `vectors_for`), `save_store`/`load_store` JSON persistence so
embeddings survive restarts, and `find_similar_pairs` all-pairs near-duplicate
detection (pure stdlib, the first dependency-free step of Distill, now with a
pluggable `CandidateReducer` Protocol for million-vector corpora)), a WEB API
(`api/`: framework-agnostic `SparkSageService` orchestration + FastAPI app
factory exposing `/api/v1/convert` and `/api/v1/generate`), `.env`-based
configuration (`config.py`: zero-dependency loader, env vars override the
file), and query-time intent recognition + rewriting (`query/`: pluggable
`IntentClassifier` / `QueryRewriter` protocols reusing `LLMClient`, LLM +
rule-based implementations, lenient→strict `QueryIntent` coercion,
`QueryProcessor` orchestration with interception policy — not yet wired to
the web layer), the end-to-end Distill de-dup pipeline (`distill/`:
`DistillPipeline` with iterative threshold refinement + hierarchical LLM
merge + lifecycle write-back via `status`/`parents`/`confidence`, pure stdlib
`ClusteringBackend` (union-find) + lazy `LouvainClusteringBackend` under
`[distill]`, reusing `find_similar_pairs` + `BlockEmbedder` + `LLMClient`
via `BlockMerger`; the pure-stdlib `LSHCandidateReducer` (random-hyperplane
LSH) accelerates the pair scan at million-vector scale while keeping
precision 1.0, auto-enabled via `select_candidate_reducer` for corpora ≥ 5000;
the async job layer (`DistillJob` / `JobManager`) wraps the pipeline in a
pollable `queued → running → success | failed | timeout | cancelled` state
machine with progress callbacks + cooperative cancellation, ready for a
future `/api/v1/distill` route), and a reproducible benchmark suite (`bench/`:
`BenchmarkRunner` comparing IdeaBlock vs a dependency-free
`RecursiveCharSplitter` baseline over the same queries/ground truth, hit@k /
MRR + token efficiency, zero-dependency HTML report).
Planned next: an OpenAI-compatible API, a `/api/v1/query` route wrapping
`QueryProcessor`, a `/api/v1/distill` route wrapping `JobManager`, and a
FAISS-backed `VectorStore` under a future `[distill]` accelerator for
million-vector corpora.
Design schema additions so the Distill lifecycle fields (`status`, `parents`,
`confidence`, `embedding`) remain usable.
