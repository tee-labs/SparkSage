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
  `__contains__` / `__len__`) — never import `numpy` or `faiss` there.
  `InMemoryVectorStore`
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
- The concrete `VectorStore` backends (`embed/backends/`) each implement the same
  Protocol and lazily import their own SDK inside `__init__` (with a clear
  `ImportError` pointing at the extra), so the core stays zero-dependency —
  install only the backend you need. `FaissVectorStore`
  (`faiss_store.py`, under the `[distill]` extra via `faiss-cpu` + `numpy`) wraps
  an exact inner-product `IndexFlatIP` behind an `IndexIDMap2` so opaque string
  `block_id`s map to FAISS's int64 ids; overwrites remove + re-insert (no in-place
  update on a flat index). `ChromaVectorStore` (`chroma_store.py`, `[chroma]`
  extra) wraps a ChromaDB collection (`cosine` space; `PersistentClient` from a
  `path=`, an injected `client=`, or an ephemeral in-process client by default)
  and reports score = `1 - distance` so it matches the dot-product convention.
  `PgvectorVectorStore` (`pgvector_store.py`, `[pgvector]` extra) owns a
  Postgres `vector(d)` table over a `psycopg` (v3) connection — vectors go in as
  the pgvector text form `[a,b,c]` (no separate `pgvector` python adapter needed),
  the table name is regex-validated since it can't be SQL-parameterized, and the
  `cosine` operator (`<=>`) is the default. All three assume L2-normalized vectors
  and return `SearchHit` scores directly comparable to
  `InMemoryVectorStore.search`.
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
  chunking strategy differs. `BenchmarkRunner.run_scaling()` (`runner.py` +
  `scaling.py`) is the nested-staircase counterpart: it slices the *same*
  corpus into a geometric staircase of increasingly large subsets (each tier
  grows by `growth_factor=`, default `1.25` per the paper §3.3), keeps the
  **question set fixed** (the first `query_count` blocks' `critical_question`),
  and runs the IdeaBlock-vs-naive comparison at every tier. `ScalingReport`
  (`scaling.py`) exposes `crossover_tier(metric=)` — the first tier where the
  strategy leader changes (the scale-dependent crossover the single-tier
  benchmark cannot see), `metric_series(metric=)` for trend plotting, and a
  `to_html()` renderer. The private `_evaluate_*` methods were generalized to
  accept a `query_blocks=` param (defaults to the full corpus for the
  single-tier path) so the staircase indexes a growing background corpus while
  querying the same fixed set.
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
  The self-query decomposition (`self_query.py`) is the LLM front-end that
  finally *produces* a `RetrievalFilter` from free text: a `SelfQueryParser`
  protocol ships an `LLMSelfQueryParser` (splits a question into a clean query
  + a `RetrievalFilter` of tags / entities / languages; tag values are read
  live from the `Tag` enum into the prompt, lenient→strict coercion reusing
  `query.schema.CoercionError`, identity fallback on a bad response) and an
   `IdentitySelfQueryParser` no-op — wire it in front of `Retriever.search` and
   pass its `filter` straight through. This is wired to the web layer via
   `POST /api/v1/query`, mirroring how `SparkSageService` wraps the ingest
   pipeline.
- The keyword-extraction core (`tags/`) is the dependency-free auto-tagging
  engine: when a document arrives without tags, a `KeywordExtractor` derives
  them from the content using classic algorithms — `RakeKeywordExtractor`,
  `TfidfKeywordExtractor`, `TextRankKeywordExtractor` (all pure stdlib).
  It depends only on the `Tokenizer` Protocol (`tokenizer.py`) and the stop-word
  sets in `stoplist.py` — never import `jieba`, NLTK, or spaCy there. CJK works
  out of the box via the dictionary-free `CharBigramTokenizer` (overlapping
  character bigrams carry strong topical signal); word-level Mandarin
  segmentation is the optional `JiebaTokenizer` (`pip install
  'sparksage[tags-zh]'`), imported lazily inside its `__init__` like every other
  optional SDK. `AutoTokenizer` inspects the text and routes CJK → bigrams,
  Latin → whitespace, and is the default tokenizer of every extractor. Tags are
  **free-form** (`KeywordScore.keyword` → `list[str]` on the document) —
  intentionally *not* the closed `Tag` enum, which keeps its coarse-grained
  semantic-filtering role. `make_extractor("rake"|"tfidf"|"textrank")` is the
  config-driven factory (unknown names fail fast). Bigrams are a *scoring*
  feature, not a word: the cohesion filter (`cohesion.py`, pure stdlib) stops
  cross-boundary CJK bigrams (`凌晨一点执行` → `晨一` / `点执`) from surfacing as
  tags. `blessed_cjk_bigrams` combines a bidirectional-conditional-probability
  cohesion floor (`f(ab)/max(f(a),f(b))`) with a per-CJK-run maximum-weight
  non-overlap selection (DP, weighted by cohesion) so the densest real-word
  segmentation wins; the extractors drop non-blessed bigrams at the token level
  (cleaning both tag output *and* TF-IDF / TextRank scoring). `min_cohesion=`
  on every extractor / `make_extractor` (default `DEFAULT_MIN_COHESION = 0.34`,
  `None` disables); `SPARKSAGE_AUTO_TAG_MIN_COHESION` is the env knob (`off` →
  disabled). Word-perfect Mandarin still needs `jieba`; this is the no-dependency
  fallback that removes the scatter.
- The document-management core (`documents/`) is the document-level counterpart
  of `schema/` — there was no *document* object, only chunk-level
  `IdeaBlock`s. `DocumentRecord` (`models.py`, Pydantic v2, `extra="forbid"`)
  carries `title` / `summary` / `body_markdown` / free-form `tags: list[str]`
  / `SourceRef` provenance / timestamps / `content_hash`; tags are de-duplicated
  and stripped on validation. The storage layer depends only on the
  `DocumentStore` Protocol (`store.py`: `save`/`get`/`list`/`delete`/`count`/
  `list_tags`/`__contains__`/`__len__`) — never import `sqlite3`-specific SQL in
  the core. `InMemoryDocumentStore` (`backends/memory.py`) is pure stdlib
  (dict-backed, defensive copies on read/write); `SqliteDocumentStore`
  (`backends/sqlite.py`, also stdlib `sqlite3`, no server) owns a `documents`
  table + a `<table>_tags` junction table for exact-match tag filtering, opened
  `check_same_thread=False` behind a `threading.Lock` (safe across the FastAPI
  threadpool), table name regex-validated since it can't be SQL-parameterized.
  `ExtractiveSummarizer` (`summarizer.py`) produces the document-level summary:
  frequency-scored sentences returned in original order, Markdown heading /
  emphasis markers stripped — depends only on the same `Tokenizer` Protocol.
- The API orchestration core (`api/pipeline.py` → `SparkSageService`) is
  framework-agnostic — never import FastAPI or any web framework there. It wires
  the existing `MarkdownConverter` / `TextCleaner` / `IdeaBlockGenerator` together
  and owns only temp-file management for uploaded bytes (the converter backends
  detect format from the file *extension*, so the temp file must carry the
  original extension; provenance is swapped back to the original filename via
  `dataclasses.replace`). It now also owns document management: optional
  `document_store` (lazily an `InMemoryDocumentStore` so ingest works with zero
  config), `keyword_extractor` (lazily RAKE), and `summarizer` (lazily
  extractive) constructor params; `ingest_document` runs convert → clean →
  (auto-tag when no tags) → (extractive summary) → store, and `list_documents`
  / `get_document` / `update_document` / `delete_document` / `retag_document`
  / `count_documents` / `list_document_tags` cover the CRUD + tag vocabulary.
  `auto_tag` decomposes over-long phrases into tag-shaped single words. FastAPI
  is an optional dependency (`pip install 'sparksage[api]'`), imported lazily
  only inside `api/app.py:create_app`. `create_app(service=...)` accepts an
  injected service (for tests); when omitted it builds one from env vars
  (`SPARKSAGE_API_KEY` / `OPENAI_API_KEY`, plus `SPARKSAGE_DOC_STORE` → a durable
  `SqliteDocumentStore`, `SPARKSAGE_AUTO_TAG_EXTRACTOR` = rake|tfidf|textrank,
  `SPARKSAGE_TAGS_ZH` → jieba). If no API key is set, `/generate` returns `503`
  while `/convert`, `/documents`, and `/tags` work LLM-free. Note: `app.py`
  deliberately omits `from __future__ import annotations` so FastAPI can
  resolve the lazily-imported route-parameter types (`UploadFile`/`File`/`Form`)
  via eager annotation evaluation.
- The async ingest job layer (`api/ingest_jobs.py`) is the engineering fix for
  the "5-file upload, second file showed failed but the backend finished" bug:
  a long ingest (convert → LLM generate → embed → index, minutes on a large
  doc) no longer holds open an HTTP connection to race a client-side timeout.
  `IngestJob` / `IngestJobManager` mirror the `distill/job.py` shape (the same
  `queued → running → success | failed | cancelled` state machine + lock-
  protected frozen `IngestJobSnapshot` + cooperative cancellation) but stay
  self-contained in the `api` package — ingest observability is an HTTP-layer
  concern, not a distill concern. The job owns no ingest config; it takes a
  fully-bound `work` callable produced by `QAService.submit_ingest`.
  `QAService.ingest_and_index` gained optional `on_progress` / `is_cancelled`
  params so the job can surface coarse phase progress (`converting` /
  `generating` / `indexing`) and abort at a phase boundary *before* the
  knowledge-base write — the cooperative `IngestCancelled` exception inverts
  the old "client gone but server still wrote" dirty-write bug (a cancelled
  ingest leaves the KB untouched). `QAService.submit_ingest` validates eagerly
  (generator configured, `kb_id` resolvable) so a bad request fails fast
  rather than producing a job that immediately errors. Wired to the web layer
  via `POST /api/v1/knowledge_base/ingest/async` (returns a `job_id`
  immediately, HTTP 202), `GET /api/v1/jobs/{job_id}` (pollable snapshot; the
  terminal-success poll carries the full `IngestAndIndexResponse` in its
  `result` field so the client gets the generated blocks in the same final
  poll — no second round-trip), and `POST /api/v1/jobs/{job_id}/cancel`
  (cooperative). The React `IngestPage` uses this async path for the
  "入库" toggle: each file is submitted + polled independently with single-
  file try/catch isolation so one failure no longer aborts the rest, and a
  per-file progress card (`phase` / `percent` / `success`/`failed`/`cancelled`)
  replaces the all-or-nothing spinner. The sync `POST .../ingest` route is
  unchanged (backward-compatible). Content updates live on the same KB path:
  `QAService.update_document_and_reindex` (`api/qa_service.py`) is the
  hash-aware update counterpart of `ingest_and_index` — convert →
  (re)generate blocks → `KnowledgeBase.update_document`, keeping `doc_id`
  stable; when the new body's `content_hash` equals the stored one it patches
  title/tags *without* re-running the LLM or re-embedding, so re-uploading the
  same file is cheap (shared generation runs through the private
  `_parallel_generate` helper). Wired to the web layer via `PUT
  /api/v1/knowledge_base/documents/{doc_id}` (multipart file + optional
  title/tags, returns the same `IngestAndIndexResponse` as ingest); metadata
  updates remain `PATCH /api/v1/documents/{id}`. Idempotent wiki-style sync is
  the external-id bridge: `DocumentRecord` / `new_record` gained
  `external_key` (a deterministic upstream id like `wiki:123`; persisted as a
  column in `SqliteDocumentStore` with an ALTER-TABLE migration for legacy
  DBs), `QAService.upsert_document` (`api/qa_service.py`) is the
  external-key-keyed counterpart of ingest/update — no doc with the key →
  ingest (`action="created"`), same `content_hash` → metadata-only patch
  (`action="unchanged"`, zero LLM cost), body changed →
  `update_document_and_reindex` keeping `doc_id` stable (`action="updated"`),
  with `find_by_external_key` / KB-scoped `list_documents` powering the
  deletion-detection diff. `ingest_and_index` / `update_document_and_reindex`
  also pass through `external_key` / `metadata` / `source.system` + `extra`
  (provenance no longer dropped on ingest/update) so citations chain back to
  the upstream wiki page. Exposed as `POST
  /api/v1/knowledge_base/documents/upsert` (multipart + `external_key` form
  field → `UpsertResponse` with `action` / `doc_id` / `block_count`) and `GET
  /api/v1/knowledge_base/documents` (KB-scoped `DocumentListResponse`
  serializing `external_key`).
- The configuration management (`api/config_manager.py`) is the pure-stdlib
  bridge between the WEB UI's `/config` page and the `.env` file. It depends
  only on `sparksage.config.parse_env_file` (never a third-party env loader).
  `read_config` reports the *effective* value of every known key (a real env
  var wins over the file, exactly like `load_dotenv`), masking any key ending in
  `_API_KEY` as `"****"` so the response is browser-safe. `write_config` is a
  *patch* (only supplied keys are touched; comments, ordering and unknown keys
  are preserved), rejects unknown key names outside the `SPARKSAGE_*` /
  `OPENAI_*` / `*_API_KEY` surface, and treats a `"****"` secret value as a
  no-op so a GET→POST round-trip never clobbers a secret with the mask. It also
  writes through to `os.environ` (12-factor: never overriding a real env var)
  so the running process stays in sync until the manual restart.
- The static frontend serving (`_mount_static_frontend` in `app.py`) lets the
  built React + Ant Design WEB UI ship from the same FastAPI origin. When a
  Vite build is found (`SPARKSAGE_WEB_DIST`, or `web/dist` next to the CWD / the
  repo root), `/assets` is mounted straight from disk and a catch-all route
  (registered *last*, so every `/api/...` route wins first) serves the SPA
  `index.html` for non-API GET paths. Any method on an unknown `/api/` path
  returns a real `404` (not a `405`) so API consumers get a clean "not found".
  The `web/` app (Vite + React 18 + TS + antd 5) covers 7 demo pages:
  `/config`, `/knowledge-bases`, `/ingest`, `/documents`, `/knowledge-base`,
  `/qa`, `/feedback` (left-side collapsible antd `Menu`); the Docker image
  builds it in a `node:20` stage and copies `web/dist` into `/app/web/dist`.
- The retrieval core (`retrieve/`) is the query-side counterpart of the ingest
  pipeline and the layer that finally *consumes* the three "designed but
  unconsumed" IdeaBlock fields. It depends only on the existing `VectorStore`
  / `BlockEmbedder` protocols plus two new ones (`LexicalRetriever`,
  `Reranker`) — never import a search engine or a rerank SDK in the core.
  `BM25Retriever` (`lexical.py`, pure stdlib) is the sparse half of hybrid
  search: each block becomes a BM25 document whose token bag weights the
  curated `keywords` field (the field the schema documents as "for BM25 /
  lexical recall boosting") plus the answer/question/name text; CJK is
  tokenized into unigrams + overlapping bigrams (dictionary-free, like the
  `tags` tokenizer). `reciprocal_rank_fusion` (`fusion.py`) is the
  score-free RRF merge of dense + lexical (or multi-query) ranked lists; it
  also supports *weighted* RRF (the WeKnora-style `w_i / (k + rank)`) via a
  `weights=` arg — only the weight *ratio* affects the ordering, so the
  equal-weight default (`None`) reproduces the original score-free RRF exactly,
  and `tune_rrf_k` / `tune_rrf_weights` empirically pick the smoothing constant
  and the dense/lexical split on labelled data. The
  `Reranker` protocol (`reranker.py`) ships an `LLMReranker` (reuses the
  existing `LLMClient`, lenient→strict index-list coercion, identity fallback
  on a bad response) and an `IdentityReranker` no-op. `Retriever`
  (`orchestrator.py`) wires it together: dense (kNN) + optional lexical →
  weighted RRF fuse (`dense_weight` / `lexical_weight`) → `RetrievalFilter`
  post-filter (`tags`/`entities`/`language`/ `kb_id`/`block_ids`, applied
  against a block registry since the store is text-agnostic) → optional rerank
  → optional score-floor guard (`min_score` + WeKnora decayed-retry /
  top-`0.15` fallback, which stops weak/irrelevant blocks from filling the
  top-`k`; applied on the rerank path and the dense-only cosine path, skipped
  on an un-reranked RRF score since RRF scores are not on an absolute scale) →
  top-k `RetrievedChunk`s. `RetrievedChunk`
  / `Citation` (`models.py`) surface `source.uri` + `source.locator` — the
  provenance the reader grounds citations in. The filter is a *post-filter*
  over an over-fetched dense pool (the store is deliberately text-agnostic);
  swap in a backend with native metadata filtering for exact filtered kNN.
  The concrete `Reranker` backends (`retrieve/backends/`) each implement the
  same Protocol and lazily import their own SDK inside `__init__` (with a clear
  `ImportError` pointing at the extra), so the core stays zero-dependency —
  install only the backend you need. `CrossEncoderReranker`
  (`cross_encoder.py`, under the `[rerank]` extra via `sentence-transformers`)
  wraps a `CrossEncoder` (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`, or
  `BAAI/bge-reranker-v2-m3` for CJK/multilingual) and re-scores the fused pool
  in one cross-attention pass per pair — the single largest point lever after
  chunking strategy, and an order of magnitude cheaper per query than
  `LLMReranker`; raw logits are squashed through a stable sigmoid into `(0, 1)`
  by default (`apply_sigmoid=`) so scores match the other rerankers' shape.
- The reader core (`reader/`) is the answer-generation stage — the missing
  "right half" of the QA pipeline. It depends only on two new protocols,
  `AnswerGenerator` and `FaithfulnessJudge`, both reusing the existing
  `LLMClient` (never invent a new LLM abstraction there). `LLMAnswerGenerator`
  (`generator.py`) feeds the model each candidate's `critical_question` +
  `trusted_answer` (the IdeaBlock QA-alignment dividend) and emits a JSON
  answer with citations referencing block ids; the lenient→strict coercion
  (`schema.py`, mirroring `query/schema.py`) binds those ids to the schema's
  `source.uri`/`source.locator` and *drops hallucinated* ids not in the
  retrieved set. `LLMFaithfulnessJudge` (`faithfulness.py`) scores how well
  the answer is supported (LLM-as-judge, degrades to a default on a bad
  response). `Reader` (`orchestrator.py`) runs generate → (judge) → abstain:
  below `min_faithfulness` or `min_confidence` it returns the abstention
  reply instead of hallucinating — the symmetric answer-side gate to
  `QueryProcessor`'s query-side `min_confidence`. `Reader` also owns the
  Context-Cliff guard: an optional `max_context_tokens` trims the best-first
  chunk list to a token budget (`reader/budget.py:trim_to_token_budget`, pure
  stdlib `len/chars_per_token` heuristic mirroring `bench.approx_tokens`, with a
  pluggable `token_counter=` for an exact tokenizer and a `keep_min` floor) *before*
  generation and judging, so the judge scores the answer against exactly the
  context the generator saw — preventing the "lost in the middle" degradation
  that a generous top-k otherwise causes.
- The QA engine (`qa/`) is the framework-agnostic orchestrator that finally
  makes SparkSage an end-to-end question-answering core. `QAEngine`
  (`engine.py`) wires query → retrieval → answer with no business logic of its
  own — every stage is a swappable protocol (`QueryProcessor` / `QueryExpander`
  / `Retriever` / `Reader` / `QACache`, all optional). It consumes the
  rewriter's `sub_queries` (COMPARISON / multi-hop decomposition) and the
  expander's variants via the same RRF-fused multi-retrieve path. An optional
  `QACache` (a `lookup`/`store` protocol the `query.SemanticCache` implements)
  short-circuits the whole pipeline for near-duplicate repeat queries. Two
  Phase-5 self-correction layers are optional knobs: (a) **intent→KB routing**
  — an `intent_router: Callable[[IntentResult], str | None]` (see the callable
  `IntentKBRouter` mapping `QueryIntent → kb_id`) merges a classified intent
  into `RetrievalFilter.kb_id` before retrieval, finally connecting the existing
  `IntentClassifier` to the existing `KnowledgeBase` multi-tenancy (a per-call
  `filter.kb_id` always wins); (b) the **self-reflective retrieval loop** — an
  optional `RetrievalGrader` (`retrieve/grader.py`, the retrieval-side
  counterpart of `FaithfulnessJudge`) scores chunk relevance after each
  retrieval, and below `min_relevance` an optional `QueryRefiner`
  (`query/refiner.py`) refines the query and the engine re-retrieves (up to
  `max_iterations`, keeping the best-graded result so refinement can never lower
  quality). This is the middle gate of the symmetric three-stage policy:
  query-side `min_confidence` → retrieval-side `min_relevance` → answer-side
  `min_faithfulness`. `QAResult` now carries `relevance` / `refined_query` /
  `iterations` for transparency. Wired to the web layer via `POST /api/v1/query`
  (a thin wrapper around `QAEngine.ask`), exactly as `SparkSageService` wraps
  ingest.
- The agentic QA core (`agent/`) is a *different orchestrator* over the *same*
  building blocks as `qa/` — it turns SparkSage from a one-shot RAG (a static
  question→answer mapping) into an **Agentic RAG** with an LLM-driven
  plan-act-observe-synthesize loop, the mode that handles the three problem
  classes one-shot RAG cannot: multi-hop reasoning, conditional filtering, and
  comparative analysis. `AgenticQAEngine` (`engine.py`) reuses the existing
  `Retriever`, `Reader`, and `QueryProcessor` unchanged and adds exactly one new
  protocol, `AgentController` (`controller.py`: `next_action(state) → AgentAction`),
  which makes the loop *paradigm-pluggable* — `LLMAgentController` drives a
  ReAct loop (thought → retrieve-another-sub-query | plan | synthesize), and
  `IdentityController` is the no-op degenerate controller (always synthesize) so
  `AgenticQAEngine(IdentityController())` collapses to the single-shot `QAEngine`
  baseline (the "off as a uniform protocol object" convention). It depends only
  on the existing `LLMClient` (never a concrete SDK) and reuses the lenient→strict
  pattern (`RawAgentAction` → `coerce_action` over a closed `ActionType` enum,
  `agent/schema.py`) so the enum stays the single source of truth; on a bad
  controller response it degrades to a synthesize action rather than aborting a
  multi-step run (`strict=True` to raise). The loop always runs a seed retrieval
  first (so the controller is consulted with evidence in hand and an empty corpus
  abstains cleanly through the reader), then iterates up to `max_iterations`
  *extra* controller-decided steps (`PLAN` or `RETRIEVE`), merging evidence
  de-duplicated by block id (best score kept, capped at `max_evidence`), and
  finally synthesizes via the `Reader` (which still enforces its
  faithfulness/confidence abstention gate — a starved agent says "I don't know",
  never hallucinates). It reuses the canonical long-running-job shape from
  `DistillPipeline` (`on_progress` callback firing thinking → retrieving →
  synthesizing → done phases + a cooperative `is_cancelled=` predicate); the
  `on_progress` hook is also plumbed through `QAService.ask` →
  `POST /api/v1/query` (`AskRequest.stream=true`, `mode="agent"`) as an SSE
  stream of `progress` events terminated by a single `result` event (so a
  long agent run streams its ReAct phases to the UI instead of a blocking
  spinner). The agent loop is no longer an "island" divorced from the
  right-half reflection components: `AgenticQAEngine.__init__` accepts optional
  `retrieval_grader` / `query_refiner` / `query_expander` (the existing
  protocols, finally connected) — each per-step retrieval is **expanded**
  (multi-query / HyDE → RRF via the shared `retrieve/multi_query.py` helper,
  the same one `QAEngine._multi_retrieve` delegates to), then **graded**
  (`ISREL` gate), then on a low score **refined + re-retrieved** (CRAG / Self-
  RAG, keeping the best-graded result so refinement can never lower quality,
  bounded by `step_max_refine`); the per-step `RelevanceResult` + refined
  query ride on `AgentStep` and surface through `AgentStepOut` for trajectory
  transparency. The `ActionType` enum gained a third member, `PLAN`
  (Plan-and-Execute): a controller emits `sub_queries: [...]`, the engine
  enqueues them and **drains the queue one per iteration without consulting
  the controller between them** (a single PLAN commits to retrieving every
  queued sub-query up to `max_iterations`), then re-consults the controller
  — the missing decomposition layer for comparison / multi-hop questions; the
  controller system prompt (`prompts.py`) documents both `plan` and the
  optional per-step `filter` (tags / entities / languages / kb_id, coerced
  leniently with unknown tags dropped rather than failing). `AgentResult`
  (`models.py`) exposes the same `query` / `text` / `citations` / `abstained`
  / `answer` / `retrieval` surface as `QAResult` (plus the `steps` /
  `evidence` / `iterations` trajectory), so the HTTP `AskResponse` serializer
  (`_to_ask_response`) works unchanged. Wired into `QAService.ask(mode=...)`
  (`"default"` | `"agent"`, the latter lazily builds a per-KB `AgenticQAEngine`
  sharing the per-KB `Retriever` + the shared `Reader` / `QueryProcessor` +
  a service-level `agent_controller` + the optional service-level
  `agent_retrieval_grader` / `agent_query_refiner` / `agent_query_expander`),
  exposed as the `mode` field on `POST /api/v1/query` (`AskRequest.mode`);
  `build_qa_service` auto-wires an `LLMAgentController` (reusing the LLM
  client) whenever an API key is set, plus the three reflection components
  (`LLMRetrievalGrader` / `LLMQueryRefiner` / `HyDEExpander`) — disable with
  `SPARKSAGE_AGENT_REFLECTION=off`, tune with
  `SPARKSAGE_AGENT_STEP_MIN_RELEVANCE` / `SPARKSAGE_AGENT_STEP_MAX_REFINE` —
  so `mode="agent"` works out of the box, bounded by
  `SPARKSAGE_AGENT_MAX_ITERATIONS`.
- The multi-query retrieve helper (`retrieve/multi_query.py`) is the shared
  recall-boost path finally extracted so both `QAEngine` (sub-query
  decomposition / multi-query expansion) and the agent's per-step retrieval
  share one implementation: retrieve each variant -> RRF-fuse the ranked
  lists -> rebuild `RetrievedChunk`s carrying the fused + dense + lexical
  scores. Pure stdlib beyond `Retriever` and `reciprocal_rank_fusion`.
- The knowledge-base core (`kb/`) is the multi-tenant aggregate root — the
  organizational entity the flat `documents/DocumentStore` lacked. `KnowledgeBase`
  (`knowledge_base.py`) owns documents + their IdeaBlocks + a dense `VectorStore`
  + a `BM25Retriever` + a `Retriever`, and crucially the **consistency**
  between them: `add_blocks` stamps `kb_id` and embeds+indexes; `remove_document`
  cascades to block vectors + registry (index↔storage consistency guarantee);
  `update_document` is an incremental re-index only when `content_hash` changed
  (hash-aware change detection); `reindex` rebuilds both indexes from the live
  registry (drift recovery). Each block carries an optional additive `kb_id`
  (`schema/ideablock.py`) so `RetrievalFilter` can scope retrieval to one KB.
  `contains_document` is the KB-scoped membership check (the document store may
  be shared across KBs, so existence in the store is not ownership).
  `KnowledgeBaseInfo` (`models.py`) is the serializable metadata; the
  `KnowledgeBaseStore` Protocol + `InMemoryKnowledgeBaseStore` (`store.py`) is
  the multi-tenant registry — live vector state stays on the aggregate.
- The query enhancements (`query/expander.py` + `query/cache.py` +
  `query/refiner.py`) extend query understanding with multi-query expansion, a
  semantic cache, and self-corrective refinement. The `QueryExpander` protocol
  ships an `LLMQueryExpander` (n paraphrase variants for RRF-fused recall,
  lenient→strict, identity fallback), a `HyDEExpander` (Hypothetical Document
  Embeddings — drafts a hypothetical answer and retrieves against it; lands in
  the `trusted_answer` semantic space IdeaBlock is embedded from, so
  "question→hypothesis→real answer" beats "question→question" for short / vague
  queries; only fires below a configurable word count to avoid hallucination on
  long queries), and an `IdentityExpander` no-op. `InMemorySemanticCache`
  (`cache.py`, pure stdlib) keys on query *meaning* via an `EmbeddingClient`
  (cosine ≥ threshold) and implements the `QACache` protocol structurally — the
  biggest cost lever, since the LLM calls dominate. `QueryRefiner`
  (`refiner.py`) is the self-reflective-retrieval companion: an `LLMQueryRefiner`
  rewrites the query using the retrieval grader's low-relevance feedback
  (lenient→strict, identity fallback) and an `IdentityRefiner` no-op; it depends
  only on the relevance score + reasoning, never on retrieved chunks, so the
  `query` package stays free of any `retrieve` dependency. All are optional
  knobs on `QAEngine`.
- The feedback core (`feedback/`) closes the query→ingest loop (the Phase-4
  flywheel). `FeedbackRecord` (`models.py`, Pydantic v2, `extra="forbid"`,
  closed `FeedbackRating` enum) captures the user's verdict on a surfaced
  answer (positive / negative / corrected) plus optional correction and the
  backing block ids. The `FeedbackStore` Protocol + `InMemoryFeedbackStore`
  (`store.py`) persist + aggregate (approval ratio, per-block breakdown).
  `extract_healing_signals` (`healing.py`, pure stdlib) turns the aggregate
  back into ingest actions: repeated low-recall queries flag a coverage gap
  (re-chunk / new content); blocks with a high bad-feedback ratio become split
  candidates (the inverse of the Distill *merge*).
- The QA conversation-history core (`qa/history.py`) is the persisted query
  log that keeps the Q&A page's turns as durable as the feedback ratings: the
  demo UI used to keep history in frontend state only, so a refresh wiped it
  while `feedback/` kept the rated answers — the asymmetry this core fixes.
  `QATurn` (Pydantic v2, `extra="forbid"`, closed `TurnRole` enum) is one
  question (`role=user`) or answer (`role=assistant`), the assistant turn
  carrying the full serialized HTTP `AskResponse` payload (`result`) so a UI
  can re-render citations / retrieved chunks / confidence without re-running
  the pipeline. The `QASessionStore` Protocol + `InMemoryQASessionStore`
  (`history.py`, pure stdlib) store turns newest-first with optional `kb_id`
  scoping and defensive copies, mirroring `feedback/store.py`. Recording lives
  in `QAService.ask` (via the canonical `_to_ask_response` serialization), so
  `GET /api/v1/query/history` / `DELETE /api/v1/query/history` restore / clear
  the conversation across reloads, and the Q&A page reloads it per selected KB.
- The evaluation core (`eval/`) is the answer-correctness counterpart of
  `bench/` (which scores retrieval alone). `QAEvaluator` (`evaluator.py`) runs
  a `QAEngine` over a `QATestCase` set and rolls per-case outcomes into a
  `QAEvalReport`: mean answer correctness, abstention rate, retrieval hit@k
  (reuses `bench.evaluate_retrieval` for comparability), mean faithfulness.
  Correctness is a pluggable `CorrectnessJudge`: the default `TokenOverlapJudge`
  is dependency-free token-F1 (fully offline); `LLMCorrectnessJudge` swaps in
  (reuses `LLMClient`, token-F1 fallback on a bad response). The adversarial
  robustness module (`eval/distractors.py`) is the "can the retriever resist
  traps?" counterpart: `DistractorInjector` generates **trap blocks** — blocks
  that mimic a target's `name` + `critical_question` (so dense retrieval finds
  them) but carry a **wrong** `trusted_answer` borrowed from a semantically
  similar donor (found via the embedder's cosine, no LLM needed); donors with
  identical answers are skipped. `RobustnessEvaluator` injects traps into the
  corpus, builds both the IdeaBlock and the naive-chunk index (the same A/B
  comparison `BenchmarkRunner` uses), queries each target's
  `critical_question`, and measures how often traps contaminate the top-k.
  `RobustnessReport` reports the **true-hit rate** vs **trap-contamination
  rate** for both strategies plus `trap_resistance_improvement` (how many times
  lower the IdeaBlock contamination is), directly quantifying the
  `trusted_answer` dividend that naive chunks cannot guarantee. Pure stdlib
  beyond `BlockEmbedder` / `InMemoryVectorStore` / `RecursiveCharSplitter`.

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
pluggable `CandidateReducer` Protocol for million-vector corpora), production
`VectorStore` backends (`embed/backends/`: `FaissVectorStore` exact IP index
under `[distill]`, `ChromaVectorStore` collection under `[chroma]`,
`PgvectorVectorStore` Postgres+pgvector table under `[pgvector]` — each lazily
imports its own SDK so the core stays zero-dependency, all return dot-product-
comparable scores)), a WEB API
(`api/`: framework-agnostic `SparkSageService` orchestration + FastAPI app
factory exposing `/api/v1/convert` and `/api/v1/generate`), `.env`-based
configuration (`config.py`: zero-dependency loader, env vars override the
file), and query-time intent recognition + rewriting (`query/`: pluggable
`IntentClassifier` / `QueryRewriter` protocols reusing `LLMClient`, LLM +
rule-based implementations, lenient→strict `QueryIntent` coercion,
  `QueryProcessor` orchestration with interception policy — now wired to the
  web layer via `POST /api/v1/query`), the end-to-end Distill de-dup pipeline (`distill/`:
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
MRR + token efficiency, zero-dependency HTML report), a dependency-free
auto-tagging engine (`tags/`: `KeywordExtractor` Protocol with pure-stdlib
RAKE / TF-IDF / TextRank over the `Tokenizer` Protocol + `stoplist.py`, CJK via
dictionary-free `CharBigramTokenizer`, optional `JiebaTokenizer` under
`[tags-zh]`; free-form tags, not the closed `Tag` enum), and a document-
management service (`documents/`: `DocumentRecord` Pydantic model with free-form
`tags: list[str]` + `summary` + `content_hash`, `DocumentStore` Protocol with
pure-stdlib `InMemoryDocumentStore` + durable `SqliteDocumentStore` (stdlib
`sqlite3`, junction-table tag filtering), `ExtractiveSummarizer`; wired into
`SparkSageService.ingest_document` → convert → clean → auto-tag → summarize →
store, with CRUD + `retag_document` + tag vocabulary, exposed as
`/api/v1/documents` (POST/GET/PATCH/DELETE) and `/api/v1/tags` routes —
`SPARKSAGE_DOC_STORE` opts into durable SQLite storage; all LLM-free), and the
full query-side "right half" that turns SparkSage into an end-to-end QA core:
hybrid retrieval (`retrieve/`: pure-stdlib `BM25Retriever` over the curated
`keywords` field + dense kNN, `reciprocal_rank_fusion`, `LLMReranker` +
`IdentityReranker`, `Retriever` orchestrator with `RetrievalFilter`
tag/entity/language/kb_id scoping and `RetrievedChunk`/`Citation` provenance),
answer generation (`reader/`: `LLMAnswerGenerator` over the QA-aligned
`critical_question`+`trusted_answer` context with citation binding to
`source.locator`, `LLMFaithfulnessJudge`, `Reader` with abstention gate), an
end-to-end `QAEngine` (`qa/`: query → retrieval → answer, multi-query /
sub-query RRF-fused retrieval, optional `QACache`), the multi-tenant
`KnowledgeBase` aggregate root (`kb/`: documents + blocks + consistent dense
+ lexical index, hash-aware `update_document`, `reindex`, `kb_id` scoping,
`KnowledgeBaseStore` registry), query enhancements (`query/expander.py`
multi-query expansion + `query/cache.py` embedding-keyed `SemanticCache`),
the feedback flywheel (`feedback/`: `FeedbackRecord` + `FeedbackStore` +
`extract_healing_signals` for coverage-gap / split-candidate signals back to
ingest), the persisted QA conversation log (`qa/history.py`: `QATurn` +
`QASessionStore` recording every ask in `QAService.ask`, restored via
`GET /api/v1/query/history` so the Q&A page keeps its turns across reloads),
and answer-correctness evaluation (`eval/`: `QAEvaluator` over a
`QATestCase` set, pluggable `TokenOverlapJudge` / `LLMCorrectnessJudge`,
reusing `bench.evaluate_retrieval` for the retrieval metric). Also implemented
now: the nested-tier scaling benchmark (`bench/scaling.py`:
`BenchmarkRunner.run_scaling()` slices the same corpus into a geometric
staircase of increasingly large subsets — each tier grows by
`growth_factor=1.25` per the paper §3.3 — keeping the question set fixed and
running the IdeaBlock-vs-naive comparison at every tier, exposing the
scale-dependent crossover the single-tier benchmark cannot see;
`ScalingReport.crossover_tier(metric=)` finds where the strategy leader
changes) and adversarial distractor injection (`eval/distractors.py`:
`DistractorInjector` generates trap blocks that mimic a target's question but
carry a wrong answer borrowed from a semantically similar donor;
`RobustnessEvaluator` injects traps and measures true-hit rate vs
trap-contamination rate for both strategies, quantifying the `trusted_answer`
dividend). Also implemented now: a cross-encoder re-ranking backend (`retrieve/backends/cross_encoder.py`:
`CrossEncoderReranker` under `[rerank]` via `sentence-transformers`, sigmoid-
normalized logits, the largest point lever after chunking strategy), the
Context-Cliff guard (`reader/budget.py`: `trim_to_token_budget` wired into
`Reader.max_context_tokens`, pure-stdlib token heuristic + pluggable tokenizer,
applied before generation *and* judging), and self-query decomposition
(`query/self_query.py`: `SelfQueryParser` protocol + `LLMSelfQueryParser`
producing a `RetrievalFilter` from free text, tag values read live from the
`Tag` enum, lenient→strict coercion, identity fallback), the self-reflective
retrieval loop (`retrieve/grader.py` `RetrievalGrader` + `query/refiner.py`
`QueryRefiner` wired into `QAEngine`: relevance grade → query refine →
re-retrieve, best-graded result kept, the symmetric middle gate between
query-side `min_confidence` and answer-side `min_faithfulness`), HyDE
(`query/expander.py:HyDEExpander`, hypothetical-answer retrieval into the
`trusted_answer` space, short-query gated), and intent→KB routing
(`qa/engine.py:IntentKBRouter`, classifies intent → `RetrievalFilter.kb_id`).
The three
previously "designed but unconsumed" IdeaBlock fields (`keywords`,
`entities`/`tags`, `source.locator`) are now all wired into retrieval /
filtering / citations.
Also implemented now: multi-knowledge-base management at the web layer
(`api/qa_service.py:QAService` holds a registry of `KnowledgeBase` aggregates
keyed by `kb_id` backed by a `KnowledgeBaseStore`, with an "active" KB as the
default routing target and per-call `kb_id=` on `ingest_and_index` / `ask` /
`list_blocks` / feedback; `api/app.py` exposes `POST/GET/DELETE
/api/v1/knowledge_bases` + `POST /api/v1/knowledge_bases/{kb_id}/activate`, the
ingest route takes a `kb_id` form field, and `AskRequest.kb_id` routes the
query to the right KB's lazily-built `QAEngine`; the React UI adds a
`/knowledge-bases` management page and a shared `KbSelector` on ingest / QA /
browse pages).
 Also implemented now: durable persistence so a Docker restart loses nothing
 (`SPARKSAGE_DATA_DIR`, defaulting to `/app/data` in the image, is the one-knob
 default). `KnowledgeBaseStore` / `FeedbackStore` each gained a stdlib-only
 `Sqlite*` counterpart (`kb/backends/sqlite.py`, `feedback/backends/sqlite.py`)
 mirroring the `SqliteDocumentStore` pattern (regex-validated table, `check_same_thread=False`
 + `threading.RLock`, defensive copies, persists across instances). A new
 `KbStateStore` Protocol (`kb/backends/state.py`) + `SqliteKbStateStore` persists
 the live block registry + dense vectors (each block's `embedding` rides along
 in its JSON, so restart never re-calls the embedding API) + document<->block
 linkage; `KnowledgeBase.__init__` takes an optional `state_store=` and writes
 through on every mutation (`add_blocks` / `remove_block` / `remove_document`),
 then hydrates the registry + vectors + lexical index on construction.
 `QAService` accepts `kb_store=` / `state_store=` / `feedback_store=` and
 reloads every persisted KB on startup (`_reload_persisted_kbs`), so the active
 KB id + indexed knowledge survive a restart. `build_qa_service` /
 `build_default_service` wire all four durable backends from `SPARKSAGE_DATA_DIR`
 (individual `SPARKSAGE_*_STORE` paths override); a Docker `VOLUME /app/data`
 mount is all a user needs. The Dockerfile sets `SPARKSAGE_DATA_DIR=/app/data`
 and declares the volume.

Also implemented now: agentic QA — a *second* QA mode that turns SparkSage from
a one-shot RAG into an Agentic RAG. The new `agent/` package is a different
orchestrator over the same `Retriever` / `Reader` / `QueryProcessor` building
blocks, adding exactly one protocol (`AgentController`: `LLMAgentController`
ReAct loop + `IdentityController` single-shot fallback) on top of the existing
`LLMClient`, with lenient→strict action coercion over a closed `ActionType`
enum (`agent/schema.py`). `AgenticQAEngine` runs a bounded plan-act-observe-
synthesize loop (seed retrieval → up to `max_iterations` controller-decided
retrievals → `Reader` synthesis), merging evidence de-duplicated by block id,
reusing the `DistillPipeline` long-run shape (`on_progress` +
`is_cancelled=`), and never hallucinating (a starved agent abstains through the
reader). `AgentResult` is shape-compatible with `QAResult`, so
`QAService.ask(mode="agent")` + the `mode` field on `POST /api/v1/query`
(`AskRequest.mode`) select it with no serializer change; `build_default_service`
auto-wires an `LLMAgentController` whenever an API key is set, bounded by
`SPARKSAGE_AGENT_MAX_ITERATIONS`.

Also implemented now: the Phase-1/2 agentic-RAG enhancements that finally
connect the agent loop to the existing right-half reflection components (the
"agent was an island" gap from the analysis report). The `ActionType` enum
gained `PLAN` (Plan-and-Execute): a controller emits `sub_queries: [...]`, the
engine enqueues them and drains the queue one per iteration without
re-consulting the controller between them, then re-plans / synthesizes — the
missing decomposition layer for comparison / multi-hop questions. The
controller system prompt (`agent/prompts.py`) now documents `plan` and the
previously-dead `filter` field (tags / entities / languages / kb_id, coerced
leniently in `agent/schema.py:_coerce_filter` with unknown tags dropped rather
than failing). `AgenticQAEngine.__init__` accepts optional
`retrieval_grader` / `query_refiner` / `query_expander` / `step_min_relevance`
/ `step_max_refine` / `expander_n_variants`: each per-step retrieval is
**expanded** (multi-query / HyDE → RRF via the new shared
`retrieve/multi_query.py:multi_query_retrieve`, which `QAEngine._multi_retrieve`
also delegates to), then **graded** (`ISREL` gate, `LLMRetrievalGrader`), then
on a low score **refined + re-retrieved** (CRAG / Self-RAG via
`LLMQueryRefiner`, keeping the best-graded result so refinement can never
lower quality, bounded by `step_max_refine`); the per-step `RelevanceResult`
+ refined query ride on `AgentStep` and surface through `AgentStepOut`
(`relevance_score` / `relevance_reasoning` / `refined_query`) for trajectory
transparency. SSE streaming of the agent loop: `POST /api/v1/query` with
`AskRequest.stream=true` + `mode="agent"` returns a `text/event-stream` of
`progress` events (one per phase — thinking / retrieving / synthesizing /
done, carrying iteration / phase / percent / evidence_count / relevance)
terminated by a single `result` event carrying the full `AskResponse` and a
`done` sentinel; the worker runs in a daemon thread (the blocking LLM /
retrieval I/O stays off the event loop) and pushes events through a
thread-safe queue. `build_qa_service` auto-wires `LLMRetrievalGrader` /
`LLMQueryRefiner` / `HyDEExpander` (reusing the LLM client) whenever an API
key is set — disable with `SPARKSAGE_AGENT_REFLECTION=off`, tune with
`SPARKSAGE_AGENT_STEP_MIN_RELEVANCE` / `SPARKSAGE_AGENT_STEP_MAX_REFINE`.
 Planned next: an OpenAI-compatible API, an `/api/v1/distill` route wrapping
 `JobManager`, and an `/api/v1/query/agent` route wrapping a pollable agent job
 (reusing the `DistillJob` state machine over the `AgenticQAEngine`
 `on_progress` / `is_cancelled` hooks so long agent runs are cancellable /
 observable exactly like a Distill run).
 Design schema additions so the Distill lifecycle fields (`status`, `parents`,
 `confidence`, `embedding`) remain usable.
