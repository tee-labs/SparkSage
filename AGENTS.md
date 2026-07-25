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
  `(a, b)` for determinism) at or above a `[0, 1]` threshold. This is the
  first, dependency-free step of the future Distill pipeline (clustering /
  Louvain / LLM-merge stay in the planned `distill/` package under the
  `[distill]` group, where approximate LSH+FAISS candidate reduction earns its
  weight at million-vector scale).
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
detection (pure stdlib, the first dependency-free step of Distill)), and a WEB
API (`api/`:
framework-agnostic `SparkSageService` orchestration + FastAPI app factory
exposing `/api/v1/convert` and `/api/v1/generate`), and `.env`-based
configuration (`config.py`: zero-dependency loader, env vars override the
file).
Planned next: Distill de-dup pipeline (LSH + FAISS + threshold iteration +
Louvain/BFS + hierarchical LLM merge, building on `find_similar_pairs` /
the `embed` vectors + the schema lifecycle fields) and an OpenAI-compatible
API. Design schema additions so the Distill lifecycle fields (`status`,
`parents`, `confidence`, `embedding`) remain usable.
