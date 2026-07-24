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
routing via `CleaningRegistry`, sits between conversion and generation), and a
  WEB API (`api/`: framework-agnostic `SparkSageService` orchestration +
  FastAPI app factory exposing `/api/v1/convert` and `/api/v1/generate`),
  `.env`-based configuration (`config.py`: zero-dependency loader, env vars
  override the file), and query-time intent recognition + rewriting
  (`query/`: pluggable `IntentClassifier` / `QueryRewriter` protocols reusing
  `LLMClient`, LLM + rule-based implementations, lenient→strict `QueryIntent`
  coercion, `QueryProcessor` orchestration with interception policy — not yet
  wired to the web layer).
Planned next: Distill de-dup pipeline (embedding + LSH + FAISS + threshold
iteration + Louvain/BFS + hierarchical LLM merge), an OpenAI-compatible API,
and a `/api/v1/query` route wrapping `QueryProcessor`.
Design schema additions so the Distill lifecycle fields (`status`, `parents`,
`confidence`, `embedding`) remain usable.
