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
- The documents core (`documents/`) is the *document*-granularity counterpart to
  the chunk-oriented IdeaBlock core. It is pure stdlib (no optional deps) and
  depends only on small Protocols: `KeywordExtractor` (auto-tagging) and
  `DocumentStore` (persistence). `DocumentService` composes the existing
  `MarkdownConverter` / `TextCleaner` and adds deterministic Markdown parsing
  (`markdown_parser`: H1 title + first-paragraph summary) plus keyword
  extraction (`keyword_extract`: position-weighted TF over headings/body with
  English+Chinese stop words, Latin words/phrases and CJK unigrams/bigrams).
  Document tags are **free-form strings** (not the controlled `Tag` enum);
  `TagSource` (`USER`/`AUTO`/`MIXED`) records how each tag set was produced.
  The `documents` package must never import from `api` — `api` is the outermost
  layer and depends on `documents`, never the reverse (hence the small temp-file
  helper is duplicated in `service.py` rather than imported from `api/pipeline`).
  `InMemoryDocumentStore` is for demos/tests/single-process; production
  implements `DocumentStore` against a DB/search engine.
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
key is set, `/generate` returns `503` while `/convert` and the whole
`/api/v1/documents` management surface (CRUD + auto-tagging) work LLM-free.
`create_app` also accepts an injected `document_service` (built from
`build_default_document_service()` otherwise, which needs `markitdown` but no
API key); the document routes are registered by `register_document_routes` in
`api/documents.py` (same lazy-FastAPI pattern as `app.py`). Note:
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

## Roadmap context

Implemented now: chunk schema (IdeaBlock + TechnicalBlock), LLM-driven
generation (`generator/`: prompt building, JSON extraction, enum coercion),
uniform file-to-Markdown conversion (`convert/`: pluggable backend built on
`markitdown`, single-file + resilient batch directory mode), customizable
text cleaning (`clean/`: composable `CleaningRule`s, source/filename-aware
routing via `CleaningRegistry`, sits between conversion and generation), a
knowledge-document management service (`documents/`: `Document` model,
deterministic Markdown title/summary parsing, keyword-extraction auto-tagger
behind a `KeywordExtractor` Protocol, pluggable `DocumentStore` +
`InMemoryDocumentStore`, `DocumentService` orchestration), a WEB API
(`api/`: framework-agnostic `SparkSageService` + `DocumentService`
orchestration + FastAPI app factory exposing `/api/v1/convert`,
`/api/v1/generate`, `/api/v1/documents` (CRUD + tag management) and
`/api/v1/tags`), and `.env`-based configuration (`config.py`: zero-dependency
loader, env vars override the file).
Planned next: Distill de-dup pipeline (embedding + LSH + FAISS + threshold
iteration + Louvain/BFS + hierarchical LLM merge) and an OpenAI-compatible API.
Design schema additions so the Distill lifecycle fields (`status`, `parents`,
`confidence`, `embedding`) remain usable.
