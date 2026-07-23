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

## Roadmap context

Implemented now: chunk schema (IdeaBlock + TechnicalBlock), LLM-driven
generation (`generator/`: prompt building, JSON extraction, enum coercion), and
uniform file-to-Markdown conversion (`convert/`: pluggable backend built on
`markitdown`, single-file + resilient batch directory mode).
Planned next: Distill de-dup pipeline (embedding + LSH + FAISS + threshold
iteration + Louvain/BFS + hierarchical LLM merge) and an OpenAI-compatible API.
Design schema additions so the Distill lifecycle fields (`status`, `parents`,
`confidence`, `embedding`) remain usable.
