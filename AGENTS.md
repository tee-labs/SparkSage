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

## Roadmap context

Implemented now: chunk schema (IdeaBlock + TechnicalBlock). Planned next:
Distill de-dup pipeline (embedding + LSH + FAISS + threshold iteration +
Louvain/BFS + hierarchical LLM merge) and an OpenAI-compatible API. Design
schema additions so the Distill lifecycle fields (`status`, `parents`,
`confidence`, `embedding`) remain usable.
