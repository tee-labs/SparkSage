# SparkSage

**Structured, question-aligned knowledge chunks for high-quality RAG.**

SparkSage replaces naive fixed-size text slicing with the **IdeaBlock** — a
small, self-contained *knowledge unit* that is aligned to how users ask
questions. Instead of embedding arbitrary text fragments (which get cut
mid-sentence and retrieve poorly), SparkSage embeds whole, verified answers.

> Status: **Pre-Alpha**. This repository implements the **chunk schema**
> (IdeaBlock + TechnicalBlock), **LLM-driven generation** (turn free text
> into many IdeaBlocks), **file-to-Markdown conversion**, **customizable
> text cleaning**, a **knowledge-document management service** (upload,
> auto-extract title/summary/tags, store, CRUD over REST — with a built-in
> keyword-extraction auto-tagger), and a **WEB API** exposing convert /
> generate / documents over HTTP. The Distill de-duplication pipeline is
> planned.

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

## Manage knowledge documents

SparkSage's core is *chunk*-oriented (the IdeaBlock), but a knowledge-management
platform works at the *document* granularity. The `documents` package adds
exactly that: upload a Markdown document, auto-extract its **title / summary /
body / tags**, store it, and manage it over CRUD — all **without an LLM**.

The headline feature: **when a document ships without tags, the system
auto-generates them from content using a keyword-extraction algorithm**
([`FrequencyKeywordExtractor`](src/sparksage/documents/keyword_extract.py),
pure stdlib — no API key, fully deterministic).

```python
from sparksage import DocumentService, MarkdownConverter

service = DocumentService(converter=MarkdownConverter())   # needs 'markitdown'

# upload WITHOUT tags -> auto-tagged from content
doc = service.upload(b"...", "onboarding.md")
print(doc.title, doc.summary, doc.tags)        # tags auto-generated
print(doc.tag_source)                          # TagSource.AUTO

# upload WITH tags -> user tags win
doc = service.upload(b"...", "design.md", tags=["architecture"])
print(doc.tag_source)                          # TagSource.USER

# CRUD + tag management
service.get(doc.id)
service.list(tag="architecture")
service.add_tags(doc.id, ["review"])
service.extract_tags(doc.id, top_n=5)          # (re)run keyword extraction
service.update(doc.id, title="New title")
service.delete(doc.id)
```

How it stays pluggable and dependency-light:

- The store is a [`DocumentStore`](src/sparksage/documents/store.py) Protocol —
  [`InMemoryDocumentStore`](src/sparksage/documents/store.py) ships for demos /
  tests / single-process deployments; implement the Protocol against a database
  or search engine for production without touching the service or HTTP layers.
- The keyword extractor is a [`KeywordExtractor`](src/sparksage/documents/keyword_extract.py)
  Protocol — swap in TextRank / YAKE / jieba without changing callers. The
  default ranks terms by **position-weighted term frequency** (Markdown headings
  outrank body prose), filters stop words (English + Chinese), scores both Latin
  words/phrases and CJK unigrams/bigrams, and drops unigrams already covered by
  a higher-ranked phrase.
- `TagSource` (`USER` / `AUTO` / `MIXED`) records how each document's tag set
  was produced, so the management UI can show provenance and decide whether to
  re-run extraction.
- The service is framework-agnostic (no HTTP imports) and fully unit-testable
  with [`FakeConverterBackend`](src/sparksage/convert/backend.py).

Offline demo (no API key, no `markitdown`):

```bash
PYTHONPATH=src python3 examples/manage_documents.py
```

---

## Serve the WEB API

SparkSage exposes the core capabilities over a small HTTP API:

* `POST /api/v1/convert` — upload a file, get back Markdown (optionally cleaned).
* `POST /api/v1/generate` — upload a file, get back a list of IdeaBlocks.
* `/api/v1/documents` (+ `/api/v1/tags`) — knowledge-document CRUD and tag
  management (auto-tagging from content when no tags are supplied). **Works
  LLM-free.**

The API layer is a thin shell over framework-agnostic services
([`SparkSageService`](src/sparksage/api/pipeline.py),
[`DocumentService`](src/sparksage/documents/service.py)). FastAPI is an
*optional* dependency.

```bash
pip install 'sparksage[api]'          # fastapi + uvicorn + python-multipart
pip install 'sparksage[convert]'      # markitdown for real file conversion
pip install 'sparksage[llm]'          # openai SDK for real generation
```

### Run the server

```bash
export SPARKSAGE_API_KEY=sk-...                    # or OPENAI_API_KEY
export SPARKSAGE_MODEL=gpt-4o-mini                 # optional
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

# 3) upload a document, auto-tag it from content (no tags supplied)
curl -F "file=@onboarding.md" \
     http://localhost:8000/api/v1/documents

# 4) list / filter / manage
curl "http://localhost:8000/api/v1/documents?tag=onboarding&limit=20"
curl http://localhost:8000/api/v1/tags
curl -X POST http://localhost:8000/api/v1/documents/<id>/tags:auto \
     -H 'content-type: application/json' -d '{"top_n": 5}'
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

`POST /api/v1/documents` (auto-tagged, no tags supplied) returns:

```json
{
  "document": {
    "id": "b3e1...c4",
    "title": "Onboarding Guide",
    "summary": "This guide explains the onboarding workflow for new engineers.",
    "content": "# Onboarding Guide\n\nThis guide explains ...",
    "tags": ["onboarding", "onboarding guide", "workflow"],
    "tag_source": "AUTO",
    "source": {"uri": "onboarding.md", "title": "Onboarding Guide"},
    "language": "en",
    "version": 1,
    "created_at": "2026-07-24T04:30:00Z",
    "updated_at": "2026-07-24T04:30:00Z"
  }
}
```

The full document surface (all under `/api/v1`): `POST /documents`,
`GET /documents` (`?tag=&limit=&offset=`), `GET/PATCH/DELETE /documents/{id}`,
`POST /documents/{id}/tags`, `POST /documents/{id}/tags:auto`,
`DELETE /documents/{id}/tags/{tag}`, and `GET /tags`.

How it stays testable and pluggable:

- The orchestration lives entirely in
  [`SparkSageService`](src/sparksage/api/pipeline.py) — no HTTP imports — so it
  is fully unit-testable offline with fakes. The FastAPI layer only does
  upload/serialization.
- `create_app(service=...)` accepts an injected service (for tests); when
  omitted it builds one from env vars (`SPARKSAGE_API_KEY` / `OPENAI_API_KEY`).
- If no API key is set, `/generate` returns a clear `503` instead of crashing;
  `/convert` works independently of any LLM.
- Uploaded bytes are written to a short-lived temp file carrying the original
  extension (so the backend picks the right format handler), and provenance is
  set back to the *original* filename — keeping cleaning-rule routing and
  `source.uri` meaningful.

Offline demo (no API key, no `markitdown`, exercises both routes via TestClient):

```bash
PYTHONPATH=src python3 examples/serve_api.py
```

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
| `SPARKSAGE_LANGUAGE`  | BCP-47 code written into each block (e.g. `en`, `zh`)|

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
├── schema/
│   ├── enums.py        # controlled vocabularies (Tag, TagSource, EntityType, ...)
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
├── documents/
│   ├── schema.py       # Document model (title/summary/body/tags)  ★
│   ├── markdown_parser.py  # deterministic title + summary extraction
│   ├── keyword_extract.py  # keyword-extraction algorithm + Protocol  ★
│   ├── store.py        # DocumentStore protocol + InMemoryDocumentStore
│   └── service.py      # convert -> parse -> tag -> store orchestration  ★
├── api/
│   ├── pipeline.py     # SparkSageService: convert→clean→generate orchestration  ★
│   ├── documents.py    # document CRUD + tag-management routes  ★
│   ├── schemas.py      # request/response Pydantic models (no fastapi)
│   └── app.py          # FastAPI app factory + routes (lazy fastapi import)
tests/                  # schema + generation + conversion + cleaning + documents + api + config
examples/               # runnable demos
```

## Development

```bash
PYTHONPATH=src python3 -m pytest -q          # tests
ruff check src tests                          # lint
```

## Roadmap

- [x] Chunk schema (IdeaBlock + TechnicalBlock) — *first release*
- [x] LLM-driven generation (text -> many IdeaBlocks via pluggable LLM client)
- [x] Uniform file-to-Markdown conversion (any format -> Markdown via markitdown)
- [x] Customizable text cleaning (business-specific rules, source-aware routing)
- [x] Knowledge-document management (upload, auto title/summary/tags, CRUD,
      keyword-extraction auto-tagger — LLM-free)
- [x] WEB API (FastAPI: file → Markdown, file → IdeaBlock list, document CRUD)
- [x] `.env` configuration (built-in loader, env vars override file)
- [ ] Distill de-duplication pipeline (embedding + LSH + FAISS kNN + threshold
      iteration + Louvain/BFS + hierarchical LLM merge)
- [ ] OpenAI-compatible ingest/distill API
- [ ] Reproducible benchmark suite

## License

Apache-2.0.
