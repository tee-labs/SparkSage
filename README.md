# SparkSage

**Structured, question-aligned knowledge chunks for high-quality RAG.**

SparkSage replaces naive fixed-size text slicing with the **IdeaBlock** — a
small, self-contained *knowledge unit* that is aligned to how users ask
questions. Instead of embedding arbitrary text fragments (which get cut
mid-sentence and retrieve poorly), SparkSage embeds whole, verified answers.

> Status: **Pre-Alpha**. This repository implements the **chunk schema**
> (IdeaBlock + TechnicalBlock), **LLM-driven generation** (turn free text
> into many IdeaBlocks), **file-to-Markdown conversion**, and **customizable
> text cleaning**. The Distill de-duplication pipeline and ingest service
> are planned.

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

## Project layout

```
src/sparksage/
├── schema/
│   ├── enums.py        # controlled vocabularies (Tag, EntityType, SentenceRole, ...)
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
tests/                  # 114 tests (schema + generation + conversion + cleaning)
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
- [ ] Distill de-duplication pipeline (embedding + LSH + FAISS kNN + threshold
      iteration + Louvain/BFS + hierarchical LLM merge)
- [ ] OpenAI-compatible ingest/distill API
- [ ] Reproducible benchmark suite

## License

Apache-2.0.
