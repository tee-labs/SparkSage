# syntax=docker/dockerfile:1
#
# SparkSage API image.
#
# Builds the library and runs the FastAPI web API (uvicorn factory). By default
# the full extras set is installed so the *complete* end-to-end QA pipeline is
# available out of the box:
#
#   * convert + llm  -> /api/v1/convert, /api/v1/generate
#   * documents      -> /api/v1/documents (CRUD), /api/v1/tags
#   * embed + rerank -> the knowledge base: /api/v1/knowledge_base/ingest,
#                       /api/v1/query (mounted when SPARKSAGE_ENABLE_QA is set)
#   * distill        -> the de-dup pipeline (in-process, /api/v1/distill later)
#   * tags-zh        -> jieba CJK segmentation for keyword extraction
#
# The QA routes are mounted automatically because SPARKSAGE_ENABLE_QA=1 is set
# below; unset it (or set SPARKSAGE_ENABLE_QA=0) to run the slim convert +
# generate + documents API only.
#
# Provide SPARKSAGE_API_KEY (or OPENAI_API_KEY) at runtime to enable
# /generate and /query; without a key the API still serves /convert, /documents,
# /tags and /health, while /generate and /query return a clear 503.
#
#   docker build -t sparksage:latest .
#   # slim image (no QA / embeddings):
#   docker build --build-arg SPARKSAGE_EXTRAS=api,convert,llm -t sparksage:slim .
#   # add production vector stores:
#   docker build \
#     --build-arg SPARKSAGE_EXTRAS=api,convert,llm,embed,rerank,distill,tags-zh,chroma,pgvector \
#     -t sparksage:full .
#
#   docker run --rm -p 8000:8000 --env-file .env sparksage:latest
#   # or mount a .env directly into the working dir (auto-loaded at startup):
#   docker run --rm -p 8000:8000 -v "$PWD/.env:/app/.env:ro" sparksage:latest
#
# Env vars (SPARKSAGE_* take priority over OPENAI_*):
#   SPARKSAGE_ENABLE_QA       Mount the full QA pipeline (default "1")
#   SPARKSAGE_API_KEY         API key (falls back to OPENAI_API_KEY)
#   SPARKSAGE_BASE_URL        OpenAI-compatible base URL (custom endpoint)
#   SPARKSAGE_MODEL           Model id (default gpt-4o-mini)
#   SPARKSAGE_STREAM          Stream the LLM response (default true)
#   SPARKSAGE_LANGUAGE        BCP-47 code written into every block
#   SPARKSAGE_EMBEDDING_API_KEY   Embedding key (falls back to the LLM key)
#   SPARKSAGE_EMBEDDING_BASE_URL  Embedding base URL (falls back to LLM base URL)
#   SPARKSAGE_EMBEDDING_MODEL     Embedding model (default text-embedding-3-small)
#   SPARKSAGE_DOC_STORE       Path to a SQLite file for durable document storage
#   SPARKSAGE_AUTO_TAG_EXTRACTOR  Auto-tag algorithm: rake|tfidf|textrank
#   SPARKSAGE_TAGS_ZH         Use jieba for CJK segmentation when truthy

ARG PYTHON_VERSION=3.11
# Full QA pipeline. Override with --build-arg SPARKSAGE_EXTRAS=... for a slim or
# extended image (e.g. append chroma / pgvector for production vector stores).
ARG SPARKSAGE_EXTRAS="api,convert,llm,embed,rerank,distill,tags-zh"

# --------------------------------------------------------------------------- #
# Frontend build stage: compile the React + Ant Design WEB UI to static assets.
# The built `web/dist` is served by FastAPI behind a catch-all route, so the
# whole product (API + UI) ships in one container on :8000.
# --------------------------------------------------------------------------- #
FROM node:20-slim AS frontend
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:${PYTHON_VERSION}-slim AS builder

ARG SPARKSAGE_EXTRAS

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Third-party dependency wheels. This layer is cached by pyproject.toml +
# SPARKSAGE_EXTRAS ONLY — never by application source. A throwaway stub package
# is created so pip can resolve the extras without needing the real src/.
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p src/sparksage && touch src/sparksage/__init__.py && \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip wheel --wheel-dir /deps ".[${SPARKSAGE_EXTRAS}]" && \
    rm -f /deps/sparksage-*.whl

# Application wheel. Rebuilds whenever src/ changes, but it is tiny (pure Python)
# and never invalidates the dependency layer above.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip wheel --no-deps --wheel-dir /wheels .

FROM python:${PYTHON_VERSION}-slim AS runtime

ARG SPARKSAGE_EXTRAS
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SPARKSAGE_EXTRAS=${SPARKSAGE_EXTRAS} \
    SPARKSAGE_ENABLE_QA=1

# Non-root user for runtime safety.
RUN groupadd --system --gid 1001 sparksage && \
    useradd --system --uid 1001 --gid sparksage --create-home sparksage

WORKDIR /app

# Stable layer: install all third-party dependencies from the pre-built local
# wheels (--no-index = no network). Cache hits as long as pyproject.toml and the
# extras set are unchanged, regardless of application source edits.
COPY --from=builder /deps /deps
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-index --find-links=/deps /deps/*.whl && \
    rm -rf /deps

# Volatile layer: only the lightweight sparksage wheel. Re-runs on every code
# change but completes in well under a second.
COPY --from=builder /wheels /wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps /wheels/sparksage-*.whl && \
    rm -rf /wheels && \
    python -c "import sparksage; print(sparksage.__version__)"

# Built WEB UI. FastAPI auto-serves it when web/dist is present next to the app
# (served behind a catch-all route, so one container exposes API + UI on :8000).
COPY --from=frontend --chown=sparksage:sparksage /web/dist /app/web/dist

USER sparksage

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health',timeout=3).read()" || exit 1

CMD ["python", "-m", "uvicorn", "sparksage.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
