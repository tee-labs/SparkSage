# syntax=docker/dockerfile:1
#
# SparkSage API image.
#
# Builds the library and runs the FastAPI web API (uvicorn factory) with the
# convert + llm extras enabled, so both /api/v1/convert and /api/v1/generate
# are functional. Provide SPARKSAGE_API_KEY (or OPENAI_API_KEY) at runtime to
# enable /generate; without a key the API still serves /convert and /health.
#
#   docker build -t sparksage:latest .
#   docker run --rm -p 8000:8000 --env-file .env sparksage:latest
#
# Env vars (SPARKSAGE_* take priority over OPENAI_*):
#   SPARKSAGE_API_KEY   API key (falls back to OPENAI_API_KEY)
#   SPARKSAGE_BASE_URL  OpenAI-compatible base URL (custom endpoint)
#   SPARKSAGE_MODEL     Model id (default gpt-4o-mini)
#   SPARKSAGE_STREAM    Stream the LLM response (default true)
#   SPARKSAGE_LANGUAGE  BCP-47 code written into every block

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip wheel --no-deps --wheel-dir /wheels .

FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user for runtime safety.
RUN groupadd --system --gid 1001 sparksage && \
    useradd --system --uid 1001 --gid sparksage --create-home sparksage

WORKDIR /app

COPY --from=builder /wheels /wheels

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install /wheels/sparksage-*.whl "sparksage[api,convert,llm]" && \
    rm -rf /wheels && \
    python -c "import sparksage; print(sparksage.__version__)"

USER sparksage

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health',timeout=3).read()" || exit 1

CMD ["python", "-m", "uvicorn", "sparksage.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
