# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# AT-KV assistant
#
# THE INDEX AND THE EMBEDDING MODEL ARE BAKED IN AT BUILD TIME.
#
# A container carrying an embedding model is slow to start from nothing. If the
# image shipped only code, every cold start would download ~470 MB of model
# weights and re-embed 538 chunks before serving its first request. With
# minReplicas 0 -- which is how this stays free in Stage 2 -- that cost lands on
# a real user, not on a deploy.
#
# So the build does the expensive work once: fetch the sources, parse, chunk,
# embed, write the FAISS index, and pre-download the model into the image.
# Startup then loads an index from disk and a model from the local filesystem.
# Query-time embedding still needs the model resident, which is why it is baked
# in rather than merely cached.
#
# The trade is image size against cold-start latency. This project has no
# tolerance for cold starts and no limit on image size that matters at EUR 0,
# so it pays in bytes.
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS base

# CPU-only torch is pinned in pyproject.toml via [tool.uv.sources], NOT here.
# UV_EXTRA_INDEX_URL does not work for this: `uv sync --frozen` honours the
# lockfile, and PyPI's torch declares its CUDA dependencies with the marker
# platform_system == "Linux" without gating on architecture -- so this ARM
# image pulled 3.3 GB of x86-only nvidia wheels anyway. Measured before the
# fix: 7.81 GB total, 42% of it CUDA that can never execute here.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # faiss and torch each bundle an OpenMP runtime; on some hosts loading a
    # torch model after faiss has used its runtime segfaults with no traceback.
    # Also set in atkv/__init__.py -- repeated here so it survives anyone
    # invoking a module directly.
    OMP_NUM_THREADS=1 \
    HF_HOME=/opt/hf \
    ATKV_ROOT=/app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# ---- dependencies (cached independently of source changes) ----------------
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ---- source ---------------------------------------------------------------
COPY src/ ./src/
COPY manifests/ ./manifests/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- bake EVERY model the service loads at startup ------------------------
# All of them, not just the embedder. A first version cached only e5-small, so
# the container downloaded the translation model on startup instead: 49.7s to
# become ready, and a hard dependency on reaching huggingface.co at boot. A
# container that needs the network to start is not really self-contained.
RUN uv run python -c "\
from sentence_transformers import SentenceTransformer; \
from transformers import MarianMTModel, MarianTokenizer; \
SentenceTransformer('intfloat/multilingual-e5-small', device='cpu'); \
MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de'); \
MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de'); \
print('models cached')"

# ---- bake the index into the image ----------------------------------------
# Needs network: the WKO PDFs are fetched from manifests/sources.yaml rather
# than committed, because they are the social partners' copyrighted documents.
# The page-1 assertion in ingest/wko.py runs here, so a build FAILS rather than
# producing an image that serves the wrong year's figures.
RUN uv run python -c "\
from pathlib import Path; \
from atkv.ingest.build import build_corpus; \
from atkv.retrieve.pipeline import RetrievalPipeline; \
c = build_corpus(Path('/app')); \
RetrievalPipeline.build(c).save(Path('/app/data/index/serving')); \
print(f'indexed {len(c)} chunks')"

# The raw PDFs are not needed at runtime and must not ship in a distributable
# image; the index holds the extracted text.
RUN rm -rf /app/data/raw

EXPOSE 8000

# Reports degraded when a source type cannot cover today, so a stale index is
# visible to the orchestrator rather than only to a puzzled user.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz',timeout=5).status==200 else 1)"

CMD ["uv", "run", "uvicorn", "atkv.api:app", "--host", "0.0.0.0", "--port", "8000"]
