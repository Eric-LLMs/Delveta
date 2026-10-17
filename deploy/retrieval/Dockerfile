# Retrieval service: the RAG pipeline (embedding + recall + rerank + rewrite) exposed over gRPC.
#
# Model inference (embedding / rerank / LLM) stays in the separate model containers; this
# service only orchestrates the retrieval DAG and owns pgvector/FTS access.
#
# Layer layout (cache discipline). The heavy `[rag]` extra pulls sentence-transformers ->
# torch + CUDA + transformers (~6.7 GB). Editing apps/ or packages/ source MUST NOT reinstall
# that layer: it is keyed only on the dependency manifest below, and the local project is
# installed separately with --no-deps. Ordered layers:
#   [1] third-party deps  <-- cache key: pyproject.toml + uv.lock (never source)
#   [2] local source      <-- cache key: apps/ + packages/ + proto/ + scripts/
# Both pip runs mount a persistent BuildKit cache (type=cache) for downloaded wheels, so a
# layer invalidation reuses the wheel cache instead of re-fetching from PyPI.
FROM python:3.12-slim

WORKDIR /app

# ── Layer 1: third-party dependencies (cache key = pyproject.toml + uv.lock) ──────────────
# uv.lock is the project's single source of truth for the resolved version set. `uv export`
# emits those locked third-party requirements WITHOUT the local project (--no-emit-project),
# so this ~6.7 GB layer is reused verbatim as long as the dependency declarations or the lock
# do not change. Base + [rag] extra are both included; the dev group is excluded. Nothing here
# reads apps/ or packages/, so source edits never reach it. No hand-written dependency list:
# the set comes from pyproject.toml as resolved by uv.lock.
COPY pyproject.toml README.md uv.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install uv \
 && uv export --frozen --no-dev --extra rag --no-emit-project -o /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt \
 && rm -f /tmp/requirements.txt

# ── Layer 2: local source + project + gRPC stubs (cache key = source content) ─────────────
# The project's declared deps are already installed by layer 1, so --no-deps installs ONLY
# the local `delveta` package (fast; no re-resolution, no version drift). Hatchling still
# needs the declared wheel packages on disk to build, so the source copies come before it.
# .dockerignore keeps the context (and this layer) small: training workspaces under scripts/,
# node_modules, data/, logs/ and .venv* are excluded.
COPY packages ./packages
COPY apps ./apps
COPY proto ./proto
COPY scripts ./scripts

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps "." \
 # Generate the protobuf/gRPC stubs (falls back to grpc_tools.protoc when buf is absent).
 && bash scripts/gen_proto.sh

# No PYTHONPATH needed: the editable install resolves `agent`/`rag`/`core`/`retrieval.v1`
# packages (see pyproject.toml [tool.hatch.build.targets.wheel].packages).

EXPOSE 50051

CMD ["python", "-m", "apps.retrieval.main"]
