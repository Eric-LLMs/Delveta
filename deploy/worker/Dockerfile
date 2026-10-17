# Worker: arq async task runner for enrichment jobs (TTS / images / AI / indexing).
#
# Model inference (LLM / embedding / TTS) stays in the separate model containers; this
# service only orchestrates the enrichment jobs, talking to those containers over HTTP and
# to Redis (arq queue) + PostgreSQL (jobs table + session data).
#
# Layer layout (cache discipline). Editing apps/ or packages/ source MUST NOT reinstall the
# OS toolchain, re-download typst/fonts, or re-resolve the Python dependency set. Ordered
# layers, cheapest-to-change last:
#   [1] OS packages + publication toolchain  <-- no source dependency (stable)
#   [2] third-party Python deps              <-- cache key: pyproject.toml + uv.lock
#   [3] local source + editable install      <-- cache key: apps/ + packages/ + plugins/
# Layers [1] and [2] never read apps/ or packages/, so a source-only edit rebuilds only [3].
FROM python:3.12-slim

WORKDIR /app

# ── Layer 1: OS packages + publication PDF toolchain (no source dependency) ──────────────
# Publication PDF toolchain (docs/19 §8): the PUBLISH node compiles report PDFs by
# default, so the worker carries the exact stack the preflight probe checks —
# typst CLI (static musl build) + the three font FAMILIES the template/probe names
# verbatim (Libertinus Serif, Noto Serif SC, DejaVu Sans Mono; apt's fonts-noto-cjk
# ships "Noto Serif CJK SC" — a different family name — hence the direct download).
# Every download is timeout-bounded and retried so a flaky mirror fails the build
# loudly instead of hanging; `typst --version` is the build-time gate.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         # antiword: legacy .doc text extraction on the ingest path (core.infrastructure.ingest).
         curl ca-certificates xz-utils unzip fontconfig fonts-dejavu-core antiword \
    && curl -fsSL --connect-timeout 15 --retry 3 --retry-delay 2 \
         -o /tmp/typst.tar.xz \
         https://github.com/typst/typst/releases/download/v0.15.1/typst-x86_64-unknown-linux-musl.tar.xz \
    && tar -xJf /tmp/typst.tar.xz -C /tmp \
    && install -m 0755 /tmp/typst-x86_64-unknown-linux-musl/typst /usr/local/bin/typst \
    && mkdir -p /usr/local/share/fonts/pdf \
    && curl -fsSL --connect-timeout 15 --retry 3 --retry-delay 2 \
         -o /usr/local/share/fonts/pdf/NotoSerifSC.ttf \
         "https://raw.githubusercontent.com/google/fonts/main/ofl/notoserifsc/NotoSerifSC%5Bwght%5D.ttf" \
    && curl -fsSL --connect-timeout 15 --retry 3 --retry-delay 2 \
         -o /tmp/libertinus.zip \
         https://github.com/alerque/libertinus/releases/download/v7.051/Libertinus-7.051.zip \
    && unzip -jq /tmp/libertinus.zip 'Libertinus-7.051/static/TTF/LibertinusSerif-*.ttf' \
         -d /usr/local/share/fonts/pdf \
    && fc-cache -f \
    && typst --version \
    && rm -rf /var/lib/apt/lists/* /tmp/typst.tar.xz /tmp/libertinus.zip /tmp/typst-x86_64-unknown-linux-musl

# ── Layer 2: third-party Python deps (cache key = pyproject.toml + uv.lock) ──────────────
# uv.lock is the resolved version set. `uv export --frozen` emits the locked third-party
# requirements WITHOUT the local project (--no-emit-project), so this layer is reused
# verbatim until the dependency declarations or the lock change. Nothing here reads apps/
# or packages/. A persistent BuildKit pip cache means a layer invalidation reuses wheels.
COPY pyproject.toml README.md uv.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install uv \
 && uv export --frozen --no-dev --no-emit-project -o /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt \
 && rm -f /tmp/requirements.txt

# ── Layer 3: local source + editable install (cache key = source content) ────────────────
# The project's declared deps are already installed by layer 2, so --no-deps installs ONLY
# the local `delveta` package (fast; no re-resolution). Source copies come first because
# hatchling needs the declared wheel packages on disk to build the editable wheel.
COPY packages ./packages
COPY apps ./apps
# Research OS plugin code: worker's research_drive task imports plugins.research.{driver,plugin}
# at runtime, and they are file-backed modules (not pip-packaged) — copy the tree verbatim.
COPY plugins ./plugins

RUN pip install --no-deps -e "." \
    # Editable install: /app/packages (+ /app/apps, /app/packages/shared/proto) become the
    # ONE import source — no frozen copies in site-packages to double-write against.
    # Hatchling's generated .pth only lists the package parents and puts /app/apps before
    # /app/packages, which would shadow the proto `retrieval` with the apps/retrieval app.
    # Pin the order explicitly (proto first) so every top-level name resolves to source.
    && printf '/app/packages/shared/proto\n/app/apps\n/app/packages\n' \
       > /usr/local/lib/python3.12/site-packages/_editable_impl_delveta.pth

CMD ["python", "-m", "apps.worker.main"]
