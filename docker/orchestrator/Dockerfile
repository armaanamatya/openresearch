# syntax=docker/dockerfile:1.6
#
# ReproLab orchestrator (control-plane) image — GCP/Azure in-cluster root loop.
#
# This is the image the GKE / AKS orchestrator Deployment + CronJob run
# (infra/gcp/helm orchestrator-{deployment,cronjob}.yaml, orchestrator.image).
# It runs `python -m backend.cli reproduce <paper> --mode rlm --model claude-oauth
# --sandbox gcp`, where the root model shells out to the `claude` CLI (which
# itself needs Node) and dispatches training cells OUTWARD as K8s Jobs onto the
# GPU-cell image (docker/gke-cell-base).  So this control-plane image needs:
#   • Python 3.12 + the reprolab backend (pip install -r backend/requirements.txt)
#   • Node >= 20.19  (the `claude` CLI runtime)
#   • The `claude` CLI  (npm install -g @anthropic-ai/claude-code@<PINNED>)
#
# It deliberately does NOT install torch/CUDA — training never runs here, it
# runs in the gke-cell-base Job pods.  Keep this image minimal.
#
# Build / push to Artifact Registry: scripts/build_orchestrator_image.sh
#   (NEVER use :latest — always a pinned git-SHA tag).
#
# Base image: same digest-pinned python:3.12-slim-bookworm as the root
# Dockerfile (a moving :3.12-slim tag silently changes the build between runs —
# see the root Dockerfile digest note).  Bump deliberately, alongside the root.

FROM python:3.12-slim-bookworm@sha256:76d4b7b6305788c6b4c6a19d6a22a3921bf802e9af4d5e1e5bd771208dba74bf

LABEL org.opencontainers.image.title="reprolab-orchestrator"
LABEL org.opencontainers.image.description="ReproLab in-cluster root reasoning loop (control plane; no CUDA)"
LABEL org.opencontainers.image.version="1"

# Pinned claude-code CLI version — NEVER floating (a `@latest` would silently
# drift the root transport between rebuilds).  Bump deliberately + rebuild.
ARG CLAUDE_CODE_VERSION=2.1.179

# Node 22 = active LTS (>= 20.19 required by the claude CLI); matches the root
# Dockerfile / CI Node major.  Bump alongside the root image.
ARG NODE_MAJOR=22

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:$PATH

# ---- System packages --------------------------------------------------------
# tini  — PID 1, signal-safe (mirrors the root image entrypoint).
# git/curl/ca-certificates — VCS deps + the claude CLI's network surface.
# build-essential — a few pinned wheels build from source on slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
        git \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Node.js (claude CLI runtime) + the pinned claude CLI -------------------
# NodeSource pinned-major setup; then a PINNED claude-code install (never
# floating).  npm cache is purged to keep the layer slim.
RUN curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/* \
    && node --version \
    && claude --version

# ---- Python deps (backend; CPU-only, NO torch/CUDA) -------------------------
# Copy the dependency manifest first so this layer caches across source edits.
RUN python -m venv "$VIRTUAL_ENV"
WORKDIR /app
COPY pyproject.toml ./
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --upgrade pip wheel \
    && pip install -r backend/requirements.txt

# ---- Application source ------------------------------------------------------
# The Helm Deployment/CronJob supplies the command (python -m backend.cli
# reproduce ...), so we COPY the backend + the runtime fixtures the CLI/ingest
# path references, but set NO CMD that runs anything.
COPY backend/ ./backend/
COPY third_party/ ./third_party/
COPY paperbench1.pdf ./paperbench1.pdf
COPY demo_paper.pdf ./demo_paper.pdf

# Run state directory (mounted/written at runtime).
RUN mkdir -p /app/runs

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---- Non-root runtime user ---------------------------------------------------
# Mirrors the root image (uid 10001, app).  HOME must be writable so the
# `claude` CLI can read/write ~/.claude (its OAuth/session state).
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER app
ENV HOME=/home/app

# tini as PID 1 for signal-safe child reaping.  NO CMD: the Helm
# Deployment/CronJob `command:` is the sole entrypoint for what runs.
ENTRYPOINT ["/usr/bin/tini", "--"]
