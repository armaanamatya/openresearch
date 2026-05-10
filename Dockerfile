# syntax=docker/dockerfile:1.6
#
# OpenResearch — full-stack image (FastAPI backend + Next.js frontend).
#
# Three-stage build:
#   1. python-deps  — installs the project's Python dependencies into a venv
#   2. frontend     — `npm ci` + `next build` for the Next.js frontend
#   3. runtime      — slim final image: Python venv + Next standalone-ish output
#
# The final image talks to a HOST Docker daemon by sharing the docker socket
# (mount /var/run/docker.sock at runtime). This keeps `LocalDockerBackend`
# working for inner sandbox runs without nesting daemons (DinD). Trade-off:
# the container effectively has root on the host's Docker; fine for local dev,
# NOT for prod. For prod, the RunPodBackend (which this image already supports)
# is the correct compute path.

# --------------------------------------------------------------------------- #
# Stage 1 — Python dependency build                                            #
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS python-deps

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

# Copy dependency manifest first so this layer caches across source edits.
WORKDIR /build
COPY pyproject.toml ./
COPY backend/requirements.txt ./backend/requirements.txt

# Install runtime deps. We install via -r so the pinned set in
# backend/requirements.txt is the source of truth for the image; pyproject
# is still copied above to satisfy any setuptools install metadata.
RUN pip install --upgrade pip wheel \
    && pip install -r backend/requirements.txt

# --------------------------------------------------------------------------- #
# Stage 2 — Frontend build                                                     #
# --------------------------------------------------------------------------- #
FROM node:20-bookworm-slim AS frontend

WORKDIR /frontend

# Lockfile-aware deps install (cached).
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund --prefer-offline

# App source + build.
COPY frontend/ ./
# Next.js doesn't require a `public/` directory and this repo doesn't ship
# one, but the runtime stage unconditionally COPYs it. Create it here so
# the multi-stage COPY always has a (possibly empty) source.
RUN mkdir -p public
# Build skips Turbopack dev cache; production build emits to .next/.
RUN rm -rf .next/dev .next/cache \
    && npm run build

# --------------------------------------------------------------------------- #
# Stage 3 — Runtime image                                                      #
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:$PATH \
    NODE_ENV=production

# Runtime needs:
#   - tini (PID 1, signal-safe)
#   - docker CLI (the Python `docker` SDK uses it for some operations)
#   - Node.js 20 (to serve `next start`)
#   - curl (healthcheck convenience)
#   - openssh-client (asyncssh / Runpod backend)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
        ca-certificates \
        curl \
        gnupg \
        openssh-client \
        docker.io \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Bring the pre-built Python venv from stage 1.
COPY --from=python-deps /opt/venv /opt/venv

# App layout.
WORKDIR /app
COPY backend/ ./backend/
COPY pyproject.toml ./
COPY start.sh ./
COPY third_party/ ./third_party/
COPY paperbench1.pdf ./paperbench1.pdf
COPY demo_paper.pdf ./demo_paper.pdf

# Frontend production build + node_modules (we serve via `next start`, not
# `next start --turbopack`, so no Turbopack runtime needed in the image).
COPY --from=frontend /frontend/.next ./frontend/.next
COPY --from=frontend /frontend/public ./frontend/public
COPY --from=frontend /frontend/node_modules ./frontend/node_modules
COPY --from=frontend /frontend/package.json ./frontend/package.json
COPY --from=frontend /frontend/next.config.ts ./frontend/next.config.ts

# Runs directory is a volume in compose; create the mount point + memory file
# parent so first-boot doesn't fail before the volume is mounted.
RUN mkdir -p /app/runs

# Single entrypoint runs both servers and forwards signals to children.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 3000
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
