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
# Digest-pinned (audit 2026-06-10; bases corrected 2026-06-11): a moving
# :3.12-slim tag silently changes the build between runs — the first digest
# grab proved it by quietly landing on Debian TRIXIE while node stayed on
# bookworm, breaking the same-base invariant the cross-stage node copy
# depends on. Both stages now pin EXPLICITLY-bookworm MULTI-ARCH INDEX
# digests (docker buildx imagetools inspect — never `docker images
# --digests`, which can return a single-platform digest that breaks the
# other arch). Node 22 = active LTS, matches CI's node-version and the
# engines range. Bump deliberately, both stages together.
FROM python:3.12-slim-bookworm@sha256:76d4b7b6305788c6b4c6a19d6a22a3921bf802e9af4d5e1e5bd771208dba74bf AS python-deps

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
FROM node:22-bookworm-slim@sha256:e21fc383b50d5347dc7a9f1cae45b8f4e2f0d39f7ade28e4eef7d2934522b752 AS frontend

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
FROM python:3.12-slim-bookworm@sha256:76d4b7b6305788c6b4c6a19d6a22a3921bf802e9af4d5e1e5bd771208dba74bf AS runtime

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
        openssh-client \
        docker.io \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Node comes from the frontend builder stage (same Debian bookworm base —
# pinned explicitly; see the digest note on stage 1), so
# the runtime serves with EXACTLY the Node that built the app. This replaces
# the old `curl -fsSL https://deb.nodesource.com/setup_20.x | bash -` — an
# unpinned remote script piped to root at build time, which could also drift
# the serving Node away from the build Node (audit 2026-06-09). gnupg was
# only needed for that nodesource flow and is gone with it.
COPY --from=frontend /usr/local/bin/node /usr/local/bin/node
COPY --from=frontend /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && node --version && npx --version

# Bring the pre-built Python venv from stage 1.
COPY --from=python-deps /opt/venv /opt/venv

# App layout. (start.sh is deliberately NOT copied: it is the host launcher —
# it references .venv/bin/uvicorn, which does not exist in this image; the
# container boots via /entrypoint.sh below.)
WORKDIR /app
COPY backend/ ./backend/
COPY pyproject.toml ./
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
# load_env.sh is the python-dotenv-delegating .env loader the entrypoint
# sources (kept as a separate file so tests pin its parse to dotenv_values).
COPY docker/load_env.sh /load_env.sh
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Non-root runtime user (audit 2026-06-10). Both servers run as `app`; the
# docker-socket capability is granted at runtime via compose `group_add`
# (see docker-compose.yml) — membership in the socket's group is still
# root-equivalent ON THE HOST by design (LocalDockerBackend needs it for
# inner sandbox runs), but the processes themselves no longer run as root:
# no root-owned FS writes, no setuid surface, container escapes land on an
# unprivileged uid. The runs/ volume is chowned at first boot by compose's
# bind mount semantics on macOS; on Linux ensure ./runs is writable by
# uid 10001 (or run `chown -R 10001 runs/`).
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER app
ENV HOME=/home/app

EXPOSE 8000 3000
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
