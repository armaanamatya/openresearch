#!/usr/bin/env bash
# Container entrypoint — boots backend (uvicorn) + frontend (next start)
# under one PID and forwards signals to both.

set -euo pipefail

cd /app

# Compose mounts .env read-only for local development. Load it here rather
# than using docker-compose env_file so `docker compose config` does not print
# secret values. Parse it as KEY=VALUE *data* instead of `source`ing it:
# python-dotenv accepts unquoted values with spaces, so a perfectly valid
# `OPENRESEARCH_RUNPOD_GPU_TYPE=NVIDIA GeForce RTX 4090` line made bash run
# `GeForce` as a command and kill the container with exit 127 under set -e
# (and `source` would happily execute $(...) in values). Vars already set in
# the container environment (compose `environment:`, `docker run -e`) keep
# winning over .env — the normal compose precedence; without that, a copied
# .env.example silently overrode the compose-set OPENRESEARCH_DATABASE_URL
# and broke event-store persistence.
if [[ -f /app/.env ]]; then
    while IFS= read -r _line || [[ -n "$_line" ]]; do
        [[ -z "$_line" || "$_line" =~ ^[[:space:]]*# ]] && continue
        [[ "$_line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
        _key="${BASH_REMATCH[1]}"
        _value="${BASH_REMATCH[2]}"
        if [[ "$_value" =~ ^\"(.*)\"$ ]]; then _value="${BASH_REMATCH[1]}"; fi
        if [[ "$_value" =~ ^\'(.*)\'$ ]]; then _value="${BASH_REMATCH[1]}"; fi
        if [[ -z "${!_key+x}" ]]; then
            export "${_key}=${_value}"
        fi
    done < /app/.env
    unset _line _key _value
fi

# --- SSH key injection (Railway / env-only deployments) ---------------------
# Railway can't mount files, so inject the private key as a base64 env var.
# Set OPENRESEARCH_RUNPOD_SSH_KEY_B64 in Railway Variables and this block writes
# it to disk and points OPENRESEARCH_RUNPOD_SSH_KEY_PATH at it automatically.
if [[ -n "${OPENRESEARCH_RUNPOD_SSH_KEY_B64:-}" ]]; then
    mkdir -p /root/.ssh
    echo "$OPENRESEARCH_RUNPOD_SSH_KEY_B64" | base64 -d > /root/.ssh/runpod_id_rsa
    chmod 600 /root/.ssh/runpod_id_rsa
    export OPENRESEARCH_RUNPOD_SSH_KEY_PATH=/root/.ssh/runpod_id_rsa
fi

# --- Backend: FastAPI via uvicorn -------------------------------------------
# Uses the venv copied from the python-deps stage. No --reload in prod.
/opt/venv/bin/python -m uvicorn backend.app:create_app \
    --factory \
    --host 0.0.0.0 \
    --port 8000 \
    --proxy-headers \
    --forwarded-allow-ips "*" &
BACKEND_PID=$!

# --- Frontend: Next.js production server ------------------------------------
# Bound to 0.0.0.0 so the host can hit it through the published port.
(cd /app/frontend && npx next start --hostname 0.0.0.0 --port "${PORT:-3000}") &
FRONTEND_PID=$!

# --- Signal forwarding ------------------------------------------------------
# tini handles PID-1 reaping; this trap propagates SIGTERM/SIGINT to children
# so `docker stop` is fast (10 s default grace) instead of waiting on the
# default 30 s SIGKILL.
trap 'echo "[entrypoint] forwarding shutdown" >&2; \
      kill -TERM "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null; \
      wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null; \
      exit 0' TERM INT

# --- Watchdog: exit if either child dies ------------------------------------
# wait -n returns when ANY background job exits. We then propagate that exit
# code so docker compose treats the container as failed (lets restart policy
# handle it instead of hanging with one healthy and one dead service).
# The `|| EXIT_CODE=$?` is load-bearing: under `set -e` a bare `wait -n` that
# returns nonzero would kill this script instantly, skipping the SIGTERM
# teardown below (the surviving child then gets namespace-SIGKILLed — which
# has corrupted the SQLite event store before).
wait -n "$BACKEND_PID" "$FRONTEND_PID" && EXIT_CODE=0 || EXIT_CODE=$?
echo "[entrypoint] one of (backend=$BACKEND_PID, frontend=$FRONTEND_PID) exited with $EXIT_CODE; tearing down" >&2
kill -TERM "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
exit "$EXIT_CODE"
