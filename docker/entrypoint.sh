#!/usr/bin/env bash
# Container entrypoint — boots backend (uvicorn) + frontend (next start)
# under one PID and forwards signals to both.

set -euo pipefail

cd /app

# Compose mounts .env read-only for local development. Source it here rather
# than using docker-compose env_file so `docker compose config` does not print
# secret values.
if [[ -f /app/.env ]]; then
    set -a
    # shellcheck disable=SC1091
    source /app/.env
    set +a
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
(cd /app/frontend && npx next start --hostname 0.0.0.0 --port 3000) &
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
wait -n "$BACKEND_PID" "$FRONTEND_PID"
EXIT_CODE=$?
echo "[entrypoint] one of (backend=$BACKEND_PID, frontend=$FRONTEND_PID) exited with $EXIT_CODE; tearing down" >&2
kill -TERM "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
exit "$EXIT_CODE"
