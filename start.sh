#!/usr/bin/env bash
# Dashboard launcher: local-by-default with a GKE (gcp) preflight and robust
# dev-stack supervision.
#
# Behavior:
#   1. Defaults dashboard sandbox to "local" unless the operator overrides
#      OPENRESEARCH_DEFAULT_SANDBOX (shell env > .env > local). Paid clouds
#      (azure/aws) and the GCP single-VM path are opt-in per run; gcp/gke
#      routes to the PARKED GKE backend (raises unless OPENRESEARCH_ALLOW_GKE=1).
#   2. Runs scripts/gke_check.sh only when sandbox is "gcp"/"gke" (skippable).
#   3. Boots the full local dev stack: backend (uvicorn, --reload) + frontend
#      (Next.js dev server), signal-forwarded and watchdog-linked so either
#      one dying tears down the other. Mirrors docker/entrypoint.sh's prod
#      two-process pattern and scripts/dev.sh's dual launcher.
#
# Escape hatches:
#   START_SKIP_PREFLIGHT=1 ./start.sh    # skip the gcp/gke preflight
#   START_FULL_SMOKE=1 ./start.sh        # also run the (operator-gated) pod smoke
#   OPENRESEARCH_DEFAULT_SANDBOX=azure ./start.sh
#                                        # temporarily force a cloud dashboard default
#   START_BACKEND_ONLY=1 ./start.sh      # backend only — no Node/frontend needed
#   START_FRONTEND_ONLY=1 ./start.sh     # frontend only — backend assumed to be
#                                        # running elsewhere
set -euo pipefail
cd "$(dirname "$0")"

ENV_FILE=".env"

# Shared dotenv-grammar .env reader (env_value_from_file). Extracted so the
# parse is pinned to python-dotenv's semantics by
# tests/scripts/test_env_file_parsers.py — the old inline copy kept trailing
# `# comments` in values and the corrupted export outranked pydantic's own
# parse (hard ValidationError on Literal fields at boot).
. scripts/lib/env_file.sh

# 1. Default sandbox for the dashboard: shell env > .env > local.
# Consulting .env here matters: this export becomes real process env, which
# pydantic-settings ranks ABOVE the .env file — exporting a value
# unconditionally would silently shadow an OPENRESEARCH_DEFAULT_SANDBOX line
# the operator put in .env.
if [[ -z "${OPENRESEARCH_DEFAULT_SANDBOX:-}" ]]; then
    OPENRESEARCH_DEFAULT_SANDBOX="$(env_value_from_file OPENRESEARCH_DEFAULT_SANDBOX "${ENV_FILE}" || true)"
fi
export OPENRESEARCH_DEFAULT_SANDBOX="${OPENRESEARCH_DEFAULT_SANDBOX:-local}"
echo "[start.sh] Dashboard default sandbox: ${OPENRESEARCH_DEFAULT_SANDBOX}"

# 2. GKE preflight (when sandbox is gcp or gke; skippable). Free read-only
# checks by default; START_FULL_SMOKE=1 passes --start-pod (an operator-gated,
# money-spending stub — exit 6). gke is an alias for gcp. Note: the gcp/gke
# backend is PARKED (raises unless OPENRESEARCH_ALLOW_GKE=1) — the preflight
# stays available so the path can be exercised once IAM perms land.
GKE_PREFLIGHT="scripts/gke_check.sh"
if [[ "${OPENRESEARCH_DEFAULT_SANDBOX}" == "gcp" || "${OPENRESEARCH_DEFAULT_SANDBOX}" == "gke" ]]; then
    if [[ "${START_SKIP_PREFLIGHT:-0}" != "1" && -x "${GKE_PREFLIGHT}" ]]; then
        gke_args=()
        if [[ "${START_FULL_SMOKE:-0}" == "1" ]]; then
            echo "[start.sh] START_FULL_SMOKE=1 — running GKE pod smoke (operator-gated stub; would spend money)."
            gke_args+=("--start-pod")
        else
            echo "[start.sh] Running GKE preflight (free)..."
        fi
        # macOS bash 3.2 empty-array guard.
        gke_rc=0
        "${GKE_PREFLIGHT}" ${gke_args[@]+"${gke_args[@]}"} || gke_rc=$?
        if [[ "${gke_rc}" -eq 6 ]]; then
            # exit 6 == the --start-pod smoke is an intentionally-unimplemented
            # operator-gated stub. Treat it as a NON-FATAL skip so
            # START_FULL_SMOKE=1 doesn't brick GKE startup; the free preflight
            # checks above already ran. Any OTHER non-zero stays fatal — including
            # exit 7 (a configured OPENRESEARCH_GCP_GPU_SKUS entry has no matching
            # live node pool: config/Terraform drift that would otherwise leave a
            # cell Pending until capacity_exhausted, ~15-25 min in).
            echo "[start.sh] GKE pod smoke unimplemented (exit 6) — skipping smoke, continuing startup."
        elif [[ "${gke_rc}" -ne 0 ]]; then
            echo "[start.sh] GKE preflight FAILED (exit ${gke_rc}) — refusing to start (set START_SKIP_PREFLIGHT=1 to bypass)."
            exit 1
        fi
    elif [[ "${START_SKIP_PREFLIGHT:-0}" == "1" ]]; then
        echo "[start.sh] START_SKIP_PREFLIGHT=1 — skipping GKE preflight."
    fi
fi

# 2b. Docker daemon preflight. build_environment does a LOCAL `docker build` only
# for sandbox `docker` and `auto`/unknown (LocalDockerBackend). `local` and
# `gcp`/`gke` short-circuit build_environment to a no-op — gcp/gke run a pre-baked
# Artifact Registry image on the GKE cluster (primitives.build_environment
# _sb_key=="gcp") — so neither needs a local daemon. A down daemon makes only
# docker/auto runs fail at build_environment with backend_unavailable, so surface
# it at startup for those modes. Warn (don't refuse): a per-run --sandbox override
# changes the requirement, and the dashboard can outlive a daemon restart.
if [[ "${OPENRESEARCH_DEFAULT_SANDBOX}" != "local" && "${OPENRESEARCH_DEFAULT_SANDBOX}" != "gcp" && "${OPENRESEARCH_DEFAULT_SANDBOX}" != "gke" && "${START_SKIP_PREFLIGHT:-0}" != "1" ]]; then
    if ! command -v docker >/dev/null 2>&1; then
        echo "[start.sh] WARNING: 'docker' CLI not found — runs with sandbox in {docker,auto} will fail at build_environment. Install OrbStack/Docker, or use --sandbox local."
    elif ! docker info >/dev/null 2>&1; then
        echo "[start.sh] WARNING: Docker daemon not reachable (sandbox=${OPENRESEARCH_DEFAULT_SANDBOX})."
        echo "[start.sh]          build_environment runs a LOCAL docker build for sandbox docker/auto —"
        echo "[start.sh]          so those runs will fail with backend_unavailable until it is up."
        echo "[start.sh]          Start OrbStack/Docker Desktop (verify: 'docker info'), or run with --sandbox local."
    else
        echo "[start.sh] Docker daemon reachable."
    fi
fi

# 3. Boot the stack: backend (uvicorn) + frontend (Next.js dev), or one side
# only via the START_BACKEND_ONLY / START_FRONTEND_ONLY escape hatches.
if [[ ! -x .venv/bin/uvicorn ]]; then
    echo "[start.sh] .venv/bin/uvicorn not found. Create the venv first:"
    echo "[start.sh]   python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt"
    exit 1
fi

if [[ "${START_BACKEND_ONLY:-0}" == "1" && "${START_FRONTEND_ONLY:-0}" == "1" ]]; then
    echo "[start.sh] START_BACKEND_ONLY=1 and START_FRONTEND_ONLY=1 are mutually exclusive."
    exit 2
fi

# 3a. Backend-only escape hatch: no Node/frontend involved at all — exec as
# the previous single-process start.sh did.
if [[ "${START_BACKEND_ONLY:-0}" == "1" ]]; then
    echo "[start.sh] START_BACKEND_ONLY=1 — starting backend only."
    exec .venv/bin/uvicorn backend.app:create_app --factory --reload --reload-dir backend --port 8000
fi

# 3b. Node selection for the frontend. System `node` can be outside Next's
# required range (>=20.19 <21 || >=22.12) — e.g. a distro node like v21.x.
# Source nvm (if installed) and select an in-range version. Do NOT trust
# frontend/.nvmrc — it pins 22, which may not actually be installed on this
# machine; 20.20.2 is the version validated for this repo's frontend
# toolchain, so it is tried first.
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm use 20.20.2 >/dev/null 2>&1 || nvm use 20 >/dev/null 2>&1 || true

node_in_range=0
node_version="not found"
if command -v node >/dev/null 2>&1; then
    node_version="$(node --version)"
    # major==20 with minor>=19, or major==22 with minor>=12 (Next's range).
    node_in_range="$(echo "${node_version}" | awk -F'[v.]' '{major=$2+0; minor=$3+0; ok=((major==20 && minor>=19) || (major==22 && minor>=12)); print ok+0}')"
fi

if [[ "${node_in_range}" != "1" ]]; then
    echo "[start.sh] ERROR: frontend needs Node >=20.19 <21 or >=22.12 (found: ${node_version})."
    echo "[start.sh]        Install one, e.g.: nvm install 22"
    echo "[start.sh]        Or run backend-only instead: START_BACKEND_ONLY=1 ./start.sh"
    exit 1
fi
echo "[start.sh] Using node ${node_version} for the frontend."

# 3c. Frontend deps guard.
if [[ ! -d frontend/node_modules ]]; then
    echo "[start.sh] frontend/node_modules missing — running 'npm ci' (this may take a minute)..."
    (cd frontend && npm ci)
fi

# 3d. Launch both processes in the background so we can signal-forward and
# watchdog them (mirrors docker/entrypoint.sh + scripts/dev.sh).
BACKEND_PID=""
FRONTEND_PID=""

if [[ "${START_FRONTEND_ONLY:-0}" != "1" ]]; then
    .venv/bin/uvicorn backend.app:create_app --factory --reload --reload-dir backend --port 8000 &
    BACKEND_PID=$!
    echo "[start.sh] backend  pid=${BACKEND_PID}  http://127.0.0.1:8000"
fi

(cd frontend && OPENRESEARCH_BACKEND_URL="http://127.0.0.1:8000" npm run dev) &
FRONTEND_PID=$!
echo "[start.sh] frontend pid=${FRONTEND_PID}  http://localhost:3000"

# 3e. Signal forwarding: propagate SIGTERM/SIGINT to whichever children are
# running so Ctrl-C (or an orchestrator's `kill`) tears down both promptly
# instead of leaving an orphaned uvicorn/next process bound to the port.
trap 'echo "[start.sh] forwarding shutdown" >&2; \
      kill -TERM ${BACKEND_PID:+"$BACKEND_PID"} ${FRONTEND_PID:+"$FRONTEND_PID"} 2>/dev/null; \
      wait ${BACKEND_PID:+"$BACKEND_PID"} ${FRONTEND_PID:+"$FRONTEND_PID"} 2>/dev/null; \
      exit 0' TERM INT

# 3f. Watchdog: exit (tearing down the survivor) as soon as EITHER child
# dies, so a crashed backend doesn't leave a zombie frontend serving 502s (or
# vice versa).
#
# PORTABILITY (BUG: macOS): `wait -n` is bash 4.3+, but the default macOS
# /bin/bash is 3.2 — under `set -euo pipefail` `wait -n` fails INSTANTLY with
# "wait: -n: invalid option", which the `|| EXIT_CODE=$?` catches as a crash and
# tears both servers down before they ever serve a request. So block by POLLING
# both PIDs with `kill -0` (presence check, no signal) until one exits — the same
# bash-3.2-safe pattern scripts/dev.sh already uses. Then reap the dead child for
# its real exit code. `kill -0`/`wait` in a loop-condition or `&&/||` chain don't
# trip `set -e`, so the teardown below always runs.
EXIT_CODE=0
if [[ -n "${BACKEND_PID}" && -n "${FRONTEND_PID}" ]]; then
    while kill -0 "${BACKEND_PID}" 2>/dev/null && kill -0 "${FRONTEND_PID}" 2>/dev/null; do
        sleep 1
    done
    if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
        wait "${BACKEND_PID}" && EXIT_CODE=0 || EXIT_CODE=$?
    else
        wait "${FRONTEND_PID}" && EXIT_CODE=0 || EXIT_CODE=$?
    fi
    echo "[start.sh] one of (backend=${BACKEND_PID}, frontend=${FRONTEND_PID}) exited with ${EXIT_CODE}; tearing down"
    kill -TERM "${BACKEND_PID}" "${FRONTEND_PID}" 2>/dev/null || true
    wait "${BACKEND_PID}" "${FRONTEND_PID}" 2>/dev/null || true
else
    # Single-process mode (START_FRONTEND_ONLY=1): nothing to multiplex.
    SOLE_PID="${BACKEND_PID}${FRONTEND_PID}"
    wait "${SOLE_PID}" && EXIT_CODE=0 || EXIT_CODE=$?
fi
exit "${EXIT_CODE}"
