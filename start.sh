#!/usr/bin/env bash
# Dashboard launcher with Runpod-as-default and robust preflight inputs.
#
# Behavior:
#   1. Defaults dashboard sandbox to "runpod" unless user overrides
#      OPENRESEARCH_DEFAULT_SANDBOX.
#   2. Runs scripts/runpod_check.sh only when sandbox is "runpod"
#      (or START_FORCE_RUNPOD_PREFLIGHT=1).
#   3. Boots the full local dev stack: backend (uvicorn, --reload) + frontend
#      (Next.js dev server), signal-forwarded and watchdog-linked so either
#      one dying tears down the other. Mirrors docker/entrypoint.sh's prod
#      two-process pattern and scripts/dev.sh's dual launcher.
#
# Escape hatches:
#   START_SKIP_PREFLIGHT=1 ./start.sh   # skip runpod preflight when selected
#   START_FORCE_RUNPOD_PREFLIGHT=1 ./start.sh
#                                       # run runpod preflight even when
#                                       # sandbox is not runpod
#   START_FULL_SMOKE=1 ./start.sh       # also boot a real pod, run nvidia-smi
#                                       # over SSH, destroy it. COSTS MONEY
#                                       # (cents-scale on RTX 4090). Use when
#                                       # you want end-to-end confidence
#                                       # before kicking off a long pipeline.
#   OPENRESEARCH_DEFAULT_SANDBOX=local ./start.sh
#                                       # temporarily force local dashboard default
#   START_BACKEND_ONLY=1 ./start.sh     # backend only — no Node/frontend needed
#   START_FRONTEND_ONLY=1 ./start.sh    # frontend only — backend assumed to be
#                                       # running elsewhere
set -euo pipefail
cd "$(dirname "$0")"

PREFLIGHT="${1:-${PREFLIGHT_SCRIPT:-scripts/runpod_check.sh}}"
ENV_FILE=".env"

# Shared dotenv-grammar .env reader (env_value_from_file). Extracted so the
# parse is pinned to python-dotenv's semantics by
# tests/scripts/test_env_file_parsers.py — the old inline copy kept trailing
# `# comments` in values and the corrupted export outranked pydantic's own
# parse (hard ValidationError on Literal fields at boot).
. scripts/lib/env_file.sh

# 1. Default sandbox for the dashboard: shell env > .env > runpod.
# Consulting .env here matters: this export becomes real process env, which
# pydantic-settings ranks ABOVE the .env file — exporting "runpod"
# unconditionally would silently shadow a `OPENRESEARCH_DEFAULT_SANDBOX=local`
# line the operator put in .env.
if [[ -z "${OPENRESEARCH_DEFAULT_SANDBOX:-}" ]]; then
    OPENRESEARCH_DEFAULT_SANDBOX="$(env_value_from_file OPENRESEARCH_DEFAULT_SANDBOX "${ENV_FILE}" || true)"
fi
export OPENRESEARCH_DEFAULT_SANDBOX="${OPENRESEARCH_DEFAULT_SANDBOX:-runpod}"
echo "[start.sh] Dashboard default sandbox: ${OPENRESEARCH_DEFAULT_SANDBOX}"

# If the public key is missing but we have a private key, derive it so
# preflight/startup remains stable with minimal .env setup.
if [[ -z "${OPENRESEARCH_RUNPOD_SSH_PUBLIC_KEY:-}" ]]; then
    if [[ -z "${OPENRESEARCH_RUNPOD_SSH_KEY_PATH:-}" ]]; then
        OPENRESEARCH_RUNPOD_SSH_KEY_PATH="$(env_value_from_file OPENRESEARCH_RUNPOD_SSH_KEY_PATH "${ENV_FILE}" || true)"
        export OPENRESEARCH_RUNPOD_SSH_KEY_PATH
    fi
    if [[ -n "${OPENRESEARCH_RUNPOD_SSH_KEY_PATH:-}" ]]; then
        # Parameter expansion, NOT `eval echo`: the value comes from .env and
        # eval would execute anything shell-special pasted into it.
        ssh_key_path="${OPENRESEARCH_RUNPOD_SSH_KEY_PATH/#\~/$HOME}"
        if [[ -f "${ssh_key_path}" ]] && command -v ssh-keygen >/dev/null 2>&1; then
            derived_pub="$(ssh-keygen -y -f "${ssh_key_path}" 2>/dev/null || true)"
            if [[ -n "${derived_pub}" ]]; then
                export OPENRESEARCH_RUNPOD_SSH_PUBLIC_KEY="${derived_pub}"
                echo "[start.sh] Derived OPENRESEARCH_RUNPOD_SSH_PUBLIC_KEY from ${ssh_key_path}."
            fi
        fi
    fi
fi

# 2. Runpod preflight (when relevant, and skippable).
runpod_preflight_needed=0
if [[ "${OPENRESEARCH_DEFAULT_SANDBOX}" == "runpod" ]]; then
    runpod_preflight_needed=1
fi
if [[ "${START_FORCE_RUNPOD_PREFLIGHT:-0}" == "1" ]]; then
    runpod_preflight_needed=1
fi
if [[ "${START_FULL_SMOKE:-0}" == "1" ]]; then
    runpod_preflight_needed=1
fi

if [[ "${runpod_preflight_needed}" == "1" ]]; then
    if [[ "${START_SKIP_PREFLIGHT:-0}" != "1" ]]; then
        if [[ -x "${PREFLIGHT}" ]]; then
            # Default = free preflight (auth + ssh key + env vars).
            # START_FULL_SMOKE=1 also boots a real pod, runs nvidia-smi, destroys
            # it — the only definitive proof the configured GPU is bookable from
            # this account, since the REST v1 API doesn't expose GPU listings.
            preflight_args=()
            if [[ "${START_FULL_SMOKE:-0}" == "1" ]]; then
                echo "[start.sh] START_FULL_SMOKE=1 — running end-to-end pod smoke (this WILL spend money)."
                preflight_args+=("--start-pod")
            else
                echo "[start.sh] Running Runpod preflight (free)..."
            fi
            # macOS bash 3.2: ${arr[@]} on an empty array fires "unbound
            # variable" under `set -u`. The `${arr[@]+...}` form expands to
            # nothing when the array is empty/unset, sidestepping it.
            if ! "${PREFLIGHT}" ${preflight_args[@]+"${preflight_args[@]}"}; then
                echo "[start.sh] Runpod preflight FAILED — refusing to start the dashboard."
                echo "[start.sh] Fix the issue, or rerun with START_SKIP_PREFLIGHT=1 to bypass."
                exit 1
            fi
        else
            echo "[start.sh] Preflight script not found at ${PREFLIGHT}; skipping."
        fi
    else
        echo "[start.sh] START_SKIP_PREFLIGHT=1 — skipping Runpod preflight."
    fi
else
    echo "[start.sh] Runpod preflight not required for sandbox=${OPENRESEARCH_DEFAULT_SANDBOX}."
fi

# 2a. GKE preflight (when sandbox is gcp or gke; skippable). Mirrors the runpod
# block: free read-only checks by default; START_FULL_SMOKE=1 passes --start-pod
# (an operator-gated, money-spending stub — exit 6). gke is an alias for gcp.
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
        # macOS bash 3.2 empty-array guard (see runpod block).
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
# for sandbox `docker` and `auto`/unknown (LocalDockerBackend). `local`, `runpod`,
# and `gcp`/`gke` short-circuit build_environment to a no-op — runpod boots its own
# pod image over SSH, gcp/gke run a pre-baked Artifact Registry image on the GKE
# cluster (primitives.build_environment _sb_key=="gcp") — so none needs a local
# daemon. A down daemon makes only docker/auto runs fail at build_environment with
# backend_unavailable, so surface it at startup for those modes. Warn (don't
# refuse): a per-run --sandbox override changes the requirement, and the dashboard
# can outlive a daemon restart.
if [[ "${OPENRESEARCH_DEFAULT_SANDBOX}" != "local" && "${OPENRESEARCH_DEFAULT_SANDBOX}" != "runpod" && "${OPENRESEARCH_DEFAULT_SANDBOX}" != "gcp" && "${OPENRESEARCH_DEFAULT_SANDBOX}" != "gke" && "${START_SKIP_PREFLIGHT:-0}" != "1" ]]; then
    if ! command -v docker >/dev/null 2>&1; then
        echo "[start.sh] WARNING: 'docker' CLI not found — runs with sandbox in {docker,auto} will fail at build_environment. Install OrbStack/Docker, or use --sandbox local/runpod."
    elif ! docker info >/dev/null 2>&1; then
        echo "[start.sh] WARNING: Docker daemon not reachable (sandbox=${OPENRESEARCH_DEFAULT_SANDBOX})."
        echo "[start.sh]          build_environment runs a LOCAL docker build for sandbox docker/auto —"
        echo "[start.sh]          so those runs will fail with backend_unavailable until it is up."
        echo "[start.sh]          Start OrbStack/Docker Desktop (verify: 'docker info'), or run with --sandbox local/runpod."
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
# vice versa). The `|| EXIT_CODE=$?` is load-bearing: this file runs under
# `set -euo pipefail`, so a bare `wait -n`/`wait` returning nonzero would kill
# this script immediately, skipping the teardown of the surviving process
# below (matches docker/entrypoint.sh's watchdog).
if [[ -n "${BACKEND_PID}" && -n "${FRONTEND_PID}" ]]; then
    wait -n "${BACKEND_PID}" "${FRONTEND_PID}" && EXIT_CODE=0 || EXIT_CODE=$?
    echo "[start.sh] one of (backend=${BACKEND_PID}, frontend=${FRONTEND_PID}) exited with ${EXIT_CODE}; tearing down"
    kill -TERM "${BACKEND_PID}" "${FRONTEND_PID}" 2>/dev/null || true
    wait "${BACKEND_PID}" "${FRONTEND_PID}" 2>/dev/null || true
else
    # Single-process mode (START_FRONTEND_ONLY=1): nothing to multiplex.
    SOLE_PID="${BACKEND_PID}${FRONTEND_PID}"
    wait "${SOLE_PID}" && EXIT_CODE=0 || EXIT_CODE=$?
fi
exit "${EXIT_CODE}"
