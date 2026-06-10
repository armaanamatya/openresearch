#!/usr/bin/env bash
# Dashboard launcher with Runpod-as-default and robust preflight inputs.
#
# Behavior:
#   1. Defaults dashboard sandbox to "runpod" unless user overrides
#      OPENRESEARCH_DEFAULT_SANDBOX.
#   2. Runs scripts/runpod_check.sh only when sandbox is "runpod"
#      (or START_FORCE_RUNPOD_PREFLIGHT=1).
#   3. Execs uvicorn for the FastAPI factory.
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

# 2b. Docker daemon preflight. build_environment does a LOCAL `docker build` only
# for sandbox `docker` and `auto`/unknown (LocalDockerBackend). Both `local` and
# `runpod` short-circuit build_environment to a no-op — runpod boots its own pod
# image over SSH — so neither needs a daemon. A down daemon makes only docker/auto
# runs fail at build_environment with backend_unavailable, so surface it at startup
# for those modes. Warn (don't refuse): a per-run --sandbox override changes the
# requirement, and the dashboard can outlive a daemon restart.
if [[ "${OPENRESEARCH_DEFAULT_SANDBOX}" != "local" && "${OPENRESEARCH_DEFAULT_SANDBOX}" != "runpod" && "${START_SKIP_PREFLIGHT:-0}" != "1" ]]; then
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

# 3. Boot the API.
if [[ ! -x .venv/bin/uvicorn ]]; then
    echo "[start.sh] .venv/bin/uvicorn not found. Create the venv first:"
    echo "[start.sh]   python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt"
    exit 1
fi
exec .venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000
