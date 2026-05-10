#!/usr/bin/env bash
# Dashboard launcher with Runpod-as-default and robust preflight inputs.
#
# Behavior:
#   1. Defaults dashboard sandbox to "runpod" unless user overrides
#      REPROLAB_DEFAULT_SANDBOX.
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
#   REPROLAB_DEFAULT_SANDBOX=local ./start.sh
#                                       # temporarily force local dashboard default
set -euo pipefail
cd "$(dirname "$0")"

PREFLIGHT="${1:-${PREFLIGHT_SCRIPT:-scripts/runpod_check.sh}}"
ENV_FILE=".env"

env_value_from_file() {
    local key="$1"
    local file="$2"
    [[ -f "${file}" ]] || return 1
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        if [[ "$line" =~ ^[[:space:]]*${key}=(.*)$ ]]; then
            local value="${BASH_REMATCH[1]}"
            if [[ "$value" =~ ^\"(.*)\"$ ]]; then value="${BASH_REMATCH[1]}"; fi
            if [[ "$value" =~ ^\'(.*)\'$ ]]; then value="${BASH_REMATCH[1]}"; fi
            printf "%s" "$value"
            return 0
        fi
    done < "${file}"
    return 1
}

# 1. Default sandbox for the dashboard. Honor explicit override.
export REPROLAB_DEFAULT_SANDBOX="${REPROLAB_DEFAULT_SANDBOX:-runpod}"
echo "[start.sh] Dashboard default sandbox: ${REPROLAB_DEFAULT_SANDBOX}"

# If the public key is missing but we have a private key, derive it so
# preflight/startup remains stable with minimal .env setup.
if [[ -z "${REPROLAB_RUNPOD_SSH_PUBLIC_KEY:-}" ]]; then
    if [[ -z "${REPROLAB_RUNPOD_SSH_KEY_PATH:-}" ]]; then
        REPROLAB_RUNPOD_SSH_KEY_PATH="$(env_value_from_file REPROLAB_RUNPOD_SSH_KEY_PATH "${ENV_FILE}" || true)"
        export REPROLAB_RUNPOD_SSH_KEY_PATH
    fi
    if [[ -n "${REPROLAB_RUNPOD_SSH_KEY_PATH:-}" ]]; then
        ssh_key_path="$(eval echo "${REPROLAB_RUNPOD_SSH_KEY_PATH}")"
        if [[ -f "${ssh_key_path}" ]] && command -v ssh-keygen >/dev/null 2>&1; then
            derived_pub="$(ssh-keygen -y -f "${ssh_key_path}" 2>/dev/null || true)"
            if [[ -n "${derived_pub}" ]]; then
                export REPROLAB_RUNPOD_SSH_PUBLIC_KEY="${derived_pub}"
                echo "[start.sh] Derived REPROLAB_RUNPOD_SSH_PUBLIC_KEY from ${ssh_key_path}."
            fi
        fi
    fi
fi

# 2. Runpod preflight (when relevant, and skippable).
runpod_preflight_needed=0
if [[ "${REPROLAB_DEFAULT_SANDBOX}" == "runpod" ]]; then
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
            if ! "${PREFLIGHT}" "${preflight_args[@]}"; then
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
    echo "[start.sh] Runpod preflight not required for sandbox=${REPROLAB_DEFAULT_SANDBOX}."
fi

# 3. Boot the API.
exec .venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000
