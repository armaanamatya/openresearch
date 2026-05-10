#!/usr/bin/env bash
# Dashboard launcher with Runpod-as-default.
#
# Behavior:
#   1. Runs scripts/runpod_check.sh as a preflight. Aborts if the API key,
#      SSH key, or env vars are misconfigured — better to fail in 5 seconds
#      here than 15 minutes into a pipeline run.
#   2. Exports REPROLAB_DEFAULT_SANDBOX=runpod so the dashboard's upload
#      form defaults to --sandbox runpod (the CLI still defaults to auto).
#   3. Execs uvicorn for the FastAPI factory.
#
# Escape hatches:
#   START_SKIP_PREFLIGHT=1 ./start.sh   # skip runpod preflight (e.g. dev work
#                                       # without a Runpod account)
#   START_FULL_SMOKE=1 ./start.sh       # also boot a real pod, run nvidia-smi
#                                       # over SSH, destroy it. COSTS MONEY
#                                       # (cents-scale on RTX 4090). Use when
#                                       # you want end-to-end confidence
#                                       # before kicking off a long pipeline.
#   REPROLAB_DEFAULT_SANDBOX=docker ./start.sh
#                                       # opt back to docker even when starting
#                                       # via this script
set -euo pipefail
cd "$(dirname "$0")"

PREFLIGHT="${1:-${PREFLIGHT_SCRIPT:-scripts/runpod_check.sh}}"

# 1. Preflight (skippable).
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

# 2. Default sandbox for the dashboard. Honor an explicit override.
export REPROLAB_DEFAULT_SANDBOX="${REPROLAB_DEFAULT_SANDBOX:-runpod}"
echo "[start.sh] Dashboard default sandbox: ${REPROLAB_DEFAULT_SANDBOX}"

# 3. Boot the API.
exec .venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000
