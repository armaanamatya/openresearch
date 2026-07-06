#!/usr/bin/env bash
# scripts/sdar_phase1_foundry.sh — deterministic SDAR Search-3B Phase-1 kickoff
# (execute mode, Anthropic-Foundry root/sub-roles, autostop ALWAYS ON).
#
# Run ON the VM (sdar-2model-a) from the repo root, fully detached so an
# operator's SSH session can drop without killing the run:
#
#   setsid nohup bash scripts/sdar_phase1_foundry.sh > runs/phase1_run.out 2>&1 < /dev/null &
#
# Requires: the staged /mnt/sdar-cache disk (SDAR repo pinned + conda envs +
# HF weights + the Search index) and a `.env` with AZURE_FOUNDRY_* (funds the
# opus-foundry root + sonnet-foundry executor/grader/verifier — fully
# OAuth-free). Do NOT execute this from the operator's laptop.
#
# Shape reconciled with the PROVEN scripts/sdar_gcp_run.sh pattern (self_stop:
# GCS-upload-BEFORE-shutdown + an EXIT trap so a crash/kill never strands a
# 4xA100 running) and the VM-resident runs/phase1_autonomous.sh precedent
# (reproduce -> upload runs/<pid>/ to GCS -> shutdown) — pinned here to the
# execute-mode Search-3B Phase-1 cell + the Anthropic-Foundry roles instead of
# grok/adapt-mode. Unlike sdar_gcp_run.sh, there is no NO_AUTOSTOP/
# FASTCRASH_STAY_UP debug escape hatch: Phase-1 is a bounded, pre-authorized
# ~$30 gate run and ALWAYS self-stops.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

GCS_BUCKET="gs://deepinvent-ext-ut-sdar-runs"
RUN_SPEC="configs/sdar_execute_run_spec.json"
CELLS_SEED="configs/sdar_execute_cells_phase1.json"

if [ ! -x .venv/bin/python ]; then
  echo "ERROR: .venv/bin/python missing — provision the venv first" >&2
  exit 1
fi
if [ ! -f "$RUN_SPEC" ]; then
  echo "ERROR: $RUN_SPEC missing" >&2
  exit 1
fi
if [ ! -f "$CELLS_SEED" ]; then
  echo "ERROR: $CELLS_SEED missing" >&2
  exit 1
fi

PID="sdar_phase1_foundry_$(date +%s)"
echo "$PID" > runs/.phase1_project_id

# --- Load the run-spec (foundry routing + execute + guards + LIFECYCLE_PRIMARY
# + CELL_RESUME_AUTO) into THIS shell's env too — belt-and-suspenders so this
# script's own echoes/gsutil calls see the same values the CLI resolves via
# --run-spec below. Every declared key (including a bare, non-OPENRESEARCH_-
# prefixed key like the spec's own "HF_HOME") is exported here; the Python-side
# loader (backend/cli.py::_load_run_spec) applies ONLY OPENRESEARCH_/REPROLAB_-
# prefixed keys plus two special-cased ones, which is why
# OPENRESEARCH_CELLS_SEED_PATH (phase-specific, deliberately NOT in the shared
# spec) is exported explicitly below regardless of what the spec contains.
mkdir -p runs/.cache
python3 - "$RUN_SPEC" > runs/.cache/phase1_env.sh <<'PY'
import json, shlex, sys
spec = json.load(open(sys.argv[1]))
for k, v in spec.items():
    print(f"export {k}={shlex.quote(str(v))}")
PY
set -a
# shellcheck disable=SC1091
source runs/.cache/phase1_env.sh
set +a

# Phase-1-specific overrides — explicit regardless of what the shared run-spec
# currently holds, so this wrapper stays correct even if the spec's defaults
# drift later.
export OPENRESEARCH_CELLS_SEED_PATH="$CELLS_SEED"
export OPENRESEARCH_REPRODUCTION_MODE="execute"
export HF_HOME="/mnt/sdar-cache/hf"

echo "[sdar_phase1_foundry] project_id=$PID run_spec=$RUN_SPEC cells_seed=$OPENRESEARCH_CELLS_SEED_PATH hf_home=$HF_HOME"

# self_stop <reason>: upload runs/<pid>/ (best-effort) to GCS BEFORE any
# shutdown, then halt the VM. Mirrors sdar_gcp_run.sh's self_stop; unlike it,
# there is no NO_AUTOSTOP/FASTCRASH_STAY_UP escape hatch — Phase-1 always stops.
_SELF_STOP_DONE=0
self_stop() {
  local reason="$1"
  _SELF_STOP_DONE=1
  echo "[sdar_phase1_foundry] self_stop triggered: $reason"
  echo "[sdar_phase1_foundry] uploading runs/$PID/ to $GCS_BUCKET/$PID/ ..."
  gsutil -m cp -r "runs/$PID" "$GCS_BUCKET/$PID/" 2>/dev/null || true
  gsutil -m cp "runs/phase1_run.out" "$GCS_BUCKET/$PID/phase1_run.out" 2>/dev/null || true
  echo "[sdar_phase1_foundry] halting GPU billing via shutdown"
  sync
  sudo shutdown -h now || sudo poweroff || true
}

# Exit trap: fires on ANY exit (success, error, signal) so an unexpected early
# abort still uploads + shuts down instead of silently leaving the VM billing.
_exit_trap() {
  local _ec=$?
  if [[ "$_SELF_STOP_DONE" == "0" ]]; then
    echo "[sdar_phase1_foundry] exit trap (rc=${_ec}): unexpected early exit — calling self_stop"
    self_stop "exit_trap rc=${_ec}"
  fi
}
trap _exit_trap EXIT

# Run the reproduction. set +e/-e brackets it so a non-zero rc does not abort
# the script before self_stop's upload+shutdown run.
set +e
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  .venv/bin/python -m backend.cli reproduce 2605.15155 \
    --project-id "$PID" --sandbox local --execution-mode max \
    --model opus-foundry \
    --scope-spec '{"models":["Qwen2.5-3B-Instruct"],"datasets":["Search-QA"]}' \
    --run-spec "$RUN_SPEC" \
    --paper-hint 2605.15155
rc=$?
set -e

if [ "$rc" -ne 0 ]; then
  self_stop "error rc=$rc"
else
  self_stop "success rc=0"
fi
exit "$rc"
