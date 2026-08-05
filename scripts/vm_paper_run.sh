#!/usr/bin/env bash
# vm_paper_run.sh — generic single-VM GCP paper launcher (any arXiv id or PDF).
#
# The generic counterpart of the SDAR-only scripted path (`scripts/sdar_gcp_run.sh`
# hardcodes arXiv 2605.15155; the 2026-07-22 runbook's "SDAR-specific" caveat and
# the "Reproduces SDAR instead of your paper" troubleshooting row both trace to
# that). This script implements the SAME validated flow, parameterized:
#
#   OPERATOR MODE (default — run from your laptop/workstation):
#     scripts/vm_paper_run.sh <paper> <project-id> [--run-spec <local.json>] [extra reproduce args...]
#     - refuses to run with harness preflights disabled (mirrors launch_sdar.sh:
#       the July 6 run burned an LLM budget on a dead credential exactly because
#       OPENRESEARCH_MODEL_PREFLIGHT=0 / OPENRESEARCH_SKIP_CRED_PREFLIGHT=1 hid it)
#     - FAILS LOUD if the four OPENRESEARCH_GCP_* identity vars are unset
#       (VmComputeProvider's built-in defaults point at a RETIRED VM:
#       sdar-a100-od / SSH user abheekp — never fall back to them)
#     - warns when OPENRESEARCH_GCP_GPU_MACHINE_TYPE is unset (the campaign VM
#       default is the expensive 4xA100-80 a2-highgpu-4g, ~$14/hr)
#     - verifies the VM is RUNNING with a prepared repo + venv (it does NOT
#       provision — provisioning stays with the campaign/VmComputeProvider path
#       or the direct recipe in docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md)
#     - ensures torch is importable in the VM venv, installing the ML training
#       deps if missing (the 2026-08-01 gotcha: `uv pip install -r
#       backend/requirements.txt` installs orchestrator deps ONLY, so every
#       training cell fails `ModuleNotFoundError: torch` -> null score)
#     - scp's the optional run-spec, then launches this same script on the VM
#       in --on-vm mode, fully detached (setsid+nohup, survives the SSH session)
#
#   VM MODE (--on-vm — what actually runs on the VM; usually launched by
#   operator mode, but directly invocable over SSH too):
#     scripts/vm_paper_run.sh --on-vm <paper> <project-id> [--run-spec <vm-path.json>] [extra reproduce args...]
#     - sources runs/.cache/vm_paper.env (or the legacy sdar_gcp.env) when present
#     - re-checks torch in the venv (installs if missing), then runs
#       `python -m backend.cli reproduce <paper> --sandbox local ...` under an
#       outer `timeout` backstop, and self-stops the VM afterwards (GCS report
#       upload first) so a finished/crashed run never idles GPU billing
#
# Money caps (env-overridable defaults; same meter mapping the campaign uses —
# child --max-usd carries the LLM share only, GPU $ via --max-run-gpu-usd, and
# gpu-hours enforced STRUCTURALLY by wall-clock co-tightening since there is no
# live hours knob):
#     OPENRESEARCH_VM_MAX_LLM_USD    (default 25)  -> reproduce --max-usd
#     OPENRESEARCH_VM_MAX_GPU_USD    (default 50)  -> reproduce --max-run-gpu-usd
#     OPENRESEARCH_VM_MAX_GPU_HOURS  (default 8)   -> --max-wall-clock = min(OPENRESEARCH_VM_MAX_WALL_S, hours*3600)
#
# Model tokens (env-overridable; API keys only — ⛔ NEVER OAuth, operator
# directive 2026-08-01; grok is NOT a validated executor — emits no
# commands.json manifest, verdict fails on evidence):
#     OPENRESEARCH_VM_ROOT_MODEL  (default opus-foundry)   -> reproduce --model
#     OPENRESEARCH_VM_MODELS      (default executor=sonnet-foundry,grader=sonnet-foundry,verifier=sonnet-foundry) -> reproduce --models
#
# Other knobs: OPENRESEARCH_VM_REPORT_GCS (gs:// prefix for the report upload),
# OPENRESEARCH_VM_NO_AUTOSTOP=1 (leave VM running), OPENRESEARCH_VM_OUTER_WALL_S
# (outer timeout backstop; default wall-clock + 3600s), OPENRESEARCH_VM_SKIP_TORCH_CHECK=1,
# OPENRESEARCH_VM_TORCH_INDEX_URL (default https://download.pytorch.org/whl/cu121),
# OPENRESEARCH_VM_FASTCRASH_STAY_UP=1 / OPENRESEARCH_VM_FASTCRASH_S (default 600).
#
# After every run: `gcloud compute instances list` — confirm no stray VM billing.
set -euo pipefail

# ---------------------------------------------------------------------------
# Shared helpers (both modes)
# ---------------------------------------------------------------------------

_lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

# Refuse to run with harness preflights disabled (mirrors launch_sdar.sh, the
# fail-loud reference). A disabled preflight is exactly how the July 6 run
# burned an LLM budget on a dead GCP credential — fail loud so the operator
# removes the stale export instead of re-learning why it exists.
refuse_disabled_preflights() {
  case "$(_lower "${OPENRESEARCH_MODEL_PREFLIGHT:-}")" in
    0|false|no|off)
      echo "ERROR: OPENRESEARCH_MODEL_PREFLIGHT=${OPENRESEARCH_MODEL_PREFLIGHT} disables the model preflight." >&2
      echo "       Preflights must stay ON (they catch dead credentials BEFORE code generation)." >&2
      echo "       Remove the export and re-run." >&2
      exit 1 ;;
  esac
  case "$(_lower "${OPENRESEARCH_SKIP_CRED_PREFLIGHT:-}")" in
    1|true|yes|on)
      echo "ERROR: OPENRESEARCH_SKIP_CRED_PREFLIGHT=${OPENRESEARCH_SKIP_CRED_PREFLIGHT} disables the credential preflight." >&2
      echo "       Preflights must stay ON (a 401 must surface before provisioning, not after)." >&2
      echo "       Remove the export and re-run." >&2
      exit 1 ;;
  esac
}

usage() {
  echo "usage: $0 [--on-vm] <paper: arXiv id | PDF path> <project-id> [--run-spec <path.json>] [extra 'reproduce' args...]" >&2
  echo "  env: OPENRESEARCH_GCP_{PROJECT,ZONE,INSTANCE,SSH_USER} (required, operator mode)," >&2
  echo "       OPENRESEARCH_REMOTE_DIR, OPENRESEARCH_GCP_GPU_MACHINE_TYPE," >&2
  echo "       OPENRESEARCH_VM_MAX_{LLM_USD,GPU_USD,GPU_HOURS}, OPENRESEARCH_VM_ROOT_MODEL, OPENRESEARCH_VM_MODELS" >&2
  exit 2
}

# ---------------------------------------------------------------------------
# Argument parsing (shared shape for both modes)
# ---------------------------------------------------------------------------

ON_VM=0
if [ "${1:-}" = "--on-vm" ]; then
  ON_VM=1
  shift
fi

# Positional paper + project-id (canonical), each falling back to an env var.
# A leading-dash first arg is treated as a flag, not a positional.
PAPER="${OPENRESEARCH_VM_PAPER:-}"
PROJECT_ID="${OPENRESEARCH_VM_PROJECT_ID:-}"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
  PAPER="$1"; shift
fi
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
  PROJECT_ID="$1"; shift
fi
if [ -z "$PAPER" ] || [ -z "$PROJECT_ID" ]; then
  echo "ERROR: paper and project-id are required (positional args 1+2, or OPENRESEARCH_VM_PAPER / OPENRESEARCH_VM_PROJECT_ID)." >&2
  usage
fi

RUN_SPEC=""
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --run-spec)
      [ $# -ge 2 ] || { echo "ERROR: --run-spec needs a path argument" >&2; usage; }
      RUN_SPEC="$2"; shift 2 ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# Model tokens + money caps (env-overridable defaults; shared by both modes so
# operator mode can forward exactly what VM mode will read).
VM_ROOT_MODEL="${OPENRESEARCH_VM_ROOT_MODEL:-opus-foundry}"
VM_MODELS="${OPENRESEARCH_VM_MODELS:-executor=sonnet-foundry,grader=sonnet-foundry,verifier=sonnet-foundry}"
VM_MAX_LLM_USD="${OPENRESEARCH_VM_MAX_LLM_USD:-25}"
VM_MAX_GPU_USD="${OPENRESEARCH_VM_MAX_GPU_USD:-50}"
VM_MAX_GPU_HOURS="${OPENRESEARCH_VM_MAX_GPU_HOURS:-8}"
VM_MAX_WALL_S="${OPENRESEARCH_VM_MAX_WALL_S:-86400}"
TORCH_INDEX_URL="${OPENRESEARCH_VM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

# Torch ensure command, run INSIDE the VM repo dir. The 2026-08-01 gotcha:
# backend/requirements.txt carries orchestrator deps only — torch + ML deps
# must be installed into the VM venv explicitly or every training cell fails
# `ModuleNotFoundError: torch` -> cell_execution_error -> null score. The
# import check makes this an idempotent no-op on a VM that already has torch.
# NOTE: read into its own var via `read -r -d ''` rather than $(cat <<EOF ...) —
# macOS bash 3.2 mis-parses a heredoc wrapped in $() (same workaround as
# sdar_gcp_run.sh's guidance heredoc). `|| true` absorbs read's expected
# non-zero exit at the delimiter-less EOF.
IFS='' read -r -d '' TORCH_ENSURE_CMD <<EOF || true
if .venv/bin/python -c 'import torch' 2>/dev/null; then
  echo '[vm_paper_run] torch OK in .venv'
else
  echo '[vm_paper_run] torch missing in .venv — installing ML training deps (2026-08-01 gotcha)'
  export PATH="\$HOME/.local/bin:\$PATH"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python torch torchvision numpy --index-url '$TORCH_INDEX_URL'
  else
    .venv/bin/python -m pip install torch torchvision numpy --index-url '$TORCH_INDEX_URL'
  fi
  .venv/bin/python -c 'import torch; print("[vm_paper_run] torch", torch.__version__, "cuda", torch.cuda.is_available())'
fi
EOF

# ===========================================================================
# OPERATOR MODE — validate identity, prepare the VM, launch detached
# ===========================================================================
if [ "$ON_VM" -eq 0 ]; then
  refuse_disabled_preflights

  # --- Required fresh-VM identity (VmComputeProvider defaults are RETIRED) ---
  _missing=0
  for var in OPENRESEARCH_GCP_PROJECT OPENRESEARCH_GCP_ZONE \
             OPENRESEARCH_GCP_INSTANCE OPENRESEARCH_GCP_SSH_USER; do
    if [ -z "$(eval "printf '%s' \"\${$var:-}\"")" ]; then
      echo "ERROR: ${var} is not set." >&2
      _missing=1
    fi
  done
  if [ "$_missing" -eq 1 ]; then
    cat >&2 <<'EOF'
The VmComputeProvider built-in defaults point at a RETIRED VM (sdar-a100-od /
SSH user abheekp). Set the fresh-VM identity explicitly, e.g.:

  export OPENRESEARCH_GCP_PROJECT=deepinvent-ext-ut
  export OPENRESEARCH_GCP_ZONE=us-central1-b
  export OPENRESEARCH_GCP_INSTANCE=<fresh-vm-name>
  export OPENRESEARCH_GCP_SSH_USER=<your-oslogin-user>   # gcloud compute os-login describe-profile --format='value(posixAccounts[0].username)'
  export OPENRESEARCH_REMOTE_DIR=/home/<your-oslogin-user>/openresearch   # recommended

See docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md.
EOF
    exit 1
  fi

  GCP_PROJECT="$OPENRESEARCH_GCP_PROJECT"
  GCP_ZONE="$OPENRESEARCH_GCP_ZONE"
  GCP_INSTANCE="$OPENRESEARCH_GCP_INSTANCE"
  GCP_SSH_USER="$OPENRESEARCH_GCP_SSH_USER"
  REMOTE_DIR="${OPENRESEARCH_REMOTE_DIR:-/home/${GCP_SSH_USER}/openresearch}"

  # Machine type: informational for an already-provisioned VM, but the
  # campaign/VmComputeProvider path reads it at `instances create` time — an
  # unset value there falls back to the EXPENSIVE 4xA100-80 default.
  if [ -z "${OPENRESEARCH_GCP_GPU_MACHINE_TYPE:-}" ]; then
    echo "WARNING: OPENRESEARCH_GCP_GPU_MACHINE_TYPE is unset — any (re)provisioning via the" >&2
    echo "         campaign/VmComputeProvider path defaults to a2-highgpu-4g (4xA100-80, ~\$14/hr)." >&2
    echo "         For CIFAR-scale papers use g2-standard-8 (1xL4, ~\$0.70/hr);" >&2
    echo "         for large-model fidelity runs use a2-ultragpu-1g (1xA100-80)." >&2
  fi

  # No GNU `timeout` wrapper around gcloud here — macOS has none (runbook gotcha);
  # gcloud's own connection timeout is the backstop.
  gssh() {
    gcloud compute ssh "${GCP_SSH_USER}@${GCP_INSTANCE}" \
      --zone "$GCP_ZONE" --project "$GCP_PROJECT" --quiet --command "$1"
  }

  echo "[vm_paper_run] paper=$PAPER project_id=$PROJECT_ID"
  echo "[vm_paper_run] VM identity: ${GCP_INSTANCE} (${GCP_PROJECT}/${GCP_ZONE}, ssh=${GCP_SSH_USER}, remote_dir=${REMOTE_DIR})"
  echo "[vm_paper_run] models: root=${VM_ROOT_MODEL} roles=${VM_MODELS}"
  echo "[vm_paper_run] caps: llm=\$${VM_MAX_LLM_USD} gpu=\$${VM_MAX_GPU_USD} gpu-hours=${VM_MAX_GPU_HOURS}"

  # --- VM must be RUNNING (this launcher does not provision) ----------------
  VM_STATUS="$(gcloud compute instances describe "$GCP_INSTANCE" \
    --zone "$GCP_ZONE" --project "$GCP_PROJECT" --format='value(status)' 2>/dev/null || true)"
  if [ "$VM_STATUS" != "RUNNING" ]; then
    echo "ERROR: VM ${GCP_INSTANCE} is not RUNNING (status: ${VM_STATUS:-not found})." >&2
    echo "       This launcher targets a prepared VM; it does not provision one." >&2
    echo "       Start it:      gcloud compute instances start ${GCP_INSTANCE} --zone ${GCP_ZONE} --project ${GCP_PROJECT}" >&2
    echo "       Or provision + prepare one via the direct recipe in" >&2
    echo "       docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md (repo staged at ${REMOTE_DIR}, .venv bootstrapped)." >&2
    exit 1
  fi

  # --- Prepared repo + venv on the VM (fail loud, name the fix) -------------
  if ! gssh "test -x ${REMOTE_DIR}/.venv/bin/python"; then
    echo "ERROR: ${REMOTE_DIR}/.venv/bin/python missing on ${GCP_INSTANCE}." >&2
    echo "       Stage the repo + bootstrap the venv first (runbook step 3-4):" >&2
    echo "         uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r backend/requirements.txt" >&2
    exit 1
  fi

  # --- Torch ensure (BEFORE launching — a missing torch = guaranteed null score)
  if [ "${OPENRESEARCH_VM_SKIP_TORCH_CHECK:-0}" != "1" ]; then
    gssh "cd ${REMOTE_DIR} && ${TORCH_ENSURE_CMD}"
  else
    echo "[vm_paper_run] torch check skipped (OPENRESEARCH_VM_SKIP_TORCH_CHECK=1)"
  fi

  # --- Optional run-spec staging --------------------------------------------
  REMOTE_SPEC_ARGS=()
  if [ -n "$RUN_SPEC" ]; then
    [ -f "$RUN_SPEC" ] || { echo "ERROR: run-spec not found: $RUN_SPEC" >&2; exit 1; }
    REMOTE_SPEC="runs/.cache/vm_paper_run_spec_${PROJECT_ID}.json"
    gssh "mkdir -p ${REMOTE_DIR}/runs/.cache"
    gcloud compute scp "$RUN_SPEC" \
      "${GCP_SSH_USER}@${GCP_INSTANCE}:${REMOTE_DIR}/${REMOTE_SPEC}" \
      --zone "$GCP_ZONE" --project "$GCP_PROJECT" --quiet
    REMOTE_SPEC_ARGS=(--run-spec "$REMOTE_SPEC")
    echo "[vm_paper_run] run-spec staged: ${RUN_SPEC} -> ${REMOTE_DIR}/${REMOTE_SPEC}"
  fi

  # --- Detached launch of VM mode -------------------------------------------
  # Forward the knobs VM mode reads (only the ones explicitly set/derived here),
  # %q-quoted so values survive the remote shell.
  FWD_ENV="export OPENRESEARCH_VM_ROOT_MODEL=$(printf '%q' "$VM_ROOT_MODEL")"
  FWD_ENV="$FWD_ENV OPENRESEARCH_VM_MODELS=$(printf '%q' "$VM_MODELS")"
  FWD_ENV="$FWD_ENV OPENRESEARCH_VM_MAX_LLM_USD=$(printf '%q' "$VM_MAX_LLM_USD")"
  FWD_ENV="$FWD_ENV OPENRESEARCH_VM_MAX_GPU_USD=$(printf '%q' "$VM_MAX_GPU_USD")"
  FWD_ENV="$FWD_ENV OPENRESEARCH_VM_MAX_GPU_HOURS=$(printf '%q' "$VM_MAX_GPU_HOURS")"
  FWD_ENV="$FWD_ENV OPENRESEARCH_VM_MAX_WALL_S=$(printf '%q' "$VM_MAX_WALL_S")"
  FWD_ENV="$FWD_ENV OPENRESEARCH_REMOTE_DIR=$(printf '%q' "$REMOTE_DIR")"
  for opt in OPENRESEARCH_VM_REPORT_GCS OPENRESEARCH_VM_NO_AUTOSTOP \
             OPENRESEARCH_VM_OUTER_WALL_S OPENRESEARCH_VM_SKIP_TORCH_CHECK \
             OPENRESEARCH_VM_FASTCRASH_STAY_UP OPENRESEARCH_VM_FASTCRASH_S \
             OPENRESEARCH_VM_TORCH_INDEX_URL OPENRESEARCH_BASELINE_EXTRA_GUIDANCE; do
    _val="$(eval "printf '%s' \"\${$opt:-}\"")"
    if [ -n "$_val" ]; then
      FWD_ENV="$FWD_ENV $opt=$(printf '%q' "$_val")"
    fi
  done

  REMOTE_ARGV="bash scripts/vm_paper_run.sh --on-vm $(printf '%q' "$PAPER") $(printf '%q' "$PROJECT_ID")"
  if [ ${#REMOTE_SPEC_ARGS[@]} -gt 0 ]; then
    REMOTE_ARGV="$REMOTE_ARGV --run-spec $(printf '%q' "${REMOTE_SPEC_ARGS[1]}")"
  fi
  if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    for a in "${EXTRA_ARGS[@]}"; do
      REMOTE_ARGV="$REMOTE_ARGV $(printf '%q' "$a")"
    done
  fi
  OUT_FILE="runs/vm_paper_run_${PROJECT_ID}.out"

  echo "[vm_paper_run] launching detached on ${GCP_INSTANCE}: ${REMOTE_ARGV}"
  gssh "cd ${REMOTE_DIR} && mkdir -p runs && ${FWD_ENV} && ( setsid nohup ${REMOTE_ARGV} > ${OUT_FILE} 2>&1 < /dev/null & ) && sleep 5 && echo '--- launch tail (${OUT_FILE}) ---' && tail -n 20 ${OUT_FILE}"

  echo "[vm_paper_run] launched. Monitor:"
  echo "  gcloud compute ssh ${GCP_SSH_USER}@${GCP_INSTANCE} --zone ${GCP_ZONE} --project ${GCP_PROJECT} --quiet --command 'tail -n 30 ${REMOTE_DIR}/${OUT_FILE}; ls ${REMOTE_DIR}/runs/${PROJECT_ID}/final_report.json 2>/dev/null'"
  echo "[vm_paper_run] after the run: gcloud compute instances list --project ${GCP_PROJECT}   # confirm no stray VM billing"
  exit 0
fi

# ===========================================================================
# VM MODE — the validated sdar_gcp_run.sh flow, minus everything SDAR-specific
# ===========================================================================

REPO_DIR="${OPENRESEARCH_REMOTE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

# Clear terminal sentinels from a previous run so the VM-side idle watchdog
# (which keys on .sdar_run_exited) does not apply its short dead-grace to a
# freshly-starting run.
rm -f "${REPO_DIR}/.vm_paper_run_exited" "${REPO_DIR}/.sdar_run_exited" 2>/dev/null || true

if [ ! -x .venv/bin/python ]; then
  echo "ERROR: .venv/bin/python missing — bootstrap the venv first:" >&2
  echo "  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r backend/requirements.txt" >&2
  exit 1
fi

# Source the prepared env file when present (generic name first, then the
# legacy SDAR one). Shell wins over .env in the harness, so these exports pin
# the run. Neither existing is fine — the CLI still reads .env via Settings.
for _env_file in runs/.cache/vm_paper.env runs/.cache/sdar_gcp.env; do
  if [ -f "$_env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$_env_file"
    set +a
    echo "[vm_paper_run] env sourced from $_env_file"
    break
  fi
done

# Preflights must stay ON no matter what the env file carried (fail loud —
# mirrors launch_sdar.sh; see refuse_disabled_preflights for the incident).
refuse_disabled_preflights

# The .env OpenAI key is dead (rotated in the 2026-07-21 leak audit) and a
# present key silently hijacks root-model selection (resolve_root_model
# prefers gpt-5 whenever OPENAI_API_KEY is set). The explicit --model below
# wins anyway, but unset it so nothing downstream trips on the dead key.
unset OPENAI_API_KEY || true

# NOTE (deliberate difference vs sdar_gcp_run.sh): NO CLAUDE_CODE_OAUTH_TOKEN
# lift from .env — OAuth is ⛔ FORBIDDEN (operator directive 2026-08-01).
# API keys only: AZURE_FOUNDRY_* for the *-foundry tokens, or a funded
# ANTHROPIC_API_KEY for --model claude.

# --- Torch ensure (right point: venv exists, before the reproduction) -------
if [ "${OPENRESEARCH_VM_SKIP_TORCH_CHECK:-0}" != "1" ]; then
  eval "$TORCH_ENSURE_CMD"
else
  echo "[vm_paper_run] torch check skipped (OPENRESEARCH_VM_SKIP_TORCH_CHECK=1)"
fi

# --- Money-cap -> reproduce-flag mapping ------------------------------------
# gpu-hours has no live knob on `reproduce`; enforce it structurally by
# co-tightening the wall-clock (same rule the campaign's enforcement plan
# applies): wall = min(OPENRESEARCH_VM_MAX_WALL_S, gpu_hours * 3600).
WALL_S="$(awk -v h="$VM_MAX_GPU_HOURS" -v w="$VM_MAX_WALL_S" \
  'BEGIN { s = h * 3600; if (s < w) printf "%d", s; else printf "%d", w }')"
# Outer wall-clock backstop: kills the reproduce process if the orchestrator
# itself wedges past its own --max-wall-clock.
OUTER_WALL_S="${OPENRESEARCH_VM_OUTER_WALL_S:-$((WALL_S + 3600))}"

echo "[vm_paper_run] paper=$PAPER project_id=$PROJECT_ID root=$VM_ROOT_MODEL models=$VM_MODELS"
echo "[vm_paper_run] caps: llm=\$${VM_MAX_LLM_USD} gpu=\$${VM_MAX_GPU_USD} gpu-hours=${VM_MAX_GPU_HOURS} wall=${WALL_S}s outer=${OUTER_WALL_S}s"

OUT_FILE="runs/vm_paper_run_${PROJECT_ID}.out"

# self_stop <reason>: best-effort GCS upload, then halt the VM to stop GPU
# billing. The boot disk (with all run artifacts) persists after shutdown.
# Set OPENRESEARCH_VM_NO_AUTOSTOP=1 to skip shutdown (leave VM running).
self_stop() {
  local reason="$1"
  _SELF_STOP_DONE=1   # prevent the EXIT trap from double-calling self_stop
  echo "[vm_paper_run] self_stop triggered: $reason"
  # Upload FIRST — before any early return — so the result is retrievable
  # laptop-independently even if the local watcher died.
  if [ -n "${OPENRESEARCH_VM_REPORT_GCS:-}" ]; then
    echo "[vm_paper_run] uploading report + diagnostics to ${OPENRESEARCH_VM_REPORT_GCS}/${PROJECT_ID}/ ..."
    gsutil -m cp \
      "runs/${PROJECT_ID}/final_report.json" \
      "runs/${PROJECT_ID}/final_report.md" \
      "runs/${PROJECT_ID}/demo_status.json" \
      "runs/${PROJECT_ID}/dashboard_events.jsonl" \
      "runs/${PROJECT_ID}/experiment_runs.jsonl" \
      "runs/${PROJECT_ID}/code/metrics.json" \
      "$OUT_FILE" \
      "${OPENRESEARCH_VM_REPORT_GCS}/${PROJECT_ID}/" 2>/dev/null || true
  fi
  if [ "${OPENRESEARCH_VM_NO_AUTOSTOP:-0}" = "1" ]; then
    echo "[vm_paper_run] autostop disabled (OPENRESEARCH_VM_NO_AUTOSTOP=1); leaving VM running"
    return 0
  fi
  # Fast-crash policy: by default even a fast error self-stops (never leave a
  # GPU VM burning after a crash). Opt-in OPENRESEARCH_VM_FASTCRASH_STAY_UP=1
  # holds the VM up on a <FASTCRASH_S crash to preserve scarce capacity
  # during an ATTENDED debug session.
  if [ "${OPENRESEARCH_VM_FASTCRASH_STAY_UP:-0}" = "1" ]; then
    case "$reason" in
      error*|exit_trap*)
        if [ "$SECONDS" -lt "${OPENRESEARCH_VM_FASTCRASH_S:-600}" ]; then
          echo "[vm_paper_run] FAST CRASH after ${SECONDS}s and FASTCRASH_STAY_UP=1 — leaving VM UP; stop it manually once inspected"
          return 0
        fi ;;
    esac
  fi
  echo "[vm_paper_run] halting GPU billing via shutdown; boot disk persists"
  sync
  sudo shutdown -h now || sudo poweroff || true
}

# Exit trap: fires on ANY exit (success, error, signal). Writes the terminal
# sentinels (both names — the deployed idle watchdog keys on .sdar_run_exited)
# and self-stops on an unexpected early exit.
_SELF_STOP_DONE=0
_exit_trap() {
  local _ec=$?
  touch "${REPO_DIR}/.vm_paper_run_exited" "${REPO_DIR}/.sdar_run_exited" 2>/dev/null || true
  echo "[vm_paper_run] exit (rc=${_ec}) — sentinel written"
  if [ "$_SELF_STOP_DONE" = "0" ]; then
    echo "[vm_paper_run] exit trap: unexpected early exit — calling self_stop"
    self_stop "exit_trap rc=${_ec}"
  fi
}
trap _exit_trap EXIT

# Run under timeout as the outer backstop. set +e so a non-zero rc (or rc=124
# on timeout) reaches self_stop instead of aborting the script.
RUN_SPEC_ARGS=()
if [ -n "$RUN_SPEC" ]; then
  [ -f "$RUN_SPEC" ] || { echo "ERROR: run-spec not found on VM: $RUN_SPEC" >&2; exit 1; }
  RUN_SPEC_ARGS=(--run-spec "$RUN_SPEC")
fi

set +e
timeout --signal=TERM --kill-after=180 "$OUTER_WALL_S" \
  .venv/bin/python -m backend.cli reproduce "$PAPER" \
    --mode rlm --sandbox local \
    --model "$VM_ROOT_MODEL" \
    --models "$VM_MODELS" \
    --max-usd "$VM_MAX_LLM_USD" \
    --max-run-gpu-usd "$VM_MAX_GPU_USD" \
    --max-wall-clock "$WALL_S" \
    --project-id "$PROJECT_ID" \
    ${RUN_SPEC_ARGS[@]+"${RUN_SPEC_ARGS[@]}"} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
rc=$?
set -e

if [ "$rc" -eq 124 ]; then
  self_stop "outer_wall_timeout rc=124"
elif [ "$rc" -ne 0 ]; then
  self_stop "error rc=$rc"
else
  self_stop "success rc=0"
fi
exit "$rc"
