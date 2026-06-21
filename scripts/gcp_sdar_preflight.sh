#!/usr/bin/env bash
# GCP SDAR VM preflight / preparation wrapper.
#
# Usage:
#   scripts/gcp_sdar_preflight.sh status
#   scripts/gcp_sdar_preflight.sh start
#   scripts/gcp_sdar_preflight.sh sync
#   scripts/gcp_sdar_preflight.sh check
#   scripts/gcp_sdar_preflight.sh prepare
#   scripts/gcp_sdar_preflight.sh launch
#   scripts/gcp_sdar_preflight.sh monitor
#   scripts/gcp_sdar_preflight.sh stop
#
# `prepare` runs on the cheap CPU machine type (no GPU billing): flips to
# CPU_MACHINE_TYPE, syncs the repo, installs SDAR deps, warms datasets/model caches,
# provisions ALFWorld/WebShop/Search-QA, then leaves the VM ready. `launch` flips
# to GPU_MACHINE_TYPE (attaches the A100s), verifies env/GPU readiness, then starts
# the reproduction detached (gate on a GREEN prepare first). Use
# `scripts/cancel_gcp_sdar_run.sh --stop-vm` or `stop` when done.
set -euo pipefail

ACTION="${1:-check}"
PROJECT="${OPENRESEARCH_GCP_PROJECT:-deepinvent-ext-ut}"
ZONE="${OPENRESEARCH_GCP_ZONE:-us-central1-c}"
INSTANCE="${OPENRESEARCH_GCP_INSTANCE:-sdar-a100-8g}"
REMOTE_DIR="${OPENRESEARCH_REMOTE_DIR:-/home/abheekp/openresearch}"
CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-/home/abheekp/.config/gcloud}"
REMOTE_USER="${OPENRESEARCH_GCP_SSH_USER:-abheekp}"
REQUIRE_SPOT="${OPENRESEARCH_REQUIRE_SPOT:-true}"
MIN_GPUS="${OPENRESEARCH_SDAR_MIN_GPUS:-8}"
CPU_MACHINE_TYPE="${OPENRESEARCH_GCP_CPU_MACHINE_TYPE:-e2-standard-16}"
GPU_MACHINE_TYPE="${OPENRESEARCH_GCP_GPU_MACHINE_TYPE:-a2-highgpu-8g}"

export CLOUDSDK_CONFIG

gcloud_base=(gcloud --project "$PROJECT")
ssh_base=(gcloud compute ssh "$REMOTE_USER@$INSTANCE" --zone "$ZONE" --project "$PROJECT" --quiet --command)

# Single EXIT-time temp-dir cleanup. One EXIT trap (not a per-function RETURN
# trap) avoids the bash pitfall where a RETURN trap set inside a function persists
# and re-fires on every LATER function return — dereferencing an out-of-scope
# `$stage` and aborting an already-SUCCESSFUL prepare under `set -u`. Null-safe via
# the `[@]:-` expansion so an empty array doesn't trip `set -u` either.
#
# `return 0` is REQUIRED: when _CLEANUP_DIRS is empty, `${arr[@]:-}` yields one
# empty string, so the loop's only iteration runs `[[ -n "" ]] && rm` — a false
# test (status 1) as the trap's last command. bash 5.2 lets a non-zero EXIT-trap
# last-command status override the script's 0 exit, so without this every action
# that stages no temp dir (status/start/stop/monitor/launch) spuriously exits 1 on
# success — which silently breaks `start && … prepare` chaining. A 0-status trap
# never masks a real failure: errexit's non-zero exit code is preserved.
_CLEANUP_DIRS=()
_cleanup() { local d; for d in "${_CLEANUP_DIRS[@]:-}"; do [[ -n "$d" ]] && rm -rf "$d"; done; return 0; }
trap _cleanup EXIT

status_only() {
  "${gcloud_base[@]}" compute instances describe "$INSTANCE" \
    --zone "$ZONE" --format='value(status)'
}

describe_instance() {
  "${gcloud_base[@]}" compute instances describe "$INSTANCE" \
    --zone "$ZONE" \
    --format='table(name,status,scheduling.provisioningModel,scheduling.preemptible,machineType.basename())'
}

ensure_spot() {
  if [[ "${REQUIRE_SPOT,,}" != "true" ]]; then
    return
  fi
  local model preemptible
  model="$("${gcloud_base[@]}" compute instances describe "$INSTANCE" --zone "$ZONE" --format='value(scheduling.provisioningModel)')"
  preemptible="$("${gcloud_base[@]}" compute instances describe "$INSTANCE" --zone "$ZONE" --format='value(scheduling.preemptible)')"
  if [[ "$model" != "SPOT" && "$preemptible" != "True" ]]; then
    echo "refusing to use non-spot GPU VM $INSTANCE (provisioningModel=$model preemptible=$preemptible)" >&2
    echo "set OPENRESEARCH_REQUIRE_SPOT=false only for an intentional on-demand run" >&2
    exit 1
  fi
}

current_machine_type() {
  "${gcloud_base[@]}" compute instances describe "$INSTANCE" \
    --zone "$ZONE" --format='value(machineType.basename())' 2>/dev/null || true
}

# ensure_machine_type <type>: if the VM is already on <type>, return immediately.
# Otherwise stop the VM (if RUNNING) and wait for TERMINATED, then set-machine-type.
# The machine type can only be changed while the VM is stopped.
ensure_machine_type() {
  local want="$1"
  local cur
  cur="$(current_machine_type)"
  if [[ "$cur" == "$want" ]]; then
    echo "[machine-type] already $want — no change"
    return 0
  fi
  echo "[machine-type] current=$cur want=$want"
  local s
  s="$(status_only 2>/dev/null || true)"
  if [[ "$s" == "RUNNING" ]]; then
    echo "[machine-type] stopping VM to flip machine type..."
    "${gcloud_base[@]}" compute instances stop "$INSTANCE" --zone "$ZONE"
    local deadline
    deadline=$((SECONDS + 300))
    while (( SECONDS < deadline )); do
      s="$(status_only 2>/dev/null || true)"
      if [[ "$s" == "TERMINATED" ]]; then break; fi
      sleep 10
    done
    if [[ "$s" != "TERMINATED" ]]; then
      echo "ERROR: VM did not reach TERMINATED within 300s; cannot flip machine type" >&2
      exit 1
    fi
  fi
  echo "[machine-type] setting machine type to $want..."
  "${gcloud_base[@]}" compute instances set-machine-type "$INSTANCE" \
    --zone "$ZONE" --machine-type "$want"
  echo "[machine-type] done — now $want"
}

start_vm() {
  local s
  ensure_spot
  s="$(status_only 2>/dev/null || true)"
  if [[ "$s" == "RUNNING" ]]; then
    echo "VM already RUNNING"
    wait_for_ssh
    return
  fi
  if [[ "$s" == "TERMINATED" ]]; then
    "${gcloud_base[@]}" compute instances start "$INSTANCE" --zone "$ZONE"
  elif [[ "$s" == "STAGING" || "$s" == "PROVISIONING" || "$s" == "STOPPING" || "$s" == "SUSPENDING" ]]; then
    echo "VM is $s; waiting for RUNNING"
  else
    echo "VM is in unexpected state '$s'; refusing to continue" >&2
    exit 1
  fi
  wait_for_running
  wait_for_ssh
}

wait_for_running() {
  local deadline s
  deadline=$((SECONDS + ${OPENRESEARCH_GCP_START_TIMEOUT_S:-900}))
  while (( SECONDS < deadline )); do
    s="$(status_only 2>/dev/null || true)"
    case "$s" in
      RUNNING)
        echo "VM is RUNNING"
        return
        ;;
      TERMINATED)
        echo "VM returned to TERMINATED during start; spot capacity may have been reclaimed" >&2
        exit 1
        ;;
      STAGING|PROVISIONING|STOPPING|SUSPENDING)
        sleep 10
        ;;
      *)
        echo "VM entered unexpected state '$s' while waiting for RUNNING" >&2
        exit 1
        ;;
    esac
  done
  echo "timed out waiting for $INSTANCE to become RUNNING" >&2
  exit 1
}

# wait_for_ssh: RUNNING != sshd-ready on a cold boot. The first SSH after a fresh
# start often fails "connect to host … port 22: Connection refused" (the
# 2026-06-19 prepare failure, and the launch gate's documented race). Poll a
# trivial SSH until it answers so the first REAL command (sync / GPU gate) does
# not abort spuriously — critical on `launch`, where the VM is already on the
# billed a2 GPU by the time the gate SSH runs. Bounded + fail-soft: on timeout it
# warns and proceeds, letting the real command surface any genuine outage.
wait_for_ssh() {
  local deadline
  deadline=$((SECONDS + ${OPENRESEARCH_GCP_SSH_TIMEOUT_S:-180}))
  while (( SECONDS < deadline )); do
    if "${ssh_base[@]}" "true" >/dev/null 2>&1; then
      echo "sshd is ready"
      return 0
    fi
    sleep 10
  done
  echo "WARNING: sshd not confirmed ready within ${OPENRESEARCH_GCP_SSH_TIMEOUT_S:-180}s; proceeding anyway" >&2
  return 0
}

stop_vm() {
  "${gcloud_base[@]}" compute instances stop "$INSTANCE" --zone "$ZONE"
}

sync_repo() {
  start_vm
  "${ssh_base[@]}" "mkdir -p $REMOTE_DIR"
  local stage
  stage="$(mktemp -d)"
  _CLEANUP_DIRS+=("$stage")
  local files=(
    requirements.txt pyproject.toml CHANGELOG.md CLAUDE.md gcp_info.md issues.md
  )
  [[ -f .env ]] && files+=(.env)
  rsync -a \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude 'backend/venv/' \
    --exclude 'runs/' \
    --exclude 'frontend/node_modules/' \
    --exclude 'frontend/.next/' \
    backend scripts docs docker infra \
    "${files[@]}" \
    "$stage/"
  # dotglob so "$stage"/* also matches dotfiles like .env. Without it the default
  # glob skips dotfiles, which silently dropped .env from every sync (the
  # 2026-06-19 token miss: creds renamed locally never reached the VM).
  ( shopt -s dotglob
    gcloud compute scp --recurse \
      --zone "$ZONE" --project "$PROJECT" --quiet --compress \
      "$stage"/* \
      "$REMOTE_USER@$INSTANCE:$REMOTE_DIR/" )
  "${ssh_base[@]}" "chmod +x $REMOTE_DIR/scripts/gcp_sdar_preflight.sh $REMOTE_DIR/scripts/sdar_gcp_assets.py $REMOTE_DIR/scripts/cancel_gcp_sdar_run.sh"
}

remote_check() {
  start_vm
  # Source the env file written by `prepare` so the standalone check sees the
  # dedicated WebShop interpreter (OPENRESEARCH_WEBSHOP_PYTHON) and cache paths.
  "${ssh_base[@]}" "cd $REMOTE_DIR && { [ -f runs/.cache/sdar_gcp.env ] && . runs/.cache/sdar_gcp.env || true; } && .venv/bin/python scripts/sdar_gcp_assets.py --check --require-gpu --min-gpus $MIN_GPUS"
}

remote_prepare() {
  ensure_machine_type "$CPU_MACHINE_TYPE"
  start_vm
  sync_repo
  # System build prerequisites for source-built Python deps. Ubuntu 24.04 ships
  # only `python3`, but some sdists shell out to bare `python` and build native
  # code (e.g. fast-downward-textworld -> cmake; alfworld/textworld -> C exts).
  # openjdk-17-jdk-headless (JDK, not just JRE) is required by pyserini's jnius,
  # which locates JAVA_HOME via `javac` — the JRE alone fails with "Unable to find
  # javac". WebShop is best-effort, so this only improves its odds of coming up.
  # Idempotent; needs passwordless sudo (GCP default for the creating user).
  "${ssh_base[@]}" "sudo bash -c 'export DEBIAN_FRONTEND=noninteractive && apt-get update -qq && apt-get install -y python-is-python3 cmake ninja-build build-essential libffi-dev openjdk-17-jdk-headless'"
  # Ensure uv is available for creating Python-version-pinned venvs (the run venv
  # at 3.12, WebShop's dedicated venv at 3.10); uv resolves the exact minor-version
  # interpreter automatically.
  "${ssh_base[@]}" "command -v uv || (curl -LsSf https://astral.sh/uv/install.sh | sh) && export PATH=\"\$HOME/.local/bin:\$PATH\""
  # Ensure the run venv is Python 3.12 — recreate it when MISSING or below the
  # harness floor. The harness requires >=3.11 (it does `from typing import Self`,
  # a 3.11+ name; pyproject requires-python = ">=3.11"), so the run venv must NOT
  # be 3.10 — only WebShop's frozen stack needs 3.10, and that lives in its own
  # dedicated venv. `--seed` gives the uv venv pip/setuptools (uv omits them by
  # default). `sudo rm` clears any root-owned files a prior `sudo pip` left behind.
  # Idempotent: an existing 3.12 venv is reused as-is.
  "${ssh_base[@]}" "export PATH=\"\$HOME/.local/bin:\$PATH\" && cd $REMOTE_DIR && if [ ! -x .venv/bin/python ] || ! .venv/bin/python --version 2>&1 | grep -q '3\\.12'; then sudo rm -rf .venv && uv venv --python 3.12 --seed .venv; fi && .venv/bin/python -m pip install -r backend/requirements.txt && .venv/bin/python scripts/sdar_gcp_assets.py --prepare --check"
}

assert_running() {
  local s
  s="$(status_only 2>/dev/null || true)"
  if [[ "$s" != "RUNNING" ]]; then
    echo "ERROR: VM $INSTANCE is '${s:-unknown}'; run 'start' or 'prepare' first" >&2
    exit 1
  fi
}

monitor_run() {
  assert_running
  local pid_proj="${OPENRESEARCH_SDAR_PROJECT_ID:-sdar_gcp_20260618}"
  "${ssh_base[@]}" "cd $REMOTE_DIR && echo '--- run.out ---' && tail -n 40 runs/sdar_gcp_run.out 2>/dev/null; echo '--- dashboard_events ---' && tail -n 20 runs/$pid_proj/dashboard_events.jsonl 2>/dev/null; echo '--- gpu ---' && nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null"
}

launch_run() {
  ensure_machine_type "$GPU_MACHINE_TYPE"
  start_vm
  # GPU-COST GATE: never start a GPU run until env + data + GPUs are verified
  # installed. Re-runs the asset readiness check (datasets, model snapshots,
  # ALFWorld/Search-QA, visible GPUs); a [RED] result aborts BEFORE any GPU work,
  # so a half-provisioned env can never burn GPU hours. Bypass for a deliberate
  # dry run via OPENRESEARCH_SDAR_SKIP_LAUNCH_CHECK=1.
  if [[ "${OPENRESEARCH_SDAR_SKIP_LAUNCH_CHECK:-0}" != "1" ]]; then
    echo "[gate] verifying env/data/GPU readiness before any GPU work..."
    if ! "${ssh_base[@]}" "cd $REMOTE_DIR && { [ -f runs/.cache/sdar_gcp.env ] && . runs/.cache/sdar_gcp.env || true; } && .venv/bin/python scripts/sdar_gcp_assets.py --check --require-gpu --min-gpus $MIN_GPUS"; then
      echo "ERROR: env/data not ready ([RED]) — run 'prepare' to [GREEN] first; refusing to start the GPU run" >&2
      exit 1
    fi
  fi
  # Push the latest run script (it may post-date the last prepare sync).
  gcloud compute scp --zone "$ZONE" --project "$PROJECT" --quiet \
    scripts/sdar_gcp_run.sh "$REMOTE_USER@$INSTANCE:$REMOTE_DIR/scripts/sdar_gcp_run.sh"

  # --- run-spec (P0.3): ship one JSON config instead of a 12-var env whitelist ---
  # Builds runs/.cache/run_spec.json from the local env and SCPs it to the VM.
  # The remote sdar_gcp_run.sh passes it via --run-spec so multi-line values
  # (e.g. OPENRESEARCH_BASELINE_EXTRA_GUIDANCE) survive intact — no shell
  # env-word-split footgun.  Optional scope-guidance text is folded into
  # baseline_extra_guidance inside the JSON; the old sdar_scope_guidance.txt
  # staging is kept as a fallback for scripts that have not yet adopted --run-spec.
  #
  # JSON spec format (matches _load_run_spec in backend/cli.py):
  #   { "OPENRESEARCH_<KEY>": "<value>", ..., "baseline_extra_guidance": "<text>" }
  # Keys that are empty/unset locally are omitted (never forwarded as empty strings).
  mkdir -p runs/.cache

  # Start with a minimal valid JSON object.
  _spec_file="runs/.cache/run_spec.json"
  printf '{' > "$_spec_file"
  _first=1

  # Helper: append a key→value pair to the JSON object (only when value is non-empty).
  _spec_add() {
    local _k="$1" _v="$2"
    [ -n "$_v" ] || return 0
    # Escape backslash and double-quote for JSON; newlines become \n.
    local _vj
    _vj="$(printf '%s' "$_v" | sed 's/\\/\\\\/g; s/"/\\"/g' | awk '{printf "%s\\n", $0}' | sed 's/\\n$//')"
    if [ "$_first" = "1" ]; then
      printf '"%s":"%s"' "$_k" "$_vj" >> "$_spec_file"
      _first=0
    else
      printf ',"%s":"%s"' "$_k" "$_vj" >> "$_spec_file"
    fi
  }

  # Forwarded OPENRESEARCH_* vars (the former whitelist, now JSON-encoded).
  _spec_add AZURE_FOUNDRY_DEPLOYMENT          "${AZURE_FOUNDRY_DEPLOYMENT:-}"
  _spec_add OPENRESEARCH_SDAR_PROJECT_ID      "${OPENRESEARCH_SDAR_PROJECT_ID:-}"
  _spec_add OPENRESEARCH_GRADER_SAMPLES       "${OPENRESEARCH_GRADER_SAMPLES:-}"
  _spec_add OPENRESEARCH_SDAR_MODELS          "${OPENRESEARCH_SDAR_MODELS:-}"
  _spec_add OPENRESEARCH_SDAR_ROOT            "${OPENRESEARCH_SDAR_ROOT:-}"
  _spec_add OPENRESEARCH_SDAR_NO_AUTOSTOP     "${OPENRESEARCH_SDAR_NO_AUTOSTOP:-}"
  _spec_add OPENRESEARCH_SDAR_OUTER_WALL_S    "${OPENRESEARCH_SDAR_OUTER_WALL_S:-}"
  _spec_add OPENRESEARCH_SDAR_REPORT_GCS      "${OPENRESEARCH_SDAR_REPORT_GCS:-}"
  _spec_add OPENRESEARCH_EVIDENCE_GATE        "${OPENRESEARCH_EVIDENCE_GATE:-}"
  _spec_add OPENRESEARCH_ARG_CONTRACTS        "${OPENRESEARCH_ARG_CONTRACTS:-}"
  _spec_add OPENRESEARCH_STUB_METRICS_GUARD   "${OPENRESEARCH_STUB_METRICS_GUARD:-}"
  _spec_add OPENRESEARCH_LLM_AUTH_STRATEGY    "${OPENRESEARCH_LLM_AUTH_STRATEGY:-}"

  # Multi-line scope guidance: fold from the staged text file (backward-compat)
  # or from OPENRESEARCH_BASELINE_EXTRA_GUIDANCE if set directly.
  _guidance=""
  if [ -f runs/.cache/sdar_scope_guidance.txt ]; then
    _guidance="$(cat runs/.cache/sdar_scope_guidance.txt)"
  fi
  if [ -n "${OPENRESEARCH_BASELINE_EXTRA_GUIDANCE:-}" ]; then
    if [ -n "$_guidance" ]; then
      _guidance="${OPENRESEARCH_BASELINE_EXTRA_GUIDANCE}"$'\n\n'"${_guidance}"
    else
      _guidance="${OPENRESEARCH_BASELINE_EXTRA_GUIDANCE}"
    fi
  fi
  _spec_add baseline_extra_guidance "$_guidance"

  printf '}' >> "$_spec_file"

  # SCP the spec to the VM (always, even if minimal — the remote side uses --run-spec).
  "${ssh_base[@]}" "mkdir -p $REMOTE_DIR/runs/.cache"
  gcloud compute scp --zone "$ZONE" --project "$PROJECT" --quiet \
    "$_spec_file" "$REMOTE_USER@$INSTANCE:$REMOTE_DIR/runs/.cache/run_spec.json"

  # Backward-compat: also SCP the raw guidance file when present, so older
  # sdar_gcp_run.sh invocations that have not yet adopted --run-spec can still read it.
  if [ -f runs/.cache/sdar_scope_guidance.txt ]; then
    gcloud compute scp --zone "$ZONE" --project "$PROJECT" --quiet \
      runs/.cache/sdar_scope_guidance.txt "$REMOTE_USER@$INSTANCE:$REMOTE_DIR/runs/.cache/sdar_scope_guidance.txt"
  fi

  # Refuse to double-launch the same paper; require a GREEN-prepared env file;
  # then start the reproduction fully detached (setsid+nohup, stdin from
  # /dev/null) so it survives this SSH session closing. The run logs to
  # runs/sdar_gcp_run.out and streams to runs/<project_id>/dashboard_events.jsonl.
  # --run-spec loads the JSON spec built above; explicit flags in sdar_gcp_run.sh
  # (--sandbox, --model, etc.) still override spec values per _load_run_spec contract.
  "${ssh_base[@]}" "cd $REMOTE_DIR \
    && chmod +x scripts/sdar_gcp_run.sh \
    && { [ -f runs/.cache/sdar_gcp.env ] || { echo 'ERROR: runs/.cache/sdar_gcp.env missing — run prepare to [GREEN] first' >&2; exit 1; }; } \
    && { ! pgrep -f '[b]ackend.cli reproduce 2605.15155' >/dev/null || { echo 'ERROR: a reproduce process for 2605.15155 is already running' >&2; exit 1; }; } \
    && ( setsid nohup bash scripts/sdar_gcp_run.sh --run-spec runs/.cache/run_spec.json > runs/sdar_gcp_run.out 2>&1 < /dev/null & ) \
    && sleep 4 \
    && echo '--- launch tail (runs/sdar_gcp_run.out) ---' \
    && tail -n 20 runs/sdar_gcp_run.out"
}

case "$ACTION" in
  status) describe_instance ;;
  start) start_vm ;;
  stop) stop_vm ;;
  sync) sync_repo ;;
  check) remote_check ;;
  prepare) remote_prepare ;;
  launch) launch_run ;;
  monitor) monitor_run ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 2
    ;;
esac
