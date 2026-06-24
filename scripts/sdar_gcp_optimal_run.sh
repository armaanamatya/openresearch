#!/usr/bin/env bash
# Capacity-aware OPTIMAL SDAR-on-GCP runner (poll on-demand -> run -> watch).
#
# One self-contained, session-survivable command that:
#   1. Pins the VM to on-demand a2-highgpu-4g and POLLS for capacity (free while the VM
#      stays TERMINATED; on-demand never preempts once it starts, unlike spot which is
#      currently being reclaimed within minutes).
#   2. On capacity: syncs the latest code, writes the OPTIMAL run env (PRIMARY lifecycle
#      mode + claude-oauth root + Sonnet sub-agents + a generous per-cell timeout), and
#      ships the capacity-bounded guidance (batched rollouts / ~20 steps / completion-
#      priority) that fixes the prior run's 11/12 cell TIMEOUTS.
#   3. Launches GREEN-gated (prepare-on-RED fallback) fully detached on the VM.
#   4. Runs the verbose lifecycle-aware watcher, which auto-stops the VM on a terminal
#      state (billing halts; the report persists on the boot disk -> `inspect` to pull).
#
# Designed to be launched detached so it outlives the controlling shell:
#   setsid nohup bash scripts/sdar_gcp_optimal_run.sh > runs/_optimal_od.log 2>&1 < /dev/null &
#
# Knobs (env): PROJECT_ID, POLL_INTERVAL (s), MAX_POLLS, ROOT, RUN_EXPERIMENT_TIMEOUT_S.
set -uo pipefail

export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-/home/abheekp/.config/gcloud}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZONE="${OPENRESEARCH_GCP_ZONE:-us-central1-b}"
PROJECT="${OPENRESEARCH_GCP_PROJECT:-deepinvent-ext-ut}"
INSTANCE="${OPENRESEARCH_GCP_INSTANCE:-sdar-a100-od}"
SSH_USER="${OPENRESEARCH_GCP_SSH_USER:-abheekp}"
REMOTE_DIR="${OPENRESEARCH_REMOTE_DIR:-/home/abheekp/openresearch}"
GPU_MT="${OPENRESEARCH_GCP_GPU_MACHINE_TYPE:-a2-highgpu-4g}"
MIN_GPUS="${OPENRESEARCH_SDAR_MIN_GPUS:-4}"

PROJECT_ID="${PROJECT_ID:-sdar_optimal_od}"
POLL_INTERVAL="${POLL_INTERVAL:-600}"
MAX_POLLS="${MAX_POLLS:-288}"            # 288 * 600s = 48h ceiling
ROOT="${ROOT:-claude-oauth}"
RUN_EXPERIMENT_TIMEOUT_S="${RUN_EXPERIMENT_TIMEOUT_S:-2400}"
SDAR_VRAM_GB="${SDAR_VRAM_GB:-40}"     # per-GPU VRAM the run reports to the capacity gate (80 for a2-ultragpu)
LIFECYCLE_MAX_IMPROVE="${LIFECYCLE_MAX_IMPROVE:-}"   # OPENRESEARCH_LIFECYCLE_MAX_IMPROVE override; empty = harness default (set -u safe)

G=(gcloud --project "$PROJECT")
ZP=(--zone "$ZONE" --project "$PROJECT")
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
ssh_() { timeout "${2:-90}" gcloud compute ssh "$SSH_USER@$INSTANCE" "${ZP[@]}" --quiet --command "$1"; }
status_() { "${G[@]}" compute instances describe "$INSTANCE" --zone "$ZONE" --format='value(status)' 2>/dev/null || true; }
wait_ssh() { local i; for i in $(seq 1 30); do ssh_ 'echo ok' 25 2>/dev/null | grep -q ok && { log "  sshd ready"; return 0; }; sleep 10; done; log "  WARN: sshd not confirmed"; }

log "=== SDAR optimal on-demand runner: project_id=$PROJECT_ID root=$ROOT zone=$ZONE ==="

# --- 1. secure an on-demand GPU VM (no preempt). NEVER stop a RUNNING VM: it may hold a
#        live run, and stopping one is exactly what cascaded a lost run + lost capacity. --
got=0
read -r _st _mt _pm < <("${G[@]}" compute instances describe "$INSTANCE" --zone "$ZONE" \
  --format='value(status,machineType.basename(),scheduling.provisioningModel)' 2>/dev/null || echo "MISSING x x")
if [[ "$_st" == "RUNNING" ]]; then
  if [[ "$_mt" == "$GPU_MT" && "$_pm" == "STANDARD" ]]; then
    log "VM already RUNNING on $GPU_MT/STANDARD — using it as-is (NOT stopping; it may hold a live run)"; got=1
  else
    log "FATAL: VM RUNNING with unexpected config ($_mt/$_pm), not $GPU_MT/STANDARD — refusing to stop a possibly-live VM; resolve manually"; exit 5
  fi
elif [[ "$_st" == "MISSING" && -n "${CREATE_IMAGE:-}" ]]; then
  # CREATE-poll: the instance does not exist (e.g. an a2-ultragpu that cannot be pre-created
  # while stocked out). Repeatedly try to CREATE it on-demand until capacity returns. On-demand
  # by default (no --preemptible); GPU types require --maintenance-policy=TERMINATE.
  log "instance $INSTANCE absent; CREATE-polling $GPU_MT in $ZONE (image $CREATE_IMAGE) every ${POLL_INTERVAL}s"
  for i in $(seq 1 "$MAX_POLLS"); do
    out="$("${G[@]}" compute instances create "$INSTANCE" --zone "$ZONE" --machine-type "$GPU_MT" \
      --image-family "$CREATE_IMAGE" --image-project "${CREATE_IMAGE_PROJECT:-deeplearning-platform-release}" \
      --maintenance-policy TERMINATE --boot-disk-size "${CREATE_DISK_GB:-1000}" --boot-disk-type pd-ssd \
      --metadata install-nvidia-driver=True --no-restart-on-failure 2>&1)"
    if [[ "$(status_)" == "RUNNING" ]]; then log "[$i] CAPACITY — created $INSTANCE ($GPU_MT in $ZONE)"; got=1; break; fi
    if echo "$out" | grep -qiE 'STOCKOUT|enough resources|EXHAUSTED|currently unavailable'; then
      log "[$i] stocked out; sleeping ${POLL_INTERVAL}s"; sleep "$POLL_INTERVAL"
    else
      log "[$i] non-stockout create error: $(echo "$out" | tail -2)"; sleep "$POLL_INTERVAL"
    fi
  done
  [[ "$got" == 1 ]] || { log "FATAL: no $GPU_MT capacity to create in $MAX_POLLS polls; giving up"; exit 2; }
  wait_ssh
fi
if [[ "$got" != 1 && "$_st" != "MISSING" ]]; then
  # Not RUNNING → wait for a stable TERMINATED, then flip machine-type FIRST (a CPU/e2
  # STANDARD type can't use --maintenance-policy=TERMINATE; only GPU types can), then flip
  # to on-demand STANDARD, then ASSERT it took (a silent flip failure leaves it SPOT and it
  # would preempt in minutes).
  for _ in $(seq 1 30); do [[ "$(status_)" == "TERMINATED" ]] && break; sleep 8; done
  log "flipping machine-type=$GPU_MT, then scheduling=on-demand (STANDARD)"
  "${G[@]}" compute instances set-machine-type "$INSTANCE" --zone "$ZONE" --machine-type="$GPU_MT" >/dev/null 2>&1 || true
  "${G[@]}" compute instances set-scheduling "$INSTANCE" --zone "$ZONE" \
    --no-preemptible --provisioning-model=STANDARD --maintenance-policy=TERMINATE \
    --clear-instance-termination-action >/dev/null 2>&1 || true
  _pm2="$("${G[@]}" compute instances describe "$INSTANCE" --zone "$ZONE" --format='value(scheduling.provisioningModel)' 2>/dev/null || true)"
  if [[ "$_pm2" != "STANDARD" ]]; then
    log "FATAL: scheduling is '$_pm2', not STANDARD — refusing to poll (would start as spot and preempt)"; exit 4
  fi
  log "confirmed scheduling=STANDARD (on-demand, no preempt)"
  for i in $(seq 1 "$MAX_POLLS"); do
    out="$("${G[@]}" compute instances start "$INSTANCE" --zone "$ZONE" 2>&1)"
    if [[ "$(status_)" == "RUNNING" ]]; then log "[$i] CAPACITY RETURNED — VM RUNNING"; got=1; break; fi
    if echo "$out" | grep -qiE 'STOCKOUT|enough resources|EXHAUSTED|currently unavailable'; then
      log "[$i] stocked out; sleeping ${POLL_INTERVAL}s"; sleep "$POLL_INTERVAL"
    else
      log "[$i] NON-STOCKOUT error: $(echo "$out" | tail -3)"; sleep "$POLL_INTERVAL"
    fi
  done
fi
[[ "$got" == 1 ]] || { log "FATAL: no on-demand capacity in $MAX_POLLS polls; giving up"; exit 2; }

# --- 1b. cross-poller launch lock (option B: a sibling poller races a different zone) ---
# Several pollers (e.g. one per zone) can run concurrently to double the catch rate, but
# only ONE may launch a run. `mkdir` is atomic: the first poller to catch capacity creates
# the lock and proceeds; any other poller that catches capacity finds the lock held, stops
# its just-started VM (releasing the scarce slot) and exits. Clean the lock between runs:
#   rmdir runs/.cache/sdar_launch.lock
LOCKDIR="${SDAR_LAUNCH_LOCK:-$HERE/runs/.cache/sdar_launch.lock}"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  log "another poller already won the launch (lock $LOCKDIR held) — stopping my VM ($ZONE/$INSTANCE) and exiting"
  "${G[@]}" compute instances stop "$INSTANCE" --zone "$ZONE" >/dev/null 2>&1 || true
  exit 0
fi
log "acquired launch lock — proceeding as the run target ($ZONE/$INSTANCE/$GPU_MT)"
wait_ssh

# --- 2. sync latest code ---------------------------------------------------------------
log "syncing latest code to the VM ..."
OPENRESEARCH_GCP_ZONE="$ZONE" OPENRESEARCH_GCP_INSTANCE="$INSTANCE" \
  OPENRESEARCH_GCP_GPU_MACHINE_TYPE="$GPU_MT" OPENRESEARCH_SDAR_MIN_GPUS="$MIN_GPUS" \
  OPENRESEARCH_GCP_PROVISIONING_MODEL=STANDARD \
  bash "$HERE/scripts/gcp_sdar_preflight.sh" sync 2>&1 | tail -4

# ensure the env file exists (prepare creates it + warms caches on a fresh disk)
if ! ssh_ "test -f $REMOTE_DIR/runs/.cache/sdar_gcp.env" 30; then
  log "sdar_gcp.env missing — running prepare to warm env/caches (first-boot path)"
  OPENRESEARCH_GCP_ZONE="$ZONE" OPENRESEARCH_GCP_INSTANCE="$INSTANCE" \
    OPENRESEARCH_GCP_GPU_MACHINE_TYPE="$GPU_MT" OPENRESEARCH_SDAR_MIN_GPUS="$MIN_GPUS" \
    OPENRESEARCH_GCP_PROVISIONING_MODEL=STANDARD \
    bash "$HERE/scripts/gcp_sdar_preflight.sh" prepare 2>&1 | tail -8
fi

# --- 3. write the OPTIMAL run env on the VM (LAST assignment wins on sourcing) ---------
log "writing optimal run env overrides ..."
ssh_ "cd $REMOTE_DIR && {
  printf '\n# --- sdar_gcp_optimal_run overrides %s ---\n' '$PROJECT_ID' >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_SDAR_PROJECT_ID=$PROJECT_ID'                       >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_SDAR_ROOT=$ROOT'                                    >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_SDAR_MODELS=executor=sonnet,grader=sonnet,verifier=sonnet' >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_LLM_AUTH_STRATEGY=oauth_only'                       >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_LIFECYCLE_PRIMARY=1'                                >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_LIFECYCLE_DRIVE=1'                                  >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_USE_AUTHOR_REPO=1'                                  >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_REPRODUCTION_MODE=adapt'                            >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_ROOT_EFFORT=high'                                   >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_PREFLIGHT_SMOKE=0'                                  >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S=$RUN_EXPERIMENT_TIMEOUT_S' >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_SDAR_VRAM_GB=$SDAR_VRAM_GB'                          >> runs/.cache/sdar_gcp.env
  [ -n '$LIFECYCLE_MAX_IMPROVE' ] && echo 'export OPENRESEARCH_LIFECYCLE_MAX_IMPROVE=$LIFECYCLE_MAX_IMPROVE' >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_GRADER_SAMPLES=3'                                   >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_SDAR_NO_AUTOSTOP=0'                                 >> runs/.cache/sdar_gcp.env
  echo 'unset OPENRESEARCH_BASELINE_EXTRA_GUIDANCE'                             >> runs/.cache/sdar_gcp.env
  set -a; . runs/.cache/sdar_gcp.env >/dev/null 2>&1; set +a
  echo \"EFFECTIVE: ROOT=\$OPENRESEARCH_SDAR_ROOT MODELS=\$OPENRESEARCH_SDAR_MODELS PRIMARY=\$OPENRESEARCH_LIFECYCLE_PRIMARY TIMEOUT=\$OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S USE_REPO=\$OPENRESEARCH_USE_AUTHOR_REPO\"
  grep -q '^CLAUDE_CODE_OAUTH_TOKEN=.' .env && echo 'oauth token: present' || echo 'oauth token: MISSING'
}" 60

# Stage the run's implementer guidance into the path the preflight launch SCPs. GUIDANCE_FILE
# (relative to the repo root) lets one runner serve both the smallest-two and full-scope runs.
if [[ -n "${GUIDANCE_FILE:-}" && -f "$HERE/$GUIDANCE_FILE" ]]; then
  cp "$HERE/$GUIDANCE_FILE" "$HERE/runs/.cache/sdar_scope_guidance.txt"
  log "staged implementer guidance from $GUIDANCE_FILE"
fi

# --- 4. launch (GREEN-gated; the guidance ships via the local --------------------------
#        runs/.cache/sdar_scope_guidance.txt that gcp_sdar_preflight.sh launch SCPs). -----
log "launching SDAR reproduction (GREEN-gated) ..."
launch_once() {
  OPENRESEARCH_GCP_ZONE="$ZONE" OPENRESEARCH_GCP_INSTANCE="$INSTANCE" \
    OPENRESEARCH_GCP_GPU_MACHINE_TYPE="$GPU_MT" OPENRESEARCH_SDAR_MIN_GPUS="$MIN_GPUS" \
    OPENRESEARCH_GCP_PROVISIONING_MODEL=STANDARD \
    OPENRESEARCH_SDAR_PROJECT_ID="$PROJECT_ID" OPENRESEARCH_SDAR_ROOT="$ROOT" \
    bash "$HERE/scripts/gcp_sdar_preflight.sh" launch 2>&1 | tail -24
}
if ! launch_once; then
  log "launch gate RED — running prepare then relaunching"
  OPENRESEARCH_GCP_ZONE="$ZONE" OPENRESEARCH_GCP_INSTANCE="$INSTANCE" \
    OPENRESEARCH_GCP_GPU_MACHINE_TYPE="$GPU_MT" OPENRESEARCH_SDAR_MIN_GPUS="$MIN_GPUS" \
    OPENRESEARCH_GCP_PROVISIONING_MODEL=STANDARD \
    bash "$HERE/scripts/gcp_sdar_preflight.sh" prepare 2>&1 | tail -8
  launch_once || { log "FATAL: launch failed after prepare"; exit 3; }
fi

# --- 5. verbose lifecycle-aware watch (auto-stops the VM on a terminal state) -----------
log "run launched; starting verbose watcher (auto-stops VM on terminal) ..."
VERBOSE=1 OPENRESEARCH_GCP_ZONE="$ZONE" OPENRESEARCH_GCP_INSTANCE="$INSTANCE" \
  OPENRESEARCH_GCP_PROJECT="$PROJECT" OPENRESEARCH_GCP_SSH_USER="$SSH_USER" \
  OPENRESEARCH_REMOTE_DIR="$REMOTE_DIR" PROJECT_ID="$PROJECT_ID" \
  bash "$HERE/scripts/sdar_gcp_watch.sh" 2>&1 | tail -120

log "=== runner finished. report persists on the boot disk; pull with: ==="
log "    PROJECT_ID=$PROJECT_ID scripts/sdar_gcp_e2e.sh inspect"
