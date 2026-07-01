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
ROOT="${ROOT:-foundry}"
RUN_EXPERIMENT_TIMEOUT_S="${RUN_EXPERIMENT_TIMEOUT_S:-2400}"
SDAR_VRAM_GB="${SDAR_VRAM_GB:-40}"     # per-GPU VRAM the run reports to the capacity gate (80 for a2-ultragpu)
LIFECYCLE_MAX_IMPROVE="${LIFECYCLE_MAX_IMPROVE:-}"   # OPENRESEARCH_LIFECYCLE_MAX_IMPROVE override; empty = harness default (set -u safe)
EVAL_PROVENANCE_GUARD="${OPENRESEARCH_EVAL_PROVENANCE_GUARD:-}"  # S1 eval-metric provenance guard; empty = default OFF (byte-identical)
REUSE_RUBRIC="${OPENRESEARCH_REUSE_RUBRIC:-}"                    # A/B: pin the pre-seeded rubric so the grader doesn't drift; empty = default
# Grok-root guardrails: prevent a non-Claude root from passing placeholder args
# or stubbing metrics.  Default ON (1); override via env before launching.
ARG_CONTRACTS="${OPENRESEARCH_ARG_CONTRACTS:-1}"
STUB_METRICS_GUARD="${OPENRESEARCH_STUB_METRICS_GUARD:-1}"
GRADER_SAMPLES="${GRADER_SAMPLES:-1}"                  # median-of-N leaf grading; 1 is σ-gate-sufficient + ~3x faster (a big grid x3 samples blew the verify cap); 3 = fidelity-critical opt-in
VERIFY_TIMEOUT_S="${VERIFY_TIMEOUT_S:-1800}"           # OPENRESEARCH_VERIFY_AGAINST_RUBRIC_TIMEOUT_S: verify wall-clock cap (table default 600s is too tight to grade a full grid)
NO_AUTOSTOP="${NO_AUTOSTOP:-1}"                        # 1 = leave the VM UP on finish so the ledger-monitor pulls the report THEN stops it; 0 = run self-stops (strands the report on the a2-ultragpu disk)
MAX_RUN_DUR="${OPENRESEARCH_GCP_MAX_RUN_DURATION:-100800s}"   # 28h hard ceiling > harness 24h budget

# --- Persistent cache disk + machine image (default OFF — no regression) -----
# OPENRESEARCH_SDAR_USE_CACHE_DISK=1: attach a named pd-ssd data disk that
# survives VM delete/recreate; repoints HF_HOME/cache/pip at /mnt/sdar-cache.
# ZONE MISMATCH NOTE: sdar-ultra (1 TB, the adoptable orphaned disk) is in
# us-central1-c while this runner's VM (sdar-a100-od) defaults to us-central1-b.
# GCP forbids cross-zone disk attach. OPERATOR STEP before enabling:
#   (a) snapshot sdar-ultra in us-central1-c then recreate it in us-central1-b
#       as 'sdar-cache', or
#   (b) set OPENRESEARCH_SDAR_CACHE_DISK_ZONE=us-central1-b and create a new
#       1000 GB pd-ssd named 'sdar-cache' in that zone.
# The attach function warns and skips (never aborts) on a zone mismatch.
USE_CACHE_DISK="${OPENRESEARCH_SDAR_USE_CACHE_DISK:-0}"
CACHE_DISK="${OPENRESEARCH_SDAR_CACHE_DISK:-sdar-cache}"
CACHE_MOUNT="${OPENRESEARCH_SDAR_CACHE_MOUNT:-/mnt/sdar-cache}"
CACHE_DISK_ZONE="${OPENRESEARCH_SDAR_CACHE_DISK_ZONE:-${ZONE}}"
# OPENRESEARCH_SDAR_USE_MI=1: boot from the named machine image at CREATE time
# (--source-machine-image instead of --image-family; MI is multi-zonal).
# Default OFF: stock DLVM CREATE path is unchanged when unset.
USE_MI="${OPENRESEARCH_SDAR_USE_MI:-0}"
MI_NAME="${OPENRESEARCH_SDAR_MI_NAME:-sdar-mi-20260620}"

G=(gcloud --project "$PROJECT")
ZP=(--zone "$ZONE" --project "$PROJECT")
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
ssh_() { timeout "${2:-90}" gcloud compute ssh "$SSH_USER@$INSTANCE" "${ZP[@]}" --quiet --command "$1"; }
status_() { "${G[@]}" compute instances describe "$INSTANCE" --zone "$ZONE" --format='value(status)' 2>/dev/null || true; }
wait_ssh() { local i; for i in $(seq 1 30); do ssh_ 'echo ok' 25 2>/dev/null | grep -q ok && { log "  sshd ready"; return 0; }; sleep 10; done; log "  WARN: sshd not confirmed"; }

# arm_max_run_duration: GCP control-plane-enforced hard stop ceiling, independent of
# any local/VM process. Must run while the instance is TERMINATED (set-scheduling rule).
# STOP preserves the boot disk; a2-ultragpu mandatory local SSDs need the discard flag.
arm_max_run_duration() {
  local extra=()
  [[ "$GPU_MT" == a2-ultragpu* ]] && extra+=(--discard-local-ssds-at-termination-timestamp=true)
  "${G[@]}" compute instances set-scheduling "$INSTANCE" --zone "$ZONE" \
    --max-run-duration="$MAX_RUN_DUR" --instance-termination-action=STOP "${extra[@]}" \
    >/dev/null 2>&1 \
    && log "armed GCP max-run-duration=$MAX_RUN_DUR action=STOP (hard idle ceiling)" \
    || log "WARN: could not arm max-run-duration (non-fatal; VM-side watchdog still covers)"
}

# attach_and_mount_cache_disk: idempotent, fail-soft (logs warning + returns 0
# on any error so the run falls back to boot-disk cache mode, never aborts).
attach_and_mount_cache_disk() {
  [[ "$USE_CACHE_DISK" != "1" ]] && return 0
  if [[ "$CACHE_DISK_ZONE" != "$ZONE" ]]; then
    log "WARN: cache disk '$CACHE_DISK' zone=$CACHE_DISK_ZONE != VM zone=$ZONE"
    log "  GCP forbids cross-zone disk attach. OPERATOR STEP: snapshot sdar-ultra"
    log "  (us-central1-c) and recreate 'sdar-cache' in us-central1-b, OR set"
    log "  OPENRESEARCH_SDAR_CACHE_DISK_ZONE=us-central1-b and create the disk there."
    log "  Falling back to boot-disk cache mode."
    return 0
  fi
  # Create the disk if absent (idempotent; 1000 GB pd-ssd matching sdar-ultra).
  if ! "${G[@]}" compute disks describe "$CACHE_DISK" --zone "$ZONE" \
       --format='value(name)' >/dev/null 2>&1; then
    log "cache disk '$CACHE_DISK' absent in $ZONE — creating 1000 GB pd-ssd ..."
    "${G[@]}" compute disks create "$CACHE_DISK" --zone "$ZONE" \
      --size=1000GB --type=pd-ssd >/dev/null 2>&1 \
      && log "  created $CACHE_DISK (1000 GB pd-ssd, $ZONE)" \
      || { log "WARN: could not create cache disk — boot-disk cache mode"; return 0; }
  fi
  # Attach if not already attached.
  local _users
  _users="$("${G[@]}" compute disks describe "$CACHE_DISK" --zone "$ZONE" \
    --format='value(users)' 2>/dev/null || true)"
  if echo "$_users" | grep -qF "$INSTANCE"; then
    log "cache disk '$CACHE_DISK' already attached to $INSTANCE"
  else
    "${G[@]}" compute instances attach-disk "$INSTANCE" --zone "$ZONE" \
      --disk "$CACHE_DISK" --mode=rw --device-name=sdar-cache >/dev/null 2>&1 \
      && log "attached cache disk '$CACHE_DISK' to $INSTANCE" \
      || { log "WARN: cache disk attach failed — boot-disk cache mode"; return 0; }
  fi
  # Format (first attach only) and mount on the VM.
  # $CACHE_MOUNT is expanded locally; \$DEV/\$MNT/\$(...) are evaluated on the VM.
  ssh_ "$(cat <<_MOUNT_EOF
set -eo pipefail
DEV=/dev/disk/by-id/google-sdar-cache
MNT=$CACHE_MOUNT
if ! blkid "\$DEV" >/dev/null 2>&1; then
  echo '[cache] formatting \$DEV as ext4 (first attach)...'
  sudo mkfs.ext4 -F "\$DEV"
fi
sudo mkdir -p "\$MNT"
if mountpoint -q "\$MNT"; then
  echo '[cache] already mounted at \$MNT'
else
  sudo mount -o discard,defaults "\$DEV" "\$MNT"
  echo '[cache] mounted \$DEV at \$MNT'
fi
sudo mkdir -p "\$MNT/hf" "\$MNT/envs" "\$MNT/pip"
sudo chown -R "\$(id -un):\$(id -gn)" "\$MNT"
echo '[cache] ready: hf/ envs/ pip/'
_MOUNT_EOF
  )" 90 \
    && log "cache disk mounted at $CACHE_MOUNT on $INSTANCE" \
    || log "WARN: VM-side cache disk mount failed — boot-disk cache mode"
}

# Source the credential preflight helper (defines preflight_root_credential +
# preflight_subagent_oauth). Fail-soft: if the helper is absent, warn and continue
# (a missing helper should never silently block a run, only a dead key should).
_CRED_PREFLIGHT_SH="${HERE}/scripts/sdar_cred_preflight.sh"
if [ -f "$_CRED_PREFLIGHT_SH" ]; then
  # shellcheck source=scripts/sdar_cred_preflight.sh
  . "$_CRED_PREFLIGHT_SH"
else
  log "WARN: sdar_cred_preflight.sh not found at $_CRED_PREFLIGHT_SH — skipping credential preflight"
  preflight_root_credential() { :; }
  preflight_subagent_oauth()   { :; }
fi

log "=== SDAR optimal on-demand runner: project_id=$PROJECT_ID root=$ROOT zone=$ZONE ==="
log "EFFECTIVE launch config: root=$ROOT sub-agents=executor=sonnet,grader=sonnet,verifier=sonnet guardrails=ARG_CONTRACTS=$ARG_CONTRACTS STUB_METRICS_GUARD=$STUB_METRICS_GUARD"

# --- CREDENTIAL PREFLIGHT (fail-fast before any VM provisioning / money spent) ---
# A dead root credential exits 1 here; a missing oauth WARN (non-fatal) prints.
preflight_root_credential "$ROOT"
preflight_subagent_oauth

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
elif [[ "$_st" == "MISSING" && ( -n "${CREATE_IMAGE:-}" || "$USE_MI" == "1" ) ]]; then
  # CREATE-poll: the instance does not exist (e.g. an a2-ultragpu that cannot be pre-created
  # while stocked out). Repeatedly try to CREATE it on-demand until capacity returns. On-demand
  # by default (no --preemptible); GPU types require --maintenance-policy=TERMINATE.
  # When USE_MI=1, boot from the named machine image (multi-zonal, warm OS+drivers+venv)
  # instead of a stock DLVM image-family (cold, ~30-60 min re-warm). DEFAULT OFF (no regression).
  if [[ "$USE_MI" == "1" ]]; then
    log "instance $INSTANCE absent; CREATE-polling $GPU_MT in $ZONE (machine-image $MI_NAME) every ${POLL_INTERVAL}s"
  else
    log "instance $INSTANCE absent; CREATE-polling $GPU_MT in $ZONE (image $CREATE_IMAGE) every ${POLL_INTERVAL}s"
  fi
  _create_mrd_extra=()
  [[ "$GPU_MT" == a2-ultragpu* ]] && _create_mrd_extra+=(--discard-local-ssds-at-termination-timestamp=true)
  for i in $(seq 1 "$MAX_POLLS"); do
    if [[ "$USE_MI" == "1" ]]; then
      # --source-machine-image and --image-family are mutually exclusive.
      # Machine images are multi-zonal so zone doesn't constrain the image choice.
      out="$("${G[@]}" compute instances create "$INSTANCE" --zone "$ZONE" --machine-type "$GPU_MT" \
        --source-machine-image "$MI_NAME" \
        --maintenance-policy TERMINATE --no-restart-on-failure \
        --max-run-duration="$MAX_RUN_DUR" --instance-termination-action=STOP "${_create_mrd_extra[@]}" 2>&1)"
    else
      out="$("${G[@]}" compute instances create "$INSTANCE" --zone "$ZONE" --machine-type "$GPU_MT" \
        --image-family "$CREATE_IMAGE" --image-project "${CREATE_IMAGE_PROJECT:-deeplearning-platform-release}" \
        --maintenance-policy TERMINATE --boot-disk-size "${CREATE_DISK_GB:-1000}" --boot-disk-type pd-ssd \
        --metadata install-nvidia-driver=True --no-restart-on-failure \
        --max-run-duration="$MAX_RUN_DUR" --instance-termination-action=STOP "${_create_mrd_extra[@]}" 2>&1)"
    fi
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
  arm_max_run_duration
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
  _lk_stop="$("${G[@]}" compute instances stop "$INSTANCE" --zone "$ZONE" 2>&1)" || true
  if echo "$_lk_stop" | grep -qiE 'local ssd|discard-local-ssd|cannot be stopped'; then
    "${G[@]}" compute instances stop "$INSTANCE" --zone "$ZONE" --discard-local-ssd=true >/dev/null 2>&1 || true
  fi
  exit 0
fi
log "acquired launch lock — proceeding as the run target ($ZONE/$INSTANCE/$GPU_MT)"
wait_ssh
# Attach + mount the persistent cache disk after the VM is up (idempotent; no-op
# when USE_CACHE_DISK=0 so boot-disk cache mode is byte-identical when unset).
attach_and_mount_cache_disk

# --- 2. sync latest code ---------------------------------------------------------------
log "syncing latest code to the VM ..."
OPENRESEARCH_GCP_ZONE="$ZONE" OPENRESEARCH_GCP_INSTANCE="$INSTANCE" \
  OPENRESEARCH_GCP_GPU_MACHINE_TYPE="$GPU_MT" OPENRESEARCH_SDAR_MIN_GPUS="$MIN_GPUS" \
  OPENRESEARCH_GCP_PROVISIONING_MODEL=STANDARD \
  bash "$HERE/scripts/gcp_sdar_preflight.sh" sync 2>&1 | tail -4

# ensure the env file exists (prepare creates it + warms caches on a fresh disk).
# When USE_CACHE_DISK=1 and the data disk already has a .warm_ok sentinel, prepare
# is fast: sdar_gcp_assets.py finds the cached models/datasets on /mnt/sdar-cache
# and skips the expensive re-download (only pip install + env provisioning runs).
_env_ok=0
ssh_ "test -f $REMOTE_DIR/runs/.cache/sdar_gcp.env" 30 2>/dev/null && _env_ok=1 || true
if [[ "$_env_ok" != "1" ]]; then
  if [[ "$USE_CACHE_DISK" == "1" && "$CACHE_DISK_ZONE" == "$ZONE" ]] && \
     ssh_ "test -f ${CACHE_MOUNT}/.warm_ok" 20 2>/dev/null; then
    log "cache disk warm (.warm_ok present) — prepare will use cached assets (fast path)"
  else
    log "sdar_gcp.env missing — running prepare to warm env/caches (first-boot path)"
  fi
  OPENRESEARCH_GCP_ZONE="$ZONE" OPENRESEARCH_GCP_INSTANCE="$INSTANCE" \
    OPENRESEARCH_GCP_GPU_MACHINE_TYPE="$GPU_MT" OPENRESEARCH_SDAR_MIN_GPUS="$MIN_GPUS" \
    OPENRESEARCH_GCP_PROVISIONING_MODEL=STANDARD \
    OPENRESEARCH_SDAR_USE_CACHE_DISK="$USE_CACHE_DISK" \
    OPENRESEARCH_SDAR_CACHE_DISK="$CACHE_DISK" \
    OPENRESEARCH_SDAR_CACHE_MOUNT="$CACHE_MOUNT" \
    OPENRESEARCH_SDAR_CACHE_DISK_ZONE="$CACHE_DISK_ZONE" \
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
  echo 'export OPENRESEARCH_VERIFY_AGAINST_RUBRIC_TIMEOUT_S=$VERIFY_TIMEOUT_S'  >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_SDAR_VRAM_GB=$SDAR_VRAM_GB'                          >> runs/.cache/sdar_gcp.env
  [ -n '$LIFECYCLE_MAX_IMPROVE' ] && echo 'export OPENRESEARCH_LIFECYCLE_MAX_IMPROVE=$LIFECYCLE_MAX_IMPROVE' >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_GRADER_SAMPLES=$GRADER_SAMPLES'                     >> runs/.cache/sdar_gcp.env
  [ -n '$EVAL_PROVENANCE_GUARD' ] && echo 'export OPENRESEARCH_EVAL_PROVENANCE_GUARD=$EVAL_PROVENANCE_GUARD' >> runs/.cache/sdar_gcp.env
  [ -n '$REUSE_RUBRIC' ] && echo 'export OPENRESEARCH_REUSE_RUBRIC=$REUSE_RUBRIC' >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_SDAR_NO_AUTOSTOP=$NO_AUTOSTOP'                      >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_ARG_CONTRACTS=$ARG_CONTRACTS'                       >> runs/.cache/sdar_gcp.env
  echo 'export OPENRESEARCH_STUB_METRICS_GUARD=$STUB_METRICS_GUARD'            >> runs/.cache/sdar_gcp.env
  echo 'unset OPENRESEARCH_BASELINE_EXTRA_GUIDANCE'                             >> runs/.cache/sdar_gcp.env
  set -a; . runs/.cache/sdar_gcp.env >/dev/null 2>&1; set +a
  echo \"EFFECTIVE: ROOT=\$OPENRESEARCH_SDAR_ROOT MODELS=\$OPENRESEARCH_SDAR_MODELS PRIMARY=\$OPENRESEARCH_LIFECYCLE_PRIMARY TIMEOUT=\$OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S USE_REPO=\$OPENRESEARCH_USE_AUTHOR_REPO ARG_CONTRACTS=\$OPENRESEARCH_ARG_CONTRACTS STUB_METRICS_GUARD=\$OPENRESEARCH_STUB_METRICS_GUARD\"
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

# --- 5. verbose lifecycle-aware watch -------------------------------------------------
# The watcher (spawned here, so it lives inside this detached runner — session-
# survivable, no fragile separate monitor) owns the terminal sequence: on a
# finalized run it PULLS the report to the local runs/<id>/, appends the honest
# outcome to the coworker ledger, THEN stops the VM. NO_AUTOSTOP=1 keeps the run
# itself from self-stopping first (which would strand the report). KEEP_UP=1
# overrides only the stop (leaves the VM up for manual inspection).
log "run launched; starting watcher (pull report → log ledger → stop VM on terminal) ..."
VERBOSE=1 PULL_AND_LOG=1 KEEP_UP="${KEEP_UP:-0}" MAX_TICKS="${MAX_TICKS:-900}" \
  OPENRESEARCH_GCP_ZONE="$ZONE" OPENRESEARCH_GCP_INSTANCE="$INSTANCE" \
  OPENRESEARCH_GCP_PROJECT="$PROJECT" OPENRESEARCH_GCP_SSH_USER="$SSH_USER" \
  OPENRESEARCH_REMOTE_DIR="$REMOTE_DIR" PROJECT_ID="$PROJECT_ID" LOCAL_REPO="$HERE" \
  bash "$HERE/scripts/sdar_gcp_watch.sh" 2>&1 | tail -120

log "=== runner finished. report persists on the boot disk; pull with: ==="
log "    PROJECT_ID=$PROJECT_ID scripts/sdar_gcp_e2e.sh inspect"
