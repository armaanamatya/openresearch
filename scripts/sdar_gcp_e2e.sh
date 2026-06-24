#!/usr/bin/env bash
# Deterministic SDAR-on-GCP end-to-end run helper.
#
# Wraps every step that was fiddly to do by hand: the VM machine-type/provisioning
# flips (with the maintenance-policy gotchas), spot/on-demand start, code sync, the
# per-run env overrides (root model / smoke on-off / project id / autodrive), launch,
# monitor, cheap CPU inspection, and teardown. The goal: a full e2e run is one or two
# reproducible commands instead of a dozen ad-hoc gcloud calls.
#
# Authored 2026-06-22 after a session that established: the pre-GPU smoke was the GCP
# blocker (fixed, RL-aware — commit bfd86d52), and a clean full reproduction is then
# gated by ROOT reliability (gpt-chat churns + spot-preempts; claude-oauth degenerates).
# See docs/runbooks/2026-06-22-sdar-gcp-e2e-and-rl-smoke-fix-handoff.md.
#
# Usage (env-parameterised):
#   ROOT=claude-oauth PROV=spot SMOKE=0 PROJECT_ID=sdar_gcp_<id> AUTODRIVE=0 USE_REPO=1 \
#     scripts/sdar_gcp_e2e.sh run        # up + setenv + launch + monitor (the common path)
#   scripts/sdar_gcp_e2e.sh up           # flip a2/<PROV>, start (spot) or poll (ondemand), sync
#   scripts/sdar_gcp_e2e.sh setenv       # write root/smoke/project-id/autodrive overrides on the VM
#   scripts/sdar_gcp_e2e.sh launch       # GREEN-gated launch (detached) via gcp_sdar_preflight.sh
#   scripts/sdar_gcp_e2e.sh monitor      # poll every 150s: GPU-training / terminal / preempt / degenerate
#                                        #   each tick also prints last 4 rendered events + live.log tail
#   scripts/sdar_gcp_e2e.sh watch        # lifecycle-aware verbose watcher (sdar_gcp_watch.sh); auto-stops VM on terminal
#   scripts/sdar_gcp_e2e.sh logs         # live-stream process stdout + .exec_live.log (streams until timeout/Ctrl-C)
#   scripts/sdar_gcp_e2e.sh events       # one-shot readable dump of last EVENTS_N (default 40) dashboard events
#   scripts/sdar_gcp_e2e.sh inspect      # cheap CPU flip, pull final_report+code+state to /tmp/sdar_inspect, stop
#   scripts/sdar_gcp_e2e.sh down         # stop the VM (halt billing)
#   scripts/sdar_gcp_e2e.sh status       # one-line VM + run snapshot
#
# ROOT  : claude-oauth (true local config; may degenerate) | foundry (gpt-chat; arg-churn) | gpt-5/claude (need a funded key)
# PROV  : spot (cheap, can preempt) | ondemand (no preempt, ~3x cost, often STOCKED OUT in us-central1-b)
# SMOKE : 0 (off — bypass the pre-GPU smoke; matches local) | 1 (on — exercises the RL-aware smoke fix)
# USE_REPO: 1 (#62 default ON for SDAR — clone github.com/ZJU-REAL/SDAR + seed code/ in adapt mode) | 0 (from scratch)
# EFFORT : low|medium|high|xhigh|max (oauth root reasoning effort; default high — fixes the reasoning_tokens:0 degenerate loop)
# ROOT_MODEL: optional oauth root model pin (e.g. opus); empty = default sonnet
set -euo pipefail

export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-/home/abheekp/.config/gcloud}"
ZONE="${OPENRESEARCH_GCP_ZONE:-us-central1-b}"
PROJECT="${OPENRESEARCH_GCP_PROJECT:-deepinvent-ext-ut}"
INSTANCE="${OPENRESEARCH_GCP_INSTANCE:-sdar-a100-od}"
SSH_USER="${OPENRESEARCH_GCP_SSH_USER:-abheekp}"
REMOTE_DIR="${OPENRESEARCH_REMOTE_DIR:-/home/abheekp/openresearch}"
GPU_MT="${OPENRESEARCH_GCP_GPU_MACHINE_TYPE:-a2-highgpu-4g}"
CPU_MT="${OPENRESEARCH_GCP_CPU_MACHINE_TYPE:-e2-standard-4}"
MIN_GPUS="${OPENRESEARCH_SDAR_MIN_GPUS:-4}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT="${ROOT:-claude-oauth}"
PROV="${PROV:-spot}"
SMOKE="${SMOKE:-0}"
PROJECT_ID="${PROJECT_ID:-sdar_gcp_e2e}"
AUTODRIVE="${AUTODRIVE:-0}"
USE_REPO="${USE_REPO:-1}"
EFFORT="${EFFORT:-high}"        # OPENRESEARCH_ROOT_EFFORT for the oauth CLI root: low|medium|high|xhigh|max
ROOT_MODEL="${ROOT_MODEL:-}"    # optional oauth root model pin (e.g. "opus"); empty = default sonnet
DRIVE="${DRIVE:-0}"            # OPENRESEARCH_LIFECYCLE_DRIVE: 1 = harness DRIVES plan/implement/run/verify when the root degenerates
PRIMARY="${PRIMARY:-0}"        # OPENRESEARCH_LIFECYCLE_PRIMARY: 1 = harness OWNS the lifecycle (proactive primary mode), root loop skipped

G=(gcloud --project "$PROJECT")
ZP=(--zone "$ZONE" --project "$PROJECT")
status_only() { "${G[@]}" compute instances describe "$INSTANCE" --zone "$ZONE" --format='value(status)' 2>/dev/null || true; }
ssh_() { timeout "${2:-70}" gcloud compute ssh "$SSH_USER@$INSTANCE" "${ZP[@]}" --quiet --command "$1"; }

# set_sched <spot|ondemand>: deterministic scheduling flip. GOTCHAS (all verified):
#   spot     -> needs BOTH --preemptible AND --provisioning-model=SPOT (each alone errors).
#   ondemand -> --no-preemptible --provisioning-model=STANDARD AND --clear-instance-termination-action
#               (a leftover spot termination-action otherwise rejects STANDARD).
#   a2 (GPU) requires --maintenance-policy=TERMINATE (can't MIGRATE); spot-e2 also accepts TERMINATE.
set_sched() {
  if [[ "$1" == spot ]]; then
    "${G[@]}" compute instances set-scheduling "$INSTANCE" --zone "$ZONE" \
      --preemptible --provisioning-model=SPOT --maintenance-policy=TERMINATE >/dev/null
  else
    "${G[@]}" compute instances set-scheduling "$INSTANCE" --zone "$ZONE" \
      --no-preemptible --provisioning-model=STANDARD --maintenance-policy=TERMINATE \
      --clear-instance-termination-action >/dev/null
  fi
}
# set_mt <type>: machine-type change requires the instance TERMINATED.
set_mt() { "${G[@]}" compute instances set-machine-type "$INSTANCE" --zone "$ZONE" --machine-type="$1" >/dev/null; }
config_line() { "${G[@]}" compute instances describe "$INSTANCE" --zone "$ZONE" --format='value(status,machineType.basename(),scheduling.provisioningModel)' 2>/dev/null; }

wait_ssh() { local i; for i in $(seq 1 30); do ssh_ 'echo ok' 25 2>/dev/null | grep -q ok && { echo "  sshd ready"; return 0; }; sleep 10; done; echo "  WARN: sshd not confirmed" >&2; }

# start the VM; for ondemand a single attempt (STOCKOUT is the common outcome -> use poll-ondemand).
start_gpu() {
  set_sched "$PROV"; set_mt "$GPU_MT"
  echo "config -> $(config_line)"
  echo "starting ($PROV) ..."
  if "${G[@]}" compute instances start "$INSTANCE" --zone "$ZONE" 2>&1 | tail -2; then :; fi
  [[ "$(status_only)" == RUNNING ]] || { echo "NOT RUNNING (stockout?). For on-demand use: $0 poll-ondemand" >&2; return 1; }
  wait_ssh
}

sync_code() {
  OPENRESEARCH_GCP_ZONE="$ZONE" OPENRESEARCH_GCP_INSTANCE="$INSTANCE" \
  OPENRESEARCH_GCP_GPU_MACHINE_TYPE="$GPU_MT" OPENRESEARCH_SDAR_MIN_GPUS="$MIN_GPUS" \
  OPENRESEARCH_GCP_PROVISIONING_MODEL="$([[ $PROV == spot ]] && echo SPOT || echo STANDARD)" \
    bash "$HERE/scripts/gcp_sdar_preflight.sh" sync
}

# append per-run overrides to the VM's runs/.cache/sdar_gcp.env (LAST assignment wins on sourcing).
set_env() {
  ssh_ "cd $REMOTE_DIR && {
    printf '\n# --- sdar_gcp_e2e overrides %s ---\n' '$PROJECT_ID' >> runs/.cache/sdar_gcp.env
    echo 'export OPENRESEARCH_SDAR_ROOT=$ROOT' >> runs/.cache/sdar_gcp.env
    echo 'export OPENRESEARCH_METRIC_REALITY_SMOKE=$SMOKE' >> runs/.cache/sdar_gcp.env
    echo 'export OPENRESEARCH_SDAR_PROJECT_ID=$PROJECT_ID' >> runs/.cache/sdar_gcp.env
    echo 'export OPENRESEARCH_OAUTH_AUTODRIVE=$AUTODRIVE' >> runs/.cache/sdar_gcp.env
    echo 'export OPENRESEARCH_ROOT_EFFORT=$EFFORT' >> runs/.cache/sdar_gcp.env
    echo 'export REPROLAB_RLM_ROOT_MODEL_NAME=$ROOT_MODEL' >> runs/.cache/sdar_gcp.env
    echo 'export OPENRESEARCH_SDAR_NO_AUTOSTOP=0' >> runs/.cache/sdar_gcp.env
    echo 'export OPENRESEARCH_USE_AUTHOR_REPO=$USE_REPO' >> runs/.cache/sdar_gcp.env
    echo 'export OPENRESEARCH_LIFECYCLE_DRIVE=$DRIVE' >> runs/.cache/sdar_gcp.env
    echo 'export OPENRESEARCH_LIFECYCLE_PRIMARY=$PRIMARY' >> runs/.cache/sdar_gcp.env
    set -a; . runs/.cache/sdar_gcp.env >/dev/null 2>&1; set +a
    echo \"EFFECTIVE: ROOT=\$OPENRESEARCH_SDAR_ROOT SMOKE=\$OPENRESEARCH_METRIC_REALITY_SMOKE PID=\$OPENRESEARCH_SDAR_PROJECT_ID AUTODRIVE=\$OPENRESEARCH_OAUTH_AUTODRIVE USE_REPO=\$OPENRESEARCH_USE_AUTHOR_REPO EFFORT=\$OPENRESEARCH_ROOT_EFFORT ROOTMODEL=\$REPROLAB_RLM_ROOT_MODEL_NAME DRIVE=\$OPENRESEARCH_LIFECYCLE_DRIVE PRIMARY=\$OPENRESEARCH_LIFECYCLE_PRIMARY\"
    grep -q '^CLAUDE_CODE_OAUTH_TOKEN=' .env && echo 'oauth token: present' || echo 'oauth token: MISSING (claude-oauth root/exec will fail)'
  }"
}

launch() {
  OPENRESEARCH_GCP_ZONE="$ZONE" OPENRESEARCH_GCP_INSTANCE="$INSTANCE" \
  OPENRESEARCH_GCP_GPU_MACHINE_TYPE="$GPU_MT" OPENRESEARCH_SDAR_MIN_GPUS="$MIN_GPUS" \
  OPENRESEARCH_GCP_PROVISIONING_MODEL="$([[ $PROV == spot ]] && echo SPOT || echo STANDARD)" \
  OPENRESEARCH_SDAR_PROJECT_ID="$PROJECT_ID" OPENRESEARCH_SDAR_ROOT="$ROOT" \
    bash "$HERE/scripts/gcp_sdar_preflight.sh" launch 2>&1 | tail -28
}

monitor() {
  local R R_recent R_live i s g deg_msg
  R='cd '"$REMOTE_DIR"'; P=runs/'"$PROJECT_ID"'; echo "XR=$(cat $P/experiment_runs.jsonl 2>/dev/null|wc -l) DEG=$(grep -c root_degenerate $P/dashboard_events.jsonl 2>/dev/null||echo 0) DONE=$(grep -cE "run_complete|run_fatal|run_interrupted" $P/dashboard_events.jsonl 2>/dev/null||echo 0) CELLS=$([ -f $P/code/cells.json ] && echo y || echo n) GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null|sort -rn|head -1)"'
  # recent events block: last 4 rendered events + last line of .exec_live.log if present
  R_recent='cd '"$REMOTE_DIR"'; P=runs/'"$PROJECT_ID"'; python3 scripts/render_run_events.py --tail 4 "$P/dashboard_events.jsonl" 2>/dev/null; L="$P/code/.exec_live.log"; [ -f "$L" ] && echo "live.log: $(tail -1 "$L" 2>/dev/null)"'
  # degenerate warning message fetch
  R_deg='cd '"$REMOTE_DIR"'; P=runs/'"$PROJECT_ID"'; grep -o "\"message\":\"[^\"]*\"" "$P/dashboard_events.jsonl" 2>/dev/null | tail -5'
  for i in $(seq 1 200); do
    s="$(status_only)"; [[ "$s" != RUNNING ]] && { echo "[$i] VM=$s (preempted/stopped)"; return 0; }
    out="$(ssh_ "$R" 55 2>/dev/null || true)"; echo "[$i $(date -u +%H:%MZ)] $(echo "$out"|tr '\n' ' ')"
    # print recent events block after each tick
    ssh_ "$R_recent" 30 2>/dev/null | sed 's/^/  /' || true
    if echo "$out" | grep -q 'DEG=[1-9]'; then
      deg_msg="$(ssh_ "$R_deg" 20 2>/dev/null | grep -i 'degenerate\|forced_iteration\|root_degenerate' | tail -3 | sed 's/^/  /' || true)"
      echo ">>> DEGENERATE (root)${deg_msg:+$'\n'"$deg_msg"}"; return 0
    fi
    echo "$out" | grep -q 'DONE=[1-9]' && { echo ">>> TERMINAL — run 'inspect'"; return 0; }
    g="$(echo "$out" | sed -n 's/.*GPU=//p' | grep -oE '^[0-9]+' || true)"; [[ "${g:-0}" -gt 60 ]] 2>/dev/null && { echo ">>> GPU TRAINING (util=$g) — real grid reached"; return 0; }
    sleep 150
  done
}

logs() {
  echo "streaming VM stdout + live training log (Ctrl-C or ${LOGS_TIMEOUT:-1800}s timeout) ..."
  ssh_ "cd $REMOTE_DIR && tail -n 60 -f runs/sdar_gcp_run.out runs/$PROJECT_ID/code/.exec_live.log 2>/dev/null" "${LOGS_TIMEOUT:-1800}"
}

events() {
  local n="${EVENTS_N:-40}"
  echo "last $n dashboard events for $PROJECT_ID:"
  ssh_ "cd $REMOTE_DIR && python3 scripts/render_run_events.py --tail $n runs/$PROJECT_ID/dashboard_events.jsonl 2>/dev/null" 60
}

inspect() {
  echo "flipping to cheap CPU ($CPU_MT spot) to read the disk ..."
  set_sched spot; set_mt "$CPU_MT"; echo "config -> $(config_line)"
  "${G[@]}" compute instances start "$INSTANCE" --zone "$ZONE" >/dev/null 2>&1 || true
  [[ "$(status_only)" == RUNNING ]] || { echo "could not start CPU inspect VM" >&2; return 1; }
  wait_ssh
  mkdir -p /tmp/sdar_inspect
  ssh_ "cd $REMOTE_DIR/runs/$PROJECT_ID && tar czf /tmp/art.tgz final_report.json final_report.md experiment_runs.jsonl dashboard_events.jsonl \$(find code -maxdepth 3 -name '*.py' 2>/dev/null) rlm_state 2>/dev/null; du -h /tmp/art.tgz" 90
  gcloud compute scp "$SSH_USER@$INSTANCE:/tmp/art.tgz" /tmp/sdar_inspect/ "${ZP[@]}" --quiet
  ( cd /tmp/sdar_inspect && tar xzf art.tgz 2>/dev/null && echo "pulled to /tmp/sdar_inspect:" && ls )
  "${G[@]}" compute instances stop "$INSTANCE" --zone "$ZONE" >/dev/null && echo "CPU inspect VM stopped"
}

poll_ondemand() {
  echo "polling on-demand $GPU_MT capacity every 600s (free per stockout) ..."
  set_sched ondemand; set_mt "$GPU_MT"; echo "config -> $(config_line)"
  local i out
  for i in $(seq 1 144); do
    out="$("${G[@]}" compute instances start "$INSTANCE" --zone "$ZONE" 2>&1)" && { echo "[$i] CAPACITY RETURNED"; wait_ssh; sync_code; echo "READY — now: $0 setenv && $0 launch && $0 monitor"; return 0; }
    echo "$out" | grep -qiE 'STOCKOUT|EXHAUSTED|enough resources' && { echo "[$i $(date -u +%H:%MZ)] stocked out"; sleep 600; } || { echo "non-stockout err: $(echo "$out"|tail -2)"; return 1; }
  done
}

case "${1:-status}" in
  up)            [[ "$PROV" == ondemand ]] && poll_ondemand || { start_gpu && sync_code; } ;;
  setenv)        set_env ;;
  launch)        launch ;;
  monitor)       monitor ;;
  watch)         VERBOSE="${VERBOSE:-1}" OPENRESEARCH_GCP_ZONE="$ZONE" OPENRESEARCH_GCP_INSTANCE="$INSTANCE" \
                   OPENRESEARCH_GCP_PROJECT="$PROJECT" OPENRESEARCH_GCP_SSH_USER="$SSH_USER" \
                   OPENRESEARCH_REMOTE_DIR="$REMOTE_DIR" PROJECT_ID="$PROJECT_ID" \
                   bash "$HERE/scripts/sdar_gcp_watch.sh" ;;
  logs)          logs ;;
  events)        events ;;
  inspect)       inspect ;;
  down)          "${G[@]}" compute instances stop "$INSTANCE" --zone "$ZONE" 2>&1 | tail -1; echo "status=$(status_only)" ;;
  poll-ondemand) poll_ondemand ;;
  status)        echo "VM: $(config_line)" ;;
  run)           { [[ "$PROV" == ondemand ]] && poll_ondemand || { start_gpu && sync_code; }; } && set_env && launch && monitor ;;
  *)             echo "unknown action: $1 (up|setenv|launch|monitor|watch|logs|events|inspect|down|poll-ondemand|status|run)" >&2; exit 2 ;;
esac
