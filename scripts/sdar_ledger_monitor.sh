#!/usr/bin/env bash
# Autonomous coworker-ledger monitor + finalizer for a long SDAR/GCP run.
#
# Polls the live run on the GPU VM; logs periodic progress to the coworker
# ledger; and on terminal it pulls the report to the LOCAL runs/<id>/ *while the
# VM is still RUNNING* (no fragile a2-ultragpu CPU-flip), logs the final outcome
# (-> ledger + /leaderboard), then stops the VM to halt billing.
#
# This replaces the runner's own watcher (which auto-stops the VM before the
# report can be pulled). Fail-soft throughout; run detached so it survives the
# launching session:
#
#   setsid nohup bash scripts/sdar_ledger_monitor.sh <project_id> [interval_s] \
#     > runs/_<project_id>_ledger.log 2>&1 < /dev/null &
set -uo pipefail
PROJECT_ID="${1:?usage: sdar_ledger_monitor.sh <project_id> [interval_s]}"
INTERVAL="${2:-900}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-/home/abheekp/.config/gcloud}"
ZONE="${OPENRESEARCH_GCP_ZONE:-us-central1-c}"
INSTANCE="${OPENRESEARCH_GCP_INSTANCE:-sdar-ultra}"
PROJECT="${OPENRESEARCH_GCP_PROJECT:-deepinvent-ext-ut}"
SSH_USER="${OPENRESEARCH_GCP_SSH_USER:-abheekp}"
REMOTE_DIR="${OPENRESEARCH_REMOTE_DIR:-/home/abheekp/openresearch}"
SANDBOX="${SANDBOX_LABEL:-gcp/$INSTANCE/2xA100-80GB}"
PY="$HERE/.venv/bin/python"
ZP=(--zone "$ZONE" --project "$PROJECT")

log() { echo "[$(date -u +%Y-%m-%dT%H:%MZ)] $*"; }
vm_status() { gcloud compute instances describe "$INSTANCE" "${ZP[@]}" --format='value(status)' 2>/dev/null || echo UNKNOWN; }
ssh_() { timeout "${2:-70}" gcloud compute ssh "$SSH_USER@$INSTANCE" "${ZP[@]}" --quiet --command "$1" 2>/dev/null; }
runlog() { "$PY" "$HERE/scripts/sdar_runlog.py" --project-id "$PROJECT_ID" --sandbox "$SANDBOX" "$@" >/dev/null 2>&1 || true; }

log "ledger-monitor up for $PROJECT_ID (poll ${INTERVAL}s on $INSTANCE/$ZONE)"
tick=0
while true; do
  st="$(vm_status)"
  if [[ "$st" != "RUNNING" ]]; then
    log "VM=$st (not RUNNING) — terminal/external-stop; finalizing"
    break
  fi
  alive="$(ssh_ "pgrep -f 'backend.cli reproduce' >/dev/null && echo yes || echo no" 45)"
  if [[ "$alive" == "no" ]]; then
    log "reproduce process gone — run terminal; finalizing"
    break
  fi
  if (( tick % 4 == 0 )); then
    cells="$(ssh_ "find $REMOTE_DIR/runs/$PROJECT_ID/code/outputs -name metrics.json 2>/dev/null | wc -l" 60)"
    runlog --event progress --note "live: cells_with_metrics=${cells:-?}"
    log "progress: cells_with_metrics=${cells:-?}"
  fi
  tick=$((tick + 1))
  sleep "$INTERVAL"
done

# --- finalize: pull the report locally (VM may still be RUNNING — no CPU flip) ---
log "pulling artifacts to runs/$PROJECT_ID ..."
mkdir -p "$HERE/runs/$PROJECT_ID"
if ssh_ "cd $REMOTE_DIR/runs/$PROJECT_ID 2>/dev/null && tar czf /tmp/art_${PROJECT_ID}.tgz \
      final_report.json final_report.md demo_status.json experiment_runs.jsonl \
      cost_ledger.jsonl dashboard_events.jsonl rlm_state generated_rubric.json 2>/dev/null; echo packed" 150; then
  if gcloud compute scp "$SSH_USER@$INSTANCE:/tmp/art_${PROJECT_ID}.tgz" "/tmp/art_${PROJECT_ID}.tgz" "${ZP[@]}" --quiet 2>/dev/null; then
    tar xzf "/tmp/art_${PROJECT_ID}.tgz" -C "$HERE/runs/$PROJECT_ID" 2>/dev/null && log "artifacts pulled" || log "WARN: extract failed"
  else
    log "WARN: scp failed — report stays on the VM disk"
  fi
fi

runlog --event finished --note "auto-finalized by ledger-monitor"
log "final outcome logged to ledger + leaderboard"

# --- halt billing ---
if [[ "$(vm_status)" == "RUNNING" ]]; then
  log "stopping VM to halt billing ..."
  gcloud compute instances stop "$INSTANCE" "${ZP[@]}" --discard-local-ssd=true 2>&1 | tail -1
fi
log "ledger-monitor done for $PROJECT_ID"
