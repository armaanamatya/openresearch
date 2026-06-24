#!/usr/bin/env bash
# Minimal, fragility-free SDAR run monitor. No `set -e` (so a transient SSH non-zero
# never kills it), no auto-stop (the run self-stops on finalize). Appends one state
# line per tick; exits when the VM is no longer RUNNING or a terminal event appears.
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-/home/abheekp/.config/gcloud}"
ZONE="${ZONE:-us-central1-b}"; PROJECT="${PROJECT:-deepinvent-ext-ut}"
INSTANCE="${INSTANCE:-sdar-a100-od}"; SSH_USER="${SSH_USER:-abheekp}"
PID="${PID:-sdar_optimal_1g}"; INTERVAL="${INTERVAL:-150}"; MAX="${MAX:-200}"
RDIR="/home/abheekp/openresearch"
ssh_() { timeout 60 gcloud compute ssh "$SSH_USER@$INSTANCE" --zone "$ZONE" --project "$PROJECT" --quiet --command "$1" 2>/dev/null; }
log() { echo "[$(date -u +%H:%MZ)] $*"; }

for i in $(seq 1 "$MAX"); do
  vm=$(gcloud --project "$PROJECT" compute instances describe "$INSTANCE" --zone "$ZONE" --format='value(status)' 2>/dev/null)
  if [ "$vm" != "RUNNING" ]; then log "VM=$vm — run self-stopped or VM gone; exiting"; break; fi
  state=$(ssh_ "cd $RDIR; P=runs/$PID; echo \"XR=\$(cat \$P/experiment_runs.jsonl 2>/dev/null|wc -l) ev=\$(cat \$P/dashboard_events.jsonl 2>/dev/null|wc -l) cells=\$([ -f \$P/code/cells.json ] && echo y||echo n) DONE=\$(grep -cE 'run_complete|run_fatal|run_interrupted' \$P/dashboard_events.jsonl 2>/dev/null) gpu=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null|head -1) score=\$(grep -o '\"overall_score\"[: ]*[0-9.]*' \$P/dashboard_events.jsonl 2>/dev/null|tail -1)\"; tail -1 \$P/code/.exec_live.log 2>/dev/null|cut -c1-100")
  log "[$i] $state"
  echo "$state" | grep -q 'DONE=[1-9]' && { log ">>> TERMINAL event — run finished"; break; }
  sleep "$INTERVAL"
done
log "monitor exit"
