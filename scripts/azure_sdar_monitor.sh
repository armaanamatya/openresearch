#!/usr/bin/env bash
# azure_sdar_monitor.sh — live babysit loop for an SDAR-on-Azure run.
# Usage: scripts/azure_sdar_monitor.sh [project_id] [interval_s=30]
set -uo pipefail
cd "$(dirname "$0")/.."

NS="${OPENRESEARCH_AZURE_NAMESPACE:-reprolab}"
INTERVAL="${2:-30}"
PID="${1:-}"
if [[ -z "$PID" ]]; then
  PID="$(ls -dt runs/prj_* 2>/dev/null | head -1 | xargs -r basename)"
fi
[[ -n "$PID" ]] || { echo "no project id; pass one explicitly"; exit 1; }
RUN="runs/$PID"
echo "Monitoring $PID (ns=$NS, every ${INTERVAL}s; Ctrl-C to stop)"

while true; do
  clear 2>/dev/null || true
  echo "=== $PID @ $(date -u '+%H:%M:%SZ') ==="
  echo "--- AKS jobs/pods ($NS) ---"
  kubectl get jobs,pods -n "$NS" 2>/dev/null | head -20 || echo "(kubectl unavailable)"
  echo "--- rubric / iteration ---"
  if [[ -f "$RUN/dashboard_events.jsonl" ]]; then
    grep -E '"(rubric_score|repl_iteration)"' "$RUN/dashboard_events.jsonl" 2>/dev/null | tail -1 \
      | jq -rc '{event, score:(.score // .overall_score // .data.score // .rubric_delta), iter:(.iteration // .data.iteration)}' 2>/dev/null || echo "(no rubric event yet)"
  fi
  echo "--- cost (USD) ---"
  if [[ -f "$RUN/cost_ledger.jsonl" ]]; then
    jq -s 'map(.usd // .cost_usd // 0) | add' "$RUN/cost_ledger.jsonl" 2>/dev/null || echo "?"
  fi
  echo "--- warnings ---"
  [[ -f "$RUN/dashboard_events.jsonl" ]] && grep -E '"(run_warning|gpu_escalated|capacity_exhausted)"' "$RUN/dashboard_events.jsonl" 2>/dev/null | tail -3
  echo "--- exec tail ---"
  [[ -f "$RUN/code/.exec_live.log" ]] && tail -4 "$RUN/code/.exec_live.log" 2>/dev/null
  echo
  echo "(recover failed cells:  scripts/azure_sdar_run.sh .env.azure --resume-cells)"
  sleep "$INTERVAL"
done
