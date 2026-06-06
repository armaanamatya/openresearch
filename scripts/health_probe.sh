#!/usr/bin/env bash
# Single-shot health snapshot for an in-flight RLM/RDR run.
# Used by the agent monitoring loop (see
# docs/superpowers/specs/2026-05-23-e2e-rlmpaper-localhost-run-design.md §
# "Monitoring loop"). Cheap, idempotent, prints a compact one-block summary
# and exits non-zero only on hard "wedged" signal.
#
# Wedged := dashboard_events.jsonl hasn't grown in WEDGE_THRESH_SEC AND
# no claude-agent-sdk worker process is currently alive AND no docker exec
# of the run container is currently alive. Mirrors the §1b false-alarm
# check in docs/runbooks/known-issues-and-monitoring.md.
#
# Usage:  scripts/health_probe.sh <projectId> [WEDGE_THRESH_SEC=600]
# Exit:   0 healthy / progressing  |  1 missing/bad args  |  2 wedged

set -euo pipefail

PROJECT="${1:-}"
THRESH="${2:-600}"

if [[ -z "$PROJECT" ]]; then
  echo "usage: $0 <projectId> [wedge_thresh_sec=600]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$REPO_ROOT/runs/$PROJECT"
EVENTS="$RUN_DIR/dashboard_events.jsonl"
STATUS_FILE="$RUN_DIR/demo_status.json"
STDERR_LOG="$RUN_DIR/runner.stderr.log"
FINAL_REPORT="$RUN_DIR/final_report.json"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERR  run dir does not exist: $RUN_DIR" >&2
  exit 1
fi

now_epoch=$(date +%s)

# ── status ────────────────────────────────────────────────────────────────
if [[ -f "$STATUS_FILE" ]]; then
  status=$(python3 -c "import json,sys; print(json.load(open('$STATUS_FILE')).get('status','?'))" 2>/dev/null || echo "?")
else
  status="(no status file yet)"
fi

# ── final_report ──────────────────────────────────────────────────────────
if [[ -f "$FINAL_REPORT" ]]; then
  final="present"
else
  final="absent"
fi

# ── event log freshness ───────────────────────────────────────────────────
if [[ -f "$EVENTS" ]]; then
  events_mtime=$(stat -f %m "$EVENTS" 2>/dev/null || stat -c %Y "$EVENTS")
  events_age=$((now_epoch - events_mtime))
  events_count=$(wc -l < "$EVENTS" | tr -d ' ')
  last_event=$(tail -1 "$EVENTS" 2>/dev/null | python3 -c "import json,sys; e=json.loads(sys.stdin.read()); print(e.get('event','?'), e.get('name',''))" 2>/dev/null || echo "(unparseable)")
else
  events_age=9999
  events_count=0
  last_event="(no event log)"
fi

# ── liveness signals ──────────────────────────────────────────────────────
sdk_workers=$(pgrep -fc "claude_agent_sdk/_bundled" 2>/dev/null || echo 0)
docker_execs=$(docker ps --filter "label=openresearch.project=$PROJECT" -q 2>/dev/null | wc -l | tr -d ' ')
recent_files=$(find "$RUN_DIR/code" -mmin -3 -type f 2>/dev/null | head -1 | wc -l | tr -d ' ')

# ── stderr tail ───────────────────────────────────────────────────────────
if [[ -f "$STDERR_LOG" ]]; then
  recent_errors=$(tail -200 "$STDERR_LOG" | grep -ciE "^(error|traceback|exception|fatal)" || true)
  recent_errors=${recent_errors:-0}
else
  recent_errors=0
fi

# ── verdict ───────────────────────────────────────────────────────────────
wedged=0
if [[ "$status" != "completed" && "$status" != "failed" && "$final" == "absent" ]]; then
  if [[ "$events_age" -gt "$THRESH" && "$sdk_workers" -lt 1 && "$docker_execs" -lt 1 && "$recent_files" -lt 1 ]]; then
    wedged=1
  fi
fi

# ── output ────────────────────────────────────────────────────────────────
printf 'project        %s\n' "$PROJECT"
printf 'status         %s\n' "$status"
printf 'final_report   %s\n' "$final"
printf 'events         count=%s age=%ss last=%s\n' "$events_count" "$events_age" "$last_event"
printf 'liveness       sdk_workers=%s docker_execs=%s recent_code_files=%s\n' "$sdk_workers" "$docker_execs" "$recent_files"
printf 'stderr         recent_error_lines=%s\n' "$recent_errors"
if [[ "$wedged" -eq 1 ]]; then
  printf 'verdict        WEDGED (no events for %ss, no live workers, no docker, no recent files)\n' "$events_age"
  exit 2
fi
printf 'verdict        healthy (progressing or terminal)\n'
exit 0
