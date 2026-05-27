#!/usr/bin/env bash
# Background data collector for the sanity-check observation cycle.
# Every 120s, snapshots: dashboard_events count, primitive_call counts,
# aclose retries, demo_status, RunPod pod count, cost ledger.
# Output: findings/sanity-observations-YYYYMMDD.jsonl (append-only)
set -e
cd "$(dirname "$0")/.."

SANITY_PID_FILE=runs/sanity-check-20260527/sanity.pid
PROJECT_DIR=runs/prj_ac41983c934a3432
OUT=findings/sanity-observations-20260527.jsonl

while true; do
  TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  ALIVE="exited"
  ETIME=""
  RSS=""
  if [ -f "$SANITY_PID_FILE" ]; then
    SP=$(cat "$SANITY_PID_FILE")
    if ps -p "$SP" > /dev/null 2>&1; then
      ALIVE="alive"
      ETIME=$(ps -p "$SP" -o etime= | tr -d ' ')
      RSS=$(ps -p "$SP" -o rss= | tr -d ' ')
    fi
  fi

  EVENT_COUNT=0
  PRIM_OK=0
  ACLOSE_COUNT=0
  POD_COUNT=0
  STATUS="?"
  KIND="?"
  ERROR=""

  if [ -f "$PROJECT_DIR/dashboard_events.jsonl" ]; then
    EVENT_COUNT=$(wc -l < "$PROJECT_DIR/dashboard_events.jsonl" | tr -d ' ')
    PRIM_OK=$(grep -c '"primitive_call"' "$PROJECT_DIR/dashboard_events.jsonl" 2>/dev/null || echo 0)
  fi
  if [ -f runs/sanity-check-20260527/sanity.stderr.log ]; then
    ACLOSE_COUNT=$(grep -c 'aclose(): asynchronous' runs/sanity-check-20260527/sanity.stderr.log 2>/dev/null || echo 0)
  fi
  if [ -f "$PROJECT_DIR/demo_status.json" ]; then
    STATUS=$(python3 -c "import json;d=json.load(open('$PROJECT_DIR/demo_status.json'));print(d.get('status','?'))" 2>/dev/null || echo "?")
    KIND=$(python3 -c "import json;d=json.load(open('$PROJECT_DIR/demo_status.json'));print(d.get('run_state',{}).get('kind','?'))" 2>/dev/null || echo "?")
    ERROR=$(python3 -c "import json;d=json.load(open('$PROJECT_DIR/demo_status.json'));print((d.get('error') or '')[:120])" 2>/dev/null || echo "")
  fi
  # RunPod pod count
  POD_COUNT=$(curl -s --max-time 5 -H "Authorization: Bearer $(grep '^REPROLAB_RUNPOD_API_KEY=' .env | cut -d= -f2)" "https://rest.runpod.io/v1/pods" 2>/dev/null | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")

  python3 -c "
import json
print(json.dumps({
    'ts': '$TS',
    'alive': '$ALIVE',
    'etime': '$ETIME',
    'rss_kb': '$RSS',
    'event_count': $EVENT_COUNT,
    'primitive_ok': $PRIM_OK,
    'aclose_retries': $ACLOSE_COUNT,
    'pods_active': '$POD_COUNT',
    'status': '$STATUS',
    'kind': '$KIND',
    'error': '$ERROR',
}))
" >> "$OUT"

  if [ "$ALIVE" = "exited" ]; then
    echo "$(date -u +%H:%M:%S) collector: process exited, stopping" >> "${OUT}.collector.log"
    break
  fi
  sleep 120
done
