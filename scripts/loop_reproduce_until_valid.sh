#!/usr/bin/env bash
# Rerun the All-CNN reproduction until it produces a VALID scored result.
#
# Each attempt is self-recovering now: the lifecycle-drive backstop rescues a
# degenerate claude-oauth root (commit 141c29e5) and the FM-001 idle timeout
# fails a hung sub-agent fast (~20 min) instead of wedging forever. So a failed
# attempt terminates in bounded time and the loop reruns. Stops on the first
# valid result, on the attempt cap, or when the RunPod balance floor is hit.
#
# Status/progress: appended to runs/_loop/allcnn_loop.log (tail -f to watch).
set -u
cd "$(dirname "$0")/.." || exit 1

PAPER="papers/allcnn.pdf"
MAX_ATTEMPTS="${LOOP_MAX_ATTEMPTS:-10}"
BALANCE_FLOOR="${LOOP_BALANCE_FLOOR:-3.0}"   # stop if RunPod balance drops below this
GAP_S="${LOOP_GAP_S:-30}"
LOGDIR="runs/_loop"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/allcnn_loop.log"
STATUS="$LOGDIR/status.json"

log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# Validity check: latest run's final_report has a real numeric score and did not
# degenerate/fail. Prints "VALID <project_id> <score>" or "INVALID".
check_valid() {
  .venv/bin/python - <<'PY'
import json, glob, os
runs = sorted(glob.glob("runs/prj_*/final_report.json"), key=os.path.getmtime, reverse=True)
if not runs:
    print("INVALID no-report"); raise SystemExit
rep = runs[0]
pid = rep.split("/")[1]
try:
    d = json.load(open(rep))
except Exception as e:
    print(f"INVALID unreadable:{pid}"); raise SystemExit
verdict = str(d.get("verdict") or (d.get("reproducibility") or {}).get("verdict") or "")
stop = str(d.get("stop_reason") or "")
rub = d.get("rubric") or {}
score = rub.get("overall_score")
if score is None:
    score = d.get("rubric_score")
degenerate = "degenerate" in stop.lower() or "root_degenerate" in stop.lower()
# Valid = a real numeric score AND not a degenerate/scoreless finish.
if isinstance(score, (int, float)) and score > 0 and not degenerate:
    print(f"VALID {pid} {score}")
else:
    print(f"INVALID {pid} score={score} verdict={verdict} stop={stop or '-'}")
PY
}

runpod_balance() {
  runpodctl user 2>/dev/null | .venv/bin/python -c "import sys,json;print(json.load(sys.stdin).get('clientBalance',0))" 2>/dev/null || echo 0
}

log "=== loop start: max_attempts=$MAX_ATTEMPTS balance_floor=\$$BALANCE_FLOOR ==="
for i in $(seq 1 "$MAX_ATTEMPTS"); do
  bal=$(runpod_balance)
  log "attempt $i/$MAX_ATTEMPTS — RunPod balance \$$bal"
  # Balance floor guard (float compare via python)
  if .venv/bin/python -c "import sys;sys.exit(0 if float('$bal') < float('$BALANCE_FLOOR') else 1)"; then
    log "STOP: balance \$$bal below floor \$$BALANCE_FLOOR"
    echo "{\"state\":\"stopped_balance\",\"attempt\":$i,\"balance\":$bal}" > "$STATUS"
    exit 3
  fi

  log "attempt $i: launching reproduction…"
  env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY OPENRESEARCH_MIN_DISK_GB=0 \
    .venv/bin/python -m backend.cli reproduce "$PAPER" \
    --sandbox runpod --model claude-oauth \
    --max-usd 6 --max-pod-seconds 5400 --max-wall-clock 7200 \
    >> "$LOG" 2>&1
  rc=$?
  log "attempt $i: reproduction exited rc=$rc"

  res=$(check_valid)
  log "attempt $i: validity -> $res"
  if [[ "$res" == VALID* ]]; then
    log "=== SUCCESS on attempt $i: $res ==="
    echo "{\"state\":\"valid\",\"attempt\":$i,\"result\":\"$res\"}" > "$STATUS"
    exit 0
  fi
  echo "{\"state\":\"retrying\",\"attempt\":$i,\"last\":\"$res\"}" > "$STATUS"
  sleep "$GAP_S"
done

log "=== EXHAUSTED $MAX_ATTEMPTS attempts without a valid result ==="
echo "{\"state\":\"exhausted\",\"attempts\":$MAX_ATTEMPTS}" > "$STATUS"
exit 4
