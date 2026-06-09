#!/usr/bin/env bash
# kill_and_restart.sh — the SDAR retry sprint kill+restart loop, scripted.
#
# Usage:
#   scripts/loops/kill_and_restart.sh <project_id> <next_attempt_n> <pdf_path>
#
# Steps:
#   1. SIGKILL the named project's CLI subprocess (escalating from TERM).
#   2. Patch runs/<project>/demo_status.json to status=killed
#      (BUG-NEW-041 workaround — the CLI subprocess doesn't write its own
#      status-on-signal yet; without this the UI shows a phantom 'running'
#      state forever and the user thinks the run is still alive).
#   3. Stage a fresh PDF copy under /tmp so the orchestrator hashes it
#      to a new project_id.
#   4. Relaunch via nohup + env -u to defeat shell-shadow of credentials
#      (the 2026-05-28 OPENAI_API_KEY shell-wins pitfall — see CLAUDE.md
#      "Shell vs .env precedence").
#
# Output:
#   prints the new attempt's PID + new project_id (once dashboard_events.jsonl
#   has been written, ~10s after launch). Caller can then re-arm Monitor
#   loops against the new project_id.
#
# Caps:
#   The 5-attempt cap is enforced by the caller (the autonomous loop in
#   the active Claude Code session). This script does NOT cap on its own —
#   it's a single transition, not a sweep.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <project_id> <next_attempt_n> <pdf_path>" >&2
  exit 2
fi

PROJECT_ID="$1"
NEXT_N="$2"
SRC_PDF="$3"

RUN_DIR="runs/${PROJECT_ID}"
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: no such run directory: ${RUN_DIR}" >&2
  exit 3
fi

# --- 1. SIGKILL the CLI subprocess ----------------------------------------
echo "[1/4] Killing CLI subprocess for ${PROJECT_ID}…"
PIDS="$(pgrep -f "${PROJECT_ID}" || true)"
if [[ -n "${PIDS}" ]]; then
  echo "  sending SIGTERM to: ${PIDS}"
  kill ${PIDS} 2>/dev/null || true
  sleep 5
  PIDS_LEFT="$(pgrep -f "${PROJECT_ID}" || true)"
  if [[ -n "${PIDS_LEFT}" ]]; then
    echo "  still alive — escalating to SIGKILL: ${PIDS_LEFT}"
    kill -9 ${PIDS_LEFT} 2>/dev/null || true
  fi
else
  echo "  no PIDs matched — assuming already dead"
fi

# --- 2. Patch demo_status.json to status=killed ---------------------------
STATUS_FILE="${RUN_DIR}/demo_status.json"
if [[ -f "${STATUS_FILE}" ]]; then
  echo "[2/4] Marking demo_status.json status=killed (BUG-NEW-041 workaround)…"
  python3 - "${STATUS_FILE}" <<'PY'
import json, sys, datetime
p = sys.argv[1]
d = json.loads(open(p).read())
ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
d['status'] = 'killed'
d['killedAt'] = ts
d['killReason'] = 'kill_and_restart.sh — sprint retry'
d['updatedAt'] = ts
open(p, 'w').write(json.dumps(d, indent=2))
print(f"  status updated at {ts}")
PY
else
  echo "[2/4] No demo_status.json — skipping status patch"
fi

# --- 3. Stage a fresh PDF copy --------------------------------------------
if [[ ! -f "${SRC_PDF}" ]]; then
  echo "ERROR: source PDF does not exist: ${SRC_PDF}" >&2
  exit 4
fi
TS="$(date -u +%Y%m%d_%H%M%S)"
BASENAME="$(basename "${SRC_PDF}" .pdf)"
# Strip any existing _attemptN_ suffix
BASENAME="${BASENAME%_attempt*}"
NEW_PDF="/tmp/${BASENAME}_attempt${NEXT_N}_${TS}.pdf"
echo "[3/4] Staging fresh PDF copy: ${NEW_PDF}"
cp "${SRC_PDF}" "${NEW_PDF}"

# --- 4. Relaunch CLI ------------------------------------------------------
LOG_FILE="/tmp/sdar-attempt${NEXT_N}.log"
echo "[4/4] Launching attempt ${NEXT_N} → log: ${LOG_FILE}"
nohup env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  ./.venv/bin/python -m backend.cli reproduce \
  "${NEW_PDF}" \
  --mode rlm --model claude-oauth --sandbox runpod --provider anthropic \
  > "${LOG_FILE}" 2>&1 &
NEW_PID=$!
disown
echo "  PID=${NEW_PID}"

# Wait for the new run dir to be written so we can report its project_id
echo "  waiting up to 30s for new run dir…"
for i in $(seq 1 30); do
  sleep 1
  NEW_RUN="$(ls -t runs/ 2>/dev/null | grep -v "^${PROJECT_ID}$" | head -1)"
  if [[ -n "${NEW_RUN}" && -f "runs/${NEW_RUN}/demo_status.json" ]]; then
    echo "  new project_id: ${NEW_RUN}"
    break
  fi
done

echo "done."
