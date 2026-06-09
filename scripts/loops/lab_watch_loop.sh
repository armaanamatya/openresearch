#!/usr/bin/env bash
# lab_watch_loop.sh — Monitoring Loop 4 (Playwright UI watch).
#
# Polls the lab UI every 5 min via the lab-watch.spec.ts Playwright spec.
# Each cycle:
#   - loads /lab?projectId=<LAB_PROJECT_ID>  (if set)
#   - loads /leaderboard
#   - captures full-page screenshots into LAB_WATCH_SCREENSHOT_DIR
#   - asserts no console errors and no React hydration-mismatch text
#
# Exit conditions:
#   - SIGINT / SIGTERM stops the loop (handled by `set -e` + trap)
#   - max LAB_WATCH_MAX_CYCLES iterations completed (0 = unlimited; default)
#
# Usage:
#   LAB_PROJECT_ID=prj_xxx \
#   LAB_BASE_URL=http://localhost:3001 \
#     scripts/loops/lab_watch_loop.sh
#
# Logs each cycle's stdout/stderr to /tmp/lab-watch-loop.log so the outer
# Claude Code Monitor can tail it. Screenshots accumulate under
# /tmp/playwright-ui-loop/ for visual diff.
#
# Requires:
#   - The dev/prod Next server already running and reachable at LAB_BASE_URL.
#   - The frontend Playwright deps installed (`cd frontend && npm ci` once).

set -euo pipefail

INTERVAL_S="${LAB_WATCH_INTERVAL_S:-300}"
MAX_CYCLES="${LAB_WATCH_MAX_CYCLES:-0}"
LOG_FILE="${LAB_WATCH_LOG_FILE:-/tmp/lab-watch-loop.log}"

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
FRONTEND_DIR="${ROOT_DIR}/frontend"

if [[ ! -d "${FRONTEND_DIR}" ]]; then
  echo "ERROR: frontend dir not found at ${FRONTEND_DIR}" >&2
  exit 2
fi

trap 'echo "[lab-watch-loop] stopping (signal)"; exit 0' INT TERM HUP

CYCLE=0
while true; do
  CYCLE=$((CYCLE + 1))
  TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo "----- ${TS} cycle=${CYCLE} project=${LAB_PROJECT_ID:-<none>} -----"
    cd "${FRONTEND_DIR}"
    # The spec auto-skips the /lab portion when LAB_PROJECT_ID is empty.
    # webServer in playwright.config.ts only fires when LAB_BASE_URL is unset,
    # so honor whatever the operator pointed us at without booting our own.
    npx playwright test lab-watch.spec.ts --reporter=list 2>&1 || {
      echo "[lab-watch-loop] cycle ${CYCLE} FAILED — see playwright report"
    }
    echo "----- end cycle=${CYCLE} -----"
  } >> "${LOG_FILE}" 2>&1

  if [[ "${MAX_CYCLES}" -gt 0 && "${CYCLE}" -ge "${MAX_CYCLES}" ]]; then
    echo "[lab-watch-loop] reached MAX_CYCLES=${MAX_CYCLES}; stopping"
    break
  fi

  sleep "${INTERVAL_S}"
done
