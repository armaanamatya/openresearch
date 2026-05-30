#!/usr/bin/env bash
# parallel_runpod.sh — launch N arxiv-paper reproductions in parallel on RunPod.
#
# Built 2026-05-29 for the "SDAR + another paper concurrent" workflow. Unlike
# scripts/batch_reproduce.py (LOCAL sandbox + local GPU leasing), this script
# uses --sandbox runpod so each run gets its own remote pod. No local GPU
# coordination needed; the only shared state is the API rate-limit budget on
# Claude OAuth + OpenAI.
#
# Usage:
#   scripts/parallel_runpod.sh 2605.15155 2403.12345
#   scripts/parallel_runpod.sh --max-wall-clock 7200 2605.15155 2403.12345
#
# Each paper gets:
#   * Its own project_id (derived from arxiv id + timestamp)
#   * Its own log file under logs/parallel_<TS>/<paper>.log
#   * Its own runs/<project_id>/ directory (default backend behavior)
#
# Env overrides honored from your shell or .env:
#   REPROLAB_RLM_ROOT_MODEL (default: claude-oauth)
#   REPROLAB_RUNPOD_CLOUD_TYPE (default: COMMUNITY)
#
# Shell rule: this script `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY` per the
# CLAUDE.md "shell vs .env precedence" guidance — stale shell exports of those
# would override .env. The reproduce process re-reads .env via pydantic-Settings.

set -euo pipefail

# Defaults — overridable via flags BEFORE positional arxiv args.
MODEL="claude-oauth"
SANDBOX="runpod"
MAX_WALL_CLOCK="5400"
MAX_USD="8"
EXTRA_ARGS=""
PYTHON_BIN=".venv/bin/python"

usage() {
  cat <<EOF
Usage: $0 [--model MODEL] [--sandbox SANDBOX] [--max-wall-clock SEC]
          [--max-usd USD] [--python PATH] [--extra "FLAGS"]
          paper1 [paper2 ...]

Defaults:
  --model           $MODEL
  --sandbox         $SANDBOX
  --max-wall-clock  $MAX_WALL_CLOCK   (per paper)
  --max-usd         $MAX_USD          (per paper)
  --python          $PYTHON_BIN

Examples:
  $0 2605.15155                                # one paper
  $0 2605.15155 2403.12345                     # two papers in parallel
  $0 --max-wall-clock 7200 --max-usd 10 2605.15155 2403.12345
  $0 --extra "--paper-hint 2605.15155" 2605.15155
EOF
  exit 1
}

# Parse flags. Stops at first non-flag (paper id).
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --sandbox) SANDBOX="$2"; shift 2 ;;
    --max-wall-clock) MAX_WALL_CLOCK="$2"; shift 2 ;;
    --max-usd) MAX_USD="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --extra) EXTRA_ARGS="$2"; shift 2 ;;
    -h|--help) usage ;;
    --*) echo "unknown flag: $1" >&2; usage ;;
    *) break ;;
  esac
done

[[ $# -lt 1 ]] && usage

PAPERS=("$@")
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/parallel_${TS}"
mkdir -p "$LOG_DIR"

# Repo-root check — refuse to run from anywhere else (project_id collisions).
[[ -d "backend" && -f "backend/cli.py" ]] || {
  echo "Error: run from the openresearch repo root (no backend/ found)." >&2
  exit 2
}

[[ -x "$PYTHON_BIN" ]] || {
  echo "Error: $PYTHON_BIN not executable. Use --python to override." >&2
  exit 3
}

echo "════════════════════════════════════════════════════════════════"
echo "  Parallel RunPod reproduction — ${#PAPERS[@]} paper(s)"
echo "════════════════════════════════════════════════════════════════"
echo "  model:           $MODEL"
echo "  sandbox:         $SANDBOX"
echo "  max-wall-clock:  ${MAX_WALL_CLOCK}s per paper"
echo "  max-usd:         \$${MAX_USD} per paper"
echo "  log dir:         $LOG_DIR"
echo "  extra:           ${EXTRA_ARGS:-(none)}"
echo

PIDS=()
SAFE_NAMES=()

for paper in "${PAPERS[@]}"; do
  safe=$(echo "$paper" | tr -c 'A-Za-z0-9.-' '_')
  log="$LOG_DIR/${safe}.log"
  SAFE_NAMES+=("$safe")

  # `env -u` defeats stale shell exports per CLAUDE.md precedence rule.
  # `nohup` keeps it alive if this script is interrupted (signals propagate
  # to children via trap below, this is just a safety net).
  # Output redirected so each paper's stdout/stderr lands in its own log.
  # shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
  nohup env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
    "$PYTHON_BIN" -m backend.cli reproduce "$paper" \
      --model "$MODEL" \
      --sandbox "$SANDBOX" \
      --max-wall-clock "$MAX_WALL_CLOCK" \
      --max-usd "$MAX_USD" \
      $EXTRA_ARGS \
      > "$log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  echo "  launched: $paper  pid=$pid  log=$log"
done
echo
echo "All ${#PAPERS[@]} runs spawned. Following the first 5s of each log..."
echo

# Brief preview so the operator can spot immediate-fail conditions
# (bad .env, missing creds, fatal import error) without tailing.
sleep 5
for i in "${!PAPERS[@]}"; do
  paper="${PAPERS[$i]}"
  log="$LOG_DIR/${SAFE_NAMES[$i]}.log"
  echo "─── ${paper} (first 12 lines) ─────────────────────────────"
  head -12 "$log" 2>/dev/null || echo "  (no output yet)"
  echo
done

cat <<EOF
════════════════════════════════════════════════════════════════
  Runs are detached. To monitor / control:
    tail -F $LOG_DIR/*.log
    pgrep -fl "backend.cli reproduce"
    .venv/bin/python -m backend.cli list-runs       # all known projects
    open http://127.0.0.1:3000/lab                  # UI (if backend + frontend up)

  To gracefully stop ALL runs from this batch:
    kill -TERM ${PIDS[*]}    # BUG-NEW-041 handler flips demo_status=killed

  To force-kill (last resort, leaves demo_status stale):
    kill -KILL ${PIDS[*]}
════════════════════════════════════════════════════════════════
EOF

# Don't wait — script returns immediately so the terminal is free.
# Each child writes its own log + runs/<project_id>/ artifacts.
exit 0
