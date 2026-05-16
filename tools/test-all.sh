#!/usr/bin/env bash
# Run every test suite in the repo. One command, copy-paste-safe.
#
# Usage:
#   bash tools/test-all.sh            # backend + frontend
#   bash tools/test-all.sh --backend  # backend only
#   bash tools/test-all.sh --frontend # frontend only
#   bash tools/test-all.sh --paper    # also run the end-to-end paper test
#                                     # (needs OPENAI_API_KEY for the real-LLM pass)
#
# Exit code: 0 = all green, non-zero = something failed.

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

run_backend=1
run_frontend=1
run_paper=0
for arg in "$@"; do
  case "$arg" in
    --backend)  run_frontend=0 ;;
    --frontend) run_backend=0 ;;
    --paper)    run_paper=1 ;;
    --no-frontend) run_frontend=0 ;;
    --no-backend)  run_backend=0 ;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0
      ;;
  esac
done

hr() { printf '\n=== %s ===\n' "$1"; }

if [[ $run_backend -eq 1 ]]; then
  hr "Backend: recursive RLM + topology + live-runs"
  if [[ ! -x "$PY" ]]; then
    echo "ERROR: $PY not found. Run: python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt"
    exit 2
  fi
  "$PY" -m pytest \
    "$ROOT/tests/test_rlm_query_recursive.py" \
    "$ROOT/tests/test_pipeline_topology_api.py" \
    "$ROOT/tests/test_live_runs_listing.py" \
    -v
fi

if [[ $run_frontend -eq 1 ]]; then
  hr "Frontend: TypeScript typecheck"
  (cd "$ROOT/frontend" && npx tsc --noEmit)

  hr "Frontend: vitest"
  (cd "$ROOT/frontend" && npm run test -- --run)
fi

if [[ $run_paper -eq 1 ]]; then
  hr "End-to-end: recursive RLM on actual arXiv paper"
  PDF=/tmp/rlm-paper.pdf
  if [[ ! -f "$PDF" ]]; then
    echo "Downloading paper to $PDF ..."
    curl -sL -o "$PDF" https://arxiv.org/pdf/2512.24601
  fi
  "$PY" "$ROOT/tools/test-rlm-on-paper.py" "$PDF"
fi

hr "ALL GREEN"
