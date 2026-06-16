#!/usr/bin/env bash
#
# scripts/readiness.sh — launch / testing readiness for OpenResearch (ReproLab).
#
# Tiered checks: dependencies → static analysis → tests → boot smoke →
# real-run smoke → deployment surface. Each tier blocks the next.
#
# Usage:
#   scripts/readiness.sh                    # full pass
#   scripts/readiness.sh --tier 3           # run tiers 1..3 only
#   scripts/readiness.sh --skip-tests       # skip TIER 4 (pytest + vitest)
#   scripts/readiness.sh --skip-smoke       # skip TIER 5 + 6 (boot + run smoke)
#   scripts/readiness.sh --skip-deploy      # skip TIER 7 (docker build, etc.)
#   scripts/readiness.sh --json             # emit machine-readable summary
#   scripts/readiness.sh --quiet            # only print failures + summary
#   scripts/readiness.sh --fix              # auto-create venv / npm ci if missing
#
# Exit codes:
#   0   all checks passed (or only WARN)
#   1   at least one TIER 1-3 check failed (blocking)
#   2   tests failed (TIER 4)
#   3   boot smoke failed (TIER 5)
#   4   run smoke failed (TIER 6)
#   5   deployment-surface check failed (TIER 7)
#   10  invalid usage / missing prerequisite the script itself needs
#
# Conventions:
#   - All paths relative to repo root (the script cd's there).
#   - All commands run from repo root unless explicitly noted.
#   - Output: [ T<n> ] <check> ............ STATUS (took Xs)
#   - Background processes spawned by boot/run smoke get killed on EXIT.

set -Eeuo pipefail

# ────────────────────────────────────────────────────────────────────
# Resolve repo root + cd
# ────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ────────────────────────────────────────────────────────────────────
# CLI parsing
# ────────────────────────────────────────────────────────────────────

MAX_TIER=7
SKIP_TESTS=0
SKIP_SMOKE=0
SKIP_DEPLOY=0
JSON_OUT=0
QUIET=0
FIX=0
RUN_SMOKE_BUDGET_USD="${OPENRESEARCH_READINESS_RUN_BUDGET_USD:-0.50}"
RUN_SMOKE_WALL_S="${OPENRESEARCH_READINESS_RUN_WALL_S:-600}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) MAX_TIER="$2"; shift 2 ;;
    --tier=*) MAX_TIER="${1#*=}"; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --skip-deploy) SKIP_DEPLOY=1; shift ;;
    --json) JSON_OUT=1; QUIET=1; shift ;;
    --quiet) QUIET=1; shift ;;
    --fix) FIX=1; shift ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1"; exit 10 ;;
  esac
done

if ! [[ "${MAX_TIER}" =~ ^[1-7]$ ]]; then
  echo "--tier must be 1-7"; exit 10
fi

# ────────────────────────────────────────────────────────────────────
# Colors + logging
# ────────────────────────────────────────────────────────────────────

if [[ -t 1 && "${NO_COLOR:-}" != "1" ]]; then
  C_RESET=$'\e[0m'; C_BOLD=$'\e[1m'; C_DIM=$'\e[2m'
  C_GREEN=$'\e[32m'; C_RED=$'\e[31m'; C_YELLOW=$'\e[33m'; C_BLUE=$'\e[34m'
else
  C_RESET=""; C_BOLD=""; C_DIM=""; C_GREEN=""; C_RED=""; C_YELLOW=""; C_BLUE=""
fi

# Result accumulators (parallel arrays — bash doesn't have nested arrays).
RESULT_TIERS=()
RESULT_NAMES=()
RESULT_STATUS=()   # PASS | FAIL | SKIP | WARN
RESULT_SECS=()
RESULT_DETAIL=()

FAILED_BLOCKING=0
FAILED_TESTS=0
FAILED_SMOKE_BOOT=0
FAILED_SMOKE_RUN=0
FAILED_DEPLOY=0
WARNED=0
TOTAL_T0=$(date +%s)

log_tier_header() {
  local tier="$1" title="$2"
  [[ "${QUIET}" -eq 1 ]] && return 0
  echo
  echo "${C_BOLD}${C_BLUE}── TIER ${tier} ── ${title}${C_RESET}"
}

log_check_start() {
  [[ "${QUIET}" -eq 1 ]] && return 0
  printf "  ${C_DIM}…${C_RESET} %-55s " "$1"
}

log_check_result() {
  # args: tier, name, status, secs, detail
  local tier="$1" name="$2" status="$3" secs="$4" detail="${5:-}"
  RESULT_TIERS+=("${tier}")
  RESULT_NAMES+=("${name}")
  RESULT_STATUS+=("${status}")
  RESULT_SECS+=("${secs}")
  RESULT_DETAIL+=("${detail}")
  if [[ "${QUIET}" -eq 1 && "${status}" == "PASS" ]]; then
    return 0
  fi
  if [[ "${QUIET}" -eq 1 ]]; then
    printf "  [T%s] %-55s " "${tier}" "${name}"
  fi
  case "${status}" in
    PASS) printf "%s%s%s (%ss)\n" "${C_GREEN}" "PASS" "${C_RESET}" "${secs}" ;;
    FAIL) printf "%s%s%s (%ss)" "${C_RED}" "FAIL" "${C_RESET}" "${secs}"; [[ -n "${detail}" ]] && printf " — %s" "${detail}"; echo ;;
    SKIP) printf "%s%s%s\n" "${C_DIM}" "SKIP" "${C_RESET}" ;;
    WARN) printf "%s%s%s (%ss)" "${C_YELLOW}" "WARN" "${C_RESET}" "${secs}"; [[ -n "${detail}" ]] && printf " — %s" "${detail}"; echo ;;
  esac
}

# Run a check: $1=tier, $2=name, $3=command-as-string.
# Captures stderr on failure. Records secs.
run_check() {
  local tier="$1" name="$2" cmd="$3"
  log_check_start "${name}"
  local t0 t1 secs status detail rc out
  t0=$(date +%s)
  set +e
  out=$(eval "${cmd}" 2>&1); rc=$?
  set -e
  t1=$(date +%s); secs=$((t1 - t0))
  if [[ ${rc} -eq 0 ]]; then
    status=PASS
    detail=""
  else
    status=FAIL
    # Compact the failure reason to a single line.
    detail=$(echo "${out}" | tail -3 | tr '\n' ' ' | sed 's/  */ /g; s/^ *//; s/ *$//' | cut -c1-200)
  fi
  log_check_result "${tier}" "${name}" "${status}" "${secs}" "${detail}"
  return ${rc}
}

skip_check() {
  log_check_start "$2"
  log_check_result "$1" "$2" "SKIP" "0" ""
}

warn_check() {
  log_check_start "$2"
  log_check_result "$1" "$2" "WARN" "0" "${3:-}"
  WARNED=$((WARNED + 1))
}

# ────────────────────────────────────────────────────────────────────
# Background-process tracking (boot/run smoke)
# ────────────────────────────────────────────────────────────────────

BG_PIDS=()
cleanup_bg() {
  local rc=$?
  if [[ ${#BG_PIDS[@]} -gt 0 ]]; then
    for pid in "${BG_PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
        # Give graceful shutdown a moment, then SIGKILL if still alive.
        sleep 1
        kill -9 "${pid}" 2>/dev/null || true
      fi
    done
  fi
  exit "${rc}"
}
trap cleanup_bg EXIT INT TERM

# ────────────────────────────────────────────────────────────────────
# TIER 1 — Environment prerequisites (fast, must pass)
# ────────────────────────────────────────────────────────────────────

tier1_environment() {
  log_tier_header 1 "Environment prerequisites"

  run_check 1 "python3 binary present" \
    "command -v python3 >/dev/null"

  run_check 1 "python ≥ 3.14" "
    v=\$(python3 -c 'import sys; print(\"%d.%d\" % sys.version_info[:2])')
    awk -v v=\"\$v\" 'BEGIN { split(v, a, \".\"); if ((a[1]<3) || (a[1]==3 && a[2]<14)) exit 1 }'
  " || true   # warn-only: 3.12+ may work for most things

  if [[ ! -d ".venv" ]]; then
    if [[ ${FIX} -eq 1 ]]; then
      run_check 1 "create .venv (--fix)" \
        "python3 -m venv .venv"
    else
      warn_check 1 ".venv exists" ".venv missing — run with --fix or 'python3 -m venv .venv'"
    fi
  else
    run_check 1 ".venv exists" "test -x .venv/bin/python"
  fi

  run_check 1 "node binary present" "command -v node >/dev/null"
  run_check 1 "node ≥ 20.19 (≠21) or ≥ 22.12" "
    v=\$(node -v | sed 's/^v//')
    awk -v v=\"\$v\" 'BEGIN {
      split(v, a, \".\");
      maj=a[1]; min=a[2];
      ok = (maj==20 && min>=19) || maj==22 && min>=12;
      if (!ok) exit 1
    }'
  "

  run_check 1 "npm binary present" "command -v npm >/dev/null"
  run_check 1 "git binary present" "command -v git >/dev/null"

  # Optional: docker + gh (only WARN if missing)
  if command -v docker >/dev/null; then
    run_check 1 "docker available" "docker --version >/dev/null"
  else
    warn_check 1 "docker available" "docker not installed — TIER 7 docker checks will SKIP"
  fi
  if command -v gh >/dev/null; then
    run_check 1 "gh CLI available" "gh --version >/dev/null"
  else
    warn_check 1 "gh CLI available" "gh not installed — manual PR review only"
  fi

  if [[ -f ".env" ]]; then
    run_check 1 ".env present" "test -s .env"
  elif [[ -f ".env.example" ]]; then
    warn_check 1 ".env present" "missing — cp .env.example .env and fill in keys"
  else
    run_check 1 ".env or .env.example" "test -f .env -o -f .env.example"
  fi
}

# ────────────────────────────────────────────────────────────────────
# TIER 2 — Dependencies installed
# ────────────────────────────────────────────────────────────────────

tier2_dependencies() {
  log_tier_header 2 "Dependencies installed"

  if [[ -d ".venv" ]]; then
    if [[ ${FIX} -eq 1 ]]; then
      run_check 2 "backend requirements (--fix)" \
        ".venv/bin/pip install -q -r backend/requirements.txt && .venv/bin/pip install -q -r backend/requirements-dev.txt"
    fi
    run_check 2 "fastapi importable" \
      ".venv/bin/python -c 'import fastapi'"
    run_check 2 "uvicorn importable" \
      ".venv/bin/python -c 'import uvicorn'"
    run_check 2 "pydantic v2 importable" \
      ".venv/bin/python -c 'import pydantic; assert pydantic.VERSION.startswith(\"2.\"), pydantic.VERSION'"
    run_check 2 "claude-agent-sdk importable" \
      ".venv/bin/python -c 'import claude_agent_sdk' 2>&1"
    run_check 2 "rlm library importable" \
      ".venv/bin/python -c 'import rlm'"
    run_check 2 "pytest available" \
      ".venv/bin/python -m pytest --version >/dev/null"
  else
    skip_check 2 "(.venv missing — install deps skipped)"
  fi

  if [[ -d "frontend/node_modules" ]]; then
    run_check 2 "frontend/node_modules present" "test -d frontend/node_modules/next"
  else
    if [[ ${FIX} -eq 1 ]]; then
      run_check 2 "npm ci (--fix)" "cd frontend && npm ci --silent"
    else
      warn_check 2 "frontend/node_modules present" "missing — run 'cd frontend && npm ci' or pass --fix"
    fi
  fi
}

# ────────────────────────────────────────────────────────────────────
# TIER 3 — Static analysis (lint, types, factory boot)
# ────────────────────────────────────────────────────────────────────

tier3_static() {
  log_tier_header 3 "Static analysis"

  if [[ -d ".venv" ]]; then
    run_check 3 "backend factory imports cleanly" \
      ".venv/bin/python -c 'from backend.app import create_app; app = create_app(); assert app is not None'"
    run_check 3 "backend.cli importable" \
      ".venv/bin/python -c 'from backend import cli; assert cli._build_parser() is not None'"
    run_check 3 "CLI --help works (rlm + rdr modes)" "
      out=\$(.venv/bin/python -m backend.cli reproduce --help 2>&1)
      echo \"\$out\" | grep -q 'rlm' && echo \"\$out\" | grep -q 'rdr'
    "
    run_check 3 "no stale --mode sdk/offline in CLI help" "
      out=\$(.venv/bin/python -m backend.cli reproduce --help 2>&1)
      ! (echo \"\$out\" | grep -qE 'mode (sdk|offline)\\b')
    "
    run_check 3 "honesty cap constant present" "
      .venv/bin/python -c '
import importlib
m = importlib.import_module(\"backend.evals.paperbench.leaf_scorer\")
assert hasattr(m, \"DEGRADED_LEAF_CEILING\"), \"DEGRADED_LEAF_CEILING missing\"
assert m.DEGRADED_LEAF_CEILING == 0.35, f\"DEGRADED_LEAF_CEILING={m.DEGRADED_LEAF_CEILING}, expected 0.35\"
'"
  else
    skip_check 3 "(.venv missing — backend static checks skipped)"
  fi

  if [[ -d "frontend/node_modules" ]]; then
    run_check 3 "frontend eslint" \
      "cd frontend && npm run lint --silent 2>&1 | tail -50"
    run_check 3 "frontend tsc --noEmit" \
      "cd frontend && npx --no-install tsc --noEmit 2>&1 | tail -50"
  else
    skip_check 3 "(frontend/node_modules missing — frontend static checks skipped)"
  fi

  # Cleanup-spec acceptance: docs ≤ 14 .md files (slightly relaxed from spec
  # §2's 12 to allow active 2026-05-23 work).
  run_check 3 "docs/*.md count ≤ 14" "
    count=\$(find docs -name '*.md' -type f | wc -l | tr -d ' ')
    [[ \$count -le 14 ]] || { echo \"\$count > 14\"; exit 1; }
  "

  # Working tree should be clean if running pre-deploy.
  if [[ -n "${READINESS_REQUIRE_CLEAN:-}" ]]; then
    run_check 3 "git status clean (READINESS_REQUIRE_CLEAN=1)" \
      "test -z \"\$(git status --porcelain)\""
  fi
}

# ────────────────────────────────────────────────────────────────────
# TIER 4 — Test suites
# ────────────────────────────────────────────────────────────────────

tier4_tests() {
  log_tier_header 4 "Test suites"

  if [[ ${SKIP_TESTS} -eq 1 ]]; then
    skip_check 4 "(--skip-tests)"
    return 0
  fi

  if [[ -d ".venv" ]]; then
    # Run with -q to suppress per-test progress; tail captures the summary line.
    # -n auto needs pytest-xdist (installed via requirements-dev.txt).
    run_check 4 "pytest tests/ -n auto" "
      out=\$(.venv/bin/python -m pytest tests/ -n auto --tb=line -q 2>&1)
      tail=\$(echo \"\$out\" | tail -3)
      echo \"\$tail\"
      echo \"\$tail\" | grep -qE '[0-9]+ passed' && ! echo \"\$tail\" | grep -qE '[0-9]+ failed'
    "
  else
    skip_check 4 "(.venv missing)"
  fi

  if [[ -d "frontend/node_modules" ]]; then
    run_check 4 "frontend vitest" \
      "cd frontend && npm test --silent 2>&1 | tail -30"
  else
    skip_check 4 "(frontend/node_modules missing)"
  fi
}

# ────────────────────────────────────────────────────────────────────
# TIER 5 — Boot smoke (start backend + frontend, hit routes)
# ────────────────────────────────────────────────────────────────────

wait_for_url() {
  local url="$1" timeout="${2:-30}" t=0
  while (( t < timeout )); do
    if curl -fs -o /dev/null --max-time 2 "${url}"; then
      return 0
    fi
    sleep 1; t=$((t + 1))
  done
  return 1
}

tier5_boot_smoke() {
  log_tier_header 5 "Boot smoke (backend + frontend)"

  if [[ ${SKIP_SMOKE} -eq 1 ]]; then
    skip_check 5 "(--skip-smoke)"
    return 0
  fi
  if [[ ! -d ".venv" ]]; then
    skip_check 5 "(.venv missing)"
    return 0
  fi

  # Backend on :8001 (avoid colliding with a local dev backend on :8000).
  local backend_port=8001
  local backend_log="${REPO_ROOT}/.readiness-backend.log"
  rm -f "${backend_log}"
  log_check_start "backend boots on :${backend_port}"
  local t0 t1 secs
  t0=$(date +%s)
  .venv/bin/uvicorn backend.app:create_app --factory --port "${backend_port}" \
    >"${backend_log}" 2>&1 &
  local backend_pid=$!
  BG_PIDS+=("${backend_pid}")
  if wait_for_url "http://127.0.0.1:${backend_port}/docs" 30; then
    t1=$(date +%s); secs=$((t1 - t0))
    log_check_result 5 "backend boots on :${backend_port}" "PASS" "${secs}"
  else
    t1=$(date +%s); secs=$((t1 - t0))
    local last_lines
    last_lines=$(tail -3 "${backend_log}" 2>/dev/null | tr '\n' ' ' | cut -c1-180)
    log_check_result 5 "backend boots on :${backend_port}" "FAIL" "${secs}" "${last_lines}"
    return 1
  fi

  run_check 5 "GET /leaderboard returns 200" \
    "curl -fs -o /dev/null -w '%{http_code}' http://127.0.0.1:${backend_port}/leaderboard | grep -q '^200$'"
  run_check 5 "GET /runs returns 200 or 401 (gated)" "
    code=\$(curl -fs -o /dev/null -w '%{http_code}' http://127.0.0.1:${backend_port}/runs)
    [[ \"\$code\" == '200' || \"\$code\" == '401' || \"\$code\" == '403' ]]
  "

  # Kill backend before frontend (Next.js dev server is heavy).
  kill "${backend_pid}" 2>/dev/null || true
  wait "${backend_pid}" 2>/dev/null || true
  # Remove from BG_PIDS to avoid double-kill in cleanup.
  BG_PIDS=("${BG_PIDS[@]/${backend_pid}}")

  # Frontend boot smoke is expensive (Next.js dev compiles on first request).
  # Skip unless explicitly requested.
  if [[ -n "${READINESS_FRONTEND_SMOKE:-}" && -d "frontend/node_modules" ]]; then
    local fe_port=3001
    local fe_log="${REPO_ROOT}/.readiness-frontend.log"
    rm -f "${fe_log}"
    log_check_start "frontend boots on :${fe_port}"
    t0=$(date +%s)
    (cd frontend && OPENRESEARCH_BACKEND_URL="http://127.0.0.1:8001" \
      PORT="${fe_port}" npm run dev --silent) >"${fe_log}" 2>&1 &
    local fe_pid=$!
    BG_PIDS+=("${fe_pid}")
    if wait_for_url "http://127.0.0.1:${fe_port}/" 90; then
      t1=$(date +%s); secs=$((t1 - t0))
      log_check_result 5 "frontend boots on :${fe_port}" "PASS" "${secs}"
    else
      t1=$(date +%s); secs=$((t1 - t0))
      log_check_result 5 "frontend boots on :${fe_port}" "FAIL" "${secs}"
    fi
    kill "${fe_pid}" 2>/dev/null || true
  else
    skip_check 5 "(frontend smoke: set READINESS_FRONTEND_SMOKE=1)"
  fi
}

# ────────────────────────────────────────────────────────────────────
# TIER 6 — Run smoke (real --mode rlm run on small bundle)
# ────────────────────────────────────────────────────────────────────

tier6_run_smoke() {
  log_tier_header 6 "Run smoke (real CLI reproduction)"

  if [[ ${SKIP_SMOKE} -eq 1 ]]; then
    skip_check 6 "(--skip-smoke)"
    return 0
  fi
  if [[ -z "${READINESS_RUN_SMOKE:-}" ]]; then
    skip_check 6 "(opt-in: set READINESS_RUN_SMOKE=1)"
    return 0
  fi
  if [[ ! -d ".venv" ]]; then
    skip_check 6 "(.venv missing)"
    return 0
  fi

  # Use a temp runs dir so we can clean up without touching real runs.
  local tmp_runs
  tmp_runs=$(mktemp -d -t reprolab-readiness.XXXXXX)
  trap "rm -rf '${tmp_runs}'" RETURN

  log_check_start "ftrl --mode rlm (≤\$${RUN_SMOKE_BUDGET_USD}, ${RUN_SMOKE_WALL_S}s)"
  local t0 t1 secs rc out
  t0=$(date +%s)
  set +e
  out=$(.venv/bin/python -m backend.cli reproduce ftrl \
    --mode rlm \
    --max-usd "${RUN_SMOKE_BUDGET_USD}" \
    --max-wall-clock "${RUN_SMOKE_WALL_S}" \
    --sandbox local \
    --runs-root "${tmp_runs}" 2>&1); rc=$?
  set -e
  t1=$(date +%s); secs=$((t1 - t0))

  if [[ ${rc} -eq 0 ]]; then
    # Verify final_report.json shape
    if find "${tmp_runs}" -name final_report.json | head -1 | xargs -I{} \
       .venv/bin/python -c "
import json, sys
data = json.load(open('{}'))
assert 'mode' in data, 'missing mode'
assert 'models' in data, 'missing models'
assert 'rubric' in data, 'missing rubric'
sys.exit(0)
" 2>/dev/null; then
      log_check_result 6 "ftrl --mode rlm smoke" "PASS" "${secs}"
    else
      log_check_result 6 "ftrl --mode rlm smoke" "FAIL" "${secs}" "final_report.json missing required keys"
      return 1
    fi
  else
    local tail3
    tail3=$(echo "${out}" | tail -3 | tr '\n' ' ' | cut -c1-200)
    log_check_result 6 "ftrl --mode rlm smoke" "FAIL" "${secs}" "${tail3}"
    return 1
  fi
}

# ────────────────────────────────────────────────────────────────────
# TIER 7 — Deployment surface (Docker, env, secrets)
# ────────────────────────────────────────────────────────────────────

tier7_deploy() {
  log_tier_header 7 "Deployment surface"

  if [[ ${SKIP_DEPLOY} -eq 1 ]]; then
    skip_check 7 "(--skip-deploy)"
    return 0
  fi

  if command -v docker >/dev/null; then
    run_check 7 "Dockerfile parses" "docker build -f Dockerfile -t reprolab:readiness-check --no-cache --target=runtime . >/dev/null 2>&1 || docker build -f Dockerfile -t reprolab:readiness-check . >/dev/null 2>&1"
  else
    skip_check 7 "(docker not installed)"
  fi

  # Demo gate sanity: if OPENRESEARCH_DEMO_SECRET set in .env, hint at it.
  if [[ -f .env ]] && grep -qE '^OPENRESEARCH_DEMO_SECRET=.+' .env; then
    warn_check 7 "demo gate configured" "OPENRESEARCH_DEMO_SECRET set — clients must send X-Demo-Secret header on POST /runs"
  fi

  # RunPod creds (only checked, not used)
  if [[ -f .env ]] && grep -qE '^REPROLAB_RUNPOD_API_KEY=.+' .env; then
    run_check 7 "RunPod creds present in .env" "true"
  else
    warn_check 7 "RunPod creds present" "missing — --sandbox runpod runs will fail at preflight"
  fi

  # OAuth subscription (macOS Keychain) OR API key
  case "$(uname -s)" in
    Darwin)
      if security find-generic-password -s "Claude Code-credentials" >/dev/null 2>&1; then
        run_check 7 "Claude OAuth subscription (Keychain)" "true"
      elif [[ -f .env ]] && grep -qE '^ANTHROPIC_API_KEY=.+' .env; then
        run_check 7 "ANTHROPIC_API_KEY in .env" "true"
      else
        warn_check 7 "Claude credentials" "no Keychain OAuth + no ANTHROPIC_API_KEY — sub-agents will fail"
      fi
      ;;
    *)
      if [[ -f "${HOME}/.claude/.credentials.json" ]]; then
        run_check 7 "Claude OAuth subscription (~/.claude)" "true"
      elif [[ -f .env ]] && grep -qE '^ANTHROPIC_API_KEY=.+' .env; then
        run_check 7 "ANTHROPIC_API_KEY in .env" "true"
      else
        warn_check 7 "Claude credentials" "no OAuth file + no ANTHROPIC_API_KEY — sub-agents will fail"
      fi
      ;;
  esac
}

# ────────────────────────────────────────────────────────────────────
# Summary + exit
# ────────────────────────────────────────────────────────────────────

emit_summary() {
  local total_secs=$(( $(date +%s) - TOTAL_T0 ))
  local n=${#RESULT_STATUS[@]}
  local pass=0 fail=0 skip=0 warn=0
  for s in "${RESULT_STATUS[@]}"; do
    case "${s}" in
      PASS) pass=$((pass + 1));;
      FAIL) fail=$((fail + 1));;
      SKIP) skip=$((skip + 1));;
      WARN) warn=$((warn + 1));;
    esac
  done

  if [[ ${JSON_OUT} -eq 1 ]]; then
    printf '{\n'
    printf '  "total_seconds": %d,\n' "${total_secs}"
    printf '  "totals": {"pass": %d, "fail": %d, "skip": %d, "warn": %d, "n": %d},\n' \
      "${pass}" "${fail}" "${skip}" "${warn}" "${n}"
    printf '  "blocking_failures": %d,\n' "${FAILED_BLOCKING}"
    printf '  "test_failures": %d,\n' "${FAILED_TESTS}"
    printf '  "boot_smoke_failures": %d,\n' "${FAILED_SMOKE_BOOT}"
    printf '  "run_smoke_failures": %d,\n' "${FAILED_SMOKE_RUN}"
    printf '  "deploy_failures": %d,\n' "${FAILED_DEPLOY}"
    printf '  "checks": [\n'
    for i in "${!RESULT_NAMES[@]}"; do
      local sep=","; [[ $((i + 1)) -eq ${n} ]] && sep=""
      printf '    {"tier": %s, "name": "%s", "status": "%s", "seconds": %s, "detail": "%s"}%s\n' \
        "${RESULT_TIERS[$i]}" "${RESULT_NAMES[$i]//\"/\\\"}" "${RESULT_STATUS[$i]}" \
        "${RESULT_SECS[$i]}" "${RESULT_DETAIL[$i]//\"/\\\"}" "${sep}"
    done
    printf '  ]\n'
    printf '}\n'
    return 0
  fi

  echo
  echo "${C_BOLD}── Summary ──${C_RESET}"
  printf "  PASS:%s%d%s  FAIL:%s%d%s  WARN:%s%d%s  SKIP:%s%d%s   total: %ds\n" \
    "${C_GREEN}" "${pass}" "${C_RESET}" \
    "${C_RED}"   "${fail}" "${C_RESET}" \
    "${C_YELLOW}" "${warn}" "${C_RESET}" \
    "${C_DIM}"   "${skip}" "${C_RESET}" \
    "${total_secs}"

  if [[ ${fail} -gt 0 ]]; then
    echo
    echo "${C_RED}Failed checks:${C_RESET}"
    for i in "${!RESULT_STATUS[@]}"; do
      if [[ "${RESULT_STATUS[$i]}" == "FAIL" ]]; then
        printf "  [T%s] %s\n    %s%s%s\n" \
          "${RESULT_TIERS[$i]}" "${RESULT_NAMES[$i]}" \
          "${C_DIM}" "${RESULT_DETAIL[$i]}" "${C_RESET}"
      fi
    done
  fi
}

# ────────────────────────────────────────────────────────────────────
# Run tiers (respect --tier ceiling, --skip-*)
# ────────────────────────────────────────────────────────────────────

# Each tier's failures contribute to a specific exit code.
# Tier 1-3 failures all map to exit 1 (blocking).
# Tier 4 → 2, 5 → 3, 6 → 4, 7 → 5.

tier1_environment || FAILED_BLOCKING=$((FAILED_BLOCKING + 1))
[[ ${MAX_TIER} -ge 2 ]] && { tier2_dependencies || FAILED_BLOCKING=$((FAILED_BLOCKING + 1)); }
[[ ${MAX_TIER} -ge 3 ]] && { tier3_static || FAILED_BLOCKING=$((FAILED_BLOCKING + 1)); }
[[ ${MAX_TIER} -ge 4 ]] && { tier4_tests || FAILED_TESTS=$((FAILED_TESTS + 1)); }
[[ ${MAX_TIER} -ge 5 ]] && { tier5_boot_smoke || FAILED_SMOKE_BOOT=$((FAILED_SMOKE_BOOT + 1)); }
[[ ${MAX_TIER} -ge 6 ]] && { tier6_run_smoke || FAILED_SMOKE_RUN=$((FAILED_SMOKE_RUN + 1)); }
[[ ${MAX_TIER} -ge 7 ]] && { tier7_deploy || FAILED_DEPLOY=$((FAILED_DEPLOY + 1)); }

# Recompute blocking from the result arrays (run_check returns rc but we may
# have additional fails from sub-checks within a tier).
real_fails_in_tier() {
  local tier="$1" n=0 i
  for i in "${!RESULT_TIERS[@]}"; do
    if [[ "${RESULT_TIERS[$i]}" == "${tier}" && "${RESULT_STATUS[$i]}" == "FAIL" ]]; then
      n=$((n + 1))
    fi
  done
  echo "${n}"
}

FAILED_BLOCKING=$(( $(real_fails_in_tier 1) + $(real_fails_in_tier 2) + $(real_fails_in_tier 3) ))
FAILED_TESTS=$(real_fails_in_tier 4)
FAILED_SMOKE_BOOT=$(real_fails_in_tier 5)
FAILED_SMOKE_RUN=$(real_fails_in_tier 6)
FAILED_DEPLOY=$(real_fails_in_tier 7)

emit_summary

if   [[ ${FAILED_BLOCKING}  -gt 0 ]]; then exit 1
elif [[ ${FAILED_TESTS}      -gt 0 ]]; then exit 2
elif [[ ${FAILED_SMOKE_BOOT} -gt 0 ]]; then exit 3
elif [[ ${FAILED_SMOKE_RUN}  -gt 0 ]]; then exit 4
elif [[ ${FAILED_DEPLOY}     -gt 0 ]]; then exit 5
else exit 0
fi
