#!/usr/bin/env bash
# Unified dev launcher.
#
# Creates logs/<TS>[-label]/ for the launch. Layout:
#
#   logs/<TS>/
#     meta.json          # launch metadata (started/ended, pids, exit codes)
#     manifest.json      # written on cleanup: every file's size + sha256
#     server/
#       backend.log      # uvicorn combined stdout+stderr
#       frontend.log     # Next.js dev combined stdout+stderr
#     prj_<id>/...       # per-pipeline-run workspaces (unchanged contract)
#
# Pipeline runs land under REPROLAB_RUNS_ROOT, which we point at logs/<TS>/ —
# so every prj_* directory the pipeline creates sits alongside server/.
#
# Stop with Ctrl-C (or TaskStop from the agent harness). meta.json is updated
# with ended_at, exit codes, and ended_reason on the way out.
#
# Flags (all optional, all additive):
#   --no-frontend    skip Next.js dev (backend + pipeline only)
#   --no-backend     skip uvicorn (frontend + pipeline only)
#   --label NAME     append "-NAME" to the timestamp dir for findability
#   --keep N         after launch, prune logs/ to N most-recent dirs
#
# See docs/design/unified-logging-launcher.md for the full design.
set -euo pipefail

cd "$(dirname "$0")/.."

WANT_FRONTEND=1
WANT_BACKEND=1
LABEL=""
KEEP=""

while [ $# -gt 0 ]; do
    case "$1" in
        --no-frontend) WANT_FRONTEND=0; shift ;;
        --no-backend)  WANT_BACKEND=0;  shift ;;
        --label)       LABEL="$2";      shift 2 ;;
        --keep)        KEEP="$2";       shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            echo "[dev.sh] unknown flag: $1" >&2
            exit 2
            ;;
    esac
done

if [ "$WANT_BACKEND" = "0" ] && [ "$WANT_FRONTEND" = "0" ]; then
    echo "[dev.sh] --no-backend and --no-frontend together leave nothing to launch" >&2
    exit 2
fi

TS="$(date +%Y%m%d-%H%M%S)"
if [ -n "$LABEL" ]; then
    # Strip anything that would make path handling awkward.
    SAFE_LABEL="$(echo "$LABEL" | tr -c 'A-Za-z0-9_-' '-' | sed 's/--*/-/g; s/^-//; s/-$//')"
    if [ -n "$SAFE_LABEL" ]; then
        TS="${TS}-${SAFE_LABEL}"
    fi
fi
LOG_DIR="logs/$TS"
SERVER_DIR="$LOG_DIR/server"
mkdir -p "$SERVER_DIR"

export REPROLAB_RUNS_ROOT="$PWD/$LOG_DIR"

# Force UTF-8 for all Python processes started under this launcher (uvicorn +
# any subprocess it spawns, e.g. the pipeline CLI). On Windows the default
# locale is cp1252 — Path.write_text / open() without explicit encoding=
# crashes on Greek letters, em-dashes, math symbols, etc. that routinely
# appear in audit JSON dumps. PYTHONUTF8=1 is Python 3.7+'s opt-in UTF-8
# mode and inherits across subprocess.Popen by default.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
SANDBOX="${REPROLAB_DEFAULT_SANDBOX:-docker}"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ENDED_REASON="exit"

BACKEND_PID=""
FRONTEND_PID=""
BACKEND_EXIT=""
FRONTEND_EXIT=""
CLEANED=0

write_meta () {
    local ended_at="${1:-}"
    local ended_field='"ended_at": null'
    if [ -n "$ended_at" ]; then
        ended_field="\"ended_at\": \"$ended_at\""
    fi
    cat > "$LOG_DIR/meta.json" <<EOF
{
  "started_at": "$STARTED_AT",
  $ended_field,
  "ended_reason": "$ENDED_REASON",
  "git_sha": "$GIT_SHA",
  "sandbox_mode": "$SANDBOX",
  "runs_root": "$REPROLAB_RUNS_ROOT",
  "label": "$LABEL",
  "backend_pid": ${BACKEND_PID:-null},
  "frontend_pid": ${FRONTEND_PID:-null},
  "backend_exit": ${BACKEND_EXIT:-null},
  "frontend_exit": ${FRONTEND_EXIT:-null}
}
EOF
}

PY_BIN=".venv/Scripts/python.exe"
if [ ! -f "$PY_BIN" ]; then PY_BIN=".venv/bin/python"; fi
if [ ! -f "$PY_BIN" ]; then
    echo "[dev.sh] no venv at .venv — create one with: python -m venv .venv && pip install -r backend/requirements.txt" >&2
    exit 1
fi

# Free :8000 / :3000 from any stale dev server so re-running this script is
# safe. Windows path uses PowerShell (lsof is unreliable on Windows + git-bash);
# macOS / Linux use lsof + kill.
free_port () {
    local port="$1"
    if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -Command \
            "Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_.OwningProcess -Force -ErrorAction SilentlyContinue }" \
            >/dev/null 2>&1 || true
    elif command -v lsof >/dev/null 2>&1; then
        local pids
        pids="$(lsof -t -i ":$port" -sTCP:LISTEN 2>/dev/null || true)"
        if [ -n "$pids" ]; then
            # shellcheck disable=SC2086
            kill -9 $pids 2>/dev/null || true
        fi
    fi
}
[ "$WANT_BACKEND"  = "1" ] && free_port 8000
[ "$WANT_FRONTEND" = "1" ] && free_port 3000

# Prune old log dirs if --keep N was given. We sort by name (the TS prefix
# sorts chronologically) so this is just `ls -1 | head -n -KEEP`.
prune_old_logs () {
    local keep="$1"
    if ! [[ "$keep" =~ ^[0-9]+$ ]] || [ "$keep" -lt 1 ]; then
        echo "[dev.sh] --keep N must be a positive integer; got '$keep' (skipping prune)" >&2
        return
    fi
    local dirs
    dirs="$(ls -1d logs/*/ 2>/dev/null | sort)"
    local total
    total="$(printf '%s\n' "$dirs" | sed '/^$/d' | wc -l | tr -d ' ')"
    if [ "$total" -le "$keep" ]; then
        return
    fi
    local victim_count=$((total - keep))
    local victims
    victims="$(printf '%s\n' "$dirs" | sed '/^$/d' | head -n "$victim_count")"
    printf '%s\n' "$victims" | while IFS= read -r d; do
        [ -z "$d" ] && continue
        # Never prune the directory we just created.
        case "$d" in
            "$LOG_DIR/"|"$LOG_DIR") continue ;;
        esac
        echo "[dev.sh] prune $d"
        rm -rf "$d"
    done
}
if [ -n "$KEEP" ]; then
    prune_old_logs "$KEEP"
fi

echo "[dev.sh] logs    -> $LOG_DIR"
echo "[dev.sh] sandbox -> $SANDBOX"
echo "[dev.sh] runs    -> $REPROLAB_RUNS_ROOT"

if [ "$WANT_BACKEND" = "1" ]; then
    # --reload-dir backend: only watch backend/ source for hot-reload. The
    # default watches cwd, which includes runs/ and logs/ — when the pipeline
    # writes generated code WatchFiles would trigger a reload mid-run.
    "$PY_BIN" -m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000 --reload --reload-dir backend \
        > "$SERVER_DIR/backend.log" 2>&1 &
    BACKEND_PID=$!
fi

if [ "$WANT_FRONTEND" = "1" ]; then
    (
        cd frontend && REPROLAB_BACKEND_URL=http://127.0.0.1:8000 npm run dev
    ) > "$SERVER_DIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!
fi

write_meta

cleanup () {
    # Idempotent — signal traps re-enter via the EXIT trap.
    [ "$CLEANED" = "1" ] && return
    CLEANED=1

    local ended_at
    ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # Stop children before sampling exit codes — otherwise `wait` blocks
    # forever on a still-running uvicorn. `wait` returns the child's exit
    # status; we disable `set -e` around it so a non-zero exit is data,
    # not a fatal error in the trap.
    set +e
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null
        wait "$BACKEND_PID" 2>/dev/null
        BACKEND_EXIT=$?
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null
        wait "$FRONTEND_PID" 2>/dev/null
        FRONTEND_EXIT=$?
    fi
    set -e

    write_meta "$ended_at"

    # Best-effort manifest. Never block exit on this.
    "$PY_BIN" scripts/_write_manifest.py "$LOG_DIR" || true

    [ "$WANT_BACKEND"  = "1" ] && free_port 8000
    [ "$WANT_FRONTEND" = "1" ] && free_port 3000
}
trap 'ENDED_REASON=int;  cleanup; exit 130' INT
trap 'ENDED_REASON=term; cleanup; exit 143' TERM
trap 'cleanup' EXIT

if [ -n "$BACKEND_PID" ]; then
    echo "[dev.sh] backend  pid=$BACKEND_PID  log=$SERVER_DIR/backend.log"
fi
if [ -n "$FRONTEND_PID" ]; then
    echo "[dev.sh] frontend pid=$FRONTEND_PID log=$SERVER_DIR/frontend.log"
fi
[ "$WANT_FRONTEND" = "1" ] && echo "[dev.sh] open http://localhost:3000/lab"

# Block until any monitored child exits. `wait -n` is bash 4.3+; default macOS
# /bin/bash is 3.2, so poll with kill -0 (presence check, no signal sent).
while :; do
    if [ -n "$BACKEND_PID" ]  && ! kill -0 "$BACKEND_PID"  2>/dev/null; then break; fi
    if [ -n "$FRONTEND_PID" ] && ! kill -0 "$FRONTEND_PID" 2>/dev/null; then break; fi
    if [ -z "$BACKEND_PID" ] && [ -z "$FRONTEND_PID" ]; then break; fi
    sleep 1
done
