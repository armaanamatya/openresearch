#!/usr/bin/env bash
# =============================================================================
# sdar_gcp_watch.sh — Lifecycle-aware monitor for an SDAR-on-GCP reproduction.
# =============================================================================
#
# PURPOSE
#   Polls a running GCP VM every INTERVAL seconds, decoding the run's dashboard
#   events and experiment-runs ledger into a single readable status line per
#   tick. Auto-stops the VM when the run reaches a terminal state (unless
#   KEEP_UP=1) to halt billing. A real gcloud call is never made to the GCP
#   control-plane except for VM-status checks and the optional auto-stop.
#
# USAGE
#   PROJECT_ID=sdar_gcp_e2e bash scripts/sdar_gcp_watch.sh
#
#   # Verbose (renders recent events + live.log tail each tick):
#   PROJECT_ID=sdar_gcp_e2e VERBOSE=1 bash scripts/sdar_gcp_watch.sh
#
#   # Keep VM alive on terminal (you handle billing):
#   PROJECT_ID=sdar_gcp_e2e KEEP_UP=1 bash scripts/sdar_gcp_watch.sh
#
#   # Faster polling / cap budget:
#   PROJECT_ID=sdar_gcp_e2e INTERVAL=60 MAX_TICKS=300 bash scripts/sdar_gcp_watch.sh
#
#   # Delegate from the e2e runner (auto-wired):
#   scripts/sdar_gcp_e2e.sh watch
#
# STAGE LABELS (priority order, printed each tick)
#   FINALIZED score=<s> verdict=<v>  — run_complete/fatal/interrupted in events
#   training(gpu N%)                 — GPU utilisation >40 %
#   repairing(N)                     — forced_repair_iteration count increased
#   harness-driving                  — lifecycle_drive event seen, GPU ≤ 40 %
#   root-degenerated                 — degenerate event seen, no lifecycle_drive
#   implementing/setup               — no experiment_runs yet, no degenerate
#   running                          — fall-through default
#
# AUTO-STOP (billing guard)
#   On a terminal run, unless KEEP_UP=1, this script calls:
#     gcloud compute instances stop <INSTANCE> --quiet
#   and prints "VM stopped (billing halted)". Set KEEP_UP=1 if you want to
#   inspect the VM yourself before it is stopped.
#
# EXIT CODES
#   0   — run reached a terminal state (run_complete / run_fatal / interrupted)
#   3   — VM is not RUNNING (preempted or stopped externally)
#   4   — MAX_TICKS exhausted; VM may still be running

set -uo pipefail
# NOTE: -e is deliberately omitted. A transient SSH blip or poll failure must
# never kill the watcher; failures are handled explicitly with `|| true` and
# `continue`.

# ---------------------------------------------------------------------------
# Config — override via env; sane defaults match sdar_gcp_e2e.sh conventions.
# ---------------------------------------------------------------------------
: "${PROJECT_ID:?PROJECT_ID is required — e.g. PROJECT_ID=sdar_gcp_e2e bash $0}"

INTERVAL="${INTERVAL:-120}"        # seconds between ticks
EVENTS_N="${EVENTS_N:-6}"         # events rendered in VERBOSE block
VERBOSE="${VERBOSE:-0}"            # 1 = render recent events + live.log each tick
KEEP_UP="${KEEP_UP:-0}"           # 1 = do NOT auto-stop VM on terminal
MAX_TICKS="${MAX_TICKS:-160}"     # hard budget: 160 × 120s = ~5.3 h

# VM identity — same defaults as sdar_gcp_e2e.sh
ZONE="${OPENRESEARCH_GCP_ZONE:-us-central1-b}"
PROJECT="${OPENRESEARCH_GCP_PROJECT:-deepinvent-ext-ut}"
INSTANCE="${OPENRESEARCH_GCP_INSTANCE:-sdar-a100-od}"
SSH_USER="${OPENRESEARCH_GCP_SSH_USER:-abheekp}"
REMOTE_DIR="${OPENRESEARCH_REMOTE_DIR:-/home/abheekp/openresearch}"

export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-/home/abheekp/.config/gcloud}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ssh_() — run COMMAND on the remote VM, with an optional TIMEOUT (default 60s).
# Matches the pattern from sdar_gcp_e2e.sh.
ssh_() { timeout "${2:-60}" gcloud compute ssh "$SSH_USER@$INSTANCE" \
           --zone "$ZONE" --project "$PROJECT" --quiet --command "$1"; }

# vm_status — returns the GCP instance status string (RUNNING / TERMINATED / …)
vm_status() {
  gcloud --project "$PROJECT" compute instances describe "$INSTANCE" \
    --zone "$ZONE" --format='value(status)' 2>/dev/null || true
}

# vm_stop — gracefully stop the VM (halt billing).
vm_stop() {
  gcloud --project "$PROJECT" compute instances stop "$INSTANCE" \
    --zone "$ZONE" --quiet 2>&1 | tail -1 || true
}

# now_utc — compact UTC timestamp for tick labels.
now_utc() { date -u +%H:%MZ; }

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  sdar_gcp_watch.sh — SDAR-on-GCP lifecycle monitor"
echo "  PROJECT_ID : $PROJECT_ID"
echo "  VM         : $INSTANCE ($ZONE)"
echo "  INTERVAL   : ${INTERVAL}s  MAX_TICKS: $MAX_TICKS  (~$((MAX_TICKS * INTERVAL / 3600))h budget)"
echo "  VERBOSE    : $VERBOSE   KEEP_UP: $KEEP_UP"
echo "  Remote dir : $REMOTE_DIR/runs/$PROJECT_ID"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# State carried across ticks (for delta detection)
# ---------------------------------------------------------------------------
prev_repair=0

# ---------------------------------------------------------------------------
# Remote poll helper (assembled as a single SSH round-trip)
# ---------------------------------------------------------------------------
# Returns a line of KEY=VAL pairs; avoids the grep -c || echo 0 double-output
# bug by using a shell function that captures the count explicitly.
#
# Keys emitted:
#   XR        — lines in experiment_runs.jsonl (wc -l)
#   DEG       — count of root_degenerate events
#   LD        — count of lifecycle_drive events
#   REPAIR    — count of forced_repair_iteration events
#   DONE      — count of terminal events (run_complete|run_fatal|run_interrupted)
#   GPU       — highest GPU utilisation % across cards (nvidia-smi; "" if unavailable)
#   SCORE     — latest lifecycle_drive rubric_score field value (may be empty)
#   FR_SCORE  — final_report.json overall_score (empty if file absent)
#   FR_VER    — final_report.json verdict        (empty if file absent)

_REMOTE_POLL='
set -uo pipefail
RD='"$REMOTE_DIR"'
P="$RD/runs/'"$PROJECT_ID"'"
EV="$P/dashboard_events.jsonl"
ER="$P/experiment_runs.jsonl"
FR="$P/final_report.json"

cnt(){ local x; x=$(grep -c "$1" "$2" 2>/dev/null); echo "${x:-0}"; }
cntE(){ local x; x=$(grep -cE "$1" "$2" 2>/dev/null); echo "${x:-0}"; }

XR=$(cat "$ER" 2>/dev/null | wc -l)
DEG=$(cnt "root_degenerate" "$EV")
LD=$(cnt "lifecycle_drive" "$EV")
REPAIR=$(cnt "forced_repair_iteration" "$EV")
DONE=$(cntE "run_complete|run_fatal|run_interrupted" "$EV")
GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
      | grep -oE "^[0-9]+" | sort -rn | head -1)
SCORE=$(grep -o "\"rubric_score\":[0-9.]*" "$EV" 2>/dev/null \
        | grep -o "[0-9.]*$" | tail -1)
if [ -f "$FR" ]; then
  FR_SCORE=$(python3 -c "
import json,sys
d=json.load(open(\"$FR\"))
r=d.get(\"rubric\",{})
s=r.get(\"overall_score\") or d.get(\"overall_score\") or d.get(\"rubric_score\")
print(s if s is not None else \"\")
" 2>/dev/null)
  FR_VER=$(python3 -c "
import json,sys
d=json.load(open(\"$FR\"))
print(d.get(\"verdict\") or d.get(\"reproducibility\",{}).get(\"verdict\") or \"\")
" 2>/dev/null)
else
  FR_SCORE=""
  FR_VER=""
fi

printf "XR=%s DEG=%s LD=%s REPAIR=%s DONE=%s GPU=%s SCORE=%s FR_SCORE=%s FR_VER=%s\n" \
  "${XR:-0}" "${DEG:-0}" "${LD:-0}" "${REPAIR:-0}" "${DONE:-0}" \
  "${GPU:-}" "${SCORE:-}" "${FR_SCORE:-}" "${FR_VER:-}"
'

# ---------------------------------------------------------------------------
# Verbose block — rendered events + live.log tail
# ---------------------------------------------------------------------------
_REMOTE_VERBOSE='
RD='"$REMOTE_DIR"'
P="$RD/runs/'"$PROJECT_ID"'"
EV="$P/dashboard_events.jsonl"
LL="$P/code/.exec_live.log"
[ -f "$EV" ] && python3 "$RD/scripts/render_run_events.py" --tail '"$EVENTS_N"' "$EV" 2>/dev/null || true
if [ -f "$LL" ]; then
  LINE=$(tail -1 "$LL" 2>/dev/null)
  [ -n "$LINE" ] && echo "  live.log: $LINE"
fi
'

# ---------------------------------------------------------------------------
# Summary printer (called on terminal or preempt)
# ---------------------------------------------------------------------------
print_summary() {
  local fr_score="${1:-}"
  local fr_ver="${2:-}"
  echo ""
  echo "------------------------------------------------------------"
  echo "  FINAL SUMMARY"
  echo "  PROJECT_ID : $PROJECT_ID"
  if [[ -n "$fr_score" || -n "$fr_ver" ]]; then
    echo "  score      : ${fr_score:-(not recorded)}"
    echo "  verdict    : ${fr_ver:-(not recorded)}"
  else
    echo "  final_report.json not yet available (run may have aborted)"
  fi
  echo "------------------------------------------------------------"
  echo ""
}

# ---------------------------------------------------------------------------
# Main tick loop
# ---------------------------------------------------------------------------
for tick in $(seq 1 "$MAX_TICKS"); do

  # 1. Check VM is still RUNNING ----------------------------------------
  vm_s="$(vm_status)"
  if [[ "$vm_s" != "RUNNING" ]]; then
    echo ""
    echo "[tick $tick $(now_utc)] VM=$vm_s (preempted or stopped externally) — exiting"
    print_summary "" ""
    exit 3
  fi

  # 2. Remote poll (single round-trip) -----------------------------------
  poll_out="$(ssh_ "$_REMOTE_POLL" 75 2>/dev/null)" || {
    echo "[tick $tick $(now_utc)] poll failed (transient) — retrying"
    sleep "$INTERVAL"
    continue
  }

  # Parse KEY=VAL pairs from poll output
  xr=0; deg=0; ld=0; repair=0; done_=0; gpu=""; score=""; fr_score=""; fr_ver=""
  while IFS='=' read -r k v; do
    case "$k" in
      XR)       xr="$v" ;;
      DEG)      deg="$v" ;;
      LD)       ld="$v" ;;
      REPAIR)   repair="$v" ;;
      DONE)     done_="$v" ;;
      GPU)      gpu="$v" ;;
      SCORE)    score="$v" ;;
      FR_SCORE) fr_score="$v" ;;
      FR_VER)   fr_ver="$v" ;;
    esac
  done < <(echo "$poll_out" | tr ' ' '\n')

  # 3. Infer stage label -------------------------------------------------
  gpu_int="${gpu:-0}"
  # coerce to integer safely (nvidia-smi may be blank on CPU ticks)
  if ! [[ "$gpu_int" =~ ^[0-9]+$ ]]; then gpu_int=0; fi

  repair_delta=$(( repair - prev_repair ))

  if [[ "${done_:-0}" -gt 0 ]]; then
    stage="FINALIZED score=${fr_score:--} verdict=${fr_ver:--}"
  elif [[ "$gpu_int" -gt 40 ]]; then
    stage="training(gpu ${gpu_int}%)"
  elif [[ "$repair_delta" -gt 0 ]]; then
    stage="repairing(${repair})"
  elif [[ "${ld:-0}" -gt 0 ]]; then
    stage="harness-driving"
  elif [[ "${deg:-0}" -gt 0 && "${ld:-0}" -eq 0 ]]; then
    stage="root-degenerated"
  elif [[ "${xr:-0}" -eq 0 && "${deg:-0}" -eq 0 ]]; then
    stage="implementing/setup"
  else
    stage="running"
  fi

  prev_repair="$repair"

  # Latest score display
  display_score="${score:-${fr_score:--}}"

  # 4. Print status line -------------------------------------------------
  printf "[tick %d %s] stage=%-38s | XR=%s DEG=%s LD=%s repair=%s gpu=%s%% score=%s\n" \
    "$tick" "$(now_utc)" "$stage" \
    "${xr:-0}" "${deg:-0}" "${ld:-0}" "${repair:-0}" \
    "$gpu_int" "$display_score"

  # 5. Verbose block -----------------------------------------------------
  if [[ "$VERBOSE" == "1" ]]; then
    echo "  --- events (last $EVENTS_N) + live.log ---"
    ssh_ "$_REMOTE_VERBOSE" 45 2>/dev/null | sed 's/^/  /' || true
    echo ""
  fi

  # 6. Terminal handling -------------------------------------------------
  if [[ "${done_:-0}" -gt 0 ]]; then
    print_summary "$fr_score" "$fr_ver"
    if [[ "$KEEP_UP" == "1" ]]; then
      echo "KEEP_UP=1 — VM left running. Stop manually: scripts/sdar_gcp_e2e.sh down"
    else
      echo "Stopping VM to halt billing..."
      vm_stop
      echo "VM stopped (billing halted)."
    fi
    exit 0
  fi

  # 7. Sleep before next tick -------------------------------------------
  sleep "$INTERVAL"

done

# ---------------------------------------------------------------------------
# Budget exhausted
# ---------------------------------------------------------------------------
echo ""
echo "watch budget exhausted ($((MAX_TICKS * INTERVAL))s elapsed across $MAX_TICKS ticks)."
echo "The VM may still be RUNNING with an active run."
echo "To continue monitoring:  scripts/sdar_gcp_e2e.sh monitor"
echo "To stop the VM:          scripts/sdar_gcp_e2e.sh down"
exit 4
