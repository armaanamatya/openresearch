#!/usr/bin/env bash
set -Eeuo pipefail

# Cancel an OpenResearch SDAR/GCP VM run and optionally stop the GPU VM.
#
# Default behavior:
#   - SSH to the VM.
#   - TERM then KILL OpenResearch reproduction processes owned by any user.
#   - Leave the VM running for inspection.
#
# Usage:
#   scripts/cancel_gcp_sdar_run.sh
#   scripts/cancel_gcp_sdar_run.sh --stop-vm
#   scripts/cancel_gcp_sdar_run.sh --project deepinvent-ext-ut --zone us-central1-c --instance sdar-a100-8g
#
# If Codex is unavailable, run this directly from any shell with gcloud auth.

PROJECT="deepinvent-ext-ut"
ZONE="us-central1-c"
INSTANCE="sdar-a100-8g"
REMOTE_DIR="/home/abheekp/openresearch"
CLOUDSDK_CONFIG_DIR="${CLOUDSDK_CONFIG:-/home/abheekp/.config/gcloud}"
STOP_VM=0
YES=0

usage() {
  sed -n '1,28p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      PROJECT="${2:?missing value for --project}"
      shift 2
      ;;
    --zone)
      ZONE="${2:?missing value for --zone}"
      shift 2
      ;;
    --instance)
      INSTANCE="${2:?missing value for --instance}"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="${2:?missing value for --remote-dir}"
      shift 2
      ;;
    --cloudsdk-config)
      CLOUDSDK_CONFIG_DIR="${2:?missing value for --cloudsdk-config}"
      shift 2
      ;;
    --stop-vm)
      STOP_VM=1
      shift
      ;;
    -y|--yes)
      YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

GCLOUD=(gcloud --project "$PROJECT")
if [[ -n "$CLOUDSDK_CONFIG_DIR" ]]; then
  export CLOUDSDK_CONFIG="$CLOUDSDK_CONFIG_DIR"
fi

confirm() {
  local prompt="$1"
  if [[ "$YES" -eq 1 ]]; then
    return 0
  fi
  read -r -p "$prompt [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" || "$ans" == "yes" || "$ans" == "YES" ]]
}

echo "Target project : $PROJECT"
echo "Target zone    : $ZONE"
echo "Target instance: $INSTANCE"
echo "Remote dir     : $REMOTE_DIR"
echo "Cloud SDK cfg  : ${CLOUDSDK_CONFIG:-<default>}"
echo

if ! confirm "Terminate OpenResearch reproduction processes on the VM?"; then
  echo "Cancelled by user."
  exit 1
fi

REMOTE_SCRIPT=$(cat <<'REMOTE_EOF'
set -Eeuo pipefail
REMOTE_DIR="$1"
cd "$REMOTE_DIR" 2>/dev/null || true

mkdir -p runs/_ops
date -Is > runs/_ops/cancel_requested_at.txt 2>/dev/null || true

echo "Before cancellation:"
pgrep -af 'python .*backend\.cli reproduce|python .*scripts/batch_reproduce\.py|backend\.cli reproduce|scripts/batch_reproduce\.py|train_cell\.py|run_experiment|uvicorn backend\.app' || true

mapfile -t pids < <(
  pgrep -f 'python .*backend\.cli reproduce|python .*scripts/batch_reproduce\.py|backend\.cli reproduce|scripts/batch_reproduce\.py|train_cell\.py|run_experiment' || true
)

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No matching OpenResearch run processes found."
else
  echo "Sending TERM to: ${pids[*]}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  sleep 10

  mapfile -t still < <(
    pgrep -f 'python .*backend\.cli reproduce|python .*scripts/batch_reproduce\.py|backend\.cli reproduce|scripts/batch_reproduce\.py|train_cell\.py|run_experiment' || true
  )
  if [[ "${#still[@]}" -gt 0 ]]; then
    echo "Sending KILL to still-running processes: ${still[*]}"
    kill -KILL "${still[@]}" 2>/dev/null || true
  fi
fi

echo
echo "After cancellation:"
pgrep -af 'python .*backend\.cli reproduce|python .*scripts/batch_reproduce\.py|backend\.cli reproduce|scripts/batch_reproduce\.py|train_cell\.py|run_experiment' || true

echo
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
fi
REMOTE_EOF
)

"${GCLOUD[@]}" compute ssh "$INSTANCE" \
  --zone "$ZONE" \
  --command "bash -s -- '$REMOTE_DIR'" <<<"$REMOTE_SCRIPT"

if [[ "$STOP_VM" -eq 1 ]]; then
  echo
  if confirm "Stop VM $INSTANCE in $ZONE to stop GPU billing?"; then
    "${GCLOUD[@]}" compute instances stop "$INSTANCE" --zone "$ZONE"
  else
    echo "VM left running."
  fi
else
  echo
  echo "VM left running. Re-run with --stop-vm to stop GPU billing after cancellation."
fi
