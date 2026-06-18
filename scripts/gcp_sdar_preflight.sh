#!/usr/bin/env bash
# GCP SDAR VM preflight / preparation wrapper.
#
# Usage:
#   scripts/gcp_sdar_preflight.sh status
#   scripts/gcp_sdar_preflight.sh start
#   scripts/gcp_sdar_preflight.sh sync
#   scripts/gcp_sdar_preflight.sh check
#   scripts/gcp_sdar_preflight.sh prepare
#   scripts/gcp_sdar_preflight.sh launch
#   scripts/gcp_sdar_preflight.sh stop
#
# `prepare` starts the VM, syncs the repo, installs SDAR deps, warms datasets and
# model caches, provisions ALFWorld/WebShop/Search-QA, then leaves the VM running
# for an immediate run. `launch` starts the reproduction detached on the VM (gate
# on a GREEN prepare first). Use `scripts/cancel_gcp_sdar_run.sh --stop-vm` or
# `stop` when done.
set -euo pipefail

ACTION="${1:-check}"
PROJECT="${OPENRESEARCH_GCP_PROJECT:-deepinvent-ext-ut}"
ZONE="${OPENRESEARCH_GCP_ZONE:-us-central1-c}"
INSTANCE="${OPENRESEARCH_GCP_INSTANCE:-sdar-a100-8g}"
REMOTE_DIR="${OPENRESEARCH_REMOTE_DIR:-/home/abheekp/openresearch}"
CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-/home/abheekp/.config/gcloud}"
REMOTE_USER="${OPENRESEARCH_GCP_SSH_USER:-abheekp}"
REQUIRE_SPOT="${OPENRESEARCH_REQUIRE_SPOT:-true}"

export CLOUDSDK_CONFIG

gcloud_base=(gcloud --project "$PROJECT")
ssh_base=(gcloud compute ssh "$REMOTE_USER@$INSTANCE" --zone "$ZONE" --project "$PROJECT" --quiet --command)

status_only() {
  "${gcloud_base[@]}" compute instances describe "$INSTANCE" \
    --zone "$ZONE" --format='value(status)'
}

describe_instance() {
  "${gcloud_base[@]}" compute instances describe "$INSTANCE" \
    --zone "$ZONE" \
    --format='table(name,status,scheduling.provisioningModel,scheduling.preemptible,machineType.basename())'
}

ensure_spot() {
  if [[ "${REQUIRE_SPOT,,}" != "true" ]]; then
    return
  fi
  local model preemptible
  model="$("${gcloud_base[@]}" compute instances describe "$INSTANCE" --zone "$ZONE" --format='value(scheduling.provisioningModel)')"
  preemptible="$("${gcloud_base[@]}" compute instances describe "$INSTANCE" --zone "$ZONE" --format='value(scheduling.preemptible)')"
  if [[ "$model" != "SPOT" && "$preemptible" != "True" ]]; then
    echo "refusing to use non-spot GPU VM $INSTANCE (provisioningModel=$model preemptible=$preemptible)" >&2
    echo "set OPENRESEARCH_REQUIRE_SPOT=false only for an intentional on-demand run" >&2
    exit 1
  fi
}

start_vm() {
  local s
  ensure_spot
  s="$(status_only 2>/dev/null || true)"
  if [[ "$s" == "RUNNING" ]]; then
    echo "VM already RUNNING"
    return
  fi
  if [[ "$s" == "TERMINATED" ]]; then
    "${gcloud_base[@]}" compute instances start "$INSTANCE" --zone "$ZONE"
  elif [[ "$s" == "STAGING" || "$s" == "PROVISIONING" || "$s" == "STOPPING" || "$s" == "SUSPENDING" ]]; then
    echo "VM is $s; waiting for RUNNING"
  else
    echo "VM is in unexpected state '$s'; refusing to continue" >&2
    exit 1
  fi
  wait_for_running
}

wait_for_running() {
  local deadline s
  deadline=$((SECONDS + ${OPENRESEARCH_GCP_START_TIMEOUT_S:-900}))
  while (( SECONDS < deadline )); do
    s="$(status_only 2>/dev/null || true)"
    case "$s" in
      RUNNING)
        echo "VM is RUNNING"
        return
        ;;
      TERMINATED)
        echo "VM returned to TERMINATED during start; spot capacity may have been reclaimed" >&2
        exit 1
        ;;
      STAGING|PROVISIONING|STOPPING|SUSPENDING)
        sleep 10
        ;;
      *)
        echo "VM entered unexpected state '$s' while waiting for RUNNING" >&2
        exit 1
        ;;
    esac
  done
  echo "timed out waiting for $INSTANCE to become RUNNING" >&2
  exit 1
}

stop_vm() {
  "${gcloud_base[@]}" compute instances stop "$INSTANCE" --zone "$ZONE"
}

sync_repo() {
  start_vm
  "${ssh_base[@]}" "mkdir -p $REMOTE_DIR"
  local stage
  stage="$(mktemp -d)"
  trap 'rm -rf "$stage"' RETURN
  local files=(
    requirements.txt pyproject.toml CHANGELOG.md CLAUDE.md gcp_info.md issues.md
  )
  [[ -f .env ]] && files+=(.env)
  rsync -a \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude 'backend/venv/' \
    --exclude 'runs/' \
    --exclude 'frontend/node_modules/' \
    --exclude 'frontend/.next/' \
    backend scripts docs docker infra \
    "${files[@]}" \
    "$stage/"
  gcloud compute scp --recurse \
    --zone "$ZONE" --project "$PROJECT" --quiet --compress \
    "$stage"/* \
    "$REMOTE_USER@$INSTANCE:$REMOTE_DIR/"
  "${ssh_base[@]}" "chmod +x $REMOTE_DIR/scripts/gcp_sdar_preflight.sh $REMOTE_DIR/scripts/sdar_gcp_assets.py $REMOTE_DIR/scripts/cancel_gcp_sdar_run.sh"
}

remote_check() {
  start_vm
  # Source the env file written by `prepare` so the standalone check sees the
  # dedicated WebShop interpreter (OPENRESEARCH_WEBSHOP_PYTHON) and cache paths.
  "${ssh_base[@]}" "cd $REMOTE_DIR && { [ -f runs/.cache/sdar_gcp.env ] && . runs/.cache/sdar_gcp.env || true; } && .venv/bin/python scripts/sdar_gcp_assets.py --check --require-gpu --min-gpus 8"
}

remote_prepare() {
  start_vm
  sync_repo
  # System build prerequisites for source-built Python deps. Ubuntu 24.04 ships
  # only `python3`, but some sdists shell out to bare `python` and build native
  # code (e.g. fast-downward-textworld -> cmake; alfworld/textworld -> C exts).
  # openjdk-17-jdk-headless (JDK, not just JRE) is required by pyserini's jnius,
  # which locates JAVA_HOME via `javac` — the JRE alone fails with "Unable to find
  # javac". WebShop is best-effort, so this only improves its odds of coming up.
  # Idempotent; needs passwordless sudo (GCP default for the creating user).
  "${ssh_base[@]}" "sudo bash -c 'export DEBIAN_FRONTEND=noninteractive && apt-get update -qq && apt-get install -y python-is-python3 cmake ninja-build build-essential libffi-dev openjdk-17-jdk-headless'"
  # Ensure uv is available for creating Python-version-pinned venvs. The run venv
  # is created at Python 3.10 so WebShop's dedicated venv (also 3.10) can share
  # the same interpreter; uv resolves the exact minor-version binary automatically.
  "${ssh_base[@]}" "command -v uv || (curl -LsSf https://astral.sh/uv/install.sh | sh) && export PATH=\"\$HOME/.local/bin:\$PATH\""
  # Ensure the run venv is Python 3.12 — recreate it when MISSING or below the
  # harness floor. The harness requires >=3.11 (it does `from typing import Self`,
  # a 3.11+ name; pyproject requires-python = ">=3.11"), so the run venv must NOT
  # be 3.10 — only WebShop's frozen stack needs 3.10, and that lives in its own
  # dedicated venv. `--seed` gives the uv venv pip/setuptools (uv omits them by
  # default). `sudo rm` clears any root-owned files a prior `sudo pip` left behind.
  # Idempotent: an existing 3.12 venv is reused as-is.
  "${ssh_base[@]}" "export PATH=\"\$HOME/.local/bin:\$PATH\" && cd $REMOTE_DIR && if [ ! -x .venv/bin/python ] || ! .venv/bin/python --version 2>&1 | grep -q '3\\.12'; then sudo rm -rf .venv && uv venv --python 3.12 --seed .venv; fi && .venv/bin/python -m pip install -r backend/requirements.txt && .venv/bin/python scripts/sdar_gcp_assets.py --prepare --check --require-gpu --min-gpus 8"
}

launch_run() {
  start_vm
  # Push the latest run script (it may post-date the last prepare sync).
  gcloud compute scp --zone "$ZONE" --project "$PROJECT" --quiet \
    scripts/sdar_gcp_run.sh "$REMOTE_USER@$INSTANCE:$REMOTE_DIR/scripts/sdar_gcp_run.sh"
  # Refuse to double-launch the same paper; require a GREEN-prepared env file;
  # then start the reproduction fully detached (setsid+nohup, stdin from
  # /dev/null) so it survives this SSH session closing. The run logs to
  # runs/sdar_gcp_run.out and streams to runs/<project_id>/dashboard_events.jsonl.
  "${ssh_base[@]}" "cd $REMOTE_DIR \
    && chmod +x scripts/sdar_gcp_run.sh \
    && { [ -f runs/.cache/sdar_gcp.env ] || { echo 'ERROR: runs/.cache/sdar_gcp.env missing — run prepare to [GREEN] first' >&2; exit 1; }; } \
    && { ! pgrep -f 'backend.cli reproduce 2605.15155' >/dev/null || { echo 'ERROR: a reproduce process for 2605.15155 is already running' >&2; exit 1; }; } \
    && ( setsid nohup bash scripts/sdar_gcp_run.sh > runs/sdar_gcp_run.out 2>&1 < /dev/null & ) \
    && sleep 4 \
    && echo '--- launch tail (runs/sdar_gcp_run.out) ---' \
    && tail -n 20 runs/sdar_gcp_run.out"
}

case "$ACTION" in
  status) describe_instance ;;
  start) start_vm ;;
  stop) stop_vm ;;
  sync) sync_repo ;;
  check) remote_check ;;
  prepare) remote_prepare ;;
  launch) launch_run ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 2
    ;;
esac
