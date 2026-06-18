#!/usr/bin/env bash
# Launch the SDAR (arXiv 2605.15155) reproduction on the GCP A100 VM.
#
# Self-contained and idempotent-friendly: it sources the env file that
# `gcp_sdar_preflight.sh prepare` wrote, pins the Azure Foundry deployment as
# every role (OAuth-free), requests the paper's full 3-model scope (the 7B
# sharded over 2 cards), and execs the reproduction. Run it on the VM from the
# repo root, or — the intended path — let `gcp_sdar_preflight.sh launch` start
# it detached. The launch is fully driven by env + flags, so a fresh session
# re-runs it with one command and no edits.
set -euo pipefail

REPO_DIR="${OPENRESEARCH_REMOTE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

ENV_FILE="runs/.cache/sdar_gcp.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE missing — run 'gcp_sdar_preflight.sh prepare' to [GREEN] first" >&2
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "ERROR: .venv/bin/python missing — run 'gcp_sdar_preflight.sh prepare' to [GREEN] first" >&2
  exit 1
fi
# Shell wins over .env in the harness, so these exports pin the run.
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

PROJECT_ID="${OPENRESEARCH_SDAR_PROJECT_ID:-sdar_gcp_20260618}"
# Default to ChatGPT-latest (reasoning-class) as root+executor+grader+verifier;
# override via AZURE_FOUNDRY_DEPLOYMENT=grok-4.3 for the grok alternative.
export AZURE_FOUNDRY_DEPLOYMENT="${AZURE_FOUNDRY_DEPLOYMENT:-gpt-chat-latest}"
export OPENRESEARCH_GRADER_SAMPLES="${OPENRESEARCH_GRADER_SAMPLES:-3}"
export OPENRESEARCH_BASELINE_EXTRA_GUIDANCE="${OPENRESEARCH_BASELINE_EXTRA_GUIDANCE:-$(cat <<'G'
FULL 3-MODEL SCOPE for this run: include ALL of Qwen/Qwen3-1.7B, Qwen/Qwen2.5-3B-Instruct,
AND Qwen/Qwen2.5-7B-Instruct (seed 0). Emit cells.json for all three across the paper's envs
(ALFWorld, WebShop, Search-QA). The 7B does NOT fit one 40GB card: set "gpus": 2 on every 7B cell
and shard it with device_map="auto" (the harness gives that cell 2 dedicated cards). Keep the 1.7B
and 3B cells at gpus:1. Set honest est_vram_gb per cell (~14 for 3B, ~32 for 7B). Do NOT descope the 7B.
G
)}"

echo "[sdar_gcp_run] project_id=$PROJECT_ID deployment=$AZURE_FOUNDRY_DEPLOYMENT grader_samples=$OPENRESEARCH_GRADER_SAMPLES"
exec env -u ANTHROPIC_API_KEY .venv/bin/python -m backend.cli reproduce 2605.15155 \
  --mode rlm --sandbox local --model grok \
  --models executor=grok,grader=grok,verifier=grok \
  --paper-hint 2605.15155 \
  --gpu-mode max --gpu-parallelism multi --vram-gb 40 \
  --no-force-single-gpu --max-wall-clock 86400 \
  --project-id "$PROJECT_ID"
