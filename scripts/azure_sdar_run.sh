#!/usr/bin/env bash
# azure_sdar_run.sh — launch the capped smallest-two SDAR reproduction with GPU
# cells on Azure AKS and the reasoning loop local (claude-oauth + Sonnet).
# Runs the preflight first and aborts if it is RED.
#
# Usage: scripts/azure_sdar_run.sh [env-file=.env.azure] [extra CLI args...]
#   The first arg is always the env file; anything after it is forwarded
#   verbatim to `backend.cli reproduce` — e.g. to retry only failed cells:
#     scripts/azure_sdar_run.sh .env.azure --resume-cells
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_FILE="${1:-.env.azure}"
shift || true   # remaining args forward to the CLI (e.g. --resume-cells)
[[ -f "$ENV_FILE" ]] && set -a && . "$ENV_FILE" && set +a

# OAuth on both reasoning surfaces — clear any shadowing API keys (CLAUDE.md gotcha).
unset OPENAI_API_KEY ANTHROPIC_API_KEY OPENRESEARCH_FORCE_SANDBOX 2>/dev/null || true

echo "== preflight =="
scripts/azure_sdar_preflight.sh "$ENV_FILE" || { echo "Preflight RED — aborting."; exit 1; }

export OPENRESEARCH_AZURE_GPU_SKUS='["azure_a100_80"]'
export OPENRESEARCH_AZURE_FILES_CACHE_ENABLED=false   # blob-only (spec §4.1)
export OPENRESEARCH_ACCELERATOR=off                   # accel needs OpenAI; off
export OPENRESEARCH_DYNAMIC_GPU=true
export OPENRESEARCH_BASELINE_EXTRA_GUIDANCE="SCOPE: reproduce SDAR using ONLY the two smallest model variants — Qwen3-1.7B and Qwen2.5-3B-Instruct. Honest-omit Qwen2.5-7B (declare in metrics.json['omitted']). Use real pretrained HF weights (no surrogate) and the real ALFWorld + Search-QA + WebShop datasets, but evaluate on a small representative slice (~32 tasks/env) to keep wall-clock practical on a single A100-80GB. Run the GRPO baseline and the proposed SDAR; the ablations (OPSD, Skill-SD, GRPO+OPSD, RLSD) may be code-present and declared. Report per_model results for 1.7B and 3B."

LOG="runs/sdar-azure-$(date +%s).log"
echo "== launch (log → $LOG) =="
.venv/bin/python -m backend.cli reproduce 2605.15155 \
  --mode rlm --sandbox azure --model claude-oauth --paper-hint 2605.15155 \
  --scope-spec '{"models":["Qwen3-1.7B","Qwen2.5-3B-Instruct"],"seeds":[42]}' \
  --force-single-gpu \
  --max-wall-clock 21600 --max-usd 30 --max-run-gpu-usd 25 --max-gpu-usd-per-hour 4 \
  "$@" \
  2>&1 | tee "$LOG"

echo "== done. Monitor with: scripts/azure_sdar_monitor.sh =="
