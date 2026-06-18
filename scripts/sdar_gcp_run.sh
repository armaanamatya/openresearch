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
# Default to grok-4.3 as root+executor (the PROVEN RLM root — chat-only
# deployments like gpt-chat-latest REFUSE to drive the REPL loop; see the
# 2026-06-18 handoff). Override the deployment via AZURE_FOUNDRY_DEPLOYMENT.
export AZURE_FOUNDRY_DEPLOYMENT="${AZURE_FOUNDRY_DEPLOYMENT:-grok-4.3}"
export OPENRESEARCH_GRADER_SAMPLES="${OPENRESEARCH_GRADER_SAMPLES:-3}"
export OPENRESEARCH_BASELINE_EXTRA_GUIDANCE="${OPENRESEARCH_BASELINE_EXTRA_GUIDANCE:-$(cat <<'SDAR_GUIDANCE_EOF'
REAL REPRODUCTION - NO FABRICATION. The harness now DETECTS and REJECTS stub models, random
log-probs, hardcoded metrics, and zero-VRAM "training" (at preflight AND at run time):
- Load the REAL models with AutoModelForCausalLM.from_pretrained: Qwen/Qwen3-1.7B,
  Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct. NEVER an nn.Linear/Identity stub.
- Generate REAL rollouts (model.generate) on the REAL env episodes; compute REAL token
  log-probs from a model forward pass. NEVER torch.randn as log-probs.
- Report MEASURED success_rate/accuracy from actual evaluation. NEVER hardcode the paper's
  Table-1 numbers (e.g. 0.844). A cell that uses ~0 GPU memory is REJECTED as fabricated.

HYPERPARAMETERS (paper Section 3): lambda_SDAR=0.01, beta=5.0 for the main runs.

METHOD (Section 2, in train.py): OPSD surrogate Delta_t = logP_T - logP_theta with reverse-KL;
gated auxiliary loss g_t*loss with a STOP-GRADIENT sigmoid gate; full GRPO loss (group sampling
G=8, importance ratio, clip). Implement ALL THREE selectable gates: Entropy g=sigmoid(beta*h_t),
Gap g=sigmoid(beta*Delta_t), Soft-OR g=sigmoid(beta*[1-(1-h_t)(1-Delta_t)]).

MATRIX: emit cells.json for the FULL 3x3 - every model x every env (ALFWorld, Search-QA; add
WebShop only if its server is up), seed 0, 150 REAL training steps each. The 7B shards over 2
cards (gpus:2, device_map="auto"); 1.7B/3B at gpus:1. Honest est_vram_gb (~14 for 3B, ~32 for 7B).

DATA: Search-QA uses NQ + HotpotQA with an E5 retriever (batch 128, max_prompt 4096); ALFWorld
batch 16, 8 rollouts, max_prompt 2048.

COMPLETENESS (write the code; run what the wall-clock allows): baseline trainers (GRPO, GRPO+OPSD,
Skill-SD, RLSD, Skill-GRPO, Skill-GRPO*); the four retrieval strategies (UCB score=rbar+c*sqrt(lnN/n),
Keyword-Matching, Full, Random) loading the SkillBank; a beta/lambda ablation sweep script on ONE
representative cell; per-step teacher-student gap-mean + gate-activation-ratio logging (Figure 5).
Name the trainer train.py and emit an evaluation script that produces the Table-1/Table-2 numbers.

PRIORITY if time-constrained: REAL training of the SDAR method on the full 3x3 matrix with correct
hyperparameters + gap logging FIRST (a real partial beats a fabricated whole), then add
baselines/retrieval/sweeps. Do NOT fake any result to increase coverage.
SDAR_GUIDANCE_EOF
)}"

# Per-role models. Default: pure grok (OAuth-free). To put a reliable ChatGPT/
# gpt-5 grader+verifier behind the grok agent (more trustworthy grading), set —
# REQUIRES a LIVE OPENAI_API_KEY in .env (the bundled one is currently dead):
#   OPENRESEARCH_SDAR_MODELS=executor=grok,grader=gpt-5,verifier=gpt-5
export OPENRESEARCH_SDAR_MODELS="${OPENRESEARCH_SDAR_MODELS:-executor=grok,grader=grok,verifier=grok}"

echo "[sdar_gcp_run] project_id=$PROJECT_ID deployment=$AZURE_FOUNDRY_DEPLOYMENT models=$OPENRESEARCH_SDAR_MODELS grader_samples=$OPENRESEARCH_GRADER_SAMPLES"
exec env -u ANTHROPIC_API_KEY .venv/bin/python -m backend.cli reproduce 2605.15155 \
  --mode rlm --sandbox local --model grok \
  --models "$OPENRESEARCH_SDAR_MODELS" \
  --paper-hint 2605.15155 \
  --gpu-mode max --gpu-parallelism multi --vram-gb 40 \
  --no-force-single-gpu --max-wall-clock 86400 \
  --project-id "$PROJECT_ID"
