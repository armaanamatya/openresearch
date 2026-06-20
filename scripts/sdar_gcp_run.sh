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

# --- Sonnet executor OAuth token (headless) ---------------------------------
# The CLI reproduce path reads .env via pydantic Settings, but the OAuth check
# (factory.py) and the claude-agent-sdk read os.environ DIRECTLY, so a token
# sitting only in .env is invisible to them. Lift it into the environment here.
# Sourced in a subshell so quoting is handled exactly as bash would and no other
# .env var leaks into the run env. No-op for OAuth-free foundry-only runs.
if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && [ -f .env ]; then
  _claude_tok="$(set -a; . ./.env >/dev/null 2>&1; printf '%s' "${CLAUDE_CODE_OAUTH_TOKEN:-}")"
  if [ -n "$_claude_tok" ]; then
    export CLAUDE_CODE_OAUTH_TOKEN="$_claude_tok"
    echo "[sdar_gcp_run] CLAUDE_CODE_OAUTH_TOKEN exported from .env (Sonnet executor auth)"
  fi
  unset _claude_tok
fi

# --- Optional scope override (e.g. smallest-two de-risking smoke) -----------
# If a guidance file was staged at runs/.cache/sdar_scope_guidance.txt, use it
# as the implementer guidance; otherwise fall through to the full-matrix default
# below. Lets a cheap smallest-two smoke precede the full 3-model run with no
# script edit. An explicit OPENRESEARCH_BASELINE_EXTRA_GUIDANCE still wins.
_SDAR_GUIDANCE_FILE="runs/.cache/sdar_scope_guidance.txt"
if [ -z "${OPENRESEARCH_BASELINE_EXTRA_GUIDANCE:-}" ] && [ -f "$_SDAR_GUIDANCE_FILE" ]; then
  OPENRESEARCH_BASELINE_EXTRA_GUIDANCE="$(cat "$_SDAR_GUIDANCE_FILE")"
  export OPENRESEARCH_BASELINE_EXTRA_GUIDANCE
  echo "[sdar_gcp_run] scope guidance loaded from $_SDAR_GUIDANCE_FILE"
fi

PROJECT_ID="${OPENRESEARCH_SDAR_PROJECT_ID:-sdar_gcp_20260618}"
# Default to gpt-chat-latest as root + every sub-role (OAuth-free). Empirically
# verified 2026-06-19 to drive the REPL loop cleanly (emits ```repl fences,
# finish_reason=stop, reasoning_tokens=0, no refusal). The earlier "chat
# deployments REFUSE to drive the loop" conclusion was a misdiagnosed max_tokens
# 400: gpt-chat-latest is a reasoning-class model requiring max_completion_tokens
# + default temperature. That is now handled on every role — _is_reasoning_model
# (primitives + grader/verifier transport) and null-param omission (rlms root
# loop + executor Agents SDK both omit max_tokens/temperature when unset).
# Override the deployment via AZURE_FOUNDRY_DEPLOYMENT (e.g. =grok-4.3, =Kimi-K2.6).
export AZURE_FOUNDRY_DEPLOYMENT="${AZURE_FOUNDRY_DEPLOYMENT:-gpt-chat-latest}"
# Root token for --model: "foundry" is the neutral alias; the actual model is
# whatever AZURE_FOUNDRY_DEPLOYMENT names. Override to switch models without
# editing this script (e.g. OPENRESEARCH_SDAR_ROOT=grok or =kimi-k2.5).
export OPENRESEARCH_SDAR_ROOT="${OPENRESEARCH_SDAR_ROOT:-foundry}"
# Self-stop controls: set NO_AUTOSTOP=1 to leave the VM running for debug.
export OPENRESEARCH_SDAR_NO_AUTOSTOP="${OPENRESEARCH_SDAR_NO_AUTOSTOP:-0}"
# Outer wall-clock backstop (seconds): kills the reproduce process if the
# orchestrator itself wedges past the harness's own --max-wall-clock 86400.
export OPENRESEARCH_SDAR_OUTER_WALL_S="${OPENRESEARCH_SDAR_OUTER_WALL_S:-90000}"
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

# Per-role models. Default: pure foundry (OAuth-free, matches the root alias).
# "foundry" is the neutral token; the actual model is AZURE_FOUNDRY_DEPLOYMENT.
# To put a reliable ChatGPT/gpt-5 grader+verifier behind the foundry agent, set —
# REQUIRES a LIVE OPENAI_API_KEY in .env (the bundled one is currently dead):
#   OPENRESEARCH_SDAR_MODELS=executor=foundry,grader=gpt-5,verifier=gpt-5
export OPENRESEARCH_SDAR_MODELS="${OPENRESEARCH_SDAR_MODELS:-executor=foundry,grader=foundry,verifier=foundry}"

echo "[sdar_gcp_run] project_id=$PROJECT_ID root=$OPENRESEARCH_SDAR_ROOT deployment=$AZURE_FOUNDRY_DEPLOYMENT models=$OPENRESEARCH_SDAR_MODELS grader_samples=$OPENRESEARCH_GRADER_SAMPLES"

# self_stop <reason>: best-effort GCS upload, then halt the VM to stop GPU billing.
# The boot disk (with all run artifacts) persists after shutdown; flip back to the
# CPU machine type and start the VM for interactive debug without GPU charges.
# Set OPENRESEARCH_SDAR_NO_AUTOSTOP=1 to skip shutdown (leave VM running).
self_stop() {
  local reason="$1"
  echo "[sdar_gcp_run] self_stop triggered: $reason"
  if [[ "${OPENRESEARCH_SDAR_NO_AUTOSTOP:-0}" == "1" ]]; then
    echo "[sdar_gcp_run] autostop disabled (OPENRESEARCH_SDAR_NO_AUTOSTOP=1); leaving VM running for debug"
    return 0
  fi
  if [[ -n "${OPENRESEARCH_SDAR_REPORT_GCS:-}" ]]; then
    echo "[sdar_gcp_run] uploading run artifacts to $OPENRESEARCH_SDAR_REPORT_GCS/$PROJECT_ID/ ..."
    gsutil -m cp \
      "runs/$PROJECT_ID/final_report.json" \
      "runs/$PROJECT_ID/final_report.md" \
      "runs/sdar_gcp_run.out" \
      "$OPENRESEARCH_SDAR_REPORT_GCS/$PROJECT_ID/" 2>/dev/null || true
  fi
  echo "[sdar_gcp_run] halting GPU billing via shutdown; boot disk persists; flip to CPU machine type to debug without GPU charges"
  sync
  sudo shutdown -h now || sudo poweroff || true
}

# Run under timeout as an outer backstop in case the orchestrator wedges past its
# own --max-wall-clock limit. set +e so the non-zero rc (or rc=124 on timeout)
# does not abort the script before self_stop runs.
set +e
timeout --signal=TERM --kill-after=180 "$OPENRESEARCH_SDAR_OUTER_WALL_S" \
  env -u ANTHROPIC_API_KEY .venv/bin/python -m backend.cli reproduce 2605.15155 \
    --mode rlm --sandbox local --model "$OPENRESEARCH_SDAR_ROOT" \
    --models "$OPENRESEARCH_SDAR_MODELS" \
    --paper-hint 2605.15155 \
    --gpu-mode max --gpu-parallelism multi --vram-gb 40 \
    --no-force-single-gpu --max-wall-clock 86400 \
    --project-id "$PROJECT_ID"
rc=$?
set -e

if [ "$rc" -eq 124 ]; then
  self_stop "outer_wall_timeout rc=124"
elif [ "$rc" -ne 0 ]; then
  self_stop "error rc=$rc"
else
  self_stop "success rc=0"
fi
exit "$rc"
