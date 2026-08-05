#!/bin/bash
# launch_sdar.sh — canonical SDAR campaign launcher (rewritten 2026-08-03).
#
# WHAT CHANGED vs the July launcher, and why:
#   - `--sandbox gcp` is GONE: GKE is NOT USED — `--sandbox gcp/gke` fail-closes
#     (`_backend_for_sandbox_mode` raises; the two IAM grants were never applied).
#     The supported GCP GPU path is the single-VM `VmComputeProvider`, reached via
#     the campaign with `--sandbox local --billing-sandbox gcp`.
#     See docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md (canonical recipe).
#   - Preflights STAY ON: the old launcher set OPENRESEARCH_MODEL_PREFLIGHT=0 and
#     OPENRESEARCH_SKIP_CRED_PREFLIGHT=1, which hid a dead GCP credential (401)
#     until AFTER code generation in the July 6 run. This script refuses to run
#     with either preflight disabled.
#   - grok root/executor is GONE: grok is NOT a validated executor — it does not
#     emit the harness's commands.json manifest, so runs verdict `failed` on
#     evidence (2026-07-22 Adam run). Canonical models: opus-foundry root +
#     sonnet-foundry executor/grader/verifier (API keys only — NEVER OAuth,
#     operator directive 2026-08-01).
#   - VM identity must be explicit: VmComputeProvider's built-in defaults point
#     at a RETIRED VM (`sdar-a100-od` / SSH user `abheekp`). The four
#     OPENRESEARCH_GCP_* identity vars are REQUIRED below.
#   - All three campaign money caps are required by the CLI; defaults here are
#     overridable via SDAR_MAX_LLM_USD / SDAR_MAX_GPU_USD / SDAR_MAX_GPU_HOURS.
#
# After every run: `gcloud compute instances list` — confirm no stray VM billing.

set -euo pipefail
# Run from the repo root the script lives in — never a hardcoded machine path.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

# ── Refuse to run with preflights disabled (inherited from the shell) ────────
# A disabled preflight is exactly how the July 6 run burned an LLM budget on a
# dead GCP credential. Fail loud instead of silently overriding, so the
# operator removes the stale export rather than re-learning why it exists.
_mp="$(echo "${OPENRESEARCH_MODEL_PREFLIGHT:-}" | tr '[:upper:]' '[:lower:]')"
_sc="$(echo "${OPENRESEARCH_SKIP_CRED_PREFLIGHT:-}" | tr '[:upper:]' '[:lower:]')"
case "${_mp}" in
  0|false|no|off)
    echo "ERROR: OPENRESEARCH_MODEL_PREFLIGHT=${OPENRESEARCH_MODEL_PREFLIGHT} disables the model preflight." >&2
    echo "       Preflights must stay ON (they catch dead credentials BEFORE code generation)." >&2
    echo "       Remove the export and re-run." >&2
    exit 1 ;;
esac
case "${_sc}" in
  1|true|yes|on)
    echo "ERROR: OPENRESEARCH_SKIP_CRED_PREFLIGHT=${OPENRESEARCH_SKIP_CRED_PREFLIGHT} disables the credential preflight." >&2
    echo "       Preflights must stay ON (a 401 must surface before provisioning, not after)." >&2
    echo "       Remove the export and re-run." >&2
    exit 1 ;;
esac

# ── Load .env (strip Windows CRLF) ───────────────────────────────────────────
if [[ -f .env ]]; then
  while IFS='=' read -r key value; do
    key=$(echo "$key" | tr -d '\r')
    value=$(echo "$value" | tr -d '\r')
    [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
    export "$key=$value" 2>/dev/null || true
  done < .env
fi

# Preflights must stay ON no matter what .env carried — unset both explicitly
# so the harness defaults (preflights enabled) apply.
unset OPENRESEARCH_MODEL_PREFLIGHT
unset OPENRESEARCH_SKIP_CRED_PREFLIGHT

# Unset the stale OpenAI key so it doesn't hijack root-model selection
# (shell/.env exports shadow explicit --model resolution; the .env key is dead
# anyway — rotated in the 2026-07-21 leak audit).
unset OPENAI_API_KEY

# ── Required fresh-VM identity (VmComputeProvider defaults are RETIRED) ──────
# Per docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md: derive the SSH user
# from `gcloud compute os-login describe-profile`; do NOT reuse `abheekp`.
_missing=0
for var in OPENRESEARCH_GCP_PROJECT OPENRESEARCH_GCP_ZONE \
           OPENRESEARCH_GCP_INSTANCE OPENRESEARCH_GCP_SSH_USER; do
  if [[ -z "${!var:-}" ]]; then
    echo "ERROR: ${var} is not set." >&2
    _missing=1
  fi
done
if [[ "${_missing}" -eq 1 ]]; then
  cat >&2 <<'EOF'
The VmComputeProvider built-in defaults point at a RETIRED VM (sdar-a100-od /
SSH user abheekp). Set the fresh-VM identity explicitly, e.g.:

  export OPENRESEARCH_GCP_PROJECT=deepinvent-ext-ut
  export OPENRESEARCH_GCP_ZONE=us-central1-b
  export OPENRESEARCH_GCP_INSTANCE=<fresh-vm-name>
  export OPENRESEARCH_GCP_SSH_USER=<your-oslogin-user>   # gcloud compute os-login describe-profile --format='value(posixAccounts[0].username)'
  export OPENRESEARCH_REMOTE_DIR=/home/<your-oslogin-user>/openresearch   # recommended

See docs/runbooks/2026-07-22-gcp-vm-e2e-run-procedure.md.
EOF
  exit 1
fi

# Machine type: unset falls back to the EXPENSIVE 4xA100-80 default
# (a2-highgpu-4g, ~$14/hr). Warn so a smoke run doesn't bill 4 A100s.
if [[ -z "${OPENRESEARCH_GCP_GPU_MACHINE_TYPE:-}" ]]; then
  echo "WARNING: OPENRESEARCH_GCP_GPU_MACHINE_TYPE is unset — the campaign VM" >&2
  echo "         defaults to a2-highgpu-4g (4xA100-80, ~\$14/hr)." >&2
  echo "         For the SDAR canonical fidelity run use a2-ultragpu-1g (1xA100-80);" >&2
  echo "         for a cheap plumbing smoke use g2-standard-8 (1xL4, ~\$0.70/hr)." >&2
fi

# ── Money caps (all three are CLI-required; override via env) ────────────────
SDAR_MAX_LLM_USD="${SDAR_MAX_LLM_USD:-25}"
SDAR_MAX_GPU_USD="${SDAR_MAX_GPU_USD:-50}"
SDAR_MAX_GPU_HOURS="${SDAR_MAX_GPU_HOURS:-8}"

# ── Python: prefer the WSL venv, fall back to the repo venv ──────────────────
if [[ -x .venv-wsl/bin/python ]]; then
  PY=.venv-wsl/bin/python
elif [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
else
  PY=python3
fi

echo "Launching SDAR campaign: caps llm=\$${SDAR_MAX_LLM_USD} gpu=\$${SDAR_MAX_GPU_USD} gpu-hours=${SDAR_MAX_GPU_HOURS}"
echo "VM identity: ${OPENRESEARCH_GCP_INSTANCE} (${OPENRESEARCH_GCP_PROJECT}/${OPENRESEARCH_GCP_ZONE}, ssh=${OPENRESEARCH_GCP_SSH_USER})"

# Per-role models: the campaign subparser has NO --model/--models flags
# (those belong to `reproduce`); campaign takes --root-model and reads roles
# from OPENRESEARCH_ROLE_MODELS (JSON or role=token,... form — role_models.py).
export OPENRESEARCH_ROLE_MODELS="executor=sonnet-foundry,grader=sonnet-foundry,verifier=sonnet-foundry"

# Paper source: an operator-staged local PDF when present (dev machines/VMs
# keep an untracked sdar.pdf at the repo root — it is not committed), else
# the bare arXiv id, which ingestion resolves and fetches itself.
SDAR_PAPER="${SDAR_PAPER:-sdar.pdf}"
if [[ ! -f "${SDAR_PAPER}" ]]; then
  echo "No local ${SDAR_PAPER} — falling back to arXiv id 2605.15155 (ingester fetches it)"
  SDAR_PAPER="2605.15155"
fi

# Canonical campaign command shape (2026-07-22 runbook): LIVE driver (the
# unified driver does not forward enforcement env to child processes —
# 2026-08-04 Tree-B root cause), LOCAL run sandbox + GCP BILLING sandbox
# (routes provisioning through VmComputeProvider; `--sandbox gcp` would hit
# the fail-loud GKE guard).
exec "${PY}" -u -m backend.cli campaign "${SDAR_PAPER}" \
  --campaign-driver live \
  --sandbox local \
  --billing-sandbox gcp \
  --paper-hint 2605.15155 \
  --root-model opus-foundry \
  --max-llm-usd "${SDAR_MAX_LLM_USD}" \
  --max-gpu-usd "${SDAR_MAX_GPU_USD}" \
  --max-gpu-hours "${SDAR_MAX_GPU_HOURS}"
