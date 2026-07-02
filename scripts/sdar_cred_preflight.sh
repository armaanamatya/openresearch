#!/usr/bin/env bash
# Credential preflight helper for SDAR-on-GCP launchers.
#
# Provides two functions (safe to source from any launcher):
#   preflight_root_credential <token>  — exits 1 (fail-fast) on a dead API key
#   preflight_subagent_oauth           — warns if no OAuth/API credential for sonnet
#
# Standalone usage:
#   scripts/sdar_cred_preflight.sh --self-check
#       Checks foundry, openai, and oauth against the current .env; no provisioning.
#   scripts/sdar_cred_preflight.sh preflight <root_token>
#       Runs both checks for the given root token.
#
# Source from another script:
#   source "$(dirname "$0")/sdar_cred_preflight.sh"
#   preflight_root_credential foundry   # exits 1 on dead credential
#   preflight_subagent_oauth            # warns (never exits 1)
set -uo pipefail

# Locate the repo root relative to this file (works when sourced or executed).
_PREFLIGHT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PREFLIGHT_REPO_DIR="$(cd "${_PREFLIGHT_SCRIPT_DIR}/.." && pwd)"
_PREFLIGHT_ENV="${_PREFLIGHT_REPO_DIR}/.env"

# _load_env_var <name>: read one variable from .env without leaking
# every .env key into the calling shell's environment.
_load_env_var() {
  local _var="$1"
  # set +u: .env files regularly source unset vars; don't abort the subshell.
  (set +u; set -a; [ -f "$_PREFLIGHT_ENV" ] && . "$_PREFLIGHT_ENV" >/dev/null 2>&1
   printf '%s' "${!_var:-}")
}

# ---------------------------------------------------------------------------
# preflight_root_credential <token>
#   Validates the credential for the given root model token.
#   Exits 1 on a dead/missing credential for API-keyed roots (fail-fast).
#   Warns and exits 0 for oauth-root or unknown tokens.
# ---------------------------------------------------------------------------
preflight_root_credential() {
  local token="${1:-foundry}"
  case "$token" in
    foundry|grok|azure-foundry|grok-4.3)
      _preflight_foundry
      ;;
    gpt-5|openai|gpt-4o)
      _preflight_openai
      ;;
    claude|anthropic)
      _preflight_anthropic_api
      ;;
    claude-oauth|anthropic-oauth)
      _preflight_claude_oauth
      ;;
    *)
      echo "[preflight] WARN: unknown root token '$token' — skipping credential check (novel root; proceed manually)"
      ;;
  esac
}

# ---------------------------------------------------------------------------
# preflight_subagent_oauth
#   Warns (never exits 1) when no sub-agent credential is available.
#   Sub-agents (executor=sonnet, grader=sonnet, verifier=sonnet) require either
#   ~/.claude/.credentials.json, CLAUDE_CODE_OAUTH_TOKEN, or ANTHROPIC_API_KEY.
# ---------------------------------------------------------------------------
preflight_subagent_oauth() {
  local has_oauth=0 has_api=0
  local tok="${CLAUDE_CODE_OAUTH_TOKEN:-$(_load_env_var CLAUDE_CODE_OAUTH_TOKEN)}"
  local key="${ANTHROPIC_API_KEY:-$(_load_env_var ANTHROPIC_API_KEY)}"
  [ -f "$HOME/.claude/.credentials.json" ] && has_oauth=1
  [ -n "$tok" ] && has_oauth=1
  [ -n "$key" ] && has_api=1

  if [[ "$has_oauth" == "0" && "$has_api" == "0" ]]; then
    echo "[preflight] WARN: sub-agent OAuth: neither ~/.claude/.credentials.json, CLAUDE_CODE_OAUTH_TOKEN,"
    echo "  nor ANTHROPIC_API_KEY is available — executor=sonnet/grader=sonnet/verifier=sonnet will fail at runtime."
    echo "  FIX: run 'claude login' to create ~/.claude/.credentials.json, or set CLAUDE_CODE_OAUTH_TOKEN in .env"
  elif [[ "$has_oauth" == "1" ]]; then
    echo "[preflight] sub-agent OAuth: credential present (sonnet executor/grader/verifier will authenticate)"
  else
    echo "[preflight] sub-agent auth: ANTHROPIC_API_KEY present (API-key path for sonnet sub-agents)"
  fi
}

# --- Private checkers -------------------------------------------------------

_preflight_foundry() {
  local endpoint key deployment
  endpoint="${AZURE_FOUNDRY_ENDPOINT:-$(_load_env_var AZURE_FOUNDRY_ENDPOINT)}"
  key="${AZURE_FOUNDRY_API_KEY:-$(_load_env_var AZURE_FOUNDRY_API_KEY)}"
  deployment="${AZURE_FOUNDRY_DEPLOYMENT:-$(_load_env_var AZURE_FOUNDRY_DEPLOYMENT)}"

  if [ -z "$endpoint" ] || [ -z "$key" ] || [ -z "$deployment" ]; then
    echo "[preflight] FAIL: foundry root — AZURE_FOUNDRY_ENDPOINT, AZURE_FOUNDRY_API_KEY, and AZURE_FOUNDRY_DEPLOYMENT must all be set" >&2
    echo "  Set these in .env and retry, or use ROOT=claude-oauth (unreliable)" >&2
    exit 1
  fi

  # Normalize endpoint: strip trailing /chat/completions and trailing slashes,
  # then append /chat/completions for the ping.
  local base="${endpoint%/chat/completions}"
  base="${base%/}"
  local url="${base}/chat/completions"
  local body="{\"model\":\"${deployment}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":5}"

  echo "[preflight] foundry: pinging deployment=${deployment} ..."
  local http_code
  http_code="$(curl -s -o /dev/null -w '%{http_code}' \
    --max-time 25 \
    -X POST "$url" \
    -H "Authorization: Bearer ${key}" \
    -H "Content-Type: application/json" \
    -d "$body" 2>/dev/null || echo "000")"

  if [ "$http_code" = "200" ]; then
    echo "[preflight] foundry: PASS / LIVE (HTTP 200 — credential valid)"
  else
    echo "[preflight] FAIL: foundry root returned HTTP ${http_code:-000} — credential dead, endpoint unreachable, or deployment mismatch" >&2
    echo "  deployment: ${deployment}" >&2
    echo "  endpoint:   ${base}" >&2
    echo "  FIX: verify AZURE_FOUNDRY_ENDPOINT, AZURE_FOUNDRY_API_KEY, AZURE_FOUNDRY_DEPLOYMENT in .env" >&2
    exit 1
  fi
}

_preflight_openai() {
  local key
  key="${OPENAI_API_KEY:-$(_load_env_var OPENAI_API_KEY)}"

  if [ -z "$key" ]; then
    echo "[preflight] FAIL: gpt-5/openai root — OPENAI_API_KEY not set" >&2
    echo "  FIX: set a live OPENAI_API_KEY in .env, or use ROOT=foundry" >&2
    exit 1
  fi

  echo "[preflight] openai: probing api.openai.com/v1/models (key: ***${key: -4}) ..."
  local http_code
  http_code="$(curl -s -o /dev/null -w '%{http_code}' \
    --max-time 25 \
    -H "Authorization: Bearer ${key}" \
    "https://api.openai.com/v1/models" 2>/dev/null || echo "000")"

  if [ "$http_code" = "200" ]; then
    echo "[preflight] openai: PASS / LIVE (HTTP 200 — API key valid)"
  else
    echo "[preflight] FAIL: OPENAI_API_KEY is dead/unfunded (HTTP ${http_code:-000})" >&2
    echo "  This is likely the known revoked sk-svcac... key — set a live key or use ROOT=foundry" >&2
    exit 1
  fi
}

_preflight_anthropic_api() {
  local key
  key="${ANTHROPIC_API_KEY:-$(_load_env_var ANTHROPIC_API_KEY)}"

  if [ -z "$key" ]; then
    echo "[preflight] FAIL: claude/anthropic root — ANTHROPIC_API_KEY not set" >&2
    echo "  FIX: set ANTHROPIC_API_KEY in .env, or use ROOT=foundry (or ROOT=claude-oauth)" >&2
    exit 1
  fi
  # Skip a live ping to avoid burning credits on a preflight check.
  echo "[preflight] claude/anthropic: PASS — ANTHROPIC_API_KEY present (key: ***${key: -4})"
  echo "[preflight]   NOTE: key presence confirmed; credit balance not verified (ping skipped to avoid cost)"
}

_preflight_claude_oauth() {
  local has=0
  local tok="${CLAUDE_CODE_OAUTH_TOKEN:-$(_load_env_var CLAUDE_CODE_OAUTH_TOKEN)}"
  [ -f "$HOME/.claude/.credentials.json" ] && has=1
  [ -n "$tok" ] && has=1

  if [ "$has" = "0" ]; then
    echo "[preflight] FAIL: claude-oauth root — neither ~/.claude/.credentials.json nor CLAUDE_CODE_OAUTH_TOKEN found" >&2
    echo "  FIX: run 'claude login' to create credentials, then retry; or use ROOT=foundry" >&2
    exit 1
  fi

  echo "[preflight] claude-oauth: PASS / PRESENT — OAuth credential found"
  echo "[preflight] WARN: claude-oauth as root is UNRELIABLE — degenerate refusal loops are common."
  echo "  RECOMMENDATION: use ROOT=foundry (grok-4.3 via Azure Foundry) for reliable root-model execution."
}

# ---------------------------------------------------------------------------
# Self-check: validates all three credential types against the current .env.
# No provisioning, no money spent. Run before launching to confirm readiness.
# ---------------------------------------------------------------------------
_self_check() {
  local env_label="${_PREFLIGHT_ENV}"
  echo "=== SDAR credential self-check (no provisioning) ==="
  echo "    .env: ${env_label}"
  echo ""

  # Read all relevant vars from .env in subshells (never leak to shell env).
  local foundry_ep foundry_key foundry_dep openai_key anthropic_key oauth_tok
  foundry_ep="$(_load_env_var AZURE_FOUNDRY_ENDPOINT)"
  foundry_key="$(_load_env_var AZURE_FOUNDRY_API_KEY)"
  foundry_dep="$(_load_env_var AZURE_FOUNDRY_DEPLOYMENT)"
  openai_key="$(_load_env_var OPENAI_API_KEY)"
  anthropic_key="$(_load_env_var ANTHROPIC_API_KEY)"
  oauth_tok="$(_load_env_var CLAUDE_CODE_OAUTH_TOKEN)"

  # 1. Foundry/grok root
  echo "--- [1] foundry/grok root  (ROOT=foundry) ---"
  if [ -z "$foundry_ep" ] || [ -z "$foundry_key" ] || [ -z "$foundry_dep" ]; then
    echo "  AZURE_FOUNDRY_*: one or more vars missing"
    echo "  result: FAIL / MISSING"
  else
    local base="${foundry_ep%/chat/completions}"
    base="${base%/}"
    local url="${base}/chat/completions"
    local body="{\"model\":\"${foundry_dep}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":5}"
    echo "  deployment: ${foundry_dep}   key: ***${foundry_key: -4}"
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 \
      -X POST "$url" \
      -H "Authorization: Bearer ${foundry_key}" \
      -H "Content-Type: application/json" \
      -d "$body" 2>/dev/null || echo "000")"
    if [ "$code" = "200" ]; then
      echo "  result: PASS / LIVE  (HTTP 200)"
    else
      echo "  result: FAIL / DEAD  (HTTP ${code:-000})"
    fi
  fi

  echo ""

  # 2. OpenAI/gpt-5 root
  echo "--- [2] openai/gpt-5 root  (ROOT=gpt-5) ---"
  if [ -z "$openai_key" ]; then
    echo "  OPENAI_API_KEY: not set"
    echo "  result: FAIL / MISSING"
  else
    echo "  key: ***${openai_key: -4}"
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 \
      -H "Authorization: Bearer ${openai_key}" \
      "https://api.openai.com/v1/models" 2>/dev/null || echo "000")"
    if [ "$code" = "200" ]; then
      echo "  result: PASS / LIVE  (HTTP 200)"
    else
      echo "  result: FAIL / DEAD  (HTTP ${code:-000}) — likely the revoked sk-svcac... key; use ROOT=foundry"
    fi
  fi

  echo ""

  # 3. Claude OAuth (sub-agent + optional root)
  echo "--- [3] claude-oauth  (sub-agents=sonnet; ROOT=claude-oauth if used) ---"
  local has_creds_file=0 has_token=0
  [ -f "$HOME/.claude/.credentials.json" ] && has_creds_file=1
  [ -n "$oauth_tok" ] && has_token=1
  if [ "$has_creds_file" = "1" ]; then
    echo "  ~/.claude/.credentials.json: present"
  fi
  if [ "$has_token" = "1" ]; then
    echo "  CLAUDE_CODE_OAUTH_TOKEN: present (in .env)"
  fi
  if [ "$has_creds_file" = "0" ] && [ "$has_token" = "0" ]; then
    echo "  result: FAIL / MISSING — neither credentials file nor token found"
    echo "  FIX: run 'claude login' or set CLAUDE_CODE_OAUTH_TOKEN in .env"
  else
    echo "  result: PASS / PRESENT"
    echo "  WARN: claude-oauth as ROOT is unreliable (degenerate loops); use ROOT=foundry"
  fi

  echo ""

  # 4. Anthropic API key (informational)
  echo "--- [4] ANTHROPIC_API_KEY (informational) ---"
  if [ -n "$anthropic_key" ]; then
    echo "  result: PRESENT (***${anthropic_key: -4})"
  else
    echo "  result: MISSING (not set in .env)"
  fi

  echo ""
  echo "=== Recommendation: ROOT=foundry — grok-4.3 live + funded ==="
}

# ---------------------------------------------------------------------------
# Standalone entry point (only runs when script is executed, not sourced).
# ---------------------------------------------------------------------------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  case "${1:-}" in
    --self-check|self-check)
      _self_check
      ;;
    preflight)
      _root="${2:-foundry}"
      echo "[preflight] running credential check for root='${_root}' ..."
      preflight_root_credential "$_root"
      echo ""
      preflight_subagent_oauth
      echo "[preflight] done"
      ;;
    *)
      echo "Usage:" >&2
      echo "  $0 --self-check              # validate all credentials; no provisioning" >&2
      echo "  $0 preflight <root_token>    # check a specific root (foundry, gpt-5, claude-oauth, ...)" >&2
      echo "" >&2
      echo "Source this file in a launcher to call preflight_root_credential() directly." >&2
      exit 2
      ;;
  esac
fi
