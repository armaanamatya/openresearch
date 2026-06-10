#!/usr/bin/env bash
# ─── One-time admin bootstrap — ReproLab Azure OIDC deploy path ───────────────
#
# Run once by the subscription Owner.  After this script succeeds, all deploys
# go through GitHub Actions OIDC — no standing human credentials are needed.
#
# Required environment variables (or export before running):
#   SUBSCRIPTION_ID   — Azure subscription GUID
#   TENANT_ID         — Entra tenant GUID
#   LOCATION          — Azure region (e.g. eastus)
#   GITHUB_ORG        — GitHub org or user that owns the repo (e.g. openresearch-ai)
#   GITHUB_REPO       — GitHub repo name (e.g. openresearch)
#   MAIN_RG_NAME      — Name of the main resource group (e.g. rg-reprolab)
#
# Optional:
#   OPERATOR_GROUP_OBJECT_ID  — Entra group object ID for the operator principal
#                               (same value as principalId in main.bicepparam).
#                               Leave unset to skip operator grants.
#   APP_NAME          — Display name for the app registration
#                       (default: openresearch-deployer)
#   GITHUB_ENVIRONMENT — GitHub environment name for the deploy protection rule
#                        (default: azure)
#
# Usage:
#   export SUBSCRIPTION_ID=... TENANT_ID=... LOCATION=... \
#          GITHUB_ORG=... GITHUB_REPO=... MAIN_RG_NAME=...
#   bash infra/azure/bicep/bootstrap/admin-bootstrap.sh

set -euo pipefail

# ─── Validate required variables ──────────────────────────────────────────────

_require() {
  local var="$1"
  local hint="${2:-}"
  if [[ -z "${!var:-}" ]]; then
    echo "ERROR: \$$var is not set." >&2
    [[ -n "$hint" ]] && echo "       $hint" >&2
    exit 1
  fi
}

_require SUBSCRIPTION_ID  "az account show --query id -o tsv"
_require TENANT_ID         "az account show --query tenantId -o tsv"
_require LOCATION          "e.g. eastus, westus3, northeurope"
_require GITHUB_ORG        "GitHub org or user that owns the repo"
_require GITHUB_REPO       "GitHub repository name (no org prefix)"
_require MAIN_RG_NAME      "Name of the main resource group (e.g. rg-reprolab)"

APP_NAME="${APP_NAME:-openresearch-deployer}"
GITHUB_ENVIRONMENT="${GITHUB_ENVIRONMENT:-azure}"

# Derive the tfstate RG name from the main RG name following the repo convention.
TFSTATE_RG_NAME="${MAIN_RG_NAME}-tfstate"

# Absolute path to main.bicep, relative to this script's directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_BICEP="${SCRIPT_DIR}/../main.bicep"

if [[ ! -f "$MAIN_BICEP" ]]; then
  echo "ERROR: main.bicep not found at $MAIN_BICEP" >&2
  exit 1
fi

echo ""
echo "=========================================================="
echo " ReproLab — one-time admin bootstrap"
echo "=========================================================="
echo " Subscription : $SUBSCRIPTION_ID"
echo " Tenant       : $TENANT_ID"
echo " Location     : $LOCATION"
echo " GitHub       : $GITHUB_ORG/$GITHUB_REPO  (environment: $GITHUB_ENVIRONMENT)"
echo " App name     : $APP_NAME"
echo " Main RG      : $MAIN_RG_NAME"
echo " Tfstate RG   : $TFSTATE_RG_NAME"
echo "=========================================================="
echo ""

# ─── Step 1: Set active subscription ──────────────────────────────────────────

echo ">>> Step 1: Setting active subscription to $SUBSCRIPTION_ID"
az account set --subscription "$SUBSCRIPTION_ID"
echo "    Done."
echo ""

# ─── Step 2: Create app registration + service principal (idempotent) ─────────

echo ">>> Step 2: Creating app registration '$APP_NAME' (idempotent)"

# Check for an existing app registration by display name.  Display names are
# NOT unique in Entra — abort on ambiguity rather than silently picking one.
APP_MATCH_COUNT="$(az ad app list --display-name "$APP_NAME" --query 'length(@)' -o tsv 2>/dev/null || echo 0)"
if [[ "$APP_MATCH_COUNT" -gt 1 ]]; then
  echo "ERROR: $APP_MATCH_COUNT app registrations share the display name '$APP_NAME'." >&2
  echo "       Delete the duplicates or set APP_NAME to a unique value, then re-run." >&2
  exit 1
fi
EXISTING_APP_ID="$(az ad app list --display-name "$APP_NAME" --query '[0].appId' -o tsv 2>/dev/null || true)"

if [[ -n "$EXISTING_APP_ID" && "$EXISTING_APP_ID" != "None" ]]; then
  echo "    App registration already exists: $EXISTING_APP_ID"
  APP_CLIENT_ID="$EXISTING_APP_ID"
else
  echo "    Creating new app registration..."
  APP_CLIENT_ID="$(az ad app create --display-name "$APP_NAME" --query appId -o tsv)"
  echo "    Created app: $APP_CLIENT_ID"
fi

# Ensure a service principal exists for the app (idempotent: show-or-create).
EXISTING_SP_ID="$(az ad sp show --id "$APP_CLIENT_ID" --query id -o tsv 2>/dev/null || true)"
if [[ -n "$EXISTING_SP_ID" && "$EXISTING_SP_ID" != "None" ]]; then
  echo "    Service principal already exists: $EXISTING_SP_ID"
  SP_OBJECT_ID="$EXISTING_SP_ID"
else
  echo "    Creating service principal..."
  SP_OBJECT_ID="$(az ad sp create --id "$APP_CLIENT_ID" --query id -o tsv)"
  echo "    Created service principal: $SP_OBJECT_ID"
fi

echo ""

# ─── Step 3: Add federated credential (idempotent upsert) ─────────────────────
#
# ONE subject only:
#   repo:<ORG>/<REPO>:environment:<ENVIRONMENT>  — workflow_dispatch deploys
#
# Deliberately NO pull_request subject: that claim binds neither actor nor
# workflow content, so any PR running a modified workflow could exchange it for
# this identity.  PR validation runs without Azure credentials.
#
# Idempotency: when the named credential exists it is UPDATED (not skipped) so
# a changed GITHUB_ORG/GITHUB_REPO/GITHUB_ENVIRONMENT never leaves a stale
# subject behind.

echo ">>> Step 3: Upserting federated credential"

_upsert_federated_credential() {
  local cred_name="$1"
  local subject="$2"
  local params
  params="{
    \"name\": \"${cred_name}\",
    \"issuer\": \"https://token.actions.githubusercontent.com\",
    \"subject\": \"${subject}\",
    \"audiences\": [\"api://AzureADTokenExchange\"],
    \"description\": \"${cred_name}\"
  }"

  local existing
  existing="$(az ad app federated-credential list --id "$APP_CLIENT_ID" \
    --query "[?name=='${cred_name}'].name" -o tsv 2>/dev/null || true)"

  if [[ -n "$existing" ]]; then
    echo "    Federated credential '$cred_name' exists — updating subject to: $subject"
    az ad app federated-credential update \
      --id "$APP_CLIENT_ID" \
      --federated-credential-id "$cred_name" \
      --parameters "$params" \
      --output none
    echo "    Updated."
  else
    echo "    Creating federated credential '$cred_name' (subject: $subject)"
    az ad app federated-credential create \
      --id "$APP_CLIENT_ID" \
      --parameters "$params" \
      --output none
    echo "    Created."
  fi
}

# Deploy credential: scoped to the named GitHub environment (approval-gated).
_upsert_federated_credential \
  "github-deploy-${GITHUB_ENVIRONMENT}" \
  "repo:${GITHUB_ORG}/${GITHUB_REPO}:environment:${GITHUB_ENVIRONMENT}"

echo ""

# ─── Step 4: What-if — admin reviews change-set before apply ──────────────────

echo ">>> Step 4: Running 'az deployment sub what-if' — please review the change-set."
echo ""

az deployment sub what-if \
  --location "$LOCATION" \
  --template-file "$MAIN_BICEP" \
  --parameters \
    location="$LOCATION" \
    mainRgName="$MAIN_RG_NAME" \
    tfstateRgName="$TFSTATE_RG_NAME" \
    principalId="${OPERATOR_GROUP_OBJECT_ID:-}" \
    deployPrincipalId="$SP_OBJECT_ID"
  # No error suppression: what-if exits 0 when it succeeds (creates included);
  # a non-zero exit is a real auth/template failure and must stop the script.

echo ""
echo "----------------------------------------------------------"
echo " REVIEW the what-if output above."
echo " It shows exactly which resource groups and role assignments"
echo " will be created or updated."
echo ""
read -r -p " Type 'yes' to proceed with the deployment, or anything else to abort: " CONFIRM
echo ""

if [[ "$CONFIRM" != "yes" ]]; then
  echo "Aborted by administrator review.  NOTE: steps 2-3 already created the"
  echo "app registration, service principal, and federated credential (Entra"
  echo "objects, idempotent, no Azure RBAC granted).  No ARM resources or role"
  echo "assignments were created.  Delete the app registration to fully revert:"
  echo "  az ad app delete --id $APP_CLIENT_ID"
  exit 0
fi

# ─── Step 5: Deploy L0 ────────────────────────────────────────────────────────

echo ">>> Step 5: Deploying L0 (az deployment sub create)"

DEPLOY_PARAMS=(
  location="$LOCATION"
  mainRgName="$MAIN_RG_NAME"
  tfstateRgName="$TFSTATE_RG_NAME"
  deployPrincipalId="$SP_OBJECT_ID"
)

# Only pass principalId when the operator group is set; otherwise omit it so
# Bicep uses its own default and skips the operator grant path.
if [[ -n "${OPERATOR_GROUP_OBJECT_ID:-}" ]]; then
  DEPLOY_PARAMS+=(principalId="$OPERATOR_GROUP_OBJECT_ID")
fi

az deployment sub create \
  --location "$LOCATION" \
  --template-file "$MAIN_BICEP" \
  --parameters "${DEPLOY_PARAMS[@]}"

echo "    Deployment complete."
echo ""

# ─── Step 6: Print the three GitHub repo variables ────────────────────────────

echo "=========================================================="
echo " Set the following as GitHub repository VARIABLES"
echo " (Settings → Secrets and variables → Actions → Variables)."
echo " These are NOT secrets — OIDC uses no client secret at all."
echo "=========================================================="
echo ""
echo "  AZURE_CLIENT_ID       = $APP_CLIENT_ID"
echo "  AZURE_TENANT_ID       = $TENANT_ID"
echo "  AZURE_SUBSCRIPTION_ID = $SUBSCRIPTION_ID"
echo ""
echo " Also create a GitHub environment named '$GITHUB_ENVIRONMENT'"
echo " (Settings → Environments) and add required reviewers to act"
echo " as the human approval gate for workflow_dispatch deploys."
echo ""
echo " No client secret was created.  OIDC is credential-free."
echo "=========================================================="
