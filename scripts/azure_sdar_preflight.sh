#!/usr/bin/env bash
# azure_sdar_preflight.sh — read-only green/red gate for an SDAR-on-Azure run.
# Exits non-zero if ANY hard check fails. Run before azure_sdar_run.sh.
#
# Usage: scripts/azure_sdar_preflight.sh [env-file=.env.azure]
set -uo pipefail

ENV_FILE="${1:-.env.azure}"
[[ -f "$ENV_FILE" ]] && set -a && . "$ENV_FILE" && set +a

RG="${OPENRESEARCH_AZURE_RESOURCE_GROUP:-rg-sciartgen-external}"
CLUSTER="${OPENRESEARCH_AZURE_AKS_CLUSTER:-sciart-aks}"
NS="${OPENRESEARCH_AZURE_NAMESPACE:-reprolab}"
SA="${OPENRESEARCH_AZURE_SERVICE_ACCOUNT:-reprolab-sa}"
SKU="${OPENRESEARCH_AZURE_GPU_SKUS:-azure_a100_80}"; SKU="${SKU//[\[\]\"]/}"; SKU="${SKU%%,*}"
REGION="${OPENRESEARCH_AZURE_REGION:-westus3}"
QUOTA_FAMILY="Standard NCADS_A100_v4 Family vCPUs"
QUOTA_MIN=24

fails=0; warns=0
ok()   { printf '  \033[32m[OK]\033[0m   %s\n' "$1"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$1"; fails=$((fails+1)); }
warn() { printf '  \033[33m[WARN]\033[0m %s\n' "$1"; warns=$((warns+1)); }
have() { command -v "$1" >/dev/null 2>&1; }

echo "== SDAR-on-Azure preflight (rg=$RG cluster=$CLUSTER ns=$NS sku=$SKU) =="

# 1. tooling
for t in az kubectl jq; do have "$t" && ok "tool: $t" || bad "tool missing: $t"; done
have kubelogin || warn "kubelogin missing (needed for disableLocalAccounts clusters): az aks install-cli"

# 2. az login + subscription
if az account show >/dev/null 2>&1; then ok "az logged in (sub $(az account show --query id -o tsv))"
else bad "not logged in — run: az login --use-device-code"; fi

# 3. cluster reachable
if kubectl get nodes >/dev/null 2>&1; then ok "kubectl reaches $CLUSTER ($(kubectl get nodes --no-headers 2>/dev/null | wc -l) node(s))"
else bad "kubectl cannot reach cluster — run: az aks get-credentials -g $RG -n $CLUSTER && kubelogin convert-kubeconfig -l azurecli"; fi

# 4. GPU quota (the #1 blocker; was 0/0)
LIMIT=$(az vm list-usage -l "$REGION" --query "[?localName=='$QUOTA_FAMILY'].limit | [0]" -o tsv 2>/dev/null || echo "")
if [[ -z "$LIMIT" ]]; then warn "could not read $QUOTA_FAMILY quota (need Reader on the sub)"
elif ! [[ "${LIMIT%.*}" =~ ^[0-9]+$ ]]; then warn "non-numeric quota value for $QUOTA_FAMILY: '$LIMIT'"
elif (( ${LIMIT%.*} >= QUOTA_MIN )); then ok "GPU quota: $QUOTA_FAMILY limit=$LIMIT (>= $QUOTA_MIN)"
else bad "GPU quota too low: $QUOTA_FAMILY limit=$LIMIT (< $QUOTA_MIN) — open a quota ticket"; fi

# 5. operator AKS-RBAC (can submit Jobs)
if kubectl auth can-i create jobs -n "$NS" >/dev/null 2>&1; then ok "RBAC: can create jobs in $NS"
else bad "RBAC: cannot create jobs in $NS — assign 'Azure Kubernetes Service RBAC Cluster Admin' on $CLUSTER"; fi

# 6. namespace + SA (Helm bootstrap ran)
if kubectl get ns "$NS" >/dev/null 2>&1; then ok "namespace $NS exists"
else bad "namespace $NS missing — run scripts/azure_sdar_bootstrap_cluster.sh"; fi
if kubectl get sa "$SA" -n "$NS" >/dev/null 2>&1; then
  cid=$(kubectl get sa "$SA" -n "$NS" -o jsonpath='{.metadata.annotations.azure\.workload\.identity/client-id}' 2>/dev/null)
  [[ -n "$cid" ]] && ok "SA $SA has workload-identity client-id" || bad "SA $SA missing workload-identity annotation"
else bad "SA $SA missing in $NS — run scripts/azure_sdar_bootstrap_cluster.sh"; fi

# 7. GPU node-pool label exists on a (scale-to-zero) pool
if az aks nodepool list -g "$RG" --cluster-name "$CLUSTER" --query "[?nodeLabels.\"reprolab/sku\"=='$SKU'] | [0].name" -o tsv 2>/dev/null | grep -q .; then
  ok "GPU pool with label reprolab/sku=$SKU present"
else bad "no node pool labelled reprolab/sku=$SKU — redeploy the Bicep stack"; fi

# 8. ACR + pinned cell image tag
if [[ -n "${OPENRESEARCH_AZURE_BASE_IMAGE:-}" ]]; then
  # Derive the ACR short-name from the image ref so the check works even when
  # OPENRESEARCH_AZURE_ACR_LOGIN_SERVER is unset (avoids an unbound-var crash).
  _acr_host="${OPENRESEARCH_AZURE_BASE_IMAGE%%/*}"                  # e.g. sciartacr.azurecr.io
  acr="${OPENRESEARCH_AZURE_ACR_LOGIN_SERVER:-$_acr_host}"; acr="${acr%%.*}"
  repo="${OPENRESEARCH_AZURE_BASE_IMAGE#*/}"; repo="${repo%%:*}"; tag="${OPENRESEARCH_AZURE_BASE_IMAGE##*:}"
  if az acr repository show-tags -n "$acr" --repository "$repo" -o tsv 2>/dev/null | grep -qx "$tag"; then ok "cell image present: $OPENRESEARCH_AZURE_BASE_IMAGE"
  else bad "cell image tag missing in ACR: $OPENRESEARCH_AZURE_BASE_IMAGE — run scripts/azure_build_cell_image.sh"; fi
else bad "OPENRESEARCH_AZURE_BASE_IMAGE unset — build + pin the cell image"; fi

# 9. storage account + blob container (the cell artifact bus)
if [[ -n "${OPENRESEARCH_AZURE_STORAGE_ACCOUNT:-}" ]] && az storage account show -n "$OPENRESEARCH_AZURE_STORAGE_ACCOUNT" -g "$RG" >/dev/null 2>&1; then
  ok "storage account $OPENRESEARCH_AZURE_STORAGE_ACCOUNT exists"
  CONTAINER="${OPENRESEARCH_AZURE_BLOB_CONTAINER:-reprolab-artifacts}"
  if az storage container show --account-name "$OPENRESEARCH_AZURE_STORAGE_ACCOUNT" --name "$CONTAINER" --auth-mode login >/dev/null 2>&1; then
    ok "blob container $CONTAINER exists"
    # Data-plane probe: verify the LOCAL az login identity has Storage Blob Data Contributor
    # (DefaultAzureCredential→AzureCliCredential is what azure_sdar_run.sh uses for code
    # upload and metrics download; a missing role causes 403 BEFORE any Job submits).
    if az storage blob list \
         --account-name "$OPENRESEARCH_AZURE_STORAGE_ACCOUNT" \
         --container-name "$CONTAINER" \
         --auth-mode login \
         --num-results 1 \
         -o none 2>/dev/null; then
      ok "blob data-plane access OK as current identity"
    else
      bad "az login identity lacks Storage Blob Data Contributor on $CONTAINER — local code-upload/metrics-download will 403; grant: az role assignment create --assignee <your-objectId> --role 'Storage Blob Data Contributor' --scope <storage-account-resource-id>"
    fi
    # NOTE: the cell Pods' workload MI is a SEPARATE identity from your az login
    # identity and also needs Storage Blob Data Contributor on this container.
    warn "verify that the cell workload MI (not your az login identity) also has Storage Blob Data Contributor on $CONTAINER — the check above proves only your local identity; see the runbook §7 Blob-403 row"
  else bad "blob container $CONTAINER not found or not reachable — confirm it was created by the Bicep stack"; fi
else bad "storage account ${OPENRESEARCH_AZURE_STORAGE_ACCOUNT:-<unset>} not found"; fi

# 10. python deps in the venv
if .venv/bin/python -c "import kubernetes, azure.identity, azure.storage.blob" 2>/dev/null; then ok "python deps: kubernetes, azure-identity, azure-storage-blob"
else bad "python deps missing — pip install -r backend/requirements.txt into .venv"; fi

# 11. claude-oauth root surface alive
if have claude && claude --print "ping" >/dev/null 2>&1; then ok "claude-oauth root surface responds"
else warn "claude --print ping failed — confirm 'claude login' (root model is claude-oauth)"; fi

echo "== preflight: $fails fail(s), $warns warn(s) =="
(( fails == 0 )) || { echo "RED — resolve the [FAIL] items above before running."; exit 1; }
echo "GREEN — safe to run scripts/azure_sdar_run.sh"
