#!/usr/bin/env bash
# azure_sdar_bootstrap_cluster.sh — one-time, idempotent cluster scaffold for
# the local-orchestrator SDAR path: namespace + cell SA (workload identity) +
# NVIDIA device plugin + RBAC + quota. NO orchestrator, NO KeyVault, NO Files.
# Requires the AKS RBAC Cluster Admin grant.
#
# Usage: scripts/azure_sdar_bootstrap_cluster.sh [env-file=.env.azure]
set -euo pipefail

ENV_FILE="${1:-.env.azure}"
[[ -f "$ENV_FILE" ]] && set -a && . "$ENV_FILE" && set +a

RG="${OPENRESEARCH_AZURE_RESOURCE_GROUP:-rg-sciartgen-external}"
WI_MI_NAME="${OPENRESEARCH_AZURE_WORKLOAD_MI_NAME:-sciart-workload-mi}"
RELEASE="reprolab-aks"
CHART="infra/azure/helm"

command -v helm >/dev/null || { echo "helm not installed: https://helm.sh/docs/intro/install/"; exit 1; }

echo "Resolving workload-identity client id from MI '$WI_MI_NAME' in $RG ..."
WI_CLIENT_ID="$(az identity show -g "$RG" -n "$WI_MI_NAME" --query clientId -o tsv)"
[[ -n "$WI_CLIENT_ID" ]] || { echo "could not resolve clientId for $WI_MI_NAME"; exit 1; }

echo "helm upgrade --install $RELEASE (orchestrator OFF, filesCache OFF) ..."
helm upgrade --install "$RELEASE" "$CHART" \
  --set orchestrator.enabled=false \
  --set storage.filesCache.enabled=false \
  --set workloadIdentity.clientId="$WI_CLIENT_ID" \
  --wait --timeout 5m

echo "Done. Scaffolded namespace + cell SA + NVIDIA device plugin + RBAC + quota."
echo "Verify: kubectl get sa,ns,daemonset -A | grep -E 'reprolab|nvidia'"
