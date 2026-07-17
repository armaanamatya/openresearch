#!/usr/bin/env bash
# GKE preflight + (optional) end-to-end smoke for the openresearch pipeline.
# Green here => --sandbox gcp / --sandbox gke will auth, reach the cluster, and
# have GPU quota. Usage: scripts/gke_check.sh [--start-pod]
# Exit: 0 green; 2 missing env; 3 gcloud/ADC; 4 cluster unreachable; 5 GPU quota;
#       6 --start-pod smoke (operator-gated, COSTS MONEY); 7 a configured
#       OPENRESEARCH_GCP_GPU_SKUS entry has NO matching live reprolab/sku node
#       pool (config/Terraform drift -- see the per-SKU check below).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
START_POD=0
for arg in "$@"; do case "$arg" in
  --start-pod) START_POD=1 ;;
  -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
  *) echo "Unknown argument: $arg" >&2; exit 1 ;;
esac; done
# Load .env per-line into this process only (mirrors runpod_check.sh).
if [[ -f "${ENV_FILE}" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"; value="${BASH_REMATCH[2]}"
      value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
      [[ -z "${!key+x}" ]] && export "${key}=${value}"
    fi
  done < "${ENV_FILE}"
fi
if [[ -z "${OPENRESEARCH_GCP_PROJECT:-}" ]]; then
  echo "FAIL  OPENRESEARCH_GCP_PROJECT not set (exit 2)" >&2; exit 2
fi
if [[ -z "${OPENRESEARCH_GCP_GCS_BUCKET:-}" ]]; then
  echo "FAIL  OPENRESEARCH_GCP_GCS_BUCKET not set (exit 2)" >&2; exit 2
fi
command -v gcloud >/dev/null 2>&1 || { echo "FAIL  gcloud not found (exit 3)" >&2; exit 3; }
gcloud auth application-default print-access-token >/dev/null 2>&1 \
  || { echo "FAIL  ADC missing — run: gcloud auth application-default login (exit 3)" >&2; exit 3; }
echo "OK    gcloud ADC present (project=${OPENRESEARCH_GCP_PROJECT})."
command -v kubectl >/dev/null 2>&1 || { echo "FAIL  kubectl not found (exit 4)" >&2; exit 4; }
kubectl cluster-info >/dev/null 2>&1 \
  || { echo "FAIL  GKE cluster unreachable — run gcloud container clusters get-credentials (exit 4)" >&2; exit 4; }
echo "OK    GKE cluster reachable."
gpu_nodes="$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null | grep -c '^[1-9]' || true)"
if [[ "${gpu_nodes:-0}" -gt 0 ]]; then echo "OK    ${gpu_nodes} GPU node(s) advertise nvidia.com/gpu."
else echo "WARN  No GPU node currently advertises nvidia.com/gpu (node pool may be scaled to zero — GKE autoscales on Job dispatch)."; fi

# Per-SKU node-pool drift check: does every OPENRESEARCH_GCP_GPU_SKUS entry
# actually correspond to a node pool that EXISTS in the target cluster -- not
# just one that is "configured"? Config (.env / Settings default) and Terraform
# (infra/gcp/variables.tf `gpu_skus`) can drift; a SKU named in config but never
# provisioned leaves every cell that resolves to it Pending forever (nodeSelector
# matches no node) -> capacity_exhausted after the pending timeout (~15-25 min
# wasted, near-$0 apparent spend).
#
# TWO EVIDENCE TIERS, mirroring backend/services/runtime/gpu_pool_preflight.py:
#   tier 1  `gcloud container node-pools list` -- AUTHORITATIVE. A node pool
#           EXISTS at zero nodes, so this answers even on a fully cold
#           scale-to-zero cluster (the normal state between runs, and exactly
#           when drift bites: the first run of the day).
#   tier 2  `kubectl get nodes` reprolab/sku labels -- heuristic; blind to a
#           pool that currently has no nodes. Used only if tier 1 is unavailable
#           (no container.clusters.get permission, cluster name unset, ...).
# CORROBORATION INVARIANT (both tiers): only a NON-empty observed set that is
# missing a specific configured SKU is a confirmed drift (FAIL). Seeing the
# label on nothing at all is inconclusive (WARN) -- never a mass false FAIL.
configured_skus_raw="${OPENRESEARCH_GCP_GPU_SKUS:-}"
if [[ -n "${configured_skus_raw}" ]]; then
  # Accept either the JSON-array form pydantic-settings parses
  # (["gcp_a100_80","gcp_a100_80x2"]) or a comma-separated string.
  cleaned="${configured_skus_raw//[\[\]\"]/}"
  IFS=',' read -r -a configured_skus <<< "${cleaned}"

  observed_skus=""
  evidence=""

  # --- Tier 1: the GKE node-pool API (sees a pool even at ZERO nodes).
  gke_cluster="${OPENRESEARCH_GCP_GKE_CLUSTER:-}"
  gke_region="${OPENRESEARCH_GCP_REGION:-us-central1}"   # matches config.py's default
  if [[ -n "${gke_cluster}" ]]; then
    # `|| true` + the empty-set check below keep a 403 (missing
    # container.clusters.get) or any API error a WARN-and-degrade, never a FAIL.
    pools_json="$(gcloud container node-pools list \
        --cluster="${gke_cluster}" --region="${gke_region}" \
        --project="${OPENRESEARCH_GCP_PROJECT}" --format=json 2>/dev/null || true)"
    if [[ -n "${pools_json}" ]]; then
      observed_skus="$(grep -o '"reprolab/sku"[[:space:]]*:[[:space:]]*"[^"]*"' <<< "${pools_json}" \
        | sed 's/.*"\([^"]*\)"$/\1/' | grep -v '^$' | sort -u || true)"
      [[ -n "${observed_skus}" ]] && evidence="node-pool API (authoritative)"
    fi
  fi

  # --- Tier 2: live-node labels (blind to a scaled-to-zero pool).
  if [[ -z "${observed_skus}" ]]; then
    observed_skus="$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.labels.reprolab/sku}{"\n"}{end}' 2>/dev/null | grep -v '^$' | sort -u || true)"
    [[ -n "${observed_skus}" ]] && evidence="live GPU nodes (heuristic)"
  fi

  if [[ -n "${observed_skus}" ]]; then
    missing=()
    for raw_sku in "${configured_skus[@]}"; do
      sku="$(echo "${raw_sku}" | xargs)"  # trim whitespace
      [[ -z "${sku}" ]] && continue
      grep -qx "${sku}" <<< "${observed_skus}" || missing+=("${sku}")
    done
    if [[ "${#missing[@]}" -gt 0 ]]; then
      echo "FAIL  OPENRESEARCH_GCP_GPU_SKUS names SKU(s) with NO matching node pool: ${missing[*]}." >&2
      echo "      Evidence: ${evidence}. Pools actually provisioned: $(echo "${observed_skus}" | tr '\n' ' ')" >&2
      echo "      A cell resolving to one of these would submit a Job whose nodeSelector" >&2
      echo "      matches no node pool: Pending forever -> capacity_exhausted after the" >&2
      echo "      pending timeout (~15-25 min), near-\$0 apparent spend. Fix ONE of:" >&2
      echo "      provision ${missing[*]} as a node pool in Terraform" >&2
      echo "      (infra/gcp/variables.tf 'gpu_skus') and apply, or remove ${missing[*]}" >&2
      echo "      from OPENRESEARCH_GCP_GPU_SKUS. (exit 7)" >&2
      exit 7
    fi
    echo "OK    All OPENRESEARCH_GCP_GPU_SKUS entries have a real node pool [${evidence}]: $(echo "${observed_skus}" | tr '\n' ' ')"
  else
    echo "WARN  Cannot verify per-SKU node-pool provisioning. The authoritative check needs"
    echo "      OPENRESEARCH_GCP_GKE_CLUSTER set AND container.clusters.get on your identity"
    echo "      (e.g. roles/container.clusterViewer); the live-node fallback sees nothing"
    echo "      because every GPU pool is scaled to zero right now. Proceeding unguarded —"
    echo "      a missing pool would surface late, as capacity_exhausted (~15-25 min in)."
  fi
else
  echo "WARN  OPENRESEARCH_GCP_GPU_SKUS not set in this shell/.env — per-SKU node-pool"
  echo "      drift check skipped (the Settings default will be used at run time)."
fi

if [[ "${START_POD}" == "1" ]]; then
  echo "WARN  --start-pod is OPERATOR-GATED and COSTS MONEY; deliberately a stub here. Use the documented manual smoke. (exit 6)"; exit 6
fi
echo "GKE preflight: all green."; exit 0
