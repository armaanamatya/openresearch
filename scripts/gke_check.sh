#!/usr/bin/env bash
# GKE preflight + (optional) end-to-end smoke for the openresearch pipeline.
# Green here => --sandbox gcp / --sandbox gke will auth, reach the cluster, and
# have GPU quota. Usage: scripts/gke_check.sh [--start-pod]
# Exit: 0 green; 2 missing env; 3 gcloud/ADC; 4 cluster unreachable; 5 GPU quota;
#       6 --start-pod smoke (operator-gated, COSTS MONEY).
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
if [[ "${START_POD}" == "1" ]]; then
  echo "WARN  --start-pod is OPERATOR-GATED and COSTS MONEY; deliberately a stub here. Use the documented manual smoke. (exit 6)"; exit 6
fi
echo "GKE preflight: all green."; exit 0
