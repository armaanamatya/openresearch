#!/usr/bin/env bash
# One-command GCP-readiness preflight for --sandbox gcp/gke.
#
# Goal: get an operator from "nothing configured" to "a real autonomous GCP
# run is bookable" in one command. This is a SUPERSET of scripts/gke_check.sh
# (it calls that script at the end) plus the steps gke_check.sh does not do:
#   1. Read OPENRESEARCH_GCP_PROJECT / _GCS_BUCKET / _BASE_IMAGE from the
#      shell env or .env (same precedence start.sh uses) and report each as
#      SET/MISSING.
#   2. Point kubectl at the actual GKE cluster via
#      `gcloud container clusters get-credentials` — this is the fix for the
#      common failure mode where the current kube-context points at some
#      OTHER cluster (e.g. a leftover Azure AKS context) and every kubectl
#      call in gke_check.sh / the backend silently targets the wrong cluster.
#   3. List the GPU node pool(s) and report current GPU-node capacity (0 is
#      NORMAL for a scale-to-zero pool, not an error — GKE autoscales a node
#      in on first Job dispatch).
#   4. Best-effort regional GPU quota check (warn-only; not a hard gate).
#   5. Delegate to scripts/gke_check.sh for the ADC / bucket / cluster-info
#      checks it already owns.
#   6. Print one FINAL SUMMARY checklist: READY/MISSING per hard requirement,
#      with the exact .env line or command to fix each MISSING item.
#
# This script is READ-ONLY plus `get-credentials` (a local kubeconfig write,
# not a cluster mutation) — it never launches a Job or a pod, so it does not
# spend money.
#
# Usage:
#   scripts/gcp_ready.sh [--cluster NAME] [--zone ZONE] [--region REGION] [--project ID]
#
# Env overrides (same names the flags set): GCP_CLUSTER, GCP_ZONE, GCP_REGION.
# --region (or $GCP_REGION) takes precedence over --zone for a regional cluster.
#
# Exit codes:
#   0  bookable — every hard requirement (project, bucket, base image, ADC,
#      kubeconfig) is satisfied.
#   1  invalid usage (bad flag).
#   2  at least one hard requirement is missing — see FINAL SUMMARY for the fix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
cd "${REPO_ROOT}"

# Shared dotenv-grammar .env reader (env_value_from_file) — same helper
# start.sh uses, so parsing of quoted/commented values matches pydantic's.
# shellcheck source=lib/env_file.sh
. "${SCRIPT_DIR}/lib/env_file.sh"

CLUSTER="${GCP_CLUSTER:-openresearch-gpu}"
ZONE="${GCP_ZONE:-us-central1-a}"
REGION="${GCP_REGION:-}"
CLI_PROJECT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cluster) CLUSTER="$2"; shift 2 ;;
    --cluster=*) CLUSTER="${1#*=}"; shift ;;
    --zone) ZONE="$2"; REGION=""; shift 2 ;;
    --zone=*) ZONE="${1#*=}"; REGION=""; shift ;;
    --region) REGION="$2"; shift 2 ;;
    --region=*) REGION="${1#*=}"; shift ;;
    --project) CLI_PROJECT="$2"; shift 2 ;;
    --project=*) CLI_PROJECT="${1#*=}"; shift ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Zonal vs regional cluster: --region (if given) wins over --zone.
LOCATION_FLAG=(--zone "$ZONE")
LOCATION_DESC="zone=${ZONE}"
if [[ -n "$REGION" ]]; then
  LOCATION_FLAG=(--region "$REGION")
  LOCATION_DESC="region=${REGION}"
fi

echo "=============================================================================="
echo " GCP readiness preflight — cluster=${CLUSTER} ${LOCATION_DESC}"
echo "=============================================================================="

# ---------------------------------------------------------------------------
# Step 1: OPENRESEARCH_GCP_PROJECT / _GCS_BUCKET / _BASE_IMAGE
# ---------------------------------------------------------------------------
# Precedence: real shell env wins over .env (mirrors pydantic-settings, which
# ranks process env above the .env file — see start.sh's own comment on this).
_read_setting() {
  local var="$1" val
  val="${!var:-}"
  if [[ -z "$val" ]]; then
    val="$(env_value_from_file "$var" "$ENV_FILE" 2>/dev/null || true)"
  fi
  printf '%s' "$val"
}

ENV_PROJECT="$(_read_setting OPENRESEARCH_GCP_PROJECT)"
BUCKET="$(_read_setting OPENRESEARCH_GCP_GCS_BUCKET)"
BASE_IMAGE="$(_read_setting OPENRESEARCH_GCP_BASE_IMAGE)"

# PROJECT is what THIS script uses to drive its own gcloud calls below
# (get-credentials / node-pools / quota). It falls back to `gcloud config
# get-value project` and --project purely so those diagnostics still work
# before .env is fully wired. That fallback does NOT satisfy the backend's
# own requirement — backend/config.py Settings (pydantic-settings) reads
# OPENRESEARCH_GCP_PROJECT only from process env / .env, never from gcloud's
# active config — so the hard-requirement check below is keyed on
# ENV_PROJECT, not PROJECT.
PROJECT="$ENV_PROJECT"
PROJECT_SOURCE="env/.env"
if [[ -n "$CLI_PROJECT" ]]; then
  PROJECT="$CLI_PROJECT"
  PROJECT_SOURCE="--project flag"
elif [[ -z "$PROJECT" ]] && command -v gcloud >/dev/null 2>&1; then
  gconf_project="$(gcloud config get-value project 2>/dev/null || true)"
  if [[ -n "$gconf_project" && "$gconf_project" != "(unset)" ]]; then
    PROJECT="$gconf_project"
    PROJECT_SOURCE="gcloud config get-value project"
  fi
fi

echo
echo "-- Settings --"
if [[ -n "$ENV_PROJECT" ]]; then
  echo "OK    OPENRESEARCH_GCP_PROJECT=${ENV_PROJECT} (from env/.env)"
elif [[ -n "$PROJECT" ]]; then
  echo "WARN  OPENRESEARCH_GCP_PROJECT not set in .env — using '${PROJECT}' (from ${PROJECT_SOURCE}) to drive the checks below only; the backend itself won't see this until it's also in .env"
else
  echo "MISS  OPENRESEARCH_GCP_PROJECT not set (and no gcloud default project)"
fi
if [[ -n "$BUCKET" ]]; then
  echo "OK    OPENRESEARCH_GCP_GCS_BUCKET=${BUCKET}"
else
  echo "MISS  OPENRESEARCH_GCP_GCS_BUCKET not set"
fi
if [[ -n "$BASE_IMAGE" ]]; then
  echo "OK    OPENRESEARCH_GCP_BASE_IMAGE=${BASE_IMAGE}"
else
  echo "MISS  OPENRESEARCH_GCP_BASE_IMAGE not set (mandatory — no ':latest' fallback)"
fi

# ---------------------------------------------------------------------------
# Step 2: point kubectl at the GKE cluster (fixes a foreign kube-context)
# ---------------------------------------------------------------------------
echo
echo "-- Step: gcloud container clusters get-credentials ${CLUSTER} (${LOCATION_DESC}) --"
GCLOUD_PRESENT=1
command -v gcloud >/dev/null 2>&1 || { echo "MISS  gcloud not found — install: https://cloud.google.com/sdk/docs/install"; GCLOUD_PRESENT=0; }
KUBECTL_PRESENT=1
command -v kubectl >/dev/null 2>&1 || { echo "MISS  kubectl not found — install: gcloud components install kubectl"; KUBECTL_PRESENT=0; }

CONTEXT_OK=0
if [[ "$GCLOUD_PRESENT" == "1" && -n "$PROJECT" ]]; then
  if gcloud container clusters get-credentials "$CLUSTER" "${LOCATION_FLAG[@]}" --project "$PROJECT" 2>&1 | sed 's/^/      /'; then
    CONTEXT_OK=1
    if [[ "$KUBECTL_PRESENT" == "1" ]]; then
      CUR_CTX="$(kubectl config current-context 2>/dev/null || echo unknown)"
      echo "OK    kubectl context now: ${CUR_CTX}"
    fi
  else
    echo "MISS  get-credentials failed for cluster=${CLUSTER} ${LOCATION_DESC} project=${PROJECT}"
    echo "      (wrong cluster/zone name, no access to the project, or the cluster does not exist there)"
  fi
elif [[ "$GCLOUD_PRESENT" == "1" ]]; then
  echo "SKIP  no GCP project resolved — cannot get-credentials yet."
else
  echo "SKIP  gcloud missing — cannot get-credentials."
fi

# ---------------------------------------------------------------------------
# Step 3: GPU node pool(s)
# ---------------------------------------------------------------------------
echo
echo "-- Step: GPU node pool --"
if [[ "$GCLOUD_PRESENT" == "1" && -n "$PROJECT" ]]; then
  echo "Node pools (gcloud container node-pools list):"
  if ! gcloud container node-pools list --cluster "$CLUSTER" "${LOCATION_FLAG[@]}" --project "$PROJECT" 2>&1 | sed 's/^/      /'; then
    echo "      WARN: could not list node pools for cluster=${CLUSTER} ${LOCATION_DESC}"
  fi
else
  echo "SKIP  no project/gcloud — cannot list node pools."
fi

if [[ "$CONTEXT_OK" == "1" && "$KUBECTL_PRESENT" == "1" ]]; then
  gpu_nodes="$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null | grep -c '^[1-9]' || true)"
  if [[ "${gpu_nodes:-0}" -gt 0 ]]; then
    echo "OK    ${gpu_nodes} GPU node(s) currently advertise nvidia.com/gpu."
  else
    echo "NOTE  No GPU node currently advertises nvidia.com/gpu — the pool is most"
    echo "      likely scaled to zero. This is EXPECTED for a scale-to-zero pool and"
    echo "      is NOT an error: GKE autoscales a node in on the first dispatched Job"
    echo "      (allow the gcp_boot_timeout_seconds/gcp_pending_timeout_seconds window,"
    echo "      code default up to 1500s, for cold-start capacity to appear)."
  fi
else
  echo "SKIP  kubectl not pointed at the cluster yet — cannot inspect GPU node capacity."
fi

# ---------------------------------------------------------------------------
# Step 4: regional GPU quota (best-effort, warn-only)
# ---------------------------------------------------------------------------
echo
echo "-- Step: regional GPU quota (best-effort, warn-only) --"
QUOTA_REGION="${REGION:-${ZONE%-*}}"
if [[ "$GCLOUD_PRESENT" == "1" && -n "$PROJECT" ]]; then
  found_any=0
  for metric in NVIDIA_A100_80GB_GPUS NVIDIA_A100_GPUS; do
    qlimit="$(gcloud compute regions describe "$QUOTA_REGION" --project "$PROJECT" \
      --format="csv[no-heading](quotas.metric,quotas.limit)" 2>/dev/null \
      | grep "^${metric}," | cut -d',' -f2 | head -1)" || true
    if [[ -n "$qlimit" ]]; then
      found_any=1
      if [[ "${qlimit%%.*}" -eq 0 ]] 2>/dev/null; then
        echo "WARN  ${metric} limit in ${QUOTA_REGION}: ${qlimit} (0 — request quota or use spot)"
      else
        echo "OK    ${metric} limit in ${QUOTA_REGION}: ${qlimit}"
      fi
    fi
  done
  [[ "$found_any" == "1" ]] || echo "WARN  could not read A100 quota metrics for region=${QUOTA_REGION} (non-blocking)"
else
  echo "SKIP  no project/gcloud — cannot check quota."
fi

# ---------------------------------------------------------------------------
# Step 5: delegate to scripts/gke_check.sh (ADC / bucket / cluster-info)
# ---------------------------------------------------------------------------
echo
echo "-- Step: scripts/gke_check.sh (ADC / bucket-env / cluster-info) --"
GKE_CHECK_RC=0
if [[ -x "${SCRIPT_DIR}/gke_check.sh" ]]; then
  "${SCRIPT_DIR}/gke_check.sh" || GKE_CHECK_RC=$?
else
  echo "WARN  scripts/gke_check.sh not found or not executable — skipping delegated check"
  GKE_CHECK_RC=127
fi

# ---------------------------------------------------------------------------
# Step 6: FINAL SUMMARY
# ---------------------------------------------------------------------------
echo
echo "=============================================================================="
echo " FINAL SUMMARY"
echo "=============================================================================="

declare -a HARD_MISSING=()
[[ -n "$ENV_PROJECT" ]] || HARD_MISSING+=("project")
[[ -n "$BUCKET" ]] || HARD_MISSING+=("bucket")
[[ -n "$BASE_IMAGE" ]] || HARD_MISSING+=("base_image")
[[ "$CONTEXT_OK" == "1" ]] || HARD_MISSING+=("kubeconfig")
# gke_check.sh's own hard requirements (ADC, gcloud, kubectl, cluster-info) are
# folded into its exit code — exit 2 there is env-only and already covered by
# project/bucket above, so only surface it as a distinct item when it fails
# for a reason NOT already listed (e.g. ADC missing).
if [[ "$GKE_CHECK_RC" != "0" ]]; then
  case "$GKE_CHECK_RC" in
    2) : ;; # already covered by project/bucket MISS above
    *) HARD_MISSING+=("gke_check(rc=${GKE_CHECK_RC})") ;;
  esac
fi

printf '  %-12s %s\n' "PROJECT:"     "$([[ -n "$ENV_PROJECT" ]] && echo "READY (${ENV_PROJECT})" || echo "MISSING (diagnostics used '${PROJECT:-none}' via ${PROJECT_SOURCE})")"
printf '  %-12s %s\n' "BUCKET:"      "$([[ -n "$BUCKET" ]] && echo "READY (${BUCKET})" || echo "MISSING")"
printf '  %-12s %s\n' "BASE_IMAGE:"  "$([[ -n "$BASE_IMAGE" ]] && echo "READY (${BASE_IMAGE})" || echo "MISSING")"
printf '  %-12s %s\n' "KUBECONFIG:"  "$([[ "$CONTEXT_OK" == "1" ]] && echo "READY" || echo "MISSING")"
printf '  %-12s %s\n' "ADC/CLUSTER:" "$([[ "$GKE_CHECK_RC" == "0" ]] && echo "READY" || echo "NOT READY (gke_check.sh rc=${GKE_CHECK_RC})")"

echo
if [[ "${#HARD_MISSING[@]}" -eq 0 && "$GKE_CHECK_RC" == "0" ]]; then
  echo "READY — a real autonomous GCP run is bookable. Example:"
  echo "  python -m backend.cli reproduce <paper> --sandbox gcp --max-gpu-usd <X> --max-gpu-hours <Y>"
  exit 0
fi

echo "NOT READY — fix the following:"
for item in "${HARD_MISSING[@]}"; do
  case "$item" in
    project)
      echo "  - Set OPENRESEARCH_GCP_PROJECT in .env, e.g.: OPENRESEARCH_GCP_PROJECT=${PROJECT:-<your-project-id>}"
      echo "    (gcloud's active config project is NOT read by the backend — it must be in .env or exported)"
      ;;
    bucket)
      echo "  - Set OPENRESEARCH_GCP_GCS_BUCKET in .env, e.g.: OPENRESEARCH_GCP_GCS_BUCKET=<bucket-name>"
      ;;
    base_image)
      echo "  - Set OPENRESEARCH_GCP_BASE_IMAGE in .env to a PINNED Artifact Registry tag, e.g.:"
      echo "      OPENRESEARCH_GCP_BASE_IMAGE=us-central1-docker.pkg.dev/${PROJECT:-<project>}/reprolab/reprolab-cell:<git-sha>"
      echo "    (never ':latest' — the backend refuses an empty/unpinned value)"
      ;;
    kubeconfig)
      echo "  - Run: gcloud container clusters get-credentials ${CLUSTER} ${LOCATION_FLAG[*]} --project ${PROJECT:-<project>}"
      ;;
    gke_check*)
      echo "  - $item: re-run scripts/gke_check.sh directly to see the detailed failure (ADC / gcloud / kubectl / cluster-info)."
      ;;
  esac
done
if [[ "$GKE_CHECK_RC" == "2" ]]; then
  echo "  - scripts/gke_check.sh also reports exit 2 (missing env) — same fix as PROJECT/BUCKET above."
fi
exit 2
