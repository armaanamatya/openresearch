#!/bin/bash
# Autonomous: wait for the gke-cell-verl build, verify its baked-in stack
# assertions, then dispatch the hand-authored UCPO execute cell to 1xA100.
# GPU spend is bounded by the cell timeout + the scale-to-zero pool.
set -uo pipefail
BUILD="${VERL_BUILD_ID:?set VERL_BUILD_ID}"
PROJ=deepinvent-ext-ut
IMG=us-central1-docker.pkg.dev/deepinvent-ext-ut/reprolab/gke-cell-verl:v1
cd /home/abheekp/openresearch

echo "[orch] waiting for build $BUILD ..."
ST=""
for i in $(seq 1 60); do
  ST=$(gcloud builds describe "$BUILD" --project "$PROJ" --format="value(status)" 2>/dev/null)
  echo "[orch] poll $i: build=$ST"
  case "$ST" in
    SUCCESS) break;;
    FAILURE|CANCELLED|TIMEOUT|EXPIRED)
      echo "[orch] BUILD $ST — ABORT dispatch. Last 40 log lines:"
      gcloud builds log "$BUILD" --project "$PROJ" 2>/dev/null | tail -40
      exit 2;;
  esac
  sleep 30
done
[ "$ST" = "SUCCESS" ] || { echo "[orch] build not SUCCESS after wait ($ST); abort"; exit 3; }

echo "[orch] BUILD SUCCESS. Baked-in stack assertions:"
gcloud builds log "$BUILD" --project "$PROJ" 2>/dev/null | grep -E "base OK|stack OK" \
  || echo "[orch] (assertion lines not in retained log — build still passed the RUN gates)"

echo "[orch] dispatching UCPO execute cell on $IMG (1xA100, <=3600s) ..."
OPENRESEARCH_GCP_BASE_IMAGE="$IMG" UCPO_CELL_TIMEOUT_S=3600 UCPO_MATRIX_TIMEOUT_S=4200 \
  .venv/bin/python scripts/prove_ucpo_cell.py

echo "[orch] === post-run money check (GPU nodes should be gone) ==="
kubectl get nodes 2>&1 | head
echo "[orch] DONE"
