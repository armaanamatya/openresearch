#!/usr/bin/env bash
# UCPO (arXiv 2605.00365) autonomous re-run on GKE — repo-first + execute-mode +
# the E-series autonomy (E1 framework->image, E4 execute-synth floor) + Option A
# (faithful HELD-OUT eval provenance).  Polls continuously for A100 capacity: the
# a100-80-rw pool is shared/contended, so the UCPO cell may sit Pending until a
# node frees or autoscales (Pending == no GPU billing).  Each outer pass launches
# (attempt 1) or --resume (later attempts, reusing the cached LLM front-half +
# synthesized cell) until a cell binds a GPU and writes metrics, or the attempt
# budget is exhausted.
#
# Repo-first logic (the user's ask): OPENRESEARCH_USE_AUTHOR_REPO=1 +
# REPRODUCTION_MODE=execute — the harness clones AnamikaLochab/UCPO into code/ and
# RUNS the authors' verl pipeline behind the deterministic execute-synth shim,
# rather than re-implementing from scratch.
#
# Money: GPU billed ONLY while the pod is bound; OPENRESEARCH_MAX_RUN_GPU_USD=35
# is the enforced run-total GPU ceiling.  A `kubectl get nodes` money check runs
# each pass.  This script never tears down nodes it did not create (other
# operators' runs share the pool).
#
# Usage:
#   PID=prj_ucpo_optA_1 scripts/ucpo_optA_gcp.sh          # fresh run + poll
#   PID=prj_ucpo_optA_1 MAX_ATTEMPTS=40 SLEEP=300 scripts/ucpo_optA_gcp.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
export PATH="$HOME/.local/bin:$PATH"   # gke-gcloud-auth-plugin (installed no-sudo)

PID="${PID:-prj_ucpo_optA_1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-40}"
SLEEP="${SLEEP:-300}"                  # 5 min between re-queue attempts
MAX_USD="${MAX_USD:-60}"
MAX_GPU_USD="${MAX_GPU_USD:-35}"
PER_ATTEMPT_WALL_S="${PER_ATTEMPT_WALL_S:-7200}"   # 2h/attempt (front-half + cell)
LOG="${LOG:-/tmp/ucpo_optA_${PID}.log}"

# --- GPU pool: 1xA100-80 (cheapest rung, least stockout-prone). GPU_COUNT=1. ---
export OPENRESEARCH_GCP_GPU_SKUS='["gcp_a100_80"]'
export OPENRESEARCH_GPU_COUNT=1 OPENRESEARCH_FORCE_SINGLE_GPU=true OPENRESEARCH_DYNAMIC_GPU_ENABLED=false

# --- image config: E1 maps a detected verl paper -> the validated gke-cell-verl ---
export OPENRESEARCH_GCP_BASE_IMAGE="us-central1-docker.pkg.dev/deepinvent-ext-ut/reprolab/gke-cell-base:v1"
export OPENRESEARCH_GCP_FRAMEWORK_IMAGES='{"verl": "us-central1-docker.pkg.dev/deepinvent-ext-ut/reprolab/gke-cell-verl:v1"}'
export OPENRESEARCH_GCP_PROJECT="deepinvent-ext-ut"
export OPENRESEARCH_GCP_GCS_BUCKET="deepinvent-ext-ut-sdar-runs"

_cell_metrics_present() {
  # A GPU cell actually ran iff an outputs/**/metrics.json carries a measured
  # reward/eval key (success_rate | reward_mean | reward). That is the STOP
  # signal — everything before it is "still waiting for / repairing the GPU path".
  python3 - "$PID" <<'PY'
import glob, json, sys
pid = sys.argv[1]
for mf in glob.glob(f"runs/{pid}/**/outputs/**/metrics.json", recursive=True):
    try:
        m = json.load(open(mf))
    except Exception:
        continue
    if isinstance(m, dict) and any(k in m for k in ("success_rate", "reward_mean", "reward")):
        print("METRICS_PRESENT", mf, {k: m.get(k) for k in ("success_rate","reward_mean","reward","metric_source","reward_key")})
        raise SystemExit(0)
raise SystemExit(1)
PY
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  RESUME=""
  [ "$attempt" -gt 1 ] && RESUME="--resume"
  echo "======================================================================"
  echo "[ucpo-optA] attempt $attempt/$MAX_ATTEMPTS  PID=$PID  resume='${RESUME:-<fresh>}'"
  echo "[ucpo-optA] budgets: --max-usd $MAX_USD  --max-run-gpu-usd $MAX_GPU_USD  wall ${PER_ATTEMPT_WALL_S}s"
  echo "[ucpo-optA] $(date -u +%FT%TZ)  log=$LOG"
  echo "======================================================================"

  UCPO_RESUME=$([ "$attempt" -gt 1 ] && echo 1 || echo 0) \
  UCPO_PID="$PID" UCPO_MAX_USD="$MAX_USD" UCPO_MAX_GPU_USD="$MAX_GPU_USD" UCPO_WALL="$PER_ATTEMPT_WALL_S" \
  .venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, os, sys, runpy
from dotenv import load_dotenv
load_dotenv('.env')                                   # cli.py does not load .env itself
# Load the Option-A / execute / repo-first / guard flags (OPENRESEARCH_* only;
# the run spec's non-OPENRESEARCH "models" key is passed on the CLI instead, and
# the bash-exported GCP image/GPU env is never overridden — the spec omits it).
spec = json.load(open('configs/ucpo_execute_run_spec.json'))
os.environ.update({k: str(v) for k, v in spec.items() if k.startswith('OPENRESEARCH_')})
argv = ['backend.cli', 'reproduce', '2605.00365',
        '--project-id', os.environ['UCPO_PID'],
        '--sandbox', 'gcp', '--mode', 'rlm', '--model', 'opus-foundry', '--provider', 'anthropic',
        '--models', 'executor=sonnet-foundry,grader=sonnet-foundry,verifier=sonnet-foundry,spec_validator=grok',
        '--max-usd', os.environ['UCPO_MAX_USD'], '--max-run-gpu-usd', os.environ['UCPO_MAX_GPU_USD'],
        '--max-wall-clock', os.environ['UCPO_WALL'], '--max-repair-iterations', '3']
if os.environ.get('UCPO_RESUME') == '1':
    argv.append('--resume')
sys.argv = argv
runpy.run_module('backend.cli', run_name='__main__')
PY

  echo "[ucpo-optA] --- money check (A100 nodes; do NOT touch others' runs) ---"
  timeout 40 kubectl get nodes 2>&1 | grep -E 'NAME|a100' | head

  if _cell_metrics_present; then
    echo "[ucpo-optA] GPU cell produced metrics — STOP polling. Final report:"
    python3 -c "import json;d=json.load(open('runs/$PID/final_report.json'));print('verdict=',d.get('verdict'));print('reproduction=',json.dumps(d.get('reproduction',{}))[:400])" 2>/dev/null
    exit 0
  fi

  echo "[ucpo-optA] no GPU cell metrics yet (A100 likely still contended). Sleeping ${SLEEP}s before re-queue..."
  sleep "$SLEEP"
done

echo "[ucpo-optA] exhausted $MAX_ATTEMPTS attempts without a GPU cell. Last money check:"
timeout 40 kubectl get nodes 2>&1 | head
exit 4
