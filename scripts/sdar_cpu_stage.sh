#!/usr/bin/env bash
# sdar_cpu_stage.sh — reliable, idempotent CPU-side staging of the SDAR cache.
#
# WHY THIS EXISTS (self-learning: every fix below was a real staging failure that
# cost a debug cycle; this script encodes them so a fresh cache NEVER re-hits them):
#   1. conda ToS       — miniconda >=25 gates `conda create` on Terms-of-Service
#                        acceptance for the anaconda main/r channels (base env
#                        creation dies with CondaToSNonInteractiveError otherwise).
#   2. g++ missing     — the DLVM `common-cu*` image ships gcc but NOT g++; the
#                        flash-attn source build fails "command 'g++' failed:
#                        No such file or directory". Install build-essential.
#   3. flash-attn OOM  — the flash-attn nvcc build is memory-hungry; cap MAX_JOBS
#                        (=4 on a 62 GB box) so parallel compile units don't OOM.
#   4. sdar verl deps  — Search-QA preprocessing imports verl -> tensordict, which
#                        the base env install omits (ModuleNotFoundError tensordict).
#   5. wiki-18 download— the ~70 GB e5 index is fetched UNAUTHENTICATED from the HF
#                        Hub -> rate-limited + flaky; the phase's built-in 5x retry
#                        gives up early. It is RESUMABLE, so loop until complete.
#   6. Java missing    — the WebShop Lucene (pyserini) index build needs a JRE.
#
# PRECONDITIONS (the VM-side provisioning the orchestrator must do first):
#   - a CPU VM (no GPU) on a Deep-Learning `common-cu*` image (provides nvcc for the
#     flash-attn build; a GPU is NOT needed to COMPILE CUDA kernels).
#   - the CLEAN cache disk formatted ext4 + mounted at /mnt/sdar-cache (a fresh disk,
#     NOT a former boot disk — those fail the by-id raw mount).
#   - miniconda installed at /mnt/sdar-cache/miniconda; git present.
#   - scripts/sdar_authors_repro.sh copied to ~/sdar_authors_repro.sh.
#
# Idempotent + resumable: safe to re-run after ANY failure; completed steps skip.
# On success writes /mnt/sdar-cache/.warm_ok (the sentinel the GPU runner checks to
# take the fast, no-re-download prepare path).
set -uo pipefail

CACHE=/mnt/sdar-cache
LOG_DIR="$CACHE/logs"; mkdir -p "$LOG_DIR"
export PATH="$CACHE/miniconda/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export HF_HOME="$CACHE/hf"
export MAX_JOBS="${MAX_JOBS:-4}"     # fix #3 — cap flash-attn build parallelism
export CXX=g++ CC=gcc
SEARCH_MAX_RETRIES="${SEARCH_MAX_RETRIES:-40}"   # fix #5 — resumable download loop

step() { echo "[stage $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
REPO="$HOME/sdar_authors_repro.sh"
[ -f "$REPO" ] || { step "FATAL: $REPO missing (scp scripts/sdar_authors_repro.sh first)"; exit 2; }

# --- 0. provisioning fixes (idempotent) --------------------------------------
step "fix #1: accept conda ToS (main + r channels)"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true
if ! command -v g++ >/dev/null; then
  step "fix #2: install build-essential (g++ for the flash-attn source build)"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential >/dev/null 2>&1
fi
if ! command -v java >/dev/null; then
  step "fix #6: install default-jre-headless (WebShop Lucene index build)"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq default-jre-headless >/dev/null 2>&1
fi

# --- 1. base (clone + models + 3 conda envs) ---------------------------------
step "phase: base"
bash "$REPO" base > "$LOG_DIR/base.log" 2>&1 || { step "FAILED: base (see logs/base.log)"; exit 1; }

# --- 2. sdar env verl deps (fix #4) ------------------------------------------
step "fix #4: install verl preprocessing deps into the sdar env (tensordict, ...)"
"$CACHE/conda/envs/sdar/bin/pip" install -q \
  tensordict codetiming omegaconf hydra-core pandas pyarrow datasets >/dev/null 2>&1 || \
  step "WARN: sdar dep install had issues (search preprocess may surface a missing module)"

# --- 3. alfworld -------------------------------------------------------------
step "phase: alfworld"
bash "$REPO" alfworld > "$LOG_DIR/alfworld.log" 2>&1 || { step "FAILED: alfworld"; exit 1; }

# --- 4. search — resilient wiki-18 download loop (fix #5) ---------------------
step "phase: search (resilient; wiki-18 e5 index ~70 GB, resumable, HF rate-limited)"
_search_ok=0
for a in $(seq 1 "$SEARCH_MAX_RETRIES"); do
  if bash "$REPO" search > "$LOG_DIR/search.log" 2>&1; then _search_ok=1; break; fi
  # the download is resumable — .cache accumulates across attempts; keep going
  step "  search attempt $a/$SEARCH_MAX_RETRIES incomplete (resuming); cache=$(du -sh "$CACHE/data/searchR1" 2>/dev/null | cut -f1)"
  sleep 15
done
if [ "$_search_ok" != 1 ] && [ ! -f "$CACHE/data/searchR1/e5_Flat.index" ]; then
  step "FAILED: search (no e5_Flat.index after $SEARCH_MAX_RETRIES attempts; see logs/search.log)"; exit 1
fi

# --- 5. webshop (flash-attn build + gdown corpus + Lucene index) -------------
step "phase: webshop"
bash "$REPO" webshop > "$LOG_DIR/webshop.log" 2>&1 || { step "FAILED: webshop (see logs/webshop.log)"; exit 1; }

# --- done --------------------------------------------------------------------
touch "$CACHE/.warm_ok"
step "=== ALL STAGING COMPLETE — /mnt/sdar-cache/.warm_ok written ==="
