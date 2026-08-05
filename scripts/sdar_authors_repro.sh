#!/usr/bin/env bash
# sdar_authors_repro.sh — Phase-able, cache-aware, idempotent orchestration
# of the ZJU-REAL/SDAR authors' training pipeline at 150 steps.
#
# Target host: GCP VM, 4×A100-80GB, us-central1-b.
# Persistent cache disk mounted at /mnt/sdar-cache.
# Requires: conda (miniconda/mambaforge), git, gzip, curl on PATH.
#
# Usage:
#   bash scripts/sdar_authors_repro.sh <phase>
#   phase: base | alfworld | webshop | search | smoke | grid | collect | all
#
# Phase order for a full run: base → alfworld → webshop → search → smoke → grid → collect
# Each phase is idempotent: safe to re-run if partially completed.
#
# NOT committed. NOT run here — executed later on the GCP VM.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — all tuneable knobs live here
# ─────────────────────────────────────────────────────────────────────────────

CACHE_ROOT="/mnt/sdar-cache"

# ── Repo ──
REPO_URL="https://github.com/ZJU-REAL/SDAR.git"
REPO_COMMIT="9f2ce6a"
REPO_DIR="${CACHE_ROOT}/SDAR"

# ── HuggingFace cache ──
export HF_HOME="${CACHE_ROOT}/hf"

# ── Conda env paths (stored on persistent disk, not ~/envs) ──
CONDA_ENVS_DIR="${CACHE_ROOT}/conda/envs"
ENV_SDAR="${CONDA_ENVS_DIR}/sdar"           # py3.12, vllm 0.11.0 — ALFWorld + Search training
ENV_WEBSHOP="${CONDA_ENVS_DIR}/verl-webshop" # py3.10, vllm 0.8.2  — WebShop training
ENV_RETRIEVER="${CONDA_ENVS_DIR}/retriever"  # py3.10, faiss-gpu    — Search retrieval server

# ── Model HF repo IDs ──
MODEL_QWEN3="Qwen/Qwen3-1.7B"
MODEL_3B="Qwen/Qwen2.5-3B-Instruct"
MODEL_7B="Qwen/Qwen2.5-7B-Instruct"
MODEL_E5="intfloat/e5-base-v2"

# ── GPU config ──
ACTUAL_GPUS=4          # VM has 4×A100-80GB
WEBSHOP_GPUS=2         # authors specify 2; exact match on our 4-GPU VM
SEARCH_GPUS=4          # authors specify 4; exact match
# FIDELITY CAVEAT — ALFWorld: authors' scripts specify trainer.n_gpus_per_node=8.
# We override to 4 (all available GPUs). Impact: ~2× fewer rollout workers per
# training step, so per-step wall-clock is similar but each "epoch" samples less
# data in parallel. All SDAR algorithm knobs (sdar_coef/gate_beta/skill_all) are
# unchanged. This is flagged in the final report as a hardware-fidelity deviation.
ALFWORLD_GPUS=4

# ── ALFWorld/WebShop OOM knobs (2026-08-03; docs/periods/2026-07.md §11) ──
# ALFWorld-3B trained and LEARNED (episode/success_rate 0.383→0.508, peaks
# 0.609) then OOMed mid-FSDP-actor-update at step 79: long episodes
# (mean ~40 turns × rollout.n=8 → ~850K-token sequences) + Adam step with no
# offload on top of vLLM holding 60% VRAM → fragmentation. The memory-safe
# config is the one already proven on Search. Applied via sed to the ALFWorld
# and WebShop launches ONLY — run_search_3b.sh is proven at 0.456 and stays
# byte-identical (do NOT add these to the Search launches).
# NOTE: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is a DEAD END here —
# vLLM's CuMemAllocator raises "AssertionError: Expandable segments are not
# compatible with memory pool" at model load (see the command-cell exemption
# in backend/agents/rlm/gpu_cell_runner.py). The last sed strips any such
# export inside the authors' scripts (run_webshop_3b_patched.sh carries one);
# nothing in THIS script sets it.
OOM_MICRO_BSZ="${SDAR_OOM_MICRO_BSZ:-16}"          # ppo_micro_batch_size_per_gpu 32→16
OOM_GPU_MEM_UTIL="${SDAR_OOM_GPU_MEM_UTIL:-0.5}"   # rollout.gpu_memory_utilization 0.6→0.5
OOM_OPT_OFFLOAD="${SDAR_OOM_OPT_OFFLOAD:-True}"    # actor.fsdp_config.optimizer_offload (Adam state → CPU)
OOM_SED_EXPRS=(
    "s/ppo_micro_batch_size_per_gpu=32/ppo_micro_batch_size_per_gpu=${OOM_MICRO_BSZ}/g"
    "s/gpu_memory_utilization=0\.6/gpu_memory_utilization=${OOM_GPU_MEM_UTIL}/g"
    "s/actor\.fsdp_config\.optimizer_offload=False/actor.fsdp_config.optimizer_offload=${OOM_OPT_OFFLOAD}/g"
    "/PYTORCH_CUDA_ALLOC_CONF=expandable_segments/d"
)

# ── Mid-training checkpoint/resume knobs (default OFF — empty = no seds) ──
# The authors' scripts hard-code trainer.save_freq=-1 and carry NO
# trainer.resume_mode arg (verified against the full hydra arg list in
# gcp_logs/sdar_authors_run/run_search_3b.log line 15), so any
# OOM/preemption/interruption restarts a 150-step run from step 0. When
# SDAR_CKPT_STEPS is non-empty, patch save_freq to that cadence and inject
# trainer.resume_mode=auto on the same anchor arg — verl then checkpoints to its
# default local dir every N steps and a rerun of the SAME script resumes from
# the latest checkpoint instead of step 0 (enables spot-GPU usage). Unlike
# OOM_SED_EXPRS (always applied), this DEFAULTS OFF: unset/empty SDAR_CKPT_STEPS
# yields an empty array = zero seds = every launch byte-identical, including the
# proven Search runs.
SDAR_CKPT_STEPS="${SDAR_CKPT_STEPS:-}"
CKPT_SED_EXPRS=()
if [[ -n "${SDAR_CKPT_STEPS}" ]]; then
    CKPT_SED_EXPRS=(
        "s/trainer\.save_freq=-1/trainer.save_freq=${SDAR_CKPT_STEPS} trainer.resume_mode=auto/g"
    )
fi

# ── Training budget (must match authors' scripts) ──
TRAIN_EPOCHS=150         # ALFWorld + WebShop: total_epochs
SEARCH_STEPS=150         # Search: total_training_steps (DIFFERENT UNIT — see recon §3)
SMOKE_EPOCHS=2           # smoke phase reduced budget

# ── Retrieval server ──
RETRIEVER_PORT=8000
RETRIEVER_PID_FILE="/tmp/sdar_retriever.pid"

# ── WandB ──
# Inherit a real key from the env to log online; otherwise run OFFLINE so the
# authors' scripts' trainer.logger=['console','wandb'] never blocks on a wandb
# login prompt in a DETACHED run (collect parses the console logs, not WandB).
WANDB_API_KEY="${WANDB_API_KEY:-}"
if [[ -z "${WANDB_API_KEY}" || "${WANDB_API_KEY}" == "your_key_here" ]]; then
    WANDB_MODE="offline"
else
    WANDB_MODE="${WANDB_MODE:-online}"
fi

# ── Logging ──
LOG_DIR="${CACHE_ROOT}/logs"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

# Retry a command up to 5× with 30s back-off (for flaky downloads).
# Uses `env "$@"` (not bare "$@") so a caller can pass an env-var prefix, e.g.
# `retry HF_HOME=/x python ...` — with bare "$@" the shell treats `HF_HOME=/x`
# as the program name ("No such file or directory") and every attempt fails
# instantly without ever running the command.
retry() {
    local n=0 max=5 delay=30
    until env "$@"; do
        n=$(( n + 1 ))
        [[ ${n} -ge ${max} ]] && die "Command failed after ${max} attempts: $*"
        log "Attempt ${n}/${max} failed — retrying in ${delay}s: $*"
        sleep "${delay}"
    done
}

# Emit conda shell hooks so 'conda activate' works in bash scripts
_conda_init() {
    local base
    base="$(conda info --base 2>/dev/null)" \
        || die "conda not found on PATH — install Miniconda/Mambaforge first"
    # shellcheck disable=SC1091
    source "${base}/etc/profile.d/conda.sh" 2>/dev/null || true
}

# Run one authors' training script inside the correct conda env.
# Applies optional sed patches to the script before executing (for GPU/epoch overrides).
# Captures combined stdout+stderr to LOG_DIR/<script-basename>.log while also streaming
# to the terminal.
#
# Usage: run_script <script_basename> <conda_env_path> <cuda_visible_devs> [sed_expr ...]
#   e.g.: run_script run_alfworld_3b.sh "${ENV_SDAR}" "0,1,2,3" \
#             's/n_gpus_per_node=8/n_gpus_per_node=4/'
run_script() {
    local script="$1" env_path="$2" cuda_devs="$3"
    shift 3
    local -a sed_exprs=("$@")

    local script_path="${REPO_DIR}/examples/sdar_trainer/${script}"
    local log_file="${LOG_DIR}/${script%.sh}.log"

    [[ -f "${script_path}" ]] || die "Authors' script not found: ${script_path}"
    mkdir -p "${LOG_DIR}"

    log "Starting ${script} (GPUs=${cuda_devs}, env=$(basename "${env_path}")) → ${log_file}"

    (
        cd "${REPO_DIR}"
        export CUDA_VISIBLE_DEVICES="${cuda_devs}"
        export HF_HOME="${HF_HOME}"
        export WANDB_API_KEY="${WANDB_API_KEY}"
        export WANDB_MODE="${WANDB_MODE}"   # offline unless a real key is present (no login hang)
        export ALFWORLD_DATA="${CACHE_ROOT}/data/alfworld"   # honored by alfworld scripts
        # Prepend the conda env's bin so python3/pip resolve from the correct env
        export PATH="${env_path}/bin:${PATH}"

        if [[ ${#sed_exprs[@]} -gt 0 ]]; then
            # Build the sed command with all -e expressions
            local -a sed_cmd=("sed")
            for expr in "${sed_exprs[@]}"; do
                sed_cmd+=("-e" "${expr}")
            done
            # Run the patched script through bash (process substitution + CWD already set)
            bash <("${sed_cmd[@]}" "${script_path}")
        else
            bash "${script_path}"
        fi
    ) 2>&1 | tee "${log_file}"

    log "Finished ${script}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval server lifecycle (used by search grid runs)
# ─────────────────────────────────────────────────────────────────────────────

retriever_start() {
    local index_file="${CACHE_ROOT}/data/searchR1/e5_Flat.index"
    local corpus_file="${CACHE_ROOT}/data/searchR1/wiki-18.jsonl"
    [[ -f "${index_file}" ]]  || die "wiki-18 index not found — run 'search' phase first"
    [[ -f "${corpus_file}" ]] || die "wiki-18 corpus not found — run 'search' phase first"

    if [[ -f "${RETRIEVER_PID_FILE}" ]]; then
        local old_pid
        old_pid="$(cat "${RETRIEVER_PID_FILE}")"
        if kill -0 "${old_pid}" 2>/dev/null; then
            log "Retrieval server already running (pid=${old_pid}) — skipping start"
            return
        fi
        rm -f "${RETRIEVER_PID_FILE}"
    fi

    local retriever_log="${LOG_DIR}/retrieval_server.log"
    mkdir -p "${LOG_DIR}"
    log "Starting retrieval server on port ${RETRIEVER_PORT} (log: ${retriever_log})"

    (
        export HF_HOME="${HF_HOME}"
        export PATH="${ENV_RETRIEVER}/bin:${PATH}"
        python3 \
            "${REPO_DIR}/examples/search/retriever/retrieval_server.py" \
            --index_path  "${index_file}" \
            --corpus_path "${corpus_file}" \
            --topk 3 \
            --retriever_name e5 \
            --retriever_model "${MODEL_E5}" \
            --faiss_gpu \
            --port "${RETRIEVER_PORT}" \
        >> "${retriever_log}" 2>&1
    ) &
    echo $! > "${RETRIEVER_PID_FILE}"
    log "Retriever server pid=$(cat "${RETRIEVER_PID_FILE}") — waiting 60s for startup"
    sleep 60

    # Lightweight health check: confirm the endpoint accepts requests
    if curl -sf --max-time 5 \
            -H 'Content-Type: application/json' \
            -d '{"query":"test","topk":1}' \
            "http://0.0.0.0:${RETRIEVER_PORT}/retrieve" > /dev/null; then
        log "Retrieval server healthy"
    else
        log "WARNING: retrieval server health-check failed — search training may error; check ${retriever_log}"
    fi
}

retriever_stop() {
    if [[ ! -f "${RETRIEVER_PID_FILE}" ]]; then return; fi
    local pid
    pid="$(cat "${RETRIEVER_PID_FILE}")"
    log "Stopping retrieval server (pid=${pid})"
    kill "${pid}" 2>/dev/null || true
    rm -f "${RETRIEVER_PID_FILE}"
    log "Retrieval server stopped"
}

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup trap — stop the retriever, then stamp the completion sentinel. The
# sentinel is read by the VM-side idle watchdog (gcp_sdar_preflight.sh) to drop
# into its short dead-run grace instead of waiting out the long idle grace.
# ─────────────────────────────────────────────────────────────────────────────
trap 'retriever_stop 2>/dev/null || true; date +%s > /run/sdar_run_exited 2>/dev/null || true' EXIT

# ─────────────────────────────────────────────────────────────────────────────
# PHASE: base
# Clone repo @ pinned commit; create 3 conda envs; symlink $HOME/data → cache.
# Skip any already-present artifact.
# ─────────────────────────────────────────────────────────────────────────────
phase_base() {
    log "=== PHASE: base ==="

    # Persistent disk layout
    mkdir -p "${CACHE_ROOT}" "${CONDA_ENVS_DIR}" "${HF_HOME}" "${LOG_DIR}" \
             "${CACHE_ROOT}/data"

    # Symlink $HOME/data → ${CACHE_ROOT}/data so all authors' scripts' hard-coded
    # $HOME/data/... paths resolve to the persistent cache automatically.
    if [[ -L "$HOME/data" ]]; then
        log "Symlink $HOME/data already exists → $(readlink "$HOME/data") — skipping"
    elif [[ -d "$HOME/data" ]]; then
        log "WARNING: $HOME/data is a real directory (not a symlink)."
        log "  Move its contents to ${CACHE_ROOT}/data manually, then remove $HOME/data"
        log "  before re-running so cache writes land on the persistent disk."
    else
        ln -sfn "${CACHE_ROOT}/data" "$HOME/data"
        log "Symlinked $HOME/data → ${CACHE_ROOT}/data"
    fi

    # Clone ZJU-REAL/SDAR at the pinned commit
    if [[ -d "${REPO_DIR}/.git" ]]; then
        log "Repo already at ${REPO_DIR} — skipping clone"
    else
        log "Cloning ${REPO_URL} @ ${REPO_COMMIT} into ${REPO_DIR}"
        git clone "${REPO_URL}" "${REPO_DIR}"
        git -C "${REPO_DIR}" checkout "${REPO_COMMIT}"
        log "Clone complete"
    fi

    _conda_init

    # ── conda env: sdar (py3.12, vllm 0.11.0) ──────────────────────────────
    # Used for: ALFWorld training, Search training
    if [[ -d "${ENV_SDAR}" ]]; then
        log "conda env sdar already at ${ENV_SDAR} — skipping"
    else
        log "Creating conda env: sdar (python=3.12)"
        conda create -p "${ENV_SDAR}" python=3.12 -y
        log "Installing vllm==0.11.0 (this takes ~10 min)"
        "${ENV_SDAR}/bin/pip3" install vllm==0.11.0
        log "Installing flash-attn==2.7.4.post1"
        "${ENV_SDAR}/bin/pip3" install flash-attn==2.7.4.post1 \
            --no-build-isolation --no-cache-dir
        log "Installing verl (pip install -e . from repo root)"
        (cd "${REPO_DIR}" && "${ENV_SDAR}/bin/pip" install -e .)
        log "sdar env ready"
    fi

    # ── conda env: verl-webshop (py3.10, vllm 0.8.2) ────────────────────────
    # Used for: WebShop training. NOTE: WebShop pins vllm 0.8.2, NOT 0.11.0 —
    # it MUST run in its own env. openjdk 11 is needed by pyserini/Lucene
    # during the 'webshop' phase; install it here so setup.sh finds it.
    if [[ -d "${ENV_WEBSHOP}" ]]; then
        log "conda env verl-webshop already at ${ENV_WEBSHOP} — skipping"
    else
        log "Creating conda env: verl-webshop (python=3.10)"
        conda create -p "${ENV_WEBSHOP}" python=3.10 -y
        log "Installing openjdk=11 (required by pyserini/Lucene index build)"
        conda install -p "${ENV_WEBSHOP}" -c conda-forge openjdk=11 -y
        log "verl-webshop base env created; full install done in 'webshop' phase"
    fi

    # ── conda env: retriever (py3.10, faiss-gpu) ─────────────────────────────
    # Used for: Search retrieval server. faiss-gpu is conda-only (no pip package).
    if [[ -d "${ENV_RETRIEVER}" ]]; then
        log "conda env retriever already at ${ENV_RETRIEVER} — skipping"
    else
        log "Creating conda env: retriever (python=3.10)"
        conda create -p "${ENV_RETRIEVER}" python=3.10 -y

        log "Pinning numpy==1.26.4 (prevents incompatible version via pip)"
        conda install -p "${ENV_RETRIEVER}" numpy=1.26.4 -y

        log "Installing torch 2.6.0 / cu124"
        "${ENV_RETRIEVER}/bin/pip" install \
            torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
            --index-url https://download.pytorch.org/whl/cu124

        log "Installing transformers/datasets/pyserini/huggingface_hub"
        "${ENV_RETRIEVER}/bin/pip" install \
            transformers datasets pyserini huggingface_hub

        log "Installing faiss-gpu==1.8.0 via conda (pip does not have GPU faiss)"
        conda install -p "${ENV_RETRIEVER}" faiss-gpu=1.8.0 -c pytorch -c nvidia -y

        log "Installing uvicorn + fastapi (retrieval server)"
        "${ENV_RETRIEVER}/bin/pip" install uvicorn fastapi

        log "retriever env ready"
    fi

    log "=== PHASE base DONE ==="
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE: alfworld
# pip install ALFWorld deps into sdar env; download game + detector data.
# Skip if already present.
# ─────────────────────────────────────────────────────────────────────────────
phase_alfworld() {
    log "=== PHASE: alfworld ==="
    [[ -d "${ENV_SDAR}" ]] || die "sdar env missing — run 'base' phase first"

    # Install ALFWorld Python packages into sdar env
    if "${ENV_SDAR}/bin/python3" -c "import alfworld" 2>/dev/null; then
        log "alfworld already importable in sdar env — skipping pip installs"
    else
        log "Installing gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld"
        "${ENV_SDAR}/bin/pip3" install gymnasium==0.29.1
        "${ENV_SDAR}/bin/pip3" install stable-baselines3==2.6.0
        "${ENV_SDAR}/bin/pip3" install alfworld
        log "ALFWorld packages installed"
    fi

    # Download ALFWorld game data.
    # ALFWORLD_DATA MUST be set BEFORE alfworld-download so games land in the
    # correct location (the package default is ~/.cache/alfworld, but run scripts
    # read $HOME/data/alfworld — honored via the $HOME/data symlink).
    export ALFWORLD_DATA="${CACHE_ROOT}/data/alfworld"
    mkdir -p "${ALFWORLD_DATA}"

    if [[ -d "${ALFWORLD_DATA}/json_2.1.1" ]]; then
        log "ALFWorld game data already at ${ALFWORLD_DATA}/json_2.1.1 — skipping download"
    else
        log "Downloading ALFWorld games + MaskRCNN detector (~3–5 GB) into ${ALFWORLD_DATA}"
        ALFWORLD_DATA="${ALFWORLD_DATA}" \
            "${ENV_SDAR}/bin/alfworld-download" -f
        log "ALFWorld data downloaded"
    fi

    log "=== PHASE alfworld DONE ==="
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE: webshop
# Run setup.sh -d all (downloads product JSON via gdown + builds Lucene index);
# complete verl-webshop env install (torch + flash-attn + verl + vllm 0.8.2).
# Skip index build if already present; retry gdown on rate-limit.
# ─────────────────────────────────────────────────────────────────────────────
phase_webshop() {
    log "=== PHASE: webshop ==="
    [[ -d "${ENV_WEBSHOP}" ]] || die "verl-webshop env missing — run 'base' phase first"

    local WEBSHOP_DIR="${REPO_DIR}/agent_system/environments/env_package/webshop/webshop"
    local INDEX_DIR="${WEBSHOP_DIR}/search_engine/indexes"

    # Index build is the slow/irreversible part — skip if present
    if [[ -d "${INDEX_DIR}" && -n "$(ls -A "${INDEX_DIR}" 2>/dev/null)" ]]; then
        log "WebShop Lucene index already at ${INDEX_DIR} — skipping setup.sh"
    else
        log "Running WebShop setup.sh -d all"
        log "  Step 1: pip install requirements.txt (webshop Python deps)"
        log "  Step 2: conda install mkl + faiss-cpu (via conda in verl-webshop env)"
        log "  Step 3: gdown product JSON from Google Drive (~3–5 GB, may rate-limit)"
        log "  Step 4: python -m spacy download en_core_web_lg/sm"
        log "  Step 5: pyserini Lucene index build (CPU + Java, ~10–30 min)"
        log "NOTE: gdown can hit Google Drive rate limits; wrapping in retry (5× / 30s)"

        # conda activate is needed so 'conda install mkl/faiss-cpu' inside setup.sh
        # targets verl-webshop, not the base env.
        _conda_init
        # Temporarily relax -e so conda shell functions don't trip the exit
        set +e
        conda activate "${ENV_WEBSHOP}"
        set -e

        (
            cd "${WEBSHOP_DIR}"
            # gdown calls inside setup.sh can fail — retry the whole setup.sh since
            # the index build at the end is fast to re-enter (prior gdown outputs are
            # cached as files in data/).
            retry bash setup.sh -d all
        )

        conda deactivate 2>/dev/null || true
        log "WebShop setup.sh complete"
    fi

    # Complete the verl-webshop env with torch/flash-attn/verl/vllm 0.8.2
    # (README says to do this AFTER setup.sh, from the repo root)
    if "${ENV_WEBSHOP}/bin/python3" -c \
            "import vllm; assert vllm.__version__=='0.8.2'" 2>/dev/null; then
        log "verl-webshop env already has vllm==0.8.2 — skipping install"
    else
        log "Installing torch 2.6.0 / cu124 into verl-webshop env"
        "${ENV_WEBSHOP}/bin/pip3" install torch==2.6.0 \
            --index-url https://download.pytorch.org/whl/cu124

        log "Installing flash-attn==2.7.4.post1 into verl-webshop env"
        "${ENV_WEBSHOP}/bin/pip3" install flash-attn==2.7.4.post1 \
            --no-build-isolation

        log "Installing verl (pip install -e . from repo root) into verl-webshop env"
        (cd "${REPO_DIR}" && "${ENV_WEBSHOP}/bin/pip" install -e .)

        log "Installing vllm==0.8.2 into verl-webshop env"
        "${ENV_WEBSHOP}/bin/pip3" install vllm==0.8.2

        log "verl-webshop env fully installed"
    fi

    # Quick import check: confirm WebAgentTextEnv is importable
    log "Verifying WebAgentTextEnv importability (in-process WebShop)"
    local webshop_pkg_dir="${WEBSHOP_DIR}"
    PYTHONPATH="${webshop_pkg_dir}:${REPO_DIR}" \
        "${ENV_WEBSHOP}/bin/python3" -c \
        "from web_agent_site.envs import WebAgentTextEnv; print('OK: WebAgentTextEnv importable')" \
        || log "WARNING: WebAgentTextEnv import failed — check verl-webshop env"

    log "=== PHASE webshop DONE ==="
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE: search
# Install skyrl_gym + gym into sdar env; preprocess QA data;
# download wiki-18 e5 index (~132 GB uncompressed) — the dominant cache cost.
# Skip each step if its artifact already exists.
# ─────────────────────────────────────────────────────────────────────────────
phase_search() {
    log "=== PHASE: search ==="
    [[ -d "${ENV_SDAR}" ]]      || die "sdar env missing — run 'base' phase first"
    [[ -d "${ENV_RETRIEVER}" ]] || die "retriever env missing — run 'base' phase first"

    # 1. Install skyrl_gym (Search-QA's RL gym) + gym==0.26.2 into sdar env
    local THIRD_PARTY="${REPO_DIR}/agent_system/environments/env_package/search/third_party"
    if "${ENV_SDAR}/bin/python3" -c "import skyrl_gym" 2>/dev/null; then
        log "skyrl_gym already importable in sdar env — skipping"
    else
        log "Installing skyrl_gym (third_party pip install -e .)"
        (cd "${THIRD_PARTY}" && "${ENV_SDAR}/bin/pip" install -e .)
        "${ENV_SDAR}/bin/pip" install "gym==0.26.2"
        log "skyrl_gym installed"
    fi

    # 2. Preprocess NQ + HotpotQA → parquet (PeterJinGo/nq_hotpotqa_train from HF)
    # Output lands at $HOME/data/searchR1_processed_direct/ via the $HOME/data symlink
    local SEARCH_DATA_DIR="${CACHE_ROOT}/data/searchR1_processed_direct"
    if [[ -f "${SEARCH_DATA_DIR}/train.parquet" && \
          -f "${SEARCH_DATA_DIR}/test.parquet" ]]; then
        log "Search-QA parquets already at ${SEARCH_DATA_DIR} — skipping preprocess"
    else
        log "Preprocessing Search-QA data (PeterJinGo/nq_hotpotqa_train, ~hundreds MB)"
        (
            cd "${REPO_DIR}"
            export HF_HOME="${HF_HOME}"
            export PATH="${ENV_SDAR}/bin:${PATH}"
            python3 -m examples.data_preprocess.preprocess_search_r1_dataset
        )
        log "Search-QA data preprocessed → ${SEARCH_DATA_DIR}"
    fi

    # 3. Download wiki-18 e5 FAISS index + corpus
    # Source: PeterJinGo/wiki-18-e5-index (part_aa + part_ab → cat → e5_Flat.index)
    #         PeterJinGo/wiki-18-corpus (wiki-18.jsonl.gz → decompress)
    # Size: ~60–70 GB download → ~132 GB uncompressed (repo-stated: docs/sglang_multiturn/)
    # This is the LARGEST single cache artifact — skip if already present.
    local WIKI18_DIR="${CACHE_ROOT}/data/searchR1"
    local INDEX_FILE="${WIKI18_DIR}/e5_Flat.index"
    local CORPUS_FILE="${WIKI18_DIR}/wiki-18.jsonl"

    mkdir -p "${WIKI18_DIR}"

    if [[ -f "${INDEX_FILE}" && -f "${CORPUS_FILE}" ]]; then
        log "wiki-18 e5 index + corpus already at ${WIKI18_DIR} — skipping (~132 GB)"
    else
        log "Downloading wiki-18 e5 index parts + corpus (~60–70 GB, expect 30–90 min)"
        log "  Source: PeterJinGo/wiki-18-e5-index (part_aa, part_ab)"
        log "          PeterJinGo/wiki-18-corpus (wiki-18.jsonl.gz)"
        log "  Retry on transient HuggingFace failures (up to 5×)"

        retry \
            HF_HOME="${HF_HOME}" \
            "${ENV_RETRIEVER}/bin/python3" \
            "${REPO_DIR}/examples/search/searchr1_download.py" \
            --local_dir "${WIKI18_DIR}"

        if [[ ! -f "${INDEX_FILE}" ]]; then
            log "Concatenating part_aa + part_ab → e5_Flat.index"
            cat "${WIKI18_DIR}/part_aa" "${WIKI18_DIR}/part_ab" > "${INDEX_FILE}"
            log "FAISS index assembled: ${INDEX_FILE} ($(du -sh "${INDEX_FILE}" | cut -f1))"
        else
            log "e5_Flat.index already assembled — skipping cat"
        fi

        if [[ ! -f "${CORPUS_FILE}" ]]; then
            log "Decompressing wiki-18.jsonl.gz (~60 GB → ~72 GB)"
            gzip -dk "${WIKI18_DIR}/wiki-18.jsonl.gz"   # -k keeps the .gz
            log "Corpus ready: ${CORPUS_FILE}"
        else
            log "wiki-18.jsonl already decompressed — skipping"
        fi
    fi

    log "=== PHASE search DONE ==="
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE: smoke
# Run ONE cheap WebShop Qwen3-1.7B cell (2 GPUs, total_epochs=2) to confirm
# the env earns non-zero reward. Uses sed to patch the authors' run script.
# ─────────────────────────────────────────────────────────────────────────────
phase_smoke() {
    log "=== PHASE: smoke ==="
    [[ -d "${ENV_WEBSHOP}" ]] || die "verl-webshop env missing — run 'base'+'webshop' phases first"

    local log_file="${LOG_DIR}/smoke_webshop_qwen3.log"
    local start_ts
    start_ts="$(date +%s)"

    log "Smoke: WebShop / Qwen3-1.7B / ${WEBSHOP_GPUS} GPUs / total_epochs=${SMOKE_EPOCHS}"
    log "Patching run_webshop_qwen3.sh: total_epochs=${TRAIN_EPOCHS}→${SMOKE_EPOCHS},"
    log "  test_freq=5→1, logger=['console','wandb']→['console'] (skip WandB for smoke)"

    run_script "run_webshop_qwen3.sh" "${ENV_WEBSHOP}" "$(seq -s, 0 $(( WEBSHOP_GPUS - 1 )))" \
        "${OOM_SED_EXPRS[@]}" \
        ${CKPT_SED_EXPRS[@]+"${CKPT_SED_EXPRS[@]}"} \
        "s/trainer\.total_epochs=${TRAIN_EPOCHS}/trainer.total_epochs=${SMOKE_EPOCHS}/" \
        "s/trainer\.test_freq=5/trainer.test_freq=1/" \
        "s/trainer\.logger=\['console','wandb'\]/trainer.logger=['console']/" \
        "s/trainer\.project_name='verl_agent_webshopv1'/trainer.project_name='verl_agent_webshopv1_smoke'/"

    local end_ts
    end_ts="$(date +%s)"
    local elapsed=$(( end_ts - start_ts ))
    log "Smoke completed in ${elapsed}s"

    # Extract final eval metric from captured log
    local last_reward
    last_reward="$(grep -oE 'val[/_](reward|win_rate)[/_]mean[^0-9]*[0-9]+\.[0-9]+' \
        "${log_file}" 2>/dev/null | tail -1 || true)"
    log "Smoke last val metric: ${last_reward:-'(not found — inspect ${log_file})'}"
    [[ -z "${last_reward}" ]] && \
        log "Tip: search the log for 'reward' or 'win_rate' to confirm non-zero learning"

    log "=== PHASE smoke DONE ==="
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE: grid
# Run all 9 authors' scripts at 150 steps with the correct algorithm knobs.
# Sequencing (4 GPU budget):
#   - WebShop: qwen3 + 3b in parallel (2+2 GPUs), then 7b alone (2 GPUs)
#   - Search: retriever up, qwen3 → 3b → 7b sequentially (4 GPUs each), retriever down
#   - ALFWorld: qwen3 → 3b → 7b sequentially (4 GPUs, overriding authors' 8)
# ─────────────────────────────────────────────────────────────────────────────
phase_grid() {
    log "=== PHASE: grid ==="
    log "Confirmed GPU budget: ${ACTUAL_GPUS}×A100-80GB"
    log "ALFWorld fidelity caveat: n_gpus_per_node overridden 8→${ALFWORLD_GPUS}."
    log "  WebShop runs are exact-match (2 GPUs); Search runs are exact-match (4 GPUs)."
    if [[ -n "${SDAR_CKPT_STEPS}" ]]; then
        log "Checkpoint/resume ON (SDAR_CKPT_STEPS): trainer.save_freq=${SDAR_CKPT_STEPS} + trainer.resume_mode=auto"
    fi

    [[ -d "${ENV_SDAR}" ]]      || die "sdar env missing — run 'base' phase"
    [[ -d "${ENV_WEBSHOP}" ]]   || die "verl-webshop env missing — run 'base'+'webshop' phases"
    [[ -d "${ENV_RETRIEVER}" ]] || die "retriever env missing — run 'base'+'search' phases"
    mkdir -p "${LOG_DIR}"

    # ── WebShop (2 GPUs per run) ─────────────────────────────────────────────
    # run qwen3 (GPUs 0,1) and 3b (GPUs 2,3) in parallel; then 7b alone.
    # All three use trainer.n_gpus_per_node=2 — exact match, no override needed.
    log "--- WEBSHOP: launching qwen3 (GPUs 0,1) + 3b (GPUs 2,3) in parallel"
    log "    (OOM knobs applied: micro_bsz=${OOM_MICRO_BSZ}, gpu_mem_util=${OOM_GPU_MEM_UTIL}, opt_offload=${OOM_OPT_OFFLOAD}, expandable_segments stripped)"
    (
        run_script "run_webshop_qwen3.sh" "${ENV_WEBSHOP}" "0,1" "${OOM_SED_EXPRS[@]}" \
            ${CKPT_SED_EXPRS[@]+"${CKPT_SED_EXPRS[@]}"} &
        PID_WS_Q=$!
        run_script "run_webshop_3b.sh"    "${ENV_WEBSHOP}" "2,3" "${OOM_SED_EXPRS[@]}" \
            ${CKPT_SED_EXPRS[@]+"${CKPT_SED_EXPRS[@]}"} &
        PID_WS_3=$!
        wait "${PID_WS_Q}" || die "run_webshop_qwen3.sh failed — check ${LOG_DIR}/run_webshop_qwen3.log"
        wait "${PID_WS_3}" || die "run_webshop_3b.sh failed — check ${LOG_DIR}/run_webshop_3b.log"
    )
    log "--- WEBSHOP: launching 7b (GPUs 0,1)"
    run_script "run_webshop_7b.sh" "${ENV_WEBSHOP}" "0,1" "${OOM_SED_EXPRS[@]}" \
        ${CKPT_SED_EXPRS[@]+"${CKPT_SED_EXPRS[@]}"}

    # ── Search (4 GPUs per run; retriever co-locates on same GPUs) ───────────
    # The retrieval server uses ~5–7 GB/GPU; training scripts set
    # actor_rollout_ref.rollout.gpu_memory_utilization=0.5 to leave room.
    # Keep the retriever alive across all 3 search runs to avoid costly restarts.
    log "--- SEARCH: starting retrieval server (uses ~5–7 GB/GPU)"
    retriever_start

    # Search launches stay byte-identical unless SDAR_CKPT_STEPS is explicitly
    # set (CKPT_SED_EXPRS is empty by default — the OOM knobs stay OFF here).
    log "--- SEARCH: running qwen3 (GPUs 0,1,2,3)"
    run_script "run_search_qwen3.sh" "${ENV_SDAR}" "0,1,2,3" \
        ${CKPT_SED_EXPRS[@]+"${CKPT_SED_EXPRS[@]}"}

    log "--- SEARCH: running 3b (GPUs 0,1,2,3)"
    run_script "run_search_3b.sh"    "${ENV_SDAR}" "0,1,2,3" \
        ${CKPT_SED_EXPRS[@]+"${CKPT_SED_EXPRS[@]}"}

    log "--- SEARCH: running 7b (GPUs 0,1,2,3)"
    run_script "run_search_7b.sh"    "${ENV_SDAR}" "0,1,2,3" \
        ${CKPT_SED_EXPRS[@]+"${CKPT_SED_EXPRS[@]}"}

    log "--- SEARCH: stopping retrieval server"
    retriever_stop

    # ── ALFWorld (authors: 8 GPUs → we use 4; see FIDELITY CAVEAT in CONFIG) ──
    # Patch trainer.n_gpus_per_node=8 → ${ALFWORLD_GPUS} via sed on the script.
    local ALF_GPU_SED="s/trainer\.n_gpus_per_node=8/trainer.n_gpus_per_node=${ALFWORLD_GPUS}/"

    log "    (OOM knobs applied: micro_bsz=${OOM_MICRO_BSZ}, gpu_mem_util=${OOM_GPU_MEM_UTIL}, opt_offload=${OOM_OPT_OFFLOAD}, expandable_segments stripped)"
    log "--- ALFWORLD: running qwen3 (${ALFWORLD_GPUS} GPUs, overriding authors' 8)"
    run_script "run_alfworld_qwen3.sh" "${ENV_SDAR}" "0,1,2,3" "${ALF_GPU_SED}" "${OOM_SED_EXPRS[@]}" \
        ${CKPT_SED_EXPRS[@]+"${CKPT_SED_EXPRS[@]}"}

    log "--- ALFWORLD: running 3b (${ALFWORLD_GPUS} GPUs)"
    run_script "run_alfworld_3b.sh"    "${ENV_SDAR}" "0,1,2,3" "${ALF_GPU_SED}" "${OOM_SED_EXPRS[@]}" \
        ${CKPT_SED_EXPRS[@]+"${CKPT_SED_EXPRS[@]}"}

    log "--- ALFWORLD: running 7b (${ALFWORLD_GPUS} GPUs)"
    run_script "run_alfworld_7b.sh"    "${ENV_SDAR}" "0,1,2,3" "${ALF_GPU_SED}" "${OOM_SED_EXPRS[@]}" \
        ${CKPT_SED_EXPRS[@]+"${CKPT_SED_EXPRS[@]}"}

    log "=== PHASE grid DONE ==="
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE: collect
# Parse captured training logs for final eval metrics; print a comparison table
# vs the paper's reported numbers; write a JSON summary.
# ─────────────────────────────────────────────────────────────────────────────
phase_collect() {
    log "=== PHASE: collect ==="
    local SUMMARY="${LOG_DIR}/sdar_repro_summary.json"

    # Paper Table-1 reported numbers at 150 steps.
    # Fill in from the paper (arxiv 2605.15155, Table 1) after confirming the
    # exact metric (win-rate for ALFWorld/WebShop, EM for Search).
    declare -A PAPER_METRIC=(
        [alfworld_qwen3]="TODO"  [alfworld_3b]="TODO"  [alfworld_7b]="TODO"
        [webshop_qwen3]="TODO"   [webshop_3b]="TODO"   [webshop_7b]="TODO"
        [search_qwen3]="TODO"    [search_3b]="TODO"     [search_7b]="TODO"
    )

    # Log file basename for each run (matches the tee target in run_script)
    declare -A LOG_NAME=(
        [alfworld_qwen3]="run_alfworld_qwen3"  [alfworld_3b]="run_alfworld_3b"   [alfworld_7b]="run_alfworld_7b"
        [webshop_qwen3]="run_webshop_qwen3"    [webshop_3b]="run_webshop_3b"     [webshop_7b]="run_webshop_7b"
        [search_qwen3]="run_search_qwen3"      [search_3b]="run_search_3b"       [search_7b]="run_search_7b"
    )

    # Success-signal notes (for context)
    declare -A METRIC_NOTE=(
        [alfworld_qwen3]="mean(won), reward=10*won"  [alfworld_3b]="mean(won)"  [alfworld_7b]="mean(won)"
        [webshop_qwen3]="mean(won), reward=10 on win"  [webshop_3b]="mean(won)"  [webshop_7b]="mean(won)"
        [search_qwen3]="exact-match EM, em_check"    [search_3b]="EM"           [search_7b]="EM"
    )

    local ordered=(
        alfworld_qwen3 alfworld_3b alfworld_7b
        webshop_qwen3  webshop_3b  webshop_7b
        search_qwen3   search_3b   search_7b
    )

    printf '\n%-26s %-22s %-14s %-30s\n' "RUN" "REPRO_METRIC" "PAPER" "SUCCESS_SIGNAL"
    printf '%0.s─' {1..92}; printf '\n'

    local json_entries=""
    for key in "${ordered[@]}"; do
        local log_file="${LOG_DIR}/${LOG_NAME[$key]}.log"
        local repro_val="N/A"

        if [[ -f "${log_file}" ]]; then
            # verl logs structured metric dicts; capture the last occurrence of a
            # val reward or EM value.  The exact format (e.g. 'val/reward/mean')
            # depends on the verl version — adjust the pattern if it doesn't match.
            repro_val="$(grep -oE \
                '(val[/_]reward[/_]mean|val[/_]win_rate|reward[/_]mean|em[/_]score|val[/_]em)[^,}]*[0-9]+\.[0-9]+' \
                "${log_file}" 2>/dev/null | tail -1 || echo "(metric line not found)")"
        else
            repro_val="(log missing: ${log_file})"
        fi

        printf '%-26s %-22s %-14s %-30s\n' \
            "${key}" "${repro_val}" "${PAPER_METRIC[$key]}" "${METRIC_NOTE[$key]}"

        # Escape for JSON
        local repro_escaped="${repro_val//\"/\\\"}"
        local paper_escaped="${PAPER_METRIC[$key]//\"/\\\"}"
        json_entries+="    {\"run\":\"${key}\",\"repro\":\"${repro_escaped}\","
        json_entries+="\"paper\":\"${paper_escaped}\","
        json_entries+="\"metric_note\":\"${METRIC_NOTE[$key]}\","
        json_entries+="\"log\":\"${log_file}\"},"$'\n'
    done

    printf '%0.s─' {1..92}; printf '\n'
    printf 'Logs dir: %s\n' "${LOG_DIR}"
    printf 'Summary : %s\n\n' "${SUMMARY}"

    # Write JSON summary
    {
        printf '{\n'
        printf '  "generated_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '  "repo_url": "%s",\n' "${REPO_URL}"
        printf '  "repo_commit": "%s",\n' "${REPO_COMMIT}"
        printf '  "train_epochs_alfworld_webshop": %s,\n' "${TRAIN_EPOCHS}"
        printf '  "train_steps_search": %s,\n' "${SEARCH_STEPS}"
        printf '  "alfworld_gpu_count_used": %s,\n' "${ALFWORLD_GPUS}"
        printf '  "alfworld_gpu_count_authors": 8,\n'
        printf '  "sdar_coef": 0.01,\n'
        printf '  "gate_beta": 5.0,\n'
        printf '  "skill_all": false,\n'
        printf '  "results": [\n'
        # Strip trailing comma+newline from last entry
        printf '%s' "${json_entries%,$'\n'}"
        printf '\n  ]\n}\n'
    } > "${SUMMARY}"

    log "Summary JSON written to ${SUMMARY}"
    log "=== PHASE collect DONE ==="
}

# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────────────────────

PHASE="${1:-}"

case "${PHASE}" in
    base)     phase_base ;;
    alfworld) phase_alfworld ;;
    webshop)  phase_webshop ;;
    search)   phase_search ;;
    smoke)    phase_smoke ;;
    grid)     phase_grid ;;
    collect)  phase_collect ;;
    all)
        phase_base
        phase_alfworld
        phase_webshop
        phase_search
        phase_smoke
        phase_grid
        phase_collect
        ;;
    *)
        cat >&2 <<'EOF'
Usage: bash scripts/sdar_authors_repro.sh <phase>

Phases (run in this order for a full reproduction):
  base      Clone ZJU-REAL/SDAR@9f2ce6a; create 3 conda envs (sdar/verl-webshop/retriever);
            symlink $HOME/data → /mnt/sdar-cache/data.
  alfworld  Install ALFWorld deps into sdar env; download game data via alfworld-download.
  webshop   Run WebShop setup.sh -d all (gdown product JSON + Lucene index build);
            complete verl-webshop env (torch/flash-attn/verl/vllm 0.8.2).
  search    Install skyrl_gym; preprocess NQ+HotpotQA; download wiki-18 e5 index (~132 GB).
  smoke     1 cheap cell: WebShop/Qwen3-1.7B/2 GPUs/total_epochs=2 — confirms env works.
  grid      Run all 9 authors' run scripts at 150 steps with correct SDAR knobs.
  collect   Parse training logs; print metric table vs paper; write JSON summary.
  all       Run all phases in order.

Cache root : /mnt/sdar-cache  (set CACHE_ROOT in the config block to override)
HF cache   : /mnt/sdar-cache/hf
Conda envs : /mnt/sdar-cache/conda/envs/{sdar,verl-webshop,retriever}
Logs       : /mnt/sdar-cache/logs/
EOF
        exit 1
        ;;
esac
