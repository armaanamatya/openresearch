#!/usr/bin/env bash
# webshop_stage_assets.sh — deterministic, resumable re-staging of ALL WebShop
# training assets onto a mounted cache disk.
#
# WHY: every WebShop asset (product corpus + Lucene index, built 2026-07-04,
# "indexes: 3.0M" per webshop_index_build.log) died with the July cache disk —
# the GCS write-through failed 6x with GcsApiError, so nothing survived in the
# bucket, and the replacement disk (sdar-cache-restore) has NO webshop data.
# See docs/runbooks/2026-08-03-webshop-asset-restaging.md (the runbook for this
# script) and docs/periods/2026-07.md ("WebShop — never trained").
#
# WHAT IT STAGES (both harness paths):
#   * the product corpus (items_shuffle.json / items_ins_v2.json + 1K subsets)
#     at the WebShop package root — this IS the "BM25 artifact" the harness
#     in-process path needs: backend/agents/rlm/webshop_env.py builds its
#     rank_bm25 BM25Okapi searcher at env construction directly from
#     items_shuffle.json (no separate on-disk index), and the adapter smoke
#     (backend/services/runtime/env_adapters/webshop.py) checks exactly
#     <WEBSHOP_DATA_DIR>/items_shuffle.json + items_ins_v2.json.
#   * the Lucene/pyserini index (<pkg>/search_engine/indexes) the FAITHFUL
#     path needs — used when the WebShop cell runs under the dedicated py3.10
#     interpreter via the OPENRESEARCH_WEBSHOP_PYTHON per-cell seam
#     (backend/agents/rlm/gpu_cell_runner.py::_cell_interpreter).
#
# Pipeline provenance — nothing here is invented:
#   * repo/commit, conda-env recipe, and the setup.sh -d all index build are
#     lifted from the proven scripts/sdar_authors_repro.sh phase_webshop
#     (base + webshop phases).
#   * corpus mirror URLs + SHA256/size pins are the byte-identical 3-way-
#     verified values from backend/services/runtime/asset_prestage.py
#     (verified 2026-07-03).
#
# USAGE:
#   bash scripts/webshop_stage_assets.sh /mnt/sdar-cache
#
# Idempotent + resumable: each stage writes a sentinel under
# <cache>/webshop_staging_state/ and is skipped on re-run when complete;
# corpus downloads resume (curl -C -). Re-run after any failure.
# A cheap no-GPU VM suffices (corpus fetch + Lucene build are CPU+Java only).
#
# Optional env knobs (all have grounded defaults):
#   SDAR_REPO_URL       (default https://github.com/ZJU-REAL/SDAR.git)
#   SDAR_REPO_COMMIT    (default 9f2ce6a — the pin sdar_authors_repro.sh uses)
#   WEBSHOP_SETUP_ARGS  (default "-d all" — args to the authors' setup.sh)
#   HF_TOKEN            (optional; mirrors are anonymously fetchable)
#   WEBSHOP_STAGE_FORCE (default 0; 1 = ignore stage sentinels and redo)
#   WEBSHOP_STAGE_TRAIN_STACK (default 0; 1 = ALSO complete the verl-webshop
#       TRAINING stack — torch cu124 + flash-attn + verl + vllm 0.8.2. Needs
#       the CUDA toolchain, so leave 0 on a CPU staging VM and run it later on
#       the GPU host, or via sdar_authors_repro.sh's webshop phase.)
#
# NOT committed to any cloud action by itself: this script only writes under
# the given cache root. bash -n clean.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# ARGS + CONFIG
# ─────────────────────────────────────────────────────────────────────────────

[[ $# -ge 1 ]] || { echo "usage: $0 <cache_root e.g. /mnt/sdar-cache>" >&2; exit 2; }
CACHE_ROOT="$1"
[[ -d "${CACHE_ROOT}" ]] || { echo "[ERROR] cache root not a directory: ${CACHE_ROOT} (is the cache disk mounted?)" >&2; exit 2; }

SDAR_REPO_URL="${SDAR_REPO_URL:-https://github.com/ZJU-REAL/SDAR.git}"
SDAR_REPO_COMMIT="${SDAR_REPO_COMMIT:-9f2ce6a}"
WEBSHOP_SETUP_ARGS="${WEBSHOP_SETUP_ARGS:--d all}"
WEBSHOP_STAGE_FORCE="${WEBSHOP_STAGE_FORCE:-0}"
WEBSHOP_STAGE_TRAIN_STACK="${WEBSHOP_STAGE_TRAIN_STACK:-0}"
HF_TOKEN="${HF_TOKEN:-}"

REPO_DIR="${CACHE_ROOT}/SDAR"
# The vendored WebShop package (== the harness WEBSHOP_DATA_DIR / the dir
# env_adapters/webshop.py auto-detects as WEBSHOP_PACKAGE_DIR's parent layout).
WEBSHOP_DIR="${REPO_DIR}/agent_system/environments/env_package/webshop/webshop"
INDEX_DIR="${WEBSHOP_DIR}/search_engine/indexes"
DATA_DIR="${WEBSHOP_DIR}/data"

CONDA_ENVS_DIR="${CACHE_ROOT}/conda/envs"
ENV_WEBSHOP="${CONDA_ENVS_DIR}/verl-webshop"   # py3.10 + openjdk 11

STATE_DIR="${CACHE_ROOT}/webshop_staging_state"
MANIFEST="${STATE_DIR}/checksums.manifest"
COMPLETE_SENTINEL="${WEBSHOP_DIR}/STAGING_COMPLETE"

# Corpus pins — byte-identical to backend/services/runtime/asset_prestage.py
# (3-way cross-mirror verified 2026-07-03). Do not edit here without editing
# there: asset_prestage.py is the source of truth.
ITEMS_SHUFFLE_SHA256="2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8"
ITEMS_SHUFFLE_MIN_SIZE=5479720229
ITEMS_INS_V2_SHA256="1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e"
ITEMS_INS_V2_MIN_SIZE=186295270
# 1K subsets: size floors only (no cross-verified sha upstream either).
ITEMS_SHUFFLE_1000_MIN_SIZE=4467013
ITEMS_INS_V2_1000_MIN_SIZE=147099

# Ordered anonymous HF mirrors (first hit that passes integrity wins).
HF_MIRRORS_FULL=(
    "YWZBrandon/webshop-data"
    "HongbangYuan/webshop"
    "quanwei0/webshop-minimal"
)
HF_MIRRORS_1000=(
    "YWZBrandon/webshop-data"
    "HongbangYuan/webshop"
)
# Operator-only last resort (needs authenticated gdown/rclone; NOT automated
# here — printed in the failure message): items_shuffle.json =
# gdrive:1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib, items_ins_v2.json =
# gdrive:1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu (asset_prestage.py).

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

stage_done() {  # stage_done <name> → 0 if sentinel present and not forced
    [[ "${WEBSHOP_STAGE_FORCE}" != "1" && -f "${STATE_DIR}/$1.done" ]]
}
mark_done() { date -u +"%Y-%m-%dT%H:%M:%SZ" > "${STATE_DIR}/$1.done"; }

retry() {  # retry <cmd...> — 5 attempts, 30s apart (gdown/HF rate limits)
    local n=0
    until "$@"; do
        n=$((n + 1))
        [[ ${n} -ge 5 ]] && die "command failed after 5 attempts: $*"
        log "attempt ${n}/5 failed; retrying in 30s: $*"
        sleep 30
    done
}

file_sha256() { sha256sum "$1" | awk '{print $1}'; }
file_size() { stat -c '%s' "$1" 2>/dev/null || stat -f '%z' "$1"; }

# verify_file <path> <min_size> [sha256] → 0 iff present + integral
verify_file() {
    local path="$1" min_size="$2" want_sha="${3:-}"
    [[ -f "${path}" ]] || return 1
    local size; size="$(file_size "${path}")"
    [[ "${size}" -ge "${min_size}" ]] || { log "integrity: ${path} size ${size} < ${min_size}"; return 1; }
    if [[ -n "${want_sha}" ]]; then
        local got; got="$(file_sha256 "${path}")"
        [[ "${got}" == "${want_sha}" ]] || { log "integrity: ${path} sha256 ${got} != ${want_sha}"; return 1; }
    fi
    return 0
}

# fetch_corpus_file <filename> <min_size> <sha256-or-empty> <mirror...>
# Downloads into ${DATA_DIR}/<filename> (resumable), trying mirrors in order;
# only a candidate passing the integrity pin is kept. Fails loud otherwise.
fetch_corpus_file() {
    local fname="$1" min_size="$2" want_sha="$3"; shift 3
    local dest="${DATA_DIR}/${fname}"
    if verify_file "${dest}" "${min_size}" "${want_sha}"; then
        log "corpus: ${fname} already present + integral — skipping"
        return 0
    fi
    local -a auth=()
    [[ -n "${HF_TOKEN}" ]] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")
    local repo url
    for repo in "$@"; do
        url="https://huggingface.co/datasets/${repo}/resolve/main/${fname}"
        log "corpus: fetching ${fname} from ${repo} (resumable)"
        # -C - resumes a partial file; a corrupt/stale partial that then fails
        # the pin is removed so the next mirror starts clean.
        if curl -fL --retry 5 --retry-delay 15 -C - "${auth[@]+"${auth[@]}"}" \
                -o "${dest}" "${url}"; then
            if verify_file "${dest}" "${min_size}" "${want_sha}"; then
                log "corpus: ${fname} OK from ${repo}"
                return 0
            fi
            log "corpus: ${fname} from ${repo} FAILED the integrity pin — discarding"
        else
            log "corpus: ${fname} fetch from ${repo} failed"
        fi
        rm -f "${dest}"
    done
    die "corpus: could not stage ${fname} from any mirror (${*}). Last resort: authenticated gdown of the gdrive ids in backend/services/runtime/asset_prestage.py, then re-run (the file is picked up in place)."
}

# link_to_pkg_root <filename> — the harness in-process smoke expects the corpus
# at the WEBSHOP_DATA_DIR (package) ROOT; setup.sh/gdown cache under data/.
# Hardlink (same filesystem) so the 5.5 GB corpus isn't duplicated; fall back
# to copy.
link_to_pkg_root() {
    local fname="$1"
    [[ -f "${WEBSHOP_DIR}/${fname}" ]] && return 0
    ln "${DATA_DIR}/${fname}" "${WEBSHOP_DIR}/${fname}" 2>/dev/null \
        || cp "${DATA_DIR}/${fname}" "${WEBSHOP_DIR}/${fname}"
}

_conda_init() {
    command -v conda >/dev/null 2>&1 || die "conda not on PATH (miniconda/mambaforge required — see runbook)"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
}

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: preflight — fail loud on missing deps BEFORE any slow work
# ─────────────────────────────────────────────────────────────────────────────
stage_preflight() {
    log "=== STAGE 1: preflight ==="
    local missing=()
    local tool
    for tool in git curl sha256sum awk stat conda; do
        command -v "${tool}" >/dev/null 2>&1 || missing+=("${tool}")
    done
    [[ ${#missing[@]} -eq 0 ]] || die "missing required tools: ${missing[*]} (install them, then re-run)"

    # Disk: corpus ~5.7 GB + repo + conda env + index headroom → require 30 GB.
    local free_kb
    free_kb="$(df -Pk "${CACHE_ROOT}" | awk 'NR==2 {print $4}')"
    [[ "${free_kb}" -ge $((30 * 1024 * 1024)) ]] \
        || die "need >= 30 GB free on ${CACHE_ROOT}; have $((free_kb / 1024 / 1024)) GB"

    mkdir -p "${STATE_DIR}" "${CONDA_ENVS_DIR}"
    log "preflight OK (tools present, $((free_kb / 1024 / 1024)) GB free)"
}

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: repo — clone + pin (mirrors sdar_authors_repro.sh)
# ─────────────────────────────────────────────────────────────────────────────
stage_repo() {
    log "=== STAGE 2: repo (${SDAR_REPO_URL} @ ${SDAR_REPO_COMMIT}) ==="
    if [[ -d "${REPO_DIR}/.git" ]]; then
        log "repo already at ${REPO_DIR}"
    else
        git clone "${SDAR_REPO_URL}" "${REPO_DIR}"
    fi
    git -C "${REPO_DIR}" checkout "${SDAR_REPO_COMMIT}"
    [[ -d "${WEBSHOP_DIR}" ]] || die "webshop package dir missing after checkout: ${WEBSHOP_DIR}"
    mkdir -p "${DATA_DIR}"
}

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: conda env — verl-webshop (py3.10 + openjdk 11 for pyserini/Lucene)
# ─────────────────────────────────────────────────────────────────────────────
stage_conda_env() {
    log "=== STAGE 3: conda env verl-webshop (python=3.10, openjdk=11) ==="
    _conda_init
    if [[ -x "${ENV_WEBSHOP}/bin/python3" ]]; then
        log "env already at ${ENV_WEBSHOP}"
    else
        conda create -p "${ENV_WEBSHOP}" python=3.10 -y
        conda install -p "${ENV_WEBSHOP}" -c conda-forge openjdk=11 -y
    fi
    # Java preflight for the Lucene stage — fail loud NOW, not mid-index-build.
    "${ENV_WEBSHOP}/bin/java" -version 2>/dev/null \
        || conda run -p "${ENV_WEBSHOP}" java -version \
        || die "java not runnable inside ${ENV_WEBSHOP} (openjdk=11 install failed?) — pyserini index build would die"
}

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: corpus — pinned-checksum fetch from HF mirrors (resumable)
# ─────────────────────────────────────────────────────────────────────────────
stage_corpus() {
    log "=== STAGE 4: product corpus (pins from asset_prestage.py) ==="
    fetch_corpus_file "items_shuffle.json" "${ITEMS_SHUFFLE_MIN_SIZE}" "${ITEMS_SHUFFLE_SHA256}" "${HF_MIRRORS_FULL[@]}"
    fetch_corpus_file "items_ins_v2.json" "${ITEMS_INS_V2_MIN_SIZE}" "${ITEMS_INS_V2_SHA256}" "${HF_MIRRORS_FULL[@]}"
    fetch_corpus_file "items_shuffle_1000.json" "${ITEMS_SHUFFLE_1000_MIN_SIZE}" "" "${HF_MIRRORS_1000[@]}"
    fetch_corpus_file "items_ins_v2_1000.json" "${ITEMS_INS_V2_1000_MIN_SIZE}" "" "${HF_MIRRORS_1000[@]}"
    # Harness in-process (rank_bm25) path reads the corpus at the pkg ROOT.
    link_to_pkg_root "items_shuffle.json"
    link_to_pkg_root "items_ins_v2.json"
    link_to_pkg_root "items_shuffle_1000.json"
    link_to_pkg_root "items_ins_v2_1000.json"
    log "corpus staged at ${DATA_DIR} (+ hardlinks at ${WEBSHOP_DIR})"
}

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5: setup + Lucene index — the authors' setup.sh (pip reqs, mkl +
# faiss-cpu, spacy models, pyserini Lucene build; CPU + Java, ~10-30 min).
# Corpus is already in place from stage 4, so a gdown rate-limit inside
# setup.sh only re-fetches what it insists on; the whole thing is retried.
# ─────────────────────────────────────────────────────────────────────────────
stage_setup_and_index() {
    log "=== STAGE 5: setup.sh ${WEBSHOP_SETUP_ARGS} + Lucene/pyserini index ==="
    if [[ -d "${INDEX_DIR}" && -n "$(ls -A "${INDEX_DIR}" 2>/dev/null)" ]]; then
        log "Lucene index already at ${INDEX_DIR} — skipping setup.sh"
        return 0
    fi
    [[ -f "${WEBSHOP_DIR}/setup.sh" ]] || die "setup.sh missing at ${WEBSHOP_DIR} (wrong commit?)"
    _conda_init
    # conda activate so 'conda install mkl/faiss-cpu' inside setup.sh targets
    # verl-webshop, not base (same dance as sdar_authors_repro.sh).
    set +e
    conda activate "${ENV_WEBSHOP}"
    set -e
    (
        cd "${WEBSHOP_DIR}"
        # shellcheck disable=SC2086 — WEBSHOP_SETUP_ARGS is intentionally word-split
        retry bash setup.sh ${WEBSHOP_SETUP_ARGS}
    )
    conda deactivate 2>/dev/null || true
    [[ -d "${INDEX_DIR}" && -n "$(ls -A "${INDEX_DIR}" 2>/dev/null)" ]] \
        || die "setup.sh finished but the Lucene index is missing/empty at ${INDEX_DIR}"
    log "Lucene index built at ${INDEX_DIR} ($(du -sh "${INDEX_DIR}" | awk '{print $1}'))"
}

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 (OPTIONAL, default OFF): complete the TRAINING stack in verl-webshop
# (torch cu124 + flash-attn + verl + vllm 0.8.2 — needs the CUDA toolchain, so
# run on the GPU host, not the cheap staging VM). Mirrors phase_webshop.
# ─────────────────────────────────────────────────────────────────────────────
stage_train_stack() {
    log "=== STAGE 6: verl-webshop training stack (torch/flash-attn/verl/vllm 0.8.2) ==="
    if "${ENV_WEBSHOP}/bin/python3" -c "import vllm; assert vllm.__version__=='0.8.2'" 2>/dev/null; then
        log "vllm==0.8.2 already installed — skipping"
        return 0
    fi
    "${ENV_WEBSHOP}/bin/pip3" install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    "${ENV_WEBSHOP}/bin/pip3" install flash-attn==2.7.4.post1 --no-build-isolation
    (cd "${REPO_DIR}" && "${ENV_WEBSHOP}/bin/pip" install -e .)
    "${ENV_WEBSHOP}/bin/pip3" install vllm==0.8.2
}

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7: verify + manifest + STAGING_COMPLETE
# ─────────────────────────────────────────────────────────────────────────────
stage_verify_and_manifest() {
    log "=== STAGE 7: verify + checksums manifest + STAGING_COMPLETE ==="
    # In-process (rank_bm25) harness path: corpus at pkg root + imports OK.
    verify_file "${WEBSHOP_DIR}/items_shuffle.json" "${ITEMS_SHUFFLE_MIN_SIZE}" "${ITEMS_SHUFFLE_SHA256}" \
        || die "verify: items_shuffle.json missing/corrupt at ${WEBSHOP_DIR}"
    verify_file "${WEBSHOP_DIR}/items_ins_v2.json" "${ITEMS_INS_V2_MIN_SIZE}" "${ITEMS_INS_V2_SHA256}" \
        || die "verify: items_ins_v2.json missing/corrupt at ${WEBSHOP_DIR}"
    PYTHONPATH="${WEBSHOP_DIR}:${REPO_DIR}" "${ENV_WEBSHOP}/bin/python3" -c \
        "import rank_bm25; from web_agent_site.envs import WebAgentTextEnv; print('OK: web_agent_site + rank_bm25 importable')" \
        || die "verify: web_agent_site/rank_bm25 not importable in ${ENV_WEBSHOP} (setup.sh stage incomplete?)"
    # Faithful (Lucene) path: non-empty index.
    [[ -d "${INDEX_DIR}" && -n "$(ls -A "${INDEX_DIR}" 2>/dev/null)" ]] \
        || die "verify: Lucene index missing/empty at ${INDEX_DIR}"

    {
        echo "# WebShop staging manifest — $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        echo "# repo=${SDAR_REPO_URL} commit=$(git -C "${REPO_DIR}" rev-parse HEAD)"
        local f
        for f in items_shuffle.json items_ins_v2.json items_shuffle_1000.json items_ins_v2_1000.json; do
            echo "$(file_sha256 "${DATA_DIR}/${f}")  data/${f}  $(file_size "${DATA_DIR}/${f}")"
        done
        echo "# index_dir=${INDEX_DIR} size=$(du -sh "${INDEX_DIR}" | awk '{print $1}') files=$(find "${INDEX_DIR}" -type f | wc -l)"
    } > "${MANIFEST}"
    log "manifest written: ${MANIFEST}"

    {
        echo "staged_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        echo "host=$(hostname)"
        echo "repo_commit=$(git -C "${REPO_DIR}" rev-parse HEAD)"
        echo "manifest=${MANIFEST}"
        echo "webshop_python=${ENV_WEBSHOP}/bin/python3"
    } > "${COMPLETE_SENTINEL}"
    log "STAGING_COMPLETE written: ${COMPLETE_SENTINEL}"
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — ordered stages, each behind a sentinel
# ─────────────────────────────────────────────────────────────────────────────
main() {
    mkdir -p "${STATE_DIR}"
    local -a stages=(preflight repo conda_env corpus setup_and_index)
    [[ "${WEBSHOP_STAGE_TRAIN_STACK}" == "1" ]] && stages+=(train_stack)
    stages+=(verify_and_manifest)

    local s
    for s in "${stages[@]}"; do
        if stage_done "${s}"; then
            log "stage ${s}: already complete (sentinel) — skipping"
            continue
        fi
        "stage_${s}"
        mark_done "${s}"
    done

    log "ALL STAGES COMPLETE."
    log "Export for the harness / training host:"
    log "  export WEBSHOP_DATA_DIR=${WEBSHOP_DIR}"
    log "  export WEBSHOP_PACKAGE_DIR=${WEBSHOP_DIR}"
    log "  export OPENRESEARCH_WEBSHOP_PYTHON=${ENV_WEBSHOP}/bin/python3"
    log "Training additionally needs the verl stack (WEBSHOP_STAGE_TRAIN_STACK=1"
    log "on the GPU host, or sdar_authors_repro.sh base+webshop phases) and the"
    log "OOM-knobbed run_webshop_3b launch — see the restaging runbook."
}

main "$@"
