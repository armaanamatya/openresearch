<!-- doc-meta: status=current; last-verified=2026-08-03 -->
# WebShop asset re-staging (corpus + Lucene index) on the restored cache disk

**Date:** 2026-08-03 · **Status:** current · **Script:** `scripts/webshop_stage_assets.sh`

## Why (the assets are gone)

Every WebShop training asset is dead:

- The corpus + Lucene index were built **2026-07-04** on the July cache disk
  (`webshop_index_build.log`: "indexes: 3.0M") — that disk is deleted.
- The GCS write-through of `items_*.json`/`checksums.actual` failed **6× with
  `GcsApiError('')`** — nothing landed in the bucket
  (docs/periods/2026-07.md §4, "WebShop — never trained").
- The surviving cache disk **`sdar-cache-restore`** (GCP, `us-central1-a`) has
  **no webshop data** at all.

WebShop training has therefore *never* fired; before it can, the assets must be
re-staged from scratch. The corpus is refetchable (verified anonymous HF
mirrors, checksum-pinned in `backend/services/runtime/asset_prestage.py`) and
the index is rebuildable (CPU + Java), so this is cheap — no GPU required.

## Where to run it

- A VM with **`sdar-cache-restore` attached and mounted** (canonically at
  `/mnt/sdar-cache`, `us-central1-a`).
- A **cheap no-GPU machine type suffices** (e.g. `e2-standard-8`): the work is
  a ~5.7 GB download + a CPU/Java pyserini index build. Do NOT lease a GPU for
  staging (see the cpu-warm-disk pattern in
  `backend/services/runtime/CLAUDE.md`).
- Prereqs on the VM: `git`, `curl`, `sha256sum`, and **conda**
  (miniconda/mambaforge) on PATH. The script fail-louds on anything missing
  (stage 1 preflight) and installs Java itself (conda `openjdk=11` inside the
  `verl-webshop` env).

## Invocation

```bash
# on the VM, cache disk mounted:
bash scripts/webshop_stage_assets.sh /mnt/sdar-cache
```

Deterministic + resumable: each stage writes a sentinel under
`/mnt/sdar-cache/webshop_staging_state/` and is skipped when already complete;
the corpus download resumes partial files (`curl -C -`). Re-run the same
command after any failure. `WEBSHOP_STAGE_FORCE=1` redoes everything.

What it stages (each stage is idempotent):

1. **Preflight** — tools + ≥30 GB free, fail loud.
2. **Repo** — clone/pin `ZJU-REAL/SDAR` @ `9f2ce6a` (same pin as
   `scripts/sdar_authors_repro.sh`) → the vendored WebShop package at
   `/mnt/sdar-cache/SDAR/agent_system/environments/env_package/webshop/webshop`.
3. **Conda env** — `verl-webshop` (python 3.10 + `openjdk=11`) at
   `/mnt/sdar-cache/conda/envs/verl-webshop`; Java preflight here, not
   mid-build.
4. **Corpus** — `items_shuffle.json`, `items_ins_v2.json` (+ the 1K subsets)
   from the ordered, 3-way-verified HF mirrors, checked against the pinned
   SHA256/size from `asset_prestage.py`; hardlinked to the package root (where
   the harness in-process smoke looks). This corpus **is** the
   "BM25 artifact" — `webshop_env.py` builds its `rank_bm25` searcher from
   `items_shuffle.json` at env construction; there is no separate BM25 file.
5. **setup.sh + Lucene index** — the authors' `setup.sh -d all` under the
   `verl-webshop` env (pip reqs, mkl+faiss-cpu, spacy models, **pyserini
   Lucene build** → `search_engine/indexes`), 5×-retried (gdown rate limits).
6. *(optional, `WEBSHOP_STAGE_TRAIN_STACK=1`, GPU host only)* — the training
   stack: torch 2.6.0/cu124 + flash-attn + `verl -e .` + **vllm 0.8.2**.
7. **Verify + manifest** — corpus pins re-checked, `web_agent_site` +
   `rank_bm25` import under the dedicated interpreter, index non-empty →
   `webshop_staging_state/checksums.manifest` + a **`STAGING_COMPLETE`**
   sentinel at the package root.

## Expected sizes / durations (from history)

| Artifact | Size | Duration | Source |
|---|---|---|---|
| `items_shuffle.json` | 5,479,720,229 B (~5.5 GB), sha-pinned | dominated by bandwidth | `asset_prestage.py` |
| `items_ins_v2.json` | 186,295,270 B (~186 MB), sha-pinned | minutes | `asset_prestage.py` |
| 1K subsets | ~4.5 MB / ~147 KB (size floors) | seconds | `asset_prestage.py` |
| Lucene/pyserini index | historical log recorded "`indexes: 3.0M`" (quote verbatim; re-measure on rebuild) | **~10–30 min, CPU + Java** | `webshop_index_build.log` refs; docs/periods/2026-07.md |
| gdown route (fallback inside setup.sh) | corpus ~3–5 GB, rate-limit-prone | retried 5×/30s | docs/periods/2026-07.md |

## Verification checklist (after the script reports ALL STAGES COMPLETE)

- [ ] `cat <pkg>/STAGING_COMPLETE` — records timestamp, host, repo commit,
      manifest path, interpreter path.
- [ ] `webshop_staging_state/checksums.manifest` — the two big files match the
      pins in `asset_prestage.py` byte-for-byte.
- [ ] `ls <pkg>/search_engine/indexes` non-empty.
- [ ] In-process smoke (what the adapter runs):
      `PYTHONPATH=<pkg> /mnt/sdar-cache/conda/envs/verl-webshop/bin/python3 -c
      "from web_agent_site.envs import WebAgentTextEnv; print('OK')"`.
- [ ] **Copy the manifest checksums somewhere durable off the disk** — the
      whole reason this runbook exists is that last time the only copies died
      with the disk and the GCS upload silently failed 6×. Verify any upload
      actually landed (`gsutil ls -l`), don't trust the exit code.

## Wiring it into a run (training is a separate step)

Staging alone does not train. A WebShop training run additionally needs:

- **Env vars** (the script prints these at the end):
  `WEBSHOP_DATA_DIR=<pkg>`, `WEBSHOP_PACKAGE_DIR=<pkg>`,
  `OPENRESEARCH_WEBSHOP_PYTHON=/mnt/sdar-cache/conda/envs/verl-webshop/bin/python3`.
- **The per-cell interpreter seam** — with `OPENRESEARCH_WEBSHOP_PYTHON` set,
  `gpu_cell_runner._cell_interpreter` runs the WebShop cell (its `cells.json`
  `env` axis or id contains `webshop`) under the dedicated interpreter and
  prepends *its* `bin/` to the cell `PATH`, so the faithful Lucene/pyserini
  search is used instead of `webshop_env.py`'s `rank_bm25` fallback
  (`backend/services/runtime/CLAUDE.md` bullet; unset = byte-identical
  default).
- **The OOM-knobbed launcher** — WebShop training uses the memory-safe
  `run_webshop_3b.sh` knobs (micro-batch, `gpu_memory_utilization=0.5`,
  optimizer offload, `expandable_segments` stripped), **already applied
  on-disk** per `docs/runbooks/2026-08-03-sdar-verl-oom-knobs-vm-application.md`
  — the VM cache-disk copies are the knobbed ones; re-apply from that runbook
  if the repo is re-cloned fresh (a fresh clone from stage 2 of this script is
  the authors' pristine `9f2ce6a`, NOT knobbed).
- **The full training stack** in `verl-webshop` (stage 6 with
  `WEBSHOP_STAGE_TRAIN_STACK=1` on the GPU host, or
  `sdar_authors_repro.sh base`+`webshop` phases — vllm **0.8.2**, not 0.11.0:
  WebShop must run in its own env).
