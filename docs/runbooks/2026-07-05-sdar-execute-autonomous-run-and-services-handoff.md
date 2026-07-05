# SDAR execute-mode autonomous run + cell-services — SESSION HANDOFF (2026-07-05, session 2)

> **Purpose:** seamless continuation. This session finished ALL the execute-mode seams, added a
> Minimal Viable Reproduction mode, built a declarative **cell-services + dynamic GPU-partition**
> layer, fixed three live adapter/route bugs, and **launched the SDAR Phase-1 execute run
> autonomously on GCP** (retrieval server + 3-GPU training, autostop ON). Read this first, then the
> two companions: [`2026-07-05-sdar-execute-mode-session-handoff.md`](2026-07-05-sdar-execute-mode-session-handoff.md)
> (the prior state) and [`2026-07-04-sdar-gcp-runs-log-analysis.md`](../audits/2026-07-04-sdar-gcp-runs-log-analysis.md) (evidence).
> Memory: `project_sdar_execute_mode_reproduction`.

---

## 0. Status at a glance

| Item | Status |
|---|---|
| Wave-1 seams (#1 command, #3 verl adapter, #4 local-repo reuse, #5 execute-owns-deps, #6 fail-loud) | ✅ committed `58f15ad9` |
| Wave-2 #2 (env passthrough + HF_HOME guard + R1 expandable_segments) + #7 (cells pre-seed) | ✅ committed `58f15ad9` |
| R2 (eval-provenance aggregate reconciliation) | ✅ committed `58f15ad9` |
| Minimal Viable Reproduction mode (`--minimal-viable`) | ✅ committed `58f15ad9` |
| verl adapter real-format fixes (B: log location, C: dict-repr regex) | ✅ committed `5ae37441` |
| **Cell services + dynamic GPU partition** (`services`, `gpus:"auto"`, `OPENRESEARCH_CELL_TRAIN_GPUS`) | ✅ committed `94d9a716` |
| Phase-1 dynamic cell config (services + auto gpus) | ✅ committed `95ed2b26` |
| **Cells-route gate for command-cells** (no train_cell.py needed) | ✅ committed `04b17bcc` |
| STEP-2 dynamic n_gpus/batch edit on the VM SDAR repo | ✅ committed on VM → `f6d0d318` |
| **Cells-seed AUTHORITY** (re-assert operator grid over executor clobber) | ✅ committed `0de48757` |
| **Phase-1 autonomous run** | 🟢 run 2 `sdar_exec_phase1_1783280123` RUNNING (run 1 `…_1783279253` proved the chain but the executor clobbered the Search seed → fixed) |
| Phase-2 grid (6 cells) | ⏳ gated on Phase-1 pass; grid config + ALFWorld/WebShop STEP-2 pending |

All branch `reconcile/grounded-self-improvement-on-main`, **pushed to deepinvent** (up to `0de48757`).

**Run 1 outcome (the fix that mattered):** the whole seam chain engaged live — repo seeded into `code/`, the **cells route engaged** (the command-cell gate fix), and the **GPU partition ran** ("3 cells across 4 GPUs"). But the foundry **executor overwrote the pre-seeded Search `cells.json` with its own ALFWorld one** (the #7 first-seed only wrote-if-absent), those ALFWorld cells `cell_execution_error`'d, the root churned 21 iters → `failed`/0.0. Fix `0de48757`: `run_experiment` re-asserts `OPENRESEARCH_CELLS_SEED_PATH` over `code/cells.json` (`force=True`, idempotent), so the operator's Search grid is what runs. **Scope note:** the effective scope is all 3 envs (`--scope-spec` only pinned the model); the reassert redirects EXECUTION to the single Search cell regardless — the other envs become honest rubric gaps. For a cleaner Phase-1 gate, also pass `--scope-spec '{"models":["Qwen2.5-3B-Instruct"],"datasets":["Search-QA"]}'`.
Every code change is default-OFF / byte-identical when its flag/field is absent.

---

## 1. The live run — how to check it

- **VM:** `sdar-2model-a`, zone `us-central1-a`, project `deepinvent-ext-ut`, 4×A100-80GB. `export CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud`.
- **project_id:** `sdar_exec_phase1_1783279253` (also in `runs/.phase1_project_id` on the VM).
- **Autostop ON:** the wrapper (`runs/phase1_autonomous.sh`) runs `reproduce` → uploads `runs/<pid>/` to `gs://deepinvent-ext-ut-sdar-runs/<pid>/` → `sudo shutdown`. So the VM **self-stops** on completion/crash; the report lands in GCS regardless.
- **Check progress (SSH):**
  ```bash
  CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud gcloud compute ssh abheekp@sdar-2model-a --zone us-central1-a --project deepinvent-ext-ut --quiet --command '
    cd /home/abheekp/openresearch; PID=$(cat runs/.phase1_project_id)
    tail -20 runs/phase1_run.out
    tail -3 runs/$PID/code/outputs/*/cells/*/cell_stdout.log 2>/dev/null   # verl val/success_rate
    find runs/$PID -name service_retrieval.log -exec tail -5 {} \;
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    cat runs/$PID/final_report.json 2>/dev/null | python3 -m json.tool | head -40'
  ```
- **If the VM already stopped** (run finished): pull the result from GCS: `gsutil -m cp -r gs://deepinvent-ext-ut-sdar-runs/sdar_exec_phase1_1783279253 .` and read `final_report.json` + `phase1_run.out`.
- **Phase-1 GATE (pass → proceed to Phase 2):** harness-driven `val/success_rate` ≥ 0.40 (target 0.456) AND the deterministic guards clean (zero-metrics / eval-provenance / env-liveness / no-learning) AND `code/metrics.json` shows a real measured value with an `eval_provenance.json` (`provenance_kind:"aggregate"`) sidecar.

---

## 2. What the run exercises (the whole seam chain, proven live this session)

`reproduce` (foundry root + `OPENRESEARCH_LIFECYCLE_DRIVE=1`) →
`implement_baseline`: execute mode **seeds the authors' SDAR repo** into `code/` (`OPENRESEARCH_REPO_LOCAL_PATH=/mnt/sdar-cache/SDAR`, pinned `OPENRESEARCH_REPO_COMMIT=f6d0d318`) **+ pre-seeds `code/cells.json`** (`OPENRESEARCH_CELLS_SEED_PATH=configs/sdar_execute_cells_phase1.json`, #7) → `run_experiment`: the **cells-route gate now engages for command-cells** (04b17bcc — cells.json + every cell has a `command`, no train_cell.py) → `gpu_cell_runner.run_matrix`:
- `gpus:"auto"` leases all 4 A100s; the `retrieval` **service reserves 1 GPU** (its 64 GB FAISS index), training gets the other 3 (`OPENRESEARCH_CELL_TRAIN_GPUS=3`);
- the service is started (`conda run -p …/retriever bash examples/search/retriever/retrieval_launch.sh`), **readiness-gated** on `http://127.0.0.1:8000/retrieve` (up to 1800 s to load the index), then **torn down** after training;
- training runs the authors' launcher verbatim (`conda run -p …/sdar bash examples/sdar_trainer/run_search_3b.sh`) with `n_gpus_per_node=$N_GPUS` + batches scaled to the GPU count (STEP-2);
- the authors' verl logs `'val/success_rate': np.float64(0.456)` to the console → the harness captures it → the **verl metrics adapter** (real dict-repr regex fix) writes `code/metrics.json` + an aggregate `eval_provenance.json` → the leaf scorer + guards read it.

Everything is **dynamic/hardware-aware**: on a box with N GPUs, retrieval takes 1, training takes N-1, and the batches scale — nothing is hard-coded.

---

## 3. Key coordinates & decisions (do NOT re-derive)

- **conda on the VM:** binary at `/mnt/sdar-cache/miniconda/bin/conda`; envs at `/mnt/sdar-cache/conda/envs/{sdar,retriever,verl-webshop}` — **NOT on PATH** (`.bashrc` has no init), so cells use the **full-path `conda run -p <prefix>`**. `sdar` (py3.12) has verl **vendored in the repo** (importable only with `cwd=code/`, which the command seam sets). `retriever` (py3.10) has faiss 1.8.0.
- **Search assets** (staged, survive VM stop): `$HOME/data/searchR1/{e5_Flat.index(64GB), wiki-18.jsonl(14GB), wiki-18.jsonl.offsets.npy}`; HF weights `/mnt/sdar-cache/hf` (Qwen2.5-3B/7B, Qwen3-1.7B); parquets `$HOME/data/searchR1_processed_direct/{train,test}.parquet`. 668 GB RAM.
- **Retrieval GPU split:** the 64 GB index fits one A100-80GB, so **retrieval on 1 GPU, training on 3** (operator decision). The verl batch-divisibility (`mini % (micro*n_gpus)==0`) forced the dynamic batch scaling: `train_data_size=N*32`, `ppo_mini=N*64`, `n_gpus_per_node=N` (per-GPU work constant; global batch scales with N). STEP-2 committed this to the VM SDAR repo → `f6d0d318`.
- **wandb:** the authors' script hardcodes a placeholder `WANDB_API_KEY` + `trainer.logger=['console','wandb']`; `cell_env` sets `WANDB_MODE=offline` so it never hangs. The scored metric is on the **console** (adapter reads it), not wandb.
- **Three live adapter/route bugs found by pre-GPU log inspection (all fixed, `5ae37441`+`04b17bcc`):** (A) the aggregate `val/success_rate` exists distinct from 7 per-dataset keys — the `success_rate_key` is correct; (B) the runner wrote the cell's stdout to `output_root/<id>.log`, a *sibling* of the dir the adapter globbed → now symlinked into `output_dir` as `cell_stdout.log`; (C) verl logs `'val/success_rate': np.float64(0.456)` (dict-repr) which the old `[:=\s]+(number)` regex couldn't parse → broadened to tolerate the quote/`np.float64(` wrapper; (route) the cells route required `train_cell.py` which command-cells lack → engages on `cells.json` + all-command.
- **LLM tiers:** all on Foundry (`AZURE_FOUNDRY_*` set in the VM `.env`); `OPENAI/ANTHROPIC` keys empty/dead (unset via `env -u`). Root=azure-foundry (grok) is NOT paper-validated (advisory warning is expected). `OPENRESEARCH_LIFECYCLE_DRIVE=1` drives plan→implement→run→verify if the root churns.

---

## 4. Phase 2 (the full grid) — what remains before it

Gated on the Phase-1 gate passing. Then:
1. **Grid config** `configs/sdar_execute_cells.json` needs the same treatment as Phase-1 for its 2 Search cells (retrieval service + full-path conda + WANDB offline + `gpus:"auto"`), and for the ALFWorld/WebShop cells: full-path conda + WANDB offline + `gpus:"auto"`; ALFWorld runs its env **in-process** (no GPU service); WebShop needs its **web_agent_site server** (a service, `gpus:0` — but the staged `verl-webshop` env showed `web_agent_site`/`faiss` import errors on 2026-07-05, so WebShop needs env repair first).
2. **STEP-2 for ALFWorld/WebShop** on `/mnt/sdar-cache/SDAR` (read-then-edit, then commit → new SHA → update `OPENRESEARCH_REPO_COMMIT`): `run_alfworld_3b_4gpu.sh` + `run_webshop_3b_patched.sh` — dynamic `N_GPUS`/batches (as Search), port Search's memory knobs (`ppo_micro_batch_size_per_gpu 32→16`, `gpu_memory_utilization 0.6→0.5`, `optimizer_offload=True`), and **strip `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (vLLM CuMemAllocator asserts; the R1 fix already stops the harness re-injecting it on command cells, but the authors' scripts set it themselves).
3. Parameterize the 3 scripts to read `$SDAR_MODEL` for the 1.7B cells (the cells pass `cell_env.SDAR_MODEL`; Search's `run_search_3b.sh` still hardcodes `Qwen/Qwen2.5-3B-Instruct` in `actor_rollout_ref.model.path` — make it `${SDAR_MODEL:-Qwen/Qwen2.5-3B-Instruct}`).
4. Launch: point `OPENRESEARCH_CELLS_SEED_PATH=configs/sdar_execute_cells.json`, `--scope-spec '{"models":["Qwen3-1.7B","Qwen2.5-3B-Instruct"]}'`, `OPENRESEARCH_MAX_RUN_GPU_USD=400`, autostop ON. Cells serialize (each leases all 4 GPUs).

---

## 5. Working discipline (unchanged)

Opus authors/reviews EVERY diff; Sonnet executes against a tight spec (dispatch **synchronously** for critical work — background subagents were lost on a mid-turn process exit, §3.E of the prior handoff). TDD, default-OFF/byte-identical, hermetic ON+OFF tests, ruff clean. Commit at milestones (few substantial commits); descriptive present-tense headlines (what+symptom+resolution); **push deepinvent only**; identity `lolout1`; **no Co-Authored-By**. Money: autostop ON always; never leave a 4×A100 idle. `/implement` for implementation.

---

## 6. Reconstitution note (if the VM is stopped/rebuilt)

The staged pd-ssd disk `sdar-cache-a` (→ `/mnt/sdar-cache`, `--discard-local-ssd=false` preserves it) holds conda + envs + repo + index + weights. The VM's `/home/abheekp/openresearch` is a **non-git rsync copy** — sync new code by `tar czf` the changed `backend/…` + `configs/…` files, `gcloud compute scp` the tarball, `tar xzf` on the VM (a full-repo tar is ~217 MB and times out scp — sync only changed files). STEP-2 edits live in the VM's `/mnt/sdar-cache/SDAR` git repo (committed, so the pin `f6d0d318` seeds them). git identity on that repo: `git config user.email abheek@deepinvent.ai; user.name lolout1`.
