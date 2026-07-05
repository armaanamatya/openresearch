# SDAR execute-mode seams + full-grid reproduction — HANDOFF

> **Purpose:** a single, self-contained entry point to *execute* the SDAR-on-GCP reproduction in a fresh session. Everything needed — context, coordinates, the implementation plan, the exact run commands, verification gates, and gotchas — is here or in the two companion docs. Written 2026-07-04; execution deferred to a new session by operator request.
>
> **Read these three, in order, before doing anything:**
> 1. **This handoff** (the executable sequence).
> 2. [`docs/audits/2026-07-04-sdar-gcp-runs-log-analysis.md`](../audits/2026-07-04-sdar-gcp-runs-log-analysis.md) — the evidence (all run outcomes, error signatures, the execute-mode gap analysis). **Source of truth for the diagnosis.**
> 3. [`docs/superpowers/specs/2026-07-04-execute-mode-verl-seams-and-sdar-repro-design.md`](../superpowers/specs/2026-07-04-execute-mode-verl-seams-and-sdar-repro-design.md) — the design (the seam interfaces, tests, risks).
>
> **Working discipline (per operator standing prefs):** Opus authors/owns the plan + reviews **every diff**; Sonnet executes against the spec via the `/implement` skill; TDD (write the per-seam guard test first). Commit infrequently at milestones; **push only to the `deepinvent` remote**; no Co-Authored-By trailer; git identity `lolout1`.

---

## 0. TL;DR — the whole sequence

The harness's repo-first `execute` mode is *just a prompt-note* with no execution machinery, so it can't actually drive the authors' `verl` SDAR trainer — the shim falls on the (stubbing) Foundry executor. **Fix = build 3 reusable harness seams so a cell can run `conda run -n sdar bash <proof>.sh` on the staged cache and have its verl metrics parsed automatically; then run the grid.**

```
STEP 1  Implement 6 harness changes (§4) + per-seam tests. Opus reviews. ruff + off-state suites green.
STEP 2  Fix the authors' verl configs (§5): ALFWorld/WebShop OOM-safe knobs; parameterize proof scripts for 1.7B+3B.
STEP 3  Seed the operator cells.json manifest (§6) + the run-spec (§7) — do NOT rely on the Foundry executor to author them.
STEP 4  Restart the VM. Phase 1: run ONLY the Search-3B cell (§8) → GATE: harness-driven val/success_rate ≈ 0.456 (≥0.40) + guards pass.
STEP 5  Phase 2: run the full 6-cell grid (§8), autostop ON, --max-run-gpu-usd 400. Pull final_report from GCS.
```

**Locked decisions (do not re-litigate):** execute mode (authors' repo verbatim) · **all LLM tiers on the existing Foundry deployment** · full **3-env × smallest-two-model** grid · **validate Search-3B first** · **~$400 GPU cap** · **autostop ON**.

---

## 1. Context (why we're here)

Reproducing **SDAR** (Self-Distilled Agentic RL, arXiv **2605.15155**): 3 Qwen sizes × 3 envs (ALFWorld/WebShop/Search-QA), GRPO + sigmoid-gated OPSD. Two reproduction tracks exist and **only one produces real numbers**:
- **Harness from-scratch (`adapt`)** — every recent run `failed`/`partial` with fabricated/zero/stub metrics (guards fired correctly) + dead WebShop/Search-QA envs. Never a valid grid.
- **Authors' `verl` trainer verbatim (`execute`)** — the only real signal: **Search-QA-3B = `val/success_rate` 0.456 @ 150 steps** (verified, hand-run bash). ALFWorld-3B was OOM-blocked (now understood + fixable); WebShop never trained (staged only).

We are making the **harness** drive the authors' verl end-to-end (so root+generation run through Foundry and a scored `final_report` is produced), reproducing 0.456 *through the harness*, then extending to the full grid.

---

## 2. Coordinates (everything you need to touch)

| Thing | Value |
|---|---|
| Dev box repo | `/home/abheekp/openresearch`, branch `reconcile/grounded-self-improvement-on-main` |
| GCP project / account | `deepinvent-ext-ut` / `abheek@deepinvent.ai` (`export CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud`) |
| GPU VM | **`sdar-2model-a`**, zone **`us-central1-a`**, `a2-ultragpu-4g`, **4×A100-80GB**, **currently STOPPED** (restart to use) |
| Cache disk (persists) | pd-ssd `sdar-cache-a` → mounted **`/mnt/sdar-cache`** on the VM |
| GCS runs bucket | `gs://deepinvent-ext-ut-sdar-runs/` (per-run reports; `sdar_gcp_run.sh::self_stop` uploads) |
| Authors' repo (staged) | `/mnt/sdar-cache/SDAR` @ commit **`9f2ce6a82a90cc5a334d73f188c16df2c4107d80`** (`github.com/ZJU-REAL/SDAR`) |
| Conda envs (staged) | `/mnt/sdar-cache/conda/envs/`: **`sdar`** (py3.12, ALFWorld+Search), `retriever`, **`verl-webshop`** |
| Weights | `HF_HOME=/mnt/sdar-cache/hf` — **Qwen2.5-3B-Instruct, Qwen3-1.7B, Qwen2.5-7B** all present |
| Proof scripts | `/mnt/sdar-cache/SDAR/examples/sdar_trainer/run_{search_3b,alfworld_3b_4gpu,webshop_3b_patched}.sh` |
| Env logs (verl) | `/mnt/sdar-cache/logs/run_{search,alfworld,webshop}_3b*.log` |
| Foundry (all tiers) | `AZURE_FOUNDRY_ENDPOINT=…appradhann-4738-resource.services.ai.azure.com/openai/v1/`, deployment `grok-4.3` (script forces `gpt-chat-latest`), key present in `.env`. Model-agnostic — served model = `AZURE_FOUNDRY_DEPLOYMENT`. |
| SDAR-GCP drivers | `scripts/sdar_gcp_e2e.sh` (start/sync/launch/monitor/autostop), `scripts/gcp_sdar_preflight.sh` (GREEN-gated launch → `scripts/sdar_gcp_run.sh --run-spec …` on the VM), `scripts/sdar_cpu_stage.sh` (CPU disk warm-up — NOT needed, cache already warm) |

**Restart the VM:** `CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud gcloud compute instances start sdar-2model-a --zone us-central1-a --project deepinvent-ext-ut`
**Emergency stop (halt cost):** `… gcloud compute instances stop sdar-2model-a --zone us-central1-a --project deepinvent-ext-ut --discard-local-ssd=false`

---

## 3. The core finding (what makes this non-trivial)

Repo-first `execute` mode = **prompt-note + verbatim code-copy, ZERO footprint in the execution machinery** (`grep execute|reproduction_mode|repo_spec` finds nothing in `run_experiment`/`gpu_cell_runner.py`/backends). The executor model must hand-author the entire shim: a `train_cell.py` shelling to `conda run -n sdar …`, dep wiring that survives the harness's forced cu121-torch pip bootstrap, and a verl→`metrics.json` parser. **The Foundry executor stubs on exactly this.** So "all-Foundry execute" only works if the harness *owns* the hard parts. Full gap list: audit §12.

---

## 4. STEP 1 — implementation plan (6 harness changes)

The elegant framing: **generalize the cell-job contract** — a cell becomes `{command?, gpus, env?, metrics_source?}` with today's `python train_cell.py` as the byte-identical default. Every change is **default-OFF / byte-identical** when its field/flag is absent, with a hermetic ON+OFF test. Detailed interfaces in the **spec §3**; summary + file targets here:

| # | Change | File targets | Test |
|---|---|---|---|
| 1 | **Cell command/launcher seam** — `cell["command"]` runs a raw command (e.g. `conda run -n sdar bash …`) instead of `python train_cell.py`, with the same injected `CUDA_VISIBLE_DEVICES`/`OUTPUT_DIR`. Fixes the `webshop_env.py:21` TODO. | `backend/agents/rlm/gpu_cell_runner.py` (`_run_cell_subprocess`/`run_matrix` ~:676), `cell_matrix.py` (`normalize_cell_axes` passthrough), `cell_scheduler.py` | `tests/rlm/test_cell_command_seam.py` |
| 2 | **Staged-env passthrough** — `OPENRESEARCH_CELL_ENV_PASSTHROUGH="HF_HOME,…"` forwards orchestrator env into cell `child_env` + monolithic `SandboxConfig.environment`, **with precedence over** the contract-prompt default + `asset_provisioning` HF_HOME clobber. | `gpu_cell_runner.py` (~:631-657), `primitives.py` (~:3829-3839), `asset_provisioning.py:340`, `backend/agents/prompts/_sandbox_contract.py:58` | `tests/rlm/test_cell_env_passthrough.py` |
| 3 | **verl→metrics adapter** — new `backend/agents/rlm/verl_metrics_adapter.py` (stdlib-only, auto-copied into `code/` via `_HARNESS_CODE_HELPERS`); runner post-step synthesizes the flat per-cell `metrics.json` + `eval_provenance.json` from verl's val output when a cell declares `metrics_source.kind=="verl"`. **Value-preserving** (copies verl's numbers, never scales). **⚠ confirm verl's real val-output path against `/mnt/sdar-cache/logs/run_search_3b.log` + a run dir before finalizing the parser.** | new module; `gpu_cell_runner.py` (post-step + copy list); `baseline_implementation.py:2816` (reference from `execute_repo_note`) | `tests/rlm/test_verl_metrics_adapter.py` |
| 4 | **Local-repo reuse + commit pin** — `OPENRESEARCH_REPO_LOCAL_PATH=/mnt/sdar-cache/SDAR` seeds `code/` from the staged repo (no GitHub clone); `OPENRESEARCH_REPO_COMMIT` pins it. | `config.py:94-98` (+2 fields), `backend/services/ingestion/repo/{resolver.py,provisioner.py}`, `run.py::_resolve_and_clone_repo` | `tests/services/ingestion/repo/test_local_repo_reuse.py` |
| 5 | **Execute owns deps** — when `mode=="execute"`, skip the cu121-torch pip bootstrap (`primitives.py:3978-4007`); the conda env owns verl/vLLM. Flag `OPENRESEARCH_EXECUTE_OWNS_DEPS` (default ON in execute). | `primitives.py:3978-4007`, `env_pin.py` | `tests/rlm/test_execute_owns_deps.py` |
| 6 | **Fail-loud on execute clone failure** — execute + failed clone must NOT silently downgrade to `mode="scratch"`; warn + fall back to the local path (§4.4) or hard-fail. | `run.py:628-637` | `tests/rlm/test_execute_clone_failure.py` |

**Parallelism for a fresh session:** 1, 4, 5, 6 are largely independent; 3 depends on 1 (the post-step lives in the same runner path); 2 touches several files but is self-contained. Reasonable fan-out: {1+3}, {2}, {4+6}, {5}. Land each with its test; Opus reviews each diff.

**Definition of done for STEP 1:** all 6 tests pass ON and assert OFF byte-identical; the full off-state suites (cells-route / execute-mode / role-model) stay green; `ruff` clean under `uv sync --frozen` (Python 3.12).

---

## 5. STEP 2 — authors' verl config fixes

Grounded in the config that already succeeded on Search (audit §11). Edit on the staged disk `/mnt/sdar-cache/SDAR/examples/sdar_trainer/` (host-side; pinned by the commit in §7).

- **ALFWorld (`run_alfworld_3b_4gpu.sh`):** `ppo_micro_batch_size_per_gpu 32→16` · `rollout.gpu_memory_utilization 0.6→0.5` · add `actor.fsdp_config.optimizer_offload=True` · **REMOVE `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (vLLM `CuMemAllocator` asserts incompatible — a proven dead end). Rationale: ALFWorld already trains + learns (0.383→0.508 @ step 79) then OOMs on fragmentation in the actor update; these knobs match the Search config that survives.
- **WebShop (`run_webshop_3b_patched.sh`):** strip the same `expandable_segments`; apply the same memory-safe knobs; **the training launch has never been fired** — this run is its first.
- **Search-QA (`run_search_3b.sh`):** unchanged (proven 0.456).
- **⚠ Parameterize all three proof scripts to accept the model** (e.g. read `$SDAR_MODEL`, default the current 3B for back-compat) — the smallest-two grid needs **Qwen3-1.7B** *and* **Qwen2.5-3B-Instruct**, but the scripts currently hardcode 3B. Confirm 1.7B fits (smaller ⇒ more OOM headroom). Note the mixed lineage: the paper's smallest-two is **Qwen3-1.7B + Qwen2.5-3B**.

---

## 6. STEP 3a — operator cells.json manifest (robustness net)

**Seed this into `code/cells.json` yourself** (via `OPENRESEARCH_BASELINE_EXTRA_GUIDANCE`, a pre-seeded file, or a paper-hint) rather than trusting the Foundry executor to author it — this is the belt-and-suspenders that makes the run robust regardless of executor stubbing. With the §4 seams, this manifest is all that's needed; the executor's job collapses to (at most) verifying it. Template (adjust script names to the parameterized proof scripts from §5):

```jsonc
{ "cells": [
  {"id":"search_qa_3b","model_key":"Qwen2.5-3B-Instruct","env":"search_qa","baseline":"sdar","gpus":4,
   "command":"conda run -n sdar bash examples/sdar_trainer/run_search_3b.sh",
   "env":{"SDAR_MODEL":"Qwen2.5-3B-Instruct"},
   "metrics_source":{"kind":"verl","log_glob":"$OUTPUT_DIR/*.log","success_rate_key":"val/success_rate"}},
  {"id":"search_qa_1p7b","model_key":"Qwen3-1.7B","env":"search_qa","baseline":"sdar","gpus":4,
   "command":"conda run -n sdar bash examples/sdar_trainer/run_search_3b.sh","env":{"SDAR_MODEL":"Qwen3-1.7B"},
   "metrics_source":{"kind":"verl","log_glob":"$OUTPUT_DIR/*.log","success_rate_key":"val/success_rate"}},
  {"id":"alfworld_3b","model_key":"Qwen2.5-3B-Instruct","env":"alfworld","baseline":"sdar","gpus":4,
   "command":"conda run -n sdar bash examples/sdar_trainer/run_alfworld_3b_4gpu.sh","env":{"SDAR_MODEL":"Qwen2.5-3B-Instruct"},
   "metrics_source":{"kind":"verl","log_glob":"$OUTPUT_DIR/*.log","success_rate_key":"episode/success_rate"}},
  {"id":"alfworld_1p7b","model_key":"Qwen3-1.7B","env":"alfworld","baseline":"sdar","gpus":4,
   "command":"conda run -n sdar bash examples/sdar_trainer/run_alfworld_3b_4gpu.sh","env":{"SDAR_MODEL":"Qwen3-1.7B"},
   "metrics_source":{"kind":"verl","log_glob":"$OUTPUT_DIR/*.log","success_rate_key":"episode/success_rate"}},
  {"id":"webshop_3b","model_key":"Qwen2.5-3B-Instruct","env":"webshop","baseline":"sdar","gpus":4,
   "command":"conda run -n verl-webshop bash examples/sdar_trainer/run_webshop_3b_patched.sh","env":{"SDAR_MODEL":"Qwen2.5-3B-Instruct"},
   "metrics_source":{"kind":"verl","log_glob":"$OUTPUT_DIR/*.log","success_rate_key":"val/success_rate"}},
  {"id":"webshop_1p7b","model_key":"Qwen3-1.7B","env":"webshop","baseline":"sdar","gpus":4,
   "command":"conda run -n verl-webshop bash examples/sdar_trainer/run_webshop_3b_patched.sh","env":{"SDAR_MODEL":"Qwen3-1.7B"},
   "metrics_source":{"kind":"verl","log_glob":"$OUTPUT_DIR/*.log","success_rate_key":"val/success_rate"}}
]}
```
> Confirm the real `success_rate_key` names + the val-output location from a live verl log before trusting the adapter (STEP 1 #3 caveat). `webshop` uses the `verl-webshop` conda env. All cells are `gpus:4` ⇒ they run **sequentially** on the 4×A100 (the scheduler serializes when the free pool < requested).

---

## 7. STEP 3b — run-spec

Create `configs/sdar_execute_run_spec.json` (loaded via `--run-spec`; CLI flags still win). The VM env must **export `HF_HOME=/mnt/sdar-cache/hf`** (and any asset dirs) so the passthrough forwards them.

```json
{
  "OPENRESEARCH_USE_AUTHOR_REPO": "1",
  "OPENRESEARCH_REPRODUCTION_MODE": "execute",
  "OPENRESEARCH_REPO_LOCAL_PATH": "/mnt/sdar-cache/SDAR",
  "OPENRESEARCH_REPO_COMMIT": "9f2ce6a82a90cc5a334d73f188c16df2c4107d80",
  "OPENRESEARCH_EXECUTE_OWNS_DEPS": "1",
  "OPENRESEARCH_CELL_ENV_PASSTHROUGH": "HF_HOME,HF_DATASETS_CACHE,ALFWORLD_DATA,WEBSHOP_URL,SEARCH_QA_INDEX_DIR",
  "OPENRESEARCH_ENV_LIVENESS_GATE": "1",
  "OPENRESEARCH_EVAL_PROVENANCE_GUARD": "1",
  "OPENRESEARCH_ZERO_METRICS_GUARD": "1",
  "OPENRESEARCH_EXTERNAL_VALIDATOR": "1",
  "OPENRESEARCH_MAX_RUN_GPU_USD": "400"
}
```

---

## 8. STEP 4–5 — execution

Runs happen **on the VM** with `--sandbox local` (the VM's own 4×A100). Two drive options:

**Option A (recommended) — via the SDAR-GCP driver** (handles start/sync/launch/monitor/GCS-upload/autostop). Point its run-spec at §7's file + seed §6's cells.json; keep `ROOT=foundry` and the existing `--models executor=foundry,grader=foundry,verifier=foundry`. Then:
```bash
OPENRESEARCH_GCP_INSTANCE=sdar-2model-a OPENRESEARCH_GCP_ZONE=us-central1-a \
PROJECT_ID=sdar_execute_search_$(date +%s) ROOT=foundry PROV=on-demand SMOKE=0 \
  scripts/sdar_gcp_e2e.sh run
```
> The `sdar_gcp_*.sh` scripts need light wiring to pass the execute-mode run-spec + seed cells.json; verify before the grid. Keep **autostop ON** (do NOT set `OPENRESEARCH_SDAR_NO_AUTOSTOP=1`).

**Option B (manual) — SSH the VM and run reproduce directly:**
```bash
# on sdar-2model-a, in /home/abheekp/openresearch, with HF_HOME exported:
python -m backend.cli reproduce 2605.15155 \
  --sandbox local --model foundry \
  --models executor=foundry,verifier=foundry,grader=foundry \
  --run-spec configs/sdar_execute_run_spec.json \
  --scope-spec '{"models":["Qwen3-1.7B","Qwen2.5-3B-Instruct"]}' \
  --paper-hint 2605.15155 \
  --max-run-gpu-usd 400 --max-wall-clock 72000 \
  --project-id sdar_execute_grid_$(date +%s)
```
Then upload artifacts to GCS + stop the VM manually (or via `sdar_gcp_run.sh::self_stop`).

**Phase 1 (validation gate).** Run the grid with **only the `search_qa_3b` cell** in cells.json first (or a `--scope-spec` pinning Search-3B). **GATE:** the harness-driven `val/success_rate` ≈ **0.456** (accept ≥ **0.40**) AND the evidence guards (`env_liveness`, `eval_provenance`, `zero_metrics`) pass + the external validator does not veto. A miss ⇒ the seams/adapter are wrong; debug before spending grid budget (~$30 slice of the $400).

**Phase 2 (full grid).** Restore all 6 cells (§6); run with `--max-run-gpu-usd 400`, autostop ON. Cells serialize (4-GPU each). Pull the report:
```bash
gcloud storage cp -r gs://deepinvent-ext-ut-sdar-runs/<PROJECT_ID>/ /tmp/sdar_out/
```

---

## 9. Verification gates (how you know each step passed)

- **STEP 1:** 6 new tests pass ON + OFF-parity; off-state suites green; `ruff` clean. Opus has reviewed every diff.
- **STEP 2:** ALFWorld cell survives past step 79 (no OOM); no `expandable_segments` anywhere; 1.7B + 3B both launch.
- **Phase 1:** harness `final_report` for Search-3B shows `val/success_rate ≥ 0.40`, `verdict` not fabrication-vetoed, evidence guards clean, `metrics.json` has a real (non-zero, value-preserved) number with an `eval_provenance.json` sidecar.
- **Phase 2:** `final_report.json` scored across the 6 cells; no `all_models_failed`/`zero_metrics`/`env_setup_failed` on cells that actually trained; GPU spend < $400; VM auto-stopped at the end.

---

## 10. Gotchas / risks / open decisions

1. **verl val-output path (STEP 1 #3):** the adapter's parse source must be confirmed against a real run dir/log first; log-regex is the robust fallback; it must fail *honest* (never fabricate a number).
2. **Foundry executor may stub even the manifest** — mitigated by pre-seeding cells.json (§6) so the executor authors ~nothing. If it still interferes, pin a validated executor (`--models executor=sonnet`) *just for the shim* while keeping root=foundry.
3. **`conda run` env leakage** (CUDA/LD_LIBRARY_PATH) — validate on Phase 1; the seam injects only the allowlisted staged env.
4. **Validator separation is `degraded`** when executor + validator share one Foundry deployment — the deterministic veto still stands, but for a real cross-check fund a second family (`--models validator=gpt-4o-azure`). **Open decision (default: accept degraded + note it).**
5. **verl config-edit location** — patch on the staged disk (simplest, pinned by commit) vs. an execute-mode `code/` overlay (cleaner provenance). **Open decision (default: staged disk).**
6. **$400 cap may trip mid-grid** — a partial grid still scores honestly; resume with a raised cap if wanted.
7. **Smallest-two mixed lineage** — Qwen3-1.7B + Qwen2.5-3B (not same family); both staged in HF_HOME.
8. **OpenAI key** in `.env` is present but assume **unfunded** (memory says dead) — do not route a tier to gpt-5 without verifying; the plan is fully Foundry/OAuth-free.
9. **Cost hygiene:** an orphaned `sdar-cache` disk (us-central1-c, 1000GB, unattached, ~$170/mo) + 9 pd-ssd disks (~$45/day) remain — cleanup is optional/independent (audit §1). Always confirm the VM auto-stopped after a run.

---

## 11. Pointers

- **Analysis / evidence:** `docs/audits/2026-07-04-sdar-gcp-runs-log-analysis.md` (Part II = confirmed diagnostics + gap list).
- **Design / spec:** `docs/superpowers/specs/2026-07-04-execute-mode-verl-seams-and-sdar-repro-design.md`.
- **Memory:** `project_sdar_execute_mode_reproduction` (+ linked `sdar_gcp_run_hardening`, `project_sdar_gcp_harness_refactor`, `project_foundry_gptchat_root_not_executor`).
- **Key code the seams touch:** `backend/agents/rlm/{gpu_cell_runner,cell_matrix,cell_scheduler,primitives,baseline_implementation,env_pin}.py`, `backend/services/ingestion/repo/{resolver,provisioner}.py`, `backend/agents/prompts/_sandbox_contract.py`, `config.py`, `run.py`.
- **Existing SDAR-GCP handoffs (background):** `docs/runbooks/2026-06-22-sdar-gcp-e2e-and-rl-smoke-fix-handoff.md`, `docs/runbooks/2026-06-20-validation-coverage-and-capped-rerun-handoff.md`.
- **Related uncommitted feature seen this session:** the external-runs monitor (`backend/services/external_monitor/`, `/external-runs` UI) — SSH-polls remote runs into `runs/_external/<id>/events.jsonl`; unrelated to this repro but note it's untracked if you branch/rebase.

---

## 12. Paste-ready kickoff prompt for the new session

> Read `docs/runbooks/2026-07-04-sdar-execute-mode-seams-and-repro-handoff.md` in full, then its two companion docs (the audit + the spec). We're building 3 harness "cell-job" seams so the harness can drive the authors' SDAR verl trainer verbatim on the staged GCP cache (all LLM tiers on Foundry), then reproducing SDAR — validate Search-3B (≈0.456) first, then the full 3-env × {Qwen3-1.7B, Qwen2.5-3B} grid at a $400 GPU cap with autostop ON. Use `/implement` with TDD (per-seam guard test first); I (Opus) review every diff before it lands; Sonnet executes. Start with STEP 1 (the 6 harness changes in §4) — propose the fan-out, then implement change #1 (cell command seam) test-first. Commit only at milestones; push to `deepinvent` only; git identity `lolout1`; no Co-Authored-By trailer. GCP: project `deepinvent-ext-ut`, VM `sdar-2model-a`/`us-central1-a` (STOPPED), `CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud`.
