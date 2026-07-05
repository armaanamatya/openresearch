# SDAR-on-GCP Runs — Consolidated Log & Failure Analysis

> **Purpose:** a single, precise, evidence-grounded snapshot of the current SDAR
> reproduction state on GCP — live infra + cost, every recent run's outcome, the
> recurring error signatures with root-cause hypotheses, and the model-config
> reality. Written to be consumed by a deeper Opus/Fable pass that will devise the
> optimal robust fix. **Precision on error signatures > breadth.**
>
> **Compiled:** 2026-07-04 (data pulled ~2026-07-05 03:30 UTC).
> **Branch:** `reconcile/grounded-self-improvement-on-main` @ `2e02cc2f` (90 ahead of `origin/main`, pushed to `deepinvent` only).
> **Method:** 4 parallel read-only agents over live `gcloud`/`gsutil`, the running VM (SSH), GCS `gs://deepinvent-ext-ut-sdar-runs/`, and local `runs/`.
> **Status:** analysis-only. No fix applied, no run launched, no VM touched.

---

## 0. TL;DR (read this first)

1. ✅ **RESOLVED — `sdar-2model-a` was stopped 2026-07-05 (`--discard-local-ssd=false`, disks + local SSD preserved).** [Was: idle ~19h, 4×A100-80GB on-demand, ~$20/hr, ~$375 sunk.] Cost halted; restart to relaunch.
2. **There are two reproduction tracks, and only one produces real numbers:**
   - **Track A — harness writes SDAR from scratch (`adapt` mode).** Every recent 2-model run ended `failed`/`partial` with **fabricated/zero/stub metrics** (guards fired correctly) and **dead WebShop + Search-QA envs**. The harness's own implementation has never produced a valid full-grid result.
   - **Track B — run the authors' `verl` trainer verbatim (repo-first `execute` mode).** The **only** genuine signal in the entire corpus: **Search-QA-3B = `val/success_rate` 0.456 @ 150 steps (verified, rc=0, real held-out eval)**. ALFWorld-3B is hard-blocked by an OOM↔vLLM catch-22; WebShop never trained (asset staging only).
3. **"Foundry sonnet" does not exist yet.** Foundry is model-agnostic (config-only), but the live deployment is `grok-4.3` and the launched SDAR script hard-forces `gpt-chat-latest`. There is **no Sonnet deployment anywhere**, and the harness would classify a Sonnet-on-Foundry as `foundry` (unvalidated) regardless. gpt-chat/grok **stub as the executor tier** for SDAR — the exact tier the guidance says to keep on Sonnet/gpt-5.

---

## 1. Live GCP inventory & cost exposure

Account `abheek@deepinvent.ai` · project `deepinvent-ext-ut` · default zone `us-central1-c`.

| Instance | Zone | Type / GPU | Provisioning | Status | Uptime | Cost risk |
|---|---|---|---|---|---|---|
| **`sdar-2model-a`** | us-central1-a | a2-ultragpu-4g / **4×A100-80GB** | on-demand | ✅ **STOPPED 2026-07-05** | — | halted (disks preserved) |
| `sdar-2m` | us-central1-a | a2-ultragpu-4g / 4×A100-80GB | — | TERMINATED | — | disk only |
| `sdar-a100-od` | us-central1-b | a2-highgpu-1g / 1×A100-40GB | spot | TERMINATED | — | disk only |
| `sdar-1model` | us-central1-c | a2-ultragpu-4g / 4×A100-80GB | — | TERMINATED | — | disk only |
| `sdar-a100-8g` | us-central1-c | a2-highgpu-4g / 4×A100-40GB | — | TERMINATED | — | disk only |

**Disks — 9 × pd-ssd = 8,000 GB ≈ ~$1,360/mo (~$45/day), billed even when VMs are off:**
- 🟠 **Orphaned** `sdar-cache` (us-central1-c, 1000 GB, **unattached**) — ~$170/mo for nothing.
- Attached: `sdar-2m`+`sdar-cache`→sdar-2m; `sdar-2model-a`+`sdar-cache-a`→running VM; `sdar-a100-od`(500); `sdar-1model`+`sdar-ultra`→sdar-1model; `sdar-a100-8g`(500).

**Actions (safe; each independent):**
```bash
# 1. Stop the idle burner (disks persist, nothing lost):
gcloud compute instances stop sdar-2model-a --zone us-central1-a --project deepinvent-ext-ut
# 2. Delete the orphaned cache disk (only if its contents are stale/duplicated):
gcloud compute disks delete sdar-cache --zone us-central1-c --project deepinvent-ext-ut
```
> Cost figures are us-central1 list-price order-of-magnitude estimates.

---

## 2. Run outcomes — every recent SDAR run

| Run id | Where | Date (UTC) | Track / config | Verdict | Score | End state |
|---|---|---|---|---|---|---|
| `sdar_2model_v2` | GCS + local | 2026-07-02 | A · root=foundry `gpt-chat-latest`, exec/grader/verifier=Sonnet | **failed** | 0.5* | SIGTERM @ 0 iters (0.5 = best-of-run salvage of a *prior* attempt) |
| `sdar_2model_a` | VM `runs/sdar_2model_a/` | 2026-07-01 | A · root=foundry gpt-chat-latest | — | — | **killed** SIGTERM ~40s after start, 0 experiments |
| `sdar_validate_full` | local | 2026-06-27 | A · root=claude-oauth + Sonnet | **partial** | **0.216** | 1 real exp (weak); WebShop+Search served 0 episodes |
| `sdar_merged_full_2g` | local | 2026-06-26 | A · root=claude-oauth + Sonnet | **failed** | None | "produced no result"; 10× fabrication_suspected |
| `_v3_capture` | local | 2026-06-20 | A · root=gpt-chat-latest + Sonnet | **failed** | None | 15 iters, 4 exps all failed (stub/anti-fab); never reached GPU |
| `prj_09047604e591d969` | local + VM | 2026-05-31 report / campaigns 2026-07-02 | A · root=claude-oauth + Sonnet | **failed** | **0.188** | all-zero metrics; 3 campaigns EXHAUSTED, gpu_usd=0 |
| **Search-3B (verl)** | VM + `runs/_external/sdar_search_3b` | 2026-07-03 | **B · authors' verl trainer** | **success** | **0.456** | step 150, rc=0, real held-out eval ✅ |
| **ALFWorld-3B (verl)** | VM + `runs/_external/sdar_alfworld_3b` | 2026-07-05 | **B · authors' verl trainer** | **failed** | — | OOM @ step ~79 → expandable-segments assertion |
| WebShop index build | VM `webshop_index_build.log` | 2026-07-04 | B · asset staging only | partial | — | index built locally; GCS upload `GcsApiError`; never trained |

\* `sdar_2model_v2`'s 0.5 is a rubric best-of-run **salvage** of an earlier in-dir attempt, not a fresh result. The final launch ran 0 iterations.

---

## 3. Track A (harness from-scratch) — failure taxonomy

Every Track-A run fails in one or more of these ways. **The guards are working correctly**; the underlying problem is that the harness's own generated SDAR implementation never produces real training numbers, and 2 of 3 envs are dead.

1. **Fabricated / zero / degenerate metrics.** `prj_09047604e591d969` exp#0: `success=True` but `metrics={status:"running", final_reward:0.0, final_loss:0.0, final_return:0.0, final_accuracy:0.0}` — all-zero, status stuck at `"running"`. `sdar_2model_v2`: all-zero `baselines_vs_sdar`/`success_rate` → **24× `fabrication_suspected`**. `sdar_merged_full_2g`: 10×. (Would be caught by `OPENRESEARCH_ZERO_METRICS_GUARD`.)
2. **Stub / hardcoded-literal metrics.** `_v3_capture`: **34× `smoke_metrics_unreal`** + `code_review_rejected` (`train.py:48` assigns literal `0.0` to every `success_rate`/`webshop_acc`). gpt-chat-latest root emits stubs; run never reaches GPU.
3. **Dead environments.** WebShop + Search-QA served **0 episodes** in every 2-model run → `env_setup_failed` (4×/2×/2×). Harness Search-QA data loader broke: `load_search_qa_tasks()` **TypeError**. (Note: the authors' *own* Search retrieval worked fine — 21 MB `retrieval_server.log` with real bamboogle scores — so this is a harness-impl bug, not an asset gap.)
4. **Truncated training.** Harness ALFWorld ran **20/150 steps** ("compute budget cap") → combined with all-zero metrics = no learning signal.
5. **Early SIGTERM @ 0 iterations.** `sdar_2model_a` (killed ~40s in) and the final `sdar_2model_v2` launch. `sdar_gcp_run.out` says "Ctrl-C → exit-trap self_stop (rc=143)". `OPENRESEARCH_SDAR_NO_AUTOSTOP=1` was set → **not the normal autostop**. Origin of the SIGTERM is unresolved (see §6).
6. **Infra / provisioning.** 3 campaigns had **no docker binary** (`chown ... 'docker': No such file or directory`) and **gpu_usd=0** (never leased GPU); campaign 3 still burned **$24.23 LLM / 4.5h wall** with zero GPU. A100 capacity stockout: `_sdar_2model.log` "stocked out; sleeping 120s" ×29. WebShop asset GCS upload: `GcsApiError`.

---

## 4. Track B (authors' `verl` trainer verbatim) — the real signal

This is the repo-first **`execute`** path: seed the authors' repo into `code/`, run their pipeline behind a thin harness shim. It is the **only** source of a genuine reproduction number.

**✅ Search-QA-3B = 0.456 @ 150 steps (verified).** `run_search_3b.log` reached step 150 (rc=0, "SEARCH-3B PROOF COMPLETE") with real SDAR mechanics live: `sdar/gate_mean:0.478`, `sdl_lambda:0.100`, `teacher_student_gap`, `actor/kl_loss`, real per-dataset held-out eval (`do_sample=False`):

| dataset | nq | triviaqa | popqa | hotpotqa | 2wiki | musique | bamboogle | **mean** |
|---|---|---|---|---|---|---|---|---|
| val/success_rate | 0.418 | 0.605 | 0.465 | 0.403 | 0.399 | 0.147 | 0.669 | **0.456** |

The trailing `BrokenPipeError: [Errno 32] Broken pipe` is harmless wandb atexit teardown, not a training failure. Captured in `runs/_external/sdar_search_3b/events.jsonl` and `final_bundles/search_3b_150step_20260703.tar.gz`.

**🔴 ALFWorld-3B — OOM ↔ vLLM expandable-segments catch-22 (the #1 blocker for a full 3-env repro):**
- **Attempt 1** (`run_alfworld_3b.attempt1_oom_step79.log`): `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.02 GiB. GPU 0 has 79.25 GiB total of which 5.85 GiB is free ... 82.24 GiB allocated by PyTorch` in `loss.backward()` at `dp_actor.py:467 update_policy`, step ~79. The colocated vLLM rollout + FSDP actor update leaves <6 GiB free on one A100-80GB. PyTorch's own hint: *"try setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`."*
- **Attempt 2** (`run_alfworld_3b.log`, after applying that hint): `AssertionError: Expandable segments are not compatible with memory pool` at `vllm/device_allocator/cumem.py:150 CuMemAllocator.__init__` during `load_model`. vLLM's `CuMemAllocator` pool (sleep/wake weight offload in the colocated setup) **explicitly refuses** `expandable_segments:True`.
- **Net:** OOM forces expandable-segments; vLLM forbids it → hard block. Resolution space (unvalidated): a 2nd GPU for the ALFWorld cell, smaller `train_batch_size`/`rollout.n`, disable vLLM sleep-mode/CuMemAllocator, or offload differently. **The idle VM has 4×A100 — single-GPU colocation is the constraint, not total capacity.**

**🟠 WebShop — never trained.** `webshop_index_build.log` (2026-07-04): indexes built locally (`indexes: 3.0M`) then **6× `GcsApiError('')`** on write-through of `items_*.json`/`checksums.actual`; corpus "remains on disk + refetchable." No WebShop *training* log exists — it never got past asset staging.

---

## 5. Evidence-quality flags (for the fabrication/fidelity guards)

- **Zero-metrics fabrication:** `prj_09047604e591d969` exp#0 (all-zero + `status:"running"`); `sdar_2model_v2` (all-zero baselines → 24 vetoes).
- **all_models_failed / env_setup:** WebShop + Search-QA = 0 episodes in every 2-model run; Search-QA also a data-loader TypeError.
- **`reproduction.execution` over-claim:** `sdar_merged_full_2g` stamps `execution.ran=true/success/metrics_produced=true` while the report body is empty/`failed` — the execution stamp disagrees with the report.
- **False "completed" from the external monitor:** both `runs/_external/*/events.jsonl` emit `terminal … status:"completed" (sentinel_exited)` on process *exit*, regardless of success — ALFWorld shows `completed` directly above its `AssertionError`. Sentinel = "process exited," **not** "succeeded"; only the error events distinguish them. (Worth hardening the monitor's terminal classification.)
- **The one clean signal:** authors' Search-3B `0.456` is a genuine 7-dataset held-out eval, **not** a `reward×100` artifact — no eval-provenance concern.

---

## 6. Open questions the logs cannot answer (targets for the deeper pass)

1. **Who SIGTERM'd `sdar_2model_a` / `sdar_2model_v2` at 0 iterations?** `NO_AUTOSTOP=1` was set, so not the normal autostop — operator interrupt, wall-clock watchdog, or a VM autostop race? This early-kill pattern recurs and blocks Track A from ever starting.
2. **Did any Track-A run reach real GPU training?** Salvaged evidence (20/150 steps, all-zero, `status:"running"`) suggests no. Needs the pre-SIGTERM `experiment_runs.jsonl` from the VM boot disk (not in the GCS pull).
3. **ALFWorld OOM — paper's batch config or already reduced?** Tails don't show the effective `train_batch_size`/`rollout.n`, so we can't yet say whether a batch-shrink alone clears it or a 2nd GPU is required.
4. **Is the WebShop corpus usable?** Index built locally but GCS upload failed; corpus may be stranded on a stopped VM's disk.
5. **Why `gpu_usd=0` for all 3 campaigns?** Docker-missing implies they failed before provisioning — did the campaign driver ever attempt a GPU lease, or bail at env setup?
6. **Local vs GCS `sdar_2model_v2` divergence** (local mtime 2026-07-01 22:03 vs GCS 2026-07-02 03:07) — re-run twice, or stale local copy?

---

## 7. Model-config reality ("foundry sonnet")

**Foundry is model-agnostic — served model = `AZURE_FOUNDRY_DEPLOYMENT`, a pure `.env` change.** Nothing in the code branches on which model is deployed; the route speaks OpenAI chat-completions only (`azure_foundry_runtime.py:44-52`, `foundry_endpoint.py:68-77`).

- **Live `.env`:** endpoint `appradhann-4738-resource.services.ai.azure.com` (`/openai/v1/`), deployment **`grok-4.3`**, key present. `OPENAI_API_KEY` present (167 chars — present ≠ funded). `ANTHROPIC_API_KEY` empty. `~/.claude/.credentials.json` present (OAuth live).
- **The canonical launched script** `sdar_gcp_run.sh` **hard-forces `AZURE_FOUNDRY_DEPLOYMENT=gpt-chat-latest`** (`:72`) and runs **all tiers on Foundry**: `--model foundry --models executor=foundry,grader=foundry,verifier=foundry` (`:76,:124`). So today's launched path = **gpt-chat-latest for root + executor + grader + verifier**, OAuth-free.
- **The fidelity-correct variant** `sdar_gcp_optimal_run.sh` = Foundry **root** + **Sonnet-OAuth** executor/grader/verifier (`:35,:496`).

**"Foundry sonnet" requires two things that don't exist yet:** (a) actually deploying a Claude Sonnet model on the `appradhann-4738-resource` Foundry resource, and (b) confirming Azure exposes it on the OpenAI-compatible `/openai/v1/chat/completions` surface (not only the Anthropic-native `/v1/messages`). Even then, the harness classifies it as `foundry` (`role_models.py:177-182`) → `paper_validated=False` + advisory `role_model_fidelity` warning (never blocks). **Known risk:** gpt-chat/grok stub as the executor for SDAR — Sonnet/gpt-5 are the only paper-validated executors. In `execute` mode (Track B) this risk shrinks (the executor only writes a thin shim, not the whole trainer).

**Exact "all tiers on Foundry" wiring** (once a deployment is chosen): set `AZURE_FOUNDRY_DEPLOYMENT=<model>`, then `env -u ANTHROPIC_API_KEY … --model foundry --models executor=foundry,verifier=foundry,grader=foundry,validator=foundry`. **Caveat:** do NOT put executor AND validator on the *same* Foundry deployment — the external-validator separation ladder reads that as `degraded` (`role_models.py:369-371`); use a different family for `validator` (e.g. `validator=gpt-4o-azure`) for a real cross-check.

---

## 8. Where logs & metrics live (retrieval cheatsheet)

```bash
export CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud   # auth = abheek@deepinvent.ai

# GCS (laptop-independent) — per-run reports:
gcloud storage ls gs://deepinvent-ext-ut-sdar-runs/
gcloud storage cp -r gs://deepinvent-ext-ut-sdar-runs/sdar_2model_v2/ /tmp/sdar_pull/
gcloud storage cat gs://deepinvent-ext-ut-sdar-runs/sdar_2model_v2/final_report.md

# Off the running VM — harness run dirs + authors'-verl logs:
gcloud compute ssh abheekp@sdar-2model-a --zone us-central1-a --project deepinvent-ext-ut --command \
  'ls -la /home/abheekp/openresearch/runs/ | tail; tail -40 /mnt/sdar-cache/logs/run_alfworld_3b.log'

# Local:
#   runs/_external/<id>/events.jsonl        (external-monitor SSE log — search/alfworld)
#   runs/{sdar_validate_full,_v3_capture,prj_09047604e591d969}/  (harness run dirs)
```

**Per-run artifact set** (uploaded by `sdar_gcp_run.sh::self_stop` before autostop): `final_report.{json,md}`, `demo_status.json`, `dashboard_events.jsonl`, `experiment_runs.jsonl`, `code/metrics.json`, `sdar_gcp_run.out`.

---

## 9. Root-cause hypotheses & fix directions — UNVALIDATED (for the deeper pass)

> Framing only — not decisions. The design is not yet approved.

- **H1 (strategy):** Track A (from-scratch) has never yielded a valid grid; Track B (authors' verl) yields real numbers. The robust path to a *reproduction* is likely **`execute` mode as ground truth**, with Track A reserved for the "improve" axis only.
- **H2 (ALFWorld OOM):** single-A100 colocation of vLLM rollout + FSDP is the binding constraint on an 80 GB card. Candidate fixes: dedicate a 2nd GPU to the ALFWorld cell (the VM has 4), shrink `train_batch_size`/`rollout.n`, or disable vLLM sleep-mode so `expandable_segments:True` is usable. Needs a GPU experiment to settle.
- **H3 (early SIGTERM):** resolve §6-Q1 before any launch — a run that dies at 0 iterations wastes the whole lease. Likely a watchdog/autostop/wall-clock interaction.
- **H4 (executor fidelity):** if generation must be OAuth-free, `execute` mode minimizes the executor's role, so a Foundry executor is far safer there than in `adapt` mode. If a real Sonnet-on-Foundry is stood up, prefer it; otherwise Sonnet-OAuth executor (the `optimal` script) is the validated choice.
- **H5 (WebShop):** corpus is built but stranded; a retrieval/GCS-upload fix + one training cell is needed before WebShop contributes to the grid.

---

## 10. Reproducibility of this analysis

Regenerate with 4 parallel read-only agents over: `gcloud compute instances/disks list`, `gcloud storage ls/cp gs://deepinvent-ext-ut-sdar-runs/`, SSH `sdar-2model-a:/mnt/sdar-cache/logs/` + `runs/`, and local `runs/_external/*`, `runs/{sdar_*,_v3_capture,prj_09047604e591d969}/`, plus `backend/agents/rlm/{models,role_models}.py`, `backend/agents/runtime/foundry_endpoint.py`, `scripts/sdar_gcp_*.sh`, `.env` (redacted).

---

# PART II — Deep diagnostics (CONFIRMED), execute-mode gap analysis, and the decision fork

> Added after two further targeted passes (VM diagnostic pull + execute-mode code read). This is the authoritative current state + the plan space. Facts here are confirmed and will NOT change based on the direction chosen; only §14 depends on the fork.

## 11. Confirmed deep diagnostics

- **VM stopped** (§0/§1). Cost halted; staged data intact on pd-ssd `sdar-cache-a`.
- **ALFWorld-3B is fixable, not fundamentally blocked.** It genuinely trains and **learns** — `episode/success_rate` 0.383 (step 53) → 0.508 (step 79, peaks 0.609) — then OOMs *mid-FSDP-actor-update* at step 79. Exact authors' config (`run_alfworld_3b_4gpu.sh`, Qwen2.5-3B, 4×A100): `train_batch_size=16`, `ppo_micro_batch_size_per_gpu=32`, `rollout.n=8`, `gpu_memory_utilization=0.6`, `tensor_model_parallel_size=2`, `env.max_steps=50`, `fsdp param/optimizer offload=False`.
  - **Root cause:** long ALFWorld episodes (mean ~40 turns, `env.max_steps=50` vs Search's `4`) × `rollout.n=8` → ~**850K-token** concatenated sequences → FSDP backward + Adam step (no offload) on top of vLLM holding 60% VRAM → **fragmentation** (28 GiB reserved-but-unallocated) fills the last GiB by step 79.
  - **Memory-safe fix (grounded in the config that already succeeded on Search, NOT yet tried):** `ppo_micro_batch_size_per_gpu 32→16`, `gpu_memory_utilization 0.6→0.5`, `actor.fsdp_config.optimizer_offload=True` (Adam state → CPU); optionally cap turn/history length. **Do NOT use `expandable_segments:True`** — it's a **dead end** (vLLM `CuMemAllocator` asserts incompatible at model-load; the operator's attempt-2 crashed on exactly this). WebShop's `run_webshop_3b_patched.sh` carries the same self-defeating `expandable_segments` and must get the same treatment.
- **SIGTERM-at-0-iters — RESOLVED (not a harness bug).** `sdar_2model_a` was **manually killed** ~40s in (root `foundry`/gpt-chat flagged `root_model_risk=unvalidated`; operator aborted and pivoted to the bash proof). Autostop was OFF and wall-clock was 24–36h, so no knob fired. Won't recur under normal operation.
- **WebShop — staged, never trained.** 5.3G corpus + built index + `verl-webshop` conda env + `run_webshop_3b_patched.sh` all present; the training launch was simply **never fired**.
- **Staged-cache reuse map** (`/mnt/sdar-cache`, pd-ssd, survives the stop): SDAR repo `ZJU-REAL/SDAR@9f2ce6a82a90cc5a334d73f188c16df2c4107d80`; conda envs `sdar` (py3.12, ALFWorld+Search), `retriever`, `verl-webshop`; `HF_HOME=/mnt/sdar-cache/hf` with **Qwen2.5-3B-Instruct + Qwen3-1.7B + Qwen2.5-7B**; `.warm_ok` + `.staging_ready` present.

## 12. Execute-mode gap analysis — the architectural crux

**Repo-first `execute` mode is a prompt-note + verbatim code-copy with ZERO footprint in the execution machinery.** (`grep execute|reproduction_mode|repo_spec` finds nothing in `run_experiment`/`gpu_cell_runner.py`/process backends.) The `execute_repo_note` (`baseline_implementation.py:2816-2829`) tells the **executor model** to hand-author a shim: `cells.json`+`train_cell.py` invoking the authors' entrypoints, dep wiring, and a metrics adapter. **The harness provides none of it, validates none of it, and `test_execute_mode.py` pins only that the note text appears in the prompt.** The proven 0.456 came from hand-run bash, bypassing the harness.

**Blocking gaps** (to reach "harness drives authors' verl on the staged cache, all-Foundry, full grid"):
- **GAP 1 (crux): no conda/launcher seam.** `gpu_cell_runner.py:676` hardcodes `[sys.executable, train_cell.py]`; running `conda run -n sdar bash run_<env>.sh` requires the agent to hand-roll `subprocess.run(["conda","run",...])` inside `train_cell.py` (brittle) — the per-cell interpreter override is a documented UNIMPLEMENTED TODO (`webshop_env.py:21-22`).
- **GAP 3: staged-cache env not first-class.** `HF_HOME`/asset dirs reach the trainer only by `os.environ` inheritance, and are actively fought by the sandbox-contract prompt (`_sandbox_contract.py:58`) + `asset_provisioning.py:340` clobber.
- **GAP 5: no verl→metrics adapter.** No parser for verl's `val/success_rate` (wandb/stdout) into the harness `metrics.json` exists in code — 100% agent-authored, aspirational.

**Sharp edges:** GAP 2 (clone is github-only, wipes `repo/`, no local-path/commit-pin → can't reuse `/mnt/sdar-cache/SDAR`); GAP 4 (`env_pin` force-installs cu121 torch → breaks the verl/vLLM stack); GAP 6 (monolithic `_resolve_distributed_launch` may accelerate-wrap the authors' launcher); GAP 8 (a failed clone **silently downgrades to from-scratch** `mode="scratch"`); **GAP 9 (Foundry executor stubs on the nontrivial shim** — the direct contradiction with "all-Foundry execute": the model that must author the shim is the one documented to stub).

**Recommended architecture** (once seams land): the **cells route** — 6 cells (model×env), each `"gpus": N`, each shelling `conda run -n sdar bash examples/sdar_trainer/run_<env>_<model>.sh` under a launcher seam, each writing a flat `metrics.json` a harness verl-adapter fills, `cell_matrix.aggregate_cell_metrics` → `per_model` → leaf scorer → `verify_against_rubric` → report (this tail already works). Full ordered gap list with file targets: agent `ad4d4f3a` investigation.

## 13. Decision fork (PENDING — set before §14 firms up)

User picks (Q asked in-session): **(A)** build the 3 harness seams → then all-Foundry execute (durable/robust, ~3 modules + guard tests); **(B)** Foundry root + pinned validated executor (Sonnet/gpt-5) authors the shim (no harness code, faster, not "all-Foundry", shim still brittle); **(C)** skip the harness, fix + run the authors' bash proof scripts directly, harness only scores/reports (lowest risk to a real number, not "end-to-end through the harness").

## 14. Recommended plan (skeleton; specializes on the §13 fork)

Common to all paths: **(P0)** fix the ALFWorld + WebShop verl configs per §11 (port Search's memory-safe knobs, strip `expandable_segments`); **(P1)** turn autostop **ON** (self-stop on completion/crash — prevents another idle burn); **(P2)** set money ceilings (`--max-run-gpu-usd` + wall-clock); **(P3)** enable honesty guards via `--run-spec` (`ENV_LIVENESS_GATE`, `EVAL_PROVENANCE_GUARD`, `ZERO_METRICS_GUARD`). Then: **(P4)** validate on the proven Search-3B cell (expect ~0.456) before committing the full grid; **(P5)** run the full 3-env × {Qwen3-1.7B, Qwen2.5-3B} grid. Option A adds the seam-build (GAP 1/3/5) before P4; Option B pins the executor before P4; Option C replaces P4/P5 with the bash proof scripts + a harness scoring wrapper.
