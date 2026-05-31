# OOM + GPU-Capacity Remediation Design (2026-05-31)

**Status:** ✅ IMPLEMENTED (2026-05-31). All components landed + full suite green (3430 passed). See the handoff doc for the per-component commit map. Local is the verified build target; cloud escalate + azure stub are designed-not-built per §6.
**Motivating failure:** `runs/prj_09047604e591d969` (SDAR baseline, `--sandbox local`, 8×A5000, profile `max`) — 11 attempts on 2026-05-31, final verdict `failed`, rubric **0.0**, ~$10 burned. Hundreds of repeated CUDA OOMs.

**What landed (component → where):**
- Comp 1 `gpu_capacity.describe_capacity(ctx) → GpuCapacity` (backend-agnostic).
- Comp 2 env ABC: `sdar_env_base.BaseEnv` (copied into `code/`) + always-on self-gating guidance + `preflight_ast._check_env_interface_contract` (recursive, self-scoping AST backstop).
- Comp 3 cell contract: `_compute_constraint_guidance` injects the GPU-budget brief + single-cell `train_cell.py`/`cells.json` contract + always-on memory discipline (no fp32 full-vocab log_softmax); suppresses torchrun multi-GPU guidance on the cell path. `caps` threaded `implement_baseline → run_with_sdk`.
- Comp 4 placement: `cell_matrix.py` pure functions (capacity_gate / dataset_url_preflight / aggregate_cell_metrics → `per_model[model][env][baseline]` leaf shape) + `run_experiment._execute_cell_matrix` routes the matrix through `gpu_cell_runner.run_matrix` (one GPU per cell) when `cells.json` is present on a local/docker GPU backend; fail-soft to the legacy path otherwise.
- Comp 4b/6 STOP: terminal `oom_shrink_exhausted` / `capacity_exhausted` classes; `forced_iteration` accepts FINAL_VAR for them (check 0.4); `run.py` calls `policy.note_terminal_failure` and stashes `ctx._terminal_stop_reason`; `final_report.json.stop_reason` surfaces it.
- Comp 5 launcher: `reserve_and_run_sdar.py` drops `REPROLAB_DISABLE_TORCHRUN_WRAP` + `REPROLAB_FORCE_SINGLE_GPU` (lease + cell runner own placement).

## 1. Root causes (reconciled across 4 investigations)

**OOM (all `search_qa`/`webshop` cells):**
1. **Single-card collapse.** Every OOM is on `GPU 0`; all 18 cells ran on **one** A5000. **Corrected cause (per integration map — supersedes the earlier "never propagated" theory):** `batch_reproduce.py:347-350` *does* propagate the leased GPU UUIDs as `CUDA_VISIBLE_DEVICES` + `REPROLAB_GPU_DEVICE_IDS`. The collapse is because (a) the agent's `code/train.py` **coordinator** hard-codes `device="cuda:0"` and loops the whole matrix in one process (`:800-801,863-900`), and (b) `reserve_and_run_sdar.py:101` sets `REPROLAB_DISABLE_TORCHRUN_WRAP=1`, disabling `_resolve_distributed_launch` — the only seam that would re-launch under `accelerate`. The `dashboard_events.jsonl gpu_count:1` is a **red herring**: it's the informational RunPod `GpuPlan`, not the real local device pinning (`ctx.gpu_device_ids`). → The harness-owned cell runner (decision #2) obviates the coordinator entirely.
2. **fp32 full-vocab logprobs blowup.** `code/sdar/utils.py` materializes `F.log_softmax(logits.float())` of shape `[B, T, 151936]`, **3× per step**, held in the autograd graph to backward ≈ **20 GB**. This is the training process's *own* 22 GB (not a foreign tenant — the PyTorch string was misread by one investigator). It is why **even Qwen3-1.7B OOMs**.
3. **7B full-FT can't fit 24 GB** regardless (~28 GB optimizer state alone).

**Repetition:** `train.py` catches the OOM and exits 0 (`silent_oom`) → bypasses the escalation gate → on `local` the catalog ladder is empty anyway → `forced_iteration` refuses `FINAL_VAR` and forces 3 iterations of the *same* failing config. No mechanical shrink exists.

**Independently fatal, non-OOM (all `alfworld` cells):**
4. **`ALFWorldEnv.build_student_prompt` AttributeError** — pure-Python interface bug; kills 18 cells with zero GPU involvement.
5. **WebShop dataset HTTP 404** — dead endpoint.

→ no `metrics.json` → rubric `degraded_no_metrics` → **0.0**. **Fixing OOM alone cannot make this run pass** (half the matrix dies on #4/#5).

**Pre-existing asset:** `backend/agents/rlm/gpu_cell_runner.py` (+ `tests/agents/rlm/test_gpu_cell_runner.py`) already implement a harness-owned, one-GPU-per-cell subprocess pool with OOM shrink-retry — written, never wired in, never copied into agent code. The failure was that it was **opt-in**.

## 2. Locked decisions (from grill-me)

| # | Decision | Choice |
|---|----------|--------|
| 1 | OOM/capacity policy | **Prevent → recover → stop.** No budget-burning re-OOM loop. |
| 2 | GPU placement | **Harness-owned & mandatory** via `gpu_cell_runner`. Agent writes only a single-cell trainer. |
| 3 | Model too big for one card | **Auto-drop + honest rubric gap.** Local 24 GB → default proven smallest-two (Qwen3-1.7B + Qwen2.5-3B). |
| 4 | Backend coverage | **Backend-agnostic abstraction now**; implement+test **local** (and runpod/brev share its shape); **Azure = documented adapter stub** (no VM provisioning this effort). |
| 5 | Cloud recovery | Cloud = shrink → **escalate SKU** → stop. **Designed but not the build/test target.** |
| 6 | Local behavior | **Use as many free A5000s as possible, split across them, never OOM.** Primary build+test target. |
| 7 | Non-OOM bug scope | OOM/GPU **+ the two cheap independently-fatal blockers** (#4 ALFWorld ABC, #5 WebShop preflight). Lossy parse out of scope. |

## 3. Architecture — one abstraction, three stages, two recovery axes

**Capacity descriptor (backend-agnostic):**
```
describe_capacity() → { num_gpus, per_gpu_vram_gb, free_gpu_ids[], can_escalate, backend_kind }
  local   → local_gpu_allocator (free cards, excludes tenant-held); can_escalate=False
  runpod  → pod SKU vram × count (gpu_plan/pod spec);               can_escalate=True
  brev    → same shape as runpod;                                    can_escalate=True
  azure   → ADAPTER STUB (raises NotImplementedError w/ guidance);   can_escalate=True (future)
```

**Stage PREVENT (capacity gate).** Clamp the matrix to *what fits one GPU of the current backend*. Same paper auto-scopes per backend: 24 GB → drop 7B; 80 GB → keep it. The agent is **told the budget** (`You have N GPUs × M GB; per-cell budget = M GB; models that fit one card: …`) so it scopes deliberately. Dropped models → explicit rubric gaps.

**Stage RECOVER — two axes:**
- **Shrink (universal):** batch↓ → grad-ckpt → drop-model. Floor on every backend; `gpu_cell_runner` already does 0.5→0.25+grad-ckpt per cell.
- **Escalate SKU (cloud only):** `gpu_catalog.find_ladder`, bounded by `REPROLAB_MAX_RUN_GPU_USD`. Local skips this axis.

**Stage STOP + report.** When live axes are exhausted → stop cleanly with a **structured stop reason**; *no* forced-iteration re-loop. Report surfaces: (a) `final_report.json.stop_reason` block, (b) a `run_error`/`run_warning` SSE event, (c) a clear console/log line.

**Local spine (the fix for the motivating bug):**
```
launcher leases all free A5000s → propagates CUDA_VISIBLE_DEVICES + REPROLAB_GPU_DEVICE_IDS
  → run_experiment routes through gpu_cell_runner.run_matrix(cells, train_cell.py, gpus=<leased>)
  → one GPU per cell, min(free_gpus, len(cells)) in parallel
  → per-cell shrink-retry → capacity gate drops 7B → never stacks, never over-scopes
```

## 4. Components / change list

1. **`gpu_capacity.py` (new)** — `describe_capacity(ctx)` + per-backend providers (local real; runpod/brev real-ish; azure stub). Single source of truth for budget + free GPUs.
2. **Capacity gate (PREVENT)** — in `run_experiment`/`plan_reproduction`: clamp scope to per-GPU VRAM, emit dropped-model gaps, inject the GPU-budget brief into the `implement_baseline` prompt.
3. **Wire `gpu_cell_runner` mandatory (PLACEMENT)** — `run_experiment`, when backend exposes ≥1 GPU, executes the matrix via `run_matrix` instead of monolithic `python train.py`. **Risk H — the aggregation contract:** the agent emits `code/cells.json` (the ONLY place the 5-baseline axis exists — `ScopeSpec` has only model×dataset×seed), and the aggregator must synthesize the exact `per_model.<m>.per_env.<e>` nesting the ~8 postflight guards + `leaf_scorer` consume, and write `scope.gaps`/`models_skipped` for capacity-dropped / URL-dead cells (keys `leaf_scorer.py:456,509` read but the harness has never written). Mutually exclusive with `_resolve_distributed_launch`.
4. **Launcher propagation fix** — `reserve_and_run_sdar.py`/`batch_reproduce.py`: lease all free A5000s and propagate the GPU id CSV. Reconcile/retire `REPROLAB_DISABLE_TORCHRUN_WRAP` for the cell-runner path.
5. **`implement_baseline` contract** — guidance produces a single-cell `train_cell.py` (reads `REPROLAB_CELL_PARAMS`/`REPROLAB_CELL_OUTPUT_DIR`, writes `metrics.json`) + a cell manifest; **forbids fp32 full-vocab `log_softmax`** (require gather + `cross_entropy`/chunked logsumexp, bf16, `use_cache=False`, grad-ckpt).
6. **OOM → stop, not loop (STOP)** — introduce a NEW terminal failure class `oom_shrink_exhausted` (do NOT reuse `silent_oom` — it correctly stays repairable for a *single* caught-OOM that hasn't exhausted shrink). Set it after `run_matrix`'s per-cell retries are spent; `forced_iteration.should_refuse` adds a bypass so it does NOT refuse `FINAL_VAR` for that class (guard all three refusal paths: repair, no-rubric, below-target).
7. **Env interface ABC (#4)** — envs are **100% agent-generated** (`code/sdar/envs/`), so this is a **copyable base module** (`sdar_env_base.py`: `BaseEnv(ABC)` + `@abstractmethod build_student_prompt`/`build_teacher_prompt`), auto-copied into `code/` via the same copy hook as #5, + guidance to subclass it, + a **preflight AST check** that fails loudly if an `*Env` is defined without the methods (the ABC makes construction raise `TypeError`, the AST check catches it before the grid). NOT a harness code change — the harness ships no env classes.
8. **Dataset preflight (#5)** — `detect_environment` runs *before* the agent writes its URLs, so the probe lives in **`run_experiment` at cell-enumeration**, keyed on an agent-declared `cells.json::dataset_url`; on 404 (bounded HEAD probe, fail-soft so a transient blip never drops a live env) drop that env's cells to an honest `scope.gaps` entry *before* the grid. Plus a dead-URL alias steering the agent off the known-dead `princeton-nlp/WebShop/master` path.

## 5. Testing

- Unit: `test_gpu_cell_runner.py` (extend — pinning, pool parallelism, OOM shrink-retry, every-cell-has-a-result). `describe_capacity` per backend (mock nvidia-smi / pod spec / azure stub raises). Capacity-gate clamp (24 GB drops 7B; 80 GB keeps it). forced-iteration OOM-exhaustion bypass.
- Contract: env ABC rejects a subclass missing `build_student_prompt`. Dataset preflight drops a 404 URL to a gap.
- Integration (local, opt-in/marked): a tiny 2-cell matrix splits across 2 GPUs, no stacking; an over-scoped cell shrinks then drops with a gap; whole run STOPS with a structured reason instead of looping.

## 6. Out of scope

Azure VM/AML provisioning (adapter stub only); ingestion lossy-parse hardening (not causal here); cloud escalate is implemented in the abstraction but local is the verified target.
