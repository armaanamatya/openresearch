# HANDOFF — OOM + GPU-Capacity Remediation (2026-05-31)

**Goal:** make the RLM reproduction harness *never* OOM-loop again. Use **all** free local GPUs, split the training matrix across them (one GPU per cell), drop work that can't fit one card (honest gap), and on unrecoverable OOM **stop cleanly + report** instead of burning iterations. Backend-agnostic (local now; runpod/brev share the shape; azure = adapter stub).

**Canonical refs (read in order):**
1. `docs/superpowers/specs/2026-05-31-oom-gpu-capacity-remediation-design.md` — the locked design + root causes (the "why").
2. **This doc** — status, exact execution plan, ops, resume prompt.

**Design is LOCKED via `/grill-me` — do NOT re-litigate the 7 decisions below.** Resume by executing §4.

---

## 1. Locked decisions (grilled)

| # | Decision | Choice |
|---|----------|--------|
| 1 | OOM/capacity policy | **Prevent → recover → stop.** No budget-burning re-OOM loop. |
| 2 | GPU placement | **Harness-owned & mandatory** via `gpu_cell_runner`. Agent writes only a single-cell trainer. |
| 3 | Model too big for one card | **Auto-drop + honest rubric gap.** Local 24 GB → proven smallest-two (Qwen3-1.7B + Qwen2.5-3B). |
| 4 | Backend coverage | **Backend-agnostic abstraction now**; implement+test **local**; azure = documented stub. |
| 5 | Cloud recovery | shrink → **escalate SKU** → stop. Designed, not the build/test target. |
| 6 | Local behavior | **Use as many free A5000s as possible, split, never OOM.** Primary target. |
| 7 | Non-OOM scope | OOM/GPU **+ the two cheap independently-fatal blockers** (env-ABC, WebShop preflight). Lossy parse out. |

**Architecture:** one `describe_capacity()` descriptor → 3 stages (**PREVENT** clamp-to-per-GPU-VRAM, **RECOVER** shrink[all]/escalate-SKU[cloud], **STOP**+report) → local spine: *launcher leases all free A5000s → `run_experiment` routes matrix through `gpu_cell_runner.run_matrix(gpus=leased)` → one GPU per cell, `min(free_gpus, cells)` parallel → per-cell shrink-retry → drop 7B.*

⚠️ **Fixing OOM alone will NOT make the run score > 0.** The 2026-05-31 run died 50/50: OOM on `search_qa`/`webshop` cells **and** a pure-Python `ALFWorldEnv.build_student_prompt` AttributeError on all `alfworld` cells + a WebShop 404. Components **2** (env ABC) and the WebShop preflight in **4** are as load-bearing as the GPU work.

---

## 2. Status — ✅ COMPLETE (2026-05-31)

**ALL components landed + tested. Full suite green: `pytest tests/ -q` → 3430 passed, 6 skipped (env-gated), 1 xfailed.** Per-component commits on `5.30.26_sdar` (authored lolout1, no Claude trailer):

| Comp | Commit | What |
|------|--------|------|
| 1 | (pre-existing) | `gpu_capacity.describe_capacity` (12 tests) |
| 6 | (pre-existing) | `forced_iteration` terminal-OOM bypass (7 tests, 84 no-regress) |
| 2 | `feat(rlm): SDAR env interface ABC + preflight backstop` | `sdar_env_base.BaseEnv` + copy hook + guidance + recursive self-scoping AST check (16 tests) |
| 3 | `feat(rlm): single-cell trainer contract + GPU-budget guidance` | budget brief + cell contract + memory discipline + torchrun suppression + `caps` threading (6 tests) |
| 4 | `feat(rlm): route run_experiment through the GPU cell runner` | `cell_matrix.py` (32 tests) + `_execute_cell_matrix` route + fail-soft (12 tests) |
| 4b | `feat(rlm): terminal-stop wiring — stop + report, never re-OOM` | `note_terminal_failure` wiring + `final_report.stop_reason` (6 tests) |
| 5 | `feat(scripts): reconcile SDAR launcher flags for the cell runner` | drop torchrun-wrap + force-single-GPU |
| 7 | `test(rlm): run_experiment cell-route branch wiring` | end-to-end branch + manifest-gate coverage |

**Next:** §8 resume prompt is satisfied; remaining work is the LIVE SDAR validation run (§7 done-criteria #2) — launch via §6 and iterate to `rubric_score > 0`.

**Verify the done work:**
```bash
.venv/bin/python -m pytest tests/services/runtime/test_gpu_capacity.py \
  tests/agents/rlm/test_forced_iteration_oom_bypass.py tests/agents/rlm/test_gpu_cell_runner.py -q
```

---

## 3. Pre-existing assets you build ON

- `backend/agents/rlm/gpu_cell_runner.py` — **complete, 37 tests pass.** One-GPU-per-cell subprocess pool + OOM shrink-retry (`run_matrix(cells, cell_script, *, output_root, gpus, max_oom_retries=2, per_cell_timeout_s)`). Per-cell env contract it SETS: `CUDA_VISIBLE_DEVICES=<one id>`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `REPROLAB_CELL_PARAMS=<cell json>`, `REPROLAB_CELL_OUTPUT_DIR`, `REPROLAB_CELL_BATCH_SCALE` (0.5 then 0.25 on retry), `REPROLAB_CELL_GRAD_CHECKPOINT=1`; argv `--cell-id`, `--output-dir`. **The agent's `train_cell.py` must READ these.**
- `backend/services/runtime/local_gpu_allocator.py` — `discover_gpus()→[GpuDevice]`, `free_devices(...)` (excludes tenant-held cards). Already used by `gpu_capacity.py`.
- `backend/services/runtime/gpu_catalog.py` — `find_ladder()` (cloud SKU escalation; local ladder is empty by design).

---

## 4. Execution plan (per remaining component)

### Component 2 — Env interface ABC (unblocks ALFWorld; independent of OOM)
**Why:** all 18 `alfworld` cells died on `AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'`. Envs are **100% agent-generated** (`runs/<id>/code/sdar/envs/`); the harness ships none — so this is a **copyable module + guidance + AST preflight**, NOT a harness env class.

**2a. New copyable module** `backend/agents/rlm/sdar_env_base.py` (mirror `gpu_cell_runner.py`):
```python
"""Copyable base class for SDAR environments — makes a missing trainer method a
construction-time TypeError, not a mid-grid AttributeError (2026-05-31 fix)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
__all__ = ["BaseEnv"]

class BaseEnv(ABC):
    # Signatures are permissive on purpose — subclasses use whatever params fit
    # their data; the contract is only that these methods EXIST.
    @abstractmethod
    def build_student_prompt(self, *args: Any, **kwargs: Any) -> str: ...
    @abstractmethod
    def build_teacher_prompt(self, *args: Any, **kwargs: Any) -> str: ...
```
Test (`tests/agents/rlm/test_sdar_env_base.py`): a subclass missing `build_student_prompt` raises `TypeError` on construction; a complete one constructs.

**2b. Auto-copy into `code/`** — there is **no existing auto-inject hook** (`rubric_guard.py` lands via the *prompt*). Add a real copy in `backend/agents/baseline_implementation.py` `run_with_sdk`, right after `code_dir.mkdir`/`_copy_source_pdf_to_code_root` (~line **2071**), mirroring `baseline_knowledge.write_curated_artifacts`:
```python
import shutil
shutil.copy2(Path(__file__).parent / "rlm" / "gpu_cell_runner.py", code_dir / "gpu_cell_runner.py")
shutil.copy2(Path(__file__).parent / "rlm" / "sdar_env_base.py", code_dir / "sdar_env_base.py")
```
**2c. Guidance** — add a block (next to `_RUBRIC_GUARD_BLOCK`, ~line **1909**) telling the agent: "every `*Env` MUST `from sdar_env_base import BaseEnv` and subclass it."
**2d. Preflight AST check** — mirror `pre_flight_validator.py`/`preflight_ast.py`: fail if any class named `*Env` is defined without `build_student_prompt`/`build_teacher_prompt` and without a `BaseEnv` base. Risk **M** (enforcement depends on the agent subclassing; AST check is the backstop).

### Component 3 — `implement_baseline` cell contract
In `backend/agents/baseline_implementation.py` `_compute_constraint_guidance` (def **1778**, append before `return` ~**2026**) add three blocks; thread `caps = describe_capacity(ctx)` in from `run_with_sdk` (**2124**) ← `implement_baseline` (`primitives.py:1322`):
- **(a) GPU budget brief:** `"You have {caps.num_gpus} GPU(s) × {caps.per_gpu_vram_gb:.0f} GB. Per-cell budget = {per_gpu} GB. A model that cannot full-fine-tune in that budget is OUT OF SCOPE — record it in scope.gaps."` (24 GB ⇒ smallest-two; never the 7B.)
- **(b) cell contract:** "Write a single-cell trainer `train_cell.py` that trains **one** cell, reading `REPROLAB_CELL_PARAMS` (JSON) + `REPROLAB_CELL_OUTPUT_DIR` + argv `--cell-id`/`--output-dir`, honoring `REPROLAB_CELL_BATCH_SCALE`/`REPROLAB_CELL_GRAD_CHECKPOINT`, and writing `metrics.json` to the output dir. Also emit `cells.json` enumerating the matrix (schema below). Do NOT write a monolithic coordinator that loops `cuda:0`." (place near `_RUBRIC_GUARD_BLOCK`, ~**1909**.)
- **(c) memory discipline (always-on, after `_EAGER_METRICS_BLOCK` ~**1860**):** "FORBIDDEN: materializing a full-vocab fp32 `log_softmax` `[B,T,vocab]`. Compute token log-probs with `torch.gather` on logits + chunked `logsumexp` (or `F.cross_entropy(reduction='none')`). Use bf16, `model.config.use_cache=False`, `gradient_checkpointing_enable()`, `mini_batch ≤ 2` for ≥3 B." (this is the ~20 GB blowup that OOM'd even the 1.7 B.)

`cells.json` schema (the ONLY place the 5-baseline axis exists — `ScopeSpec` has only model×dataset×seed):
```json
{"cells": [{"id": "qwen3_1_7b__sdar__search_qa__s42", "model_id": "Qwen/Qwen3-1.7B",
  "model_key": "qwen3_1_7b", "baseline": "sdar", "env": "search_qa", "seed": 42,
  "dataset_url": "https://...", "est_vram_gb": 14.0}]}
```

### Component 4 — Wire `run_matrix` into `run_experiment` (RISK **H** — the load-bearing change)
`backend/agents/rlm/primitives.py`, `run_experiment` (def **3533**). Insert the route **after** commands/env resolution (~**3692**) and **before** the escalation `while True:` (**3735**):
```
caps = describe_capacity(ctx)
if caps.backend_kind in ("local","docker") and not caps.is_empty:
    cells = json.load(code/"cells.json")["cells"]
    cells = _dataset_url_preflight(cells)        # HEAD-probe dataset_url; 404 → scope.gaps (fail-soft, bounded)
    cells = _capacity_gate(cells, caps)          # drop est_vram_gb > caps.per_gpu_vram_gb/headroom → scope.gaps
    res  = gpu_cell_runner.run_matrix(cells, code/"train_cell.py",
                output_root=artifact_root, gpus=list(ctx.gpu_device_ids) or caps.free_gpu_ids,
                per_cell_timeout_s=resolve_experiment_timeout_s(ctx))
    metrics = _aggregate_cell_metrics(res, dropped_gaps)   # ⇐ see contract
    # fall through to the SAME postflight guards (3907-4096) + _persist_experiment_result
```
**Aggregation contract (the H-risk):** `_aggregate_cell_metrics` MUST emit the exact `metrics.json` shape the ~8 postflight guards (`_metrics_completeness_violation`, `_validate_scope_metrics`, `_training_health_violation`, rubric_contract, metrics_shape) and `backend/evals/paperbench/leaf_scorer.py` (gaps read at **456, 499-541**) consume — i.e. `per_model.<model_key>.per_env.<env>...` nesting + `scope.gaps`/`scope.models_skipped`. **De-risk:** have `train_cell.py` write per-cell metrics already in leaf shape so the aggregator just nests + merges; validate against a real prior `metrics.json` sample (e.g. `runs/prj_09047604e591d969/code/outputs/*/metrics.json`) before trusting it.
**Terminal stop (uses component 6):** if every cell is `oom_failed` after shrink-exhaustion, set `result["stop_reason"]={"kind":"oom_shrink_exhausted",...}`, `success=False`, and call `policy.note_terminal_failure("oom_shrink_exhausted")` (wire near the repair-attempt call at `run.py:660-672`) so FINAL_VAR is accepted → run stops + reports. **Do NOT reuse `silent_oom`** (it correctly stays repairable for a *single* caught OOM; `_training_health_violation` at **2190**, markers **2017/2206**).
**Mutual exclusion:** skip `_resolve_distributed_launch` (**2401-2509**, fires only when `len(gpu_device_ids)>1`) on the cell-runner branch.

### Component 5 — Launcher flag
`scripts/reserve_and_run_sdar.py:101` sets `env["REPROLAB_DISABLE_TORCHRUN_WRAP"]="1"`. With harness-owned one-GPU-per-cell, remove it (or gate it off the cell path). **`batch_reproduce.py:347-350` already propagates `CUDA_VISIBLE_DEVICES`+`REPROLAB_GPU_DEVICE_IDS` correctly — leave it.** (The 2026-05-31 collapse was the agent coordinator looping `cuda:0` + this flag, NOT a propagation gap.) Risk **L**.

### Component 7 — Integration/contract tests + full suite
Capacity-gate clamp (24 GB drops 7B, 80 GB keeps it); env-ABC rejects incomplete subclass; dataset preflight drops a 404 to a gap; aggregation yields leaf-scorer-shaped metrics (assert against a real sample); `run_matrix` route splits a 2-cell matrix across 2 mock GPUs with no stacking. Then `.venv/bin/python -m pytest tests/ -q` green; keep the 37 `gpu_cell_runner` + 84 forced-iteration tests passing. Mock nvidia-smi/subprocess exactly as `test_gpu_cell_runner.py` does.

---

## 5. Gotchas (these contradict the naive reading — from the integration map)

- **Two `train.py` files:** `code/train.py` (53 KB **coordinator**, the `commands.json` target, loops `cuda:0`) vs `code/sdar/train.py` (per-cell trainer, has `train_cell()`). Your `train_cell.py` REPLACES the coordinator's role.
- **`gpu_resolved gpu_count:1` is a RED HERRING on local** — it's the informational RunPod `GpuPlan`, not the real pinning (`ctx.gpu_device_ids`). Don't key local decisions on it.
- **`ScopeSpec` has no baseline axis** (`schemas.py:866`) — the 5 baselines live only in agent code. The harness needs `cells.json` to know the full matrix.
- **`scope.gaps`/`models_skipped` were always agent-written** — the aggregator (comp 4) is the first harness code to WRITE them; the leaf scorer already reads them.
- **Dataset preflight can't live in `detect_environment`** (`primitives.py:641`) — it runs before the agent writes its URLs (`code/sdar/envs/webshop.py:62` hardcodes the dead `princeton-nlp/WebShop/master` 404). Do it at cell-enumeration in `run_experiment`, keyed on `cells.json::dataset_url`. Probe must be bounded + fail-soft.
- **The 22 GB on GPU 0 was the training process's OWN memory** (fp32 logprobs ×3), not a foreign tenant — comp 3(c) is the real fix.

---

## 6. Ops / environment

- **Box:** shared 8×NVIDIA RTX A5000 (24 GB). `local_gpu_allocator` excludes tenant-held cards — "use as much as possible" = all currently-free cards.
- **Run tests:** `.venv/bin/python -m pytest tests/<path> -q` (pytest config in `pyproject.toml`; `.venv` is the project venv, Python 3.12).
- **Launch an SDAR run:** `.venv/bin/python scripts/reserve_and_run_sdar.py` (reserves GPUs, sets `SDAR_GUIDANCE_FILE`/`REPROLAB_BASELINE_EXTRA_GUIDANCE`, delegates to `scripts/batch_reproduce.py --gpus-per-run N --model claude-oauth`). Full context: `docs/runbooks/2026-05-23-sdar-baseline-handoff.md`.
- **Guidance file:** ensure `reserve_and_run_sdar.py` points `SDAR_GUIDANCE_FILE` at the **smallest-two** guidance (`extra_guidance_sdar.txt`), NOT `extra_guidance_sdar_full.txt` (the full-7B-matrix file regressed the run). Drop the `REPROLAB_FORCE_SINGLE_GPU="false"` override — with the cell runner, GPU count is owned by the lease, not a flag.
- **Auth (CLAUDE.md):** leave `ANTHROPIC_API_KEY=` empty, `claude login` once, `--model claude-oauth` for $0 sub-agents; shell env shadows `.env` (prefix `env -u OPENAI_API_KEY` if a stale shell key bites).
- **Verify the GPU split live:** during a run, `nvidia-smi` should show **multiple** A5000s busy (one per concurrent cell), never all load on GPU 0. Each cell's log shows its pinned `CUDA_VISIBLE_DEVICES`.

---

## 7. Done-criteria

1. `pytest tests/ -q` green (incl. the 19 new + 37 cell-runner + 84 forced-iteration).
2. Live SDAR run: cells **split across ≥2 GPUs** (nvidia-smi), no `cuda:0` stacking; **all `alfworld` cells produce metrics** (no `build_student_prompt` AttributeError); WebShop 404 → honest gap, not a crash; any residual OOM **stops cleanly with `stop_reason.kind`** and ships a report (no 3-iteration re-OOM loop); **rubric_score > 0**.
3. `final_report.json` carries `stop_reason` when capacity-stopped.

---

## 8. Resume prompt (paste into a fresh session)

> Read `docs/runbooks/2026-05-31-oom-gpu-remediation-handoff.md` and `docs/superpowers/specs/2026-05-31-oom-gpu-capacity-remediation-design.md`. The design is locked (grilled) — do not re-litigate. Components 1 (`gpu_capacity.py`) and 6 (forced_iteration terminal-OOM bypass) are DONE + tested. Execute the remaining components **2 → 3 → 4 → 5 → 7** per §4, landing each with tests green before the next. Component 4 is risk-H: validate the metrics aggregation shape against a real `runs/*/code/outputs/*/metrics.json` sample and `leaf_scorer.py`. Match surrounding code style; tests alongside each component; do not restore eval/exec in the REPL patch; update both the spec and `CLAUDE.md` when you add the new primitive contract/SSE behavior. Run `pytest tests/ -q` at the end.

**Best practices to carry:** small tested increments in dependency order; the harness-owned cell runner and `_resolve_distributed_launch` are mutually exclusive; clamp config at the harness (don't trust agent-written code to be OOM-safe); honest gaps over silent drops; OOM → stop+report, never loop. **Skills used:** `/grill-me` (design locked — don't redo), superpowers spec-driven (spec is the source of truth). Commit messages end with the `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer; branch off `main` before committing.
