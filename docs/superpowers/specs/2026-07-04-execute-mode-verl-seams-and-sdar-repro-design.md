# Execute-mode execution seams + SDAR full-grid reproduction — design

> **Status:** DRAFT for user review (brainstorming output). Author: Opus (design), execution to be delegated to Sonnet with Opus diff-review.
> **Date:** 2026-07-04. **Branch:** `reconcile/grounded-self-improvement-on-main`.
> **Companion analysis (source of truth for the evidence):** [`docs/audits/2026-07-04-sdar-gcp-runs-log-analysis.md`](../../audits/2026-07-04-sdar-gcp-runs-log-analysis.md).
> **Locked decisions (user, 2026-07-04):** Option A (build the harness seams → then all-Foundry execute) · validate Search-3B (≈0.456) before the full grid · ~$400 GPU ceiling · autostop ON.

## 1. Goal & non-goals

**Goal.** Make the harness *actually* drive the authors' SDAR `verl` trainer verbatim, end-to-end, on the pre-staged GCP cache disk — root + executor + grader + verifier all on the existing Foundry deployment — and produce a scored `final_report` for the full 3-env × smallest-two-model grid ({Qwen3-1.7B, Qwen2.5-3B} × {ALFWorld, WebShop, Search-QA}). The proven signal (Search-3B = 0.456) must be reproducible *through the harness*, not just hand-run bash.

**Root cause being fixed.** Repo-first `execute` mode is today only a prompt-note + verbatim code-copy — it has **zero footprint in the execution machinery**, so the entire conda-launch + verl-run + metrics-parse "shim" falls on the executor model, and the Foundry executor stubs on exactly that (audit §12). The fix is to move the shim's hard parts *into the harness* as reusable seams, collapsing the executor's job to a fillable manifest.

**Non-goals.** (a) Not changing the LLM-grade path or any scoring semantics — the aggregate → `per_model` → leaf-scorer → report tail already works once `metrics.json` is populated. (b) Not touching adapt/reference modes or non-execute runs — every change is default-OFF / byte-identical when the new fields/flags are absent. (c) Not making WebShop *serve* better as an environment (that's a separate provisioning task) — only launching its already-staged training. (d) No autonomous default-flips; the SDAR run is operator-driven.

## 2. The unifying abstraction — generalize the cell job

Today a cell is implicitly `{id, gpus?}` and the runner always launches `[sys.executable, "train_cell.py", --cell-id, --output-dir]` (`gpu_cell_runner.py:676`), inheriting `os.environ`, and reads a flat `metrics.json` from the output dir. We generalize the **cell job contract** so a cell can describe *how it launches* and *where its metrics come from*, with the current behavior as the default:

```jsonc
// cells.json entry (all new fields OPTIONAL; absent ⇒ byte-identical to today)
{
  "id": "search_qa_3b",
  "model_key": "Qwen2.5-3B-Instruct", "env": "search_qa", "baseline": "sdar",
  "gpus": 4,                               // existing; k GPUs pinned via CUDA_VISIBLE_DEVICES
  "command": "conda run -n sdar bash examples/sdar_trainer/run_search_3b.sh",  // NEW: raw launch; default = python train_cell.py
  "env": { "SDAR_STEPS": "150" },          // NEW: per-cell extra env (merged beneath staged-env)
  "metrics_source": {                       // NEW: how to synthesize the flat metrics.json when `command` doesn't write one
     "kind": "verl",
     "log_glob": "$OUTPUT_DIR/*.log",       // where verl's val/success_rate is emitted
     "success_rate_key": "val/success_rate"
  }
}
```

This one contract change subsumes the three seams: `command` = the **launcher seam**, `metrics_source` = the **verl adapter hook**, and the operator **staged-env** injects beneath `env`. A cell with none of the new fields runs exactly as today.

## 3. Design — Part 1: execution seams

Each item states interface · files · default-OFF guarantee · test.

### 3.1 Cell command/launcher seam (GAP 1 — the crux)
- **Interface.** If `cell["command"]` is present, the runner executes it via the shell in the cell's working dir (`code/`) instead of `[sys.executable, train_cell.py]`. The command receives the same injected env the harness sets today for a cell — `CUDA_VISIBLE_DEVICES` (the pinned set), `OPENRESEARCH_CELL_OUTPUT_DIR`/`--output-dir` equivalent exported as `OUTPUT_DIR`, `OPENRESEARCH_CELL_ID` — plus the merged env (§3.2). A raw command need not be `python`; `conda run -n sdar bash …` is the canonical case. Also fixes the documented TODO at `webshop_env.py:21-22`.
- **Files.** `backend/agents/rlm/gpu_cell_runner.py` (`_run_cell_subprocess`/`run_matrix`), `backend/agents/rlm/cell_matrix.py` (`normalize_cell_axes` passes the field through), `backend/agents/rlm/cell_scheduler.py` (shared with the Azure path).
- **Default-OFF.** No `command` ⇒ existing `[sys.executable, train_cell.py, …]` path unchanged.
- **Test.** `tests/rlm/test_cell_command_seam.py`: a cell with `command` runs it with the pinned `CUDA_VISIBLE_DEVICES` + `OUTPUT_DIR`; a cell without `command` is byte-identical to today (assert argv).

### 3.2 Staged-environment passthrough (GAP 3)
- **Interface.** An operator-declared allowlist forwards already-set orchestrator env vars into every cell's `child_env` **and** the monolithic `SandboxConfig.environment`, with **precedence over** the sandbox-contract prompt default and `asset_provisioning`'s `HF_HOME` set. Mechanism: `OPENRESEARCH_CELL_ENV_PASSTHROUGH="HF_HOME,HF_DATASETS_CACHE,ALFWORLD_DATA,WEBSHOP_URL,SEARCH_QA_INDEX_DIR,…"` (comma list of names to forward from the launching process). Precedence order into the child: harness-fixed (`CUDA_VISIBLE_DEVICES`, `OUTPUT_DIR`) > **passthrough allowlist** > `cell["env"]` > inherited `os.environ`.
- **Clobber guard.** When `HF_HOME` is in the allowlist, `asset_provisioning.py:340` and `_sandbox_contract.py:58` must NOT override it (guard both sites on "operator HF_HOME already declared").
- **Files.** `gpu_cell_runner.py` (child_env build, ~:631-657), `primitives.py` (`SandboxConfig.environment`, ~:3829-3839), `asset_provisioning.py:340`, `backend/agents/prompts/_sandbox_contract.py:58`.
- **Default-OFF.** Empty/unset allowlist ⇒ nothing forwarded ⇒ byte-identical.
- **Test.** `tests/rlm/test_cell_env_passthrough.py`: listed vars reach child_env with correct precedence; a pre-set `HF_HOME` survives asset_provisioning; unset allowlist forwards nothing.

### 3.3 verl → metrics adapter (GAP 5)
- **Interface.** New copyable helper `backend/agents/rlm/verl_metrics_adapter.py` (zero non-stdlib dep, auto-copied into `code/` via `_HARNESS_CODE_HELPERS`, mirroring `eval_provenance.py`). Public: `write_cell_metrics_from_verl(output_dir, *, model_key, env, baseline, log_glob, success_rate_key="val/success_rate", extra_keys=(...)) -> dict` — locates verl's val output (prefer a machine-readable val JSON if the trainer writes one; fallback to the last matching `success_rate_key` occurrence in `log_glob`), writes the flat per-cell `metrics.json` (`{success_rate, reward, loss, sdar_gate_mean, teacher_student_gap, status:"success"}`) **plus** an `eval_provenance.json` sidecar (value-preserving; records the source path + parsed line for auditability), and returns the dict. **Value-preserving:** it copies verl's measured numbers verbatim — never recomputes or scales (avoids the `reward×100` class, [[project_sdar_eval_provenance_fidelity]]).
- **Runner hook.** When a cell declares `metrics_source.kind=="verl"` and no flat `metrics.json` exists after `command` exits, the runner invokes the adapter as a post-step (so the parse never depends on the stubbing executor). This is a small `_finalize_cell_metrics` post-step in `gpu_cell_runner.py`.
- **Implementation task (must confirm against a real run dir):** pin the exact verl val-output location using `/mnt/sdar-cache/logs/run_search_3b.log` + the trainer's output dir before finalizing the parser; the log-regex path is the robust fallback, a val JSON is preferred if present.
- **Files.** new `verl_metrics_adapter.py`; `gpu_cell_runner.py` (post-step + `_HARNESS_CODE_HELPERS` copy list); reference it from `execute_repo_note` (`baseline_implementation.py:2816`).
- **Default-OFF.** No `metrics_source` ⇒ runner reads `metrics.json` as today.
- **Test.** `tests/rlm/test_verl_metrics_adapter.py`: parses a fixture verl log/JSON → correct flat metrics + provenance sidecar; value-preserving (no scaling); missing source ⇒ honest `status:"failed"`, never a fabricated number.

### 3.4 Local-repo reuse + commit pin (GAP 2 — required to use the staged cache)
- **Interface.** `OPENRESEARCH_REPO_LOCAL_PATH=/mnt/sdar-cache/SDAR` seeds `code/` from the pre-staged repo instead of a GitHub clone; `OPENRESEARCH_REPO_COMMIT=<sha>` pins the commit (records it in `repo_spec.json`). `RepoResolver` accepts a local-path source; `RepoProvisioner` copies (or `git clone <local>`) instead of wiping + cloning from github.
- **Files.** `config.py:94-98` (+2 fields), `backend/services/ingestion/repo/resolver.py` (accept local source), `provisioner.py:37-96` (copytree/commit-checkout), `run.py::_resolve_and_clone_repo`.
- **Default-OFF.** Unset ⇒ github-clone path unchanged.
- **Test.** `tests/services/ingestion/repo/test_local_repo_reuse.py`: local path seeds `code/`; commit recorded; unset ⇒ github path intact.

### 3.5 Execute mode owns deps (GAP 4)
- **Interface.** When `repo_spec.mode == "execute"`, skip the local `env_pin`/cu121-torch pip bootstrap (`primitives.py:3978-4007`) — the conda env owns the verl/vLLM stack; the harness must not restack it. Opt via `OPENRESEARCH_EXECUTE_OWNS_DEPS=1` (default ON when mode==execute).
- **Files.** `primitives.py:3978-4007` (gate), `env_pin.py` (no-op when execute-owns-deps).
- **Default-OFF.** Non-execute runs unchanged.
- **Test.** `tests/rlm/test_execute_owns_deps.py`: execute-mode run skips the pip bootstrap; adapt-mode still hardens.

### 3.6 Fail-loud on execute clone failure (GAP 8)
- **Interface.** In `execute` mode, a failed resolve/clone must NOT silently downgrade to `mode="scratch"` (a from-scratch reimplementation — the opposite of intent). Emit a `repo_execute_unavailable` `run_warning` and either hard-fail or fall back to `OPENRESEARCH_REPO_LOCAL_PATH` (§3.4) when set.
- **Files.** `run.py:628-637`.
- **Test.** `tests/rlm/test_execute_clone_failure.py`: execute + failed clone + no local path ⇒ loud failure (not scratch); + local path ⇒ falls back to it.

## 4. Design — Part 2: authors' verl config fixes

Grounded in the config that already succeeded on Search (audit §11). Applied to the on-disk proof scripts under `/mnt/sdar-cache/SDAR/examples/sdar_trainer/` (host-side, one-time), tracked in the run's `code/` via execute-mode seeding.

- **ALFWorld (`run_alfworld_3b_4gpu.sh`):** `ppo_micro_batch_size_per_gpu 32→16`, `rollout.gpu_memory_utilization 0.6→0.5`, add `actor.fsdp_config.optimizer_offload=True`. **Remove `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (vLLM `CuMemAllocator` asserts incompatible — a dead end, audit §11). If still tight, cap turn/history length (secondary).
- **WebShop (`run_webshop_3b_patched.sh`):** strip the same `expandable_segments`; apply the same memory-safe knobs; **fire the training launch** (it never has been).
- **Search-QA (`run_search_3b.sh`):** unchanged (proven at 0.456).
- **Validation:** each fix is a config edit, not code; verified live by the Search-3B smoke (unchanged) + the ALFWorld cell surviving past step 79.

## 5. Design — Part 3: run orchestration & validation

- **Repo/cache reuse.** `OPENRESEARCH_USE_AUTHOR_REPO=1`, `OPENRESEARCH_REPRODUCTION_MODE=execute`, `OPENRESEARCH_REPO_LOCAL_PATH=/mnt/sdar-cache/SDAR`, `OPENRESEARCH_REPO_COMMIT=9f2ce6a8…`. `OPENRESEARCH_CELL_ENV_PASSTHROUGH` forwards `HF_HOME=/mnt/sdar-cache/hf` + the env asset dirs.
- **Models (all-Foundry).** `--model foundry --models executor=foundry,verifier=foundry,grader=foundry`. Validator on a **different** deployment/family if funded (e.g. `validator=gpt-4o-azure`) to avoid the `degraded` separation; else accept the deterministic-veto-only validator (audit §7). Foundry deployment = the existing one (grok-4.3 / gpt-chat-latest per script).
- **GPU scheduling.** 4×A100-80GB. The authors' proof scripts are 4-GPU single-env. **Phase 1 (validation):** the Search-3B cell alone, `"gpus": 4`. **Phase 2 (grid):** 6 cells (3 env × 2 model). With 4-GPU proof scripts, cells run **sequentially** (one 4-GPU cell at a time) — the cells scheduler already serializes when the free-GPU pool is smaller than the requested set. (A future 2-GPU config could pair cells; out of scope here.)
- **Money & safety.** `--max-run-gpu-usd 400` + `OPENRESEARCH_MAX_RUN_GPU_USD=400`; explicit `--max-wall-clock` co-tightened to the budget; **autostop ON** (`OPENRESEARCH_SDAR_NO_AUTOSTOP` unset) so a crash/finish self-stops the VM (prevents another idle burn). Honesty guards via `--run-spec`: `OPENRESEARCH_ENV_LIVENESS_GATE`, `OPENRESEARCH_EVAL_PROVENANCE_GUARD`, `OPENRESEARCH_ZERO_METRICS_GUARD`, `OPENRESEARCH_EXTERNAL_VALIDATOR`.
- **Validation gate (Phase 1 → 2).** Proceed to the full grid only if the harness-driven Search-3B cell reproduces `val/success_rate` within a tolerance band of 0.456 (e.g. ≥0.40) AND the evidence guards pass. A miss means the seams/adapter are wrong — debug before spending the grid budget.

## 6. Testing strategy

- Per-seam hermetic tests (§3.x) — each asserts BOTH the ON behavior and OFF byte-identical parity.
- Full off-state regression: the existing cells-route / execute-mode / role-model suites must stay green with all new flags unset.
- No live GPU in CI. The Search-3B live validation is the operator gate, not a test.
- `ruff` clean; `uv sync --frozen` env.

## 7. Risks & mitigations

1. **verl val-output location assumption (§3.3).** Mitigation: confirm against a real run dir before finalizing; log-regex fallback; adapter fails honest (never fabricates).
2. **Foundry executor still stubs even the thin manifest.** Mitigation: once the harness owns launcher + adapter, the executor writes only a small `cells.json`; if it still stubs, fall back to a harness-templated `cells.json` for SDAR (operator-seeded) — the executor's authoring burden is then ~zero. (Escape hatch, not the plan.)
3. **Conda-in-subprocess env leakage** (CUDA paths, LD_LIBRARY_PATH). Mitigation: `conda run -n sdar` sets its own; the seam injects only the allowlisted staged env; validate on Search-3B.
4. **$400 cap trips mid-grid.** Acceptable — a partial grid is scored honestly (the tail handles missing cells); resume with a raised cap if desired.
5. **Execute-mode seeding + on-disk config edits drift.** Mitigation: `OPENRESEARCH_REPO_COMMIT` pins the repo; config edits live in the pinned tree.

## 8. Rollout / operator steps (after implementation + Opus review)

1. Land the seams (§3) + tests; `ruff`/off-state suites green.
2. Apply the verl config fixes (§4) on `/mnt/sdar-cache/SDAR` (host-side) or as an execute-mode overlay.
3. Restart `sdar-2model-a`; run **Phase 1** (Search-3B, all-Foundry, ~$30 slice of the $400) → confirm ≈0.456 + guards pass.
4. On pass, run **Phase 2** (full 6-cell grid) with autostop ON + `--max-run-gpu-usd 400`.
5. Pull `final_report` from GCS; update the audit doc with the outcome.

## 9. Open decisions for the reviewer (you)

- **Validator separation:** accept the `degraded` single-Foundry validator (deterministic veto still stands), or fund a second deployment/family for a real cross-check? (Default here: accept degraded; note it.)
- **verl config edits location:** patch on the staged disk (simplest, host-side) vs. an execute-mode overlay in `code/` (cleaner provenance)? (Default here: staged disk, pinned by commit.)
