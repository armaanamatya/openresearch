# Phase 2 — Preflight / Retry-Reduction (2026-06-07)

**Status:** 🟡 PROPOSED. **Mode-agnostic** pre-run preflight — independent of the BES work (decision D2). Catches cheap errors *off the GPU path* so runs stop burning hours on dep/schema/code-bug failures.
**Goal:** Reduce failed/retried GPU runs by extending the existing pre-run gate, not by adding BES.

> **Codex review (2026-06-07) — applied.** (1) Component A: the local bootstrap uses `|| true` (`primitives.py:2760-2781`), so a synthesized `requirements.txt` does **not** guarantee the import succeeds — A must **verify imports post-bootstrap** and return a `repairable` dependency failure; "matplotlib impossible" softened. (2) Component B: `CUDA_VISIBLE_DEVICES=""` is **not** isolation — importing agent-generated code is code execution and must run **inside the sandbox** (no secrets/network, read-only workspace, resource limits, hard timeout). (3) This phase also **owns the mode-agnostic RDR pre-run-gate wiring** (moved out of the BES deltas — flag `REPROLAB_RDR_PREFLIGHT_GATE`); the corrected seam detail is in Phase 3 §2.

---

## 1. Grounding — the pre-run gate already exists (extend it, don't invent it)

**Verified (Agent 1):** there is already a closed pre-run loop. `run_experiment` calls `validate_code_pre_flight` (`pre_flight_validator.py:1344-1358`) at `primitives.py:3840-3865` — **after** code-write, **before** sandbox dispatch. A hard violation short-circuits to `{"success": False, "pre_flight_blocked": True}`, which `failure_classifier` maps to `preflight_blocked` (`failure_classifier.py:187-188`), a **repairable** class → the agent re-implements. Good closed loop.

**`preflight_ast.scan_code_dir` (`preflight_ast.py:756-839`) checks today:**
1. Syntax (`_parse_with_syntax_check:719`).
2. Same-file missing-attribute access (`_check_missing_attr_access:286`).
3. Local `import-from` of a nonexistent symbol (`_check_local_import_from:476`).
4. SDAR env interface contract (`_check_env_interface_contract:622`, recursive `rglob`, self-scoped).

**Design contract (`preflight_ast.py:12-31`):** *narrow scope — prefer false negatives over false positives.* `scan_code_dir` is fail-soft (`:803`,`:836`): never raises. **`_check_undefined_names` is defined (`:377-473`) but intentionally NOT called** — the reason is quoted at `:808-818`: prohibitively high false-positive rate without full import-resolution tracking (`TODO(PR-γ-followup)`).

## 2. Component A — `requirements.txt` synthesis on the `local` sandbox (the matplotlib fix)

**The bug (verified CONFIRMED, Agent 1):** the failing run's `code/` had **no `requirements.txt` at all**; `train.py:89 import matplotlib` → `ModuleNotFoundError`. Root cause: the dependency bootstrap that synthesizes a requirements file (`ensure_requirements_txt`) lives **only inside the `"runpod"` block** (`primitives.py:2710-2726`); the `local` path gates on the file *already existing* (`primitives.py:2760` — `if "local" in _mode_str and requirements_path.exists()`). So on local, a missing file means nothing installs and nothing is synthesized. The synthesizer itself is fully capable (`requirements_derive.py:300-349`) — it's just never invoked on local.

**The change (PROPOSED, XS, low risk):** before the `requirements_path.exists()` gate at `primitives.py:2760`, call `ensure_requirements_txt(code_dir, dockerfile_path=…, base_image=env_id)` on the local path too, mirroring `:2710-2726`, AST-collecting top-level third-party imports. **⚠ Synthesis alone is not enough:** the local bootstrap uses `|| true` (`primitives.py:2760-2781`), so a failed install is silently ignored and the import can still die. So Component A must **verify the imports resolve after bootstrap** (an import-probe in the target env) and return a `repairable` dependency failure when they don't — *not* rely on the `|| true` install succeeding.

## 3. Component B — env/dataloader construction smoke (the largest bucket)

**Why:** `cell_execution_error` "likely code bugs" is the **largest** observed bucket (6×). Many are construction-time `TypeError`/bad-id/`AttributeError` that a 2-second CPU import-and-construct would surface for ~$0, before any GPU spins up. This generalizes the *static* `_check_env_interface_contract` into a *runtime construction* probe.

**The change (PROPOSED, L, gated):** a new helper invoked from `validate_code_pre_flight` (near the `scan_code_dir` bridge, `pre_flight_validator.py:1344`) that `import`s the agent's training module and **constructs** each `*Env`/dataset object, catching + reporting failures.
- **⚠ Isolation (Codex):** importing agent-generated code is *code execution* — `CUDA_VISIBLE_DEVICES=""` is **not** isolation (no filesystem, credential, process, or network containment). Run the construct probe **inside the existing sandbox** (Docker/local backend) with **no secrets, no network, a read-only workspace, resource limits, and a hard timeout**. Today's preflight (`pre_flight_validator.py:1344-1358`) is static AST only — this is the first preflight step that *executes* code, so it must be sandboxed accordingly.
- Gate behind `REPROLAB_PREFLIGHT_SMOKE=1`, time-boxed.
- FP risk: low-medium — construction may legitimately need GPU/network; report-not-block on ambiguous failures, per the `:12` contract.
- Pairs with the **Phase 0 Component C routing fix** so the surfaced code-bug becomes a floor-enforced repairable patch.

## 4. Component C — swallowed-backward-OOM AST check (verify the site first)

**The pattern:** `try: loss.backward()… except (RuntimeError|OutOfMemoryError|Exception): <no re-raise, no batch-shrink> continue` — the script catches the OOM, skips the step, exits 0. Today `silent_oom` is detected **postflight** by a log scan (`_training_health_violation:2209`, `_OOM_LOG_MARKERS:2036-2042`), i.e. *after* the GPU run.

**Verdict (verified PARTIAL, Agent 1):** statically detectable in principle — find `ast.Try` nodes whose body calls `.backward()`/`optimizer.step()` and whose handler catches `RuntimeError`/`OutOfMemoryError`/bare `Exception` and **neither re-raises nor mutates a batch/scale variable**. BUT:
- **Not the most frequent failure** (1 distinct run vs `cell_execution_error` 6× / `compute_scope_invalid` 4×) — so this is lower priority than A/B.
- **FP risk medium:** a legitimate *catch-OOM-shrink-and-retry* pattern looks similar. Keep it `soft`, scope tightly to handlers that skip-not-retry.

**The change (PROPOSED, M):** new `_check_swallowed_backward_oom(tree, path, out)` in `preflight_ast.py`, wired into the per-file loop at `:819-828`, reusing `_OOM_LOG_MARKERS` semantics for the except-type match. **⚠ Verify against the real swallow site first** (read the agent-emitted `train_cell.py`/`train.py` try/except) before shipping — the offending block was not extracted during recon.

## 5. Component D — re-enable `_check_undefined_names` (deferred, highest-risk)

Only after building the shallow import-resolution the `:808-818` comment asks for (resolve `import x`/`from x import y` alias tables before flagging a bare name). Uncomment the call at `:819-828`; extend `_collect_top_level_names` (`:193`) to track aliases. Ship **`soft`, never `hard`**, until the FP rate is measured. **Lowest priority** — the existing TODO deferred it for a reason.

## 6. Ranked order

| Rank | Component | Prevents (observed) | Effort | FP risk |
|---|---|---|---|---|
| 1 | A — `requirements.txt` on local | matplotlib `ModuleNotFoundError` (1×) | XS | very low |
| 1 | (Phase 0 C) `cell_execution_error` routing | code-bug cells (6×) → repair floor | XS | low |
| 2 | B — env/dataloader construct smoke | code-bug cells (6×), pre-GPU | L | low-med |
| 3 | C — swallowed-OOM AST | `silent_oom` (1 run) | M | medium |
| 4 | D — undefined-names | NameError subset of (6×) | M | high (deferred) |

## 7. Testing

- A: a run on `local` with a third-party import and no `requirements.txt` synthesizes one; assert the import resolves.
- B: a `*Env` that raises in `__init__` is caught by the smoke probe and reported as `preflight_blocked` (repairable) without GPU dispatch.
- C: AST fixtures — a swallow-and-skip block flags; a catch-shrink-retry block does NOT (the FP guard).
- All: assert `scan_code_dir` still never raises (fail-soft contract preserved).

## 8. Definition of done

- [ ] Component A lands: requirements synthesized **and imports verified post-bootstrap** on local; a dep failure returns `repairable` (no silent `|| true` pass-through).
- [ ] Component B behind `REPROLAB_PREFLIGHT_SMOKE`, time-boxed, report-not-block on ambiguity.
- [ ] Component C only after the real swallow site is read; ships `soft`.
- [ ] Component D remains deferred unless import-resolution groundwork lands.
- [ ] No new `hard` block raises the false-positive rate above today's (regression-measured on the run corpus).

## 9. Expected effect

Eliminates the matplotlib-class failure on local (A: synthesize requirements + **verify imports post-bootstrap**) and converts the largest failure bucket into pre-GPU repairables (B + Phase 0 C). No score change directly — this is **wall-clock and cost**: fewer hours-long GPU runs that die on a cheap error. Makes every Phase 1/3/4 iteration cheaper.
