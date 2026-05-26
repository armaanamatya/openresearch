# Systemic Failure Resolution — Comprehensive Design

**Status:** proposed (2026-05-26)
**Scope:** every failure mode observed across Adam (1412.6980), Dropout (1207.0580), and VAE (1312.6114) runs over the past two days, grouped by root cause, with a single canonical fix per category.
**Author:** Opus (claude-opus-4-7), reviewed by Codex.

---

## All observed failures, in one table

| # | Run | Date | Failure surface | Root cause |
|---|---|---|---|---|
| F1 | Adam `prj_6d41d2f09c026403` | 2026-05-24 | `RUNPOD_BALANCE_TOO_LOW: 500` on all 4 `run_experiment` calls → rubric 0.0 | RunPod credit ran out mid-day |
| F2 | Dropout `prj_9afa700e444c1df7` | 2026-05-24 | same | same |
| F3 | VAE `prj_9554a1396eb993aa` | 2026-05-25 | container started with `DeviceRequests: null` → CPU only | `SandboxConfig` dropped `gpu_mode` ❶ |
| F4 | VAE `prj_db45c0304ce455a6` | 2026-05-25 | same (caught earlier, killed) | same ❶ |
| F5 | VAE `prj_3080fe2a02c20164` | 2026-05-25 | `detect_environment` ValidationError: `hardware_clues` non-list | LLM passed bare string ❷ |
| F6 | VAE same | 2026-05-26 | Phase 3 importance-sampling crash: `WakeSleepVAE` has no `reparameterize` | Agent code-writing bug ❸ |
| F7 | Adam `prj_d02bd4fdf24aedf2` iter 1 | 2026-05-25 | `Connection closed` SSH on artifact-sync end + 16 `rubric_guard` contract violations | RunPod SSH flake ❹ + nested-vs-flat key mismatch ❺ |
| F8 | Adam same iter 2 | 2026-05-26 | (in flight) | TBD |
| F9 | All RunPod runs | ongoing | wall-clock estimate 3-5× under reality | Single-LLM workload extractor, no empirical anchor ❻ |

❶ = Configuration-threading bug — FIXED today (commit `7294ee1`)
❷ = Schema rigidity vs LLM creativity — FIXED today (commit `7294ee1`)
❸ = Agent code-writing bug — UNFIXED, dominant failure class going forward
❹ = RunPod infra transient — UNFIXED
❺ = Schema rigidity (output side) — FIXED today (commit `befb51c`)
❻ = Estimator architecture — designed, not yet shipped

---

## Failure taxonomy → six root-cause categories

After deduplication, every observed failure falls into one of these six categories:

### Category I — Infrastructure transients (RunPod-specific)
**Examples:** F1, F2, F7-SSH
**Mechanism:** RunPod's SSH transport drops mid-command; pod gets recycled; credit check fires; 500s from REST. Today these are fatal to `run_experiment` even when the underlying training succeeded.
**Cost imposed:** $0.58/iter on retry, ~50 min wall clock per iteration, sometimes total rubric=0.0.

### Category II — Configuration-threading bugs
**Examples:** F3, F4 (gpu_mode), and historical (wall-clock-ceiling, etc.)
**Mechanism:** A value set on the outer config (CLI flag, env var) isn't forwarded all the way to the runtime layer. Silent — code runs but in the wrong mode.
**Cost imposed:** entire run wasted in the wrong mode.

### Category III — Schema rigidity vs LLM creativity (input side)
**Examples:** F5 (hardware_clues), historical (claims, datasets, metrics, training_recipe, model_architecture, evaluation_protocol, core_contribution, ambiguities)
**Mechanism:** The LLM root passes `method_spec["hardware_clues"]` as a string when the schema expects `list[str]`. Pydantic rejects → primitive errors → root burns one iteration recovering.
**Cost imposed:** ~1 min + tokens per occurrence; bug surface infinite until coerced.

### Category IV — Schema rigidity vs agent creativity (output side)
**Examples:** F7-keys (nested vs flat), historical (paper_claims list-vs-dict)
**Mechanism:** The agent's `train.py` writes `metrics.json` with a nested shape; `rubric_guard` expects flat keys derived from rubric leaves. 16 contract violations → `run_experiment` returns FAIL.
**Cost imposed:** entire iteration burned, full re-implementation needed.

### Category V — Agent code-writing bugs (the hard ones)
**Examples:** F6 (WakeSleepVAE.reparameterize missing), historical (torchvision missing, num_workers>0 on CPU, wrong API shape)
**Mechanism:** Sonnet writes `train.py` that calls `model.reparameterize()` on a class that doesn't define it. Crashes mid-execution, often after hours of valid work.
**Cost imposed:** all subsequent work in that script lost. The dominant runtime-crash class.

### Category VI — Estimation / observability
**Examples:** F9 (time estimator off), Codex diagnostic findings from today
**Mechanism:** Single LLM opinion, no empirical anchor, no shape modeling, no retry-rate modeling, no hardware-throughput modeling.
**Cost imposed:** wrong-by-5× ETA, makes batch planning impossible.

---

## Six canonical solutions, one per category

### Solution I — Transient-error retry policy + sandbox fallback

A single `transient_error_classifier(exception)` function returns one of:
- `fatal` (RUNPOD_BALANCE_TOO_LOW, RUNPOD_AUTH_FAILED) — user must act
- `transient` (Connection closed, 500 Internal Server Error, NO_CAPACITY_AVAILABLE) — retry with exponential backoff, up to 3×
- `code_bug` (AttributeError, NameError from agent code) — feed to repair loop, NOT retry

Wrap `_execute_in_sandbox`'s create+exec+destroy lifecycle in retry logic that consults this classifier. Emit `sandbox_retry` SSE events. After max retries on RunPod transient, automatically fall back to `local docker` if the host supports it; emit `sandbox_fallback` event.

**File:** `backend/services/runtime/transient_classifier.py` (new) + `backend/services/runtime/runpod_backend.py` (wraps create/exec/destroy)
**Tests:** synthetic exceptions, fallback path, max-retry exit
**Ship cost:** ~150 lines + 50 lines tests

### Solution II — Configuration audit + lockdown

Audit every config knob (gpu_mode, sandbox_mode, execution_mode, max_wall_clock, max_pod_seconds, max_usd) for end-to-end threading from `StartRunRequest` → CLI args → subprocess env → `RunContext` → primitives → sandbox runtime. For each, add a one-line "threading test" that asserts the value reaches `SandboxConfig.create_sandbox`'s observable side effect.

**File:** `tests/integration/test_config_threading.py` (new)
**Ship cost:** ~60 lines (one test per config knob × ~6 knobs)

### Solution III — Single canonical input-coercion module

We've shipped 8 `@field_validator(mode="before")` coercers on `PaperClaimMap` to fix this pattern, plus similar ones on related schemas. Consolidate them into `backend/agents/llm_coercion.py` with three reusable helpers:
- `coerce_to_str_list(v)` — bare str / tuple / None → `list[str]`
- `coerce_to_string(v)` — dict / list / None → `str`
- `coerce_list_of_records(v, key="name")` — bare str items → `{key: item}` dicts

Refactor existing validators to call these helpers. Future LLM-facing list/string fields use the canonical helper. Document the pattern in the module docstring as the official answer to schema rigidity vs LLM creativity.

**File:** `backend/agents/llm_coercion.py` (new) + refactor `backend/agents/schemas.py`
**Tests:** consolidate existing schema tests
**Ship cost:** ~100 lines new module, refactor of ~150 existing lines

### Solution IV — Output-side fingerprint matching + agent-prompt structure declaration

Two parts:
1. The `rubric_guard` fingerprint matcher SHIPPED today (commit `befb51c`) already accepts nested-or-flat shapes via token-subsequence matching.
2. NEW: the planning agent's prompt now requests an explicit `metrics_shape` declaration in `ReproductionContract` — an array of paths the agent will emit (e.g., `["per_model.mnist_logistic.per_dataset.mnist.adam_final_nll", ...]`). `implement_baseline`'s Sonnet sub-agent receives this declaration and is bound to emit ONLY those paths. RubricGuard validates against the declared shape, not against the rubric's expected flat keys.

This is the **clean fix** for Category IV — make the agent declare its own contract instead of guessing the rubric's.

**File:** `backend/agents/schemas.py` (new `metrics_shape` field on `ReproductionContract`), `backend/agents/rlm/primitives.py` (prompt update for plan_reproduction + verify_against_rubric)
**Tests:** stub LLM emits shape, downstream binding tests
**Ship cost:** ~200 lines

### Solution V — Agent-code AST pre-flight (the BIG one)

Before sandbox `exec` of `train.py`, run a fast static-analysis pass that catches:
- **Missing attribute access**: `model.reparameterize()` called on a class that doesn't define it (today's VAE bug F6)
- **Undefined function call**: calling a name not in scope
- **Import-from of nonexistent module**: `from torchvision import datasets` when torchvision isn't in requirements
- **Mismatched signature**: function called with wrong arg count

Implementation via Python's built-in `ast` module + a lightweight class/method resolver. NO LLM call — pure AST + symbol-table walk. Surfaces violations as a structured `pre_flight_violation` list; emits as `run_warning` events; presents to the agent's repair loop with file:line context so the next `implement_baseline` iteration knows the exact fix.

Existing task #77 ("Pre-flight: tensor device-mismatch AST scan") is the spiritual sibling — this is its bigger cousin.

**File:** `backend/agents/rlm/preflight_ast.py` (new) + hook in `_execute_in_sandbox`
**Tests:** 8 violation classes × happy/sad fixtures
**Ship cost:** ~300 lines + ~150 lines tests

### Solution VI — Three-source ensemble estimator + multi-stage workload + iteration modeling

The spec `docs/superpowers/specs/2026-05-25-three-source-budget-estimator-design.md` plus Codex's diagnostic refinements from today:
- **Capture wall-clock per preserved run** to `timing.json` (extract from `dashboard_events.jsonl`)
- **k-NN over preserved wall-clocks** keyed on paper feature vector
- **Multi-stage workload schema**: `{main_experiments, lr_probes, baselines, sweeps}` instead of single `epochs × experiment_count`
- **Hardware throughput catalog** (`seconds_per_epoch_per_M_params` per SKU; add RTX 2060 row)
- **Iteration-count modeling**: `expected_iterations = base + p(rubric_guard_fail) × retry_overhead`
- **Kill the hardcoded 0.15 compression** in favor of agent-declared `compute_scope.declared_reductions`

**Files:** consolidate the existing 3-source-ensemble + compute-adjusted-rubric specs into one PR.
**Ship cost:** ~600 lines + ~300 lines tests (largest of the six solutions)

---

### Solution VII — Orchestrator (`run.py` / RLM loop) hardening

The orchestrator's failure-response policy today is *entirely* delegated to the root LLM: any non-success on `run_experiment` puts the failure dict into the next REPL turn, and the root model decides whether to retry, re-implement, or propose improvements. This works but is wasteful:

- **No failure-class routing**. SSH-close and "agent wrote a buggy class" both come back as "run_experiment failed" with a stringified error. The root has to read the error and guess the right next action. Often it re-implements when retry would have been sufficient (cost: ~50 min wasted on Adam iter 1). Or it propose_improvements when re-implementation was needed (cost: a quality regression iteration).

- **No iteration budget**. The user can't say "spend max 3 iterations total". Today's max_invocations is per-agent, not per-run-experiment-cycle. RLM can loop indefinitely on a recoverable failure.

- **Partial-work caching missing**. When Sonnet's `train.py` has Lines 1-800 correct and Line 924 is broken, the next `implement_baseline` iteration re-writes the entire file. A patch-mode (input the broken file + the error → emit minimal diff) would save 95% of the LLM tokens and most of the agent's wall-time per recovery iteration.

- **Iteration cost is invisible to user**. The SSE stream has cost events but no rolling cost-per-iteration so the user can spot a runaway loop early. By the time they notice, the loop has burned $5-10.

- **Watchdog wedge detection is overly conservative on long valid runs**. Today's VAE went 3.5h between `iteration_heartbeat` events because `run_experiment` legitimately takes that long. The watchdog's "no update for N minutes" rule would have killed a healthy run if N were tighter.

**Concrete changes:**

1. **Failure-class router** in `run.py`'s REPL turn handler. Inspects the `failure_class` field on the result dict (set by transient_classifier from Solution I and by preflight_ast from Solution V). Maps to a recommended next action — `retry_same` for transients, `repair_diff` for code-bugs, `re_implement` for shape/contract drift, `propose_improvements` for "code is correct but metric below target". Surface as a `recommended_next_action` field the root model can choose to follow or override.

2. **Per-run iteration budget**: `--max-rlm-iterations N` CLI flag (default 5, configurable). Soft warning at N-1, hard stop at N. `iteration_budget_exceeded` SSE event with cost summary.

3. **Patch-mode `implement_baseline`**: when failure_class is `code_bug` and the prior train.py is on disk, pass the current file + the error trace to Sonnet with a prompt like "emit a minimal diff to fix this specific exception, do not rewrite the file". Falls back to full rewrite if patch can't be applied cleanly. Probably 80% of recovery iterations qualify.

4. **Rolling cost surfacing** in `demo_status.json::cost_summary` updated every 30s with `usd_this_iter`, `usd_total`, `iter_count`. UI can render a cost trend chart and warn when iter cost exceeds a threshold.

5. **Watchdog gradient**: instead of "N minutes since last event = wedge", use `min(N_baseline, 2 × p95_observed_idle_for_this_primitive)`. Run-time-aware idle threshold so a legitimate 3h `run_experiment` doesn't get falsely killed.

**Files:** `backend/agents/rlm/run.py`, `backend/agents/rlm/primitives.py` (implement_baseline patch-mode), `backend/agents/rlm/run_watchdog.py`
**Tests:** failure-class routing (4 classes × happy/sad), iteration budget (3 scenarios), patch-mode fall-through, watchdog gradient
**Ship cost:** ~400 lines + ~200 lines tests

---

## Partial-success acceptance (cross-cutting)

VAE today is a perfect motivator: 95% of the science completed, crashed on the last 1%. The orchestrator's current binary `success: bool` treats this as "broken". A `success: "ok" | "partial" | "failed"` enum with explicit partial-acceptance rules:
- `metrics_dict` non-empty + at least one named primary metric → `"partial"` not `"failed"`
- `degraded=False` on partial (don't cap leaves at 0.35)
- Grader gets full evidence; rubric scores against captured numbers; missing leaves get the conservative 0.0
- UI badge: "partial — N metrics captured before crash"

**File:** `backend/agents/rlm/run.py`, `backend/agents/rlm/report.py`, `backend/evals/paperbench/leaf_scorer.py` (DEGRADED_LEAF_CEILING gates on success state)
**Tests:** partial-success path, full-success path, full-failure path
**Ship cost:** ~100 lines

---

## Phasing — three PRs

### PR-1 (ship tonight, ~4h Sonnet work)
**Theme: stop bleeding compute on transient errors, broken agent code, and orchestrator-level overreaction**
- Solution I (transient retry + fallback)
- Solution V (AST pre-flight)
- Solution VII (orchestrator hardening — failure-class router + iteration budget + cost surfacing)
- Partial-success acceptance

These four together would have prevented the rubric=0.0 from F1+F2 (transient retry), caught F6's bug in seconds (AST), and saved Adam iter 1's wasted 50 minutes (transient classifier → retry_same instead of re_implement). Iteration-budget cap means a runaway loop has a hard ceiling.

### PR-2 (ship next day, ~2h Sonnet work)
**Theme: schema robustness consolidation**
- Solution II (config threading tests)
- Solution III (canonical coercion module)
- Solution IV (metrics_shape declaration)

Removes future occurrences of Categories II, III, IV.

### PR-3 (ship overnight, ~4h Sonnet work)
**Theme: estimation precision**
- Solution VI (3-source ensemble + multi-stage workload + iteration modeling)

The big-shape estimator refactor. Depends on Solution VII's iteration modeling being shipped first (PR-1).

---

## What this design does NOT change

- The RLM root model architecture (we're not rewriting the orchestrator)
- The rubric leaf scorer's LLM prompt (it's working — see Codex's analysis from earlier; the issue was input not the grader)
- The choice of Sonnet for sub-agent code writing (model choice is good; the AST pre-flight catches its consistent bug class)
- The PaperBench rubric tree format

---

## Critique invitations (for Codex)

**Q1:** Is the six-category taxonomy exhaustive, or are there observed failures from the past two days I missed? Are any of my categorizations wrong?

**Q2:** Solution V (AST pre-flight) is the biggest piece — is it actually feasible without an LLM call? Pure AST + symbol table might miss dynamic patterns (monkey-patching, `setattr`, etc.) that legitimately add `reparameterize` at runtime. What's the false-positive rate likely to be on existing reproductions?

**Q3:** Partial-success acceptance shifts the "did this run succeed?" decision from binary to ternary. Does that interact badly with anything in the leaderboard projection, the demo_status state machine, or the SSE event allowlist?

**Q4:** Is PR-1's "stop bleeding compute" focus the right priority? Or should PR-2 (schema lockdown) come first because schema bugs are cheaper to fix and provide a stable foundation for the bigger AST work?

**Q5:** Is there a 7th category I missed? Specifically, anything in the LLM-as-judge layer (verify_against_rubric, rubric_gen) that's contributed to scoring noise?
