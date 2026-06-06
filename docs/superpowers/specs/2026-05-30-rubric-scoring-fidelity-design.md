# RLM Rubric-Scoring Fidelity Remediation — Design

**Date:** 2026-05-30
**Status:** Design (grilled + locked); implementation pending
**Motivating run:** SDAR smallest-two (`prj_09047604e591d969`) scored area rolls
Method 0.44 / Data 0.13 / Execution 0.02 / **Eval 0.00 / Result 0.00** / Artifact 0.07
despite a faithful algorithm + both models trained.

## 1. Problem

A faithful reproduction scores near-0 on areas it actually satisfies. The score
does **not** reflect the work done. Three failure classes, established by a
file:line investigation of `leaf_scorer.py` + the `run_experiment` postflight:

1. **Measured-values blindness (the dominant cause — Eval/Result/Execution = ~half the weight).**
   Every guard checks **presence / shape / exit-code**, never **populated measured
   values**. A `train.py` that writes a placeholder `metrics.json
   {status:"running", final_accuracy:0.0, per_model:{m:{}}}` and exits 0:
   - `success = all(r.succeeded …)` is exit-code-only → `success=True` (`primitives.py:2920`).
   - The placeholder is **non-empty** → `degraded=False` (`primitives.py:4176`) → the
     0.35 honesty ceiling (`leaf_scorer.py:868`) never engages → **no floor**.
   - `verify_against_rubric` ignores the passed `results` and **re-reads
     `metrics.json` from disk** (`leaf_scorer.py:205`); the conservative LLM grader
     ("score 0.0 when no evidence", `leaf_scorer.py:486`) gives every eval/result/
     execution leaf 0.0.
   - The guard that *should* catch it, `_degenerate_training_violation`, **only
     inspects models with `status="ok"`** — an empty `{}` entry has `status=""` →
     skipped (`primitives.py:2084`).
   - 🔥 **The guidance instructs the placeholder**: `_EAGER_METRICS_BLOCK`
     (`baseline_implementation.py:1016`) tells the agent to write `status:"running"`
     "as you go" — the exact poison shape.

2. **Quick-scope penalization.** Data (paper Search-QA = 7 QA sets; quick run uses
   2) and Result-match (paper's full-training accuracy targets vs a 50–100-step
   run) score low for work intentionally not done. The `skip_set` mechanism that
   already excludes the 7B model (`leaf_scorer.py` `_detect_data_unavailable_leaves`
   → `roll_up(skip_set)`) does **not** extend to datasets or training budget.

3. **Wrong eval metric.** Even with measured numbers, the run emits a *training*
   token-F1 reward labelled "accuracy"; the Eval-protocol area wants the paper's
   eval metric (Search-QA **accuracy %** per model). `metrics_shape` is passed but
   not bound to the paper's keys nor required to be populated.

4. **Silent failures (cross-cutting).** All of the above were invisible: the
   placeholder, the empty `per_model`, and the per-leaf scoring reasons were never
   logged or surfaced. Debugging required reconstructing state from fragments.

## 2. Design (locked via grill)

### A. Metrics-completeness guard — 3 layers (fixes Eval/Result/Execution)
1. **Hard postflight** `_metrics_completeness_violation(result, scope)` wired in the
   success-gated band at `primitives.py:~3768` (beside `_training_health_violation`).
   Flips `success→False`, `failure_class="incomplete_metrics"` (repairable) when a
   `success=True` run is unmeasured:
   - top-level `status ∈ {running,in_progress,pending,started}`, **or**
   - every `per_model[m]` entry empty / carries no numeric leaf value, **or**
   - multi-model scope but **zero** per-model entries carry a measured metric.
   Reuses the value-walkers `_max_train_steps` / `_reward_curve` / `_scalar_rewards`
   (`primitives.py:1991-2056`). The repair message names the empty model(s) + the
   required keys → the loop re-runs to real numbers before it can score/finalize.
   Register `incomplete_metrics` in `FAILURE_CLASSES` + `_RUN_EXPERIMENT_REPAIRABLE_FAILURES`.
2. **Degraded floor (defense-in-depth)** — widen the `degraded` predicate at
   `primitives.py:4176` to treat placeholder/non-terminal-status metrics as
   `degraded=True`, so the 0.35 ceiling engages even if the postflight is bypassed
   near the wall-clock.
3. **Guidance fix** — `_EAGER_METRICS_BLOCK` (`baseline_implementation.py:1016`):
   incremental writes OK for liveness, but a `status:"running"`/empty-`per_model`
   `metrics.json` at run-end is a **FAILED** run; the final flush MUST set a terminal
   status and populate every per-model entry claimed.

### B. Dynamic scope scoring (fixes Data + Result-match for declared scope)
- **Dataset exclusion** — extend the `skip_set` mechanism to out-of-scope
  *datasets*: `ScopeSpec` already carries `datasets`; add `skip_datasets` (mirror
  `skip_models`) and teach `_detect_data_unavailable_leaves` to add leaves whose
  requirement names an unused QA set to `skip_set` (excluded from numerator AND
  denominator). Same anti-gaming guard as the 7B: the run must honestly not-attempt
  + declare it in `scope.gaps`.
- **Directional result-match** — for a run that declares a reduced training budget
  (`ScopeSpec.budget_per_model` / a `reduced_budget` flag), `_check_result_match`
  (`rubric_contract.py:194`) scores the paper's **directional** claim
  (SDAR ≥ GRPO/baselines at convergence — from the YAML, expressed as
  `relative_targets`) instead of absolute full-training numbers. A full run keeps
  absolute matching. The YAML (`docs/papers/2605.15155.yaml`) gains a
  `directional_claims` block (e.g. `searchqa_sdar_reward >= searchqa_grpo_reward`).

### C. Eval-metric schema enforcement (fixes Eval-protocol correctness)
- Derive the per-(model,env) **eval-metric keys** from the paper YAML
  `paper_targets` (e.g. `searchqa_<model>_accuracy`), pass them as the
  `metrics_shape` contract bound to the agent (the θ hook already threads
  `metrics_shape` into `run_with_sdk`), and have the **completeness guard (A1)**
  require those keys **populated with measured values** (extend
  `assert_metrics_schema` `rubric_guard.py:248/257` so a resolved path whose value
  is `{}`/None counts as *missing*, not present).
- **Guidance (how-to)** — SDAR file + general baseline block: after training, run a
  held-out Search-QA eval and emit **per-model accuracy** at exactly those keys,
  distinct from the training reward.

### D. Descriptive diagnostics (cross-cutting)
- **Structured events** — every guard/decision emits a `run_warning`/dashboard event
  with a stable `code` + a `WHAT / WHY / NEXT` message: `metrics_incomplete`,
  `scope_leaf_excluded`, `result_match_directional`, `rubric_degraded_floor`,
  `eval_metric_missing`. Visible in the UI + `dashboard_events.jsonl`.
- **`rubric_breakdown.json`** — the leaf scorer persists the full
  area→leaf→score→justification to `runs/<id>/rubric_breakdown.json`, and the
  `rubric_score` event carries a per-leaf summary (currently only area rolls are
  emitted). So *why* each leaf scored is durable + visible.
- **Per-run `debug.jsonl`** — a `_diag(ctx, stage, **fields)` helper writes a
  structured line (and `logger.info`) at each guard evaluation + the exact metrics
  state at scoring time. One grep reconstructs any failure without re-running.

## 3. Implementation order (each its own tested commit)
1. **A** metrics-completeness guard + failure class + degraded floor + guidance fix
   (highest value; unblocks Eval/Result/Execution). Tests: placeholder/empty/
   non-terminal → repairable; populated → ok; degraded predicate.
2. **D** diagnostics (lands with A so A's firings are visible). Tests: event codes,
   `rubric_breakdown.json` shape, `debug.jsonl` lines.
3. **C** eval-metric schema (paper-derived keys + populated-value enforcement +
   guidance). Tests: missing/zero accuracy → repairable; schema derivation from YAML.
4. **B** dynamic scope (skip_datasets + directional result-match). Tests:
   dataset-leaf exclusion from num+den; directional vs absolute by budget flag;
   anti-gaming (declared-but-attempted not excluded).

## 4. Risks / edge cases
- **Anti-gaming** is the central risk for B (and C's exclusions): a run could declare
  everything out-of-scope for a trivially high score. Mitigation: reuse the existing
  honesty guard — a leaf is only excluded when the run **did not attempt** it
  (no matching metric/artifact) AND declared it; an *attempted-but-failed* item is
  NOT excluded (it scores its real value). Mirror `_gap_in_load_failures`'s
  code-bug-laundering guard.
- **Completeness false-positives**: a legitimately CPU-only smoke (no GPU) emits
  reduced metrics. Mitigation: the guard requires *measured eval values for the
  declared in-scope (model,env)* — a smoke that declares reduced scope is judged
  against its own declared keys, not the paper's full set.
- **Directional result-match** must not let a *full* run off the hook — gate it on an
  explicit `reduced_budget`/`budget_per_model` declaration; default = absolute.
- **Backward-compat**: all new guards default-on but env-toggleable
  (`OPENRESEARCH_METRICS_COMPLETENESS_CHECK`, `OPENRESEARCH_DIRECTIONAL_RESULT_MATCH`), and
  `own`-style empty defaults preserve existing single-model behavior.

## 5. Files
`backend/agents/rlm/primitives.py` (guard, degraded, diag, wiring),
`backend/agents/rlm/failure_classifier.py` (incomplete_metrics),
`backend/agents/rlm/rubric_guard.py` (non-empty value enforcement),
`backend/agents/rlm/rubric_contract.py` (directional result-match),
`backend/evals/paperbench/leaf_scorer.py` (skip_datasets, rubric_breakdown.json),
`backend/agents/schemas.py` (ScopeSpec.skip_datasets / reduced_budget),
`backend/agents/baseline_implementation.py` (_EAGER_METRICS_BLOCK, eval how-to),
`docs/papers/2605.15155.yaml` (directional_claims, eval-metric keys),
`backend/agents/rlm/context.py` (diag/debug.jsonl path). Tests alongside each.
