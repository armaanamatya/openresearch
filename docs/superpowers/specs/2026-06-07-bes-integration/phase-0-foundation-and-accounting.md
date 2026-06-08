# Phase 0 — Foundation & Honest Accounting (2026-06-07)

**Status:** 🟡 PROPOSED. Three independent, near-free changes. No BES code. Highest ROI in the program.
**Goal:** Land the prerequisites and recover the rubric points the harness has *already earned but failed to count* — before spending effort on coverage (Phase 1) or BES (Phase 3).
**Targets issues:** breadth-unblock (merge), accounting-under-delivery (re-score), code-bug repair floor (routing).

> **Codex review (2026-06-07) — applied.** Two blockers on Component B (finalize re-score): (1) re-running `score_reproduction` does **fresh LLM grading** (`leaf_scorer.py:997-1038`), so `max(in_loop, finalized)` would score-select grader variance — instead **deterministically re-roll-up the already-persisted leaf scores**, never `max()` across policies. (2) It must **depend on Phase 4's env-guard** — `environments_skipped` is honored unconditionally and even `scope.gaps` alone is honored (`leaf_scorer.py:386-394`), so there is no safe "ship-before-the-guard" path. The **0.431** figure is the arithmetic with *both* env-sets excluded; under the guard only WebShop (principled) is excludable, so the credible finalized number is lower (recompute under the guard).

---

## 1. Component A — merge `feat/full-scope-envs`

**Why:** It is the single biggest honest-score lever and is **purely additive**. `git merge-tree c19b78c 5.30.26_sdar origin/feat/full-scope-envs` → **0 conflict hunks**. Since the merge-base `c19b78c`, `5.30.26_sdar` touched **only `best_runs/` and `docs/`** (zero backend code), so the env work lands clean.

**What it ships (env capability — NOT baselines, NOT learning):**

| New file | LOC | What it is | file:line |
|---|---|---|---|
| `backend/agents/rlm/search_qa_env.py` | 915 | Real multi-turn retrieval QA — `Retriever` with **dense E5 (`intfloat/e5-base-v2`) over a cached wiki-18 FAISS index** → BM25 → lexical fallback; `search()`/`answer()` grammar | `:151` |
| `backend/agents/rlm/alfworld_env.py` | 702 | Real ALFWorld TextWorld (headless); admissible-command nav; **sparse `float(won)` reward** | `:1-23`, reward `:458` |
| `backend/agents/rlm/webshop_env.py` | 629 | Real WebShop (server via `env_cache`); `search:`/`click:` grammar | — |
| `backend/services/runtime/env_cache.py` | 574 | `EnvCacheManager` — host-shared crash-safe **construct-once** cache (ALFWorld data, WebShop server, Search-QA index) | `:11`, `:249` |
| `backend/agents/rlm/exclusion.py` | 395 | Structured **verified-exclusion** contract — only harness-produced (`verified=True`) skips leave the rubric | — |
| `backend/agents/rlm/sdar_env_base.py` (+200) | — | `BaseEnv`/`AgenticEnv` ABC with abstract `build_student_prompt`/`build_teacher_prompt`; per-step `StepResult.reward` field | `:67`,`:83`,`:102-110` |

Plus `agentic_rollout.py` (centralized multi-turn rollout → flat `Trajectory`), `leaf_scorer.py` (+90, the verified-exclusion env gate `~:431`), and ~20 test files. 13 modified files, **none conflict**.

**⚠ Two caveats that make this necessary-but-not-sufficient (verified, Agent 3):**
1. It adds the three **environments + E5 retrieval**, but **not** the missing baselines (Skill-SD/RLSD/OPSD are agent-side `train.py`, see Phase 1).
2. ALFWorld is **real but not learnable** — sparse `float(won)` reward, zero warm-start/shaping. Bringing it in-scope without Phase 4 *lowers* the score (the §6 sequencing trap in the README). Land the capability now; gate its *activation* on Phase 4.

**Change:** standard merge of `origin/feat/full-scope-envs` into `5.30.26_sdar`. Run the full suite; the env files are import-guarded, so they're inert until cells reference them.

## 2. Component B — finalize-time re-score hook

**The bug (verified, Agent 2):** the 0.3556 was **frozen at iteration 6 (02:11:31)**, when `environments_skipped` was still `[]`. The agent declared `environments_skipped: ['alfworld','webshop']` at **02:41:34 (iter 7)** and **never re-called `verify_against_rubric`**. The exclusion code (`7f6db16`, an ancestor of HEAD) was present and correct — it simply had no signal to act on at scoring time. The late declaration sat on disk (`final_report.json::scope.gaps`, written 03:15) but **nothing re-scored** to honor it:
- `write_final_report_rlm` (`report.py:873-922`) **merges** the existing `rubric_evaluation.json` and only fills keys *not already present* (`:909-922`) — no call to `score_reproduction`.
- `_best_recorded_rubric_score` (`report.py:626-659`) is a scalar high-water mark over emitted events — it cannot apply newly-declared exclusions.

**The recompute (arithmetic, Agent 2):** excluding the **7** ALFWorld/WebShop env leaves (NOT the 8th, `d2c1a0a8`/SkillBank — that's genuinely-unimplemented in-scope work that must stay scored 0.0) plus the already-excluded 7B leaf (`6d1b8d3b`):

| Area | Weight | As-scored | Re-scored over surviving leaves |
|---|---|---|---|
| Method & code | 0.38 | 0.600 | 0.600 |
| Data & preprocessing | 0.14 | 0.090 | 0.180 |
| Experiment execution | 0.18 | 0.145 | 0.4143 |
| Evaluation protocol | 0.15 | 0.2846 | 0.2846 |
| Result match | 0.10 | 0.140 | 0.400 |
| Artifact completeness | 0.05 | 0.420 | 0.420 |

`0.600×0.38 + 0.180×0.14 + 0.41429×0.18 + 0.2846×0.15 + 0.400×0.10 + 0.420×0.05` = **0.4315**.
(The brainstorm's "0.448" excluded all 8 weak-leaves — it over-counts. **0.431 is the both-env-excluded arithmetic;** still short of 0.6. ⚠ Under the Phase 4 env-guard, ALFWorld's exclusion is *non-principled* (it crashed) and stays counted — so the **credible** finalized number is lower than 0.431, WebShop-only; recompute under the guard.)

**The change (PROPOSED) — deterministic, guard-dependent (Codex-corrected):** at finalize, re-run **only the eligibility + `roll_up`** over the **already-persisted leaf scores** — do **not** re-grade:
- Hook: `write_final_report_rlm` (`report.py:873`), before the merge at `:909`.
- Reuse the persisted `leaf_scores` from `rubric_evaluation.json`; re-run `_detect_data_unavailable_leaves` + `roll_up` (`leaf_scorer.py:358`,`:75-132`) under the **final** `scope` and the **Phase 4 env-guard**. Do **not** call `score_reproduction` (it does fresh LLM grading, `leaf_scorer.py:997-1038`); do **not** take `max(last_in_loop, finalized)` — that maximizes across *different exclusion policies* (and `report.py:838-855` would preserve a stale high-water score even after the guard rejects its exclusions). Replace the score outright with the deterministic re-roll-up and record which policy produced it.
- **Hard dependency: Phase 4 Component B (env-axis guard).** Without it the finalize re-score is an automated laundering path. **Do not ship this Component before Phase 4 Component B.**

## 3. Component C — `cell_execution_error` repair-routing fix

**The bug (verified, Agent 1 — symptom corrected from the brainstorm):** the cell-matrix code-bug branch returns `failure_class="cell_execution_error"` **with a populated `metrics` dict** (`primitives.py:3698-3703`, because `aggregate_cell_metrics` always returns non-empty — `cell_matrix.py:445`,`:474-487`). `_classify_run_experiment_outcome` (`primitives.py:219-237`) checks `metrics` truthiness **first** (`:224-226`) → returns `partial_evidence` and **never reaches the `failure_class` lookup** (`:228-237`). `cell_execution_error` is absent from all of `_RUN_EXPERIMENT_REPAIRABLE_FAILURES` (`:46-66`), the retryable/fatal sets, and `failure_classifier.FAILURE_CLASSES`.

**Consequence (corrected):** it does **not** cause restarts (`partial_evidence` keeps the root iterating in-place; not terminal — absent from `forced_iteration._TERMINAL_FAILURE_CLASSES:64`). The real cost: it **skips the repair-iteration floor** — `record_repair_attempt` fires only on `== "repairable"` (`run.py:661-663`), so the floor (`REPROLAB_MIN_REPAIR_ITERATIONS`, `forced_iteration.py:319-326`) never insists the agent fix a code-bug cell — and it gets no canonical `suggested_fix` (the hint is hand-rolled at `primitives.py:3702-3703`).

**The change (PROPOSED, XS):**
1. Add `cell_execution_error` to `_RUN_EXPERIMENT_REPAIRABLE_FAILURES` (`primitives.py:46`).
2. In `_classify_run_experiment_outcome` (`:224`), consult `failure_class` **before** the metrics-first branch *for known-repairable code-bug classes* — so a code-bug cell engages the repair floor instead of being silently typed `partial_evidence`.
3. Register `cell_execution_error` in `failure_classifier.FAILURE_CLASSES` with a canonical `suggested_fix`.

**Risk:** low — narrows an existing too-broad `partial_evidence` bucket. Must NOT reclassify a *genuinely partial* result (some cells ok, some bug) as fully repairable; gate on "all run-cells errored, zero ok" (the `status=="failed"` signal `aggregate_cell_metrics` already sets).

## 4. Testing

- **A:** full suite green post-merge; assert env modules import inert when no cell references them.
- **B:** golden test — synthetic run with `environments_skipped` declared *after* last in-loop verify; assert finalized `overall_score` applies the exclusion and equals `max(in_loop, recomputed)`; regression-pin the 0.431 recompute on the `prj_09047604e591d969` fixture.
- **C:** unit test — a `run_experiment` result with `failure_class=cell_execution_error` + all-cells-errored metrics classifies as `repairable` and fires `record_repair_attempt`; a mixed partial result still classifies `partial_evidence`.

## 5. Definition of done

- [ ] `full-scope-envs` merged; suite green; env activation deferred to Phase 4. **Rollback:** revert the merge commit (purely additive, no schema change).
- [ ] Finalize re-score is **deterministic** (re-roll-up of persisted scores; no re-grade; no `max()`), **gated on Phase 4's env-guard**, and records the policy used. 0.431 is the both-env-excluded arithmetic; the guarded figure (WebShop-only) is lower — recompute under the guard. **Rollback:** feature-flag the finalize re-score; off ⇒ today's behavior.
- [ ] `cell_execution_error` engages the repair floor; mixed-partial (`status != failed`) unaffected. **Rollback:** revert the set-membership + classifier ordering change.
- [ ] No change to flag-off behavior of any other run (all three are bugfixes/merges, not new modes).

## 6. Expected effect

Recovers ~**+0.075** rubric for free (re-score), unblocks the entire Data area for Phase 1 (merge), and converts the largest observed failure bucket (6× code-bug cells) into floor-enforced cheap patches (routing). **No GPU spend, no BES.**
