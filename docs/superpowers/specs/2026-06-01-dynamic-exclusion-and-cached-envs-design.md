# Dynamic Verified Rubric Exclusion + Cached Full-Scope Envs (2026-06-01)

**Status:** in progress. Split build — **Part A (Claude)** + **Part B (Codex)**, on
`feat/full-scope-envs`, against the shared `Exclusion` contract committed first.
Author: lolout1. No Claude/Codex co-author trailer.

## 1. Problem

The canonical SDAR baseline (`prj_09047604e591d969`) plateaus at rubric **~0.36 /
0.60**. Two of the three loss drivers are *fairness* bugs, not reproduction
failures — the run is being **docked for experiments it was never asked to run**:

1. **Operator scope is silently zeroed.** The run uses the cost-bounded
   "smallest-two" scope (`Qwen3-1.7B` + `Qwen2.5-3B`, **Search-QA only**).
   ALFWorld and WebShop are deliberately out of scope. But the cell route
   synthesises `metrics.json` from `cells.json` via
   `cell_matrix.aggregate_cell_metrics`, whose caller never declares those
   environments as skipped → `scope.environments_skipped: []`. The leaf scorer
   then finds the ALFWorld/WebShop leaves ungraded and scores them **0.0**,
   dragging the weighted overall down.

2. **No anti-gaming gate on the environment axis.** `leaf_scorer`
   `_detect_data_unavailable_leaves` already gates the *model* axis
   (`operator_skip_models`: an operator-de-scoped model is excluded; a
   *requested* model whose load failed in agent code is **not** — it stays scored
   as a code bug). The *environment* axis has **no such gate** — every
   `environments_skipped` entry is honoured unconditionally. That is both the
   reason (1) is unfixable today by just writing the skip (it would be a
   gaming hole) and a latent hole on its own.

The user's directive (the **fairness principle**): never dock the rubric for an
experiment blocked by an out-of-our-control limit (VRAM / OOM / dead dataset /
un-installable env) **or** deliberately de-scoped by the operator — **exclude**
it (drop from numerator *and* denominator), **dynamically**, **verified-only**
(anti-gaming). And, separately, make the full scope actually *runnable* so we
stop needing to exclude it (Part B).

## 2. What already exists (do not rebuild)

`backend/evals/paperbench/leaf_scorer.py`:
- `roll_up(node, leaf_scores, skip_set)` — already excludes skipped leaves from
  **both** numerator and denominator at every tree level.
- `_detect_data_unavailable_leaves(...)` — already reads `scope.environments_skipped`,
  `scope.models_skipped`, `model_load_failures`, and `scope.gaps` (structured
  dicts **and** prose) from the newest `metrics.json` + `final_report.json`, and
  matches leaves by **requirement-text token-superset** (not hash id).
- `operator_skip_models` — the existing anti-gaming gate for the model axis.
- `score_reproduction(...)` — emits `overall_score`, `coverage_pct`,
  `eligible_count`, `unavailable_count`.

`backend/agents/rlm/cell_matrix.py`:
- `capacity_gate` (→ `capacity` gaps + `models_skipped`), `dataset_url_preflight`
  (→ `dataset_unavailable` gaps + `environments_skipped`), and
  `aggregate_cell_metrics(..., models_skipped=, environments_skipped=,
  capacity_gaps=, dataset_gaps=)` which **already** emits them when passed.

So Part A is **wiring + one new anti-gaming gate**, not a new scorer.

## 3. Shared contract — `backend/agents/rlm/exclusion.py` (DONE)

Stdlib-only, copyable into the agent sandbox (like `gpu_cell_runner`/`cell_matrix`).

```
Exclusion(item, axis, kind, reason, verified, evidence)
  axis ∈ {environment, model, dataset, baseline}
  kind ∈ {capacity_vram, dataset_dead, oom_shrink_exhausted, env_setup_failed,  # HARD_LIMIT_KINDS
          operator_scope}
  verified: True  ⇒ harness-produced (measured limit OR operator ScopeSpec)  → EXCLUDED from strict score
            False ⇒ agent-declared, un-corroborated                         → STAYS in scoring
```
Helpers: `operator_scope_exclusions(full, active, axis)`, `verified_only`,
`verified_items_by_axis`, `build_scope_block(exclusions, models_run, existing)`
(emits structured `exclusions` + derives legacy `environments_skipped` /
`models_skipped` / `gaps` from **verified** records only), `Exclusion.from_gap` /
`exclusions_from_gaps` (round-trip legacy on-disk gaps).

**Invariant:** only `verified=True` exclusions ever enter a skip list or the
strict `skip_set`. Unverified ones are recorded in `scope.exclusions` for
transparency but never reduce the denominator.

## 4. Part A — dynamic verified exclusion (owner: Claude)

**A1. Mint exclusions at the cell route.** Where `aggregate_cell_metrics` is
called (`run_experiment` / its cell-route helper), build the exclusion set:
- `operator_scope` for the paper's full axis minus the active `ScopeSpec` axis
  (full SDAR envs `{ALFWorld, WebShop, Search-QA}` − active `{Search-QA}` →
  exclude ALFWorld + WebShop), `verified=True`, evidence = scope-spec path.
  Source of "full scope": the rubric/paper-hint declared environments (fall back
  to a paper-scope constant for SDAR); active scope from `ScopeSpec`.
- `capacity_vram` / `dataset_dead` from the existing gate gaps
  (`Exclusion.from_gap`).
- `oom_shrink_exhausted` from cells whose record is OOM-after-shrink-exhaustion.
- `env_setup_failed` from Part B's env-setup fallback (consumed when present).
Persist via `build_scope_block(...)` so `metrics.json::scope` carries
`exclusions` + the derived legacy lists.

**A2. Close the environment anti-gaming hole in `leaf_scorer`.** Add
`operator_skip_environments` (and, preferably, consume `scope.exclusions`
directly): an `environments_skipped` / env-axis exclusion is honoured **only**
when it is operator-intended or otherwise `verified`. A *requested* environment
that failed in agent code (not verified) is treated like a requested-but-failed
model — **stays scored**. Mirror the existing `operator_skip_models` logic
exactly; keep requirement-**text** token matching.

**A3. Two numbers.** `score_reproduction` already yields the strict
`overall_score` (verified exclusions removed). Also compute/return
`compute_adjusted_score` = strict score with operator-scope exclusions folded in
too (answers "what would we score on the scope we attempted?"), and
`compute_scope` describing what was excluded. Thread both into
`amend_final_report` (the rubric block already has slots for them).

**A4. Tests.** Unit-test `exclusion.py` (validation, operator_scope diff,
build_scope_block verified-only derivation, gap round-trip); cell-route minting;
the env anti-gaming gate (operator-de-scoped env excluded; agent-failed env
stays scored); the strict-vs-adjusted numbers on a fixture rubric. Use the real
on-disk SDAR `metrics.json` sample where practical.

## 5. Part B — cached, scalable ALFWorld + WebShop (owner: Codex)

**B1. `backend/services/runtime/env_cache.py` — `EnvCacheManager`.** Idempotent,
file-locked, crash-safe (mirror `local_gpu_allocator`'s `fcntl` discipline):
- ALFWorld: run `alfworld-download` **once** into a shared cache dir
  (`REPROLAB_ENV_CACHE_DIR`, default under `runs/.cache/envs`); subsequent runs
  reuse it. Expose the data path to cells via env var.
- WebShop: start **one** shared WebShop server (indexed product data) per host,
  ref-counted; hand cells its base URL; tear down when the last lease drops.
- Public API returns a typed result so a failure becomes a structured
  `env_setup_failed` **`Exclusion`** (verified=True), never an exception that
  zeroes the grid.

**B2. `scripts/batch_reproduce.py` wiring.** Acquire ALFWorld download + WebShop
server through `EnvCacheManager` before launching cells; pass cache locations
into the per-run env; release on exit (SIGINT-safe, like the GPU leases).

**B3. SDAR guidance.** Extend `REPROLAB_BASELINE_EXTRA_GUIDANCE` (and/or the SDAR
guidance constant) so the agent writes `ALFWorldEnv` / `WebShopEnv` subclassing
`sdar_env_base.BaseEnv` (implementing `build_student_prompt` /
`build_teacher_prompt`) **and** adds their cells to `cells.json` (model × env ×
baseline × seed). The `preflight_ast._check_env_interface_contract` already
rejects a non-subclassing `*Env` before the grid runs.

**B4. Fallback.** If an env cannot be set up on this host, emit an
`env_setup_failed` `Exclusion` (verified=True) and continue — the grid runs the
envs that work; the rubric excludes (not zeroes) the rest. Same `Exclusion`
contract as Part A; `build_scope_block` merges them.

**B5. Tests.** Unit-test `EnvCacheManager` (idempotent download, ref-counted
server lifecycle, lock/crash-safety, failure→Exclusion). Real ALFWorld/WebShop
integration is **not** end-to-end verifiable in this session (no spare GPU; live
run holds the box) — unit-test the manager and document the integration as
untested-pending-a-clean-host.

## 6. Anti-gaming invariants (both halves uphold)

1. A skip enters the denominator-reducing set **iff** `verified=True`.
2. `operator_scope` is verified **only** against the operator's `ScopeSpec`, not
   the agent's `train.py` declaration.
3. A *requested* axis value that fails in agent code is a repairable bug, **not**
   an exclusion (logged, stays scored) — preserve the existing model-axis
   behaviour and extend it to environments.
4. Leaf matching is by requirement **text** (token-superset), never hash id, and
   never excludes an in-scope SDAR leaf.

## 7. File map

| File | Part | Change |
|---|---|---|
| `backend/agents/rlm/exclusion.py` | shared | NEW — contract (DONE) |
| `backend/agents/rlm/<cell-route caller>` | A | mint + persist exclusions via `build_scope_block` |
| `backend/evals/paperbench/leaf_scorer.py` | A | `operator_skip_environments` gate + consume `scope.exclusions`; compute_adjusted_score |
| `backend/agents/rlm/report.py` / final_report | A | surface `exclusions` + `compute_scope` |
| `tests/...` | A | exclusion + scorer-gate + numbers |
| `backend/services/runtime/env_cache.py` | B | NEW — `EnvCacheManager` |
| `scripts/batch_reproduce.py` | B | acquire/release env cache |
| SDAR guidance constant | B | ALFWorld/WebShop env + cells guidance |
| `tests/services/runtime/test_env_cache.py` | B | manager lifecycle |

## 8. Acceptance

- `exclusion.py` + Part A scorer wiring: full unit coverage, 0 regressions in the
  existing leaf-scorer/cell-matrix suites.
- Re-scoring the live run's rubric with ALFWorld/WebShop as verified
  `operator_scope` exclusions yields the **fairness-adjusted** overall (the
  user's "score without the 2 disabled experiments") — reported with the delta.
- Part B `EnvCacheManager` unit-green; integration documented as pending.
- Merge Codex's branch into `feat/full-scope-envs` (disjoint files), push.

## 9. Part A — landed (2026-06-01)

- `ScopeSpec.skip_datasets` added (mirrors `skip_models`); `merge_with_paper_default`
  auto-derives it from a narrowed `datasets` list (operator narrowing IS the
  decision) and reconciles it out of the effective datasets.
- `primitives._operator_scope_exclusions` / `_apply_operator_scope` mint verified
  `operator_scope` Exclusions from `skip_models`/`skip_datasets`, recover the
  capacity/dataset gate gaps as structured exclusions, and fold all of them into
  `metrics.scope` via `exclusion.build_scope_block` at both cell-route aggregate
  sites.
- `leaf_scorer._detect_data_unavailable_leaves` consumes `scope.exclusions`
  (verified-only) and gates `environments_skipped` by an `operator_skip_environments`
  set — closing the env anti-gaming hole. Verified env/model exclusions are
  authoritative; an agent-declared (unverified) skip stays scored.
  `score_reproduction` threads `operator_skip_environments` (from
  `scope_spec.skip_datasets`).

**A3 decision:** because `operator_scope` is `verified=True` (the `--scope-spec`
is the evidence), de-scoped environments are excluded from the *official*
`overall_score` directly — identical to how operator-skipped models already are.
No separate `compute_adjusted_score` is introduced; the strict score is the fair
score. The pre-existing β3 `compute_adjusted_score` (floor-anchored result-match)
is a distinct mechanism and is left untouched.

**Result on the live run (`prj_09047604e591d969`):** re-scoring its actual rubric
through the real `_detect_data_unavailable_leaves` + `roll_up` with ALFWorld/WebShop
as verified `operator_scope` exclusions: **0.356 → 0.413** (9 of 26 leaves
excluded). Tests: 199 green (174 existing + 25 new), 0 regressions. The live run
(launched on pre-fix code) won't pick this up mid-flight; future runs + a re-score do.
