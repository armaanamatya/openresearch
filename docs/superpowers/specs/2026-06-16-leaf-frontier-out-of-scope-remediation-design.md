# Leaf-Frontier & Out-of-Scope Remediation — Design Spec

**Date:** 2026-06-16 · **Branch:** `feat/grader-fidelity` (shipped tier) → continues on the same branch · **Status:** L1–L3 SHIPPED (commit `45c3d32`); L4–L6 designed, flag-gated, not built.

**Sibling specs (this doc does NOT re-design any of them):**
- `2026-06-16-grader-fidelity-and-harness-remediation-design.md` — the LOCKED grader-fidelity workstreams **A1–A7 / B / C / D / E** (median-of-N, `deterministic_leaf_checker`, evidence-fingerprint, champion-artifact, decoupled transport, EVIDENCE_GATE, `ab_compare` validator, BES). **Owned by the operator.**
- `2026-06-16-grader-noise-and-harness-remediation-design.md` — the SOTA-literature companion (actively being refined by another agent). **Do not edit.**
- Handoffs: `2026-06-16-grader-fidelity-remediation-handoff.md`, `2026-06-08-agent-codegen-tdd-hardening-handoff.md`, `2026-06-08-execution-reliability-redesign-handoff.md`.

---

## 0. Thesis

The two best current runs lose their remaining points **not to failed training but to two distinct, mechanical causes** the grading pipeline already has machinery for — it just doesn't *fire* it:

1. **Unfair docking for out-of-scope work.** A leaf about a dataset the operator never scoped in (ImageNet/COCO on a CIFAR-10 run) scored `0.0` in the *in-loop* grade, even though *finalize* correctly excluded it. The agent was then shown those `0.0` leaves as "weak" and told to fix the un-fixable. **[SHIPPED — L1.]**
2. **Open-loop repair.** `leaf_triage` *diagnoses* every recoverable leaf (render a figure, aggregate cells, re-run a failed cell, tune a per-condition LR) and writes a directive into the implementer prompt — then **hopes the agent acts**. On the real Adam run it often didn't: `fe5e7900` shipped un-rendered and `ac4006bf` shipped un-re-run, each a clean `0.0`. **[L2/L3 SHIPPED the *classification*; L4–L6 propose closing the loop.]**

**The unifying frame (novel contribution):** make the diagnosis *actuated*. `leaf_triage` is the diagnosis; the existing harness mechanisms (`staged_search` synthesizer, `cell_matrix` aggregate, `describe_capacity`, `emit_figure_sidecar`) are the cure; today there is no wire between them. We add a thin, **deterministic, budget-aware, fail-soft, default-OFF** actuator dispatch so the harness *dynamically resolves* the cheapest/safest leaf classes itself instead of docking and moving on — feeding the repaired evidence straight back into the operator's grading pipeline (A1 median-of-N → A2 deterministic → A3 evidence-fingerprint floor → A4 champion).

Every change below is `=today` byte-for-byte until a flag is set, fail-soft, and test-gated — the codebase's standing change discipline.

---

## 1. Ground truth (from the run logs, not memory)

| Run | id | shipped | in-loop | gap cause |
|---|---|---|---|---|
| ResNet | `prj_4627097f8362928c` | `final_report` **0.6201** | `rubric_evaluation` **0.3685** | 10 ImageNet/COCO/bottleneck leaves scored `0.0` in-loop, excluded only at finalize |
| Adam | `prj_29bf688e15d86b59` | `final_report` **0.764** (`meets_target=True`, 46 leaves) | — | 2 clean `0.0` leaves the agent could have repaired but didn't |

Verbatim leaf evidence (Adam `rubric_evaluation.json`):
- `fe5e7900` `0.0` — *"The code/file listing enumerates outputs/…/metrics.json and .log files but **shows zero image or figure artifacts**; train.py includes a fail-so[ft mpl guard]"* → a figure that needed rendering from data already on disk.
- `ac4006bf` `0.0` — *"metrics.json per_model has no 'imdb_logreg' entry and scope.models_run does not include it; **provenance.json lists imdb_logreg cells but they** [failed to produce output]"* → an in-scope cell attempted but errored.

ResNet's 10 excluded ids are all ImageNet-training / COCO-detection / 3-layer-bottleneck / 10-crop — none feasible on a CIFAR-10-scoped run.

---

## 2. Leaf taxonomy (L1–L6) — relabelled to avoid the handoff's A–F *workstream* letters

| # | Class | Real leaf | Root cause | Cost to fix | Status |
|---|---|---|---|---|---|
| **L1** | Out-of-inclusion-scope | ResNet ×10 | in-loop grade lacked the inclusion param finalize had | none (exclude) | **SHIPPED** |
| **L2** | Render-artifact phrasing | Adam `fe5e7900` | render regex missed "shows zero … artifacts" | none (render) | **SHIPPED (classify)** |
| **L3** | In-scope cell failure | Adam `ac4006bf` | no class for "attempted but errored" | targeted re-run | **SHIPPED (classify)** |
| **L4** | Per-condition HP fidelity | Adam optimizer-ordering leaves | one shared LR inverts the paper's ordering | targeted re-run (sweep) | **designed** |
| **L5** | Single-seed variance | ResNet 1-seed-vs-5 (3×`0.4`) | leaf wants mean±std; we ran 1 seed | GPU (N seeds) | **designed** |
| **L6** | Aggregation-completeness + arch | VAE β-sweep | ran cells absent from aggregate; arch crash | none / preflight | **designed** |

---

## 3. SHIPPED tier (commit `45c3d32`, `lolout1`, no-trailer; 2157 blast-radius tests, 0 regress)

### L1 — out-of-inclusion-scope exclusion, applied **in-loop** (the dominant fix)
**Root finding (reproducible):** `_detect_out_of_inclusion_scope_leaves` + flag `REPROLAB_SCOPE_INCLUSION_EXCLUDE` ran at **finalize only** (`finalize_rescore`). The in-loop grade (`score_reproduction`) had **no `operator_dataset_inclusion` parameter at all** — it excluded data-unavailable + theory leaves but not out-of-scope ones. So the two artifacts disagreed by 0.25 **and** the agent was shown 10 un-fixable "weak leaves".

**Fix:** `score_reproduction` gains `operator_dataset_inclusion`, applies the detector in its `skip_set` (excluded leaves also skip LLM grading), plumbed from `ctx.scope_spec.datasets` at the verify site (`primitives.py:6644`) and the freshness-regrade site (`finalize_regrade.py:214` `maybe_regrade` — **not** the no-ctx `regrade_for_hard_stop`). Operator-sourced (paper-hint `default_scope` / `--scope-spec`, never agent prose); evidence-safe (a leaf naming any in-scope dataset is never excluded). No-op unless the flag is on **and** an inclusion list is provided.

**Proof (pure-Python re-roll of the real ResNet leaves):** in-loop `0.3685 → 0.6201`, exactly 10 leaves excluded, matching finalize byte-for-byte.

**Composes with — does not duplicate — the locked spec:** this *realizes* the handoff's stated preference **F4** ("Prefer operator inclusion-scope over fuzzy matching; reserve the alias map for un-inferrable synonyms"). It runs *before* A3's evidence-fingerprint floor and A7's EVIDENCE_GATE — a leaf excluded here never reaches either.

### L2 — render-artifact phrasing
`leaf_triage` render regex (`leaf_triage.py:62`) now also matches "**zero/no** {image|figure|plot|curve|chart|visualis…} … artifacts" — Adam `fe5e7900`'s exact wording, which previously fell through to bare `review`. Still grounded: demotes to `protocol_gap` when no history/curves/sweep is on disk (`_ground`, `:204`).

### L3 — in-scope cell failure
New `cell_failure` class (`leaf_triage.py:97`, checked **last** so a contradiction still wins `result_quality`): an in-scope cell that `provenance.json`/`cells.json` records as **attempted but produced no result** → `targeted_rerun` directive *"RE-RUN the failed cell(s), don't exclude … excluding it would hide a real miss."* (`:137`). This is the deliberate complement to L1: out-of-scope → exclude; in-scope-but-failed → re-run, never exclude.

---

## 4. Proposed tier — the leaf-repair control loop (L4–L6)

### 4.0 The seam (why this is the elegant move)
`leaf_triage` today is **open-loop**: classify → directive string → implementer prompt → *hope*. Every actuator it would need already exists somewhere in the tree:

| Repair class | Existing actuator | File |
|---|---|---|
| `render_artifact` | `emit_figure_sidecar` / mpl render | agent codegen helpers |
| `aggregation_gap` | `aggregate_cell_metrics` / `normalize_cell_axes` | `cell_matrix.py` |
| `result_quality` (per-condition LR) | `synthesize_search_from_hint` → `run_staged_search` | `staged_search.py:193,404` |
| multi-seed | `ScopeSpec` seed axis + `describe_capacity` | `gpu_capacity.py:89` |
| arch crash | `execution_smoke` / `preflight_ast` | agent-codegen-TDD handoff |

**There is no wire between the diagnosis and the actuators.** We add one thin dispatcher, gated by a single master flag and a cost ceiling, that executes the deterministic repairs the agent skipped, then **re-verifies only the affected leaves** so the repaired evidence enters the operator's grading pipeline.

```
REPROLAB_LEAF_ACTUATE=0            # master gate (default OFF == today)
REPROLAB_LEAF_ACTUATE_MAX_COST=none   # ceiling: none | targeted_rerun  (free repairs only by default)
REPROLAB_LEAF_ACTUATE_SEEDS=0     # sub-gate for the GPU-cost seed expansion (L5)
REPROLAB_LEAF_SEED_MAX=5          # seed ceiling (default = paper N, hard cap 5)
```

Dispatch contract (pure, fail-soft, one actuator per leaf, idempotent): each `plan[]` entry already carries `repair_class` + `cost`. The dispatcher runs only entries with `cost ≤ ceiling`, cheapest-first (the plan is already sorted), each wrapped so a failure logs a `leaf_actuate_failed` warning and falls back to the *existing* advisory directive — i.e. raising the flag can only ever *add* a repair attempt, never remove today's behaviour.

### 4.1 L4 — per-condition hyperparameter fidelity (Adam's optimizer-ordering leaves)
**Symptom:** the paper reports an ordering ("Adam < RMSprop < SGD-mom on MNIST"); a single shared LR makes the favored method plateau and inverts it → `result_quality` leaf at `0.0`. The directive already names the cause (`leaf_triage.py:129` *"each optimizer/method/ablation needs ITS OWN best lr"*).

**Novel, grounded actuator:** `synthesize_search_from_hint(cells, lr_search)` already builds a `search` section that `run_staged_search` tunes (candidates → `select_winner` → budget-preflight → one full cell per group). We add a **second trigger**: `synthesize_search_from_leaf(cells, plan_entry, hint)` — when a `result_quality` leaf names a per-condition axis the rubric/hint enumerates (the optimizers), emit a `search` group over *that axis* and hand it to the **same** `run_staged_search`. Zero new tuning machinery; one new synthesizer keyed off the diagnosis instead of a prose hint. Shape-gated (no axis enumerable → fall back to today's advisory directive). This is the "diagnose-before-build" lesson applied: the mechanism exists, we add a trigger.

**Composes:** rides the existing staged-search route (memory `staged-search-cells-route`); the operator's deferred VAE-arch-preflight lever (same memory) is orthogonal and stays theirs.

### 4.2 L5 — budget-gated multi-seed (ResNet's 1-seed-vs-5, 3×`0.4`)
**Symptom:** a leaf demands mean±std / CI over the paper's N seeds; we ran 1 → capped at `0.4` ("single seed, no variance"). This is a *capacity* decision, not an agent bug.

**Actuator (budget-gated, no silent cap):** a **seed-replication planner** that fires only when (a) a leaf's text demands variance/CI **and** (b) `describe_capacity(ctx)` + `ctx.remaining_s()` + `RunBudget.check_run_gpu_usd` confirm `N_extra_seeds` fit the remaining wall-clock and USD budget. It expands the cell's existing `ScopeSpec` seed axis to `min(paper_N, REPROLAB_LEAF_SEED_MAX, budget_fit)` and `log()`s exactly how many it dropped (the codebase's no-silent-truncation rule). Behind `REPROLAB_LEAF_ACTUATE_SEEDS` because it is the only GPU-cost actuator; default OFF.

**Composes:** seed is already a first-class `ScopeSpec`/`cells.json` axis, so this re-uses the one-GPU-per-cell `run_matrix` placement unchanged; it borrows nothing from the locked BES/A-B workstream D.

### 4.3 L6 — aggregation-completeness critic + arch preflight (VAE β-sweep)
Two sub-issues, both `cost: none` or preflight:

**(a) Aggregation-completeness critic.** Ran cells whose axis (the β value) never appears in the canonical aggregate → the leaf can't see the comparison it asked for. `normalize_cell_axes`/`aggregate_cell_metrics` already derive missing `model_key`/`env`/`baseline` (the 2026-06-09 "a ran cell never vanishes" fix). We add a **deterministic post-aggregate assertion**: every axis a rubric leaf references must have a key in `metrics.json`; a ran-but-unaggregated axis emits a `cell_axes_derived`-style `aggregation_gap` directive (and, under actuation, folds it in). Pure, stdlib-only, unit-tested against an on-disk `metrics.json` — same discipline as `cell_matrix.py`.

**(b) Arch-fidelity preflight.** The VAE historically crashed mid-grid (device-side assert) → zeroed cells. This is *exactly* the agent-codegen-TDD handoff's `execution_smoke`/`preflight_ast` surface. We do **not** rebuild it — L6(b) is a note that flipping that handoff's preflight ON catches the arch bug pre-GPU. Owned by that handoff; cross-referenced here only so the taxonomy is complete.

---

## 5. Non-conflict matrix (explicit, per the "don't conflict with other handoffs" instruction)

| This doc | Locked grader-fidelity / noise / TDD | Relationship |
|---|---|---|
| L1 in-loop out-of-scope exclusion | handoff **F4** (prefer inclusion-scope) | **realizes** F4; runs *before* A3/A7 |
| L2/L3 leaf-triage classes | A2 `deterministic_leaf_checker` | **distinct surface** — A2 changes *scores*; leaf_triage is *advisory/actuation* over evidence |
| L4 leaf→search synthesizer | `staged_search` (existing) | new *trigger* on existing machinery; not in any locked workstream |
| L5 budget-gated seeds | D (BES/A-B), `describe_capacity` | re-uses capacity/budget; **no** overlap with BES SELECT |
| L6(a) aggregation critic | `cell_matrix` derive-not-drop | extends the 06-09 fix; pure-Python |
| L6(b) arch preflight | agent-codegen-TDD `execution_smoke` | **owned there**; cross-ref only |
| actuated evidence → re-verify | A1 median-of-N, A3 fingerprint floor, A4 champion | actuators **feed** their pipeline; ordering seam in §6 |

**Files I will touch:** `leaf_triage.py` (actuator dispatch), `staged_search.py` (`synthesize_search_from_leaf`), a new `leaf_actuator.py`, `cell_matrix.py` (completeness assertion), tests. **Files I will NOT touch:** anything the locked spec lists under "Files of record" for A1–A7 beyond the already-shipped additive `operator_dataset_inclusion` param (which the operator's F4 endorses), and `2026-06-16-grader-noise-…md`.

---

## 6. Ordering seam with A3/A4 (correctness, not optional)
Actuators mutate `code/` (render a figure, fold cells, re-run a cell). To stay honest under the operator's evidence-fingerprint floor (A3) and champion-artifact (A4): an actuator runs **before** the verify whose snapshot it intends to influence; the scoped re-verify uses the **same `evidence_key`** so A3 compares like-for-like and A4 snapshots the post-repair `code/`. An actuation that does not raise the median is therefore *not* adopted — A3/A4 remain the backstop. This is why actuation is safe to default-OFF and safe to flip per-paper: it can only *propose* a better evidence state, never *assert* one.

---

## 7. Rollout sequence (each independently flag-gated + validated)
1. **[DONE]** L1/L2/L3 — `45c3d32`. Validate: set `REPROLAB_SCOPE_INCLUSION_EXCLUDE=1` + a `--scope-spec` on a ResNet re-grade; confirm in-loop == shipped (0.6201).
2. **L6(a)** aggregation-completeness critic (pure, free) — lowest risk, ship first of the new tier.
3. **L4** leaf→search synthesizer (`cost: targeted_rerun`, rides existing staged-search).
4. **L5** budget-gated seeds (`REPROLAB_LEAF_ACTUATE_SEEDS`, GPU-cost) — last, behind its own sub-gate.
5. Master `REPROLAB_LEAF_ACTUATE` flips ON only after a clean A/B on Adam+ResNet shows actuation ≥ advisory at equal budget.

## 8. Validation gate (how we prove it helped, not just changed)
- **Deterministic re-roll** (no GPU, like the L1 proof): replay the saved Adam/ResNet leaves through the dispatcher; assert the free actuators (`render`/`aggregate`) recover the named `0.0` leaves and the score rises, with the excluded set unchanged.
- **A/B** (the operator's `ab_compare.py` validator, D1): one paper, `REPROLAB_LEAF_ACTUATE` on vs off, same seed/budget/`generated_rubric.json` (`REPROLAB_REUSE_RUBRIC=1`); adopt only if Δ ≥ 0 at equal cost and no leaf regressed.
- **No silent caps:** every dropped seed / skipped actuation emits a `log()` line.

## 9. Test plan
- `tests/rlm/test_leaf_actuator.py` — dispatch respects the cost ceiling; per-class actuator fail-soft → advisory fallback; idempotent; never raises on garbage (mirror `test_leaf_triage.py:194`).
- `tests/rlm/test_staged_search.py` — `synthesize_search_from_leaf` shape-gates (no enumerable axis → `[]`), caps candidates, defers to legacy when absent.
- `tests/runtime/test_cell_matrix.py` — completeness assertion flags a ran-but-unaggregated axis; pure, on-disk fixture.
- `tests/rlm/test_leaf_seed_planner.py` — expands to `min(N, MAX, budget_fit)`; budget-exhausted → no expansion + a logged drop.
- Re-run the L1 blast radius (`tests/evals/ tests/rlm/ tests/agents/rlm/test_cell_matrix.py tests/test_p3_grader_stance.py`) for 0-regress.

## 10. Risks / adversarial
- **Actuator masks a real failure** (re-runs until green). Mitigated: L3/L5 re-run is capped (one targeted re-run; seeds ≤ MAX), and A3's floor only adopts a *higher median* — a flaky pass that doesn't move the median is discarded.
- **Render fabricates a figure from absent data.** Mitigated: `_ground` already demotes `render_artifact`→`protocol_gap` when no history/curves/sweep on disk; the actuator inherits that gate (renders only from proven on-disk series).
- **Per-condition sweep gaming** (write a fake best). Mitigated: the sweep runs *real* cells through `run_staged_search`; A2/A7 value-sanity (locked spec) still grades the result.
- **Budget blowout from seeds.** Mitigated: hard `describe_capacity` + `RunBudget` gate, default-OFF sub-flag.

## 11. Files of record
- Shipped: `backend/evals/paperbench/leaf_scorer.py`, `backend/agents/rlm/{primitives,finalize_regrade,leaf_triage}.py`, `tests/rlm/{test_leaf_triage,test_finalize_floor_and_inclusion}.py` (commit `45c3d32`).
- Proposed: `backend/agents/rlm/leaf_actuator.py` (new), `backend/agents/rlm/staged_search.py` (`synthesize_search_from_leaf`), `backend/agents/rlm/cell_matrix.py` (completeness assertion), the four test files in §9.
- Memory: `in-loop-out-of-scope-exclusion.md`, `leaf-frontier-remediation.md`, `staged-search-cells-route.md`.

## 12. Definition of done
L1–L3 shipped + validated (done). Each of L4/L5/L6 lands flag-gated, fail-soft, test-gated, default == today, with a deterministic re-roll proof and a paired A/B before its flag is considered for default-ON — and zero edits to the operator's locked grader-fidelity/noise specs.
