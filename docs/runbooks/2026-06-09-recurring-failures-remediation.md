# 2026-06-09 — Recurring-failure remediation (Adam + All-CNN forensics)

Forensic sweep of all 15 Adam (`prj_6d41d2f09c026403`) and All-CNN
(`prj_0a3202fc187bb692`) attempts since 2026-06-07, root-causing why Adam
regressed from its 0.8308 best (06-07 21:10) to 0.69/0.0/0.0/broken/0.736/0.762,
and why All-CNN ended `status=failed, score=None` with 13/14 cells trained to
paper-grade accuracy. Every fix below is harness-generic (no paper-specific
hacks), default-on, fail-soft, and regression-tested (suite 4616 green).

## Root causes → fixes

| # | Root cause (evidence) | Fix | Code |
|---|---|---|---|
| 1 | **Silent cell discard**: agent `cells.json` lacked `model_key`/`env`/`baseline`; `aggregate_cell_metrics` skipped every ran cell → `{status: failed, per_model: {}}` while per-cell `metrics.json` held `test_accuracy 0.8945, 350/350 epochs`. Adam's `scope_shape_violation` was the downstream symptom (empty `per_model` → wrong guard branch). | `normalize_cell_axes` derives axes from synonyms (`dataset`→env, `variant`→baseline, … id fallback) at manifest load; aggregation derives as 2nd layer — a ran cell can never vanish. Agent informed via `cell_axes_derived` warning + `contract_warnings` on the result. | `cell_matrix.py`, `primitives.py::_execute_cell_matrix` |
| 2 | **Hard-stop lost the earned score**: watchdog/SIGTERM path built a bare report (rubric `None` defaults), bypassing the best-of-run floor; the rubric_evaluation merge treated `overall_score: None` as "present". All-CNN recorded 0.4908 in events, shipped `None`. | `_salvage_partial_report`: floor from events, verdict reconciled (≤ partial), structured `stop_reason`; merge now fills score keys when `None`. | `run.py`, `report.py` |
| 3 | **Stale rubric leakage across attempts**: `rubric_evaluation.json`/`rubric_tree.json`/telemetry sidecars weren't archived per attempt — 6 broken attempts reported the *previous* attempt's weak_leaves. | Added to `_ARCHIVE_FILES`: rubric_evaluation/rubric_tree/timing/tokens_total/worker_reports/environment_spec. | `attempt_isolation.py` |
| 4 | **Dataset-coverage false negatives**: env keys (`cifar10_noaug`) failed equality match vs scope datasets (`CIFAR-10`). | Digit-aware containment (`cifar100` still ≠ `cifar10`) in `_validate_scope_metrics`, both branches. | `primitives.py` |
| 5 | **`import backend` runtime crash** (Adam 06-08: `No module named 'backend'`, cost a full experiment). | Preflight AST hard violation for *unguarded* harness imports; the try/except-ImportError copy-helper pattern passes. | `preflight_ast.py` |
| 6 | **Garbage requirements line kills pip** (`Invalid requirement: '(Section'` → all deps missing → `missing_module`). | `sanitize_requirements` drops never-parseable lines pre-install (kept lines unchanged; commented in `requirements.hardened.txt`). | `env_pin.py`, `primitives.py` |
| 7 | **`compute_scope_invalid` on 15/15 attempts**: agent sends prose; warning went to dashboard only — agent never learned. | Shape correction now returned ON the plan (`warnings` re-attached post contract-dump). | `primitives.py` |
| 8 | **Grounding false alarms**: dict-reprs/LLM prose grepped against paper text. | `_extract_name`: dicts→name field, serialized literals parsed, prose skipped. | `paper_grounding.py` |
| 9 | **"CPU only (no GPU required)" planning** on every paper that predates GPU-mention culture, on a 2×GPU box → timid CPU-scale experiment plans. | `detect_environment` appends harness-measured `ENV-RT1` assumption (real GPU count/VRAM, "scale experiments to this"). | `primitives.py` |
| 10 | **Disk exhaustion** (10.9 GB free killed an Adam attempt; per-cell `datasets/` copies + `model.pt` ≈ 15 GB/run). | `scripts/gc_runs.py` — dry-run by default, `.preserved` skipped, only recomputable bulk (datasets/weights/caches); record files always survive. ~18 GB currently reclaimable with `--include-preserved`. | new script |

Non-bugs identified: the 06-08 23:37→06-09 04:08 `impl=broken` cluster was
operator test-batches SIGTERM-killed at ~25 min (expected), which *exposed* #2
and #3. SDK pre-emit stall (900 s watchdog → repairable, root retries) behaved
as designed. Two separate `batch_reproduce` invocations sharing one box halve
per-run GPU allocation (lease files are per-invocation) — launch both papers in
ONE batch invocation instead.

## Validation
- Unit/regression: `tests/rlm/test_cell_axis_normalization.py` (11),
  `tests/rlm/test_hard_stop_salvage.py` (8),
  `tests/rlm/test_harness_feedback_fixes.py` (16); legacy
  `test_malformed_cells_skipped_not_raised` re-pinned to the new derive-not-drop
  contract. Full suite **4616 passed / 0 failed**.
- Artifact replays against the real runs:
  - All-CNN per-cell outputs re-aggregated → `status: partial`, 14 leaves,
    `a_allcnn` 0.8945 accuracy preserved (was `failed/{}`).
  - All-CNN report re-finalized from real events+eval → `overall_score 0.4908`,
    verdict `partial`, `stop_reason wall_clock_watchdog` (was `None`).

## Adam 0.83 gap — what remains agent-side
The 0.831→0.762 delta decomposes into: a wasted first experiment
(`scope_shape_violation` — eliminated by #1/#4), rubric-leaf misses both runs
share (no θ̄ temporal-averaging implementation; VAE β1-sweep shape; CIFAR
early-epoch ordering), and scorer-config drift between attempts (theory-leaf
exclusion + fidelity flags changed denominators — compare attempts only under
identical flags). With #1/#2/#4 the harness no longer destroys earned points;
the remaining climb is implementation fidelity, which the rubric checklist +
repair loop now see clean signals for.
