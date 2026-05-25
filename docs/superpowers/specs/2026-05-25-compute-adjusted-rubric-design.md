# Compute-Adjusted Rubric — Design

**Status:** proposed (2026-05-25)
**Resolves:** existing task #80 ("Compute-adjusted rubric on plan_reproduction").
**Related:** `2026-05-25-three-source-budget-estimator-design.md` (orthogonal — that one is about pre-Begin estimation; this is about post-run grading).

---

## Problem

Today's rubric is **compute-blind**. It grades a reproduction's metrics against the paper's reported numbers without considering the compute budget that was deliberately clipped. In efficient mode (or with `minimize_compute=True`), a correct implementation that ran fewer epochs gets penalized identically to broken code. Result: efficient mode silently destroys ~30% of the score (the `result_match` area's weight), even when the algorithm is implemented exactly right.

User's framing: *"efficient mode shouldn't penalize scoring as it's a toggle and out of control to save time."*

---

## Anchor — floor-anchored credit (user-approved)

The planning agent — which already understands the paper's algorithm and the user's compute budget — declares an `expected_metric_floor` per result_match leaf. The floor is the metric value plausibly reachable given the actual compute. The grader scores against the floor instead of the paper's headline target.

Concretely:
- `actual_metric` ≥ `floor` → 1.0 (full credit)
- `actual_metric` between `floor` and `paper_target` → linear interp
- `actual_metric` worse than `floor` → 0.0 (the algorithm is broken; the floor was achievable)

This punishes broken code (won't reach floor), doesn't punish correct code that ran shorter, and the floor is grounded in the planning agent's domain reasoning (not a hardcoded epoch-ratio heuristic).

---

## Components

### 1. `ComputeScope` schema (new field on `ReproductionContract`)

```python
class MetricFloor(BaseModel):
    metric: str                # claim id or metric name matching the rubric leaf
    direction: Literal["higher", "lower"]   # higher-is-better vs lower-is-better
    paper_target: float        # paper's headline number for this metric
    floor: float               # plausibly-reachable in actual_epochs
    rationale: str             # one-sentence justification ("1/10 epoch budget → ~30% of final improvement")

class ComputeScope(BaseModel):
    is_clipped: bool                       # True when any axis is reduced vs paper
    paper_epochs: int | None
    actual_epochs: int | None
    paper_dataset_size: int | None
    actual_dataset_size: int | None
    rationale: str                          # "efficient mode: 5 epochs of paper's 50; ConvNet recipe unchanged"
    metric_floors: list[MetricFloor]        # one entry per result_match leaf

class ReproductionContract(BaseModel):
    # ...existing fields...
    compute_scope: ComputeScope | None = None
```

`compute_scope = None` when execution_profile.mode == "max" AND minimize_compute=False (the existing behavior — no clipping declared). Pydantic default keeps every existing call site working without change.

### 2. `plan_reproduction` primitive (prompt update)

When `execution_profile.mode == "efficient"` OR `minimize_compute=True`, the planning agent's prompt receives an additional `<compute_budget>` section:

```
You are planning a reproduction with a deliberately clipped compute budget.

Paper's stated compute (extracted from §method):
  - epochs: {paper_epochs}
  - dataset_size: {paper_dataset_size}

Available budget:
  - epochs: {actual_epochs}
  - dataset_size: {actual_dataset_size}

For each metric the paper claims, declare a metric_floor that is the value
plausibly reachable given the actual budget. Use your understanding of
convergence trajectories for this algorithm class (Adam-style is exponential
approach to floor; SGD with cosine is closer to logarithmic; GAN losses are
non-monotonic — be honest if a meaningful floor is undefined).

Format your answer as ComputeScope JSON. The floor is what we'll grade against;
overstate the floor and a working implementation will fail to reach it.
```

The planning agent returns the full ComputeScope inside the existing ReproductionContract. No new primitive call — extends an existing one.

### 3. `verify_against_rubric` grader (prompt update)

The grader's system prompt receives a `<compute_context>` section when `compute_scope` is present:

```
This reproduction ran with clipped compute:
  - paper expected {paper_epochs} epochs; actual {actual_epochs}
  - rationale: {compute_scope.rationale}

When grading the rubric, use these per-metric floors (NOT the paper's headline
targets) for result_match leaves:

  metric             direction   paper_target   floor    rationale
  -----------------  ----------  -------------  -------  -----------------
  mnist_test_cost    lower       0.500          0.850    1/10 epoch budget
  cifar_test_acc     higher      0.910          0.780    ConvNet, 5/45 epochs
  ...

Scoring rule for result_match leaves:
  * actual_metric meets or exceeds floor → 1.0
  * actual_metric between floor and paper_target → linear partial credit:
      higher_is_better: (actual - floor) / (paper_target - floor)
      lower_is_better:  (floor - actual) / (floor - paper_target)
  * actual_metric worse than floor → 0.0 (the algorithm isn't working)
  * actual_metric BETTER than paper_target → 1.0 (no upside cap)

Method/data/eval/artifact leaves are graded as you normally would —
compute clipping doesn't affect those.
```

Grader emits two scores per area (raw + adjusted) and the final report carries both.

### 4. Final report (additive)

```jsonc
{
  "rubric": {
    "overall_score": 0.42,                      // raw, against paper targets (unchanged)
    "compute_adjusted_score": 0.87,             // new — scored against floors
    "compute_scope": { ... },                   // snapshot of ComputeScope actually used
    "areas": [
      {
        "area": "Result match versus the paper's reported targets",
        "score": 0.10,                          // raw
        "compute_adjusted_score": 0.90,         // new — same leaves, floor-anchored
        "weight": 0.169
      },
      ...
    ]
  }
}
```

When `compute_scope.is_clipped == False`, `compute_adjusted_score == overall_score` (no clipping engaged — they're identical). When `compute_scope == None` (old reports), the field is null and the UI falls back to the raw score.

### 5. UI changes

`RubricStrip` (`frontend/src/components/lab/rlm/rubric-strip.tsx`):
- When `compute_scope.is_clipped == true`: headline is `compute_adjusted_score`, raw appears as small "vs paper headline: 0.42" footnote with hover-tip explaining the floor methodology.
- Per-area chips show adjusted scores on clipped runs; raw stays in the hover-detail.
- New badge: "compute-adjusted (efficient)" rendered next to the score on clipped runs.

Leaderboard:
- Existing endpoint adds two optional columns: `compute_adjusted_score` (float | null) and `mode` (`"efficient"` | `"max"`).
- Default sort key flips to `compute_adjusted_score` (so efficient and max runs are comparable). Old `overall_score` available as alt-sort.
- Filter: "show only max-mode runs" for strict comparisons (purists).

---

## Why this design

**Floor is grounded.** The planning agent has already read the paper and the compute budget — it's the right place to commit to a number. The grader downstream is mechanical (look up floor, do arithmetic) rather than re-reasoning from scratch.

**Two scores, not one replaced.** Raw stays in the report so a curious reader can always see "did this match the paper's headline." Adjusted is the headline only when clipping is declared. No information is hidden.

**No epoch-ratio heuristic.** Convergence isn't linear in epochs. Adam is exponential; GAN losses are non-monotonic; LR-schedule effects make naive ratios wrong. Letting the LLM declare the floor leverages domain knowledge a ratio can't capture.

**Backward-compat is free.** `compute_scope: ComputeScope | None = None` means every existing run/test/serialization path keeps working. The UI gracefully degrades when the field is absent.

---

## Failure modes & mitigations

| Risk | Mitigation |
|---|---|
| Planning agent overstates the floor (rewards weak code) | Floor must be ≤ paper_target by definition (validator); grader's "worse than floor → 0.0" rule prevents cheating |
| Planning agent understates the floor (defeats the purpose) | Floor compared against actual metric; if metric > floor by a lot, only get 1.0 (same as overstating) — but score is at least not penalized |
| Metric direction misclassified (loss treated as higher-is-better) | `MetricFloor.direction` is required; planning agent prompt has examples for both directions; pydantic enum enforces "higher"|"lower" only |
| Compute_scope missing for one leaf but not others | Leaves without a matching floor in the dict fall back to raw scoring (graceful degradation) |
| Two metrics with the same `metric` key | Pydantic model_validator de-duplicates with a logged warning; keeps first occurrence |
| Planning agent in `efficient` mode produces zero floors | When `is_clipped=True` but `metric_floors == []`, treat as un-clipped (fall back to raw scoring) and emit a `run_warning` event |

---

## Out of scope

- Auto-detecting "clipping happened" (e.g., OOM forced a smaller model, wall-clock budget tripped mid-run). That's a different signal — `compute_scope` is opt-in, agent-declared. A follow-up could add a `clip_reason` enum if we want to distinguish opt-in vs forced clipping.
- Calibrating floors against the 3-source budget estimator. They're independent for now; could share `PaperFeatures` in a follow-up if the data accumulates.
- Adversarial-floor detection (a planning agent that systematically overstates floors to game the leaderboard). Not a real threat at single-tenant; would need to address if multi-tenant.

---

## Testing

Schema (`tests/test_compute_scope.py`):
- ComputeScope round-trip; MetricFloor validation (direction enum, paper_target≥floor for higher-is-better, ≤ for lower-is-better)
- is_clipped derivation: True when any axis < paper; False otherwise
- Backward-compat: ReproductionContract with no compute_scope deserializes to None

Planning agent (`tests/rlm/test_plan_reproduction_compute_scope.py`):
- Stub LLM emits ComputeScope in `efficient` mode; assert shape, floor presence per result_match leaf
- `max` mode: no compute_scope emitted (None)
- Floor direction inference: leaf says "test_loss" → direction="lower" required; "test_acc" → "higher"

Grader (`tests/rlm/test_verify_compute_adjusted.py`):
- Higher-is-better at floor → 1.0
- Higher-is-better between floor and target → linear interp
- Higher-is-better worse than floor → 0.0
- Higher-is-better above target → 1.0 (no upside cap)
- Lower-is-better same four cases (inverted)
- Missing floor for a leaf → falls back to raw scoring with a warning
- compute_scope=None → both scores equal raw (unchanged behavior)

Integration (`tests/integration/test_efficient_rubric_e2e.py`):
- Full plan→implement→run→grade cycle with stubbed sandbox; asserts both scores are present and ordered (adjusted ≥ raw on clipped runs).

---

## Implementation phasing (1 PR, 5 commits)

| Phase | Output | Tests |
|---|---|---|
| 1 | Schema: ComputeScope + ReproductionContract field | test_compute_scope.py |
| 2 | plan_reproduction prompt update; planning agent emits compute_scope | test_plan_reproduction_compute_scope.py |
| 3 | verify_against_rubric prompt + scoring logic; emits both scores | test_verify_compute_adjusted.py |
| 4 | Final report serialization; leaderboard projection adds 2 columns | test_final_report_compute_adjusted.py + leaderboard route test |
| 5 | UI: RubricStrip swap headline; leaderboard sort + filter | Vitest snapshot + Playwright |

Codex adversarial review at end (mirroring how we shipped the budget estimator).

---

## Open question

Just one: when **max** mode is used, should we still emit a `compute_adjusted_score` (equal to raw)? Two views:

- **Always emit, always equal**: simpler API (UI never has to handle null), leaderboard always sorts on adjusted. Costs one redundant float on every report.
- **Only emit on clipped runs**: cleaner semantics — `compute_adjusted_score is not None` means "clipping happened." UI checks null to decide which score is the headline.

I lean **always emit, always equal** — null-handling is the most common UI bug source and the marginal cost is one float per report. If you disagree, say so before I write the implementation plan.
