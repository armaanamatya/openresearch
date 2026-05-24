# Option D — Inject sandbox time-budget into agent prompts

**Status:** Plan · 2026-05-19
**Depends on:** PRs #43 / #44 / #45 — all merged on `main`.
**Triggering evidence:** Real run `prj_9cbac43dcba7d926` (2026-05-19 00:17 → 01:52, 95 min total) — Gate 2 returned `failed_reproduction` with rubric 0.268 / 0.700 because the generated full reproduction script could not complete inside the 60-min sandbox `command_timeout` cap.

## What happened (the live evidence)

The `baseline-implementation` agent generated `run_experiments.sh` with this scope:

| Phase | Scope |
|---|---|
| smoke | 1 env × 1 seed × 5,000 steps |
| **mujoco** | **4 envs × 3 seeds × 1,000,000 steps** |
| atari | 5 games × 2 seeds × 10,000,000 steps |
| ablation | 3 surrogates × 3 seeds × 1,000,000 steps |

`mujoco_run.log` shows the script started Phase 1 (HalfCheetah seed=1) and was actively learning:

```
update=1/488   step=2048    mean_ep_ret=-742.48  sps=33
update=50/488  step=102400  mean_ep_ret=-379.81  sps=65
update=100/488 step=204800  mean_ep_ret=  37.64  sps=87
update=150/488 step=307200  mean_ep_ret= 339.41  sps=99   ← killed by 3600s timeout
```

At `sps=99`, one HalfCheetah run alone needs ~168 minutes — **2.8× the 60-min sandbox cap.** Twelve MuJoCo runs + 10 Atari runs + 9 ablations would need ~30 hours. The agent's code was correct; the scope was unboundedly paper-faithful.

## Why this isn't a bug in PRs #43-45

| PR | Expected behavior | Observed |
|---|---|---|
| **#43** `_finalize_partial` writes `final_report.{json,md}` when rubric below target | ✅ Files written at 01:52:44 (9172 + 2581 bytes); `demo_status.json.benchmark.verdict` correctly updated to `failed`. |
| **#44** Gate 2 partial fall-through | ✅ Correctly did NOT fire — verdict was `failed_reproduction`, not `partial_reproduction`. |
| **#45** Gate 3 blocked halt | ✅ Correctly did NOT fire — Gate 2 halted before Gate 3 was reached. |

The pipeline produced an honest terminal artifact. The implementation was correct; the experiment scope was too large for the runtime budget.

## The architectural gap

`reproduction-planner` and `baseline-implementation` agents have **no visibility into `command_timeout`**. They scope at paper-scale. The sandbox enforces 60 min. Mismatch is systematic.

## Three fix options (ranked)

### D.1 — Inject `time_budget_seconds` into the agent prompt context (recommended)

Pass the sandbox's `command_timeout` value as an explicit field in the agent context. Modify the prompt templates for `reproduction-planner` and `baseline-implementation` to include:

> Your generated code MUST fit within `${time_budget_seconds}` seconds inside the sandbox. Choose env/seed/timestep counts to fit. The pipeline records a smoke test plus a primary reduced-scope run; full paper-scale reproduction is out of scope for this sandbox.

**Surface (~30 LOC + test):**

| File | Change |
|---|---|
| `backend/agents/orchestrator.py` | In `run_baseline_implementation` and `run_reproduction_planner`, add `time_budget_seconds` to the agent context dict. Source from `self.execution_profile.command_timeout` (or whatever the field is named — verify in `backend/agents/execution.py`). |
| `backend/agents/prompts/baseline_implementation.py` | Add explicit "fit within {time_budget_seconds}s" instruction. |
| `backend/agents/prompts/reproduction_planner.py` | Same — the planner should commit to a scope that fits. |
| `tests/test_baseline_implementation_time_budget.py` | New test asserting `time_budget_seconds` appears in the rendered prompt. Lightweight — no full orchestrator needed. |

**Why this is right:**
- Same surgical-edit shape as PRs #43-45.
- No new Settings field — `command_timeout` already exists.
- The agent is the right place to enforce the budget; it's the one writing the code.
- Smoke + one reduced-scope full run is still a faithful reproduction CHECK (not a paper-replication), which matches the pipeline's actual purpose.

### D.2 — Have the planner emit a "compute budget" estimate

Reproduction-planner produces a structured `reproduction_contract.json` already (training_config, env_ids, seeds). Add a `compute_budget` field: estimated total seconds the planned run will take. The orchestrator validates against `command_timeout` and rejects the plan (round-trips for replanning) if it exceeds the budget.

**Pro:** explicit, testable contract.
**Con:** ~80 LOC; adds a validation gate; planner needs a cost model for steps→time per env. Heavier.

### D.3 — Bump `command_timeout`

Default is 3600s. Could raise to e.g. 28800s (8 hours). Operator-friendly knob, no agent changes.

**Pro:** trivial — change one config default.
**Con:** doesn't scale to the actual paper scope (30+ hours on CPU). Hides the architectural problem rather than fixing it. Still wouldn't fit Schulman 2017 in full.

## Recommendation

**Ship D.1.** Same shape as the recent PR sequence. Single file changes + one test. Lets the agent self-budget; the rubric verifier will then have real metric deltas to score against, which lifts `comparison_quality` from 0.00 → ~0.7+ and `baseline_implementation` from 0.38 → ~0.85+. Overall rubric should clear the 0.70 target on a future run of the same paper.

D.2 is the right long-term answer if the planner→validator round-trip pattern is wanted elsewhere. D.3 alone won't fix it.

## Validation plan after D.1 lands

- [ ] Unit test: rendered prompt for `baseline-implementation` contains `time_budget_seconds=3600` (or whatever the configured value is).
- [ ] Integration: re-run the demo paper. Expect the generated `run_experiments.sh` to scope smaller (e.g. 200K steps × 2 seeds × 1-2 envs), complete within 60 min, produce non-empty `metrics.json`, lift rubric score above target.
- [ ] Regression: existing test surface (`tests/test_rubric_verifier.py`, `tests/test_gate3_blocked_halts.py`, etc.) stays green.

## Open question (worth thinking about before D.1 lands)

**What's the "reduced scope" semantics?** Two valid framings:
1. **"Demonstration scope"**: ~5% of paper-scale — proves the code works, smoke test + one short training run. Quick, cheap, doesn't validate the paper's claim.
2. **"Confidence scope"**: ~20-40% of paper-scale on the cheapest env — provides a real-ish metric the rubric can score against. Slower but produces the comparison the rubric needs.

D.1 should pick one and communicate it explicitly in the prompt. Recommend framing 2: rubric needs real metrics; framing 1 leaves `comparison_quality` at 0 again.

---

## Linked artifacts

- The failed run: `logs/20260519-001647/prj_9cbac43dcba7d926/` (final_report.{json,md}, baseline/mujoco_run.log)
- Prior design docs: `docs/design/option-a-…`, `option-b-…`, `option-c-…` — same PR shape applies.
