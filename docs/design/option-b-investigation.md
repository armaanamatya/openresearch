# Option B Investigation — Why no improvement loop on Gate 2 fail

**Status:** Diagnosed · 2026-05-18
**Question:** `prj_b9306d43600e3d5c` had `baseline_verification.overall_score=0.27` against `rubric_target_score=0.70` with `rubric_max_improvement_iterations=2`. The pipeline halted at `gate_2_passed` with `improvement_paths=0`. Why didn't the improvement loop fire?

## Root cause: gating is on `gate_2.passed`, not on rubric score

In `backend/agents/orchestrator.py:~2425-2445` (post-pipeline section of `orchestrator.run()`):

```python
# Check gate results
if state.gate_1 and not state.gate_1.passed:
    print("  X Gate 1 FAILED")
    return state
if state.gate_2 and not state.gate_2.passed:
    if state.environment_build_attempts > 0 and not state.environment_build_ok:
        # Track 4 fail-soft: docker build never succeeded — let the run
        # finish with a partial verdict instead of halting for a human.
        print("  ! Gate 2 failed on un-buildable environment -- continuing fail-soft")
    else:
        print("  X Gate 2 FAILED")
        return state          # ← THE BUG: user's case lands here
```

For `prj_b9306d43600e3d5c`:
- Supervisor-verifier returned `GateStatus.partial_reproduction` → `state.gate_2.passed = False`.
- Docker build succeeded (`environment_build_ok = True`, `environment_build_attempts = 1`).
- So the `else` branch fired → `return state` → pipeline halted with no improvement attempt.

**The reiteration loop never executes because the function call is gated upstream by `state.gate_2.passed`.** The whole reiteration-loop code path (`_run_improvement_reiteration_loop`, the `_should_reiterate` predicate, and `run_improvements` itself) was never reachable.

## Why the existing tests look like the loop "works"

`tests/test_rubric_verifier.py` exercises `_run_improvement_reiteration_loop` and `_should_reiterate` directly — they bypass the supervisor gate entirely. So unit tests pass while the real flow halts.

## What the design actually does today

| Signal | Type | What it gates |
|---|---|---|
| `state.gate_1.passed` | supervisor-verifier checklist | run halts if False |
| `state.gate_2.passed` | supervisor-verifier checklist on baseline | run halts if False (except Track 4 fail-soft) |
| `state.baseline_verification.meets_target` | rubric-verifier score on baseline | **NOT checked** before improvement orchestration; only used inside the loop |
| `state.improved_verification.meets_target` | rubric-verifier score on improved | drives `_should_reiterate` to keep looping |

The rubric verifier's `meets_target` on the **baseline** is observable in artifacts but does not affect control flow today. Only `improved_verification.meets_target` matters, and `improved_verification` is only populated after one improvement round has completed — which can't happen if Gate 2 already halted.

## The three real options for B

### B.1 — Soften the supervisor gate
Map `GateStatus.partial_reproduction` to `gate_2.passed=True`. Pipeline continues to improvements unconditionally; supervisor-verifier's job becomes pure diagnostic, not gating.

**Pro:** simplest. One conditional change.
**Con:** the supervisor gate exists for a reason — to catch broken artifacts (missing metrics, divergent training, malformed configs). Letting `partial_reproduction` through removes that guardrail wholesale.

### B.2 — Treat rubric-below-target as its own fail-soft mode
Mirror Track 4's `environment_build_ok` escape. When `gate_2.passed=False` AND `state.baseline_verification` is present AND `not baseline_verification.meets_target`, take the "continuing fail-soft" branch instead of `return state`. The improvement loop then runs once; if it raises rubric above target, gate_2 stays False but the run produces a meaningfully different artifact.

**Pro:** matches the existing fail-soft pattern. Targeted: only kicks in when there's evidence improvement might help (verifier ran, score is below target). Preserves the supervisor gate for runs where the baseline is broken in non-rubric ways.
**Con:** ~15 LOC change in a hot path. Adds a sub-condition to a comment-rich branch that reviewers will read carefully.

### B.3 — Add an explicit `rubric_drives_improvement` config flag
New `Settings.rubric_drives_improvement: bool = False` (default off — current behavior preserved). When True, ALSO check `state.baseline_verification.meets_target` in the gate-2 result branch; if False, take the fail-soft path even when supervisor gate failed.

**Pro:** opt-in. No behavior change for existing deployments. Operator can flip it via env var.
**Con:** one more knob in an already-crowded `Settings`. Likely the same code as B.2 but behind a feature flag.

## What I'd actually do

**Implement B.2 directly** — it's the right semantic and the comment on the existing fail-soft branch already establishes "continue with honest verdict" as the runbook for gate failures the user can't fix. Adding "rubric below target" as a second trigger for the same branch is faithful to existing intent.

Estimated change: ~15 LOC in `orchestrator.run()` + a unit test that pins the new branch. Single PR.

**Reasoning against B.1:** the supervisor gate catches things the rubric verifier doesn't (e.g. baseline_result.commands_to_run is empty, metrics.json is missing). Wholesale softening loses that signal.

**Reasoning against B.3 (the flag-gated version):** the operator-friendly knob pattern (`provider_fallback_disabled`, `force_llm_provider`) is reserved for things deployments need to disable. Improvement-on-rubric-miss is a *capability* the rubric verifier was explicitly designed to enable; gating it behind a flag advertises "this feature exists but we don't trust it." Better to either commit to it (B.2) or remove it.

## Pre-condition for B.2

**Option A must land first.** B.2 routes Gate-2-failed runs through the improvement loop. If the loop itself fails (LLM errors, run-budget exhausted, etc.), the orchestrator currently still doesn't write a final report. Option A's `_finalize_partial` is the safety net that makes B.2 safe to ship: even if the improvement attempt bombs, the partial report still lands on disk.

## Open questions for the user — ANSWERED (2026-05-18)

### Q1: Is `partial_reproduction` "halt" or "try harder"?

**Answer: salvageable, try improvements first. Only commit to partial as a terminal state once Track 3 (the improvement loop) has had its chance.**

Reasoning from three angles:

1. **Semantic.** `partial_reproduction` literally names a salvageable outcome. The sibling `blocked_requires_human` is the *terminal-halt* verdict — its name explicitly says "needs human review." If `partial_reproduction` also meant terminal halt, the enum would have one verdict, not two.
2. **Existing comment.** The Track 4 fail-soft branch reads: *"let the run complete with an honest partial-reproduction verdict instead of halting for a human."* This phrases partial as the *outcome we accept after exhausting other options*, not as the cause to halt. Today the code halts *before* exhausting Track 3 — that's the gap.
3. **Track 3 vs Track 4 intent.** Track 4's fail-soft is "we ran out of build attempts; commit to partial." Track 3's improvement loop is "we have rubric headroom; try to lift the score." They're independent features. Today only Track 4 has its fail-soft wired into the gate-2 guard; Track 3 is locked behind that same guard with no escape.

**Consequence for the fix:** the right discriminator at the gate-2 result check is `state.gate_2.status` (the *verdict*), not just `state.gate_2.passed` (the *boolean*). Treat `partial_reproduction` and `blocked_requires_human` as semantically different states, even though they both currently set `passed=False`.

### Q2: Should `_should_reiterate` fire on baseline-only state?

**Answer: no — `_should_reiterate` is correct as-is, don't change it.**

Reasoning:

- `_should_reiterate(verification, iteration, max_iterations)` answers "should we go around the loop *again*?" By the time it's evaluated, one improvement round has already run (the unconditional `run_improvements` + `run_gate_3` calls before the loop body). It reads `state.improved_verification` because that's the correct signal for "did the last improvement attempt clear the bar?"
- Making it accept `baseline_verification` as a fallback would conflate "should we *start* the loop" with "should we *continue* the loop." Those should remain separate concerns.
- The unconditional first-iteration is implicit elsewhere: lines 2449 (`run_improvements`) and 2452 (`run_gate_3`) execute regardless of rubric score. That's the right place for the "always-try-once" semantics; `_should_reiterate` rightly stays focused on the loop-continuation question.

**Consequence for the fix:** no changes to `_should_reiterate`, `_run_improvement_reiteration_loop`, or the existing test surface. The whole rubric-driven loop machinery is correct in isolation; the bug is purely in the upstream gate-2 guard.

## Refined recommendation: B.2-status-aware (replaces the earlier B.2)

Instead of adding a rubric-score sub-condition to the existing fail-soft branch, **branch on `state.gate_2.status`**:

```python
if state.gate_2 and not state.gate_2.passed:
    if state.environment_build_attempts > 0 and not state.environment_build_ok:
        # Track 4 fail-soft (EXISTING): env-build cap exhausted, accept partial.
        print("  ! Gate 2 failed on un-buildable environment "
              "-- continuing fail-soft toward an honest verdict")
    elif state.gate_2.status == GateStatus.partial_reproduction:
        # Track 3 give-it-a-shot (NEW): the supervisor judged baseline as
        # partial — let the rubric-driven improvement loop try to lift the
        # score above target before we commit to partial as the verdict.
        print("  ! Gate 2 partial_reproduction "
              "-- entering improvement orchestration to try to lift the rubric")
    else:
        # blocked_requires_human or any unrecognized terminal verdict.
        print(f"  X Gate 2 FAILED: {state.gate_2.status.value}")
        return state
```

**What this is, exactly:**
- One new `elif` branch keyed on `state.gate_2.status == GateStatus.partial_reproduction`.
- Same fall-through behavior as Track 4 fail-soft: control reaches the existing `run_improvements`, `run_gate_3`, and `_run_improvement_reiteration_loop` calls below.
- If those steps later raise, Option A's `_finalize_partial` is the safety net (PR #43 — must merge first).
- The hard-halt branch is preserved for `blocked_requires_human` and any unknown verdict — defense-in-depth.

**Why this is right (vs. earlier B.2 framing):**
- Branches on declared semantics (status verdict), not on a derived rubric inequality. The verdict is already the authoritative signal.
- Zero new config flags. The existing `rubric_verifier_enabled` and `rubric_max_improvement_iterations` continue to gate the loop's depth.
- Preserves Track 4's existing escape exactly. Adds Track 3's escape as a sibling.
- Symmetric with the supervisor-verifier's design intent: `partial_reproduction` is the verdict that says "salvageable"; this code finally treats it that way.

**Implementation surface (~15 LOC + one test):**

| File | Change |
|---|---|
| `backend/agents/orchestrator.py` | Add the `elif state.gate_2.status == GateStatus.partial_reproduction:` branch with its print statement and fall-through. Import `GateStatus` if not already in scope. |
| `tests/test_orchestrator_partial_continues.py` | New test: build a state with `gate_2.passed=False`, `gate_2.status=partial_reproduction`, `environment_build_ok=True`, and assert that `run_improvements` is called (mock it) — i.e. the orchestrator did NOT early-return. |

## Validation plan after B.2-status-aware lands

- [ ] Re-trigger `prj_b9306d43600e3d5c`'s scenario (supervisor returns `partial_reproduction`, rubric `0.27 < 0.70`). Expect: at least one `run_improvements` round executes, `improvement_iteration >= 1` in final state, `improved_verification` populated, `final_report.json` exists.
- [ ] Verify the hard-halt path still works for `blocked_requires_human` (or any non-partial non-passing verdict). New unit test.
- [ ] Existing `test_rubric_verifier.py::test_reiteration_loop_*` stays green (they exercise the loop in isolation; the gate-2 guard change is upstream of them).
- [ ] PR #43 (Option A) must be merged first so `_finalize_partial` exists as the safety net when the improvement loop itself fails.

## Validation plan after B.2 lands

- [ ] Re-trigger `prj_b9306d43600e3d5c`'s scenario (or a synthetic equivalent: supervisor returns partial_reproduction + rubric below target). Confirm: one improvement round attempted, final_report.json exists with `improvement_iterations >= 1`.
- [ ] Existing test surface stays green (especially `test_rubric_verifier.py::test_reiteration_loop_*` which exercise the loop in isolation).
- [ ] New test: gate_2.passed=False + baseline_verification below target → orchestrator calls run_improvements at least once.
