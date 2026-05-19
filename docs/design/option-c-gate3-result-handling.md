# Gate 3 Result Handling + Improvement-Loop Edge Cases

> **Superseded 2026-05-19 by [`option-d-q1q2-refactor.md`](option-d-q1q2-refactor.md).**
> Option C kept `partial_reproduction` as a fall-through verdict at Gate 3
> (the reiteration loop's `!= partial_reproduction` exemption). Option D
> inverts that: any non-pass at Gate 2 or Gate 3 — including
> `partial_reproduction` — is a terminal supervisor verdict. This file is
> retained as historical record for the design conversation that produced
> commit `263ebbb`; the implementation has been replaced.

**Status:** Plan · 2026-05-18 *(implementation superseded 2026-05-19)*
**Depends on:** PR #43 (Option A — `_finalize_partial`) and PR #44 (Option B — Gate 2 partial fall-through) — both merged on `main`.
**Triggering question:** With PR #44 now routing `gate_2.status=partial_reproduction` runs through the improvement orchestration, what happens when **Gate 3** also returns non-success — particularly when an improvement round returns `partial_reproduction` (or worse)?

## Current state after PRs #43 + #44

The post-Gate-2 flow in `orchestrator.run()`:

```
gate_2 fail
  ├── env unbuildable (Track 4) → fall-through to improvements
  ├── status=partial_reproduction (Track 3, PR #44) → fall-through to improvements
  └── status=anything-else → halt + _finalize_partial via run_gate_2 (PR #43)

improvements phase
  ├── run_improvements (one unconditional first round)
  ├── run_gate_3 (one unconditional first call)
  └── _run_improvement_reiteration_loop (capped, keyed on improved_verification.meets_target)

then unconditionally:
  ├── generate_research_map (stage 9)
  └── final_report.{json,md} written via success path
  └── advance to COMPLETE
```

## The four Gate 3 outcomes

| `gate_3.status` | `passed` | `improved_verification.meets_target` | Current handling | Issue? |
|---|---|---|---|---|
| `verified` | True | True | Falls through to research map ✓ | None |
| `verified_with_caveats` | True | True or False | Falls through; reiteration loop may run if rubric below target | None |
| `partial_reproduction` | False | Likely False | Reiteration loop fires (up to cap), then research map ✓ | Cost: may burn `max_iterations` LLM rounds even when supervisor is signaling the approach is fundamentally salvageable but slow |
| `blocked_requires_human` | False | False | **Reiteration loop still fires** — supervisor says "halt for human" but loop runs another round anyway | **Real gap.** Wastes LLM cost on a path the supervisor declared blocked, AND ignores the human-review escape that `blocked_requires_human` was designed to be |
| `failed_reproduction` / `invalid_claim` | False | False | Same as blocked | Same gap |

## The minimum-viable fix

**Treat `gate_3.status` symmetrically with `gate_2.status` (PR #44):**

- `partial_reproduction` → continue. The reiteration loop's existing termination logic (rubric meets target OR iteration cap reached) handles this case correctly today; no change needed.
- `blocked_requires_human` / `failed_reproduction` / `invalid_claim` → halt with `_finalize_partial`. Mirror Gate 2's behavior so the human-review escape that the supervisor encoded survives.

## Three implementation options

### C.1 — Halt only on the unconditional first `run_gate_3` call (smallest)

Add one new check at `orchestrator.py:~2470`, immediately after the unconditional `state = await self.run_gate_3(state)`:

```python
if (
    state.gate_3 is not None
    and not state.gate_3.passed
    and state.gate_3.status != GateStatus.partial_reproduction
):
    print(f"  X Gate 3 FAILED: {state.gate_3.status.value}", file=sys.stderr, flush=True)
    self._finalize_partial(state)
    return state
```

**Pro:** smallest, symmetric with Gate 2's structure.
**Con:** the reiteration loop inside `_run_improvement_reiteration_loop` also calls `run_gate_3` (line ~2210) and is NOT covered by this check. If iteration 1's gate_3 returns blocked, iteration 2 still fires.

### C.2 — Halt at every `run_gate_3` invocation (most robust)

C.1 plus: inside `_run_improvement_reiteration_loop`, immediately after the inner `state = await self.run_gate_3(state)`, check the same condition and `break` the loop. The outer `orchestrator.run()` flow will then check the same condition after the loop returns and call `_finalize_partial` if blocked.

**Pro:** covers every gate_3 invocation. No wasted rounds after a blocked verdict.
**Con:** ~6 LOC more than C.1. Reviewer reads two changes instead of one.

### C.3 — Refactor: extract `_should_halt_after_gate3(state) -> bool`

Same as C.2 but pull the decision into a helper so the outer and inner check are literally the same call. Pure cleanup of duplication.

**Pro:** single source of truth for the halt rule. Easy to test the predicate in isolation.
**Con:** +5 LOC vs C.2 with no observable behavior change.

## Recommendation

**Ship C.2.** The reiteration loop's whole point is to retry on partial; running more iterations after a `blocked_requires_human` verdict is a bug, not a tradeoff. C.1 would land that bug. C.3 is fine but the duplication is two call sites, which is the threshold above which extraction usually pays off — and below which inline is OK. C.2 reads cleanly: "after every gate_3, check if we should halt."

## What the existing reiteration cap already protects against

Worth being explicit so the PR description is accurate about scope:

| Concern | Already handled by | Action needed? |
|---|---|---|
| All rounds return partial, run never terminates | `rubric_max_improvement_iterations` cap; `_should_reiterate(iteration < max_iterations)` | No |
| Verifier itself fails mid-loop | `verification_history` length check inside the loop | No |
| LLM exception inside `run_improvements`/`run_gate_3` | Try/except inside the loop body, `break` on Exception | No |
| Gate 3 returns `blocked_requires_human` on first call | **NOT handled — gap that C.2 closes** | Yes |
| Gate 3 returns `blocked_requires_human` on iteration N | **NOT handled — same gap** | Yes |
| Improved verification meets target | `_should_reiterate` returns False | No |
| Cap reached with rubric still below | Loop exits → research_map runs → final_report describes partial state | No |

The cap + verification-history-length check are robust against the cost/termination class of bug. The only outstanding issue is the supervisor's halt verdict being ignored.

## Implementation surface (~20 LOC + tests)

| File | Change |
|---|---|
| `backend/agents/orchestrator.py` | (1) After the unconditional `run_gate_3` call in `orchestrator.run()`, check `state.gate_3.passed is False` + status not in `{partial_reproduction}` → print + `_finalize_partial` + return. (2) Inside `_run_improvement_reiteration_loop`, same check after each inner `run_gate_3` → `break` the loop. (3) After the loop returns, the outer flow already would catch the case if needed, but the inner break is enough — no need to re-check. |
| `tests/test_gate3_blocked_halts.py` | New file. Three tests: (a) gate_3 returns `blocked_requires_human` on first call → `_finalize_partial` is called, run ends at COMPLETE, `run_improvements` NOT called a second time. (b) gate_3 returns `partial_reproduction` on first call → reiteration loop is reached (existing behavior, just pin it). (c) gate_3 returns `partial_reproduction` then `blocked_requires_human` on round 2 → loop breaks early; `_finalize_partial` from outer flow OR success path writes final_report. |

## Test fixture strategy (reusing PR #44 patterns)

The PR #44 tests built the infrastructure: `_orch(tmp_path, monkeypatch, project_id)` + `_seed_state_at_baseline_run` + `_install_stage_mocks`. The new tests should follow the same shape:

- Same `SimpleNamespace`-mocked settings (rubric_verifier_enabled toggled per test).
- Same checkpoint-then-`orch.run(resume=True)` entry pattern.
- The new mock for `run_gate_3` records the call count AND sets `state.gate_3.status` to the test's chosen value.

For the reiteration-loop test (c), enable the loop (`rubric_verifier_enabled=True`, `rubric_max_improvement_iterations=2`) and have `run_gate_3`'s side-effect return `partial_reproduction` on call 1 and `blocked_requires_human` on call 2. Assert that `run_improvements` was called exactly twice (one unconditional + one reiteration) and not three times.

## What this PR does NOT do

- It doesn't touch `_should_reiterate` — that predicate is correct as-is (keyed on rubric verification, not gate_3 status). The new behavior is encoded as an EARLY BREAK in the loop body before `_should_reiterate` is re-evaluated.
- It doesn't change `partial_reproduction`'s behavior on Gate 3. That's still "let the reiteration loop decide" — same as Gate 2 (PR #44).
- It doesn't add a Settings flag. Like PR #44, the verdict enum already carries the right signal; we just read it.
- It doesn't add a Gate 1 result handling change. Gate 1 (plan verification) already halts on any non-pass — no rubric loop there.

## Validation plan

- [ ] `pytest tests/test_gate3_blocked_halts.py -v` — all new tests pass.
- [ ] `pytest -k "orchestrator or rubric or gate or partial"` — regression sweep stays green.
- [ ] Manual: trigger a real run where the improvement path is fundamentally broken (e.g., baseline imports a missing module). Expect: first `run_gate_3` returns `blocked_requires_human`; pipeline halts at COMPLETE with `final_report.json` describing the blocked outcome (via Option A's `_finalize_partial`).

## Open questions (NONE that block this PR)

The previous design rounds had open questions (Q1 about `partial_reproduction` semantics, Q2 about `_should_reiterate` baseline-seeding). Those have answers now and don't apply to Gate 3 — `blocked_requires_human` has the unambiguous "halt for human" semantic, and the reiteration loop's first-iteration trigger doesn't change.
