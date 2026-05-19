# Option D — Gate halts on partial; rubric verifier drives the loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement task-by-task. Tasks use checkbox (`- [ ]`) syntax.

**Status:** Plan · 2026-05-19
**Supersedes:** PR #44 (`76a9a2e`) and the `!= partial_reproduction` exemptions in `263ebbb`.
**Depends on:** PR #43 (`d133f95` — `_finalize_partial`) — preserved as-is.

**Goal:** Collapse two signals (supervisor verdict + rubric score) into clean, non-overlapping roles. The supervisor gate becomes a binary halt gate — any non-pass at Gate 2 or Gate 3, including `partial_reproduction`, ends the run with `_finalize_partial`. The rubric verifier becomes the sole driver of the improvement loop, seeded from `baseline_verification` on the first iteration.

**Architecture:** `_should_reiterate` widens to consult `baseline_verification` as a fallback when `improved_verification` is `None`. The unconditional pre-loop `run_improvements + run_gate_3` pair in `orchestrator.run()` is deleted; the rubric-enabled path goes entirely through the loop. The rubric-disabled path retains a single-round legacy branch (production default is `enabled=True`, but `False` is a documented config). One outer Gate-3 halt check replaces today's three.

**Tech Stack:** Python 3.11, `pytest`, `pytest-asyncio`, `pyproject.toml` testpaths = ["tests"].

---

## Constraints discovered (audit, 2026-05-19)

| Concern | Finding | Plan response |
|---|---|---|
| `state.improvement_iteration` is operator-visible | Persisted in `pipeline_state.json`, exposed to UI as `improvementIterations` (`live_runs.py:1291`, `pipeline-dashboard.ts:1203`), printed in report markdown as "across N re-iteration round(s)" (`report_generator.py:618,628`). | Accept the semantic shift (now counts *total* improvement rounds). Update report wording in Task 5. The UI just renders the integer — no schema change. |
| `rubric_verifier_enabled` is a real config | Default `True` in `backend/config.py:111`, documented in `CLAUDE.md` and `system_overview.md`, can be flipped via env. | Keep an explicit rubric-disabled fallback in the outer flow (single unconditional round). |
| `_should_reiterate(None, None, 0, max)` must return False | The proposed predicate `if verification is None or verification.meets_target: return False` handles this naturally. | Pin with a unit test in Task 1. |

## File-by-file change summary

| File | Region | Change |
|---|---|---|
| `backend/agents/orchestrator.py` | `_should_reiterate` (~L102-114) | Widen signature: `(improved, baseline, iteration, max_iterations)`. Body: `verification = improved if improved is not None else baseline`, then the existing meets_target / cap logic. |
| `backend/agents/orchestrator.py` | `_run_improvement_reiteration_loop` (~L2155-2253) | Pass both verifications to `_should_reiterate`. Drop the `!= partial_reproduction` exemption from the inner Gate-3 break — any non-pass breaks the loop. Update docstring to reflect "counts total improvement rounds." |
| `backend/agents/orchestrator.py` | `orchestrator.run()` Gate 2 result block (~L2444-2476) | Delete the PR #44 `elif state.gate_2.status == GateStatus.partial_reproduction` branch. `partial_reproduction` collapses into the existing `else: return state` halt. (`_finalize_partial` in `run_gate_2` itself, from PR #43, still writes the artifact.) |
| `backend/agents/orchestrator.py` | `orchestrator.run()` improvement-phase block (~L2478-2533) | Delete the unconditional pre-loop `run_improvements + run_gate_3` pair. Branch on `settings.rubric_verifier_enabled`: True → only `_run_improvement_reiteration_loop`; False → legacy single-round path. Replace today's three Gate-3 halt checks with **one** outer check (drop the `!= partial_reproduction` exemption). |
| `backend/agents/report_generator.py` | L618, L628 | Change "across N re-iteration round(s)" → "across N improvement round(s)" to match new counter semantics. |
| `tests/test_gate2_partial_continues_to_improvements.py` | `test_gate2_partial_reproduction_falls_through_to_improvements` | Invert: assert `calls["improvements"] == 0`, `calls["finalize_partial"] == 1`, `final_state.stage is PipelineStage.COMPLETE` (because `_finalize_partial` in `run_gate_2` advances to COMPLETE). Rename test and file to `_halts.py`. |
| `tests/test_gate2_partial_continues_to_improvements.py` | `test_gate2_blocked_requires_human_still_halts` | Unchanged — same halt behavior preserved. |
| `tests/test_gate3_blocked_halts.py` | `test_gate3_partial_on_first_call_falls_through_to_reiteration` | Invert: mirror the `blocked` test exactly but with `status=partial_reproduction`. One Gate-3 call, `_finalize_partial` called once, no research map. |
| `tests/test_gate3_blocked_halts.py` | `test_gate3_partial_then_blocked_breaks_loop_and_halts` | Consolidate per advisor: rewrite as `test_gate3_partial_in_loop_halts` — one round of reiteration, partial verdict, loop breaks, single `_finalize_partial` call. Two cases collapse to one. |
| `tests/test_should_reiterate_baseline_fallback.py` *(new)* | Whole file | Unit tests for the widened predicate: baseline seeds first call; both-None returns False; baseline-meets-target returns False; cap bounds termination. |
| `tests/test_orchestrator_rubric_disabled_single_round.py` *(new)* | Whole file | Pin the legacy fallback: `rubric_verifier_enabled=False` runs `run_improvements + run_gate_3` exactly once each, no loop, halts on Gate-3 non-pass. |
| `tests/test_rubric_verifier.py` | Counter assertions (L166-667) | Audit semantics: each `assert ... improvement_iteration == N` now means "N total rounds," not "N re-iterations beyond the first." Mostly already-correct because most assertions are about resume round-trips, but verify each. |
| `docs/design/option-c-gate3-result-handling.md` | Top of file | Append "**Superseded 2026-05-19 by Option D** — see `option-d-q1q2-refactor.md`." Keep the doc as historical record. |
| `docs/design/option-b-investigation.md` | Q1/Q2 answer blocks | Append a "**Revised 2026-05-19**" note. The original answers (Q1=salvageable, Q2=no) are inverted. |
| `CLAUDE.md` | Rubric verification + self-improvement bullet (~L86) | Update to reflect new semantics: partial verdicts terminate; cap is "max total improvement rounds" not "max re-iterations." |

---

## Task 0: Prep — sync local with origin

**Files:** none.

- [ ] **Step 0.1: Fetch and confirm local is on the right base**

```bash
git fetch origin
git status
git log --oneline -3
git log --oneline origin/feat/gate3-result-handling-and-coverage -3
```

Expected: local HEAD is `1e358d4` (Merge PR #44); origin tip is `4c59ebe`. Local is 2 commits behind.

- [ ] **Step 0.2: Fast-forward local to origin**

```bash
git pull --ff-only origin feat/gate3-result-handling-and-coverage
```

Expected: fast-forward succeeds; `HEAD` is now `4c59ebe`.

- [ ] **Step 0.3: Confirm pre-existing test baseline is green**

```bash
.venv/bin/python -m pytest tests/test_gate2_partial_continues_to_improvements.py tests/test_gate3_blocked_halts.py tests/test_partial_final_report.py tests/test_rubric_verifier.py -v
```

Expected: all green (≈158 tests + 1 chromadb skip). This is the *before* baseline we'll invert.

---

## Task 1: Widen `_should_reiterate` to accept baseline fallback

**Files:**
- Modify: `backend/agents/orchestrator.py` (~L102-114)
- Create: `tests/test_should_reiterate_baseline_fallback.py`

**Invariants:**
- The predicate returns `False` whenever both verifications are `None` (no rubric signal to drive iteration).
- The predicate returns `False` whenever the active verification (improved if present, else baseline) has `meets_target=True`.
- The predicate returns `False` whenever `iteration >= max_iterations`.
- Otherwise returns `True`.

- [ ] **Step 1.1: Write the failing unit test file**

Create `tests/test_should_reiterate_baseline_fallback.py` with five tests (no mocks needed; this is a pure-function unit test):

```python
from backend.agents.orchestrator import _should_reiterate
from backend.agents.schemas import RubricVerification

def _verif(score: float, target: float) -> RubricVerification:
    return RubricVerification(
        overall_score=score, target_score=target,
        meets_target=score >= target, rubric_source="generated",
    )

def test_both_none_returns_false():
    assert _should_reiterate(None, None, 0, 5) is False

def test_baseline_below_target_with_no_improved_returns_true():
    assert _should_reiterate(None, _verif(0.3, 0.7), 0, 2) is True

def test_baseline_meets_target_with_no_improved_returns_false():
    assert _should_reiterate(None, _verif(0.8, 0.7), 0, 2) is False

def test_improved_overrides_baseline_when_present():
    # Improved met target — should stop, even if baseline didn't.
    assert _should_reiterate(_verif(0.9, 0.7), _verif(0.3, 0.7), 1, 2) is False
    # Improved below target — should continue, even if baseline met.
    assert _should_reiterate(_verif(0.3, 0.7), _verif(0.9, 0.7), 1, 2) is True

def test_cap_bounds_termination():
    assert _should_reiterate(None, _verif(0.3, 0.7), 2, 2) is False
```

- [ ] **Step 1.2: Run — expect failure**

```bash
.venv/bin/python -m pytest tests/test_should_reiterate_baseline_fallback.py -v
```

Expected: `TypeError: _should_reiterate() takes 3 positional arguments but 4 were given` on every test.

- [ ] **Step 1.3: Widen the predicate**

Edit `backend/agents/orchestrator.py:~102-114`. Replace the existing 3-arg signature with:

```python
def _should_reiterate(
    improved: RubricVerification | None,
    baseline: RubricVerification | None,
    iteration: int,
    max_iterations: int,
) -> bool:
    """Whether the rubric-driven improvement loop should run another iteration.

    Seeds from baseline_verification when improved_verification is None
    (the first iteration, before any improvement round has populated improved).
    Subsequent iterations key on improved_verification. Returns False — and
    terminates the loop — when there's no verification signal, the target
    is met, or the iteration cap is reached.
    """
    verification = improved if improved is not None else baseline
    if verification is None or verification.meets_target:
        return False
    return iteration < max_iterations
```

- [ ] **Step 1.4: Run — predicate tests should pass**

```bash
.venv/bin/python -m pytest tests/test_should_reiterate_baseline_fallback.py -v
```

Expected: 5 passed.

- [ ] **Step 1.5: Run — existing caller tests should now fail (the loop body still passes 3 args)**

```bash
.venv/bin/python -m pytest tests/test_rubric_verifier.py -v 2>&1 | head -40
```

Expected: TypeError-class failures from inside `_run_improvement_reiteration_loop` calling `_should_reiterate` with 3 args. We'll fix in Task 2 — no commit yet.

- [ ] **Step 1.6: Hold the commit** until Task 2 is done; the predicate change is incomplete without its caller update.

---

## Task 2: Refactor `_run_improvement_reiteration_loop` body

**Files:**
- Modify: `backend/agents/orchestrator.py` (~L2155-2253)

**Invariants:**
- The loop fires from iteration 0 (no pre-loop unconditional pair).
- First iteration uses `state.baseline_verification` (via the widened predicate).
- Inner Gate-3 break fires on **any** non-pass — drop the `!= partial_reproduction` exemption.
- `verification_history` length check (dead-verifier escape) preserved.
- Counter semantics: `state.improvement_iteration` now equals the number of completed improvement rounds (was "re-iterations beyond the first").

- [ ] **Step 2.1: Edit the loop body**

In `_run_improvement_reiteration_loop`:

1. Change the `while _should_reiterate(...)` call (~L2177-2181) to pass both verifications:
   ```python
   while _should_reiterate(
       state.improved_verification,
       state.baseline_verification,
       state.improvement_iteration,
       max_iterations,
   ):
   ```
2. Change the iteration-body's verification pickup (~L2183) from `verification = state.improved_verification` + assertion to:
   ```python
   verification = state.improved_verification or state.baseline_verification
   assert verification is not None  # guaranteed by _should_reiterate
   ```
3. In the inner Gate-3 break (~L2230-2247), drop `and state.gate_3.status != GateStatus.partial_reproduction` so the condition reads:
   ```python
   if state.gate_3 is not None and not state.gate_3.passed:
       logger.warning(
           "improvement round %d: Gate 3 returned %s — stopping the loop",
           next_iteration,
           state.gate_3.status.value,
       )
       break
   ```
4. Update the docstring to say "counts total completed improvement rounds" and reference the Q1+Q2 ruling.
5. Rename log strings from "re-iteration" to "improvement round" for clarity (this is operator-visible in `pipeline.log`).

- [ ] **Step 2.2: Run rubric-verifier tests**

```bash
.venv/bin/python -m pytest tests/test_rubric_verifier.py -v 2>&1 | tail -40
```

Expected: failures shift from TypeError to assertion mismatches on counter values (because the +1 quirk is gone). Some tests will still pass (resume round-trips); some will need updating in Task 4.

- [ ] **Step 2.3: Don't commit yet** — the outer flow still has the unconditional pre-loop pair, which will double-run improvements. Continue to Task 3.

---

## Task 3: Refactor `orchestrator.run()` — Gate 2 halt, single Gate 3 halt, rubric-mode branch

**Files:**
- Modify: `backend/agents/orchestrator.py` (~L2444-2545)

**Invariants:**
- Gate 2 non-pass (excluding Track 4 fail-soft) always halts; `_finalize_partial` (PR #43) inside `run_gate_2` already wrote the artifact.
- Improvement phase branches by `settings.rubric_verifier_enabled`:
  - `True`: only `_run_improvement_reiteration_loop` runs.
  - `False`: legacy single-round (`run_improvements` + `run_gate_3`) so research_map has data.
- Exactly **one** outer Gate-3 halt check, after both branches converge.
- That check halts on any non-pass — no `partial_reproduction` exemption.
- `_finalize_partial` fires at most once per run.

- [ ] **Step 3.1: Delete the PR #44 elif branch**

In the Gate 2 result block (~L2459-2474), remove the `elif state.gate_2.status == GateStatus.partial_reproduction:` branch entirely. The `else: print("X Gate 2 FAILED ..."); return state` catches partial along with everything else.

The Track 4 fail-soft branch (un-buildable env) is preserved exactly.

- [ ] **Step 3.2: Replace the improvement-phase block + Gate 3 checks**

Delete lines ~2478-2545 (the unconditional `run_improvements` + `run_gate_3`, the two Gate 3 halt checks, and the `_run_improvement_reiteration_loop` call). Replace with:

```python
# Improvement phase — branch by rubric verifier mode.
settings = get_settings()
if settings.rubric_verifier_enabled:
    # Rubric drives the loop. Seeds from baseline_verification on iteration 1.
    state = await self._run_improvement_reiteration_loop(
        state,
        user_hints=user_hints,
        n_improvement_paths=n_improvement_paths,
    )
    current_idx = stages_order.index(state.stage)
else:
    # Rubric disabled — single unconditional round so research_map has
    # improvement data. Preserves pre-Option-D behavior for this config path.
    if current_idx < stages_order.index(PipelineStage.IMPROVEMENTS_RUN):
        state = await self.run_improvements(
            state, user_hints=user_hints, n_paths=n_improvement_paths,
        )
        current_idx = stages_order.index(state.stage)
    if current_idx < stages_order.index(PipelineStage.GATE_3_PASSED):
        state = await self.run_gate_3(state)
        current_idx = stages_order.index(state.stage)

# Single Gate 3 halt check — any non-pass terminates (Q1 ruling 2026-05-19).
# Covers both the rubric-driven loop break path and the rubric-disabled
# single-round path. See docs/design/option-d-q1q2-refactor.md.
if state.gate_3 is not None and not state.gate_3.passed:
    print(
        f"  X Gate 3 FAILED: {state.gate_3.status.value}",
        file=sys.stderr,
        flush=True,
    )
    self._finalize_partial(state)
    return state

if current_idx < stages_order.index(PipelineStage.RESEARCH_MAP_GENERATED):
    state = await self.generate_research_map(state)
```

- [ ] **Step 3.3: Run the orchestrator-test sweep**

```bash
.venv/bin/python -m pytest tests/test_rubric_verifier.py tests/test_orchestrator_rlm_plumbing.py -v 2>&1 | tail -30
```

Expected: most pass; some test_gate2/test_gate3 tests still fail because they assert the old fall-through behavior. Those are inverted in Task 4.

- [ ] **Step 3.4: Commit Tasks 1+2+3 together**

```bash
git add backend/agents/orchestrator.py tests/test_should_reiterate_baseline_fallback.py
git commit -m "$(cat <<'EOF'
refactor(orchestrator): supervisor gate = binary halt, rubric drives loop (Q1+Q2)

Q1 (2026-05-19 revised): partial_reproduction is a terminal supervisor
verdict, not "try harder". Halt + _finalize_partial at every gate.
Reverts the PR #44 fall-through and the != partial_reproduction
exemptions added in 263ebbb.

Q2 (2026-05-19 revised): _should_reiterate now seeds from
baseline_verification when improved_verification is None. The
unconditional pre-loop run_improvements + run_gate_3 pair is deleted;
the rubric-enabled path runs entirely through _run_improvement_reiteration_loop.
A rubric-disabled fallback (single unconditional round) is preserved
because rubric_verifier_enabled=False is a real config knob.

Net change: one Gate 2 halt branch, one Gate 3 halt check (was three),
one improvement-phase branch keyed on rubric verifier mode. The
PipelineStage enum is unchanged. _finalize_partial (PR #43) is the
single terminal-artifact writer for every halt path.

Counter semantics shift: state.improvement_iteration now counts total
completed improvement rounds (was "re-iterations beyond the first").
Operators relying on the +1 quirk should bump
rubric_max_improvement_iterations by 1 to preserve total-rounds
behavior. Report wording fix in follow-up commit.

See docs/design/option-d-q1q2-refactor.md.
EOF
)"
```

---

## Task 4: Invert and consolidate existing gate tests

**Files:**
- Modify + rename: `tests/test_gate2_partial_continues_to_improvements.py` → `tests/test_gate2_partial_halts.py`
- Modify: `tests/test_gate3_blocked_halts.py`

**Invariants:**
- The negative defense-in-depth tests stay exactly as-is (blocked still halts, blocked + reiteration still blocked).
- The positive "partial falls through" tests invert to "partial halts."
- The two-state `test_gate3_partial_then_blocked_breaks_loop_and_halts` consolidates to a single `test_gate3_partial_in_loop_halts` — under new semantics, partial alone breaks the loop, no need to chain blocked.

- [ ] **Step 4.1: Rename and rewrite Gate 2 test**

```bash
git mv tests/test_gate2_partial_continues_to_improvements.py tests/test_gate2_partial_halts.py
```

In the renamed file:
- Rename `test_gate2_partial_reproduction_falls_through_to_improvements` → `test_gate2_partial_reproduction_now_halts`.
- Replace the positive-flow assertions with:
  ```python
  assert calls["improvements"] == 0, (
      "GateStatus.partial_reproduction must halt the pipeline; "
      "Option D reverts the PR #44 fall-through"
  )
  # _finalize_partial inside run_gate_2 (PR #43) advances stage to COMPLETE.
  assert final_state.stage is PipelineStage.COMPLETE
  ```
- Update the module docstring to reference `option-d-q1q2-refactor.md` instead of `option-b-investigation.md`.
- `test_gate2_blocked_requires_human_still_halts` — unchanged behavior, but update its docstring to note it's now defense-in-depth against accidentally re-introducing the partial fall-through. The assertion `final_state.stage is PipelineStage.GATE_2_PASSED` becomes `final_state.stage is PipelineStage.COMPLETE` because `_finalize_partial` (from `run_gate_2`'s rubric-miss check) now fires for blocked too. **Verify this empirically: does `run_gate_2` call `_finalize_partial` when supervisor returns blocked and rubric verifier wasn't run?** If `baseline_verification is None` (rubric disabled), `_finalize_partial` is *not* called by `run_gate_2`, and the test should keep `GATE_2_PASSED`. The fixture has `rubric_verifier_enabled=False`, so rubric won't run — stage stays at `GATE_2_PASSED`. Test assertion: unchanged. (Add a comment explaining the resolution.)

- [ ] **Step 4.2: Run the Gate 2 test**

```bash
.venv/bin/python -m pytest tests/test_gate2_partial_halts.py -v
```

Expected: 2 passed.

- [ ] **Step 4.3: Rewrite Gate 3 tests**

In `tests/test_gate3_blocked_halts.py`:
- `test_gate3_blocked_on_first_call_halts_before_reiteration` — unchanged.
- `test_gate3_partial_on_first_call_falls_through_to_reiteration` → rename to `test_gate3_partial_on_first_call_now_halts`. Mirror the blocked test's structure exactly but with `status=partial_reproduction`. Assertions: `gate_3 == 1`, `finalize_partial == 1`, `research_map == 0`, `stage is COMPLETE`.
- `test_gate3_partial_then_blocked_breaks_loop_and_halts` → consolidate into `test_gate3_partial_in_loop_halts`:
  - Setup: `rubric_verifier_enabled=True`, `rubric_max_improvement_iterations=3`, seed `state.improved_verification` below target so the loop enters.
  - `fake_run_gate_3` returns `partial_reproduction` on **every** call (no chain to blocked).
  - Assert: `gate3_calls == 2` (unconditional first + one loop iter that breaks on partial). Wait — with the pre-loop deleted, there's no "unconditional first." The first call is iter 1 of the loop. So `gate3_calls == 1`. Assert `gate3_calls == 1`, `finalize_calls == 1`, no research_map.
  - Drop the `_finalize_partial` outer-vs-inner double-check rationale from the docstring — there's only one halt site now.
- Update file docstring to reference `option-d-q1q2-refactor.md`.

- [ ] **Step 4.4: Run the Gate 3 tests**

```bash
.venv/bin/python -m pytest tests/test_gate3_blocked_halts.py -v
```

Expected: 3 passed (blocked + partial-first-call + partial-in-loop).

- [ ] **Step 4.5: Audit `test_rubric_verifier.py` counter assertions**

```bash
.venv/bin/python -m pytest tests/test_rubric_verifier.py -v 2>&1 | tail -40
```

For each failing test, the failure should be a counter-off-by-one due to the deleted pre-loop. Update assertions:
- Anywhere the test seeds `improvement_iteration=N` then expects the loop to do `max-N` more iterations, the cap is now `max` total — verify each.
- The `dead-round-not-counted` test (L667) should still pass: dead rounds don't increment.

If any test isn't a pure off-by-one — pause and call advisor.

- [ ] **Step 4.6: Commit Task 4**

```bash
git add tests/test_gate2_partial_halts.py tests/test_gate3_blocked_halts.py tests/test_rubric_verifier.py
git commit -m "$(cat <<'EOF'
test(orchestrator): invert gate tests for Option D semantics

Gate 2 partial → halts (was: falls through to improvements).
Gate 3 partial → halts (was: falls through to reiteration loop).
Consolidate two Gate 3 partial tests into one — under new semantics
partial alone breaks the loop, no chain to blocked needed.

test_rubric_verifier.py counter assertions audited for the deleted
pre-loop pair — improvement_iteration now counts total rounds.

See docs/design/option-d-q1q2-refactor.md Task 4.
EOF
)"
```

---

## Task 5: Add the rubric-disabled single-round contract test + update report wording

**Files:**
- Create: `tests/test_orchestrator_rubric_disabled_single_round.py`
- Modify: `backend/agents/report_generator.py` (L618, L628)

**Invariants:**
- `rubric_verifier_enabled=False` runs `run_improvements` exactly once and `run_gate_3` exactly once.
- `_run_improvement_reiteration_loop` is *not* invoked (early-returns trivially).
- If Gate 3 in that single round returns non-pass, outer halt check fires.
- Report markdown wording reflects "improvement round(s)" not "re-iteration round(s)" since the counter now includes the first round.

- [ ] **Step 5.1: Write the failing test**

Create `tests/test_orchestrator_rubric_disabled_single_round.py` using the same `_orch` / `_seed_state_past_baseline_run` patterns from `test_gate3_blocked_halts.py`. Two cases:
1. Gate 3 passes → research_map runs, stage COMPLETE, `improvements` called once, `gate_3` called once.
2. Gate 3 returns blocked → outer halt fires, `_finalize_partial` called once, research_map NOT called.

- [ ] **Step 5.2: Run — expect pass** (the implementation already landed in Task 3)

```bash
.venv/bin/python -m pytest tests/test_orchestrator_rubric_disabled_single_round.py -v
```

Expected: 2 passed.

- [ ] **Step 5.3: Update report wording**

In `backend/agents/report_generator.py:618,628`, change "re-iteration round(s)" → "improvement round(s)". Two occurrences.

- [ ] **Step 5.4: Run report-generator tests**

```bash
.venv/bin/python -m pytest tests/ -k "report_generator or final_report" -v
```

Expected: tests pass (these test the structure of the report, not the exact wording — but if any test pins the literal string, update it now).

- [ ] **Step 5.5: Full regression sweep**

```bash
.venv/bin/python -m pytest tests/ -v 2>&1 | tail -20
```

Expected: all green except the unrelated chromadb skip. If anything else fails, pause — likely indicates a missed counter-semantics site.

- [ ] **Step 5.6: Commit Task 5**

```bash
git add tests/test_orchestrator_rubric_disabled_single_round.py backend/agents/report_generator.py
git commit -m "$(cat <<'EOF'
feat(orchestrator): pin rubric-disabled single-round contract; update report wording

Adds the missing contract test for rubric_verifier_enabled=False:
exactly one improvement round + one Gate 3 call, outer halt fires on
non-pass.

Updates final_report.md wording from "across N re-iteration round(s)"
to "across N improvement round(s)" to match Option D counter semantics
(N is now total completed improvement rounds, not re-iterations beyond
an implicit first).
EOF
)"
```

---

## Task 6: Update narrative docs

**Files:**
- Modify: `docs/design/option-c-gate3-result-handling.md`
- Modify: `docs/design/option-b-investigation.md`
- Modify: `CLAUDE.md`
- Modify: `system_overview.md` if it documents the counter semantics or gate halt rules

- [ ] **Step 6.1: Append "Superseded" note to option-c**

At the top of `docs/design/option-c-gate3-result-handling.md`, insert under the status line:

```markdown
> **Superseded 2026-05-19 by `option-d-q1q2-refactor.md`.** Option C kept
> `partial_reproduction` as a fall-through verdict at both Gate 2 and Gate 3.
> Option D inverts that: any non-pass at either supervisor gate is terminal.
> This file is retained as historical record.
```

- [ ] **Step 6.2: Append "Revised" note to option-b's Q1/Q2 blocks**

In `docs/design/option-b-investigation.md`, immediately before each of Q1 and Q2's "Answer:" lines, add:

```markdown
> **Revised 2026-05-19 — see `option-d-q1q2-refactor.md`.** The answer
> below was overturned: Q1 is now "halt" and Q2 is now "yes (seed from
> baseline)". The reasoning preserved below is the historical context.
```

- [ ] **Step 6.3: Update CLAUDE.md**

In the "Rubric verification + self-improvement" bullet (~L86), revise the description of the cap:
- Before: "Below `rubric_target_score`, the orchestrator loops improvement-selection + Gate 3, capped by `rubric_max_improvement_iterations`."
- After: "Below `rubric_target_score`, the orchestrator runs improvement-selection + Gate 3 in a loop seeded from `baseline_verification`, capped by `rubric_max_improvement_iterations` (total improvement rounds). Any supervisor non-pass at Gate 2 or Gate 3 is terminal; `_finalize_partial` writes the artifact."

- [ ] **Step 6.4: Commit Task 6**

```bash
git add docs/design/option-c-gate3-result-handling.md docs/design/option-b-investigation.md CLAUDE.md docs/design/option-d-q1q2-refactor.md
git commit -m "$(cat <<'EOF'
docs: Option D semantic shift — partial halts, rubric drives loop

Supersede option-c-gate3-result-handling.md (kept as historical record).
Revise option-b-investigation.md's Q1/Q2 answer blocks. Update CLAUDE.md
to reflect that the cap now counts total improvement rounds and that
partial verdicts terminate.

This commit closes out Option D (PR-equivalent local work). The
implementation lives in the previous three commits.
EOF
)"
```

---

## Verification (end-of-plan)

Run all five test files plus the rubric-verifier and orchestrator sweeps:

```bash
.venv/bin/python -m pytest \
  tests/test_should_reiterate_baseline_fallback.py \
  tests/test_gate2_partial_halts.py \
  tests/test_gate3_blocked_halts.py \
  tests/test_orchestrator_rubric_disabled_single_round.py \
  tests/test_rubric_verifier.py \
  tests/test_partial_final_report.py \
  tests/test_orchestrator_rlm_plumbing.py \
  -v
```

Expected: all green (≈170+ tests, 1 chromadb skip).

Manual check after merge (a real run resuming a stuck `prj_*` with baseline rubric below target):
1. With `rubric_verifier_enabled=True` and `rubric_max_improvement_iterations=2`: exactly 2 improvement rounds fire (was 3 = 1 pre + 2 reiter under old semantics). If the second round still misses target, run completes with `final_report.json` describing partial. If Gate 3 returns blocked on round 1, run halts after 1 round.
2. With `rubric_verifier_enabled=False`: exactly 1 improvement round fires, research_map runs (or outer halt fires on Gate 3 non-pass).
3. `demo_status.json.benchmark.verdict` reflects honest terminal state at every halt site — no `pending_pipeline_result` hangs.

## Rollback

If any task fails to land cleanly, `git revert` the per-task commits in reverse order (6 → 5 → 4 → 3+2+1). Each task's commit is independently revertable except Tasks 1+2+3 which are a single commit by design (the refactor is atomic — a partial revert would leave the orchestrator in a broken state).
