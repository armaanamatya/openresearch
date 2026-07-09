# Lifecycle-Primary Hardening (iteration-counter keystone) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make lifecycle-primary produce a downstream contract indistinguishable from the completion path by fixing the one keystone gap — the iteration counter — plus finalize sourcing and improve-phase observability, with parity tests.

**Architecture:** Post-extraction the design collapsed: the lifecycle driver calls the *already-wrapped* primitives, so `binding.wrap_primitive` already emits `rubric_score` and sets `ctx.latest_rubric_*` under lifecycle mode — they just carry `iteration=0` because `ctx.current_iteration` is only advanced by the root-loop logger (dead in lifecycle mode). Fix: the lifecycle driver owns/advances `ctx.current_iteration` per step (this makes the events + ctx state correct *for free*, since binding reads it); finalize sources the count from the driver summary; the climb emits per-hypothesis events. The risky `binding.wrap_primitive` refactor is **dropped**.

**Tech Stack:** Python 3.12, pytest + pytest-socket (hermetic).

**Spec:** `docs/superpowers/specs/2026-07-09-lifecycle-primary-hardening-design.md` (see the "Post-extraction correction" block).

**Deferred to a follow-on plan (NOT in scope here):** stage-checkpoint resume, `evidence_bundle.json` write from the driver, campaign-loop integration test, the ideation seam. These need more recon (checkpoint resume shape, the evidence-bundle builder API, the campaign attempt loop) and are enhancements, not blockers.

**Baseline:** the 168 lifecycle tests are green. Run tests with `export OPENRESEARCH_MIN_DISK_GB=0` and the worktree's absolute venv interpreter. Commit rule: present-tense headline, NO Conventional-Commit prefix, NO `Co-Authored-By` trailer.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `backend/agents/rlm/lifecycle_driver.py` | Deterministic stage driver | Advance `ctx.current_iteration` per step; stamp `summary["iterations"]`; emit iteration in drive-step + a climb event |
| `backend/agents/rlm/run.py` | Orchestrator wiring + finalize | Source `iterations=` from the summary when `_primary_active` |
| Tests | — | keystone unit test + a binding-integration parity test |

---

## Task 1: Advance ctx.current_iteration per lifecycle step (the keystone)

**Files:**
- Modify: `backend/agents/rlm/lifecycle_driver.py` — the `_step()` helper (lines 293-333) and the `run_lifecycle_primary` return (~line 668-674).
- Test: `tests/rlm/test_lifecycle_iteration_counter.py`

**Why:** `binding.wrap_primitive` reads `ctx.current_iteration` for the `rubric_score` event (`binding.py:1016`) and `ctx.latest_rubric_iteration` (`binding.py:921`). In lifecycle mode nothing advances it, so everything stamps 0. Advancing it in `_step` fixes the events + ctx state for free.

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_lifecycle_iteration_counter.py
"""Lifecycle-primary must advance ctx.current_iteration per driven step so the
rubric_score event + ctx.latest_rubric_iteration + finalize's iteration count
are real (not 0). binding.wrap_primitive reads ctx.current_iteration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from backend.agents.rlm.lifecycle_driver import drive_lifecycle_chain, run_lifecycle_primary


def _ctx(tmp_path):
    d = tmp_path / "proj"
    d.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(project_dir=d, remaining_s=lambda: None, current_iteration=0)


def _tool(ret):
    m = MagicMock()
    m.return_value = ret
    return m


def _tools():
    return {
        "understand_section": {"tool": _tool({"sections": []})},
        "detect_environment": {"tool": _tool({"env": "conda"})},
        "plan_reproduction": {"tool": _tool({"method_spec": "x"})},
        "implement_baseline": {"tool": _tool({"ok": True, "code_path": "/c"})},
        "run_experiment": {"tool": _tool({"success": True, "metrics": {"r": 0.7}})},
        "verify_against_rubric": {"tool": _tool({"overall_score": 0.9, "meets_target": True})},
    }


def test_full_backbone_advances_iteration_to_six(tmp_path):
    ctx = _ctx(tmp_path)
    drive_lifecycle_chain(
        tools=_tools(), ctx=ctx, paper_text="p", rubric_spec={"target_score": 0.7},
        start_stage="need_baseline", emit=lambda e: None,
    )
    # understand, detect, plan, implement, run, verify = 6 steps
    assert ctx.current_iteration == 6


def test_run_primary_stamps_iterations_in_summary(tmp_path):
    ctx = _ctx(tmp_path)
    summary = run_lifecycle_primary(
        tools=_tools(), ctx=ctx, paper_text="p", rubric_spec={"target_score": 0.7},
        emit=lambda e: None, target_score=0.7, max_improve_iterations=0,
    )
    assert summary["iterations"] == ctx.current_iteration
    assert summary["iterations"] >= 6
```

- [ ] **Step 2: Run it, verify FAIL**

`export OPENRESEARCH_MIN_DISK_GB=0 && <VENV> -m pytest tests/rlm/test_lifecycle_iteration_counter.py -v`
Expected: FAIL (`ctx.current_iteration` stays 0; `summary["iterations"]` KeyError).

- [ ] **Step 3: Advance the counter in `_step`**

In `backend/agents/rlm/lifecycle_driver.py`, inside `_step` (lines 293-333), find the emit block:
```python
        _safe_emit(
            emit,
            {"event": "lifecycle_drive_step", "stage": start_stage, "primitive": name},
        )
```
Immediately BEFORE it, add the counter advance, and ADD an `iteration` field to the event:
```python
        # Advance the harness-owned iteration counter BEFORE the tool runs so
        # binding.wrap_primitive (which reads ctx.current_iteration for the
        # rubric_score event + ctx.latest_rubric_iteration policy state) records
        # real, non-zero iterations under lifecycle-primary. The root-loop logger
        # that normally advances it never runs here.
        try:
            _it = int(getattr(ctx, "current_iteration", 0) or 0) + 1
            setattr(ctx, "current_iteration", _it)
        except Exception:  # noqa: BLE001 — counter is best-effort, never blocks a step
            _it = int(getattr(ctx, "current_iteration", 0) or 0)

        _safe_emit(
            emit,
            {"event": "lifecycle_drive_step", "stage": start_stage,
             "primitive": name, "iteration": _it},
        )
```

- [ ] **Step 4: Stamp `summary["iterations"]` in run_lifecycle_primary**

In `run_lifecycle_primary`, just before the final `return summary` (after `summary["improved"] = improved` / the `summary["rubric_score"]`/`summary["verify_result"]` writes, ~line 668-674), add:
```python
    summary["iterations"] = int(getattr(ctx, "current_iteration", 0) or 0)
```
ALSO add the same stamp in `drive_lifecycle_chain` right before EACH `return summary` path is not required — but add it once at the normal terminal return of `drive_lifecycle_chain` (the final `return summary` at the end of the function) so a backbone-only drive also reports iterations:
```python
    summary["iterations"] = int(getattr(ctx, "current_iteration", 0) or 0)
    return summary
```
(If `drive_lifecycle_chain` has multiple early `return summary` statements, they already carry the advanced counter via ctx; only the FINAL normal return needs the explicit stamp for the happy path. Do not chase every early return — the keystone is the ctx advance.)

- [ ] **Step 5: Run the test, verify PASS**

`export OPENRESEARCH_MIN_DISK_GB=0 && <VENV> -m pytest tests/rlm/test_lifecycle_iteration_counter.py -v`
Expected: 2 passed.

- [ ] **Step 6: Regression — the existing lifecycle suite must stay green**

`export OPENRESEARCH_MIN_DISK_GB=0 && <VENV> -m pytest tests/rlm/test_lifecycle_driver.py tests/rlm/test_run_lifecycle_primary.py tests/rlm/test_lifecycle_driver_hardening.py -q`
Expected: all pass (existing order-assertion tests unaffected; they check `driven`, not iteration). If any test asserted the event dict EXACTLY (without `iteration`), update it to allow the new field.

- [ ] **Step 7: Commit**

```bash
git add backend/agents/rlm/lifecycle_driver.py tests/rlm/test_lifecycle_iteration_counter.py
git commit -m "Advance ctx.current_iteration per lifecycle step so rubric events carry real iterations"
```

---

## Task 2: Source finalize's iteration count from the driver summary under lifecycle-primary

**Files:**
- Modify: `backend/agents/rlm/run.py` — the `_primary_active` block (~4468-4517) and the finalize call (`iterations=rlm_logger.iteration_count`, line 4614).
- Test: `tests/rlm/test_lifecycle_primary_iterations_finalize.py`

**Why:** `run.py:4614` passes `iterations=rlm_logger.iteration_count`, which is 0 in lifecycle mode (the root logger never ran). Source it from `summary["iterations"]` instead when `_primary_active`.

- [ ] **Step 1: Write the failing test**

```python
# tests/rlm/test_lifecycle_primary_iterations_finalize.py
"""Under lifecycle-primary, the finalize iteration count must come from the
driver summary, not rlm_logger.iteration_count (which is 0 in this mode)."""

from __future__ import annotations

from backend.agents.rlm import run as run_mod


def test_primary_iterations_prefers_summary_over_logger():
    # The helper picks the summary's iteration count when primary is active,
    # else the logger's. (Task 2 introduces _resolve_run_iterations.)
    assert run_mod._resolve_run_iterations(
        primary_active=True, summary={"iterations": 6}, logger_iterations=0
    ) == 6
    assert run_mod._resolve_run_iterations(
        primary_active=False, summary=None, logger_iterations=4
    ) == 4
    # Missing summary iterations under primary -> fall back to logger (defensive).
    assert run_mod._resolve_run_iterations(
        primary_active=True, summary={}, logger_iterations=3
    ) == 3
```

- [ ] **Step 2: Run it, verify FAIL** (`_resolve_run_iterations` doesn't exist).

- [ ] **Step 3: Add the resolver helper**

In `backend/agents/rlm/run.py`, near the other `_primary_*`/`_drive_*` helpers (around lines 1174-1214), add:
```python
def _resolve_run_iterations(
    *, primary_active: bool, summary: dict | None, logger_iterations: int
) -> int:
    """Pick the iteration count for the final report.

    Lifecycle-primary drives its own loop, so ``rlm_logger.iteration_count`` is 0
    there — use the driver summary's ``iterations`` instead. Fall back to the
    logger count when not primary, or when the summary lacks the key (defensive).
    """
    if primary_active and summary:
        val = summary.get("iterations")
        if isinstance(val, int) and val > 0:
            return val
    return int(logger_iterations or 0)
```

- [ ] **Step 4: Run the helper test, verify PASS (3 asserts).**

- [ ] **Step 5: Wire it at the finalize call**

In the `_primary_active` block, the driver `summary` is already in scope. Keep a reference usable at finalize time — right after `result_obj = _synth_result_from_summary(summary, ctx)`, add:
```python
            _primary_summary = summary
```
And ensure `_primary_summary` is initialized to `None` before the branch (add `_primary_summary = None` next to `_primary_active = False` at the top of the block, ~line 4468).
Then at the finalize call (line 4614), change:
```python
            iterations=rlm_logger.iteration_count,
```
to:
```python
            iterations=_resolve_run_iterations(
                primary_active=_primary_active,
                summary=_primary_summary,
                logger_iterations=rlm_logger.iteration_count,
            ),
```
Confirm `_primary_active` and `_primary_summary` are in scope at line 4614 (they are set earlier in the same function). If `rlm_logger` is not in scope in some branch, keep the existing reference — only the `iterations=` argument changes.

- [ ] **Step 6: Regression**

`export OPENRESEARCH_MIN_DISK_GB=0 && <VENV> -m pytest tests/rlm/test_lifecycle_primary_iterations_finalize.py tests/rlm/test_run_lifecycle_primary.py tests/rlm/test_run_lifecycle_drive.py -q`
Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add backend/agents/rlm/run.py tests/rlm/test_lifecycle_primary_iterations_finalize.py
git commit -m "Source finalize iteration count from the driver summary under lifecycle-primary"
```

---

## Task 3: Emit a per-hypothesis event during the improvement climb

**Files:**
- Modify: `backend/agents/rlm/lifecycle_driver.py` — the climb loop in `run_lifecycle_primary` (~577-674).
- Test: extend `tests/rlm/test_lifecycle_iteration_counter.py` (or a new small test).

**Why:** the climb currently emits only the nested `drive_lifecycle_chain` steps; the dashboard can't see which hypothesis was tried. Emit one event per climb iteration.

- [ ] **Step 1: Write the failing test**

Add to a test file:
```python
def test_climb_emits_hypothesis_event(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from backend.agents.rlm.lifecycle_driver import run_lifecycle_primary

    events = []
    d = tmp_path / "proj"; d.mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(project_dir=d, remaining_s=lambda: None, current_iteration=0)

    def _t(ret):
        m = MagicMock(); m.return_value = ret; return m

    # Backbone scores 0.5 (< target 0.7) so the climb runs once; propose returns
    # one hypothesis; the sub-drive re-verifies to 0.8.
    verify = _t({"overall_score": 0.5, "target_score": 0.7})
    tools = {
        "understand_section": {"tool": _t({})},
        "detect_environment": {"tool": _t({})},
        "plan_reproduction": {"tool": _t({})},
        "implement_baseline": {"tool": _t({"ok": True, "code_path": "/c"})},
        "run_experiment": {"tool": _t({"success": True, "metrics": {"r": 0.5}})},
        "verify_against_rubric": {"tool": verify},
        "propose_improvements": {"tool": _t([{"hypothesis": "raise lr", "success": True}])},
    }
    run_lifecycle_primary(
        tools=tools, ctx=ctx, paper_text="p", rubric_spec={"target_score": 0.7},
        emit=events.append, target_score=0.7, max_improve_iterations=1,
    )
    assert any(
        e.get("event") == "lifecycle_drive_step" and e.get("phase") == "improve"
        for e in events
    ), events
```

- [ ] **Step 2: Run it, verify FAIL** (no `phase == "improve"` event).

- [ ] **Step 3: Emit the climb event**

In the climb loop, right after `chosen = valid[0]` (before the `implement_baseline` call), add:
```python
        _safe_emit(
            emit,
            {"event": "lifecycle_drive_step", "stage": "improve", "phase": "improve",
             "primitive": "propose_improvements",
             "hypothesis": chosen.get("hypothesis"),
             "iteration": int(getattr(ctx, "current_iteration", 0) or 0)},
        )
```

- [ ] **Step 4: Run the test, verify PASS.**

- [ ] **Step 5: Regression**

`export OPENRESEARCH_MIN_DISK_GB=0 && <VENV> -m pytest tests/rlm/test_run_lifecycle_primary.py tests/rlm/test_lifecycle_iteration_counter.py -q`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add backend/agents/rlm/lifecycle_driver.py tests/rlm/test_lifecycle_iteration_counter.py
git commit -m "Emit a per-hypothesis event during the lifecycle improvement climb"
```

---

## Task 4: Parity — binding-integration test proving rubric events carry a real iteration

**Files:**
- Test only: `tests/rlm/test_lifecycle_binding_parity.py`

**Why:** Tasks 1-3 are unit-tested against raw mock tools, which bypass `binding`. This task proves the KEYSTONE end-to-end: when the driver runs a `binding.wrap_primitive`-wrapped `verify_against_rubric`, the emitted `rubric_score` event carries `iteration > 0` (i.e., the counter advance actually reaches binding).

- [ ] **Step 1: Recon the binding test fixtures**

Read `tests/rlm/test_binding.py` to see how it constructs a `RunContext`-like object with `cost_ledger`, `emit`, `current_iteration`, `provider`, `model`, `project_dir`, and `llm_client` for `wrap_primitive`. Reuse that exact fixture style. (If a shared helper/fixture exists in `tests/rlm/conftest.py`, use it.)

- [ ] **Step 2: Write the test**

Construct a ctx with a capturing `emit` (append to a list) and `current_iteration=0`. Wrap a fake `verify_against_rubric` (returns `{"overall_score": 0.9, "target_score": 0.7, "areas": []}`) via `binding.wrap_primitive("verify_against_rubric", fake, ctx)`. Build a `tools` dict `{"verify_against_rubric": {"tool": wrapped}, ... other stages as raw mocks ...}` and drive `drive_lifecycle_chain(..., start_stage="need_verification")` (verify-only) so exactly one step runs. Assert:
```python
    rubric_events = [e for e in emitted if e.get("event") == "rubric_score"]
    assert rubric_events, emitted
    assert rubric_events[0]["iteration"] >= 1   # NOT 0 — the keystone reached binding
    assert ctx.latest_rubric_iteration >= 1
```
Match the real `RunContext` field names from Step 1. If constructing a full RunContext is heavy, import and instantiate the real `RunContext` with a tmp project_dir and a real/So fake `cost_ledger` per the binding test's pattern.

- [ ] **Step 3: Run it — verify it PASSES with Task 1's change** (and would fail at 0 without it). If the driver's `need_verification` start_stage advances the counter (it calls `_step` for verify), `iteration` should be 1.

`export OPENRESEARCH_MIN_DISK_GB=0 && <VENV> -m pytest tests/rlm/test_lifecycle_binding_parity.py -v`

- [ ] **Step 4: If the wiring is too heavy to construct a real ctx**, report DONE_WITH_CONCERNS with what blocked it rather than faking the assertion. The unit tests in Task 1 already prove the counter advance; this test is the end-to-end confirmation.

- [ ] **Step 5: Commit**

```bash
git add tests/rlm/test_lifecycle_binding_parity.py
git commit -m "Add binding-integration parity test: lifecycle rubric events carry a real iteration"
```

---

## Task 5: OFF byte-identical confirmation + full lifecycle regression

**Files:** none (verification + a guard test).

- [ ] **Step 1: Confirm the completion path is untouched (OFF byte-identical)**

`export OPENRESEARCH_MIN_DISK_GB=0 && <VENV> -m pytest tests/rlm -q -k "lifecycle or run_lifecycle or binding or sse_bridge"`
Expected: all pass. The only production behavior change is that lifecycle mode now advances `ctx.current_iteration` — the completion path (root loop) is unchanged because it advances the counter via its own logger as before.

- [ ] **Step 2: Full lifecycle suite**

`export OPENRESEARCH_MIN_DISK_GB=0 && <VENV> -m pytest tests/rlm/test_lifecycle_driver.py tests/rlm/test_lifecycle_driver_hardening.py tests/rlm/test_run_lifecycle_primary.py tests/rlm/test_run_lifecycle_drive.py tests/rlm/test_lifecycle_primary_inputs.py tests/rlm/test_binding_lifecycle_ledger.py tests/rlm/test_lifecycle_ledger.py tests/rlm/test_lifecycle_iteration_counter.py tests/rlm/test_lifecycle_primary_iterations_finalize.py tests/rlm/test_lifecycle_binding_parity.py -q`
Expected: ≥ baseline 168 + the new tests, all green.

- [ ] **Step 3: No commit** (verification only). If Step 1/2 surface a regression, fix in the owning task and re-run.

---

## Self-review checklist (run after writing code)

- Spec coverage: keystone (iteration counter) = Task 1; finalize source = Task 2; improve observability = Task 3; parity proof = Task 4; OFF-identical = Task 5. Deferred items are explicitly out of scope (top of plan).
- Type consistency: `_resolve_run_iterations(primary_active, summary, logger_iterations)` signature matches its test and call site; `summary["iterations"]` written in Task 1, read in Task 2; event `phase == "improve"` matches Task 3 test.
- Placeholder scan: `<VENV>` = `/Volumes/CS_Stuff/scientific_article_generator/.venv/bin/python` (the executing session sets the real worktree path).

## Next after this plan
Spec B core landed → the Cutout (arXiv 1708.04552) A/B run on GCP (completion vs lifecycle-primary, same paper/seed) through the grader-σ gate, with the 30-min monitoring loop.
