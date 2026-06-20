# EvidenceAudit → run_experiment Wiring — Implementation Plan (Plan 2 of N)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Wire the unified `result_is_fabricated` veto into `run_experiment` via a unit-tested `apply_result_veto` helper, so that when the master flag is ON the unified critic catches the SDAR-v6 / stub / low-VRAM fabrications even if the individual guard flags are off — and byte-identical when OFF.

**Architecture:** Add one helper `apply_result_veto(result, ctx, *, peak_vram_gb, emit) -> result` to `evidence_audit.py` (fully unit-tested), then add ONE `if result.get("success"):` seam block in `run_experiment` after the existing four guards that calls it. The helper degrades a flagged result to the existing `failure_class="fabrication_suspected"` shape and emits the existing `run_warning`. The seam runs on the aggregated `result` for BOTH the monolithic and cells routes (both flow through this section), so it is wired once.

**Tech Stack:** Python (3.11 floor / 3.12 image / 3.14 dev), stdlib only in the helper, pytest, ruff.

## Global Constraints

- **Red line:** the critic reads deterministic evidence, never an LLM grade.
- **Byte-identical when OFF:** `result_is_fabricated` returns `None` when `OPENRESEARCH_EVIDENCE_AUDIT` is unset, so `apply_result_veto` returns the input `result` object unchanged and emits nothing.
- **Fail-soft:** `apply_result_veto` never raises; the `emit` callback's exceptions are swallowed.
- **Deferred import at the seam:** import `apply_result_veto` INSIDE the `run_experiment` seam block (`# noqa: PLC0415`), matching the existing guard blocks in that function.
- **Minimal blast radius:** do NOT modify or remove the existing four guards (VRAM antifab, stub, zero-metrics, metric-semantics). ADD the seam after them.
- **Lint:** `uvx ruff@0.15.16 check backend/agents/rlm/evidence_audit.py backend/agents/rlm/primitives.py` clean.
- **Test location:** `tests/rlm/test_evidence_audit.py` (append).
- **Commit policy:** standing authorization to make ONE milestone commit at plan end (no push). The implementer does NOT commit — the controller commits after the inline diff review.
- **Test command:** `.venv/bin/python -m pytest tests/rlm/test_evidence_audit.py -v` then `.venv/bin/python -m pytest tests/rlm/ -q`.

---

### Task 1: `apply_result_veto()` helper + unit tests

**Files:**
- Modify: `backend/agents/rlm/evidence_audit.py` (add `Callable` import; add `apply_result_veto` after `result_is_fabricated`)
- Test: `tests/rlm/test_evidence_audit.py` (append)

**Interfaces:**
- Consumes: `result_is_fabricated` (Plan 1).
- Produces: `apply_result_veto(result: Any, ctx: Any, *, peak_vram_gb: float | None = None, emit: Callable[[str], None] | None = None) -> Any`. Returns the input `result` unchanged when not flagged; otherwise a NEW dict `{**result, "success": False, "failure_class": "fabrication_suspected", "error": <reason>}`, calling `emit(reason)` (failures swallowed).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/rlm/test_evidence_audit.py

# ---------------------------------------------------------------------------
# Plan 2: apply_result_veto()
# ---------------------------------------------------------------------------

_SDAR_V6 = {"success": True, "metrics": {"per_model": {"Qwen/Qwen3-1.7B": {"alfworld": {"sdar":
           {"status": "ok", "device": "cuda", "success_rate": 0.0, "reward": 0.0}}}}}}


def test_apply_result_veto_unchanged_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENRESEARCH_EVIDENCE_AUDIT", raising=False)
    from backend.agents.rlm.evidence_audit import apply_result_veto
    src = dict(_SDAR_V6)
    out = apply_result_veto(src, _ctx(tmp_path))
    assert out is src  # byte-identical: same object returned, untouched


def test_apply_result_veto_degrades_when_flag_on(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import apply_result_veto
    emitted = []
    out = apply_result_veto(dict(_SDAR_V6), _ctx(tmp_path), emit=emitted.append)
    assert out["success"] is False
    assert out["failure_class"] == "fabrication_suspected"
    assert out["error"]
    assert emitted and ("zero" in emitted[0].lower() or "constant" in emitted[0].lower())


def test_apply_result_veto_clean_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import apply_result_veto
    clean = {"success": True, "metrics": {"per_model": {"m": {"e": {"b":
            {"status": "ok", "accuracy": 0.83}}}}}}
    out = apply_result_veto(clean, _ctx(tmp_path))
    assert out is clean  # identity: untouched


def test_apply_result_veto_preserves_other_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import apply_result_veto
    src = dict(_SDAR_V6, logs="x", wall_time_s=1.5)
    out = apply_result_veto(src, _ctx(tmp_path))
    assert out["success"] is False
    assert out["logs"] == "x" and out["wall_time_s"] == 1.5


def test_apply_result_veto_emit_failsoft(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import apply_result_veto

    def bad_emit(_msg):
        raise RuntimeError("boom")

    out = apply_result_veto(dict(_SDAR_V6), _ctx(tmp_path), emit=bad_emit)
    assert out["success"] is False  # degraded despite emit failure
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/rlm/test_evidence_audit.py -k apply_result_veto -v`
Expected: FAIL — `ImportError: cannot import name 'apply_result_veto'`.

- [ ] **Step 3: Write the minimal implementation**

Change the import line in `backend/agents/rlm/evidence_audit.py`:
```python
from collections.abc import Callable
from typing import Any
```
(add `from collections.abc import Callable` next to the existing `from typing import Any`, grouped at the top.)

Add after `result_is_fabricated`:
```python
def apply_result_veto(
    result: Any,
    ctx: Any,
    *,
    peak_vram_gb: float | None = None,
    emit: Callable[[str], None] | None = None,
) -> Any:
    """Degrade a result the unified critic flags to a repairable fabrication_suspected
    failure (mirrors the existing per-guard pattern in run_experiment). Returns the input
    ``result`` unchanged when not flagged — byte-identical when the master flag is off
    (``result_is_fabricated`` returns None). ``emit`` (if given) is called with the reason
    for a run_warning; emit failures are swallowed (diagnostics must never break a run)."""
    reason = result_is_fabricated(result, ctx, peak_vram_gb=peak_vram_gb)
    if reason is None:
        return result
    degraded = {
        **result,
        "success": False,
        "failure_class": "fabrication_suspected",
        "error": reason,
    }
    if emit is not None:
        try:
            emit(reason)
        except Exception:
            pass
    return degraded
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/rlm/test_evidence_audit.py -v`
Expected: PASS (Plan 1's 27 + 5 new = 32).

Run: `uvx ruff@0.15.16 check backend/agents/rlm/evidence_audit.py`
Expected: clean.

---

### Task 2: Wire the seam into `run_experiment`

**Files:**
- Modify: `backend/agents/rlm/primitives.py` (insert one block in `run_experiment`, after the metric-semantics guard, before the finalize-on-timeout block)

**Interfaces:**
- Consumes: `apply_result_veto` (Task 1); in-scope locals `result`, `ctx`, `_antifab_peak_vram_gb`, and `_emit_dashboard_event`.
- Produces: no new public symbol — behavior change is gated entirely by the master flag inside `result_is_fabricated`.

**Context:** In `backend/agents/rlm/primitives.py`, `run_experiment` runs four success-gated fabrication guards in sequence (VRAM antifab, stub, zero-metrics, metric-semantics), each degrading to `failure_class="fabrication_suspected"`. Immediately after the LAST one (the metric-semantics guard) and BEFORE the `# Finalize-on-timeout` block, add the unified seam. `_antifab_peak_vram_gb` is the monolithic-path peak (None on the cells route → the VRAM sub-check no-ops there, which is correct; the cells route has its own per-cell VRAM check, and the stub/zero checks still apply to the aggregated metrics).

- [ ] **Step 1: Add the seam via Edit**

In `backend/agents/rlm/primitives.py`, find this exact text (the end of the metric-semantics guard, immediately followed by the finalize-on-timeout comment):

```python
        except Exception:  # noqa: BLE001 — the metric-semantics guard must never crash the run
            logger.debug("run_experiment: metric-semantics guard failed", exc_info=True)

    # Finalize-on-timeout (2026-06-08): a timed-out / stalled experiment must SCORE its
```

Replace it with (the same two anchor lines, the new seam inserted before the comment):

```python
        except Exception:  # noqa: BLE001 — the metric-semantics guard must never crash the run
            logger.debug("run_experiment: metric-semantics guard failed", exc_info=True)

    # Unified evidence-audit veto (Pillar-1 wiring, spec 2026-06-20 §3). Flag-gated INSIDE
    # result_is_fabricated (OPENRESEARCH_EVIDENCE_AUDIT, default OFF → returns None → this is
    # byte-identical). When ON, the unified critic catches the SDAR-v6 all-0.0-real-keys / stub
    # / low-VRAM fabrications even if the individual guard flags above are off — one master
    # switch. Re-checks success (a prior guard may have flipped it). Fail-soft in the helper.
    if result.get("success"):
        from backend.agents.rlm.evidence_audit import apply_result_veto  # noqa: PLC0415
        result = apply_result_veto(
            result,
            ctx,
            peak_vram_gb=_antifab_peak_vram_gb,
            emit=lambda _msg: _emit_dashboard_event(
                ctx,
                event_type="run_warning",
                payload={"code": "fabrication_suspected", "message": _msg},
            ),
        )

    # Finalize-on-timeout (2026-06-08): a timed-out / stalled experiment must SCORE its
```

- [ ] **Step 2: Lint**

Run: `uvx ruff@0.15.16 check backend/agents/rlm/primitives.py`
Expected: clean.

- [ ] **Step 3: Full-suite regression (byte-identical proof)**

Run: `.venv/bin/python -m pytest tests/rlm/ -q`
Expected: PASS — same counts as before the change (the seam is inert with the master flag off; the existing guard + run_matrix tests are unaffected).

**Note on wiring coverage:** the one-line seam in `run_experiment` is verified by inspection + the full-suite regression, NOT a dedicated behavior test — consistent with the existing monolithic guards, which are also covered at the predicate/helper level (`apply_result_veto`'s 5 unit tests) plus the cells-route integration in `test_antifab_guard.py::TestVramGuard`. A full `run_experiment` integration test (sandbox + code dir) is out of scope here; the final Codex review is the backstop.

---

### Task 3: Milestone commit

- [ ] **Step 1: Controller reviews the diff** (`git --no-pager diff` on `evidence_audit.py`, `primitives.py`, `tests/rlm/test_evidence_audit.py`).

- [ ] **Step 2: Commit (controller, standing authorization — no push)**

```bash
git add backend/agents/rlm/evidence_audit.py backend/agents/rlm/primitives.py \
        tests/rlm/test_evidence_audit.py \
        docs/superpowers/plans/2026-06-20-evidence-audit-run-experiment-wiring.md
git commit -m "Wire the unified evidence-audit veto into run_experiment (flag-gated)

Adds apply_result_veto() to evidence_audit.py and a single success-gated seam in
run_experiment after the existing four fabrication guards. Gated inside
result_is_fabricated by OPENRESEARCH_EVIDENCE_AUDIT (default OFF → byte-identical);
when ON the unified critic catches the SDAR-v6 all-0.0-real-keys / stub / low-VRAM
fabrications even if the individual guard flags are off. Covers both the monolithic
and cells routes (both flow through this section). 5 new unit tests; tests/rlm/ green."
```

Do NOT push.

---

## Self-Review

**1. Spec coverage:** Wires Pillar-1's `result_is_fabricated` into the live `run_experiment` path behind the master flag (spec §3.2 "run_experiment calls result_is_fabricated"). Byte-identical-when-off invariant preserved (§9 Phase 1).

**2. Placeholder scan:** None. Every step has complete code; the Edit anchor is an exact, unique block quoted from the file.

**3. Type consistency:** `apply_result_veto(result, ctx, *, peak_vram_gb=None, emit=None)` matches the seam call (`peak_vram_gb=_antifab_peak_vram_gb`, `emit=lambda _msg: ...`). `result_is_fabricated` signature reused exactly from Plan 1. `_emit_dashboard_event`, `_antifab_peak_vram_gb`, `result`, `ctx` confirmed in scope at the insertion point (lines ~6536, ~6242, throughout `run_experiment`).
