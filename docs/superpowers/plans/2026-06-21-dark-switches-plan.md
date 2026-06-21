# Dark-Switches Implementation Plan (T1/T6 — flip the built-but-OFF machinery)

> **For agentic workers:** REQUIRED SUB-SKILL — use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax.

**Goal:** Realize the immediate cost + reliability wins from the system-improvement
research (`docs/superpowers/specs/2026-06-21-system-improvement-opportunities.md`, themes
T1/T6) by shipping the pure-win mechanism fixes hermetically NOW, and readying the
behavior-changing default-flips for operator A/B validation.

**Architecture:** Two phases by validation-need. **Phase 1** = pure correctness/reliability
fixes with no quality tradeoff → implement + hermetic test + ship (default-ON where it's a
flag). **Phase 2** = switches that change run *behavior/quality* → implement/wire the
mechanism, keep the default OFF, and hand the operator a ready-to-run paired A/B (repo rule:
**≥3 paired A/B runs before flipping any default**). No GPU/live-API spend in this plan.

**Tech stack:** Python 3.12, pytest (socket-hermetic), `OPENRESEARCH_*` env flags, the
existing A/B harness (`experiment_arm`, `scripts/ab_compare.py`).

**Branch:** `feat/dark-switches` (off the trunk `feat/bes-conversion-correctness`), kept
**separate from `main`** per standing operator constraint.

**Guardrails (every task):** default-OFF→ON flips only where Phase-1-eligible; unset must
stay byte-identical for Phase-2; fail-soft; TDD red→green; full hermetic suite green before
commit (`OPENRESEARCH_MIN_DISK_GB=0 .venv/bin/python -m pytest tests/ -q`); ruff clean.

---

# PHASE 1 — Ship now (pure wins, hermetic)

### Task 1: Re-run AST preflight after a patch-mode write (close the patch-churn hole)

**Why:** `implement_baseline` patch-mode applies a diff and writes `train.py` back with NO
re-validation (`primitives.py`, after the `_os.replace(_tmp, _train_py_path)` in the
`_patch_success` branch). A patch can apply cleanly yet not fix (or re-introduce) the
targeted violation → a GPU dispatch is burned to rediscover it. Re-scan; if the targeted
violations persist, fall through to the full rewrite instead of returning success.

**Files:**
- Modify: `backend/agents/rlm/primitives.py` (the `_patch_success` branch in
  `implement_baseline`, anchor: the `_os.replace(_tmp, _train_py_path)` + the immediately
  following `return str(ctx.project_dir / "code")`).
- Test: `tests/rlm/test_patch_repreflight.py` (new).

- [ ] **Step 1: Write the failing test**
```python
# tests/rlm/test_patch_repreflight.py
"""Patch-mode must re-validate: a patch that applies but doesn't clear the targeted
preflight violation must NOT return success (it should fall through to full rewrite)."""
from pathlib import Path
import backend.agents.rlm.primitives as prim


def test_patch_that_leaves_violation_does_not_return_success(monkeypatch, tmp_path):
    code = tmp_path / "code"; code.mkdir()
    (code / "train.py").write_text("import torch\nx = undefined_symbol\n")
    # Patch "succeeds" textually but the violation persists on re-scan.
    monkeypatch.setattr(prim, "_apply_patch_mode", lambda *a, **k: (True, "import torch\nx = still_undefined\n"), raising=False)
    # scan still reports a violation after the patch
    monkeypatch.setattr(prim.preflight_ast, "scan_code_dir",
                        lambda d: [{"kind": "undefined_name", "file": "train.py", "line": 2}])
    # The unit under test is the re-preflight gate helper (Step 3 extracts it):
    persisted = prim._patch_left_violations(str(code), targeted=[{"kind": "undefined_name"}])
    assert persisted is True
```

- [ ] **Step 2: Run it — expect FAIL** (`_patch_left_violations` undefined)
Run: `OPENRESEARCH_MIN_DISK_GB=0 .venv/bin/python -m pytest tests/rlm/test_patch_repreflight.py -q`
Expected: FAIL (AttributeError / not defined).

- [ ] **Step 3: Implement** a small helper + wire it into the `_patch_success` branch.
```python
# primitives.py — new helper near the other patch helpers
def _patch_left_violations(code_dir: str, targeted: list[dict]) -> bool:
    """True if any of the targeted preflight violation kinds still appear after the patch."""
    try:
        remaining = preflight_ast.scan_code_dir(code_dir)
    except Exception:  # noqa: BLE001 — re-scan must never crash the patch path
        return False  # fail-soft: don't block on a scan error
    remaining_kinds = {v.get("kind") for v in remaining}
    return any(t.get("kind") in remaining_kinds for t in (targeted or []))
```
Then in the `_patch_success` branch, AFTER `_os.replace(_tmp, _train_py_path)` and BEFORE
`return str(ctx.project_dir / "code")`:
```python
            if _patch_left_violations(str(ctx.project_dir / "code"), _violations):
                logger.warning(
                    "implement_baseline[%s]: patch applied but targeted violations persist "
                    "— falling through to full rewrite", ctx.project_id)
                _patch_success = False  # fall into the existing else/full-rewrite path
            else:
                return str(ctx.project_dir / "code")
```
(Confirm `_violations` is the in-scope targeted-violation list at that point; if named
differently, bind to the actual variable.)

- [ ] **Step 4: Run the test — expect PASS.**
- [ ] **Step 5: Run touched-area tests** (`test_baseline_implementation*`, `test_preflight_ast*`) + ruff.
- [ ] **Step 6: Commit** `feat(implement): re-run AST preflight after patch-mode; don't ship an unfixed patch`.

### Task 2: Orphan-guard default-ON (stop VRAM zombies starving retries)

**Why:** `orphan_guard_enabled()` (`orphan_guard.py`) defaults OFF; the kill is already
fail-soft per-pgid. An abandoned training subprocess holds a full GPU until pod teardown,
starving the retry the timeout authorized. Pure reliability → default ON, opt-OUT.

**Files:** Modify `backend/agents/rlm/orphan_guard.py` (`orphan_guard_enabled`). Test:
`tests/rlm/test_orphan_guard.py` (extend or new).

- [ ] **Step 1: Failing test** — assert unset ⇒ enabled, and explicit `0/false/off` ⇒ disabled.
```python
def test_orphan_guard_default_on(monkeypatch):
    from backend.agents.rlm import orphan_guard
    monkeypatch.delenv("OPENRESEARCH_ORPHAN_GUARD", raising=False)
    assert orphan_guard.orphan_guard_enabled() is True          # NEW default
    monkeypatch.setenv("OPENRESEARCH_ORPHAN_GUARD", "0")
    assert orphan_guard.orphan_guard_enabled() is False
    monkeypatch.setenv("OPENRESEARCH_ORPHAN_GUARD", "off")
    assert orphan_guard.orphan_guard_enabled() is False
```
- [ ] **Step 2: Run — FAIL** (currently default OFF).
- [ ] **Step 3: Implement** — invert the default:
```python
def orphan_guard_enabled() -> bool:
    """True unless OPENRESEARCH_ORPHAN_GUARD explicitly opts OUT (default ON 2026-06-21)."""
    return os.environ.get("OPENRESEARCH_ORPHAN_GUARD", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )
```
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Suite slice + ruff.**
- [ ] **Step 6: Commit** `feat(reliability): orphan-guard default-ON (kill abandoned GPU subprocs; opt-out)`.

### Task 3: Import preflight-smoke default-ON for cost-bearing sandboxes

**Why:** `preflight_smoke` runs an import-resolution test on CPU (`CUDA_VISIBLE_DEVICES=""`),
catching the whole `ModuleNotFoundError` class before a paid pod boots; false-positive rate
"essentially zero". Default-OFF today (`OPENRESEARCH_PREFLIGHT_SMOKE`). Make it ON when the
sandbox bills money (runpod/brev), opt-OUT.

**Files:** Modify `backend/agents/rlm/preflight_smoke.py` (find the `_enabled`/gate function —
the docstring is at top; the gate reads `OPENRESEARCH_PREFLIGHT_SMOKE`). Test:
`tests/rlm/test_preflight_smoke.py`.

- [ ] **Step 1: Failing test** — cost sandbox ⇒ enabled by default; local ⇒ unchanged (OFF);
  explicit `0` ⇒ off everywhere.
```python
def test_preflight_smoke_default_on_for_cost_sandbox(monkeypatch):
    from backend.agents.rlm import preflight_smoke as ps
    monkeypatch.delenv("OPENRESEARCH_PREFLIGHT_SMOKE", raising=False)
    assert ps.preflight_smoke_enabled(sandbox_mode="runpod") is True   # NEW
    assert ps.preflight_smoke_enabled(sandbox_mode="local") is False   # unchanged
    monkeypatch.setenv("OPENRESEARCH_PREFLIGHT_SMOKE", "0")
    assert ps.preflight_smoke_enabled(sandbox_mode="runpod") is False  # explicit opt-out
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — make the gate sandbox-aware (add the `sandbox_mode` param if the
  current signature lacks it; thread the caller's `ctx.sandbox_mode`):
```python
_COST_SANDBOXES = {"runpod", "brev"}
def preflight_smoke_enabled(sandbox_mode: str | None = None) -> bool:
    raw = os.environ.get("OPENRESEARCH_PREFLIGHT_SMOKE", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return (sandbox_mode or "").strip().lower() in _COST_SANDBOXES  # unset ⇒ cost-sandbox default
```
  Update the call site to pass `ctx.sandbox_mode`.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Suite slice + ruff.**
- [ ] **Step 6: Commit** `feat(cost): preflight import-smoke default-ON on runpod/brev; opt-out`.

### Task 4 (Phase 1 gate): full suite + lint
- [ ] Run `OPENRESEARCH_MIN_DISK_GB=0 .venv/bin/python -m pytest tests/ -q` → 0 failures.
- [ ] `uvx ruff@0.15.16 check .` → clean.
- [ ] Push `feat/dark-switches`; open PR to the **trunk** (NOT main).

---

# PHASE 2 — Wire + A/B-gate (default stays OFF; operator validates)

Each Phase-2 switch is a behavior/quality change → per repo rule it needs ≥3 paired A/B runs
before the default flips. Implement/verify the mechanism + a hermetic test that the mechanism
works when the flag is ON, keep the default OFF, and the operator runs the A/B. **Do NOT flip
these defaults in code.**

### Task 5: Cell-resume — make it trivially A/B-able + add the resume hermetic test
- Mechanism exists (`cell_scheduler.should_skip_cell`, `OPENRESEARCH_RESUME_CELLS`). Add a
  hermetic test proving an ok+fingerprint-matched cell is skipped with no subprocess when the
  flag is ON. Keep default OFF. (Anchor: `cell_scheduler.py:235`.)
- Deliver the A/B command (below) in the handoff.

### Task 6: Dead-training early-stop — hermetic test of the detector on synthetic loss
- `dead_training_guard.DeadTrainingDetector` — add/confirm a unit test feeding a flat loss
  series trips `training_diverged`, and a healthy descending series does not (false-positive
  guard). Keep `OPENRESEARCH_DEAD_LOSS_EARLYSTOP` default OFF. (Anchor: `dead_training_guard.py:65`.)

### Task 7: OOM hard-memcap — hermetic test of the sitecustomize shim emission
- Confirm `OPENRESEARCH_OOM_ENFORCE` emits the `set_per_process_memory_fraction` shim and the
  batch-scale ladder; unit-test the emitted shim content. Default OFF. (Anchor: `gpu_cell_runner.py:102`.)

### Task 8: HF/dataset cache persistence — wire + hermetic payload test
- `runpod_backend` cache mounts are gated on `network_volume_id` (default `""`). Add a config
  knob to auto-attach a configured volume; hermetic test the pod-create payload includes the
  cache env when a volume id is set. **Provisioning a real volume = operator.** (Anchors:
  `runpod_backend.py:709`, `config.py:282`.)

### Task 9: Spot/interruptible GPUs (the M-effort $ giant)
- Add `interruptible` + bid to the pod-create payload and handle the preempt→requeue signal in
  the escalation/retry loop (resume scaffolding already at `gpu_cell_runner.py:483`). Behind
  `OPENRESEARCH_RUNPOD_INTERRUPTIBLE` (default OFF). Hermetic test the payload + the
  preempt-handling branch. (Anchor: `runpod_backend.py:727`.) Operator validates on a real pod.

### Task 10 (Phase 2 gate): full suite + lint + update the handoff with per-switch A/B status.

---

## Self-review checklist (run after writing/executing)
- Every Phase-1 flip: unset behaves as the NEW safe default; explicit opt-out works; hermetic test covers both.
- Every Phase-2 switch: unset is byte-identical to today; the mechanism is tested ON; the default is NOT flipped in code.
- No GPU/live-API spend anywhere; full suite green; ruff clean; branch separate from main.
