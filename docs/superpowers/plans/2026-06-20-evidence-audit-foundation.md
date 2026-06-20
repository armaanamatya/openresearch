# EvidenceAudit Foundation — Implementation Plan (Plan 1 of N)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `backend/agents/rlm/evidence_audit.py` — the single deterministic evidence critic that *composes* (never reimplements) the existing fabrication-predicate modules into one aggregate, dormant by default so the change is byte-identical.

**Architecture:** A new leaf module with one master flag (`OPENRESEARCH_EVIDENCE_AUDIT`, default OFF), one frozen snapshot dataclass (`EvidenceAudit`), one per-result veto seam (`result_is_fabricated`), and one run-level disk/ledger snapshot (`audit_evidence`). It calls the *pure* predicates from `zero_metrics_detection`, `stub_detection`, `gpu_cell_runner` (VRAM), `evidence_key`, plus `report.run_experiment_success_count` and `external_validator.check_provenance_present`, via **deferred imports** (the codebase pattern for avoiding cycles). NO consumer is rewired in this plan — the module ships dormant and fully tested; wiring is Plan 2.

**Tech Stack:** Python (3.11 floor / 3.12 image / 3.14 dev venv), stdlib only in the new module, pytest, ruff.

## Global Constraints

- **Red line:** this module is the deterministic evidence signal; it must NEVER read or return an LLM grade.
- **Fail-soft:** every predicate composition is wrapped so a bug in the critic returns the all-pass/safe value and NEVER raises into a run. (`audit_evidence` returns an all-true `EvidenceAudit`; `result_is_fabricated` returns `None`.)
- **Byte-identical when OFF:** `result_is_fabricated` returns `None` unless `OPENRESEARCH_EVIDENCE_AUDIT` is truthy. `audit_evidence` is a read-only pure snapshot (always safe to compute; consumers gate on the flag — but no consumer exists in this plan).
- **Deferred cross-module imports:** all imports of `report`, `leaf_scorer`, `external_validator`, `zero_metrics_detection`, `stub_detection`, `gpu_cell_runner`, `evidence_key` happen INSIDE functions with `# noqa: PLC0415` (matches `run.py` pattern; avoids import cycles — `report` imports heavily).
- **Flag parsing:** `os.environ.get("OPENRESEARCH_EVIDENCE_AUDIT", "").strip().lower() in ("1","true","yes","on")` — identical to the sibling guard flags.
- **Lint:** `uvx ruff@0.15.16 check backend/agents/rlm/evidence_audit.py` clean (E4/E7/E9/F).
- **Test location:** `tests/rlm/test_evidence_audit.py` (mirror the existing `tests/rlm/test_zero_metrics_detection.py`; if that file lives under `tests/agents/rlm/` instead, match it — confirm with `ls tests/rlm/test_zero_metrics_detection.py tests/agents/rlm/test_zero_metrics_detection.py 2>/dev/null` and use the directory that resolves).
- **Commit policy (user preference overrides the skill's per-task commits):** NO per-task commits. Each task ends at the green-test gate. ONE milestone commit at the end (Task 5), made only on the user's go; the final Codex code-review runs after all pillars. Never push unless asked (and only to the `deepinvent` remote).
- **Test command:** `.venv/bin/python -m pytest tests/rlm/test_evidence_audit.py -v` (suite is socket-hermetic; all tests here are local file/monkeypatch only).

---

### Task 1: Module scaffold — `evidence_audit_enabled()` + `_provenance_on_disk()`

**Files:**
- Create: `backend/agents/rlm/evidence_audit.py`
- Test: `tests/rlm/test_evidence_audit.py`

**Interfaces:**
- Consumes: nothing (leaf).
- Produces: `evidence_audit_enabled() -> bool`; `_provenance_on_disk(ctx) -> bool` (private helper; reused by Tasks 3–4).

- [ ] **Step 1: Write the failing tests**

```python
# tests/rlm/test_evidence_audit.py
from types import SimpleNamespace


def test_evidence_audit_disabled_by_default(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EVIDENCE_AUDIT", raising=False)
    from backend.agents.rlm.evidence_audit import evidence_audit_enabled
    assert evidence_audit_enabled() is False


def test_evidence_audit_enabled_truthy(monkeypatch):
    from backend.agents.rlm.evidence_audit import evidence_audit_enabled
    for v in ("1", "true", "on", "yes", "TRUE", " On "):
        monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", v)
        assert evidence_audit_enabled() is True


def test_evidence_audit_disabled_falsey(monkeypatch):
    from backend.agents.rlm.evidence_audit import evidence_audit_enabled
    for v in ("0", "false", "off", "no", ""):
        monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", v)
        assert evidence_audit_enabled() is False


def test_provenance_on_disk_true_when_file_present(tmp_path):
    from backend.agents.rlm.evidence_audit import _provenance_on_disk
    code = tmp_path / "code"
    code.mkdir()
    (code / "provenance.json").write_text('{"schema_version": 1}')
    assert _provenance_on_disk(SimpleNamespace(project_dir=tmp_path)) is True


def test_provenance_on_disk_false_when_absent(tmp_path):
    from backend.agents.rlm.evidence_audit import _provenance_on_disk
    (tmp_path / "code").mkdir()
    assert _provenance_on_disk(SimpleNamespace(project_dir=tmp_path)) is False


def test_provenance_on_disk_failsoft_on_bad_ctx():
    from backend.agents.rlm.evidence_audit import _provenance_on_disk
    assert _provenance_on_disk(SimpleNamespace()) is False  # no project_dir -> False, no raise
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/rlm/test_evidence_audit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.agents.rlm.evidence_audit'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# backend/agents/rlm/evidence_audit.py
"""Unified deterministic evidence critic.

The single aggregation point over the existing fabrication-predicate modules.
Composes them — never reimplements. The fitness signal is this deterministic
evidence layer, NEVER the LLM grade (the red line).

Master flag ``OPENRESEARCH_EVIDENCE_AUDIT`` (default OFF) -> dormant -> the
per-result veto returns None -> byte-identical to today. See
docs/superpowers/specs/2026-06-20-actor-critic-evidence-critic-redesign-design.md.
"""
from __future__ import annotations

import os
from typing import Any

_TRUE = ("1", "true", "yes", "on")


def evidence_audit_enabled() -> bool:
    """True iff OPENRESEARCH_EVIDENCE_AUDIT opts the unified critic ON. Default OFF."""
    return os.environ.get("OPENRESEARCH_EVIDENCE_AUDIT", "").strip().lower() in _TRUE


def _provenance_on_disk(ctx: Any) -> bool:
    """True iff code/provenance.json exists under the run dir. Fail-soft -> False.

    Mirrors the inline check at primitives.py:6596 (existence only; well-formedness
    is the validator's concern, not this discriminator's)."""
    try:
        code = ctx.project_dir / "code"
        return code.is_dir() and any(code.rglob("provenance.json"))
    except Exception:
        return False
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/rlm/test_evidence_audit.py -v`
Expected: PASS (6 tests).

---

### Task 2: `EvidenceAudit` dataclass + `run_level_clean`

**Files:**
- Modify: `backend/agents/rlm/evidence_audit.py`
- Test: `tests/rlm/test_evidence_audit.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `EvidenceAudit` frozen dataclass with fields `backed_by_ledger: bool`, `provenance_present: bool`, `metrics_non_degenerate: bool`, `metric_keys_real: bool`, `fingerprint: str`, `reasons: tuple[str, ...] = ()`, `rerun_agrees: bool | None = None`; property `run_level_clean: bool`.

**Design note:** `provenance_present` is recorded but is **NOT** an AND-term of `run_level_clean` (a legit CPU/non-provenance baseline must not be marked unclean — matches the current verdict gate, which never universally requires provenance). `rerun_agrees` defaults `None` (populated in Pillar 2); `None` is treated as "no disagreement."

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/rlm/test_evidence_audit.py

def _audit(**kw):
    from backend.agents.rlm.evidence_audit import EvidenceAudit
    base = dict(backed_by_ledger=True, provenance_present=True,
               metrics_non_degenerate=True, metric_keys_real=True, fingerprint="fp")
    base.update(kw)
    return EvidenceAudit(**base)


def test_run_level_clean_all_true():
    assert _audit().run_level_clean is True


def test_run_level_clean_false_when_unbacked():
    assert _audit(backed_by_ledger=False).run_level_clean is False


def test_run_level_clean_false_when_degenerate():
    assert _audit(metrics_non_degenerate=False).run_level_clean is False


def test_run_level_clean_false_when_keys_not_real():
    assert _audit(metric_keys_real=False).run_level_clean is False


def test_run_level_clean_ignores_provenance():
    # provenance absent but everything else real -> still clean
    assert _audit(provenance_present=False).run_level_clean is True


def test_run_level_clean_false_when_rerun_disagrees():
    assert _audit(rerun_agrees=False).run_level_clean is False


def test_run_level_clean_true_when_rerun_none_or_agrees():
    assert _audit(rerun_agrees=None).run_level_clean is True
    assert _audit(rerun_agrees=True).run_level_clean is True


def test_evidence_audit_is_frozen():
    import dataclasses
    import pytest
    a = _audit()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.backed_by_ledger = False  # type: ignore[misc]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/rlm/test_evidence_audit.py -k "run_level or frozen" -v`
Expected: FAIL — `ImportError: cannot import name 'EvidenceAudit'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# add to backend/agents/rlm/evidence_audit.py (after the imports, before the functions)
from dataclasses import dataclass  # noqa: E402  (kept with the other top-level imports in practice)


@dataclass(frozen=True)
class EvidenceAudit:
    """Run-level deterministic evidence snapshot. Built from on-disk state by
    ``audit_evidence``; consumed by the verdict gate, recipe admission, and the
    validator. ``run_level_clean`` is the ONE run-level evidence predicate."""

    backed_by_ledger: bool
    provenance_present: bool
    metrics_non_degenerate: bool
    metric_keys_real: bool
    fingerprint: str
    reasons: tuple[str, ...] = ()
    rerun_agrees: bool | None = None  # populated in Pillar 2; None == no disagreement

    @property
    def run_level_clean(self) -> bool:
        return (
            self.backed_by_ledger
            and self.metrics_non_degenerate
            and self.metric_keys_real
            and (self.rerun_agrees is None or self.rerun_agrees)
        )
```

(Place `from dataclasses import dataclass` and `from typing import Any` together at the top of the file; the inline `# noqa` annotation above is illustrative — keep all module-level imports grouped at the top to satisfy ruff.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/rlm/test_evidence_audit.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

---

### Task 3: `result_is_fabricated()` — the per-result veto seam

**Files:**
- Modify: `backend/agents/rlm/evidence_audit.py`
- Test: `tests/rlm/test_evidence_audit.py`

**Interfaces:**
- Consumes: `evidence_audit_enabled`, `_provenance_on_disk` (Task 1). Pure predicates `looks_like_zero_metrics`, `zero_metrics_repair_message` (`zero_metrics_detection`), `looks_like_stub_metrics`, `stub_repair_message` (`stub_detection`), `metrics_claim_gpu_training`, `vram_evidence_verdict` (`gpu_cell_runner`).
- Produces: `result_is_fabricated(result: Any, ctx: Any, *, peak_vram_gb: float | None = None) -> str | None`. Returns a repair-reason string when fabrication is suspected, else `None`. Returns `None` when the master flag is OFF (byte-identical) or `result` is not a successful dict.

**Order of checks (cheapest-first; stub before zero so placeholder-only keys are named precisely):** stub → zero-metrics(+gpu_claim, no provenance) → VRAM(+gpu_claim).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/rlm/test_evidence_audit.py

def _ctx(tmp_path):
    return SimpleNamespace(project_dir=tmp_path)


def test_result_is_fabricated_none_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENRESEARCH_EVIDENCE_AUDIT", raising=False)
    from backend.agents.rlm.evidence_audit import result_is_fabricated
    result = {"success": True, "metrics": {"per_model": {"m": {"e": {"b":
             {"status": "ok", "device": "cuda", "reward": 0.0, "success_rate": 0.0}}}}}}
    assert result_is_fabricated(result, _ctx(tmp_path)) is None  # dormant


def test_result_is_fabricated_none_when_not_success(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import result_is_fabricated
    assert result_is_fabricated({"success": False, "metrics": {}}, _ctx(tmp_path)) is None
    assert result_is_fabricated("not a dict", _ctx(tmp_path)) is None


def test_result_is_fabricated_vetoes_sdar_v6_zero_metrics(monkeypatch, tmp_path):
    # The SDAR-v6 hallucination: real metric keys, all 0.0, GPU claim, NO provenance.
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import result_is_fabricated
    result = {"success": True, "metrics": {"per_model": {"Qwen/Qwen3-1.7B": {"alfworld": {"sdar":
             {"status": "ok", "device": "cuda", "success_rate": 0.0, "reward": 0.0}}}}}}
    reason = result_is_fabricated(result, _ctx(tmp_path))
    assert reason is not None
    assert "zero" in reason.lower() or "constant" in reason.lower()


def test_result_is_fabricated_vetoes_stub(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import result_is_fabricated
    result = {"success": True, "metrics": {"total_length": 5, "chunk_count": 2}}
    assert result_is_fabricated(result, _ctx(tmp_path)) is not None


def test_result_is_fabricated_passes_clean(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import result_is_fabricated
    result = {"success": True, "metrics": {"per_model": {"m": {"e": {"b":
             {"status": "ok", "accuracy": 0.83}}}}}}
    assert result_is_fabricated(result, _ctx(tmp_path)) is None


def test_result_is_fabricated_legit_zero_with_provenance(monkeypatch, tmp_path):
    # all-zero + GPU claim BUT provenance.json present -> real 0 baseline, NOT vetoed.
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    from backend.agents.rlm.evidence_audit import result_is_fabricated
    code = tmp_path / "code"
    code.mkdir()
    (code / "provenance.json").write_text('{"schema_version": 1}')
    result = {"success": True, "metrics": {"per_model": {"m": {"e": {"b":
             {"status": "ok", "device": "cuda", "reward": 0.0, "success_rate": 0.0}}}}}}
    assert result_is_fabricated(result, _ctx(tmp_path)) is None


def test_result_is_fabricated_vetoes_low_vram(monkeypatch, tmp_path):
    # GPU claimed, non-zero metric, but peak VRAM below the 1.5 GiB floor.
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
    monkeypatch.setenv("OPENRESEARCH_ANTIFAB_GUARD", "1")
    from backend.agents.rlm.evidence_audit import result_is_fabricated
    result = {"success": True, "metrics": {"per_model": {"m": {"e": {"b":
             {"status": "ok", "device": "cuda", "accuracy": 0.42}}}}}}
    reason = result_is_fabricated(result, _ctx(tmp_path), peak_vram_gb=0.2)
    assert reason is not None and "vram" in reason.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/rlm/test_evidence_audit.py -k fabricated -v`
Expected: FAIL — `ImportError: cannot import name 'result_is_fabricated'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# add to backend/agents/rlm/evidence_audit.py

def result_is_fabricated(result: Any, ctx: Any, *, peak_vram_gb: float | None = None) -> str | None:
    """Unified per-result fabrication veto for run_experiment.

    Composes the zero-metrics, stub, and VRAM-antifab predicates into one seam,
    returning a repair-reason string when fabrication is suspected, else None.
    Dormant (returns None) unless the master flag is on -> byte-identical to today.
    Fail-soft: any internal error returns None (the critic never blocks a run on its
    own bug). Wiring into run_experiment is Plan 2 (this ships dormant + tested).
    """
    if not evidence_audit_enabled():
        return None
    if not isinstance(result, dict) or not result.get("success"):
        return None
    metrics = result.get("metrics")
    try:
        from backend.agents.rlm.zero_metrics_detection import (  # noqa: PLC0415
            looks_like_zero_metrics,
            zero_metrics_repair_message,
        )
        from backend.agents.rlm.stub_detection import (  # noqa: PLC0415
            looks_like_stub_metrics,
            stub_repair_message,
        )
        from backend.agents.rlm.gpu_cell_runner import (  # noqa: PLC0415
            metrics_claim_gpu_training,
            vram_evidence_verdict,
        )
    except Exception:
        return None

    try:
        if looks_like_stub_metrics(metrics):
            return stub_repair_message(metrics)
        gpu_claim = metrics_claim_gpu_training(metrics)
        provenance = _provenance_on_disk(ctx)
        if gpu_claim and not provenance and looks_like_zero_metrics(metrics):
            return zero_metrics_repair_message(metrics)
        if vram_evidence_verdict(peak_vram_gb, claims_gpu_training=gpu_claim):
            return (
                f"gpu training claimed but peak VRAM {peak_vram_gb:.2f} GiB "
                f"is below the fabrication floor"
            )
    except Exception:
        return None
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/rlm/test_evidence_audit.py -v`
Expected: PASS (all tasks so far).

---

### Task 4: `audit_evidence()` — run-level disk/ledger snapshot + fingerprint

**Files:**
- Modify: `backend/agents/rlm/evidence_audit.py`
- Test: `tests/rlm/test_evidence_audit.py`

**Interfaces:**
- Consumes: `_provenance_on_disk`, `EvidenceAudit` (Tasks 1–2). `run_experiment_success_count` (`report`), `looks_like_zero_metrics` (`zero_metrics_detection`), `looks_like_stub_metrics` (`stub_detection`), `evidence_key` (`evidence_key`), `_latest_metrics_path` (`leaf_scorer`, with a direct `code/metrics.json` fallback).
- Produces: `audit_evidence(ctx: Any) -> EvidenceAudit`. A read-only pure snapshot of on-disk evidence + ledger session counters. `backed_by_ledger` is `True` unless the ledger KNOWS the in-process success count is 0 (None ⇒ replay/postmortem ⇒ trust content, matching the current gate's fallback).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/rlm/test_evidence_audit.py
import json


def _ctx_dir(tmp_path, ledger_ok=None):
    ledger = None
    if ledger_ok is not None:
        ledger = SimpleNamespace(
            session_success_compatible_count=lambda agent_id: ledger_ok,
            session_call_count=lambda agent_id: max(ledger_ok, 1),
        )
    return SimpleNamespace(project_dir=tmp_path, cost_ledger=ledger)


def _write_metrics(tmp_path, metrics, provenance=False):
    code = tmp_path / "code"
    code.mkdir(exist_ok=True)
    (code / "metrics.json").write_text(json.dumps(metrics))
    if provenance:
        (code / "provenance.json").write_text('{"schema_version": 1}')


def test_audit_evidence_clean(tmp_path):
    from backend.agents.rlm.evidence_audit import audit_evidence
    _write_metrics(tmp_path, {"per_model": {"m": {"e": {"b":
                  {"status": "ok", "accuracy": 0.8}}}}}, provenance=True)
    a = audit_evidence(_ctx_dir(tmp_path, ledger_ok=1))
    assert a.backed_by_ledger and a.metrics_non_degenerate and a.metric_keys_real
    assert a.provenance_present and a.run_level_clean and a.fingerprint


def test_audit_evidence_zero_metrics_not_clean(tmp_path):
    from backend.agents.rlm.evidence_audit import audit_evidence
    _write_metrics(tmp_path, {"per_model": {"m": {"e": {"b":
                  {"status": "ok", "reward": 0.0, "success_rate": 0.0}}}}})
    a = audit_evidence(_ctx_dir(tmp_path, ledger_ok=1))
    assert a.metrics_non_degenerate is False
    assert a.run_level_clean is False
    assert any("zero" in r.lower() or "constant" in r.lower() for r in a.reasons)


def test_audit_evidence_unbacked_when_ok_count_zero(tmp_path):
    from backend.agents.rlm.evidence_audit import audit_evidence
    _write_metrics(tmp_path, {"per_model": {"m": {"e": {"b":
                  {"status": "ok", "accuracy": 0.8}}}}})
    a = audit_evidence(_ctx_dir(tmp_path, ledger_ok=0))
    assert a.backed_by_ledger is False
    assert a.run_level_clean is False


def test_audit_evidence_backed_when_no_ledger(tmp_path):
    # None ledger (replay/postmortem) -> trust content, do not fail closed.
    from backend.agents.rlm.evidence_audit import audit_evidence
    _write_metrics(tmp_path, {"per_model": {"m": {"e": {"b":
                  {"status": "ok", "accuracy": 0.8}}}}})
    a = audit_evidence(_ctx_dir(tmp_path, ledger_ok=None))
    assert a.backed_by_ledger is True


def test_audit_evidence_fingerprint_deterministic_and_sensitive(tmp_path):
    from backend.agents.rlm.evidence_audit import audit_evidence
    _write_metrics(tmp_path, {"per_model": {"m": {"e": {"b":
                  {"status": "ok", "accuracy": 0.8}}}}})
    ctx = _ctx_dir(tmp_path, ledger_ok=1)
    fp1 = audit_evidence(ctx).fingerprint
    fp2 = audit_evidence(ctx).fingerprint
    assert fp1 and fp1 == fp2
    _write_metrics(tmp_path, {"per_model": {"m": {"e": {"b":
                  {"status": "ok", "accuracy": 0.9}}}}})
    assert audit_evidence(ctx).fingerprint != fp1


def test_audit_evidence_failsoft_on_missing_dir(tmp_path):
    # No code/ dir at all -> empty metrics, no raise, backed defaults True.
    from backend.agents.rlm.evidence_audit import audit_evidence
    a = audit_evidence(_ctx_dir(tmp_path, ledger_ok=None))
    assert isinstance(a.fingerprint, str)
    assert a.provenance_present is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/rlm/test_evidence_audit.py -k audit_evidence -v`
Expected: FAIL — `ImportError: cannot import name 'audit_evidence'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# add to backend/agents/rlm/evidence_audit.py

def _load_latest_metrics(ctx: Any) -> dict:
    """Latest on-disk metrics for the run. Tries leaf_scorer._latest_metrics_path,
    falls back to code/metrics.json. Fail-soft -> {}."""
    import json  # noqa: PLC0415

    path = None
    try:
        from backend.evals.paperbench.leaf_scorer import _latest_metrics_path  # noqa: PLC0415

        path = _latest_metrics_path(ctx.project_dir)
    except Exception:
        path = None
    try:
        if path is None:
            cand = ctx.project_dir / "code" / "metrics.json"
            path = cand if cand.exists() else None
        if path is not None and path.exists():
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def audit_evidence(ctx: Any) -> EvidenceAudit:
    """Run-level deterministic evidence snapshot from on-disk state + the ledger.

    Pure read-only function (modulo ledger session counters) -> deterministic
    ``fingerprint``. Consumed by the verdict gate, recipe admission, and the
    validator. Fail-soft: any error yields the safe all-true snapshot."""
    metrics = _load_latest_metrics(ctx)

    backed = True
    try:
        from backend.agents.rlm.report import run_experiment_success_count  # noqa: PLC0415

        ok = run_experiment_success_count(ctx)
        if ok is not None:
            backed = ok >= 1
    except Exception:
        backed = True

    non_degen = True
    keys_real = True
    try:
        from backend.agents.rlm.zero_metrics_detection import looks_like_zero_metrics  # noqa: PLC0415
        from backend.agents.rlm.stub_detection import looks_like_stub_metrics  # noqa: PLC0415

        non_degen = not looks_like_zero_metrics(metrics)
        keys_real = not looks_like_stub_metrics(metrics)
    except Exception:
        non_degen = True
        keys_real = True

    fingerprint = ""
    try:
        from backend.agents.rlm.evidence_key import evidence_key  # noqa: PLC0415

        scope = metrics.get("scope") if isinstance(metrics, dict) else None
        fingerprint = evidence_key(metrics if isinstance(metrics, dict) else {}, scope)
    except Exception:
        fingerprint = ""

    reasons: list[str] = []
    if not backed:
        reasons.append("no in-process run_experiment success")
    if not non_degen:
        reasons.append("metrics all-zero/constant")
    if not keys_real:
        reasons.append("metrics keys are placeholders")

    return EvidenceAudit(
        backed_by_ledger=backed,
        provenance_present=_provenance_on_disk(ctx),
        metrics_non_degenerate=non_degen,
        metric_keys_real=keys_real,
        fingerprint=fingerprint,
        reasons=tuple(reasons),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/rlm/test_evidence_audit.py -v`
Expected: PASS (all tasks).

- [ ] **Step 5: Lint + full-suite regression (byte-identical proof)**

Run: `uvx ruff@0.15.16 check backend/agents/rlm/evidence_audit.py`
Expected: clean.

Run: `.venv/bin/python -m pytest tests/rlm/ -q`
Expected: PASS — no regression (the module is dormant; nothing else imports it yet).

---

### Task 5: Milestone commit (single — gated on user go)

**Files:** none new.

- [ ] **Step 1: Show the diff for Opus review**

Run: `git status && git --no-pager diff --stat`
Expected: only `backend/agents/rlm/evidence_audit.py`, `tests/rlm/test_evidence_audit.py`, and the two docs (`spec`, `plan`) are new/modified.

- [ ] **Step 2: Commit (ONLY after the user approves — do not auto-commit)**

```bash
git add backend/agents/rlm/evidence_audit.py tests/rlm/test_evidence_audit.py \
        docs/superpowers/specs/2026-06-20-actor-critic-evidence-critic-redesign-design.md \
        docs/superpowers/plans/2026-06-20-evidence-audit-foundation.md
git commit -m "Add the unified EvidenceAudit deterministic critic (dormant foundation)

New backend/agents/rlm/evidence_audit.py composes the existing fabrication
predicates (zero-metrics, stub, VRAM antifab, evidence_key, ledger backing,
provenance) into one EvidenceAudit snapshot + a result_is_fabricated veto seam.
Default-OFF master flag OPENRESEARCH_EVIDENCE_AUDIT -> byte-identical; no consumer
rewired yet (Plan 2). Keystone for the actor-critic redesign spec."
```

Do NOT push (user pushes to `deepinvent` only on request). The final Codex code-review runs after all pillars land.

---

## Self-Review

**1. Spec coverage (Pillar 1 portion of §3):** `EvidenceAudit` dataclass + `run_level_clean` (Task 2 ✓); composes existing predicates not rewrites (Tasks 3–4 ✓); `result_is_fabricated` veto seam subsuming zero/stub/VRAM (Task 3 ✓); `audit_evidence` run-level snapshot + `fingerprint` reusing `evidence_key` (Task 4 ✓); master flag default-OFF byte-identical (Tasks 1,3 + Task 4 regression ✓); SDAR-v6 regression fixture (Task 3 ✓); determinism property (Task 4 ✓). `leaf_substantiated` re-export and consumer rewiring are explicitly deferred to Plan 2 (YAGNI — no consumer here). The `run_experiment` split, `finalize_pipeline`, and Pillars 2–5 are separate plans per §13.

**2. Placeholder scan:** No TBD/TODO. Every code step shows complete code. The only `...`-free; the one illustrative `# noqa` note in Task 2 Step 3 is annotated as illustrative with the real instruction (group imports at top).

**3. Type consistency:** `EvidenceAudit` field names/types identical across Tasks 2/4 and the `_audit()`/`audit_evidence` constructors. `result_is_fabricated(result, ctx, *, peak_vram_gb=None) -> str | None` and `audit_evidence(ctx) -> EvidenceAudit` match the spec §3.1 signatures. Helper `_provenance_on_disk(ctx)` consumed identically in Tasks 3–4. Composed external signatures (`zero_metrics_should_veto` not used — the pure `looks_like_zero_metrics` is, deliberately, so the master flag is the sole gate; `vram_evidence_verdict(peak_vram_gb, *, claims_gpu_training)`; `evidence_key(metrics, scope)`; `run_experiment_success_count(ctx)`) match the extracted reference.

**Note on VRAM gating:** `vram_evidence_verdict` self-gates on `OPENRESEARCH_ANTIFAB_GUARD` (default ON), so under the master flag the VRAM branch fires unless antifab is explicitly disabled. Fully subordinating it to the single master flag is a Plan 2/Phase-3 concern (§9); acceptable here because the module is dormant.
