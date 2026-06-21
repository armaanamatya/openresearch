"""Tests for the unified-critic consumer wiring (§3.2 of the grounded
self-improvement spec 2026-06-20).

Covers:
  1. ``audit_evidence_from_dir`` — ctx-free variant; delegation from ``audit_evidence``.
  2. ``_apply_evidence_gate`` with OPENRESEARCH_EVIDENCE_AUDIT ON/OFF.
  3. Default-off hard invariant: flag unset → gate is byte-identical to today.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_metrics(project_dir: Path, metrics: dict, *, provenance: bool = False) -> None:
    code = project_dir / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if provenance:
        (code / "provenance.json").write_text('{"schema_version": 1}', encoding="utf-8")


def _write_exp_runs(project_dir: Path, *, success: bool = True, metrics: dict | None = None) -> None:
    """Write a minimal experiment_runs.jsonl so _has_experiment_evidence passes."""
    row = {"success": success, "experiment_run_id": "run1", "metrics": metrics or {"accuracy": 0.9}}
    (project_dir / "experiment_runs.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def _make_report(verdict: str = "partial"):
    """Build a minimal RLMFinalReport with the given verdict."""
    from backend.agents.rlm.report import RLMFinalReport

    return RLMFinalReport(
        verdict=verdict,
        reproduction_summary="test summary",
        baseline_metrics={"accuracy": 0.9},
    )


# ---------------------------------------------------------------------------
# Task 1: audit_evidence_from_dir
# ---------------------------------------------------------------------------


class TestAuditEvidenceFromDir:
    def test_clean_evidence(self, tmp_path):
        from backend.agents.rlm.evidence_audit import audit_evidence_from_dir

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "accuracy": 0.8}}}}},
            provenance=True,
        )
        a = audit_evidence_from_dir(tmp_path, ok_count=1)
        assert a.backed_by_ledger is True
        assert a.metrics_non_degenerate is True
        assert a.metric_keys_real is True
        assert a.provenance_present is True
        assert a.run_level_clean is True
        assert a.fingerprint

    def test_degenerate_zero_metrics_not_clean(self, tmp_path):
        from backend.agents.rlm.evidence_audit import audit_evidence_from_dir

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "device": "cuda",
                                             "reward": 0.0, "success_rate": 0.0}}}}},
        )
        a = audit_evidence_from_dir(tmp_path, ok_count=1)
        assert a.metrics_non_degenerate is False
        assert a.run_level_clean is False
        assert any("zero" in r.lower() or "constant" in r.lower() for r in a.reasons)

    def test_ok_count_none_trusts_content(self, tmp_path):
        """ok_count=None (no ledger) -> backed_by_ledger=True (replay rule)."""
        from backend.agents.rlm.evidence_audit import audit_evidence_from_dir

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "accuracy": 0.8}}}}},
        )
        a = audit_evidence_from_dir(tmp_path, ok_count=None)
        assert a.backed_by_ledger is True

    def test_ok_count_zero_marks_unbacked(self, tmp_path):
        from backend.agents.rlm.evidence_audit import audit_evidence_from_dir

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "accuracy": 0.8}}}}},
        )
        a = audit_evidence_from_dir(tmp_path, ok_count=0)
        assert a.backed_by_ledger is False
        assert a.run_level_clean is False

    def test_missing_dir_failsoft(self, tmp_path):
        """No code/ dir at all -> no raise, safe defaults."""
        from backend.agents.rlm.evidence_audit import audit_evidence_from_dir

        a = audit_evidence_from_dir(tmp_path, ok_count=None)
        assert isinstance(a.fingerprint, str)
        assert a.provenance_present is False

    def test_provenance_detected(self, tmp_path):
        from backend.agents.rlm.evidence_audit import audit_evidence_from_dir

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "accuracy": 0.8}}}}},
            provenance=True,
        )
        a = audit_evidence_from_dir(tmp_path, ok_count=1)
        assert a.provenance_present is True

    def test_fingerprint_deterministic_and_sensitive(self, tmp_path):
        from backend.agents.rlm.evidence_audit import audit_evidence_from_dir

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "accuracy": 0.8}}}}},
        )
        fp1 = audit_evidence_from_dir(tmp_path, ok_count=1).fingerprint
        fp2 = audit_evidence_from_dir(tmp_path, ok_count=1).fingerprint
        assert fp1 and fp1 == fp2
        # Change metrics -> different fingerprint
        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "accuracy": 0.99}}}}},
        )
        assert audit_evidence_from_dir(tmp_path, ok_count=1).fingerprint != fp1


# ---------------------------------------------------------------------------
# Task 1 (continued): audit_evidence delegates to audit_evidence_from_dir
# ---------------------------------------------------------------------------


class TestAuditEvidenceDelegation:
    def _make_ctx(self, project_dir, ok_count=None):
        ledger = None
        if ok_count is not None:
            ledger = SimpleNamespace(
                session_success_compatible_count=lambda agent_id: ok_count,
                session_call_count=lambda agent_id: max(ok_count, 1),
            )
        return SimpleNamespace(project_dir=project_dir, cost_ledger=ledger)

    def test_ctx_and_dir_variants_agree(self, tmp_path):
        """audit_evidence(ctx) == audit_evidence_from_dir(dir, ok_count) for same state."""
        from backend.agents.rlm.evidence_audit import audit_evidence, audit_evidence_from_dir

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "accuracy": 0.8}}}}},
            provenance=True,
        )
        ctx = self._make_ctx(tmp_path, ok_count=1)
        a_ctx = audit_evidence(ctx)
        a_dir = audit_evidence_from_dir(tmp_path, ok_count=1)

        assert a_ctx.backed_by_ledger == a_dir.backed_by_ledger
        assert a_ctx.provenance_present == a_dir.provenance_present
        assert a_ctx.metrics_non_degenerate == a_dir.metrics_non_degenerate
        assert a_ctx.metric_keys_real == a_dir.metric_keys_real
        assert a_ctx.fingerprint == a_dir.fingerprint
        assert a_ctx.run_level_clean == a_dir.run_level_clean

    def test_ctx_degenerate_agrees_with_dir(self, tmp_path):
        from backend.agents.rlm.evidence_audit import audit_evidence, audit_evidence_from_dir

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "device": "cuda",
                                             "reward": 0.0, "success_rate": 0.0}}}}},
        )
        ctx = self._make_ctx(tmp_path, ok_count=1)
        a_ctx = audit_evidence(ctx)
        a_dir = audit_evidence_from_dir(tmp_path, ok_count=1)

        assert a_ctx.run_level_clean == a_dir.run_level_clean is False
        assert a_ctx.metrics_non_degenerate == a_dir.metrics_non_degenerate is False

    def test_ctx_no_ledger_agrees_with_dir_none_ok_count(self, tmp_path):
        from backend.agents.rlm.evidence_audit import audit_evidence, audit_evidence_from_dir

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "accuracy": 0.7}}}}},
        )
        ctx = self._make_ctx(tmp_path, ok_count=None)
        a_ctx = audit_evidence(ctx)
        a_dir = audit_evidence_from_dir(tmp_path, ok_count=None)

        assert a_ctx.backed_by_ledger == a_dir.backed_by_ledger is True


# ---------------------------------------------------------------------------
# Task 2/3: _apply_evidence_gate with unified critic
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Ensure both flags are cleared before each test."""
    monkeypatch.delenv("OPENRESEARCH_EVIDENCE_AUDIT", raising=False)
    monkeypatch.delenv("OPENRESEARCH_EVIDENCE_GATE", raising=False)


class TestEvidenceGateUnifiedCriticOff:
    """Flag OFF → gate is byte-identical to today (default-off invariant)."""

    def test_flag_unset_no_audit_consulted(self, tmp_path, monkeypatch):
        """With flag unset, a report with degenerate metrics still keeps its verdict
        (the audit is never consulted — the gate only looks at content evidence)."""
        from backend.agents.rlm.report import _apply_evidence_gate

        # Degenerate on-disk metrics (all-zero)
        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "device": "cuda",
                                             "reward": 0.0, "success_rate": 0.0}}}}},
        )
        # Backed by content evidence
        _write_exp_runs(tmp_path, success=True, metrics={"accuracy": 0.9})

        report = _make_report("partial")
        result = _apply_evidence_gate(report, tmp_path, run_experiment_ok_calls=1)
        # Gate sees real content evidence; audit is OFF -> verdict is NOT changed
        assert result.verdict == "partial"

    def test_flag_explicitly_off_no_audit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "0")
        from backend.agents.rlm.report import _apply_evidence_gate

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "device": "cuda",
                                             "reward": 0.0, "success_rate": 0.0}}}}},
        )
        _write_exp_runs(tmp_path, success=True, metrics={"accuracy": 0.9})

        report = _make_report("reproduced")
        result = _apply_evidence_gate(report, tmp_path, run_experiment_ok_calls=1)
        assert result.verdict == "reproduced"


class TestEvidenceGateUnifiedCriticOn:
    """Flag ON → unified critic AND-ed into verdict gate."""

    def test_degenerate_metrics_downgrades_partial(self, tmp_path, monkeypatch):
        """Content evidence present + degenerate on-disk metrics + flag ON
        -> verdict downgraded to 'failed'."""
        monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
        from backend.agents.rlm.report import _apply_evidence_gate

        # All-zero metrics (SDAR v6 hallucination shape)
        _write_metrics(
            tmp_path,
            {"per_model": {"Qwen/Qwen3-1.7B": {"alfworld": {"sdar":
                {"status": "ok", "device": "cuda", "reward": 0.0, "success_rate": 0.0}}}}},
        )
        _write_exp_runs(tmp_path, success=True, metrics={"accuracy": 0.9})

        report = _make_report("partial")
        result = _apply_evidence_gate(report, tmp_path, run_experiment_ok_calls=1)
        assert result.verdict == "failed"
        assert "evidence_audit" in result.reproduction_summary

    def test_degenerate_metrics_downgrades_reproduced(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
        from backend.agents.rlm.report import _apply_evidence_gate

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "device": "cuda",
                                             "reward": 0.0, "success_rate": 0.0}}}}},
        )
        _write_exp_runs(tmp_path, success=True, metrics={"accuracy": 0.9})

        report = _make_report("reproduced")
        result = _apply_evidence_gate(report, tmp_path, run_experiment_ok_calls=1)
        assert result.verdict == "failed"

    def test_clean_evidence_preserves_verdict(self, tmp_path, monkeypatch):
        """Flag ON + clean (non-degenerate) metrics -> verdict preserved."""
        monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
        from backend.agents.rlm.report import _apply_evidence_gate

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "accuracy": 0.83}}}}},
        )
        _write_exp_runs(tmp_path, success=True, metrics={"accuracy": 0.83})

        report = _make_report("partial")
        result = _apply_evidence_gate(report, tmp_path, run_experiment_ok_calls=1)
        assert result.verdict == "partial"

    def test_clean_evidence_reproduced_preserved(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
        from backend.agents.rlm.report import _apply_evidence_gate

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "accuracy": 0.83}}}}},
        )
        _write_exp_runs(tmp_path, success=True, metrics={"accuracy": 0.83})

        report = _make_report("reproduced")
        result = _apply_evidence_gate(report, tmp_path, run_experiment_ok_calls=1)
        assert result.verdict == "reproduced"

    def test_failed_verdict_unchanged_even_with_audit_on(self, tmp_path, monkeypatch):
        """A 'failed' verdict is never changed by the audit."""
        monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
        from backend.agents.rlm.report import _apply_evidence_gate

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "device": "cuda",
                                             "reward": 0.0, "success_rate": 0.0}}}}},
        )
        _write_exp_runs(tmp_path, success=True)

        report = _make_report("failed")
        result = _apply_evidence_gate(report, tmp_path, run_experiment_ok_calls=1)
        assert result.verdict == "failed"

    def test_audit_note_appended_on_downgrade(self, tmp_path, monkeypatch):
        """Downgrade note names the audit reason."""
        monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
        from backend.agents.rlm.report import _apply_evidence_gate

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "device": "cuda",
                                             "reward": 0.0, "success_rate": 0.0}}}}},
        )
        _write_exp_runs(tmp_path, success=True, metrics={"accuracy": 0.9})

        report = _make_report("partial")
        result = _apply_evidence_gate(report, tmp_path, run_experiment_ok_calls=1)
        assert "[evidence_audit]" in result.reproduction_summary

    def test_no_content_evidence_gate_fires_first(self, tmp_path, monkeypatch):
        """When content evidence is missing the primary gate fires; audit ON
        doesn't change that (still 'failed')."""
        monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")
        from backend.agents.rlm.report import _apply_evidence_gate

        # No experiment_runs.jsonl -> no content evidence
        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "accuracy": 0.8}}}}},
        )

        report = _make_report("partial")
        result = _apply_evidence_gate(report, tmp_path, run_experiment_ok_calls=1)
        # Primary gate fires (no content evidence) before audit branch
        assert result.verdict == "failed"
        assert "[evidence_gap]" in result.reproduction_summary

    def test_audit_failsoft_leaves_verdict_unchanged(self, tmp_path, monkeypatch):
        """If the audit itself raises, the gate decision is unchanged (fail-soft)."""
        monkeypatch.setenv("OPENRESEARCH_EVIDENCE_AUDIT", "1")

        # Patch audit_evidence_from_dir to raise. _apply_evidence_gate imports it
        # DEFERRED (at call time), so monkeypatching the module attribute is picked
        # up without reloading the report module. Do NOT importlib.reload(report) here:
        # a reload mints a fresh RLMFinalReport class and pollutes every later test's
        # isinstance(..., RLMFinalReport) checks (the regression this replaces).
        import backend.agents.rlm.evidence_audit as _ea_mod

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated audit failure")

        monkeypatch.setattr(_ea_mod, "audit_evidence_from_dir", _boom)

        from backend.agents.rlm.report import _apply_evidence_gate

        _write_metrics(
            tmp_path,
            {"per_model": {"m": {"e": {"b": {"status": "ok", "device": "cuda",
                                             "reward": 0.0, "success_rate": 0.0}}}}},
        )
        _write_exp_runs(tmp_path, success=True, metrics={"accuracy": 0.9})

        report = _make_report("partial")
        result = _apply_evidence_gate(report, tmp_path, run_experiment_ok_calls=1)
        # Fail-soft: audit boom -> verdict kept by the primary gate's decision
        # (content evidence present, no forge -> partial stays partial)
        assert result.verdict == "partial"
