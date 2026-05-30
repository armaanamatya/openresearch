"""Phase 3 — final-report evidence gate (FM-004).

Every finished /runs report shipped baseline_metrics={} + rubric 0 yet
verdict="partial"; pb_…784 even claimed "implemented and executed" with no
run_experiment in its trace. The write-time gate downgrades a success-ish verdict
that has no experiment evidence to "failed", across ALL writers (FINAL_VAR,
watchdog, fatal-abort).
"""
from __future__ import annotations

import json

from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm


def test_partial_without_evidence_downgrades_to_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_EVIDENCE_GATE", "1")
    # No experiment_runs.jsonl at all.
    report = RLMFinalReport(
        verdict="partial", reproduction_summary="did stuff", baseline_metrics={}
    )
    json_path, _ = write_final_report_rlm(report, tmp_path)
    data = json.loads(json_path.read_text())
    assert data["verdict"] == "failed"
    assert "evidence_gap" in data["reproduction_summary"]


def test_partial_with_metrics_evidence_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_EVIDENCE_GATE", "1")
    (tmp_path / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True, "metrics": {"accuracy": 0.41}}) + "\n"
    )
    report = RLMFinalReport(
        verdict="partial", reproduction_summary="ran it",
        baseline_metrics={"accuracy": 0.41},
    )
    json_path, _ = write_final_report_rlm(report, tmp_path)
    assert json.loads(json_path.read_text())["verdict"] == "partial"


def test_partial_with_jsonl_evidence_but_empty_inline_metrics_is_kept(tmp_path, monkeypatch):
    """Evidence exists on disk even if the self-attested baseline_metrics is empty."""
    monkeypatch.setenv("REPROLAB_EVIDENCE_GATE", "1")
    (tmp_path / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True, "metrics": {"loss": 1.2}}) + "\n"
    )
    report = RLMFinalReport(
        verdict="partial", reproduction_summary="ran but didn't copy metrics",
        baseline_metrics={},
    )
    json_path, _ = write_final_report_rlm(report, tmp_path)
    # Has disk evidence → keep partial (don't punish a missing self-attest copy).
    assert json.loads(json_path.read_text())["verdict"] == "partial"


def test_failed_verdict_is_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_EVIDENCE_GATE", "1")
    report = RLMFinalReport(verdict="failed", reproduction_summary="nope", baseline_metrics={})
    json_path, _ = write_final_report_rlm(report, tmp_path)
    assert json.loads(json_path.read_text())["verdict"] == "failed"


def test_gate_disabled_keeps_verdict(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_EVIDENCE_GATE", "0")
    report = RLMFinalReport(verdict="partial", reproduction_summary="x", baseline_metrics={})
    json_path, _ = write_final_report_rlm(report, tmp_path)
    assert json.loads(json_path.read_text())["verdict"] == "partial"
