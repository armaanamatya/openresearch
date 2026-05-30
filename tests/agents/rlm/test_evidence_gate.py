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


# --- Tightened predicate: evidence requires success==true AND non-empty metrics ---
# (user request 2026-05-30). A failed experiment that emitted metrics-like junk is
# NOT evidence; only a row that BOTH succeeded AND produced real metrics counts.

from backend.agents.rlm.report import _has_experiment_evidence  # noqa: E402


def _write_rows(tmp_path, *rows):
    (tmp_path / "experiment_runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )


def test_predicate_failed_experiment_with_junk_metrics_does_not_count(tmp_path):
    _write_rows(tmp_path, {"success": False, "metrics": {"accuracy": 0.99}})
    assert _has_experiment_evidence(tmp_path) is False


def test_predicate_successful_experiment_with_empty_metrics_does_not_count(tmp_path):
    _write_rows(tmp_path, {"success": True, "metrics": {}})
    assert _has_experiment_evidence(tmp_path) is False


def test_predicate_successful_experiment_with_nonempty_metrics_counts(tmp_path):
    _write_rows(tmp_path, {"success": True, "metrics": {"accuracy": 0.41}})
    assert _has_experiment_evidence(tmp_path) is True


def test_predicate_missing_file_fails_safe(tmp_path):
    assert _has_experiment_evidence(tmp_path) is False


def test_partial_rescued_only_by_a_successful_experiment_row(tmp_path, monkeypatch):
    """End-to-end: a failed experiment with junk metrics no longer rescues a
    partial verdict; a successful one with real metrics does."""
    monkeypatch.setenv("REPROLAB_EVIDENCE_GATE", "1")
    # Only a failed row with junk metrics on disk → no real evidence → downgrade.
    _write_rows(tmp_path, {"success": False, "metrics": {"accuracy": 0.99}})
    report = RLMFinalReport(verdict="partial", reproduction_summary="ran", baseline_metrics={})
    json_path, _ = write_final_report_rlm(report, tmp_path)
    assert json.loads(json_path.read_text())["verdict"] == "failed"

    # Add a genuinely successful row → evidence exists → partial stands.
    _write_rows(
        tmp_path,
        {"success": False, "metrics": {"accuracy": 0.99}},
        {"success": True, "metrics": {"accuracy": 0.41}},
    )
    report2 = RLMFinalReport(verdict="partial", reproduction_summary="ran", baseline_metrics={})
    json_path2, _ = write_final_report_rlm(report2, tmp_path)
    assert json.loads(json_path2.read_text())["verdict"] == "partial"
