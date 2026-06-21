"""Tests for the P2.3 validation panel stamp in write_final_report_rlm.

Hermetic (tmp_path only): no network, no external process, no fixture files
from the repo.  The validator verdict is planted on disk; write_final_report_rlm
is called; we assert final_report.json.validation is populated or empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm.external_validator import (
    ValidatorVerdict,
    PredicateVerdict,
    evidence_fingerprint,
    persist_verdict,
)
from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics() -> dict:
    return {"mean_reward": 0.72, "accuracy": 0.88}


def _make_verdict(fp: str, *, status: str = "clean", vetoed: bool = False) -> ValidatorVerdict:
    pred = PredicateVerdict(
        predicate="not_all_constant",
        metric_ref="mean_reward",
        violated=vetoed,
        detail="test detail",
    )
    return ValidatorVerdict(
        status=status,
        veto_set=["mean_reward"] if vetoed else [],
        predicates=[pred],
        panel_models=["validator:azure:gpt-4o"],
        separation="independent",
        evidence_fingerprint=fp,
    )


def _make_report(metrics: dict | None = None) -> RLMFinalReport:
    r = RLMFinalReport()
    r.baseline_metrics = metrics or {}
    return r


# ---------------------------------------------------------------------------
# Test 1 — matching fingerprint → validation field populated
# ---------------------------------------------------------------------------


def test_matching_fingerprint_stamps_validation(tmp_path: Path) -> None:
    """A planted verdict whose fingerprint matches the shipped metrics is stamped."""
    metrics = _make_metrics()
    fp = evidence_fingerprint(metrics)
    verdict = _make_verdict(fp, status="clean")
    persist_verdict(tmp_path, verdict)

    report = _make_report(metrics)
    write_final_report_rlm(report, tmp_path)

    result = json.loads((tmp_path / "final_report.json").read_text())
    val = result.get("validation", {})
    assert val.get("status") == "clean"
    assert val.get("separation") == "independent"
    assert val.get("evidence_fingerprint") == fp
    assert val.get("veto_set") == []
    assert isinstance(val.get("predicates"), list)
    assert len(val["predicates"]) == 1
    p = val["predicates"][0]
    assert p["predicate"] == "not_all_constant"
    assert p["violated"] is False


# ---------------------------------------------------------------------------
# Test 2 — vetoed verdict is stamped correctly
# ---------------------------------------------------------------------------


def test_vetoed_verdict_stamped(tmp_path: Path) -> None:
    """A vetoed verdict stamps veto_set and status=vetoed."""
    metrics = _make_metrics()
    fp = evidence_fingerprint(metrics)
    verdict = _make_verdict(fp, status="vetoed", vetoed=True)
    persist_verdict(tmp_path, verdict)

    report = _make_report(metrics)
    write_final_report_rlm(report, tmp_path)

    result = json.loads((tmp_path / "final_report.json").read_text())
    val = result.get("validation", {})
    assert val.get("status") == "vetoed"
    assert "mean_reward" in val.get("veto_set", [])


# ---------------------------------------------------------------------------
# Test 3 — mismatched fingerprint → validation empty (stale ignored)
# ---------------------------------------------------------------------------


def test_mismatched_fingerprint_leaves_validation_empty(tmp_path: Path) -> None:
    """A verdict whose fingerprint does not match the shipped metrics is ignored."""
    stale_metrics = {"other_metric": 0.5}
    fp = evidence_fingerprint(stale_metrics)
    verdict = _make_verdict(fp, status="clean")
    persist_verdict(tmp_path, verdict)

    # Ship DIFFERENT metrics — fingerprint will not match
    shipped_metrics = _make_metrics()
    assert evidence_fingerprint(shipped_metrics) != fp  # sanity

    report = _make_report(shipped_metrics)
    write_final_report_rlm(report, tmp_path)

    result = json.loads((tmp_path / "final_report.json").read_text())
    val = result.get("validation", {})
    assert val == {}, f"expected empty validation, got {val!r}"


# ---------------------------------------------------------------------------
# Test 4 — no verdict file + OPENRESEARCH_EXTERNAL_VALIDATOR unset →
#          validation empty, report otherwise unchanged
# ---------------------------------------------------------------------------


def test_no_verdict_file_validation_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the validator is disabled and no verdict file exists, validation is empty.

    Note: the evidence gate may legitimately downgrade verdict when no experiment
    evidence is present in tmp_path; that is existing correct behavior unrelated to
    the validation stamp.  We assert only that validation is empty and iterations is
    preserved (a field the gate never touches).
    """
    monkeypatch.delenv("OPENRESEARCH_EXTERNAL_VALIDATOR", raising=False)

    report = _make_report(_make_metrics())
    report.iterations = 3

    write_final_report_rlm(report, tmp_path)

    result = json.loads((tmp_path / "final_report.json").read_text())
    assert result.get("validation", {}) == {}
    # iterations is a field the gate/stamp logic never modifies
    assert result["iterations"] == 3


# ---------------------------------------------------------------------------
# Test 5 — metrics sourced from on-disk code/metrics.json when baseline_metrics
#          is empty but verdict was planted against that file
# ---------------------------------------------------------------------------


def test_metrics_loaded_from_disk_when_baseline_empty(tmp_path: Path) -> None:
    """When baseline_metrics is empty, fingerprint is computed from code/metrics.json."""
    disk_metrics = {"val_loss": 0.33, "accuracy": 0.91}
    code_dir = tmp_path / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "metrics.json").write_text(json.dumps(disk_metrics))

    fp = evidence_fingerprint(disk_metrics)
    verdict = _make_verdict(fp, status="clean")
    persist_verdict(tmp_path, verdict)

    # baseline_metrics is deliberately empty
    report = _make_report({})
    write_final_report_rlm(report, tmp_path)

    result = json.loads((tmp_path / "final_report.json").read_text())
    val = result.get("validation", {})
    assert val.get("status") == "clean"
    assert val.get("evidence_fingerprint") == fp
