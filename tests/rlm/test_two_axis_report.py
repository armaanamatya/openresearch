"""U11 integration: attach the two-axis verdict to a report (two_axis_report.py).

Covers the flag gate, the conservative no-artifact default, fail-soft loading,
and the end-to-end A4 guarantee: a faithful-but-contradicted artifact set
projects legacy ``verdict="reproduced"`` — NOT ``failed``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm import two_axis_report as tar

_FLAG = "REPROLAB_TWO_AXIS_VERDICT"


def _rubric(fidelity: float = 0.9, result_match: float = 0.1) -> dict:
    return {
        "overall_score": 0.5,
        "areas": [
            {"area": "Method and code fidelity to the paper", "score": fidelity, "weight": 0.4},
            {"area": "Experiment execution and reproducibility", "score": fidelity, "weight": 0.2},
            {"area": "Result match versus the paper's reported targets", "score": result_match, "weight": 0.2},
        ],
    }


def _write(run_dir: Path, rel: str, payload: dict) -> None:
    p = run_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")


def _green_certificate() -> dict:
    return {
        "invariant_tests_ran": True,
        "invariant_tests_passed": True,
        "mutation_confirmed": True,
        "blinded_extraction_agreed": True,
        "obligation_profile": "end_to_end",
        "profile_satisfied": True,
    }


def _claim(effects: list[float], *, scope: dict, measured_scope: dict, claimed: float = 9.4) -> dict:
    return {
        "comparison": {
            "claim_id": "primary",
            "description": "method beats baseline",
            "metric_name": "success_rate",
            "direction": "higher_is_better",
            "estimate_kind": "percentage_points",
            "baseline_label": "GRPO",
            "claimed_effect": claimed,
            "equivalence_margin": 1.0,
            "scope": scope,
            "is_primary": True,
        },
        "seed_bundle": {"seeds": [42, 43], "per_seed_effect": effects, "rng_independent": True},
        "measured_scope": measured_scope,
    }


# --------------------------------------------------------------------------- #

def test_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv(_FLAG, raising=False)
    report = {"verdict": "reproduced", "rubric": _rubric()}
    before = dict(report)
    assert tar.compute_and_attach(report, tmp_path) is False
    assert report == before  # untouched
    assert "schema_version" not in report


def test_enabled_no_artifacts_is_partial_not_broken(monkeypatch, tmp_path):
    """Flag on, metrics present, but no certificate/claims → honest
    (partial, inconclusive); legacy verdict projected from fidelity = 'partial'."""
    monkeypatch.setenv(_FLAG, "1")
    _write(tmp_path, "code/metrics.json", {"status": "ok", "per_model": {"m": {"v": 1.0}}})
    report = {"verdict": "reproduced", "rubric": _rubric()}
    assert tar.compute_and_attach(report, tmp_path) is True
    assert report["schema_version"] == 2
    assert report["implementation_verdict"] == "partial"  # ran, not certified — NOT broken
    assert report["replication_verdict"] == "inconclusive"
    assert report["verdict"] == "partial"


def test_no_metrics_is_broken(monkeypatch, tmp_path):
    monkeypatch.setenv(_FLAG, "1")
    report = {"verdict": "reproduced", "rubric": _rubric()}
    assert tar.compute_and_attach(report, tmp_path) is True
    assert report["implementation_verdict"] == "broken"
    assert report["verdict"] == "failed"


def test_end_to_end_faithful_contradicted_is_NOT_failed(monkeypatch, tmp_path):
    """THE A4 GUARANTEE end-to-end: a green certificate + a contradicting primary
    claim at matching scope projects legacy verdict 'reproduced', never 'failed'."""
    monkeypatch.setenv(_FLAG, "1")
    _write(tmp_path, "code/metrics.json", {"status": "ok", "per_model": {"m": {"v": 1.0}}})
    _write(tmp_path, "rlm_state/fidelity_certificate.json", _green_certificate())
    scope = {"model": "Qwen2.5-3B", "dataset": "ALFWorld", "split": "test"}
    _write(tmp_path, "rlm_state/repro_spec.json",
           {"claims": [_claim([-3.0, -2.8], scope=scope, measured_scope=scope)]})
    report = {"verdict": "partial", "rubric": _rubric()}

    assert tar.compute_and_attach(report, tmp_path) is True
    assert report["implementation_verdict"] == "faithful"
    assert report["replication_verdict"] == "contradicted"
    assert report["verdict"] == "reproduced"      # <- legacy projection from FIDELITY
    assert report["verdict"] != "failed"          # the whole point of A4
    assert report["reproducibility"]["per_claim"][0]["status"] == "contradicted"


def test_end_to_end_scope_mismatch_is_inconclusive(monkeypatch, tmp_path):
    """A 7B-specific claim measured on a 3B run → inconclusive, never contradicted
    (A2), even with a green certificate and an inverting effect."""
    monkeypatch.setenv(_FLAG, "1")
    _write(tmp_path, "code/metrics.json", {"status": "ok", "per_model": {"m": {"v": 1.0}}})
    _write(tmp_path, "rlm_state/fidelity_certificate.json", _green_certificate())
    _write(tmp_path, "rlm_state/repro_spec.json", {"claims": [_claim(
        [-3.0, -2.8],
        scope={"model": "Qwen2.5-7B", "dataset": "ALFWorld", "split": "test"},
        measured_scope={"model": "Qwen2.5-3B", "dataset": "ALFWorld", "split": "test"},
    )]})
    report = {"verdict": "partial", "rubric": _rubric()}
    assert tar.compute_and_attach(report, tmp_path) is True
    assert report["implementation_verdict"] == "faithful"
    assert report["replication_verdict"] == "inconclusive"


def test_fidelity_score_excludes_result_match():
    rubric = _rubric(fidelity=0.8, result_match=0.0)
    # only the two non-result-match areas (both 0.8) count → 0.8, not dragged by 0.0
    assert tar.fidelity_score_from_rubric(rubric) == pytest.approx(0.8)


def test_malformed_repro_spec_is_failsoft(monkeypatch, tmp_path):
    monkeypatch.setenv(_FLAG, "1")
    _write(tmp_path, "code/metrics.json", {"status": "ok", "per_model": {"m": {"v": 1.0}}})
    (tmp_path / "rlm_state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "rlm_state" / "repro_spec.json").write_text("{not valid json", encoding="utf-8")
    report = {"verdict": "partial", "rubric": _rubric()}
    # must not raise; bad artifact → no claims → inconclusive
    assert tar.compute_and_attach(report, tmp_path) is True
    assert report["replication_verdict"] == "inconclusive"


def test_live_write_path_attaches_verdict(monkeypatch, tmp_path):
    """The LIVE finalize path (write_final_report_rlm) attaches the two-axis
    verdict to final_report.json and projects 'failed'→'reproduced' for a
    faithful-contradicted run — A4 end-to-end through the real writer."""
    monkeypatch.setenv(_FLAG, "1")
    monkeypatch.setenv("REPROLAB_UPDATE_CALIBRATION", "false")
    _write(tmp_path, "code/metrics.json", {"status": "ok", "per_model": {"m": {"v": 1.0}}})
    _write(tmp_path, "rlm_state/fidelity_certificate.json", _green_certificate())
    scope = {"model": "Qwen2.5-3B", "dataset": "ALFWorld", "split": "test"}
    _write(tmp_path, "rlm_state/repro_spec.json",
           {"claims": [_claim([-3.0, -2.8], scope=scope, measured_scope=scope)]})

    from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm
    report = RLMFinalReport(verdict="failed", rubric=_rubric())
    write_final_report_rlm(report, tmp_path)

    written = json.loads((tmp_path / "final_report.json").read_text())
    assert written["implementation_verdict"] == "faithful"
    assert written["replication_verdict"] == "contradicted"
    assert written["verdict"] == "reproduced"   # A4 — projected from fidelity, via the real writer
    assert written["schema_version"] == 2


def test_live_write_path_unchanged_when_flag_off(monkeypatch, tmp_path):
    """Flag off → final_report.json is the plain model dump (no two-axis fields)."""
    monkeypatch.delenv(_FLAG, raising=False)
    monkeypatch.setenv("REPROLAB_UPDATE_CALIBRATION", "false")
    from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm
    report = RLMFinalReport(verdict="partial", rubric=_rubric())
    write_final_report_rlm(report, tmp_path)
    written = json.loads((tmp_path / "final_report.json").read_text())
    assert "implementation_verdict" not in written
    assert written["verdict"] == "partial"


def test_ambiguous_claim_blocks_contradiction(monkeypatch, tmp_path):
    """An extractor-flagged ambiguous claim can't contradict even with a green
    cert + inverting effect (A1)."""
    monkeypatch.setenv(_FLAG, "1")
    _write(tmp_path, "code/metrics.json", {"status": "ok", "per_model": {"m": {"v": 1.0}}})
    _write(tmp_path, "rlm_state/fidelity_certificate.json", _green_certificate())
    scope = {"model": "Qwen2.5-3B", "dataset": "ALFWorld", "split": "test"}
    claim = _claim([-3.0, -2.8], scope=scope, measured_scope=scope)
    claim["comparison"]["ambiguous"] = True
    claim["comparison"]["ambiguity_reason"] = "pp vs relative-% undetermined"
    _write(tmp_path, "rlm_state/repro_spec.json", {"claims": [claim]})
    report = {"verdict": "partial", "rubric": _rubric()}
    assert tar.compute_and_attach(report, tmp_path) is True
    assert report["replication_verdict"] == "inconclusive"
