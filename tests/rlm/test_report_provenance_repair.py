"""Tests for repair_projection_from_disk — Task 6 (BES conversion-correctness).

Verifies that a final report whose provenance is empty but whose grader scored
a populated code/metrics.json gets its baseline_metrics repopulated, and that
already-coherent reports are left untouched (no-op).
"""
import json
from backend.agents.rlm.conversion_guard import detect_projection_incoherence
from backend.agents.rlm.report import repair_projection_from_disk


def test_repair_populates_baseline_metrics_from_disk(tmp_path):
    metrics = {"cifar10_cnn": {"top1": 0.91}}
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text(json.dumps(metrics))
    kwargs = {"baseline_metrics": {}, "experiment_run_id": None, "primitive_trace": {}}
    rubric = {"overall_score": 0.53, "evidence_cites_metrics": True}
    fixed = repair_projection_from_disk(kwargs, rubric, tmp_path)
    assert fixed["baseline_metrics"] == metrics
    assert fixed.get("provenance_repaired") is True


def test_repair_is_noop_when_coherent(tmp_path):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text(json.dumps({"x": 1}))
    kwargs = {"baseline_metrics": {"x": 1}, "experiment_run_id": "r1", "primitive_trace": {"run_experiment": 1}}
    rubric = {"overall_score": 0.5, "evidence_cites_metrics": True}
    assert repair_projection_from_disk(kwargs, rubric, tmp_path).get("provenance_repaired") is None


def test_repair_is_noop_when_no_metrics_file(tmp_path):
    """No code/metrics.json → no repair possible even if provenance is empty."""
    kwargs = {"baseline_metrics": {}, "experiment_run_id": None, "primitive_trace": {}}
    rubric = {"overall_score": 0.53, "evidence_cites_metrics": True}
    fixed = repair_projection_from_disk(kwargs, rubric, tmp_path)
    assert fixed.get("provenance_repaired") is None
    assert fixed["baseline_metrics"] == {}


def test_repair_is_noop_when_rubric_no_evidence(tmp_path):
    """Rubric with no evidence signal (score=0, evidence_cites_metrics absent) → no-op."""
    metrics = {"cifar10_cnn": {"top1": 0.91}}
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text(json.dumps(metrics))
    kwargs = {"baseline_metrics": {}, "experiment_run_id": None, "primitive_trace": {}}
    rubric = {"overall_score": 0.0}  # no evidence_cites_metrics, score=0
    fixed = repair_projection_from_disk(kwargs, rubric, tmp_path)
    assert fixed.get("provenance_repaired") is None
