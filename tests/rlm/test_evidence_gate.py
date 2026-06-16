"""A7 EVIDENCE_GATE — the honest backstop (2026-06-16).

OPENRESEARCH_EVIDENCE_GATE default OFF → no veto (byte-for-byte today). ON → a
RESULT-CLAIMING leaf the grader credited (>0) whose cited per_model cell has NO
successful on-disk evidence (matching on BOTH model and dataset) is vetoed to
0.0 — the grader cannot credit a result that was never computed. Judgment leaves
and substantiated result leaves are never touched.
"""

from __future__ import annotations

import json
import re

import pytest

from backend.agents.rlm.evidence_gate import (
    gate_decision,
    leaf_claims_measured_result,
)
from backend.evals.paperbench.leaf_scorer import (
    _result_leaf_substantiated,
    score_reproduction,
)

RUBRIC = {
    "id": "root", "requirements": "r", "weight": 1.0, "target_score": 0.7,
    "sub_tasks": [
        {"id": "r_cifar", "weight": 1.0, "sub_tasks": [], "task_category": "Result Analysis",
         "requirements": "ResNet accuracy on CIFAR10 matches the reported value"},
        {"id": "r_imagenet", "weight": 1.0, "sub_tasks": [], "task_category": "Result Analysis",
         "requirements": "ResNet accuracy on ImageNet matches the reported value"},
        {"id": "judgment", "weight": 1.0, "sub_tasks": [],
         "requirements": "the network uses residual skip connections"},
    ],
}

# resnet+cifar10 ran successfully; imagenet never ran.
METRICS = {"per_model": {"resnet": {"cifar10": {"baseline": {"status": "ok", "accuracy": 0.91}}}}}


class _Stub:
    """Grades every leaf the prompt asks about at a fixed score."""

    def __init__(self, score: float = 0.8):
        self.score = score
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        ids = re.findall(r'"leaf_id":\s*"([^"]+)"', user)
        return json.dumps(
            [{"leaf_id": i, "score": self.score, "justification": "x"} for i in ids]
        )


def _write_metrics(run_dir, obj=METRICS):
    (run_dir / "code").mkdir(parents=True, exist_ok=True)
    (run_dir / "code" / "metrics.json").write_text(json.dumps(obj), encoding="utf-8")


def _toks(s: str) -> frozenset[str]:
    return frozenset(t for t in re.split(r"[^a-z0-9]+", s.lower()) if t)


def _by_id(result):
    return {r["id"]: r for r in result["leaf_scores"]}


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    for v in (
        "OPENRESEARCH_EVIDENCE_GATE", "OPENRESEARCH_GRADER_SAMPLES",
        "OPENRESEARCH_GRADER_BACKEND", "OPENRESEARCH_DETERMINISTIC_LEAVES",
    ):
        monkeypatch.delenv(v, raising=False)


# --- pure module --------------------------------------------------------------

def test_gate_decision_vetoes_only_credited_result_without_evidence():
    assert gate_decision(score=0.8, claims_result=True, has_disk_evidence=False) == (0.0, True)
    assert gate_decision(score=0.8, claims_result=True, has_disk_evidence=True) == (0.8, False)
    assert gate_decision(score=0.8, claims_result=False, has_disk_evidence=False) == (0.8, False)
    assert gate_decision(score=0.0, claims_result=True, has_disk_evidence=False) == (0.0, False)
    assert gate_decision(score=None, claims_result=True, has_disk_evidence=False) == (None, False)


def test_claims_measured_result_is_conservative():
    assert leaf_claims_measured_result({"task_category": "Result Analysis"}) is True
    assert leaf_claims_measured_result({"finegrained_task_category": "metric match"}) is True
    assert leaf_claims_measured_result({"check_kind": "deterministic:numeric"}) is True
    assert leaf_claims_measured_result({"evidence_required": True}) is True
    # no category + no annotation → NOT a confident result claim → not gated
    assert leaf_claims_measured_result({"requirements": "uses a sigmoid gate"}) is False
    assert leaf_claims_measured_result({"task_category": "Code Development"}) is False


def test_substantiated_requires_both_model_and_dataset():
    assert _result_leaf_substantiated(_toks("resnet accuracy on cifar10"), METRICS) is True
    # model overlaps but dataset does not → cross-dataset fabrication → NOT substantiated
    assert _result_leaf_substantiated(_toks("resnet accuracy on imagenet"), METRICS) is False
    # dataset overlaps but model does not
    assert _result_leaf_substantiated(_toks("vgg accuracy on cifar10"), METRICS) is False
    # nothing ran
    assert _result_leaf_substantiated(_toks("resnet on cifar10"), {}) is False


# --- integration via score_reproduction --------------------------------------

def test_gate_off_is_byte_for_byte_today(tmp_path):
    _write_metrics(tmp_path)
    result = score_reproduction(RUBRIC, tmp_path, _Stub(0.8), degraded=False)
    recs = _by_id(result)
    assert recs["r_imagenet"]["score"] == pytest.approx(0.8)  # NOT vetoed (flag off)
    assert "evidence_gate_vetoed" not in recs["r_imagenet"]


def test_gate_on_vetoes_unsubstantiated_result_only(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "1")
    _write_metrics(tmp_path)
    result = score_reproduction(RUBRIC, tmp_path, _Stub(0.8), degraded=False)
    recs = _by_id(result)
    # imagenet result credited 0.8 but never ran → vetoed to 0.0
    assert recs["r_imagenet"]["score"] == 0.0
    assert recs["r_imagenet"]["evidence_gate_vetoed"] is True
    assert recs["r_imagenet"]["original_score"] == pytest.approx(0.8)
    # cifar result DID run → kept
    assert recs["r_cifar"]["score"] == pytest.approx(0.8)
    assert "evidence_gate_vetoed" not in recs["r_cifar"]
    # judgment leaf (no result category) → never gated
    assert recs["judgment"]["score"] == pytest.approx(0.8)
    assert "evidence_gate_vetoed" not in recs["judgment"]


def test_gate_on_empty_metrics_vetoes_all_result_claims(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "1")
    _write_metrics(tmp_path, {"per_model": {}})  # credited results, nothing computed
    result = score_reproduction(RUBRIC, tmp_path, _Stub(0.8), degraded=False)
    recs = _by_id(result)
    assert recs["r_cifar"]["score"] == 0.0 and recs["r_cifar"]["evidence_gate_vetoed"] is True
    assert recs["r_imagenet"]["score"] == 0.0 and recs["r_imagenet"]["evidence_gate_vetoed"] is True
    # judgment leaf still untouched — it makes no measured-result claim
    assert recs["judgment"]["score"] == pytest.approx(0.8)
