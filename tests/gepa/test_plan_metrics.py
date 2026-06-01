"""Tests for plan_reproduction scoring metrics."""
import pytest
from backend.agents.gepa.metrics.plan_metrics import metrics_shape_score, contract_completeness

# A minimal valid ReproductionContract dict
VALID_CONTRACT = {
    "baseline_plan": "train ResNet on CIFAR-10",
    "smoke_test_plan": "run 1 epoch",
    "full_run_plan": "run 100 epochs",
    "expected_artifacts": ["checkpoints/final.pt", "metrics.json"],
    "dataset_plan": {"name": "CIFAR-10", "source": "torchvision"},
    "evaluation_plan": {"metrics": ["top1_acc"]},
    "verification_checklist": ["check accuracy >= 0.90"],
    "metrics_shape": [
        {"metric_id": "top1_acc", "json_path": "cifar10_resnet_top1_acc", "rubric_leaf_ids": []},
        {"metric_id": "top5_acc", "json_path": "cifar10_resnet_top5_acc", "rubric_leaf_ids": []},
    ],
}

EXPECTED_METRIC_IDS = ["top1_acc", "top5_acc"]


def test_metrics_shape_score_perfect():
    score, feedback = metrics_shape_score(VALID_CONTRACT, EXPECTED_METRIC_IDS)
    assert score == 1.0
    assert "missing" not in feedback.lower()


def test_metrics_shape_score_partial():
    contract = {**VALID_CONTRACT, "metrics_shape": [
        {"metric_id": "top1_acc", "json_path": "cifar10_top1", "rubric_leaf_ids": []},
    ]}
    score, feedback = metrics_shape_score(contract, EXPECTED_METRIC_IDS)
    assert score == 0.5
    assert "top5_acc" in feedback


def test_metrics_shape_score_empty_metrics_shape():
    contract = {**VALID_CONTRACT, "metrics_shape": []}
    score, feedback = metrics_shape_score(contract, EXPECTED_METRIC_IDS)
    assert score == 0.0
    assert "missing" in feedback.lower()


def test_metrics_shape_score_no_expected_ids():
    """If expected_metric_ids is empty, score is 1.0 (no constraint)."""
    score, feedback = metrics_shape_score(VALID_CONTRACT, [])
    assert score == 1.0


def test_metrics_shape_score_invalid_json():
    score, feedback = metrics_shape_score("not a dict", EXPECTED_METRIC_IDS)
    assert score == 0.0
    assert "invalid" in feedback.lower()


def test_contract_completeness_all_fields():
    score = contract_completeness(VALID_CONTRACT)
    assert score == 1.0


def test_contract_completeness_missing_field():
    contract = {k: v for k, v in VALID_CONTRACT.items() if k != "dataset_plan"}
    score = contract_completeness(contract)
    assert score < 1.0


def test_contract_completeness_missing_metrics_shape():
    contract = {k: v for k, v in VALID_CONTRACT.items() if k != "metrics_shape"}
    score = contract_completeness(contract)
    assert score < 1.0


def test_contract_completeness_empty_string_field():
    contract = {**VALID_CONTRACT, "baseline_plan": ""}
    score = contract_completeness(contract)
    assert score < 1.0
