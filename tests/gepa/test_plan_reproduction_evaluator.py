"""PlanReproductionEvaluator: GEPA evaluator for plan_reproduction prompt."""
from unittest.mock import MagicMock

import pytest

from backend.agents.gepa.adapters.plan_reproduction import PlanReproductionEvaluator
from backend.agents.gepa.trainset.paper_examples import PaperExample


@pytest.fixture()
def llm_client():
    client = MagicMock()
    client.complete = MagicMock(return_value="""{
        "baseline_plan": "train ResNet",
        "smoke_test_plan": "1 epoch",
        "full_run_plan": "100 epochs",
        "expected_artifacts": ["metrics.json"],
        "dataset_plan": {"name": "CIFAR-10"},
        "evaluation_plan": {"metrics": ["top1"]},
        "verification_checklist": ["check acc"],
        "metrics_shape": [
            {"metric_id": "top1_acc", "json_path": "cifar10_top1", "rubric_leaf_ids": []}
        ]
    }""")
    client._last_usage = {}
    return client


@pytest.fixture()
def example():
    return PaperExample(
        method_spec={"datasets": [], "metrics": [], "training_recipe": {}, "hardware_clues": []},
        env_spec={"base_image": "pytorch/pytorch:2.1.0"},
        expected_metric_ids=["top1_acc"],
    )


def test_evaluate_returns_score_and_feedback(llm_client, example):
    evaluator = PlanReproductionEvaluator(llm_client=llm_client)
    score, side_info = evaluator(
        candidate={"plan_reproduction_system": "You are a planner."},
        example=example,
    )
    assert 0.0 <= score <= 1.0
    assert isinstance(side_info, dict)


def test_evaluate_bad_json_returns_zero(llm_client, example):
    llm_client.complete.return_value = "not json at all"
    evaluator = PlanReproductionEvaluator(llm_client=llm_client)
    score, side_info = evaluator(
        candidate={"plan_reproduction_system": "broken prompt"},
        example=example,
    )
    assert score == 0.0
    assert "error" in side_info


def test_evaluate_uses_candidate_as_system_prompt(llm_client, example):
    evaluator = PlanReproductionEvaluator(llm_client=llm_client)
    custom_system = "Custom system prompt for testing."
    evaluator(candidate={"plan_reproduction_system": custom_system}, example=example)
    call_kwargs = llm_client.complete.call_args
    # The system passed to llm_client.complete must START with the custom_system
    # (it has _METRICS_SHAPE_INSTRUCTION appended)
    assert call_kwargs.kwargs["system"].startswith(custom_system)
