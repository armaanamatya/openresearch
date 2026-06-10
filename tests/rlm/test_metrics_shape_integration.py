"""Integration test: metrics_shape flows from plan_reproduction through
implement_baseline prompt and rubric_guard (PR-θ, Changes θ.1–θ.5).

Stubs the LLM and SDK calls so no real network/subprocess is needed.
Verifies the full contract:
  plan_reproduction → ReproductionContract.metrics_shape →
  implement_baseline prompt contains METRICS CONTRACT block →
  rubric_guard honors declared paths (not fingerprint).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.agents.rlm.primitives import plan_reproduction
from backend.agents.rlm.rubric_guard import (
    RubricGuardFailure,
    assert_metrics_schema,
)
from backend.agents.schemas import MetricPath, ReproductionContract


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_VALID_METRICS_SHAPE = [
    {
        "metric_id": "mnist_logistic_adam_final_nll",
        "json_path": "per_model.mnist_logistic.per_dataset.mnist.adam_final_nll",
        "rubric_leaf_ids": [],
    },
    {
        "metric_id": "cifar10_cnn_adam_final_loss",
        "json_path": "cifar10_cnn.cifar10.adam_final_loss",
        "rubric_leaf_ids": [],
    },
]


def _plan_response(metrics_shape=None) -> str:
    """Build a minimal plan_reproduction LLM response."""
    d: dict = {
        "reproduction_definition": "Reproduce baseline.",
        "smoke_test_plan": "Quick smoke.",
        "full_run_plan": "Full run.",
        "expected_outputs": ["metrics.json"],
        "evaluation_plan": "Evaluate on test set.",
    }
    if metrics_shape is not None:
        d["metrics_shape"] = metrics_shape
    return json.dumps(d)


# ---------------------------------------------------------------------------
# θ.2 + θ.3 integration: plan_reproduction → implement_baseline prompt
# ---------------------------------------------------------------------------

def test_metrics_shape_flows_to_implement_baseline_prompt(make_context, tmp_path) -> None:
    """A non-empty metrics_shape from plan_reproduction must appear in the
    implement_baseline guidance block as a METRICS CONTRACT section."""
    from backend.agents.baseline_implementation import (
        _compute_constraint_guidance,
    )

    # 1. Simulate plan_reproduction result with metrics_shape.
    ctx = make_context(tmp_path, llm_responses=[_plan_response(_VALID_METRICS_SHAPE)])
    plan_result = plan_reproduction({"core_contribution": "X"}, {}, ctx=ctx)

    assert len(plan_result["metrics_shape"]) == 2

    # 2. Extract metrics_shape as dicts (as primitives.implement_baseline does).
    contract = ReproductionContract(**plan_result)
    shape_dicts = [
        mp.model_dump() if hasattr(mp, "model_dump") else dict(mp)
        for mp in contract.metrics_shape
    ]

    # 3. Verify the guidance block contains the METRICS CONTRACT section.
    guidance = _compute_constraint_guidance(
        sandbox_mode=None,
        gpu_mode=None,
        metrics_shape=shape_dicts,
    )
    assert "METRICS CONTRACT" in guidance
    assert "mnist_logistic_adam_final_nll" in guidance
    assert "per_model.mnist_logistic.per_dataset.mnist.adam_final_nll" in guidance
    assert "cifar10_cnn.cifar10.adam_final_loss" in guidance


def test_empty_metrics_shape_omits_metrics_contract_block(make_context, tmp_path) -> None:
    """When metrics_shape is empty, the METRICS CONTRACT block is NOT injected."""
    from backend.agents.baseline_implementation import _compute_constraint_guidance

    guidance = _compute_constraint_guidance(
        sandbox_mode=None,
        gpu_mode=None,
        metrics_shape=[],
    )
    # Block should be absent; rest of guidance should be present.
    assert "METRICS CONTRACT" not in guidance
    assert "RUBRIC GUARD" in guidance


def test_no_metrics_shape_arg_omits_metrics_contract_block(make_context, tmp_path) -> None:
    """When metrics_shape is not passed (None), the METRICS CONTRACT block is absent."""
    from backend.agents.baseline_implementation import _compute_constraint_guidance

    guidance = _compute_constraint_guidance(sandbox_mode=None, gpu_mode=None)
    assert "METRICS CONTRACT" not in guidance


# ---------------------------------------------------------------------------
# θ.4 integration: rubric_guard honors declared paths from metrics_shape
# ---------------------------------------------------------------------------

def test_rubric_guard_authoritative_path_beats_fingerprint() -> None:
    """When metrics_shape is set, rubric_guard checks the declared json_path
    exactly — not the required_keys fingerprint. This means a declared path
    that is present passes, even if required_keys would have fingerprint-matched
    a different shape."""
    # Metrics has the exact declared path.
    metrics = {
        "per_model": {
            "mnist_logistic": {
                "per_dataset": {
                    "mnist": {"adam_final_nll": 0.4137}
                }
            }
        }
    }
    metrics_shape = [
        {
            "metric_id": "mnist_logistic_adam_final_nll",
            "json_path": "per_model.mnist_logistic.per_dataset.mnist.adam_final_nll",
            "rubric_leaf_ids": [],
        }
    ]
    # Must not raise.
    assert_metrics_schema(metrics, required_keys=[], metrics_shape=metrics_shape)


def test_rubric_guard_declared_path_wrong_shape_fails() -> None:
    """When the agent emits a FLAT shape but contract declared NESTED path → fail."""
    # Agent wrote flat key instead of nested.
    metrics = {
        "mnist_logistic_adam_final_nll": 0.4137,  # flat, not nested
    }
    metrics_shape = [
        {
            "metric_id": "mnist_logistic_adam_final_nll",
            # Contract says nested — agent didn't honor it.
            "json_path": "per_model.mnist_logistic.per_dataset.mnist.adam_final_nll",
            "rubric_leaf_ids": [],
        }
    ]
    with pytest.raises(RubricGuardFailure) as excinfo:
        assert_metrics_schema(metrics, required_keys=[], metrics_shape=metrics_shape)
    detail = json.loads(str(excinfo.value))
    # The declared (nested) path must appear in the failure.
    assert any("per_model.mnist_logistic" in k for k in detail["missing_keys"])


# ---------------------------------------------------------------------------
# θ.5 integration: run_experiment wires metrics_shape from ctx.reproduction_contract
# ---------------------------------------------------------------------------

def test_run_experiment_uses_metrics_shape_from_contract(make_context, tmp_path) -> None:
    """When ctx.reproduction_contract.metrics_shape is non-empty, run_experiment
    performs a post-run check with the declared paths and attaches violations."""
    from backend.agents.rlm.primitives import run_experiment
    from backend.agents.schemas import ReproductionContract

    # Set up code dir with commands.json and metrics.json (missing the declared path).
    code_dir = tmp_path / "test_proj" / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "commands.json").write_text(json.dumps(["echo done"]))

    # Metrics written by agent is flat, but contract declared nested.
    metrics_data = {
        "mnist_logistic_adam_final_nll": 0.42,  # flat — wrong vs declaration
    }

    # Mock _execute_in_sandbox to return the metrics directly.
    from backend.agents.rlm import primitives as _prims
    mock_sandbox_result = {
        "success": True,
        "metrics": metrics_data,
        "logs": "",
        "exit_code": 0,
    }

    ctx = make_context(tmp_path)
    # Set reproduction_contract with non-empty metrics_shape declaring nested path.
    ctx.reproduction_contract = ReproductionContract(
        metrics_shape=[
            MetricPath(
                metric_id="mnist_logistic_adam_final_nll",
                json_path="per_model.mnist_logistic.per_dataset.mnist.adam_final_nll",
            )
        ]
    )

    with patch.object(_prims, "_execute_in_sandbox", new=AsyncMock(return_value=mock_sandbox_result)):
        result = run_experiment(
            str(code_dir),
            "test_image:latest",
            ctx=ctx,
        )

    # The result should have contract_violations listing the missing declared path.
    violations = result.get("contract_violations", [])
    assert len(violations) >= 1
    violation_texts = [str(v) for v in violations]
    combined = " ".join(violation_texts)
    assert "per_model.mnist_logistic.per_dataset.mnist.adam_final_nll" in combined


def test_run_experiment_no_metrics_shape_no_extra_violations(make_context, tmp_path) -> None:
    """When reproduction_contract.metrics_shape is empty, no extra violations added."""
    from backend.agents.rlm.primitives import run_experiment
    from backend.agents.schemas import ReproductionContract

    code_dir = tmp_path / "test_proj" / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "commands.json").write_text(json.dumps(["echo done"]))

    metrics_data = {"flat_acc": 0.81}

    from backend.agents.rlm import primitives as _prims
    mock_sandbox_result = {
        "success": True,
        "metrics": metrics_data,
        "logs": "",
        "exit_code": 0,
    }

    ctx = make_context(tmp_path)
    # Empty metrics_shape → no extra violations.
    ctx.reproduction_contract = ReproductionContract(metrics_shape=[])

    with patch.object(_prims, "_execute_in_sandbox", new=AsyncMock(return_value=mock_sandbox_result)):
        result = run_experiment(
            str(code_dir),
            "test_image:latest",
            ctx=ctx,
        )

    # No contract_violations from metrics_shape (may have other violations from
    # rubric_contract.validate if yaml exists, but metrics_shape violations are absent).
    violations = result.get("contract_violations", [])
    ms_violations = [v for v in violations if "metrics_shape" in str(v) or "declared path" in str(v).lower()]
    assert ms_violations == []
