"""Tests for plan_reproduction metrics_shape parsing (PR-θ, Change θ.2).

Verifies that plan_reproduction:
  1. Parses a valid metrics_shape list from the LLM response.
  2. Attaches the parsed shape to the returned ReproductionContract.
  3. Returns an empty metrics_shape when the LLM omits the field (backward compat).
  4. Silently skips malformed items instead of crashing.
"""

from __future__ import annotations

import json

import pytest

from backend.agents.rlm.primitives import plan_reproduction, _METRICS_SHAPE_INSTRUCTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contract_with_shape(metrics_shape_value) -> str:
    """Build a minimal ReproductionContract JSON with the given metrics_shape."""
    return json.dumps({
        "reproduction_definition": "Reproduce the paper.",
        "smoke_test_plan": "Quick smoke.",
        "full_run_plan": "Full run.",
        "expected_outputs": ["metrics.json"],
        "evaluation_plan": "Evaluate on test set.",
        "metrics_shape": metrics_shape_value,
    })


def _contract_without_shape() -> str:
    """Minimal ReproductionContract JSON with NO metrics_shape key."""
    return json.dumps({
        "reproduction_definition": "Reproduce the paper.",
        "smoke_test_plan": "Quick smoke.",
        "full_run_plan": "Full run.",
        "expected_outputs": ["metrics.json"],
        "evaluation_plan": "Evaluate on test set.",
    })


# ---------------------------------------------------------------------------
# θ.2 tests
# ---------------------------------------------------------------------------

def test_plan_reproduction_parses_metrics_shape(make_context, tmp_path) -> None:
    """LLM emits metrics_shape → plan_reproduction attaches it to the contract."""
    llm_response = _contract_with_shape([
        {
            "metric_id": "mnist_logistic_adam_final_nll",
            "json_path": "per_model.mnist_logistic.per_dataset.mnist.adam_final_nll",
            "rubric_leaf_ids": [],
        },
        {
            "metric_id": "cifar10_cnn_adam_final_loss",
            "json_path": "cifar10_cnn.cifar10.adam_final_loss",
            "rubric_leaf_ids": ["leaf_a"],
        },
    ])
    ctx = make_context(tmp_path, llm_responses=[llm_response])
    result = plan_reproduction({"core_contribution": "X"}, {}, ctx=ctx)

    assert "metrics_shape" in result
    assert len(result["metrics_shape"]) == 2
    first = result["metrics_shape"][0]
    assert first["metric_id"] == "mnist_logistic_adam_final_nll"
    assert first["json_path"] == "per_model.mnist_logistic.per_dataset.mnist.adam_final_nll"
    assert first["rubric_leaf_ids"] == []


def test_plan_reproduction_empty_metrics_shape_list(make_context, tmp_path) -> None:
    """LLM emits metrics_shape: [] → contract has empty list (no fingerprint needed)."""
    llm_response = _contract_with_shape([])
    ctx = make_context(tmp_path, llm_responses=[llm_response])
    result = plan_reproduction({"core_contribution": "X"}, {}, ctx=ctx)

    assert result["metrics_shape"] == []


def test_plan_reproduction_backward_compat_no_metrics_shape(make_context, tmp_path) -> None:
    """LLM omits metrics_shape → contract has empty list (backward compat)."""
    llm_response = _contract_without_shape()
    ctx = make_context(tmp_path, llm_responses=[llm_response])
    result = plan_reproduction({"core_contribution": "X"}, {}, ctx=ctx)

    # Empty list — fingerprint fallback will apply in rubric_guard.
    assert result["metrics_shape"] == []


def test_plan_reproduction_malformed_item_skipped(make_context, tmp_path) -> None:
    """Malformed metrics_shape items are silently skipped (not a crash)."""
    llm_response = _contract_with_shape([
        # Valid item first
        {"metric_id": "val_loss", "json_path": "results.val_loss", "rubric_leaf_ids": []},
        # Missing required json_path → should be skipped
        {"metric_id": "incomplete_no_path"},
        # Non-dict item → should be skipped
        "just a string",
        # None → should be skipped
        None,
    ])
    ctx = make_context(tmp_path, llm_responses=[llm_response])
    result = plan_reproduction({"core_contribution": "X"}, {}, ctx=ctx)

    # Only the valid item survives; malformed ones are dropped.
    assert len(result["metrics_shape"]) == 1
    assert result["metrics_shape"][0]["metric_id"] == "val_loss"


def test_plan_reproduction_metrics_shape_not_list(make_context, tmp_path) -> None:
    """metrics_shape that is not a list at all → treated as empty (not a crash)."""
    llm_response = _contract_with_shape("not_a_list")
    ctx = make_context(tmp_path, llm_responses=[llm_response])
    result = plan_reproduction({"core_contribution": "X"}, {}, ctx=ctx)

    assert result["metrics_shape"] == []


def test_plan_reproduction_metrics_shape_instruction_in_system_prompt(make_context, tmp_path) -> None:
    """The _METRICS_SHAPE_INSTRUCTION is always appended to the system prompt."""
    llm_response = _contract_without_shape()
    ctx = make_context(tmp_path, llm_responses=[llm_response])
    plan_reproduction({"core_contribution": "X"}, {}, ctx=ctx)

    system_prompt = ctx.llm_client.calls[0]["system"]
    # Key phrases from _METRICS_SHAPE_INSTRUCTION must appear in the system prompt.
    assert "metrics_shape" in system_prompt
    assert "json_path" in system_prompt
    assert "metric_id" in system_prompt
