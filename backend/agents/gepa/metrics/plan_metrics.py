"""Deterministic scoring functions for plan_reproduction outputs."""
from __future__ import annotations

_REQUIRED_CONTRACT_FIELDS = frozenset({
    "baseline_plan", "smoke_test_plan", "full_run_plan",
    "expected_artifacts", "dataset_plan", "evaluation_plan",
    "verification_checklist", "metrics_shape",
})


def metrics_shape_score(
    contract: object,
    expected_metric_ids: list[str],
) -> tuple[float, str]:
    """Score how many expected metric IDs appear in contract['metrics_shape'].

    Returns (score 0.0-1.0, feedback_string).
    score = fraction of expected_metric_ids found in metrics_shape[*].metric_id.
    If expected_metric_ids is empty, returns (1.0, "no constraints").
    """
    if not isinstance(contract, dict):
        return 0.0, "invalid contract: not a dict"
    if not expected_metric_ids:
        return 1.0, "no metric_id constraints"
    metrics_shape = contract.get("metrics_shape")
    if not isinstance(metrics_shape, list) or len(metrics_shape) == 0:
        return 0.0, f"missing metrics_shape; expected ids: {expected_metric_ids}"
    declared_ids = {
        entry.get("metric_id", "") for entry in metrics_shape if isinstance(entry, dict)
    }
    found = [mid for mid in expected_metric_ids if mid in declared_ids]
    missing = [mid for mid in expected_metric_ids if mid not in declared_ids]
    score = len(found) / len(expected_metric_ids)
    if missing:
        feedback = f"missing metric_ids: {missing}; declared: {sorted(declared_ids)}"
    else:
        feedback = f"all {len(found)} metric_ids found"
    return score, feedback


def contract_completeness(contract: object) -> float:
    """Score structural completeness of a ReproductionContract dict.

    Returns fraction of required fields that are present and non-empty.
    """
    if not isinstance(contract, dict):
        return 0.0
    scores = []
    for field in _REQUIRED_CONTRACT_FIELDS:
        val = contract.get(field)
        if val is None:
            scores.append(0.0)
        elif isinstance(val, (str, list, dict)) and len(val) == 0:
            scores.append(0.0)
        else:
            scores.append(1.0)
    return sum(scores) / len(scores)


def combined_plan_score(
    contract: object,
    expected_metric_ids: list[str],
    *,
    shape_weight: float = 0.6,
    completeness_weight: float = 0.4,
) -> tuple[float, str]:
    """Combined score: shape coverage + structural completeness."""
    shape, fb = metrics_shape_score(contract, expected_metric_ids)
    completeness = contract_completeness(contract)
    score = shape_weight * shape + completeness_weight * completeness
    return score, f"shape={shape:.2f} completeness={completeness:.2f} | {fb}"
