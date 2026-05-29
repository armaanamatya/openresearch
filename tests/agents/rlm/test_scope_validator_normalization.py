"""Scope-metrics validator must match scope display names to sanitized metrics keys.

2026-05-29 SDAR bug: scope.models are display names ("Qwen3-1.7B-Instruct") but the
agent's metrics.json per_model keys are sanitized ("qwen3_1_7b"), so a correctly-run
model was falsely flagged `per_model_incomplete` (scope_shape_violation).
"""
from __future__ import annotations

from backend.agents.schemas import ScopeSpec
from backend.agents.rlm.paper_invariants import canonical_model_key
from backend.agents.rlm.primitives import _validate_scope_metrics


def test_canonical_model_key_strips_format_and_suffix():
    assert canonical_model_key("Qwen3-1.7B-Instruct") == "qwen3_1_7b"
    assert canonical_model_key("Qwen/Qwen2.5-3B-Instruct") == "qwen2_5_3b"
    assert canonical_model_key("qwen3_1_7b") == "qwen3_1_7b"  # idempotent


def test_scope_validator_matches_display_names_to_sanitized_keys():
    scope = ScopeSpec(models=["Qwen3-1.7B-Instruct", "Qwen2.5-3B-Instruct"])
    metrics = {"per_model": {"qwen3_1_7b": {"alfworld": {}}, "qwen2_5_3b": {"alfworld": {}}}}
    assert _validate_scope_metrics(scope, metrics) is None


def test_scope_validator_flags_genuinely_missing_model():
    scope = ScopeSpec(models=["Qwen3-1.7B-Instruct", "Qwen2.5-3B-Instruct"])
    metrics = {"per_model": {"qwen3_1_7b": {"alfworld": {}}}}  # 3B genuinely absent
    hint = _validate_scope_metrics(scope, metrics)
    assert hint is not None and "per_model_incomplete" in hint


def test_scope_validator_accepts_env_keyed_nesting():
    """Agents often write per_model[model][env] directly (no per_dataset wrapper)."""
    scope = ScopeSpec(
        models=["Qwen3-1.7B-Instruct", "Qwen2.5-3B-Instruct"],
        datasets=["ALFWorld", "Search-QA"],
    )
    metrics = {"per_model": {
        "qwen3_1_7b": {"alfworld": {"sdar": {}}, "searchqa": {"sdar": {}}},
        "qwen2_5_3b": {"alfworld": {"sdar": {}}, "searchqa": {"sdar": {}}},
    }}
    assert _validate_scope_metrics(scope, metrics) is None


def test_scope_validator_matches_dataset_display_names():
    """Multi-dataset: 'Search-QA' (scope) must match 'searchqa' (metrics key)."""
    scope = ScopeSpec(
        models=["Qwen3-1.7B-Instruct", "Qwen2.5-3B-Instruct"],
        datasets=["ALFWorld", "Search-QA"],
    )
    metrics = {"per_model": {
        "qwen3_1_7b": {"per_dataset": {"alfworld": {}, "searchqa": {}}},
        "qwen2_5_3b": {"per_dataset": {"alfworld": {}, "searchqa": {}}},
    }}
    assert _validate_scope_metrics(scope, metrics) is None
