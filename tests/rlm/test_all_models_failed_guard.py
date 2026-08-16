"""Tests for the all-models-failed postflight guard (Workstream C Fix 1).

The monolithic ``run_experiment`` path sets ``success = all(r.succeeded …)`` —
subprocess exit only. An experiment where EVERY model errored at load (e.g.
``per_model.qwen3 = {status:"failed", accuracy:0.0}``) still exits 0 and reports
``success=true`` — a latent fake-green. The existing completeness guard only
fires when per_model entries are EMPTY placeholders; an error-bearing entry with
a ``0.0`` numeric passes ``_per_model_has_measured_value``. The degenerate-training
guard only judges ``_OK_STATUSES`` models, so it skips a ``status:"failed"`` model.

This guard closes the narrow gap: per_model is non-empty but NO entry has an ok
status, yet success=true → repairable failure ``all_models_failed``.

DEFAULT-OFF behind ``OPENRESEARCH_PER_MODEL_STATUS_GATE`` (1/true/yes = ON); unset
is byte-for-byte today.
"""

from backend.agents.rlm.primitives import (
    _OK_STATUSES,
    _all_models_failed_violation,
)

_FLAG = "OPENRESEARCH_PER_MODEL_STATUS_GATE"


def _all_failed_result() -> dict:
    return {
        "metrics": {
            "per_model": {
                "qwen3": {"status": "failed", "error": "ValueError", "accuracy": 0.0},
                "qwen2": {"status": "error", "accuracy": 0.0},
            }
        }
    }


def test_all_failed_with_flag_on_fires(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    out = _all_models_failed_violation(_all_failed_result())
    assert out is not None
    cls, msg = out
    assert cls == "all_models_failed"
    # message names the offending models
    assert "qwen3" in msg
    assert "qwen2" in msg


def test_flag_off_unset_returns_none_byte_for_byte(monkeypatch):
    """CRITICAL regression: unset flag → None → all-failed row still success=true today."""
    monkeypatch.delenv(_FLAG, raising=False)
    assert _all_models_failed_violation(_all_failed_result()) is None


def test_flag_explicit_off_returns_none(monkeypatch):
    monkeypatch.setenv(_FLAG, "0")
    assert _all_models_failed_violation(_all_failed_result()) is None


def test_one_ok_status_with_flag_on_returns_none(monkeypatch):
    """≥1 ok status → no false positive."""
    monkeypatch.setenv(_FLAG, "1")
    ok_status = sorted(_OK_STATUSES)[0]
    result = {
        "metrics": {
            "per_model": {
                "qwen3": {"status": "failed", "accuracy": 0.0},
                "qwen2": {"status": ok_status, "accuracy": 0.71},
            }
        }
    }
    assert _all_models_failed_violation(result) is None


def test_empty_per_model_with_flag_on_returns_none(monkeypatch):
    """Empty per_model is the completeness guard's job, not this one."""
    monkeypatch.setenv(_FLAG, "1")
    assert _all_models_failed_violation({"metrics": {"per_model": {}}}) is None


def test_no_per_model_key_with_flag_on_returns_none(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    assert _all_models_failed_violation({"metrics": {"status": "done"}}) is None


def test_per_model_not_a_dict_with_flag_on_returns_none(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    assert _all_models_failed_violation({"metrics": {"per_model": ["qwen3"]}}) is None


def test_no_metrics_with_flag_on_returns_none(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    assert _all_models_failed_violation({}) is None


# ---------------------------------------------------------------------------
# CELLS-ROUTE nesting (2026-08-02 regression): the cells route builds a NESTED
# per_model — ``per_model[model][env][baseline] = {status, metrics}`` (cell_matrix
# aggregate_cell_metrics) — so the leaf ``status`` lives 2 levels below the model
# key, NOT at the model level. The guard must descend to the leaf (mirroring the
# cells-route any_ok rule), else it FALSE-POSITIVES on every completed cells run
# (real incident: base_rn ResNet, 11/11 leaf cells status=ok, wrongly flagged
# all_models_failed → success flipped false → verdict clamped failed).
# ---------------------------------------------------------------------------


def _nested_cells_result(leaf_statuses: dict) -> dict:
    """per_model[model]['cifar10']['residual'] = {status, test_error_pct}.

    ``leaf_statuses`` maps model_key -> status string.
    """
    per_model = {}
    for model_key, status in leaf_statuses.items():
        per_model[model_key] = {
            "cifar10": {"residual": {"status": status, "test_error_pct": 9.5, "cell_id": model_key}}
        }
    return {"metrics": {"status": "complete", "per_model": per_model}}


def test_nested_all_ok_leaves_with_flag_on_returns_none(monkeypatch):
    """The base_rn false-positive: nested per_model, every leaf ok → MUST NOT fire."""
    monkeypatch.setenv(_FLAG, "1")
    ok = sorted(_OK_STATUSES)[0]
    result = _nested_cells_result({"plain20": ok, "resnet20": ok, "resnet110": ok})
    assert _all_models_failed_violation(result) is None


def test_nested_all_failed_leaves_with_flag_on_fires(monkeypatch):
    """Anti-fabrication preserved: nested per_model, every leaf failed → MUST fire."""
    monkeypatch.setenv(_FLAG, "1")
    result = _nested_cells_result({"plain20": "failed", "resnet20": "error", "resnet110": "failed"})
    out = _all_models_failed_violation(result)
    assert out is not None
    cls, _msg = out
    assert cls == "all_models_failed"


def test_nested_mixed_one_ok_leaf_with_flag_on_returns_none(monkeypatch):
    """≥1 ok leaf anywhere in the nested tree → no false positive."""
    monkeypatch.setenv(_FLAG, "1")
    ok = sorted(_OK_STATUSES)[0]
    result = _nested_cells_result({"plain20": "failed", "resnet20": ok, "resnet110": "failed"})
    assert _all_models_failed_violation(result) is None


def test_nested_all_failed_flag_off_returns_none_byte_for_byte(monkeypatch):
    """OFF stays byte-identical even for the nested shape."""
    monkeypatch.delenv(_FLAG, raising=False)
    result = _nested_cells_result({"plain20": "failed", "resnet20": "failed"})
    assert _all_models_failed_violation(result) is None
