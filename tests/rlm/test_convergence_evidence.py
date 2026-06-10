"""Tests for convergence_evidence — the enforced structured-evidence backstop.

Anchored on the 2026-06-09 Adam reproduction failure (0.7364): the eval-protocol area
crashed to 0.21 because sweep results / regret time-series / per-epoch curves never reached
structured metrics and a figure used the wrong axis. These tests pin the behaviour that turns
each of those 0.0 leaves into an enforced requirement.
"""

from __future__ import annotations

import os

import pytest

from backend.agents.rlm import convergence_evidence as ce


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setenv(ce.ENV_FLAG, "1")


@pytest.fixture
def disarmed(monkeypatch):
    monkeypatch.delenv(ce.ENV_FLAG, raising=False)


# --------------------------------------------------------------------------- flag

def test_is_enabled_default_off(disarmed):
    assert ce.is_enabled() is False


@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("on", True), ("yes", True),
    ("0", False), ("false", False), ("", False), ("off", False), ("no", False),
])
def test_is_enabled_values(monkeypatch, val, expected):
    monkeypatch.setenv(ce.ENV_FLAG, val)
    assert ce.is_enabled() is expected


# --------------------------------------------------------------------------- iters_to_threshold

def test_iters_to_threshold_descending_loss():
    x = [0, 1, 2, 3, 4]
    y = [2.0, 1.5, 1.0, 0.5, 0.2]
    # first epoch where loss <= 1.0 is x=2
    assert ce.iterations_to_threshold(x, y, 1.0) == 2


def test_iters_to_threshold_interpolates():
    x = [0, 10]
    y = [2.0, 0.0]
    # threshold 1.0 sits halfway → x=5
    assert ce.iterations_to_threshold(x, y, 1.0) == pytest.approx(5.0)


def test_iters_to_threshold_never_crossed_is_none():
    assert ce.iterations_to_threshold([0, 1, 2], [3.0, 2.9, 2.8], 1.0) is None


def test_iters_to_threshold_ascending_accuracy():
    x = [0, 1, 2, 3]
    y = [0.2, 0.5, 0.8, 0.9]
    assert ce.iterations_to_threshold(x, y, 0.8, descending=False) == 2


def test_auc_trapezoid():
    assert ce.area_under_curve([0, 1, 2], [1.0, 1.0, 1.0]) == pytest.approx(2.0)


def test_auc_too_short_is_none():
    assert ce.area_under_curve([0], [1.0]) is None


# --------------------------------------------------------------------------- derive

def test_derive_orders_adam_faster():
    # Adam reaches low loss faster; SGD ends at the same final but later → lower iters + lower AUC
    history = {
        "adam": {"epoch": [0, 1, 2, 3, 4], "train_loss": [2.0, 0.8, 0.4, 0.3, 0.25]},
        "sgd": {"epoch": [0, 1, 2, 3, 4], "train_loss": [2.0, 1.6, 1.1, 0.6, 0.25]},
    }
    d = ce.derive_convergence_metrics(history)
    assert set(d) == {"adam", "sgd"}
    # both reach the common (slowest-final-eased) target, Adam in fewer iters
    assert d["adam"]["iters_to_threshold"] < d["sgd"]["iters_to_threshold"]
    assert d["adam"]["auc"] < d["sgd"]["auc"]
    assert d["adam"]["final"] == pytest.approx(0.25)


def test_derive_empty_or_malformed_is_empty():
    assert ce.derive_convergence_metrics({}) == {}
    assert ce.derive_convergence_metrics({"adam": "not a dict"}) == {}
    assert ce.derive_convergence_metrics("nope") == {}  # type: ignore[arg-type]


def test_derive_drops_nan():
    history = {"m": {"epoch": [0, 1, 2], "train_loss": [2.0, float("nan"), 0.5]}}
    d = ce.derive_convergence_metrics(history)
    # NaN dropped → aligned to the shorter finite prefix, still produces a final
    assert d["m"]["final"] is not None


# --------------------------------------------------------------------------- missing_structured_evidence

def test_missing_off_flag_is_empty(disarmed):
    # even with everything absent, the disarmed path enforces nothing
    assert ce.missing_structured_evidence({}, {"history_methods": ["adam"], "sweeps": ["s"]}) == []


def test_missing_none_requirement_is_empty(armed):
    assert ce.missing_structured_evidence({"anything": 1}, None) == []


def test_missing_history_detected(armed):
    metrics = {"final_acc": 0.9}  # no history at all
    miss = ce.missing_structured_evidence(metrics, {"history_methods": ["adam", "sgd"]})
    assert any("history.adam" in m for m in miss)
    assert any("history.sgd" in m for m in miss)


def test_present_history_passes(armed):
    metrics = {"history": {"adam": {"epoch": [0, 1], "train_loss": [2.0, 0.5]}}}
    miss = ce.missing_structured_evidence(metrics, {"history_methods": ["adam"]})
    assert miss == []


def test_present_nested_history_passes(armed):
    # Adam-shaped: history nested per-experiment, optimizers one level down
    metrics = {"history": {
        "mnist_lr": {"adam": {"epoch": [0, 1, 2], "train_nll": [2.3, 0.9, 0.31]},
                     "sgd_nesterov": {"epoch": [0, 1, 2], "train_nll": [2.3, 1.1, 0.33]}},
    }}
    miss = ce.missing_structured_evidence(
        metrics, {"history_methods": ["adam", "sgd_nesterov"]}
    )
    assert miss == []


def test_missing_method_in_nested_history_detected(armed):
    metrics = {"history": {"mnist_lr": {"adam": {"epoch": [0, 1], "train_nll": [2.3, 0.3]}}}}
    # adagrad absent from the nested history → flagged
    miss = ce.missing_structured_evidence(metrics, {"history_methods": ["adam", "adagrad"]})
    assert any("adagrad" in m for m in miss)
    assert not any("history.adam" in m for m in miss)


def test_missing_sweep_detected(armed):
    miss = ce.missing_structured_evidence({"history": {}}, {"sweeps": ["vae_lr_sweep"]})
    assert any("vae_lr_sweep" in m for m in miss)


def test_present_sweep_passes(armed):
    metrics = {"vae_lr_sweep": {"lr1e-3": {"elbo": -100}}}
    assert ce.missing_structured_evidence(metrics, {"sweeps": ["vae_lr_sweep"]}) == []


def test_missing_series_detected_when_scalar(armed):
    # regret present but only as a single scalar → must be a time-series
    metrics = {"regret_final_cumulative": 0.0542}
    miss = ce.missing_structured_evidence(metrics, {"series": ["regret"]})
    assert any("regret" in m for m in miss)


def test_series_present_as_array_passes(armed):
    metrics = {"regret": {"t": [1, 2, 3], "cumulative": [0.1, 0.08, 0.05]}}
    assert ce.missing_structured_evidence(metrics, {"series": ["regret"]}) == []


def test_missing_fail_soft_on_bad_requirement(armed):
    # a malformed requirement must not raise — degrade to nothing-missing
    assert ce.missing_structured_evidence({}, {"history_methods": "not-a-list"}) == []


# --------------------------------------------------------------------------- figure axis

def test_figure_axis_off_flag_true(disarmed):
    assert ce.figure_axis_matches({"axis": {"x": {"scale": "linear"}}}, {"x": {"scale": "log"}}) is True


def test_figure_axis_wrong_scale_caught(armed):
    # the Adam Fig-4 failure: x is linear 'Epoch', paper needs log10(alpha)
    sidecar = {"axis": {"x": {"label": "Epoch", "scale": "linear"}}}
    assert ce.figure_axis_matches(sidecar, {"x": {"scale": "log"}}) is False


def test_figure_axis_correct_passes(armed):
    sidecar = {"axis": {"x": {"label": "log10(alpha)", "scale": "log"},
                        "y": {"label": "Test loss"}}}
    assert ce.figure_axis_matches(
        sidecar, {"x": {"scale": "log"}, "y": {"label_contains": "loss"}}
    ) is True


def test_figure_axis_missing_sidecar_caught(armed):
    assert ce.figure_axis_matches(None, {"x": {"scale": "log"}}) is False
