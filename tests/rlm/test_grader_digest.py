"""Unit tests for backend.evals.paperbench.grader_digest (Workstream A6).

Properties under test:
  * a wide grid (20 models) -> digest contains ALL 20 (no cell vanishes)
  * a placeholder model {} is distinguished from a measured one
  * per_model_has_measured_value True / False cases
  * both metrics shapes: 3-level m/e/b and flat m->cell
  * headline metric + n_epochs resolution
  * deterministic, stable ordering
"""

from __future__ import annotations

from backend.evals.paperbench.grader_digest import (
    build_grader_digest,
    per_model_has_measured_value,
)


def _cell(err: float, epochs: int = 100, status: str = "ok") -> dict:
    return {
        "status": status,
        "test_error_pct": err,
        "best_test_accuracy": 1.0 - err / 100.0,
        "epochs_run": epochs,
        "history": {
            "epoch": list(range(epochs)),
            "train_loss": [1.0 / (i + 1) for i in range(epochs)],
        },
    }


def _wide_grid(n: int = 20) -> dict:
    per_model = {}
    for i in range(n):
        per_model[f"model_{i:02d}"] = {"cifar10": {"plain": _cell(9.0 + i * 0.01, epochs=100 + i)}}
    return {"per_model": per_model}


# --- wide grid: no cell vanishes -------------------------------------------

def test_wide_grid_all_cells_present():
    dig = build_grader_digest(_wide_grid(20))
    assert dig["count"] == 20
    keys = {c["model_key"] for c in dig["cells"]}
    assert keys == {f"model_{i:02d}" for i in range(20)}


def test_wide_grid_every_cell_has_headline_and_epochs():
    dig = build_grader_digest(_wide_grid(20))
    for c in dig["cells"]:
        assert c["headline_metric"] is not None
        assert c["headline_metric"]["name"] == "test_error_pct"
        assert c["n_epochs"] >= 100
        assert c["measured"] is True


def test_digest_is_deterministically_sorted():
    dig = build_grader_digest(_wide_grid(20))
    triples = [(c["model_key"], c["env"], c["baseline"]) for c in dig["cells"]]
    assert triples == sorted(triples, key=lambda t: tuple(x or "" for x in t))


# --- placeholder vs measured ----------------------------------------------

def test_placeholder_cell_distinguished_from_measured():
    metrics = {
        "per_model": {
            "real": {"cifar10": {"plain": _cell(9.1)}},
            "placeholder": {},  # empty model-level placeholder
            "declared_only": {"cifar10": {"plain": {"status": "failed", "test_error_pct": None}}},
        }
    }
    dig = build_grader_digest(metrics)
    by_model = {c["model_key"]: c for c in dig["cells"]}
    # All three surface (no cell vanishes) ...
    assert set(by_model) == {"real", "placeholder", "declared_only"}
    # ... but only the real one is measured.
    assert by_model["real"]["measured"] is True
    assert by_model["placeholder"]["measured"] is False
    assert by_model["declared_only"]["measured"] is False
    assert by_model["declared_only"]["headline_metric"] is None
    assert dig["measured_count"] == 1


# --- per_model_has_measured_value ------------------------------------------

def test_has_measured_true_on_real_grid():
    assert per_model_has_measured_value(_wide_grid(3)) is True


def test_has_measured_false_on_empty_placeholder():
    assert per_model_has_measured_value({"per_model": {"m": {}}}) is False


def test_has_measured_false_on_none_only_cell():
    metrics = {"per_model": {"m": {"e": {"b": {"status": "failed", "test_error_pct": None, "acc": None}}}}}
    assert per_model_has_measured_value(metrics) is False


def test_has_measured_true_when_history_only():
    metrics = {"per_model": {"m": {"e": {"b": {"status": "ok", "history": {"train_loss": [1.0, 0.5]}}}}}}
    assert per_model_has_measured_value(metrics) is True


def test_has_measured_false_no_per_model():
    assert per_model_has_measured_value({"comparison": {}}) is False
    assert per_model_has_measured_value({}) is False


def test_has_measured_false_on_non_dict():
    assert per_model_has_measured_value(None) is False  # type: ignore[arg-type]
    assert per_model_has_measured_value([]) is False  # type: ignore[arg-type]


def test_placeholder_does_not_outrank_measured():
    # The ranking foot-gun this fixes: a truthy-but-empty placeholder must NOT
    # read as measured (which would let it outrank genuine older data).
    placeholder = {"per_model": {"m": {}}}  # truthy dict, no measurement
    measured = {"per_model": {"m": {"e": {"b": _cell(9.0)}}}}
    assert per_model_has_measured_value(placeholder) is False
    assert per_model_has_measured_value(measured) is True


# --- both metrics shapes ---------------------------------------------------

def test_flat_model_level_cell_shape():
    # Older shape: per_model[model] is itself the cell (carries status directly).
    metrics = {
        "per_model": {
            "mnist_logreg": {"status": "ok", "test_accuracy": 0.92, "epochs_run": 30},
            "mnist_mlp": {"status": "ok", "test_accuracy": 0.97, "epochs_run": 30},
        }
    }
    dig = build_grader_digest(metrics)
    assert dig["count"] == 2
    by_model = {c["model_key"]: c for c in dig["cells"]}
    assert by_model["mnist_logreg"]["env"] is None
    assert by_model["mnist_logreg"]["baseline"] is None
    assert by_model["mnist_logreg"]["headline_metric"]["name"] == "test_accuracy"
    assert by_model["mnist_logreg"]["headline_metric"]["value"] == 0.92
    assert by_model["mnist_logreg"]["measured"] is True


def test_three_level_shape_env_and_baseline_recorded():
    metrics = {"per_model": {"plain_20": {"cifar10": {"plain": _cell(9.22, epochs=164)}}}}
    dig = build_grader_digest(metrics)
    assert dig["count"] == 1
    c = dig["cells"][0]
    assert c["model_key"] == "plain_20"
    assert c["env"] == "cifar10"
    assert c["baseline"] == "plain"
    assert c["n_epochs"] == 164


def test_mixed_shapes_in_one_grid():
    metrics = {
        "per_model": {
            "flat_model": {"status": "ok", "accuracy": 0.8},
            "nested_model": {"cifar10": {"plain": _cell(9.0)}},
        }
    }
    dig = build_grader_digest(metrics)
    assert dig["count"] == 2
    by_model = {c["model_key"]: c for c in dig["cells"]}
    assert by_model["flat_model"]["env"] is None
    assert by_model["nested_model"]["env"] == "cifar10"


# --- headline / n_epochs edge cases ----------------------------------------

def test_headline_priority_order():
    # test_error_pct outranks final_train_loss when both present.
    cell = {"status": "ok", "final_train_loss": 0.03, "test_error_pct": 9.1}
    dig = build_grader_digest({"per_model": {"m": {"e": {"b": cell}}}})
    assert dig["cells"][0]["headline_metric"]["name"] == "test_error_pct"


def test_headline_fallback_to_any_numeric():
    # No priority key present -> first non-bookkeeping numeric (sorted key).
    cell = {"status": "ok", "custom_metric_z": 0.5, "custom_metric_a": 0.9}
    dig = build_grader_digest({"per_model": {"m": {"e": {"b": cell}}}})
    hm = dig["cells"][0]["headline_metric"]
    assert hm["name"] == "custom_metric_a"  # sorted-key deterministic
    assert hm["value"] == 0.9


def test_n_epochs_from_history_when_no_explicit():
    cell = {"status": "ok", "test_error_pct": 9.0, "history": {"train_loss": [1, 2, 3, 4]}}
    dig = build_grader_digest({"per_model": {"m": {"e": {"b": cell}}}})
    assert dig["cells"][0]["n_epochs"] == 4


def test_n_epochs_explicit_wins():
    cell = {"status": "ok", "epochs_run": 164, "history": {"train_loss": [1, 2, 3]}}
    dig = build_grader_digest({"per_model": {"m": {"e": {"b": cell}}}})
    assert dig["cells"][0]["n_epochs"] == 164


def test_bool_is_not_a_measured_numeric():
    # A bool field (e.g. use_residual) must not count as a headline/measurement.
    cell = {"status": "ok", "use_residual": True}
    assert per_model_has_measured_value({"per_model": {"m": {"e": {"b": cell}}}}) is False
    dig = build_grader_digest({"per_model": {"m": {"e": {"b": cell}}}})
    assert dig["cells"][0]["headline_metric"] is None


# --- empty / odd inputs ----------------------------------------------------

def test_empty_metrics_returns_empty_digest():
    dig = build_grader_digest({})
    assert dig["count"] == 0
    assert dig["cells"] == []
    assert dig["measured_count"] == 0


def test_non_dict_metrics_returns_empty_digest():
    dig = build_grader_digest(None)  # type: ignore[arg-type]
    assert dig["count"] == 0
