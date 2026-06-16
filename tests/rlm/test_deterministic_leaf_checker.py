"""Unit tests for the deterministic-by-construction leaf checker (A2).

Each test writes a small ``provenance.json`` / ``metrics.json`` into a tmp run
dir (the on-disk contract the producers actually write) and asserts the
checker's verdict + uniform return shape. The load-bearing case is the
NO-ANNOTATION fall-through: an un-annotated leaf must return ``None`` so the
caller routes it to the LLM (the backwards-compat guarantee).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.evals.paperbench.deterministic_leaf_checker import (
    DETERMINISTIC_CHECK_KINDS,
    check_leaf,
)


# --------------------------------------------------------------------------- #
# fixtures: build a run dir with provenance/metrics on disk.
# --------------------------------------------------------------------------- #
def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """A run dir with a code/ subdir (the implement_baseline output contract)."""
    (tmp_path / "code").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _provenance(run_dir: Path, experiments: dict, *, top_level: dict | None = None) -> None:
    payload = {"schema_version": 1, "run_id": "r1", "experiments": experiments}
    if top_level:
        payload.update(top_level)
    _write(run_dir / "code" / "provenance.json", payload)


def _metrics(run_dir: Path, payload: dict, *, subdir: str | None = None) -> Path:
    if subdir:
        target = run_dir / "code" / "outputs" / subdir / "metrics.json"
    else:
        target = run_dir / "code" / "metrics.json"
    _write(target, payload)
    return target


def _assert_shape(rec: dict, *, kind: str, leaf_id: str) -> None:
    """Every graded record must match the LLM grader's per-leaf record shape."""
    assert set(rec) == {"id", "score", "justification", "_graded", "check_kind"}
    assert rec["id"] == str(leaf_id)
    assert isinstance(rec["score"], float)
    assert 0.0 <= rec["score"] <= 1.0
    assert isinstance(rec["justification"], str) and rec["justification"]
    assert rec["_graded"] is True
    assert rec["check_kind"] == kind


# --------------------------------------------------------------------------- #
# backwards-compat: NO annotation → None (route to LLM). THE load-bearing case.
# --------------------------------------------------------------------------- #
def test_no_check_kind_falls_through_to_llm(run_dir: Path) -> None:
    leaf = {"id": "abc", "criterion": "model implements the gate g_t=sigmoid(...)"}
    assert check_leaf(leaf, run_dir) is None


def test_unknown_check_kind_falls_through(run_dir: Path) -> None:
    leaf = {"id": "abc", "check_kind": "judgment", "assertion": {"field": "x"}}
    assert check_leaf(leaf, run_dir) is None


def test_recognized_kind_but_no_assertion_falls_through(run_dir: Path) -> None:
    leaf = {"id": "abc", "check_kind": "deterministic:hparam"}
    assert check_leaf(leaf, run_dir) is None
    leaf2 = {"id": "abc", "check_kind": "deterministic:hparam", "assertion": {}}
    assert check_leaf(leaf2, run_dir) is None


def test_malformed_assertion_falls_through(run_dir: Path) -> None:
    # hparam with an unknown op, numeric with an unknown direction, artifact
    # with no glob → all malformed → route to LLM (None), NOT a 0.0 grade.
    _provenance(run_dir, {"e1": {"epochs": 45}})
    assert check_leaf(
        {"id": "1", "check_kind": "deterministic:hparam",
         "assertion": {"field": "epochs", "op": "≈≈", "value": 45}}, run_dir) is None
    assert check_leaf(
        {"id": "2", "check_kind": "deterministic:numeric",
         "assertion": {"metric_key": "acc", "direction": "best", "target": 0.9}}, run_dir) is None
    assert check_leaf(
        {"id": "3", "check_kind": "deterministic:artifact", "assertion": {"path": "x.py"}},
        run_dir) is None


# --------------------------------------------------------------------------- #
# deterministic:hparam
# --------------------------------------------------------------------------- #
def test_hparam_pass_per_experiment_field(run_dir: Path) -> None:
    # the agent writes epochs INSIDE each experiment, not at the manifest root.
    _provenance(run_dir, {"mnist__adam": {"model_key": "mnist", "epochs": 45, "seed": 42}})
    leaf = {"id": "ep-45", "check_kind": "deterministic:hparam",
            "assertion": {"field": "epochs", "op": "==", "value": 45}}
    rec = check_leaf(leaf, run_dir)
    _assert_shape(rec, kind="deterministic:hparam", leaf_id="ep-45")
    assert rec["score"] == 1.0


def test_hparam_pass_top_level_field(run_dir: Path) -> None:
    _provenance(run_dir, {"e1": {}}, top_level={"run_id": "r1"})
    leaf = {"id": "rid", "check_kind": "deterministic:hparam",
            "assertion": {"field": "run_id", "op": "==", "value": "r1"}}
    rec = check_leaf(leaf, run_dir)
    assert rec["score"] == 1.0


def test_hparam_pass_dotted_per_optimizer(run_dir: Path) -> None:
    _provenance(run_dir, {"adam": {"baseline": "adam", "per_optimizer": {"lr": 0.001}}})
    leaf = {"id": "lr", "check_kind": "deterministic:hparam",
            "assertion": {"field": "per_optimizer.lr", "op": "~=", "value": 0.001,
                          "tolerance": 1e-6}}
    rec = check_leaf(leaf, run_dir)
    assert rec["score"] == 1.0


def test_hparam_mismatch(run_dir: Path) -> None:
    _provenance(run_dir, {"e1": {"epochs": 10}})
    leaf = {"id": "ep", "check_kind": "deterministic:hparam",
            "assertion": {"field": "epochs", "op": "==", "value": 45}}
    rec = check_leaf(leaf, run_dir)
    _assert_shape(rec, kind="deterministic:hparam", leaf_id="ep")
    assert rec["score"] == 0.0
    assert "fails" in rec["justification"]


def test_hparam_missing_file(run_dir: Path) -> None:
    # no provenance.json on disk at all.
    leaf = {"id": "ep", "check_kind": "deterministic:hparam",
            "assertion": {"field": "epochs", "op": "==", "value": 45}}
    rec = check_leaf(leaf, run_dir)
    _assert_shape(rec, kind="deterministic:hparam", leaf_id="ep")
    assert rec["score"] == 0.0
    assert rec["justification"] == "provenance_missing:epochs"


def test_hparam_missing_field(run_dir: Path) -> None:
    _provenance(run_dir, {"e1": {"seed": 42}})  # has the file, lacks the field.
    leaf = {"id": "wd", "check_kind": "deterministic:hparam",
            "assertion": {"field": "weight_decay", "op": "==", "value": 5e-4}}
    rec = check_leaf(leaf, run_dir)
    assert rec["score"] == 0.0
    assert rec["justification"] == "provenance_missing:weight_decay"


def test_hparam_ops(run_dir: Path) -> None:
    _provenance(run_dir, {"e1": {"momentum": 0.9, "wd": 5e-4, "name": "Adam"}})
    cases = [
        ({"field": "momentum", "op": ">=", "value": 0.85}, 1.0),
        ({"field": "momentum", "op": "<=", "value": 0.85}, 0.0),
        ({"field": "wd", "op": "!=", "value": 0.0}, 1.0),
        ({"field": "wd", "op": "~=", "value": 0.0005, "tolerance": 1e-6}, 1.0),
        # string equality is case-insensitive.
        ({"field": "name", "op": "==", "value": "adam"}, 1.0),
        # numeric-tolerant ==: 0.9 == "0.9".
        ({"field": "momentum", "op": "==", "value": "0.9"}, 1.0),
    ]
    for assertion, expected in cases:
        leaf = {"id": "x", "check_kind": "deterministic:hparam", "assertion": assertion}
        rec = check_leaf(leaf, run_dir)
        assert rec["score"] == expected, (assertion, rec["justification"])


def test_hparam_newest_provenance_with_field_wins(run_dir: Path) -> None:
    # An older outputs/ manifest carries the field; the newest top-level one
    # does not — the checker must find the field rather than stop at the newest.
    import os
    import time
    old = run_dir / "code" / "outputs" / "r1" / "provenance.json"
    _write(old, {"experiments": {"e1": {"epochs": 45}}})
    new = run_dir / "code" / "provenance.json"
    _write(new, {"experiments": {"e1": {"seed": 42}}})  # no epochs.
    # make `new` strictly newer.
    now = time.time()
    os.utime(old, (now - 100, now - 100))
    os.utime(new, (now, now))
    leaf = {"id": "ep", "check_kind": "deterministic:hparam",
            "assertion": {"field": "epochs", "op": "==", "value": 45}}
    rec = check_leaf(leaf, run_dir)
    assert rec["score"] == 1.0


# --------------------------------------------------------------------------- #
# deterministic:artifact
# --------------------------------------------------------------------------- #
def test_artifact_found_top_level_glob(run_dir: Path) -> None:
    (run_dir / "code" / "train.py").write_text("# trainer", encoding="utf-8")
    leaf = {"id": "has-train", "check_kind": "deterministic:artifact",
            "assertion": {"glob": "train.py"}}
    rec = check_leaf(leaf, run_dir)
    _assert_shape(rec, kind="deterministic:artifact", leaf_id="has-train")
    assert rec["score"] == 1.0


def test_artifact_found_recursive_bare_name(run_dir: Path) -> None:
    nested = run_dir / "code" / "src" / "models" / "model.py"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text("class Net: ...", encoding="utf-8")
    leaf = {"id": "has-model", "check_kind": "deterministic:artifact",
            "assertion": {"glob": "model.py"}}
    rec = check_leaf(leaf, run_dir)
    assert rec["score"] == 1.0
    assert "recursive" in rec["justification"]


def test_artifact_found_any_of_list(run_dir: Path) -> None:
    (run_dir / "code" / "Dockerfile").write_text("FROM x", encoding="utf-8")
    leaf = {"id": "env-def", "check_kind": "deterministic:artifact",
            "assertion": {"glob": ["environment.yml", "Dockerfile", "requirements.txt"]}}
    rec = check_leaf(leaf, run_dir)
    assert rec["score"] == 1.0


def test_artifact_missing(run_dir: Path) -> None:
    leaf = {"id": "no-arch", "check_kind": "deterministic:artifact",
            "assertion": {"glob": "architecture_definition.py"}}
    rec = check_leaf(leaf, run_dir)
    _assert_shape(rec, kind="deterministic:artifact", leaf_id="no-arch")
    assert rec["score"] == 0.0
    assert "artifact_missing" in rec["justification"]


def test_artifact_empty_glob_list_falls_through(run_dir: Path) -> None:
    leaf = {"id": "x", "check_kind": "deterministic:artifact", "assertion": {"glob": []}}
    assert check_leaf(leaf, run_dir) is None


# --------------------------------------------------------------------------- #
# deterministic:numeric
# --------------------------------------------------------------------------- #
def _metrics_with_metric(run_dir: Path, value, *, key: str = "metric") -> None:
    # canonical aggregate shape: per_model[model][env][baseline] = {status, metric}
    _metrics(run_dir, {
        "status": "ok",
        "per_model": {"mnist": {"mnist": {"adam": {"status": "ok", key: value}}}},
    })


def test_numeric_higher_better_pass(run_dir: Path) -> None:
    _metrics_with_metric(run_dir, 0.93)
    leaf = {"id": "acc", "check_kind": "deterministic:numeric",
            "assertion": {"metric_key": "metric", "target": 0.90,
                          "tolerance": 0.01, "direction": "higher_better"}}
    rec = check_leaf(leaf, run_dir)
    _assert_shape(rec, kind="deterministic:numeric", leaf_id="acc")
    assert rec["score"] == 1.0


def test_numeric_higher_better_fail(run_dir: Path) -> None:
    _metrics_with_metric(run_dir, 0.50)
    leaf = {"id": "acc", "check_kind": "deterministic:numeric",
            "assertion": {"metric_key": "metric", "target": 0.90,
                          "tolerance": 0.01, "direction": "higher_better"}}
    rec = check_leaf(leaf, run_dir)
    assert rec["score"] == 0.0


def test_numeric_higher_better_within_tolerance(run_dir: Path) -> None:
    # value just below target but inside tolerance → pass (trend, not magnitude).
    _metrics_with_metric(run_dir, 0.895)
    leaf = {"id": "acc", "check_kind": "deterministic:numeric",
            "assertion": {"metric_key": "metric", "target": 0.90,
                          "tolerance": 0.01, "direction": "higher_better"}}
    rec = check_leaf(leaf, run_dir)
    assert rec["score"] == 1.0


def test_numeric_lower_better(run_dir: Path) -> None:
    _metrics_with_metric(run_dir, 0.12, key="loss")
    passing = {"id": "loss", "check_kind": "deterministic:numeric",
               "assertion": {"metric_key": "loss", "target": 0.20,
                             "direction": "lower_better"}}
    assert check_leaf(passing, run_dir)["score"] == 1.0
    failing = {"id": "loss", "check_kind": "deterministic:numeric",
               "assertion": {"metric_key": "loss", "target": 0.05,
                             "direction": "lower_better"}}
    assert check_leaf(failing, run_dir)["score"] == 0.0


def test_numeric_within(run_dir: Path) -> None:
    _metrics_with_metric(run_dir, 7.05, key="error_rate")
    leaf = {"id": "err", "check_kind": "deterministic:numeric",
            "assertion": {"metric_key": "error_rate", "target": 7.0,
                          "tolerance": 0.1, "direction": "within"}}
    assert check_leaf(leaf, run_dir)["score"] == 1.0
    leaf_off = {"id": "err", "check_kind": "deterministic:numeric",
                "assertion": {"metric_key": "error_rate", "target": 7.0,
                              "tolerance": 0.01, "direction": "within"}}
    assert check_leaf(leaf_off, run_dir)["score"] == 0.0


def test_numeric_trend_up_raw_list(run_dir: Path) -> None:
    _metrics(run_dir, {"per_model": {"m": {"e": {"b": {"reward_curve": [0.1, 0.3, 0.8]}}}}})
    leaf = {"id": "rew", "check_kind": "deterministic:numeric",
            "assertion": {"metric_key": "reward_curve", "direction": "trend_up"}}
    rec = check_leaf(leaf, run_dir)
    _assert_shape(rec, kind="deterministic:numeric", leaf_id="rew")
    assert rec["score"] == 1.0
    assert "rose" in rec["justification"]


def test_numeric_trend_up_fails_when_falling(run_dir: Path) -> None:
    _metrics(run_dir, {"per_model": {"m": {"e": {"b": {"reward_curve": [0.9, 0.5, 0.1]}}}}})
    leaf = {"id": "rew", "check_kind": "deterministic:numeric",
            "assertion": {"metric_key": "reward_curve", "direction": "trend_up"}}
    assert check_leaf(leaf, run_dir)["score"] == 0.0


def test_numeric_trend_down_on_summary_dict(run_dir: Path) -> None:
    # provenance _summarize_series form: {"first","last","len","sampled",...}
    _metrics(run_dir, {"per_model": {"m": {"e": {"b": {
        "loss": {"len": 100, "first": 2.3, "last": 0.05, "min": 0.05, "max": 2.3}}}}}})
    leaf = {"id": "loss", "check_kind": "deterministic:numeric",
            "assertion": {"metric_key": "loss", "direction": "trend_down"}}
    rec = check_leaf(leaf, run_dir)
    assert rec["score"] == 1.0
    assert "fell" in rec["justification"]


def test_numeric_metric_missing(run_dir: Path) -> None:
    _metrics_with_metric(run_dir, 0.9)  # has 'metric' but not 'top1_accuracy'.
    leaf = {"id": "acc", "check_kind": "deterministic:numeric",
            "assertion": {"metric_key": "top1_accuracy", "target": 0.9,
                          "direction": "higher_better"}}
    rec = check_leaf(leaf, run_dir)
    assert rec["score"] == 0.0
    assert rec["justification"] == "metric_missing:top1_accuracy"


def test_numeric_no_metrics_file(run_dir: Path) -> None:
    leaf = {"id": "acc", "check_kind": "deterministic:numeric",
            "assertion": {"metric_key": "metric", "target": 0.9,
                          "direction": "higher_better"}}
    rec = check_leaf(leaf, run_dir)
    assert rec["score"] == 0.0
    assert rec["justification"] == "metric_missing:metric"


def test_numeric_threshold_direction_without_target_falls_through(run_dir: Path) -> None:
    _metrics_with_metric(run_dir, 0.9)
    leaf = {"id": "acc", "check_kind": "deterministic:numeric",
            "assertion": {"metric_key": "metric", "direction": "higher_better"}}
    assert check_leaf(leaf, run_dir) is None  # no target → malformed → LLM.


def test_numeric_dotted_path(run_dir: Path) -> None:
    _metrics(run_dir, {"per_model": {"mnist": {"mnist": {"adam": {"metric": 0.95}}}}})
    leaf = {"id": "acc", "check_kind": "deterministic:numeric",
            "assertion": {"metric_key": "per_model.mnist.mnist.adam.metric",
                          "target": 0.9, "direction": "higher_better"}}
    assert check_leaf(leaf, run_dir)["score"] == 1.0


def test_numeric_prefers_measured_over_empty_placeholder(run_dir: Path) -> None:
    # An empty placeholder metrics.json (newer mtime) must NOT outrank an older
    # measured one — A6 ranking parity.
    import os
    import time
    measured = _metrics(run_dir, {"per_model": {"m": {"e": {"b": {"metric": 0.95}}}}},
                         subdir="old_run")
    placeholder = _metrics(run_dir, {"per_model": {"m": {}}}, subdir="new_run")
    now = time.time()
    os.utime(measured, (now - 100, now - 100))
    os.utime(placeholder, (now, now))
    leaf = {"id": "acc", "check_kind": "deterministic:numeric",
            "assertion": {"metric_key": "metric", "target": 0.9,
                          "direction": "higher_better"}}
    rec = check_leaf(leaf, run_dir)
    assert rec["score"] == 1.0  # found the measured 0.95, not the empty placeholder.


# --------------------------------------------------------------------------- #
# fail-soft: malformed JSON / bad input must never crash → graded 0.0 or None.
# --------------------------------------------------------------------------- #
def test_malformed_provenance_json_no_crash(run_dir: Path) -> None:
    (run_dir / "code" / "provenance.json").write_text("{not valid json,,", encoding="utf-8")
    leaf = {"id": "ep", "check_kind": "deterministic:hparam",
            "assertion": {"field": "epochs", "op": "==", "value": 45}}
    rec = check_leaf(leaf, run_dir)
    # file exists but unparseable → field not found → graded 0.0, no raise.
    assert rec["score"] == 0.0
    assert rec["justification"] == "provenance_missing:epochs"


def test_malformed_metrics_json_no_crash(run_dir: Path) -> None:
    _metrics(run_dir, {"per_model": {"m": {"e": {"b": {"metric": 0.9}}}}})
    # corrupt the chosen file's sibling to ensure parse errors are swallowed.
    bad = run_dir / "code" / "outputs" / "x" / "metrics.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("}}}garbage", encoding="utf-8")
    leaf = {"id": "acc", "check_kind": "deterministic:numeric",
            "assertion": {"metric_key": "metric", "target": 0.9,
                          "direction": "higher_better"}}
    rec = check_leaf(leaf, run_dir)
    assert rec["score"] == 1.0  # the good file is still selected.


def test_non_dict_leaf_returns_none(run_dir: Path) -> None:
    assert check_leaf(None, run_dir) is None  # type: ignore[arg-type]
    assert check_leaf("not a leaf", run_dir) is None  # type: ignore[arg-type]
    assert check_leaf([], run_dir) is None  # type: ignore[arg-type]


def test_leaf_id_coerced_to_str(run_dir: Path) -> None:
    (run_dir / "code" / "train.py").write_text("x", encoding="utf-8")
    leaf = {"id": 12345, "check_kind": "deterministic:artifact",
            "assertion": {"glob": "train.py"}}
    rec = check_leaf(leaf, run_dir)
    assert rec["id"] == "12345"


def test_missing_id_defaults_to_empty_str(run_dir: Path) -> None:
    (run_dir / "code" / "train.py").write_text("x", encoding="utf-8")
    leaf = {"check_kind": "deterministic:artifact", "assertion": {"glob": "train.py"}}
    rec = check_leaf(leaf, run_dir)
    assert rec["id"] == ""


def test_run_dir_accepts_str(run_dir: Path) -> None:
    (run_dir / "code" / "train.py").write_text("x", encoding="utf-8")
    leaf = {"id": "t", "check_kind": "deterministic:artifact", "assertion": {"glob": "train.py"}}
    rec = check_leaf(leaf, str(run_dir))  # str path, not Path.
    assert rec["score"] == 1.0


def test_check_kinds_constant_is_the_three_kinds() -> None:
    assert DETERMINISTIC_CHECK_KINDS == frozenset({
        "deterministic:hparam", "deterministic:artifact", "deterministic:numeric"})
