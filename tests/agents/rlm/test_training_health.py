"""Postflight training-health detection — silent OOM + insufficient train steps.

The 2026-05-29 SDAR run exited 0 while catching a backward OOM every step (no gradient
updates, all-0 metrics) and grinding for hours. The postflight now flips such a run to a
repairable failure so the loop reduces memory / trains longer instead of accepting it.
"""
from __future__ import annotations

from backend.agents.rlm.primitives import _training_health_violation, _max_train_steps
from backend.agents.rlm.failure_classifier import classify_failure, FAILURE_CLASSES


def test_max_train_steps_walks_tree():
    m = {"per_model": {"a": {"alfworld": {"sdar": {"train_steps": 15}, "grpo": {"train_steps": 30}}}}}
    assert _max_train_steps(m) == 30


def test_max_train_steps_none_when_absent():
    assert _max_train_steps({"reward": 0.0}) is None


def test_silent_oom_detected_from_logs():
    res = {"success": True, "logs": "WARNING Loss/backward OOM: CUDA out of memory", "metrics": {}}
    out = _training_health_violation(res)
    assert out is not None and out[0] == "silent_oom"


def test_no_violation_when_clean():
    res = {"success": True, "logs": "epoch 1 loss 0.5\nepoch 2 loss 0.3", "metrics": {"train_steps": 200}}
    assert _training_health_violation(res) is None


def test_insufficient_train_steps_opt_in(monkeypatch):
    monkeypatch.setenv("REPROLAB_MIN_TRAIN_STEPS", "100")
    res = {"success": True, "logs": "all good", "metrics": {"per_model": {"a": {"e": {"x": {"train_steps": 15}}}}}}
    out = _training_health_violation(res)
    assert out is not None and out[0] == "insufficient_train_steps"


def test_min_steps_disabled_by_default(monkeypatch):
    monkeypatch.delenv("REPROLAB_MIN_TRAIN_STEPS", raising=False)
    res = {"success": True, "logs": "ok", "metrics": {"train_steps": 5}}
    assert _training_health_violation(res) is None  # default 0 = disabled


def test_classifier_respects_preset_silent_oom():
    assert "silent_oom" in FAILURE_CLASSES
    cls, fix = classify_failure({
        "success": False, "failure_class": "silent_oom",
        "error": "silent_oom: ...", "logs": "",
    })
    assert cls == "silent_oom"
    assert "memory" in fix.lower() or "backward" in fix.lower()


# --- no-smokes guard (insufficient_training) ---------------------------------
# A seconds-long smoke must never be the scored reproduction. Opt-in like the
# step floor: REPROLAB_MIN_TRAIN_WALL_S (default 0 = off). The 2026-05-29 SDAR
# failure scored a 2 s smoke after real FSDP training crashed.


def test_no_smoke_guard_disabled_by_default(monkeypatch):
    monkeypatch.delenv("REPROLAB_MIN_TRAIN_WALL_S", raising=False)
    res = {"success": True, "logs": "ok", "wall_time_s": 2.0,
           "metrics": {"per_model": {"a": {"accuracy": 0.1}}}}
    assert _training_health_violation(res) is None  # floor 0 = disabled


def test_no_smoke_guard_flags_short_wall_when_enabled(monkeypatch):
    monkeypatch.setenv("REPROLAB_MIN_TRAIN_WALL_S", "120")
    res = {"success": True, "logs": "ok", "wall_time_s": 2.0,
           "metrics": {"per_model": {"a": {"accuracy": 0.1}}}}
    out = _training_health_violation(res)
    assert out is not None and out[0] == "insufficient_training"
    assert "2.0s" in out[1] and "SMOKE" in out[1]


def test_no_smoke_guard_exempts_long_run(monkeypatch):
    monkeypatch.setenv("REPROLAB_MIN_TRAIN_WALL_S", "120")
    res = {"success": True, "logs": "ok", "wall_time_s": 900.0,
           "metrics": {"per_model": {"a": {"accuracy": 0.5, "train_steps": 300}}}}
    assert _training_health_violation(res) is None  # above the floor


def test_no_smoke_guard_exempts_fast_but_many_steps(monkeypatch):
    # A genuinely fast-but-real run (many optimizer steps in little wall-clock)
    # is exempted by the step floor, so the guard cannot false-flag it.
    monkeypatch.setenv("REPROLAB_MIN_TRAIN_WALL_S", "120")
    monkeypatch.setenv("REPROLAB_MIN_REAL_TRAIN_STEPS", "5")
    res = {"success": True, "logs": "ok", "wall_time_s": 10.0,
           "metrics": {"per_model": {"a": {"train_steps": 200}}}}
    assert _training_health_violation(res) is None


def test_no_smoke_guard_ignores_missing_wall(monkeypatch):
    # No wall_time_s recorded (old result / backend without timing) → cannot judge.
    monkeypatch.setenv("REPROLAB_MIN_TRAIN_WALL_S", "120")
    res = {"success": True, "logs": "ok",
           "metrics": {"per_model": {"a": {"accuracy": 0.1}}}}
    assert _training_health_violation(res) is None


def test_insufficient_training_registered_and_classified():
    assert "insufficient_training" in FAILURE_CLASSES
    cls, fix = classify_failure({
        "success": False, "failure_class": "insufficient_training",
        "error": "insufficient_training: ran only 2.0s ...", "logs": "",
    })
    assert cls == "insufficient_training"
    assert "smoke" in fix.lower()
