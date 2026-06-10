"""Harness-level enforcement (2026-05-30): agent-written train.py bugs can't
silently ship a degenerate, leaf-excluded result. Covers the three fixes —
masked-code-bug reclassification, degenerate-training detection, disk guard —
plus the failure_classifier wiring. All deterministic, no LLM / no sandbox.
"""
from __future__ import annotations

import pytest

from backend.agents.rlm.failure_classifier import FAILURE_CLASSES, classify_failure
from backend.agents.rlm.primitives import (
    _RUN_EXPERIMENT_REPAIRABLE_FAILURES,
    _data_load_failure_is_code_bug,
    _degenerate_training_violation,
    _disk_floor_violation,
    _gap_in_load_failures,
    _max_train_steps,
    _reclassify_masked_code_bugs,
    _reward_curve,
    _scalar_rewards,
)


# ── FIX 1: code-bug vs data-unavailability classification ────────────────────

@pytest.mark.parametrize("err", [
    "TypeError: float() argument must be a string or a real number, not 'tuple'",
    "AttributeError: module 'alfworld.agents.environment' has no attribute 'AlfredTWEnv'",
    "HfUriError: Invalid HF URI 'hf://datasets/nq_open@x/.huggingface.yaml'",
    "Qwen/Qwen3-1.7B-Instruct is not a valid model identifier",
    "the dataset slice returned 0 rows",
    "Cannot re-initialize CUDA in forked subprocess",
    "[Errno 32] Broken pipe",
    "RuntimeError: Repository id must be 'namespace/name', got 'nq_open'",  # via phrase
    "init error: [Errno 2] No such file or directory: '.../base_config.yaml'",
    "FileNotFoundError: alfworld/config/base_config.yaml",
])
def test_code_bug_signatures_are_repairable(err):
    assert _data_load_failure_is_code_bug(err) is True


@pytest.mark.parametrize("err", [
    "HTTPError: HTTP Error 404: Not Found",
    "HTTP Error 403: Forbidden",
    "dataset is gated; requires authentication",
    "DatasetNotFoundError: Dataset 'musique/musique' doesn't exist on the Hub",
    "connection timed out",
    "ValueError: Unknown split 'test'. Should be one of ['train', 'validation'].",  # HF data
    "RuntimeError: Couldn't reach the Hugging Face Hub (connection)",               # HF transient
    "a banner mentioning ValueError appeared in the 404 body",                      # word in prose
    "",
])
def test_data_unavailable_is_not_code_bug(err):
    assert _data_load_failure_is_code_bug(err) is False


def test_datasetnotfound_with_badid_signal_is_code_bug():
    # Ambiguous DatasetNotFoundError + a bad-id/URI co-signal → code bug.
    assert _data_load_failure_is_code_bug(
        "DatasetNotFoundError ... Invalid HF URI 'hf://datasets/x'"
    ) is True


@pytest.mark.parametrize("err", [
    # F-03: a bare missing DATA path — no config/source co-signal, no exception
    # class name — is a provably-unobtainable dataset, NOT a code bug. Treating
    # it as code_bug both wastes a repair iteration AND blocks force-reduce.
    "OSError: [Errno 2] No such file or directory: '/data/webshop/items.json'",
    "could not find /data/searchqa/nq_open — no such file or directory",
])
def test_bare_missing_data_path_is_not_code_bug(err):
    assert _data_load_failure_is_code_bug(err) is False


def test_missing_config_path_is_still_code_bug():
    # A missing CONFIG/source path (the SDAR alfworld base_config.yaml shape)
    # keeps the co-signal → code_bug, so F-03 doesn't regress it.
    assert _data_load_failure_is_code_bug(
        "OSError: [Errno 2] No such file or directory: 'configs/base_config.yaml'"
    ) is True


def test_reclassify_picks_only_code_bug_entries():
    res = {"success": True, "metrics": {"data_load_failures": [
        {"dataset": "webshop", "error": "HTTP Error 404"},          # data
        {"dataset": "alfworld", "error": "TypeError: float() ... tuple"},  # code
        "AttributeError: x has no attribute y",                      # code (str form)
    ]}}
    out = _reclassify_masked_code_bugs(res)
    assert out is not None
    cls, bugs = out
    assert cls == "code_bug"
    assert any("alfworld" in b for b in bugs) and any("AttributeError" in b for b in bugs)
    assert not any("webshop" in b for b in bugs)  # genuine 404 left for the leaf scorer


def test_reclassify_all_data_returns_none():
    assert _reclassify_masked_code_bugs(
        {"metrics": {"data_load_failures": [{"dataset": "ws", "error": "404"}]}}
    ) is None


def test_reclassify_empty_returns_none():
    assert _reclassify_masked_code_bugs({"metrics": {}}) is None
    assert _reclassify_masked_code_bugs({}) is None


def test_reclassify_scans_model_load_failures():
    out = _reclassify_masked_code_bugs({"metrics": {"model_load_failures": [
        {"model": "qwen3_1_7b", "error": "Qwen/Qwen3-1.7B-Instruct is not a valid model identifier"},
    ]}})
    assert out is not None and out[0] == "code_bug" and any("qwen3" in b for b in out[1])


def test_gap_in_load_failures_does_not_launder_a_code_bug():
    # A code bug recorded as a data_load_failure must NOT force-reduce its scope gap.
    hint = "per_dataset_incomplete: model 'm' missing datasets ['ALFWorld']."
    code = {"data_load_failures": [{"dataset": "alfworld",
            "error": "init error: [Errno 2] No such file or directory: base_config.yaml"}]}
    data = {"data_load_failures": [{"dataset": "alfworld", "error": "HTTP Error 404"}]}
    assert _gap_in_load_failures(hint, code) is False   # code bug → not force-reduced
    assert _gap_in_load_failures(hint, data) is True    # genuine 404 → force-reduced


# ── FIX 2: degenerate-training detection ─────────────────────────────────────

def test_all_zero_scalar_rewards_status_ok_is_degenerate():
    out = _degenerate_training_violation({"per_model": {"m": {
        "status": "ok", "alfworld_reward": 0.0, "webshop_reward": 0.0, "searchqa_reward": 0.0,
    }}})
    assert out is not None and out[0] == "degenerate_training" and "m" in out[1]


def test_weak_but_nonzero_reward_is_not_degenerate():
    # The live Search-QA run (~0.04-0.06) must NOT be flagged.
    assert _degenerate_training_violation(
        {"per_model": {"m": {"status": "ok", "searchqa_reward": 0.043}}}
    ) is None


def test_all_zero_curve_is_degenerate():
    out = _degenerate_training_violation({"per_model": {"m": {
        "status": "ok", "training_curves": {"reward": [0.0] * 10}}}})
    assert out is not None and out[0] == "degenerate_training"


def test_constant_nonzero_curve_is_not_degenerate():
    # A converged plateau (constant but non-zero) must NOT be flagged — only |reward|~0 is.
    assert _degenerate_training_violation({"per_model": {"m": {
        "status": "ok", "training_curves": {"reward": [0.2] * 10}}}}) is None


def test_negative_reward_is_not_degenerate():
    # Negative rewards (step/length/KL penalties) are normal RL — NOT degenerate.
    assert _degenerate_training_violation({"per_model": {"m": {
        "status": "ok", "mean_reward": -1.8, "final_reward": -1.2,
        "training_curves": {"reward": [-5.0, -3.0, -1.8]}}}}) is None


def test_reward_config_fields_do_not_fake_signal():
    # reward_std / reward_scale / baseline_reward must NOT count as outcome reward,
    # so an all-zero outcome with config noise is still flagged.
    out = _degenerate_training_violation({"per_model": {"m": {
        "status": "ok", "searchqa_reward": 0.0, "reward_std": 0.0,
        "reward_scale": 1.0, "baseline_reward": 0.5}}})
    assert out is not None and out[0] == "degenerate_training"


def test_rising_curve_is_not_degenerate():
    assert _degenerate_training_violation({"per_model": {"m": {
        "status": "ok", "training_curves": {"reward": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}}}}) is None


def test_status_ok_zero_steps_is_degenerate():
    out = _degenerate_training_violation({"per_model": {"m": {"status": "ok", "train_steps": 0}}})
    assert out is not None and out[0] == "degenerate_training"


def test_non_ok_status_is_not_judged():
    # A model that already reports failure isn't re-flagged as degenerate.
    assert _degenerate_training_violation({"per_model": {"m": {"status": "failed", "searchqa_reward": 0.0}}}) is None


def test_mixed_per_model_flags_the_degenerate_one():
    out = _degenerate_training_violation({"per_model": {
        "good": {"status": "ok", "searchqa_reward": 0.3},
        "bad":  {"status": "ok", "searchqa_reward": 0.0},
    }})
    assert out is not None and "bad" in out[1]


def test_missing_per_model_returns_none():
    assert _degenerate_training_violation({}) is None
    assert _degenerate_training_violation({"per_model": "nope"}) is None


def test_short_curve_falls_back_not_flagged_on_two_points():
    # 2-point warmup is not enough to call zero-variance; with a nonzero scalar it's fine.
    assert _degenerate_training_violation({"per_model": {"m": {
        "status": "ok", "searchqa_reward": 0.2, "training_curves": {"reward": [0.2, 0.2]}}}}) is None


def test_reward_helpers():
    assert _reward_curve({"training_curves": {"reward": [1, 2, 3]}}) == [1.0, 2.0, 3.0]
    assert _reward_curve({"training_curves": [{"reward": 1}, {"reward": 2}]}) == [1.0, 2.0]
    assert sorted(_scalar_rewards({"a_reward": 0.1, "nested": {"b_reward": 0.2}, "loss": 9})) == [0.1, 0.2]
    assert _max_train_steps({"per_model": {"m": {"global_step": 42}}}) == 42  # alias
    assert _max_train_steps({"optimizer_steps": 7}) == 7


# ── FIX 3: disk guard ────────────────────────────────────────────────────────

def test_disk_floor_violation_trips_below_floor(monkeypatch, tmp_path):
    monkeypatch.setenv("REPROLAB_DISK_FLOOR_GB", "999999")  # force a violation
    out = _disk_floor_violation([str(tmp_path)])
    assert out is not None and out[0] == "disk_exhausted"


def test_disk_floor_ok_when_above(monkeypatch, tmp_path):
    monkeypatch.setenv("REPROLAB_DISK_FLOOR_GB", "0.000001")
    assert _disk_floor_violation([str(tmp_path)]) is None


def test_disk_floor_disabled_at_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("REPROLAB_DISK_FLOOR_GB", "0")
    assert _disk_floor_violation([str(tmp_path)]) is None


def test_disk_floor_ignores_bad_paths(monkeypatch):
    monkeypatch.setenv("REPROLAB_DISK_FLOOR_GB", "15")
    assert _disk_floor_violation(["", "/nonexistent/zzz"]) is None  # unresolvable → skipped


# ── failure_classifier wiring ────────────────────────────────────────────────

@pytest.mark.parametrize("cls", ["code_bug", "degenerate_training", "disk_exhausted"])
def test_new_classes_registered_and_repairable(cls):
    assert cls in FAILURE_CLASSES
    assert cls in _RUN_EXPERIMENT_REPAIRABLE_FAILURES
    got_cls, suggestion = classify_failure({"failure_class": cls})
    assert got_cls == cls and suggestion  # preset honored + has a suggested fix
