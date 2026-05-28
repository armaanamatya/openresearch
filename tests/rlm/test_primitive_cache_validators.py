"""Tests for primitive_cache hit-time schema validators.

Symptom we're guarding against: a cached primitive result that no longer
matches its expected schema (bad first call during a transient LLM outage,
contract change between primitive version and cache version, partial write
to disk) silently returns the bad answer on every hit until the cache is
manually purged.

The validators are STRUCTURAL only — they check shape, not semantic
correctness; correctness checks would defeat the cache's purpose.  On
failed validation, ``maybe_get`` skips the entry (logged warning, treated
as miss) and continues scanning subsequent JSONL entries.  ``put`` then
appends a fresh good entry, retiring the bad one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm import primitive_cache as pc


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(pc._DISABLE_ENV_VAR, raising=False)
    yield


# ---------------------------------------------------------------------------
# Per-primitive validators — positive + negative cases
# ---------------------------------------------------------------------------


def test_understand_section_valid_passes() -> None:
    r = {
        "datasets": ["MNIST"], "metrics": ["accuracy"], "training_recipe": {},
        "hardware_clues": ["GPU"], "ambiguities": [],
    }
    assert pc._v_understand_section(r) is True


def test_understand_section_missing_field_fails() -> None:
    r = {"datasets": [], "metrics": [], "training_recipe": {}, "hardware_clues": []}
    # missing ambiguities
    assert pc._v_understand_section(r) is False


def test_understand_section_non_dict_fails() -> None:
    assert pc._v_understand_section([1, 2, 3]) is False  # type: ignore[arg-type]
    assert pc._v_understand_section("string") is False  # type: ignore[arg-type]


def test_extract_hyperparameters_with_one_key_passes() -> None:
    # extract_hyperparameters returns may be partial — accept any single hparam.
    assert pc._v_extract_hyperparameters({"optimizer": "Adam"}) is True
    assert pc._v_extract_hyperparameters({"learning_rate": 0.001}) is True
    assert pc._v_extract_hyperparameters({"_meta": {"partial": True}}) is True


def test_extract_hyperparameters_empty_dict_fails() -> None:
    assert pc._v_extract_hyperparameters({}) is False


def test_extract_hyperparameters_unrelated_keys_fails() -> None:
    assert pc._v_extract_hyperparameters({"foo": 1, "bar": 2}) is False


def test_detect_environment_valid_passes() -> None:
    r = {
        "dockerfile": "FROM python:3.11", "python_version": "3.11",
        "framework": "pytorch", "framework_version": "2.2.0",
    }
    assert pc._v_detect_environment(r) is True


def test_detect_environment_missing_dockerfile_fails() -> None:
    assert pc._v_detect_environment({"python_version": "3.11", "framework": "pytorch"}) is False


def test_detect_environment_missing_framework_fails() -> None:
    assert pc._v_detect_environment({"dockerfile": "FROM ...", "python_version": "3.11"}) is False


def test_plan_reproduction_valid_passes() -> None:
    r = {"smoke_test_plan": "...", "eval_plan": "...", "datasets": ["MNIST"]}
    assert pc._v_plan_reproduction(r) is True


def test_plan_reproduction_with_just_primary_metric_passes() -> None:
    assert pc._v_plan_reproduction({"primary_metric": "accuracy"}) is True


def test_plan_reproduction_error_dict_fails() -> None:
    # We don't want to cache an error result — next attempt should retry.
    r = {"success": False, "error": "LLM timed out"}
    assert pc._v_plan_reproduction(r) is False


def test_plan_reproduction_empty_dict_fails() -> None:
    assert pc._v_plan_reproduction({}) is False


def test_verify_against_rubric_valid_passes() -> None:
    r = {"overall_score": 0.75, "target_score": 0.6, "areas": [], "leaf_count": 12}
    assert pc._v_verify_against_rubric(r) is True


def test_verify_against_rubric_missing_overall_fails() -> None:
    assert pc._v_verify_against_rubric({"target_score": 0.6, "areas": []}) is False


def test_verify_against_rubric_missing_areas_fails() -> None:
    assert pc._v_verify_against_rubric({"overall_score": 0.75, "target_score": 0.6}) is False


def test_implement_baseline_path_wrapper_valid_passes() -> None:
    # implement_baseline cache wraps str path as {_kind: "path", value: <str>}
    r = {"_kind": "path", "value": "/tmp/code/dir"}
    assert pc._v_implement_baseline(r) is True


def test_implement_baseline_ok_envelope_valid_passes() -> None:
    r = {
        "ok": True,
        "code_path": "/tmp/code/dir",
        "files": ["commands.json", "train.py"],
    }
    assert pc._v_implement_baseline(r) is True


def test_implement_baseline_ok_envelope_without_manifest_fails() -> None:
    r = {
        "ok": True,
        "code_path": "/tmp/code/dir",
        "files": ["train.py"],
    }
    assert pc._v_implement_baseline(r) is False


def test_implement_baseline_path_wrapper_empty_value_fails() -> None:
    assert pc._v_implement_baseline({"_kind": "path", "value": ""}) is False
    assert pc._v_implement_baseline({"_kind": "path", "value": None}) is False  # type: ignore[arg-type]


def test_implement_baseline_error_dict_with_message_passes() -> None:
    # A real timeout error dict carries an "error" string — that's worth caching
    # so a 4h retry returns the same fail-soft answer.
    r = {"success": False, "error": "implement_baseline: timed out after 14400 s"}
    assert pc._v_implement_baseline(r) is True


def test_implement_baseline_empty_error_dict_fails() -> None:
    # Malformed: claim of failure with no error message → re-run
    assert pc._v_implement_baseline({"success": False}) is False


# ---------------------------------------------------------------------------
# Integration: maybe_get behavior with poisoned entries
# ---------------------------------------------------------------------------


def _write_cache_entries(project_dir: Path, entries: list[dict]) -> None:
    cache_dir = project_dir / "rlm_state"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / pc._CACHE_FILENAME
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


def test_maybe_get_skips_poisoned_entry(tmp_path: Path) -> None:
    """A cached entry that fails validation should NOT be returned —
    next call recomputes and stores a fresh good entry."""
    key = pc.make_key("understand_section", payload={"text_slice": "abc"})
    bad_entry = {
        "key": key,
        "primitive": "understand_section",
        "result": {"datasets": []},  # missing 4 required keys
    }
    _write_cache_entries(tmp_path, [bad_entry])
    got = pc.maybe_get(tmp_path, "understand_section", payload={"text_slice": "abc"})
    assert got is None  # validation failed → miss


def test_maybe_get_returns_good_entry_after_poisoned(tmp_path: Path) -> None:
    """When the JSONL has both a poisoned entry and a good entry for the
    same key (older entry first), maybe_get should return the good one."""
    key = pc.make_key("understand_section", payload={"text_slice": "abc"})
    bad_entry = {
        "key": key, "primitive": "understand_section",
        "result": {"datasets": []},  # invalid
    }
    good_entry = {
        "key": key, "primitive": "understand_section",
        "result": {
            "datasets": ["MNIST"], "metrics": ["acc"], "training_recipe": {},
            "hardware_clues": [], "ambiguities": [],
        },
    }
    _write_cache_entries(tmp_path, [bad_entry, good_entry])
    got = pc.maybe_get(tmp_path, "understand_section", payload={"text_slice": "abc"})
    assert got is not None
    assert got["datasets"] == ["MNIST"]


def test_maybe_get_returns_valid_entry(tmp_path: Path) -> None:
    """Sanity: a well-formed cache entry comes back unchanged."""
    key = pc.make_key("plan_reproduction", payload={"plan": "abc"})
    good_entry = {
        "key": key, "primitive": "plan_reproduction",
        "result": {
            "smoke_test_plan": "run pytest", "eval_plan": "run eval.py",
            "datasets": ["MNIST"],
        },
    }
    _write_cache_entries(tmp_path, [good_entry])
    got = pc.maybe_get(tmp_path, "plan_reproduction", payload={"plan": "abc"})
    assert got is not None
    assert got["smoke_test_plan"] == "run pytest"


def test_maybe_get_skips_error_dict_for_plan_reproduction(tmp_path: Path) -> None:
    """A cached error result for plan_reproduction should be treated as miss
    so the next attempt retries the primitive."""
    key = pc.make_key("plan_reproduction", payload={"plan": "abc"})
    err_entry = {
        "key": key, "primitive": "plan_reproduction",
        "result": {"success": False, "error": "LLM rate-limited"},
    }
    _write_cache_entries(tmp_path, [err_entry])
    got = pc.maybe_get(tmp_path, "plan_reproduction", payload={"plan": "abc"})
    assert got is None  # error result rejected by validator


def test_maybe_get_implement_baseline_path_wrapper_round_trip(tmp_path: Path) -> None:
    """Verify the {_kind:path, value:<str>} wrapper is preserved through validation."""
    key = pc.make_key("implement_baseline", payload={"plan": "p1", "arxiv_id": "1207.0580"})
    entry = {
        "key": key, "primitive": "implement_baseline",
        "result": {"_kind": "path", "value": "/runs/proj/code"},
    }
    _write_cache_entries(tmp_path, [entry])
    got = pc.maybe_get(tmp_path, "implement_baseline", payload={"plan": "p1", "arxiv_id": "1207.0580"})
    assert got is not None
    assert got["_kind"] == "path"
    assert got["value"] == "/runs/proj/code"


# ---------------------------------------------------------------------------
# Validator dispatch table
# ---------------------------------------------------------------------------


def test_every_cacheable_primitive_has_a_validator() -> None:
    """If CACHEABLE_PRIMITIVES grows but _CACHE_VALIDATORS doesn't, hits
    on the new primitive would skip validation silently — that's a class
    of regression we want to catch at test time, not at runtime."""
    missing = pc.CACHEABLE_PRIMITIVES - set(pc._CACHE_VALIDATORS.keys())
    assert not missing, f"primitives without validators: {sorted(missing)}"


# ---------------------------------------------------------------------------
# --no-cache CLI wiring
# ---------------------------------------------------------------------------


def test_disable_env_var_short_circuits_maybe_get(tmp_path: Path, monkeypatch) -> None:
    """The CLI's --no-cache sets REPROLAB_PRIMITIVE_CACHE=disabled. Even a
    valid cache entry must be skipped when the env var is set."""
    key = pc.make_key("understand_section", payload={"text_slice": "abc"})
    good_entry = {
        "key": key, "primitive": "understand_section",
        "result": {
            "datasets": ["MNIST"], "metrics": ["acc"], "training_recipe": {},
            "hardware_clues": [], "ambiguities": [],
        },
    }
    _write_cache_entries(tmp_path, [good_entry])
    # Sanity: cache hit when env var unset
    assert pc.maybe_get(tmp_path, "understand_section", payload={"text_slice": "abc"}) is not None
    # Now flip the env var (as --no-cache does in cli.main)
    monkeypatch.setenv(pc._DISABLE_ENV_VAR, "disabled")
    assert pc.maybe_get(tmp_path, "understand_section", payload={"text_slice": "abc"}) is None
