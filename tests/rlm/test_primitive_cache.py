"""Tests for backend.agents.rlm.primitive_cache.

Pinned guarantees:

  * make_key is deterministic and version-prefixed
  * Allow-list gate: only CACHEABLE_PRIMITIVES are cached
  * Disable env var (OPENRESEARCH_PRIMITIVE_CACHE=disabled) short-circuits both put and get
  * maybe_get returns None on every fail-soft path (no file, no dir, corrupt JSON, wrong shape)
  * put is fail-soft: corrupted FS path doesn't raise
  * Round-trip: put -> maybe_get returns the same dict
  * Different payload → different key → cache miss
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
# make_key
# ---------------------------------------------------------------------------


def test_make_key_deterministic() -> None:
    a = pc.make_key("understand_section", payload={"text_slice": "abc"})
    b = pc.make_key("understand_section", payload={"text_slice": "abc"})
    assert a == b


def test_make_key_includes_version_and_primitive() -> None:
    key = pc.make_key("understand_section", payload={"x": 1})
    assert key.startswith("v1:understand_section:")


def test_make_key_distinct_for_different_payload() -> None:
    a = pc.make_key("understand_section", payload={"text_slice": "abc"})
    b = pc.make_key("understand_section", payload={"text_slice": "abcd"})
    assert a != b


def test_make_key_canonical_for_dict_key_order() -> None:
    a = pc.make_key("plan_reproduction", payload={"a": 1, "b": 2})
    b = pc.make_key("plan_reproduction", payload={"b": 2, "a": 1})
    assert a == b  # JSON sort_keys=True canonicalizes


# ---------------------------------------------------------------------------
# Allow-list gate
# ---------------------------------------------------------------------------


def test_non_cacheable_primitive_returns_none_on_get(tmp_path: Path) -> None:
    # run_experiment is NOT in CACHEABLE_PRIMITIVES — must not be cached.
    # (implement_baseline was added in Lane A — pick a primitive that is
    # genuinely never cached: run_experiment depends on real-world state.)
    pc.put(tmp_path, "run_experiment", payload={"x": 1}, result={"y": 2})
    assert pc.maybe_get(tmp_path, "run_experiment", payload={"x": 1}) is None


def test_cacheable_primitives_listed() -> None:
    expected = {
        "understand_section",
        "extract_hyperparameters",
        "detect_environment",
        "plan_reproduction",
        "verify_against_rubric",
        # Lane A — warm-retry cache.  implement_baseline became cacheable so
        # kill-and-relaunch can reuse the prior attempt's code.
        "implement_baseline",
    }
    assert pc.CACHEABLE_PRIMITIVES == expected


# ---------------------------------------------------------------------------
# Disable env var
# ---------------------------------------------------------------------------


def test_disable_via_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(pc._DISABLE_ENV_VAR, "disabled")
    pc.put(tmp_path, "understand_section", payload={"x": 1}, result={"y": 2})
    assert pc.maybe_get(tmp_path, "understand_section", payload={"x": 1}) is None
    # File should never have been written either
    cache_file = tmp_path / "rlm_state" / pc._CACHE_FILENAME
    assert not cache_file.exists()


def test_enabled_default() -> None:
    assert pc.is_enabled() is True


def test_disable_value_case_insensitive(monkeypatch) -> None:
    monkeypatch.setenv(pc._DISABLE_ENV_VAR, "DISABLED")
    assert pc.is_enabled() is False
    monkeypatch.setenv(pc._DISABLE_ENV_VAR, "Disabled")
    assert pc.is_enabled() is False


# ---------------------------------------------------------------------------
# maybe_get fail-soft paths
# ---------------------------------------------------------------------------


def test_maybe_get_no_project_dir(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does-not-exist"
    assert pc.maybe_get(nonexistent, "understand_section", payload={"x": 1}) is None


def test_maybe_get_no_cache_file(tmp_path: Path) -> None:
    # project dir exists, rlm_state/ does not
    (tmp_path / "rlm_state").mkdir()
    assert pc.maybe_get(tmp_path, "understand_section", payload={"x": 1}) is None


def test_maybe_get_skips_corrupt_lines(tmp_path: Path) -> None:
    cache_dir = tmp_path / "rlm_state"
    cache_dir.mkdir()
    cache_file = cache_dir / pc._CACHE_FILENAME
    # Mix of corrupt + valid lines; the valid one should still be readable.
    # Result shape must pass the hit-time schema validator (5 understand_section keys).
    key = pc.make_key("understand_section", payload={"x": 1})
    valid_result = {
        "datasets": ["MNIST"], "metrics": ["accuracy"], "training_recipe": {},
        "hardware_clues": [], "ambiguities": [],
    }
    cache_file.write_text(
        "not-json\n"
        '{"key": "wrong"}\n'
        f'{{"key": "{key}", "primitive": "understand_section", "result": {json.dumps(valid_result)}}}\n'
    )
    got = pc.maybe_get(tmp_path, "understand_section", payload={"x": 1})
    assert got == valid_result


def test_maybe_get_skips_validator_failures(tmp_path: Path) -> None:
    """Lane M: hit-time schema validation rejects malformed cached results.
    A cached entry that doesn't match the primitive's contract is treated
    as a miss, even if its key matches."""
    cache_dir = tmp_path / "rlm_state"
    cache_dir.mkdir()
    cache_file = cache_dir / pc._CACHE_FILENAME
    key = pc.make_key("understand_section", payload={"x": 1})
    # {"y": 7} is a structurally invalid understand_section result.
    cache_file.write_text(
        f'{{"key": "{key}", "primitive": "understand_section", "result": {{"y": 7}}}}\n'
    )
    assert pc.maybe_get(tmp_path, "understand_section", payload={"x": 1}) is None


def test_maybe_get_wrong_payload_misses(tmp_path: Path) -> None:
    pc.put(tmp_path, "understand_section", payload={"x": 1}, result={"y": 2})
    assert pc.maybe_get(tmp_path, "understand_section", payload={"x": 99}) is None


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_put_then_get_roundtrip(tmp_path: Path) -> None:
    # Result shape must pass the hit-time schema validator.
    valid = {
        "datasets": ["MNIST"], "metrics": ["accuracy"], "training_recipe": {},
        "hardware_clues": [], "ambiguities": [],
    }
    pc.put(tmp_path, "understand_section", payload={"x": 1}, result=valid)
    out = pc.maybe_get(tmp_path, "understand_section", payload={"x": 1})
    assert out == valid


def test_put_creates_rlm_state(tmp_path: Path) -> None:
    pc.put(tmp_path, "understand_section", payload={"x": 1}, result={"y": 2})
    assert (tmp_path / "rlm_state" / pc._CACHE_FILENAME).exists()


def test_put_appends(tmp_path: Path) -> None:
    pc.put(tmp_path, "understand_section", payload={"x": 1}, result={"y": 2})
    pc.put(tmp_path, "extract_hyperparameters", payload={"x": 2}, result={"y": 3})
    contents = (tmp_path / "rlm_state" / pc._CACHE_FILENAME).read_text().strip().split("\n")
    assert len(contents) == 2


def test_put_rejects_non_dict_result(tmp_path: Path) -> None:
    pc.put(tmp_path, "understand_section", payload={"x": 1}, result="not a dict")  # type: ignore[arg-type]
    assert pc.maybe_get(tmp_path, "understand_section", payload={"x": 1}) is None


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_empty(tmp_path: Path) -> None:
    assert pc.stats(tmp_path) == {}


def test_stats_counts_per_primitive(tmp_path: Path) -> None:
    pc.put(tmp_path, "understand_section", payload={"i": 1}, result={"x": 1})
    pc.put(tmp_path, "understand_section", payload={"i": 2}, result={"x": 2})
    pc.put(tmp_path, "extract_hyperparameters", payload={"j": 1}, result={"x": 3})
    out = pc.stats(tmp_path)
    assert out == {"understand_section": 2, "extract_hyperparameters": 1}
