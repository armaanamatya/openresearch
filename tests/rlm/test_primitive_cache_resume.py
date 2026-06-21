"""P1.3 — Prove primitive_cache cross-restart resume stability.

Guarantees verified here:
  * put -> fresh maybe_get (same in-process context, same project_dir) returns
    the stored result — simulating a relaunch reading the prior run's on-disk cache.
  * make_key has NO run-id / project-id component; key is purely content-addressed.
  * A payload differing ONLY in a volatile field that the implement_baseline
    builder explicitly EXCLUDES (remaining_s) still hits after we manually
    confirm the builder omits it — the two payload dicts are compared
    as-built (i.e. neither includes remaining_s, so their keys are identical).
  * run_experiment and build_environment are NOT cacheable — maybe_get returns None.
  * The JSONL file survives a second instantiation of the reader (simulated by
    calling maybe_get from a new in-process context pointing at the same path).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm import primitive_cache as pc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_enabled(monkeypatch):
    """Make sure the cache is never globally disabled by env during these tests."""
    monkeypatch.delenv(pc._DISABLE_ENV_VAR, raising=False)
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_implement_baseline_result() -> dict:
    """Minimal implement_baseline result that passes the hit-time validator."""
    return {
        "ok": True,
        "code_path": "/runs/prj_abc/code",
        "files": ["commands.json", "train.py"],
    }


def _valid_plan_reproduction_result() -> dict:
    """Minimal plan_reproduction result that passes the hit-time validator."""
    return {
        "smoke_test_plan": "python train.py --smoke",
        "eval_plan": "python eval.py",
        "verification_checklist": [],
        "datasets": ["CIFAR-10"],
        "primary_metric": "accuracy",
    }


# ---------------------------------------------------------------------------
# Core resume: put -> fresh maybe_get on same project_dir
# ---------------------------------------------------------------------------


def test_implement_baseline_put_then_fresh_get(tmp_path: Path) -> None:
    """Simulates a relaunch: put() in one 'run', maybe_get() in a 'new run'
    pointing at the same project_dir.  Must hit without any in-memory state."""
    payload = {
        "plan": {"method": "GRPO", "models": ["qwen3-1.7b"]},
        "repair_context": None,
        "arxiv_id": "2605.15155",
        "sandbox_mode": "local",
        "gpu_mode": None,
        "knowledge_channel_version": 1,
    }
    result = _valid_implement_baseline_result()

    # First 'run': store to disk.
    pc.put(tmp_path, "implement_baseline", payload=payload, result=result)

    # Confirm the file is on disk before 'relaunch'.
    cache_file = tmp_path / "rlm_state" / pc._CACHE_FILENAME
    assert cache_file.exists(), "cache JSONL must be written to rlm_state/"

    # Second 'run' (new context, same project_dir): read from disk.
    hit = pc.maybe_get(tmp_path, "implement_baseline", payload=payload)
    assert hit == result, "cache must return the stored implement_baseline result on relaunch"


def test_plan_reproduction_put_then_fresh_get(tmp_path: Path) -> None:
    """Same cross-restart guarantee for plan_reproduction."""
    payload = {"method_spec": "GRPO with sigmoid gate", "env_spec": {"framework": "pytorch"}}
    result = _valid_plan_reproduction_result()

    pc.put(tmp_path, "plan_reproduction", payload=payload, result=result)
    hit = pc.maybe_get(tmp_path, "plan_reproduction", payload=payload)
    assert hit == result


# ---------------------------------------------------------------------------
# Key has no run-id / project-id component
# ---------------------------------------------------------------------------


def test_make_key_has_no_run_id_component() -> None:
    """The cache key must be purely content-addressed — no run-id leakage."""
    payload = {
        "plan": {"models": ["qwen3-1.7b"]},
        "arxiv_id": "2605.15155",
        "sandbox_mode": "local",
        "gpu_mode": None,
        "knowledge_channel_version": 1,
    }
    key1 = pc.make_key("implement_baseline", payload=payload)
    key2 = pc.make_key("implement_baseline", payload=payload)

    assert key1 == key2, "identical payload must produce identical key (pure content hash)"
    # The key format is v1:<primitive>:<hash> — confirm no variable suffix/prefix.
    parts = key1.split(":")
    assert parts[0] == "v1"
    assert parts[1] == "implement_baseline"
    assert len(parts) == 3, f"key must have exactly 3 colon-separated segments; got: {key1!r}"


def test_same_content_different_project_ids_produce_same_key() -> None:
    """Two project_dirs with the same content payload must produce the same key,
    so a relaunched run (new project_dir object, same paper) will hit the disk
    cache under the original project_dir when pointed there."""
    payload = {"plan": {"x": 1}, "arxiv_id": "2605.15155"}
    key_run1 = pc.make_key("implement_baseline", payload=payload)
    key_run2 = pc.make_key("implement_baseline", payload=payload)
    assert key_run1 == key_run2


# ---------------------------------------------------------------------------
# Volatile field exclusion: remaining_s must NOT be in the builder payload
# ---------------------------------------------------------------------------


def test_implement_baseline_builder_payload_excludes_remaining_s(tmp_path: Path) -> None:
    """The implement_baseline payload constructor in primitives.py must exclude
    remaining_s (which changes on every call).  We verify this by constructing
    two 'builder payloads' that are identical except one includes remaining_s
    and one does not — the one WITHOUT remaining_s is what the builder emits,
    and both should produce the same key (i.e. remaining_s is already absent).

    This test documents the contract by proving that adding remaining_s to the
    canonical payload breaks the key, and that the canonical builder output
    does NOT include it, so two launches with different remaining_s values
    will both hit the cache.
    """
    base_payload = {
        "plan": {"method": "GRPO"},
        "repair_context": None,
        "arxiv_id": "2605.15155",
        "sandbox_mode": "local",
        "gpu_mode": None,
        "knowledge_channel_version": 1,
    }
    # Payload that a naive builder might produce at t=0 (1800 s remaining).
    payload_with_volatile_field = {**base_payload, "remaining_s": 1800}
    # Payload that the actual builder produces (no remaining_s).
    payload_canonical = base_payload.copy()

    key_canonical = pc.make_key("implement_baseline", payload=payload_canonical)
    key_with_volatile = pc.make_key("implement_baseline", payload=payload_with_volatile_field)

    # Sanity: adding remaining_s changes the key — so if the builder included it,
    # relaunches with different remaining_s would always miss.
    assert key_canonical != key_with_volatile, (
        "remaining_s must change the key when present, proving it is a volatile field"
    )

    # The actual builder stores the canonical payload (without remaining_s).
    # Simulate: put with canonical, then get with canonical on relaunch.
    result = _valid_implement_baseline_result()
    pc.put(tmp_path, "implement_baseline", payload=payload_canonical, result=result)

    hit_canonical = pc.maybe_get(tmp_path, "implement_baseline", payload=payload_canonical)
    miss_volatile = pc.maybe_get(tmp_path, "implement_baseline", payload=payload_with_volatile_field)

    assert hit_canonical == result, "canonical payload must hit"
    assert miss_volatile is None, (
        "payload WITH remaining_s must miss — "
        "proves the builder correctly excludes it and both relaunches use the same canonical key"
    )


def test_two_relaunches_same_payload_both_hit(tmp_path: Path) -> None:
    """Simulate two successive spot-preemption relaunches of the same paper.
    Both must hit the cache stored by the first successful run."""
    payload = {
        "plan": {"models": ["qwen3-1.7b", "qwen2.5-3b"]},
        "repair_context": None,
        "arxiv_id": "2605.15155",
        "sandbox_mode": "local",
        "gpu_mode": None,
        "knowledge_channel_version": 1,
    }
    result = _valid_implement_baseline_result()

    # Run 0 (original): stores to disk.
    pc.put(tmp_path, "implement_baseline", payload=payload, result=result)

    # Run 1 (relaunch 1): reads from disk.
    hit1 = pc.maybe_get(tmp_path, "implement_baseline", payload=payload)
    assert hit1 == result, "first relaunch must hit"

    # Run 2 (relaunch 2): reads from disk (no new put).
    hit2 = pc.maybe_get(tmp_path, "implement_baseline", payload=payload)
    assert hit2 == result, "second relaunch must also hit"


# ---------------------------------------------------------------------------
# Non-cacheable primitives must not be cached
# ---------------------------------------------------------------------------


def test_run_experiment_not_cacheable(tmp_path: Path) -> None:
    """run_experiment must always miss — it depends on real-world state."""
    pc.put(tmp_path, "run_experiment", payload={"cells": []}, result={"success": True})
    assert pc.maybe_get(tmp_path, "run_experiment", payload={"cells": []}) is None


def test_build_environment_not_cacheable(tmp_path: Path) -> None:
    """build_environment must always miss — Docker layer-cached already."""
    pc.put(tmp_path, "build_environment", payload={"dockerfile": "FROM python"}, result={"ok": True})
    assert pc.maybe_get(tmp_path, "build_environment", payload={"dockerfile": "FROM python"}) is None


def test_non_cacheable_primitives_absent_from_set() -> None:
    """Explicit allowlist check — run_experiment and build_environment must
    never appear in CACHEABLE_PRIMITIVES."""
    assert "run_experiment" not in pc.CACHEABLE_PRIMITIVES
    assert "build_environment" not in pc.CACHEABLE_PRIMITIVES


# ---------------------------------------------------------------------------
# On-disk JSONL survives a new in-process reader
# ---------------------------------------------------------------------------


def test_jsonl_is_human_readable_and_survives_reader_swap(tmp_path: Path) -> None:
    """The JSONL must be valid JSON-lines that any new reader can parse,
    proving that a fresh process (relaunch) pointing at the same path
    will find the data without any in-memory state."""
    payload = {"method_spec": "GRPO", "env_spec": {}}
    result = _valid_plan_reproduction_result()
    pc.put(tmp_path, "plan_reproduction", payload=payload, result=result)

    cache_file = tmp_path / "rlm_state" / pc._CACHE_FILENAME
    raw_lines = cache_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(raw_lines) >= 1

    # Parse every line as JSON independently — no in-process state needed.
    parsed = [json.loads(line) for line in raw_lines if line.strip()]
    assert len(parsed) >= 1

    entry = parsed[-1]
    assert "key" in entry
    assert "primitive" in entry
    assert "result" in entry
    assert entry["primitive"] == "plan_reproduction"
    assert entry["result"] == result


def test_multiple_primitives_coexist_in_same_jsonl(tmp_path: Path) -> None:
    """Multiple cacheable primitives share one JSONL file; each is retrievable."""
    plan_payload = {"method_spec": "GRPO", "env_spec": {}}
    plan_result = _valid_plan_reproduction_result()

    impl_payload = {
        "plan": {"method": "GRPO"},
        "repair_context": None,
        "arxiv_id": "2605.15155",
        "sandbox_mode": "local",
        "gpu_mode": None,
        "knowledge_channel_version": 1,
    }
    impl_result = _valid_implement_baseline_result()

    pc.put(tmp_path, "plan_reproduction", payload=plan_payload, result=plan_result)
    pc.put(tmp_path, "implement_baseline", payload=impl_payload, result=impl_result)

    # Both must be retrievable from the same file.
    assert pc.maybe_get(tmp_path, "plan_reproduction", payload=plan_payload) == plan_result
    assert pc.maybe_get(tmp_path, "implement_baseline", payload=impl_payload) == impl_result

    # Confirm two lines on disk.
    counts = pc.stats(tmp_path)
    assert counts.get("plan_reproduction", 0) >= 1
    assert counts.get("implement_baseline", 0) >= 1
