"""Round-trip + version-invalidation tests for the estimate cache.

Spec invariant 4: CATALOG_SCHEMA_VERSION bump invalidates cached estimates.
Spec invariant 8: cache key includes both schema versions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import backend.services.pricing.cache as cache_module
from backend.services.pricing.cache import get_cached, set_cached


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    r = tmp_path / "runs"
    r.mkdir()
    return r


def test_round_trip(runs_root: Path):
    sha = "abcdef1234567890"
    payload = {"estimate_id": "abcdef12_strict_1_1", "gpu": {"sku_id": "rtx4090"}}
    set_cached(runs_root, sha, "strict", payload)
    result = get_cached(runs_root, sha, "strict")
    assert result is not None
    assert result["gpu"]["sku_id"] == "rtx4090"


def test_different_recipe_modes_are_independent(runs_root: Path):
    sha = "abcdef1234567890"
    set_cached(runs_root, sha, "strict", {"recipe": "strict"})
    set_cached(runs_root, sha, "compressed", {"recipe": "compressed"})

    strict_result = get_cached(runs_root, sha, "strict")
    compressed_result = get_cached(runs_root, sha, "compressed")
    assert strict_result["recipe"] == "strict"
    assert compressed_result["recipe"] == "compressed"


def test_miss_when_no_file(runs_root: Path):
    result = get_cached(runs_root, "nonexistent_sha", "strict")
    assert result is None


def test_catalog_version_mismatch_is_cache_miss(runs_root: Path, monkeypatch):
    sha = "abcdef1234567890"
    set_cached(runs_root, sha, "strict", {"data": "v1"})
    result = get_cached(runs_root, sha, "strict")
    assert result is not None, "should hit before version bump"

    # Simulate a catalog version bump.
    monkeypatch.setattr(cache_module, "CATALOG_SCHEMA_VERSION", 99)
    result_after = get_cached(runs_root, sha, "strict")
    assert result_after is None, "version bump must invalidate cache"


def test_calibration_version_mismatch_is_cache_miss(runs_root: Path, monkeypatch):
    sha = "abcdef1234567890"
    set_cached(runs_root, sha, "strict", {"data": "cal_v1"})
    result = get_cached(runs_root, sha, "strict")
    assert result is not None

    monkeypatch.setattr(cache_module, "CALIBRATION_SCHEMA_VERSION", 99)
    result_after = get_cached(runs_root, sha, "strict")
    assert result_after is None, "calibration version bump must invalidate cache"


def test_corrupt_cache_file_is_cache_miss(runs_root: Path):
    sha = "badf00dbadf00dba"
    cache_dir = runs_root / "_estimates"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_module._cache_key(sha, "strict")
    (cache_dir / f"{key}.json").write_text("{{not valid json", encoding="utf-8")
    result = get_cached(runs_root, sha, "strict")
    assert result is None


def test_atomic_write_uses_tmp(runs_root: Path, monkeypatch):
    """set_cached must write .tmp then os.replace — never partially-written files."""
    import os
    writes: list[str] = []
    real_replace = os.replace

    def _track_replace(src, dst):
        writes.append(str(src))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _track_replace)
    set_cached(runs_root, "aabbccdd11223344", "strict", {"x": 1})
    assert any(".tmp" in w for w in writes), "expected .tmp file in os.replace calls"
