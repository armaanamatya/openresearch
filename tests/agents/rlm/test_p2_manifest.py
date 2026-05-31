"""P2 / provenance manifest — Unit A: metric↔artifact binding (invariant 2).

_manifest_enrichment enriches a run_experiment result IN PLACE with sandbox_backend
+ metrics_sha256 (the sha256 of the canonical metrics.json that a final-report
metric is later tied to). Best-effort + fail-soft.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from backend.agents.rlm.primitives import _manifest_enrichment


def test_enriches_sandbox_backend_and_metrics_hash(tmp_path: Path):
    (tmp_path / "metrics.json").write_text('{"acc": 0.5}', encoding="utf-8")
    result = {
        "success": True,
        "artifact_dir": str(tmp_path),
        "resource_limits": {"sandbox_mode": "local"},
    }
    _manifest_enrichment(result)
    assert result["sandbox_backend"] == "local"
    expected = hashlib.sha256((tmp_path / "metrics.json").read_bytes()).hexdigest()
    assert result["metrics_sha256"] == expected


def test_graceful_without_artifact_dir():
    """A pre-flight / early-fail result (no artifact) must not raise or invent fields."""
    result = {"success": False, "error": "disk floor"}
    _manifest_enrichment(result)
    assert "metrics_sha256" not in result
    assert "sandbox_backend" not in result


def test_backend_without_metrics_file(tmp_path: Path):
    """artifact_dir present but no metrics.json (failed run) → backend recorded, no hash."""
    result = {"artifact_dir": str(tmp_path), "resource_limits": {"sandbox_mode": "docker"}}
    _manifest_enrichment(result)
    assert result["sandbox_backend"] == "docker"
    assert "metrics_sha256" not in result


def test_does_not_overwrite_existing_fields(tmp_path: Path):
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    result = {
        "artifact_dir": str(tmp_path),
        "resource_limits": {"sandbox_mode": "local"},
        "sandbox_backend": "runpod",  # already set (e.g. by a richer caller)
        "metrics_sha256": "preexisting",
    }
    _manifest_enrichment(result)
    assert result["sandbox_backend"] == "runpod"
    assert result["metrics_sha256"] == "preexisting"


def test_hash_is_byte_exact(tmp_path: Path):
    """Two different metrics.json bytes ⇒ different hashes (real binding, not a stub)."""
    (tmp_path / "metrics.json").write_text('{"a": 1}', encoding="utf-8")
    r1 = {"artifact_dir": str(tmp_path), "resource_limits": {}}
    _manifest_enrichment(r1)
    (tmp_path / "metrics.json").write_text('{"a": 2}', encoding="utf-8")
    r2 = {"artifact_dir": str(tmp_path), "resource_limits": {}}
    _manifest_enrichment(r2)
    assert r1["metrics_sha256"] != r2["metrics_sha256"]
