"""P2 / provenance manifest — Unit A: metric↔artifact binding (invariant 2).

_manifest_enrichment enriches a run_experiment result IN PLACE with sandbox_backend
+ metrics_sha256 (the sha256 of the canonical metrics.json that a final-report
metric is later tied to). Best-effort + fail-soft.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from backend.agents.rlm.primitives import _manifest_enrichment, _stamp_manifest_ids


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


# --- Unit B: _stamp_manifest_ids (experiment_run_id + env_id + commands) ---

def test_stamp_records_run_id_env_id_commands():
    result = {"success": True, "metrics": {"acc": 0.5}}
    _stamp_manifest_ids(result, run_id="prj_x-ab12cd34", env_id="img:tag", commands=["python train.py"])
    assert result["experiment_run_id"] == "prj_x-ab12cd34"
    assert result["env_id"] == "img:tag"
    assert result["commands"] == ["python train.py"]


def test_stamp_does_not_clobber_existing():
    result = {"experiment_run_id": "already", "env_id": "keep", "commands": ["orig"]}
    _stamp_manifest_ids(result, run_id="new", env_id="new", commands=["new"])
    assert result["experiment_run_id"] == "already"
    assert result["env_id"] == "keep"
    assert result["commands"] == ["orig"]


def test_stamp_empty_commands_is_list():
    result = {}
    _stamp_manifest_ids(result, run_id="r", env_id="e", commands=[])
    assert result["commands"] == []


def test_stamp_non_dict_is_noop():
    _stamp_manifest_ids(None, run_id="r", env_id="e", commands=[])  # must not raise


def test_full_manifest_round_trip(tmp_path: Path):
    """A finalized success result carries all five manifest fields (A + B)."""
    (tmp_path / "metrics.json").write_text('{"f1": 0.7}', encoding="utf-8")
    result = {
        "success": True,
        "artifact_dir": str(tmp_path),
        "resource_limits": {"sandbox_mode": "local"},
    }
    _stamp_manifest_ids(result, run_id="prj_x-ff00", env_id="__local__", commands=["python train.py"])
    _manifest_enrichment(result)
    for key in ("experiment_run_id", "env_id", "commands", "sandbox_backend", "metrics_sha256"):
        assert key in result, key
