"""
Tests for _describe_azure in gpu_capacity.py.

Verifies that the descriptor returns a GpuCapacity with:
- backend_kind == "azure"
- num_gpus == azure_max_nodes (always — AKS concurrency cap, not plan gpu_count)
- per_gpu_vram_gb derived from gpu_plan.json when azure plan present, else settings
- can_escalate == False
- no NotImplementedError raised
- fail-soft on corrupt/unreadable gpu_plan.json
- non-azure plan on disk → settings used (plan ignored)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.config import Settings
from backend.services.runtime.gpu_capacity import GpuCapacity, _describe_azure, describe_capacity


def _reset_settings_cache():
    import backend.config as _config
    _config._settings_cache = None


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch):
    _reset_settings_cache()
    yield
    _reset_settings_cache()


def _azure_ctx(sandbox_mode_value: str = "azure") -> SimpleNamespace:
    """Minimal duck-typed ctx with sandbox_mode matching azure."""
    mode = SimpleNamespace(value=sandbox_mode_value)
    return SimpleNamespace(sandbox_mode=mode, gpu_device_ids=None, gpu_plan=None)


def test_describe_azure_returns_gpu_capacity():
    """_describe_azure must return a GpuCapacity, not raise NotImplementedError."""
    ctx = _azure_ctx()
    result = _describe_azure(ctx)
    assert isinstance(result, GpuCapacity), f"Expected GpuCapacity, got {type(result)}"


def test_describe_azure_backend_kind():
    """backend_kind must be 'azure'."""
    result = _describe_azure(_azure_ctx())
    assert result.backend_kind == "azure"


def test_describe_azure_num_gpus_equals_azure_max_nodes(monkeypatch):
    """num_gpus must equal azure_max_nodes from Settings."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_MAX_NODES", "6")
    _reset_settings_cache()
    result = _describe_azure(_azure_ctx())
    assert result.num_gpus == 6


def test_describe_azure_per_gpu_vram_equals_azure_per_gpu_vram_gb(monkeypatch):
    """per_gpu_vram_gb must equal azure_per_gpu_vram_gb from Settings."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_PER_GPU_VRAM_GB", "40.0")
    _reset_settings_cache()
    result = _describe_azure(_azure_ctx())
    assert result.per_gpu_vram_gb == pytest.approx(40.0)


def test_describe_azure_can_escalate_false():
    """can_escalate must be False for the azure backend.

    Rationale: can_escalate guards the *run_experiment monolithic SKU-ladder
    escalation loop* used by the RunPod backend to advance through
    gpu_plan.ladder_remaining on OOM.  That loop does NOT apply to AKS — the
    azure K8s runner dispatches Jobs with a node-pool SKU selected at dispatch
    time and escalates through a separate mechanism.  Setting True would cause
    the RunPod escalation loop to fire incorrectly on azure pods.
    """
    result = _describe_azure(_azure_ctx())
    assert result.can_escalate is False


def test_describe_azure_free_gpu_ids_match_node_count(monkeypatch):
    """free_gpu_ids must be ("0", "1", ...) up to azure_max_nodes."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_MAX_NODES", "3")
    _reset_settings_cache()
    result = _describe_azure(_azure_ctx())
    assert result.free_gpu_ids == ("0", "1", "2")


def test_describe_azure_default_vram_is_80():
    """Default azure_per_gpu_vram_gb must be 80.0 (A100-80GB spec)."""
    for k in ("OPENRESEARCH_AZURE_PER_GPU_VRAM_GB",):
        import os
        os.environ.pop(k, None)
    _reset_settings_cache()
    s = Settings(_env_file=None)
    assert s.azure_per_gpu_vram_gb == pytest.approx(80.0)


def test_describe_capacity_routes_azure_ctx():
    """describe_capacity dispatches to _describe_azure for sandbox_mode='azure'."""
    ctx = _azure_ctx()
    result = describe_capacity(ctx)
    assert result.backend_kind == "azure"
    assert result.can_escalate is False


# ---------------------------------------------------------------------------
# Plan-aware tests (new — plan loading + fall-through behaviour)
# ---------------------------------------------------------------------------

def _make_gpu_plan(tmp_path: Path, *, short_name: str, cloud_type: str, vram_gb: int = 24,
                   gpu_count: int = 1) -> Path:
    """Write a minimal gpu_plan.json into tmp_path/rlm_state/ and return the plan path."""
    rlm_state = tmp_path / "rlm_state"
    rlm_state.mkdir(parents=True, exist_ok=True)
    plan = {
        "runpod_id": "azure-a10g-24",
        "short_name": short_name,
        "vram_gb": vram_gb,
        "gpu_count": gpu_count,
        "cloud_type": cloud_type,
        "sku_usd_per_hr": 2.0,
        "total_usd_per_hr": 2.0,
        "container_disk_gb": 50,
        "volume_gb": 100,
        "source": "paper",
        "requirements": {
            "vram_gb": vram_gb,
            "num_gpus": gpu_count,
            "paper_gpu_string": None,
            "paper_gpu_count": None,
            "reasoning": "test",
            "confidence": 0.9,
        },
        "ladder_remaining": [],
        "resolved_at": "2026-06-07T00:00:00Z",
    }
    plan_path = rlm_state / "gpu_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path


def _azure_ctx_with_dir(project_dir) -> SimpleNamespace:
    """ctx with sandbox_mode=azure and a project_dir attribute."""
    mode = SimpleNamespace(value="azure")
    return SimpleNamespace(sandbox_mode=mode, gpu_device_ids=None, gpu_plan=None,
                           project_dir=project_dir)


def test_no_plan_falls_back_to_settings(tmp_path, monkeypatch):
    """No gpu_plan.json on disk → per_gpu_vram_gb comes from Settings."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_PER_GPU_VRAM_GB", "80.0")
    _reset_settings_cache()
    ctx = _azure_ctx_with_dir(tmp_path)
    result = _describe_azure(ctx)
    assert result.per_gpu_vram_gb == pytest.approx(80.0)
    assert result.backend_kind == "azure"


def test_azure_plan_on_disk_overrides_vram(tmp_path, monkeypatch):
    """An azure gpu_plan.json (short_name starts with 'azure_') → vram_gb=24 used."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_PER_GPU_VRAM_GB", "80.0")
    _reset_settings_cache()
    _make_gpu_plan(tmp_path, short_name="azure_a10_24", cloud_type="ONDEMAND", vram_gb=24)
    ctx = _azure_ctx_with_dir(tmp_path)
    result = _describe_azure(ctx)
    assert result.per_gpu_vram_gb == pytest.approx(24.0)
    assert result.backend_kind == "azure"
    assert result.can_escalate is False


def test_azure_plan_ondemand_cloud_type_overrides_vram(tmp_path, monkeypatch):
    """cloud_type='ONDEMAND' (even without 'azure_' prefix) is treated as azure plan."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_PER_GPU_VRAM_GB", "80.0")
    _reset_settings_cache()
    _make_gpu_plan(tmp_path, short_name="h100_80", cloud_type="ONDEMAND", vram_gb=48)
    ctx = _azure_ctx_with_dir(tmp_path)
    result = _describe_azure(ctx)
    assert result.per_gpu_vram_gb == pytest.approx(48.0)


def test_corrupt_gpu_plan_falls_back_to_settings(tmp_path, monkeypatch):
    """Corrupt/unreadable gpu_plan.json → fail-soft; settings value used, no raise."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_PER_GPU_VRAM_GB", "80.0")
    _reset_settings_cache()
    rlm_state = tmp_path / "rlm_state"
    rlm_state.mkdir(parents=True, exist_ok=True)
    (rlm_state / "gpu_plan.json").write_text("{not valid json", encoding="utf-8")
    ctx = _azure_ctx_with_dir(tmp_path)
    # Must not raise
    result = _describe_azure(ctx)
    assert result.per_gpu_vram_gb == pytest.approx(80.0)
    assert result.backend_kind == "azure"


def test_non_azure_plan_uses_settings(tmp_path, monkeypatch):
    """A RunPod COMMUNITY plan on disk → not an azure plan → settings vram used."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_PER_GPU_VRAM_GB", "80.0")
    _reset_settings_cache()
    _make_gpu_plan(tmp_path, short_name="rtx4090_24", cloud_type="COMMUNITY", vram_gb=24)
    ctx = _azure_ctx_with_dir(tmp_path)
    result = _describe_azure(ctx)
    # Non-azure plan must be ignored; settings default (80 GB) applies
    assert result.per_gpu_vram_gb == pytest.approx(80.0)


def test_num_gpus_always_comes_from_azure_max_nodes(tmp_path, monkeypatch):
    """Plan's gpu_count is irrelevant; num_gpus is always azure_max_nodes."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_MAX_NODES", "4")
    _reset_settings_cache()
    # Azure plan with gpu_count=8 — should NOT affect num_gpus
    _make_gpu_plan(tmp_path, short_name="azure_a100_80", cloud_type="ONDEMAND",
                   vram_gb=80, gpu_count=8)
    ctx = _azure_ctx_with_dir(tmp_path)
    result = _describe_azure(ctx)
    assert result.num_gpus == 4
    assert result.free_gpu_ids == ("0", "1", "2", "3")


def test_no_project_dir_on_ctx_falls_back_to_settings(monkeypatch):
    """ctx without project_dir attr → settings path taken (no AttributeError)."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_PER_GPU_VRAM_GB", "80.0")
    _reset_settings_cache()
    ctx = _azure_ctx()  # no project_dir attribute
    result = _describe_azure(ctx)
    assert result.per_gpu_vram_gb == pytest.approx(80.0)
