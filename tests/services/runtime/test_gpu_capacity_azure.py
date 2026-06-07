"""
Tests for _describe_azure in gpu_capacity.py.

Verifies that the settings-driven descriptor returns a GpuCapacity with:
- backend_kind == "azure"
- num_gpus == azure_max_nodes
- per_gpu_vram_gb == azure_per_gpu_vram_gb
- can_escalate == False
- no NotImplementedError raised
"""

from __future__ import annotations

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
    monkeypatch.setenv("REPROLAB_AZURE_MAX_NODES", "6")
    _reset_settings_cache()
    result = _describe_azure(_azure_ctx())
    assert result.num_gpus == 6


def test_describe_azure_per_gpu_vram_equals_azure_per_gpu_vram_gb(monkeypatch):
    """per_gpu_vram_gb must equal azure_per_gpu_vram_gb from Settings."""
    monkeypatch.setenv("REPROLAB_AZURE_PER_GPU_VRAM_GB", "40.0")
    _reset_settings_cache()
    result = _describe_azure(_azure_ctx())
    assert result.per_gpu_vram_gb == pytest.approx(40.0)


def test_describe_azure_can_escalate_false():
    """can_escalate must be False (single node-pool, no catalog ladder)."""
    result = _describe_azure(_azure_ctx())
    assert result.can_escalate is False


def test_describe_azure_free_gpu_ids_match_node_count(monkeypatch):
    """free_gpu_ids must be ("0", "1", ...) up to azure_max_nodes."""
    monkeypatch.setenv("REPROLAB_AZURE_MAX_NODES", "3")
    _reset_settings_cache()
    result = _describe_azure(_azure_ctx())
    assert result.free_gpu_ids == ("0", "1", "2")


def test_describe_azure_default_vram_is_80():
    """Default azure_per_gpu_vram_gb must be 80.0 (A100-80GB spec)."""
    for k in ("REPROLAB_AZURE_PER_GPU_VRAM_GB",):
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
