"""
Tests for _describe_gcp in gpu_capacity.py.

Verifies that the descriptor returns a GpuCapacity with:
- backend_kind == "gcp"
- num_gpus == gcp_max_nodes (always — GKE concurrency cap, not plan gpu_count)
- per_gpu_vram_gb derived from gpu_plan.json when a gcp_ plan is present, else settings
- can_escalate == False
- no error raised
- fail-soft on corrupt/unreadable gpu_plan.json
- non-gcp plan on disk (including azure_ prefix) → settings used (plan ignored)
- returns backend kind "gcp" via describe_capacity dispatch
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.config import Settings
from backend.services.runtime.gpu_capacity import GpuCapacity, _describe_gcp, describe_capacity


def _reset_settings_cache():
    import backend.config as _config
    _config._settings_cache = None


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch):
    _reset_settings_cache()
    yield
    _reset_settings_cache()


def _gcp_ctx(sandbox_mode_value: str = "gcp") -> SimpleNamespace:
    """Minimal duck-typed ctx with sandbox_mode matching gcp."""
    mode = SimpleNamespace(value=sandbox_mode_value)
    return SimpleNamespace(sandbox_mode=mode, gpu_device_ids=None, gpu_plan=None)


def test_describe_gcp_returns_gpu_capacity():
    """_describe_gcp must return a GpuCapacity, not raise."""
    ctx = _gcp_ctx()
    result = _describe_gcp(ctx)
    assert isinstance(result, GpuCapacity), f"Expected GpuCapacity, got {type(result)}"


def test_describe_gcp_backend_kind():
    """backend_kind must be 'gcp'."""
    result = _describe_gcp(_gcp_ctx())
    assert result.backend_kind == "gcp"


def test_describe_gcp_num_gpus_equals_gcp_max_nodes(monkeypatch):
    """num_gpus must equal gcp_max_nodes from Settings."""
    monkeypatch.setenv("OPENRESEARCH_GCP_MAX_NODES", "6")
    _reset_settings_cache()
    result = _describe_gcp(_gcp_ctx())
    assert result.num_gpus == 6


def test_describe_gcp_per_gpu_vram_equals_gcp_per_gpu_vram_gb(monkeypatch):
    """per_gpu_vram_gb must equal gcp_per_gpu_vram_gb from Settings."""
    monkeypatch.setenv("OPENRESEARCH_GCP_PER_GPU_VRAM_GB", "40.0")
    _reset_settings_cache()
    result = _describe_gcp(_gcp_ctx())
    assert result.per_gpu_vram_gb == pytest.approx(40.0)


def test_describe_gcp_can_escalate_false():
    """can_escalate must be False for the gcp backend.

    GKE dispatches Kubernetes Jobs with node-pool SKU selection at dispatch time;
    the RunPod monolithic SKU-ladder escalation loop does not apply to GKE.
    """
    result = _describe_gcp(_gcp_ctx())
    assert result.can_escalate is False


def test_describe_gcp_free_gpu_ids_match_node_count(monkeypatch):
    """free_gpu_ids must be ("0", "1", ...) up to gcp_max_nodes."""
    monkeypatch.setenv("OPENRESEARCH_GCP_MAX_NODES", "3")
    _reset_settings_cache()
    result = _describe_gcp(_gcp_ctx())
    assert result.free_gpu_ids == ("0", "1", "2")


def test_describe_gcp_default_vram_is_80():
    """Default gcp_per_gpu_vram_gb must be 80.0 (A100-80GB spec)."""
    os.environ.pop("OPENRESEARCH_GCP_PER_GPU_VRAM_GB", None)
    _reset_settings_cache()
    s = Settings(_env_file=None)
    assert s.gcp_per_gpu_vram_gb == pytest.approx(80.0)


def test_describe_capacity_routes_gcp_ctx():
    """describe_capacity dispatches to _describe_gcp for sandbox_mode='gcp'."""
    ctx = _gcp_ctx()
    result = describe_capacity(ctx)
    assert result.backend_kind == "gcp"
    assert result.can_escalate is False


# ---------------------------------------------------------------------------
# Plan-aware tests
# ---------------------------------------------------------------------------

def _make_gpu_plan(tmp_path: Path, *, short_name: str, cloud_type: str, vram_gb: int = 24,
                   gpu_count: int = 1) -> Path:
    """Write a minimal gpu_plan.json into tmp_path/rlm_state/ and return the plan path."""
    rlm_state = tmp_path / "rlm_state"
    rlm_state.mkdir(parents=True, exist_ok=True)
    plan = {
        "runpod_id": "gcp-a100-80",
        "short_name": short_name,
        "vram_gb": vram_gb,
        "gpu_count": gpu_count,
        "cloud_type": cloud_type,
        "sku_usd_per_hr": 3.93,
        "total_usd_per_hr": 3.93,
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
        "resolved_at": "2026-06-16T00:00:00Z",
    }
    plan_path = rlm_state / "gpu_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path


def _gcp_ctx_with_dir(project_dir) -> SimpleNamespace:
    """ctx with sandbox_mode=gcp and a project_dir attribute."""
    mode = SimpleNamespace(value="gcp")
    return SimpleNamespace(sandbox_mode=mode, gpu_device_ids=None, gpu_plan=None,
                           project_dir=project_dir)


def test_no_plan_falls_back_to_settings(tmp_path, monkeypatch):
    """No gpu_plan.json on disk → per_gpu_vram_gb comes from Settings."""
    monkeypatch.setenv("OPENRESEARCH_GCP_PER_GPU_VRAM_GB", "80.0")
    _reset_settings_cache()
    ctx = _gcp_ctx_with_dir(tmp_path)
    result = _describe_gcp(ctx)
    assert result.per_gpu_vram_gb == pytest.approx(80.0)
    assert result.backend_kind == "gcp"


def test_gcp_plan_on_disk_overrides_vram(tmp_path, monkeypatch):
    """A gcp gpu_plan.json (short_name starts with 'gcp_') → vram_gb=24 used."""
    monkeypatch.setenv("OPENRESEARCH_GCP_PER_GPU_VRAM_GB", "80.0")
    _reset_settings_cache()
    _make_gpu_plan(tmp_path, short_name="gcp_a100_80", cloud_type="ONDEMAND", vram_gb=24)
    ctx = _gcp_ctx_with_dir(tmp_path)
    result = _describe_gcp(ctx)
    assert result.per_gpu_vram_gb == pytest.approx(24.0)
    assert result.backend_kind == "gcp"
    assert result.can_escalate is False


def test_azure_plan_on_disk_does_not_override_gcp_capacity(tmp_path, monkeypatch):
    """CRITICAL: A plan with short_name starting 'azure_' must NOT override gcp capacity.

    Both azure and GCP plans carry cloud_type='ONDEMAND'.  The GCP descriptor must
    use the 'gcp_' prefix check and reject an azure_ plan — so an azure plan on disk
    never contaminates a GCP sandbox's capacity descriptor.
    """
    monkeypatch.setenv("OPENRESEARCH_GCP_PER_GPU_VRAM_GB", "80.0")
    _reset_settings_cache()
    _make_gpu_plan(tmp_path, short_name="azure_nc24ads_a100", cloud_type="ONDEMAND", vram_gb=24)
    ctx = _gcp_ctx_with_dir(tmp_path)
    result = _describe_gcp(ctx)
    # azure_ plan must be ignored; settings default (80 GB) applies
    assert result.per_gpu_vram_gb == pytest.approx(80.0)
    assert result.backend_kind == "gcp"


def test_runpod_community_plan_does_not_override_gcp_capacity(tmp_path, monkeypatch):
    """A RunPod COMMUNITY plan on disk → not a gcp plan → settings vram used."""
    monkeypatch.setenv("OPENRESEARCH_GCP_PER_GPU_VRAM_GB", "80.0")
    _reset_settings_cache()
    _make_gpu_plan(tmp_path, short_name="rtx4090_24", cloud_type="COMMUNITY", vram_gb=24)
    ctx = _gcp_ctx_with_dir(tmp_path)
    result = _describe_gcp(ctx)
    assert result.per_gpu_vram_gb == pytest.approx(80.0)


def test_corrupt_gpu_plan_falls_back_to_settings(tmp_path, monkeypatch):
    """Corrupt/unreadable gpu_plan.json → fail-soft; settings value used, no raise."""
    monkeypatch.setenv("OPENRESEARCH_GCP_PER_GPU_VRAM_GB", "80.0")
    _reset_settings_cache()
    rlm_state = tmp_path / "rlm_state"
    rlm_state.mkdir(parents=True, exist_ok=True)
    (rlm_state / "gpu_plan.json").write_text("{not valid json", encoding="utf-8")
    ctx = _gcp_ctx_with_dir(tmp_path)
    # Must not raise
    result = _describe_gcp(ctx)
    assert result.per_gpu_vram_gb == pytest.approx(80.0)
    assert result.backend_kind == "gcp"


def test_num_gpus_always_comes_from_gcp_max_nodes(tmp_path, monkeypatch):
    """Plan's gpu_count is irrelevant; num_gpus is always gcp_max_nodes."""
    monkeypatch.setenv("OPENRESEARCH_GCP_MAX_NODES", "4")
    _reset_settings_cache()
    # GCP plan with gpu_count=8 — should NOT affect num_gpus
    _make_gpu_plan(tmp_path, short_name="gcp_a100_80", cloud_type="ONDEMAND",
                   vram_gb=80, gpu_count=8)
    ctx = _gcp_ctx_with_dir(tmp_path)
    result = _describe_gcp(ctx)
    assert result.num_gpus == 4
    assert result.free_gpu_ids == ("0", "1", "2", "3")


def test_no_project_dir_on_ctx_falls_back_to_settings(monkeypatch):
    """ctx without project_dir attr → settings path taken (no AttributeError)."""
    monkeypatch.setenv("OPENRESEARCH_GCP_PER_GPU_VRAM_GB", "80.0")
    _reset_settings_cache()
    ctx = _gcp_ctx()  # no project_dir attribute
    result = _describe_gcp(ctx)
    assert result.per_gpu_vram_gb == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# gpus_per_node — total schedulable GPUs = max_nodes × gpus_per_node
# ---------------------------------------------------------------------------

def test_default_gpus_per_node_is_byte_identical_to_node_count(monkeypatch):
    """Default gcp_gpus_per_node=1 ⇒ num_gpus == gcp_max_nodes (unchanged)."""
    monkeypatch.setenv("OPENRESEARCH_GCP_MAX_NODES", "4")
    monkeypatch.delenv("OPENRESEARCH_GCP_GPUS_PER_NODE", raising=False)
    _reset_settings_cache()
    result = _describe_gcp(_gcp_ctx())
    assert result.num_gpus == 4  # == gcp_max_nodes


def test_gpus_per_node_multiplies_schedulable_gpus(monkeypatch):
    """gcp_gpus_per_node=8 with gcp_max_nodes=1 ⇒ num_gpus == 8."""
    monkeypatch.setenv("OPENRESEARCH_GCP_MAX_NODES", "1")
    monkeypatch.setenv("OPENRESEARCH_GCP_GPUS_PER_NODE", "8")
    _reset_settings_cache()
    result = _describe_gcp(_gcp_ctx())
    assert result.num_gpus == 8
    assert result.free_gpu_ids == tuple(str(i) for i in range(8))


def test_gpus_per_node_default_is_one(monkeypatch):
    """Settings default for gcp_gpus_per_node must be 1."""
    monkeypatch.delenv("OPENRESEARCH_GCP_GPUS_PER_NODE", raising=False)
    _reset_settings_cache()
    s = Settings(_env_file=None)
    assert s.gcp_gpus_per_node == 1
