"""Fail-closed, provider-only EKS GPU metadata tests (socket-hermetic)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.agents.schemas import GpuRequirements
from backend.services.runtime import gpu_capacity
from backend.services.runtime.gpu_resolver import GpuResolutionError, resolve_configured_aws


def _settings(**overrides):
    values = {
        "aws_gpu_skus": ["eks-a100-80"],
        "aws_max_nodes": 2,
        "aws_gpus_per_node": 1,
        "aws_per_gpu_vram_gb": 80.0,
        "aws_gpu_usd_per_hour": 3.25,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_aws_capacity_is_empty_when_metering_metadata_is_missing(monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(aws_gpu_usd_per_hour=0.0))
    cap = gpu_capacity.describe_capacity(SimpleNamespace(sandbox_mode="aws", gpu_plan=None))

    assert cap.is_empty
    assert "aws_gpu_usd_per_hour" in cap.detail["configuration_error"]


def test_aws_capacity_rejects_foreign_gpu_plan_label(monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings())
    cap = gpu_capacity.describe_capacity(SimpleNamespace(
        sandbox_mode="aws", gpu_plan={"short_name": "rtx4090", "gpu_count": 1},
    ))

    assert cap.is_empty
    assert "not an aws_gpu_skus label" in cap.detail["configuration_error"]


def test_aws_capacity_rejects_multi_gpu_nodes_and_plans(monkeypatch):
    monkeypatch.setattr("backend.config.get_settings", lambda: _settings(aws_gpus_per_node=2))
    cap = gpu_capacity.describe_capacity(SimpleNamespace(
        sandbox_mode="aws", gpu_plan={"short_name": "eks-a100-80", "gpu_count": 1},
    ))

    assert cap.is_empty
    assert "must equal 1" in cap.detail["configuration_error"]


def test_configured_aws_plan_obeys_declared_rate_and_vram_not_public_catalog():
    plan = resolve_configured_aws(
        GpuRequirements(estimated_vram_gb=60, confidence=0.9),
        gpu_skus=("private-eks-a100",), per_gpu_vram_gb=80, per_gpu_usd_per_hour=3.25,
        gpus_per_node=1, dynamic_gpu_enabled=True, force_single_gpu=True,
        max_gpu_usd_per_hour=4.0, headroom_multiplier=1.0,
    )

    assert plan.short_name == "private-eks-a100"
    assert plan.sku_usd_per_hr == pytest.approx(3.25)
    assert plan.ladder_remaining == ()


def test_primitive_resolves_aws_only_from_configured_pool_metadata(monkeypatch, tmp_path):
    from backend.agents.execution import SandboxMode
    from backend.agents.rlm.primitives import resolve_gpu_requirements

    settings = _settings(
        dynamic_gpu_enabled=True, force_single_gpu=True, max_gpu_usd_per_hour=4.0,
        dynamic_gpu_headroom=1.0, gpu_count=None, dynamic_gpu_fallback_vram_gb=24,
    )
    monkeypatch.setattr("backend.config.get_settings", lambda: settings)
    monkeypatch.setattr("backend.agents.rlm.primitives._emit_dashboard_event", lambda *a, **k: None)
    payload = resolve_gpu_requirements(
        {"estimated_vram_gb": 60, "confidence": 0.9},
        ctx=SimpleNamespace(project_dir=tmp_path, sandbox_mode=SandboxMode.aws),
    )

    assert payload["short_name"] == "eks-a100-80"
    assert payload["sku_usd_per_hr"] == pytest.approx(3.25)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"per_gpu_usd_per_hour": 0.0}, "requires positive"),
        ({"per_gpu_vram_gb": 40.0}, "requires >= 60 GB"),
        ({"max_gpu_usd_per_hour": 3.0}, "exceeds"),
        ({"gpus_per_node": 2}, "gpus_per_node=1"),
    ],
)
def test_configured_aws_plan_fails_closed_on_bad_metadata_or_cost(kwargs, match):
    values = dict(
        gpu_skus=("eks-a100",), per_gpu_vram_gb=80.0, per_gpu_usd_per_hour=3.25,
        gpus_per_node=1, dynamic_gpu_enabled=True, force_single_gpu=True,
        max_gpu_usd_per_hour=4.0, headroom_multiplier=1.0,
    )
    values.update(kwargs)
    with pytest.raises(GpuResolutionError, match=match):
        resolve_configured_aws(GpuRequirements(estimated_vram_gb=60, confidence=0.9), **values)
