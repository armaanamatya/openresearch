from __future__ import annotations

from unittest.mock import patch


from backend.agents.execution import SandboxMode
from backend.agents.rlm.primitives import _backend_for_sandbox_mode
from backend.agents.schemas import GpuPlan, GpuRequirements


def _plan() -> GpuPlan:
    return GpuPlan(
        runpod_id="NVIDIA A100 40GB PCIe", short_name="a100_40", vram_gb=40, gpu_count=1,
        cloud_type="COMMUNITY", sku_usd_per_hr=1.19, total_usd_per_hr=1.19,
        container_disk_gb=50, volume_gb=20, source="paper",
        requirements=GpuRequirements(
            estimated_vram_gb=32, paper_gpu_string="A100",
            paper_gpu_count=1, reasoning="", confidence=0.9,
        ),
        ladder_remaining=("a100_80",), resolved_at="2026-05-23T00:00:00+00:00",
    )


def test_backend_for_sandbox_mode_passes_gpu_plan_to_runpod_backend():
    plan = _plan()
    with patch("backend.services.runtime.ensure_runpod_available"):
        backend = _backend_for_sandbox_mode(SandboxMode.runpod, gpu_plan=plan)
    assert backend.__class__.__name__ == "RunpodBackend"
    assert backend.gpu_type == "NVIDIA A100 40GB PCIe"


def test_backend_for_sandbox_mode_local_docker_ignores_gpu_plan():
    plan = _plan()
    backend = _backend_for_sandbox_mode(SandboxMode.docker, gpu_plan=plan)
    assert backend.__class__.__name__ == "LocalDockerBackend"
