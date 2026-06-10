from __future__ import annotations


from backend.agents.schemas import GpuPlan, GpuRequirements
from backend.services.runtime.runpod_backend import RunpodBackend


def _plan(**overrides) -> GpuPlan:
    base = dict(
        runpod_id="NVIDIA A100 80GB PCIe",
        short_name="a100_80",
        vram_gb=80,
        gpu_count=1,
        cloud_type="COMMUNITY",
        sku_usd_per_hr=1.89,
        total_usd_per_hr=1.89,
        container_disk_gb=80,
        volume_gb=20,
        source="paper",
        requirements=GpuRequirements(
            estimated_vram_gb=64, paper_gpu_string="A100",
            paper_gpu_count=1, reasoning="", confidence=0.9,
        ),
        ladder_remaining=("h100_80",),
        resolved_at="2026-05-23T00:00:00+00:00",
    )
    base.update(overrides)
    return GpuPlan(**base)


def test_backend_uses_gpu_plan_when_provided():
    plan = _plan()
    backend = RunpodBackend(api_key="dummy", gpu_plan=plan)
    assert backend.gpu_type == "NVIDIA A100 80GB PCIe"
    assert backend.gpu_count == 1
    assert backend.cloud_type == "COMMUNITY"
    assert backend.container_disk_gb >= 80
    assert backend.volume_gb >= 20


def test_backend_back_compat_no_plan_uses_settings():
    """When gpu_plan is None, backend falls back to legacy Settings defaults."""
    backend = RunpodBackend(api_key="dummy", gpu_plan=None)
    # Default per repo: OPENRESEARCH_RUNPOD_GPU_TYPE="NVIDIA GeForce RTX 4090"
    assert "RTX 4090" in backend.gpu_type or "4090" in backend.gpu_type
    assert backend.gpu_count == 1


def test_backend_plan_overrides_explicit_init_args():
    """If both gpu_plan and gpu_type=... are passed, gpu_plan wins for type/count."""
    plan = _plan(short_name="rtx4090", runpod_id="NVIDIA GeForce RTX 4090", vram_gb=24)
    backend = RunpodBackend(api_key="dummy", gpu_type="OTHER_GPU", gpu_count=4, gpu_plan=plan)
    assert backend.gpu_type == "NVIDIA GeForce RTX 4090"
    assert backend.gpu_count == 1


def test_backend_informational_plan_is_ignored():
    """source='informational' means dynamic_gpu_enabled=off; legacy path."""
    plan = _plan(source="informational")
    backend = RunpodBackend(api_key="dummy", gpu_plan=plan, gpu_type="NVIDIA GeForce RTX 4090")
    # Backend ignores informational plans and uses the explicit gpu_type arg / settings default.
    assert backend.gpu_type == "NVIDIA GeForce RTX 4090"
