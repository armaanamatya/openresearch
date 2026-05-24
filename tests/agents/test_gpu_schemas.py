from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.agents.schemas import GpuPlan, GpuRequirements


def test_gpu_requirements_accepts_complete_payload():
    req = GpuRequirements(
        estimated_vram_gb=48,
        paper_gpu_string="A100 80GB",
        paper_gpu_count=8,
        reasoning="paper states 80GB; eval harness adds ~10GB",
        confidence=0.85,
    )
    assert req.estimated_vram_gb == 48
    assert req.confidence == pytest.approx(0.85)


def test_gpu_requirements_allows_none_estimate():
    req = GpuRequirements(
        estimated_vram_gb=None,
        paper_gpu_string=None,
        paper_gpu_count=None,
        reasoning="no hardware clues found in paper",
        confidence=0.1,
    )
    assert req.estimated_vram_gb is None


def test_gpu_requirements_rejects_negative_vram():
    with pytest.raises(ValidationError):
        GpuRequirements(
            estimated_vram_gb=-5,
            paper_gpu_string=None,
            paper_gpu_count=None,
            reasoning="",
            confidence=0.5,
        )


def test_gpu_requirements_clamps_confidence_range():
    with pytest.raises(ValidationError):
        GpuRequirements(
            estimated_vram_gb=24,
            paper_gpu_string=None,
            paper_gpu_count=None,
            reasoning="",
            confidence=1.5,
        )


def test_gpu_plan_complete_payload():
    plan = GpuPlan(
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
            estimated_vram_gb=64,
            paper_gpu_string="A100 80GB",
            paper_gpu_count=8,
            reasoning="test",
            confidence=0.9,
        ),
        ladder_remaining=("h100_80",),
        resolved_at="2026-05-23T00:00:00+00:00",
    )
    assert plan.gpu_count == 1
    assert plan.ladder_remaining == ("h100_80",)


def test_gpu_plan_source_accepts_only_known_values():
    with pytest.raises(ValidationError):
        GpuPlan(
            runpod_id="x", short_name="x", vram_gb=24, gpu_count=1,
            cloud_type="COMMUNITY", sku_usd_per_hr=0.34, total_usd_per_hr=0.34,
            container_disk_gb=50, volume_gb=20,
            source="bogus_source",  # not in {paper, fallback, manual, informational}
            requirements=GpuRequirements(
                estimated_vram_gb=24, paper_gpu_string=None, paper_gpu_count=None,
                reasoning="", confidence=0.5,
            ),
            ladder_remaining=(),
            resolved_at="2026-05-23T00:00:00+00:00",
        )
