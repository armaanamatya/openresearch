"""Operator GPU-count override (OPENRESEARCH_GPU_COUNT) — resolver behaviour.

A user-selected GPU count pins ``GpuPlan.gpu_count`` (and thus the K8s
``nvidia.com/gpu`` request), overriding the ``force_single_gpu`` default. It must
relax ``force_single_gpu`` for SKU SELECTION so an N-GPU-only provisioned pool
(e.g. GCP's default ``gcp_a100_80x8``) is not filtered out, then cap the final
count to the chosen SKU's physical ``gpu_count``. Pure tests — no cloud/network.
"""
from __future__ import annotations

from backend.agents.schemas import GpuRequirements
from backend.services.runtime import gpu_resolver as r


def _req(vram: int, conf: float = 0.9) -> GpuRequirements:
    return GpuRequirements(estimated_vram_gb=vram, confidence=conf, paper_gpu_count=1)


def _resolve(req: GpuRequirements, **kw):
    base = dict(
        dynamic_gpu_enabled=True,
        force_single_gpu=False,
        max_gpu_usd_per_hour=None,
        headroom_multiplier=1.0,
        fallback_vram_gb=24,
        provider="gcp",
    )
    base.update(kw)
    return r.resolve(req, **base)


class TestGpuCountOverride:
    def test_override_pins_gpu_count_and_stamps_manual(self):
        plan = _resolve(_req(80), provisioned_skus=("gcp_a100_80x8",), gpu_count_override=4)
        assert plan.gpu_count == 4
        assert plan.source == "manual"
        assert plan.short_name == "gcp_a100_80x8"

    def test_override_recomputes_total_rate(self):
        plan = _resolve(_req(80), provisioned_skus=("gcp_a100_80x8",), gpu_count_override=4)
        # total = per-GPU rate * requested count (not the SKU's physical 8)
        assert plan.total_usd_per_hr == round(plan.sku_usd_per_hr * 4, 4)

    def test_override_capped_to_sku_physical_count(self):
        # Ask for more GPUs than the machine physically has → capped to the SKU max.
        plan = _resolve(_req(80), provisioned_skus=("gcp_a100_80x8",), gpu_count_override=16)
        assert plan.gpu_count == 8

    def test_override_floors_at_one(self):
        plan = _resolve(_req(80), provisioned_skus=("gcp_a100_80x8",), gpu_count_override=0)
        assert plan.gpu_count == 1

    def test_override_relaxes_force_single_on_multi_gpu_only_pool(self):
        # The GCP default provisions ONLY the 8-GPU pool. With force_single_gpu=True the
        # ladder filter (gpu_count==1) would EXCLUDE it and raise. The override must relax
        # force_single_gpu for selection so the request succeeds with the pinned count.
        plan = _resolve(
            _req(80),
            force_single_gpu=True,
            provisioned_skus=("gcp_a100_80x8",),
            gpu_count_override=2,
        )
        assert plan.gpu_count == 2
        assert plan.source == "manual"

    def test_override_none_is_byte_identical(self):
        a = _resolve(_req(80), provisioned_skus=("gcp_a100_80x8",)).model_dump()
        b = _resolve(_req(80), provisioned_skus=("gcp_a100_80x8",), gpu_count_override=None).model_dump()
        # resolved_at is a wall-clock stamp — exclude it from the equality check.
        a.pop("resolved_at", None)
        b.pop("resolved_at", None)
        assert a == b
        assert a["source"] != "manual"

    def test_override_on_runpod_path_also_pins_count(self):
        # Provider-agnostic: the override applies on the RunPod path too.
        plan = r.resolve(
            _req(24),
            dynamic_gpu_enabled=True,
            force_single_gpu=True,
            max_gpu_usd_per_hour=None,
            headroom_multiplier=1.0,
            fallback_vram_gb=24,
            provider="runpod",
            gpu_count_override=1,
        )
        assert plan.gpu_count == 1
        assert plan.source == "manual"
