from __future__ import annotations

import pytest

from backend.agents.schemas import GpuPlan, GpuRequirements
from backend.services.runtime.gpu_resolver import (
    GpuResolutionError,
    resolve,
)


def _req(vram: int | None = 40, count: int | None = 8, conf: float = 0.85) -> GpuRequirements:
    return GpuRequirements(
        estimated_vram_gb=vram,
        paper_gpu_string="A100 80GB" if vram else None,
        paper_gpu_count=count,
        reasoning="test",
        confidence=conf,
    )


def test_resolve_picks_cheapest_meeting_vram_with_multiplier():
    # Multiplier 1.25 on 40 -> 50; cheapest SKU with vram>=50 under cap is L40S (48GB? no — 48<50; A100 80GB 1.89, but A6000 is 48 which is <50; correct cheapest is A100 80 at 1.89)
    # Actually 48 < 50 (after multiplier 50), so we tier up to 80 (A100 80GB at $1.89).
    plan = resolve(
        _req(vram=40),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        cloud_types=("COMMUNITY",),
    )
    assert plan.short_name == "a100_80"
    assert plan.gpu_count == 1
    assert plan.source == "paper"


def test_resolve_force_single_gpu_caps_count_at_one():
    plan = resolve(
        _req(vram=20, count=8),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        cloud_types=("COMMUNITY",),
    )
    assert plan.gpu_count == 1


def test_resolve_multi_gpu_bounded_by_cost_cap():
    # Without force_single: paper says 8x; SKU A100 80GB is $1.89/hr; cap $10/hr
    # -> floor(10 / 1.89) = 5; min(8, 5) = 5.
    plan = resolve(
        _req(vram=64, count=8),
        dynamic_gpu_enabled=True,
        force_single_gpu=False,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        cloud_types=("COMMUNITY",),
    )
    assert plan.gpu_count == 5
    assert plan.total_usd_per_hr <= 10.0


def test_resolve_low_confidence_triggers_fallback_sku():
    plan = resolve(
        _req(vram=40, conf=0.2),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        cloud_types=("COMMUNITY",),
    )
    assert plan.source == "fallback"
    assert plan.short_name == "rtx4090"
    assert plan.gpu_count == 1


def test_resolve_none_estimate_triggers_fallback():
    plan = resolve(
        _req(vram=None, conf=0.9),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        cloud_types=("COMMUNITY",),
    )
    assert plan.source == "fallback"


def test_resolve_raises_when_no_sku_under_cap_meets_vram():
    with pytest.raises(GpuResolutionError) as exc:
        resolve(
            _req(vram=200),  # > largest SKU vram (141)
            dynamic_gpu_enabled=True,
            force_single_gpu=True,
            max_gpu_usd_per_hour=10.0,
            headroom_multiplier=1.25,
            fallback_vram_gb=24,
            cloud_types=("COMMUNITY", "SECURE"),
        )
    assert "200" in str(exc.value) or "no SKU" in str(exc.value).lower()


def test_resolve_raises_when_required_sku_exceeds_cap():
    # H100 80GB needed but cap forces only RTX 4090/A6000/L40S/A100s
    # vram 80 required; cap $0.50/hr -> only RTX 4090 ($0.34), A5000 ($0.36), A6000 ($0.49)
    # none have vram>=80 -> raise
    with pytest.raises(GpuResolutionError):
        resolve(
            _req(vram=80),
            dynamic_gpu_enabled=True,
            force_single_gpu=True,
            max_gpu_usd_per_hour=0.50,
            headroom_multiplier=1.0,
            fallback_vram_gb=24,
            cloud_types=("COMMUNITY",),
        )


def test_resolve_ladder_contains_next_larger_skus():
    plan = resolve(
        _req(vram=24),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.0,
        fallback_vram_gb=24,
        cloud_types=("COMMUNITY",),
    )
    # picked: RTX 4090 ($0.34), 24GB. Next-cheapest with vram>=24: A5000.
    # ladder_remaining should contain A5000 (and onward up the ladder).
    assert plan.short_name == "rtx4090"
    assert "a5000" in plan.ladder_remaining


def test_resolve_disabled_dynamic_returns_informational_plan_from_fallback_default():
    plan = resolve(
        _req(vram=80),
        dynamic_gpu_enabled=False,  # OFF
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        cloud_types=("COMMUNITY",),
    )
    assert plan.source == "informational"
    assert plan.short_name == "rtx4090"


def test_resolve_is_pure_no_io_imports():
    """Smoke: gpu_resolver must not import anything that performs network/disk I/O."""
    import importlib
    import sys
    mod = importlib.import_module("backend.services.runtime.gpu_resolver")
    forbidden = {"requests", "httpx", "urllib", "subprocess", "socket", "asyncio", "asyncssh"}
    deps = set()
    for name, m in list(sys.modules.items()):
        if m and getattr(m, "__file__", "") and "backend/services/runtime/gpu_resolver" in (m.__file__ or ""):
            for v in vars(m).values():
                modname = getattr(getattr(v, "__module__", None), "split", lambda *_: [""])(".")[0] if v else ""
                if modname:
                    deps.add(modname)
    assert not (deps & forbidden), f"resolver pulled in I/O modules: {deps & forbidden}"
