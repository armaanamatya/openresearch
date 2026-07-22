from __future__ import annotations

import pytest

from backend.agents.schemas import GpuRequirements
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
    # Default provider=gcp (ONDEMAND). Multiplier 1.25 on 40 -> 50; cheapest gcp
    # single-GPU SKU with vram>=50 is the 80GB A100 (gcp_a100_80).
    plan = resolve(
        _req(vram=40),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
    )
    assert plan.short_name == "gcp_a100_80"
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
    )
    assert plan.gpu_count == 1


def test_resolve_low_confidence_triggers_fallback_sku():
    plan = resolve(
        _req(vram=40, conf=0.2),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
    )
    assert plan.source == "fallback"
    assert plan.short_name == "gcp_a100_40"
    assert plan.gpu_count == 1


def test_resolve_none_estimate_triggers_fallback():
    plan = resolve(
        _req(vram=None, conf=0.9),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
    )
    assert plan.source == "fallback"


def test_resolve_raises_when_no_sku_under_cap_meets_vram():
    with pytest.raises(GpuResolutionError) as exc:
        resolve(
            _req(vram=1024),  # > largest effective SKU vram
            dynamic_gpu_enabled=True,
            force_single_gpu=True,
            max_gpu_usd_per_hour=100.0,
            headroom_multiplier=1.25,
            fallback_vram_gb=24,
        )
    assert "1024" in str(exc.value) or "no" in str(exc.value).lower()


def test_resolve_disabled_dynamic_returns_informational_plan_from_fallback_default():
    plan = resolve(
        _req(vram=80),
        dynamic_gpu_enabled=False,  # OFF
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
    )
    assert plan.source == "informational"
    assert plan.short_name == "gcp_a100_40"


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


# ---------------------------------------------------------------------------
# Regression: default provider="runpod" must produce the same plan as before
# ---------------------------------------------------------------------------

def test_resolve_gcp_default_provider_unchanged():
    """Explicit provider="gcp" must be byte-for-byte equal to the no-provider call."""
    req = _req(vram=40)
    kwargs = dict(
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=10.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
    )
    plan_implicit = resolve(req, **kwargs)
    plan_explicit = resolve(req, **kwargs, provider="gcp")
    # Same SKU and structural fields — timestamps will differ by microseconds so
    # compare everything except resolved_at.
    assert plan_explicit.short_name == plan_implicit.short_name
    assert plan_explicit.runpod_id == plan_implicit.runpod_id
    assert plan_explicit.gpu_count == plan_implicit.gpu_count
    assert plan_explicit.cloud_type == plan_implicit.cloud_type
    assert plan_explicit.sku_usd_per_hr == plan_implicit.sku_usd_per_hr
    assert plan_explicit.total_usd_per_hr == plan_implicit.total_usd_per_hr
    assert plan_explicit.ladder_remaining == plan_implicit.ladder_remaining
    assert plan_explicit.source == plan_implicit.source
    assert plan_implicit.cloud_type == "ONDEMAND"


# ---------------------------------------------------------------------------
# Azure-specific tests
# ---------------------------------------------------------------------------

def _azure_req(vram: int, count: int = 1, conf: float = 0.85) -> GpuRequirements:
    return GpuRequirements(
        estimated_vram_gb=vram,
        paper_gpu_string=f"{count}x A100",
        paper_gpu_count=count,
        reasoning="azure test",
        confidence=conf,
    )


def test_resolve_azure_70gb_picks_a100_80_single():
    """~70 GB requirement → a100_80 (NC24ads, 80 GB, 1 GPU, $3.67/hr).

    With force_single_gpu=False the multi-GPU SKUs appear in the ladder;
    with force_single_gpu=True only single-GPU azure rows are considered so the
    ladder above a100_80 is empty (it is the largest single-GPU azure SKU).
    """
    plan = resolve(
        _azure_req(vram=70),
        dynamic_gpu_enabled=True,
        force_single_gpu=False,  # allow multi-GPU ladder
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.0,
        fallback_vram_gb=24,
        provider="azure",
    )
    assert plan.short_name == "azure_a100_80"
    assert plan.gpu_count == 1
    assert plan.cloud_type == "ONDEMAND"
    # runpod_id carries the Azure VM size string.
    assert plan.runpod_id == "Standard_NC24ads_A100_v4"
    assert plan.sku_usd_per_hr == pytest.approx(3.67)
    assert plan.total_usd_per_hr == pytest.approx(3.67)
    assert plan.source == "paper"
    # ladder_remaining must include larger azure SKUs.
    assert "azure_a100_80x2" in plan.ladder_remaining
    assert "azure_a100_80x4" in plan.ladder_remaining


def test_resolve_azure_70gb_force_single_no_ladder():
    """force_single_gpu=True → only single-GPU azure rows; azure_a100_80 is the largest, ladder empty."""
    plan = resolve(
        _azure_req(vram=70),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.0,
        fallback_vram_gb=24,
        provider="azure",
    )
    assert plan.short_name == "azure_a100_80"
    assert plan.gpu_count == 1
    # No larger single-GPU Azure SKU exists above 80 GB.
    assert plan.ladder_remaining == ()


def test_resolve_azure_ladder_remaining_ordering():
    """ladder_remaining for azure_a100_80 is sorted by effective VRAM then price (multi-GPU allowed)."""
    plan = resolve(
        _azure_req(vram=70),
        dynamic_gpu_enabled=True,
        force_single_gpu=False,
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.0,
        fallback_vram_gb=24,
        provider="azure",
    )
    # azure_a100_80x2 (160 GB eff) should come before azure_a100_80x4 (320 GB eff).
    assert plan.ladder_remaining.index("azure_a100_80x2") < plan.ladder_remaining.index("azure_a100_80x4")


def test_resolve_azure_over_80gb_with_force_single_raises():
    """Requirement >80 GB with force_single_gpu=True → no qualifying SKU → error."""
    with pytest.raises(GpuResolutionError):
        resolve(
            _azure_req(vram=100),
            dynamic_gpu_enabled=True,
            force_single_gpu=True,
            max_gpu_usd_per_hour=20.0,
            headroom_multiplier=1.0,
            fallback_vram_gb=24,
            provider="azure",
        )


def test_resolve_azure_over_80gb_multi_gpu_picks_x2():
    """Requirement >80 GB with force_single_gpu=False → picks a100_80x2 (160 GB eff)."""
    plan = resolve(
        _azure_req(vram=100, count=2),
        dynamic_gpu_enabled=True,
        force_single_gpu=False,
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.0,
        fallback_vram_gb=24,
        provider="azure",
    )
    assert plan.short_name == "azure_a100_80x2"
    assert plan.gpu_count == 2
    assert plan.cloud_type == "ONDEMAND"
    assert plan.runpod_id == "Standard_NC48ads_A100_v4"
    # GpuPlan's schema defines sku_usd_per_hr as the PER-GPU rate and
    # total_usd_per_hr as (per-GPU x gpu_count). The catalog's approx_usd_per_hr
    # ($7.35) is the WHOLE-MACHINE rate for this 2-GPU VM, so the per-GPU rate is
    # $3.675 and the machine total is $7.35 -- NOT the $14.70 the old assertion
    # encoded (7.35 x 2), which double-billed every multi-GPU node.
    assert plan.sku_usd_per_hr == pytest.approx(7.35 / 2)
    assert plan.total_usd_per_hr == pytest.approx(7.35)
    assert "azure_a100_80x4" in plan.ladder_remaining


def test_resolve_azure_cost_fields_populated():
    """sku_usd_per_hr and total_usd_per_hr are taken from the catalog."""
    plan = resolve(
        _azure_req(vram=20),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.0,
        fallback_vram_gb=24,
        provider="azure",
    )
    # Cheapest azure SKU with effective_vram >= 20 is azure_a10_24 ($1.20).
    assert plan.short_name == "azure_a10_24"
    assert plan.sku_usd_per_hr == pytest.approx(1.20)
    assert plan.total_usd_per_hr == pytest.approx(1.20)
    assert plan.gpu_count == 1


def test_resolve_azure_low_confidence_returns_fallback():
    """Low confidence → fallback to cheapest azure SKU (azure_a10_24)."""
    plan = resolve(
        _azure_req(vram=80, conf=0.2),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        provider="azure",
    )
    assert plan.source == "fallback"
    assert plan.short_name == "azure_a10_24"


def test_resolve_azure_disabled_dynamic_returns_informational():
    """dynamic_gpu_enabled=False → informational plan from cheapest azure SKU."""
    plan = resolve(
        _azure_req(vram=80),
        dynamic_gpu_enabled=False,
        force_single_gpu=True,
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        provider="azure",
    )
    assert plan.source == "informational"
    assert plan.short_name == "azure_a10_24"
    assert plan.cloud_type == "ONDEMAND"


def test_resolve_azure_gcp_rows_not_mixed_in():
    """Azure resolution must never return a GCP SKU and vice-versa."""
    azure_plan = resolve(
        _azure_req(vram=24),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.0,
        fallback_vram_gb=24,
        provider="azure",
    )
    gcp_plan = resolve(
        _req(vram=24),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.0,
        fallback_vram_gb=24,
        provider="gcp",
    )
    assert azure_plan.cloud_type == "ONDEMAND"
    assert gcp_plan.cloud_type == "ONDEMAND"
    # They must draw from different provider sections.
    assert azure_plan.short_name.startswith("azure_")
    assert gcp_plan.short_name.startswith("gcp_")
    assert azure_plan.runpod_id != gcp_plan.runpod_id


# ---------------------------------------------------------------------------
# provisioned_skus filter tests (P1 fix)
# ---------------------------------------------------------------------------

def test_provisioned_skus_filters_primary_and_ladder():
    """provisioned_skus=("azure_a100_80",) → primary must be azure_a100_80,
    and x2/x4 must NOT appear in primary or ladder_remaining."""
    plan = resolve(
        _azure_req(vram=70),
        dynamic_gpu_enabled=True,
        force_single_gpu=False,
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.0,
        fallback_vram_gb=24,
        provider="azure",
        provisioned_skus=("azure_a100_80",),
    )
    assert plan.short_name == "azure_a100_80"
    assert "azure_a100_80x2" not in plan.ladder_remaining
    assert "azure_a100_80x4" not in plan.ladder_remaining


def test_provisioned_skus_none_unchanged_behavior():
    """provisioned_skus=None → identical behaviour to calling without the param (regression)."""
    kwargs = dict(
        dynamic_gpu_enabled=True,
        force_single_gpu=False,
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.0,
        fallback_vram_gb=24,
        provider="azure",
    )
    req = _azure_req(vram=70)
    plan_no_param = resolve(req, **kwargs)
    plan_none = resolve(req, **kwargs, provisioned_skus=None)
    assert plan_no_param.short_name == plan_none.short_name
    assert plan_no_param.ladder_remaining == plan_none.ladder_remaining
    assert plan_no_param.source == plan_none.source


def test_provisioned_skus_single_entry_ladder_empty():
    """With only one provisioned SKU and it is the primary, ladder_remaining is empty."""
    plan = resolve(
        _azure_req(vram=70),
        dynamic_gpu_enabled=True,
        force_single_gpu=False,
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.0,
        fallback_vram_gb=24,
        provider="azure",
        provisioned_skus=("azure_a100_80",),
    )
    assert plan.short_name == "azure_a100_80"
    assert plan.ladder_remaining == ()


def test_provisioned_skus_low_confidence_fallback_filters_ladder():
    """Even on the fallback path, provisioned_skus restricts ladder_remaining."""
    plan = resolve(
        _azure_req(vram=80, conf=0.2),
        dynamic_gpu_enabled=True,
        force_single_gpu=False,
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.0,
        fallback_vram_gb=24,
        provider="azure",
        provisioned_skus=("azure_a10_24", "azure_a100_80"),
    )
    assert plan.source == "fallback"
    # x2/x4 are not in provisioned_skus → must not appear in ladder
    assert "azure_a100_80x2" not in plan.ladder_remaining
    assert "azure_a100_80x4" not in plan.ladder_remaining


# ---------------------------------------------------------------------------
# P1 fix: fallback primary pick must target a provisioned pool
# ---------------------------------------------------------------------------

def test_azure_fallback_picks_provisioned_sku_when_estimate_none():
    """No estimate + provisioned_skus=("azure_a100_80",) → fallback must pick azure_a100_80.

    Previously the fallback path called _by_short_name(_AZURE_FALLBACK_SHORT_NAME)
    unconditionally, returning azure_a10_24 even when that pool is not provisioned,
    causing the K8s Job to hang Pending.
    """
    plan = resolve(
        GpuRequirements(
            estimated_vram_gb=None,
            paper_gpu_string=None,
            paper_gpu_count=1,
            reasoning="no estimate",
            confidence=0.9,
        ),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        provider="azure",
        provisioned_skus=("azure_a100_80",),
    )
    assert plan.short_name == "azure_a100_80"
    assert plan.gpu_count == 1
    assert plan.source == "fallback"


def test_azure_fallback_low_confidence_picks_provisioned():
    """Low-confidence estimate + provisioned_skus=("azure_a100_80",) → picks azure_a100_80.

    Confidence below _CONFIDENCE_FLOOR (0.4) also hits the fallback branch; the
    primary pick must still target the provisioned pool, not the unconstrained default.
    """
    plan = resolve(
        GpuRequirements(
            estimated_vram_gb=80,
            paper_gpu_string="1x A100",
            paper_gpu_count=1,
            reasoning="low confidence",
            confidence=0.1,  # below _CONFIDENCE_FLOOR=0.4
        ),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        provider="azure",
        provisioned_skus=("azure_a100_80",),
    )
    assert plan.short_name == "azure_a100_80"
    assert plan.source == "fallback"


def test_azure_fallback_unconstrained_preserves_a10():
    """No estimate + provisioned_skus=() → unchanged behavior: global fallback azure_a10_24."""
    plan = resolve(
        GpuRequirements(
            estimated_vram_gb=None,
            paper_gpu_string=None,
            paper_gpu_count=1,
            reasoning="no estimate unconstrained",
            confidence=0.9,
        ),
        dynamic_gpu_enabled=True,
        force_single_gpu=True,
        max_gpu_usd_per_hour=20.0,
        headroom_multiplier=1.25,
        fallback_vram_gb=24,
        provider="azure",
        provisioned_skus=(),
    )
    assert plan.short_name == "azure_a10_24"
    assert plan.source == "fallback"
