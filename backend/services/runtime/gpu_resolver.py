"""Pure-logic GPU resolver. No I/O. No network. No imports beyond stdlib + schemas + catalog.

Given `GpuRequirements` + settings + budget, returns a `GpuPlan` with the cheapest
RunPod SKU that meets the VRAM target (after headroom multiplier + tier-up), respecting
the per-GPU $/hr cap and the force_single_gpu invariant.

See `docs/superpowers/specs/2026-05-23-dynamic-gpu-selection-design.md` §Resolver.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from backend.agents.schemas import GpuPlan, GpuRequirements
from backend.services.runtime.gpu_catalog import GpuSku, find_ladder

# Confidence threshold below which we treat the LLM estimate as unusable.
_CONFIDENCE_FLOOR: float = 0.4

# When estimate is unusable, we use this SKU regardless of paper.
_FALLBACK_SHORT_NAME: str = "rtx4090"


class GpuResolutionError(RuntimeError):
    """Raised when no SKU can satisfy (VRAM + $/hr cap + cloud_type) constraints."""


def resolve(
    requirements: GpuRequirements,
    *,
    dynamic_gpu_enabled: bool,
    force_single_gpu: bool,
    max_gpu_usd_per_hour: float | None,
    headroom_multiplier: float,
    fallback_vram_gb: int,
    cloud_types: tuple[str, ...] = ("COMMUNITY",),
) -> GpuPlan:
    """Resolve requirements to a GpuPlan. Pure function — no I/O.

    Returns:
        GpuPlan with source in {"paper", "fallback", "manual", "informational"}.

    Raises:
        GpuResolutionError when constraints are infeasible (e.g., VRAM > largest catalog
        SKU, or required SKU exceeds the per-GPU $/hr cap). The error message names the
        cheapest SKU that would have satisfied VRAM if the cap were lifted.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # Dynamic disabled → return informational plan from the fallback SKU. RunpodBackend
    # ignores `gpu_plan` and uses legacy Settings.runpod_gpu_type when source is
    # "informational" (see primitive caller).
    if not dynamic_gpu_enabled:
        sku = _by_short_name(_FALLBACK_SHORT_NAME)
        return _build_plan(sku, gpu_count=1, source="informational",
                           requirements=requirements, ladder=(), now_iso=now_iso)

    estimate = requirements.estimated_vram_gb
    confidence = requirements.confidence

    # Fallback path: no estimate OR confidence too low.
    if estimate is None or confidence < _CONFIDENCE_FLOOR:
        sku = _by_short_name(_FALLBACK_SHORT_NAME)
        # Populate the full escalation ladder even on the fallback path.
        # When the default SKU (RTX 4090) is unavailable due to capacity issues
        # or RunPod API errors, the escalation loop in run_experiment can advance
        # to the next cheapest GPU rather than failing immediately.
        full_ladder = find_ladder(
            min_vram_gb=sku.vram_gb,
            max_per_gpu_usd_per_hr=None,
            cloud_types=cloud_types,
        )
        remaining = tuple(s.short_name for s in full_ladder if s.short_name != sku.short_name)
        return _build_plan(sku, gpu_count=1, source="fallback",
                           requirements=requirements, ladder=remaining, now_iso=now_iso)

    # Apply headroom multiplier; round up.
    needed_vram = math.ceil(estimate * max(headroom_multiplier, 1.0))

    # Find ladder under cap.
    ladder = find_ladder(
        min_vram_gb=needed_vram,
        max_per_gpu_usd_per_hr=max_gpu_usd_per_hour,
        cloud_types=cloud_types,
    )

    if not ladder:
        # Diagnose: would a SKU exist if we lifted the cap?
        unconstrained = find_ladder(
            min_vram_gb=needed_vram,
            max_per_gpu_usd_per_hr=None,
            cloud_types=cloud_types,
        )
        if unconstrained:
            cheapest = unconstrained[0]
            raise GpuResolutionError(
                f"Paper requires >= {needed_vram} GB VRAM (after {headroom_multiplier}x headroom on "
                f"estimate={estimate}). Cheapest SKU is {cheapest.short_name} at "
                f"${cheapest.approx_usd_per_hr:.2f}/hr, but `max_gpu_usd_per_hour` cap is "
                f"${max_gpu_usd_per_hour}. Raise the cap or set --vram-gb to a lower override."
            )
        raise GpuResolutionError(
            f"Paper requires >= {needed_vram} GB VRAM (estimate={estimate}, multiplier={headroom_multiplier}x) "
            f"but no catalog SKU has that much VRAM. "
            f"Largest available is {_largest_vram(cloud_types)} GB. Consider multi-GPU "
            f"reproduction or scoping down the experiment."
        )

    pick = ladder[0]

    # Count: force_single_gpu wins; else min(paper_count, cap_allows_count).
    if force_single_gpu:
        gpu_count = 1
    else:
        paper_count = max(1, requirements.paper_gpu_count or 1)
        if max_gpu_usd_per_hour and max_gpu_usd_per_hour > 0:
            cap_allows = max(1, int(math.floor(max_gpu_usd_per_hour / pick.approx_usd_per_hr)))
        else:
            cap_allows = paper_count
        gpu_count = min(paper_count, cap_allows)

    remaining = tuple(s.short_name for s in ladder[1:])
    return _build_plan(pick, gpu_count=gpu_count, source="paper",
                       requirements=requirements, ladder=remaining, now_iso=now_iso)


def _by_short_name(short_name: str) -> GpuSku:
    from backend.services.runtime.gpu_catalog import CATALOG
    for sku in CATALOG:
        if sku.short_name == short_name:
            return sku
    raise GpuResolutionError(f"Catalog has no SKU with short_name={short_name!r} (programmer error)")


def _largest_vram(cloud_types: tuple[str, ...]) -> int:
    from backend.services.runtime.gpu_catalog import CATALOG
    candidates = [s.vram_gb for s in CATALOG if s.cloud_type in cloud_types]
    return max(candidates) if candidates else 0


def _build_plan(
    sku: GpuSku,
    *,
    gpu_count: int,
    source: str,
    requirements: GpuRequirements,
    ladder: tuple[str, ...],
    now_iso: str,
) -> GpuPlan:
    return GpuPlan(
        runpod_id=sku.runpod_id,
        short_name=sku.short_name,
        vram_gb=sku.vram_gb,
        gpu_count=gpu_count,
        cloud_type=sku.cloud_type,
        sku_usd_per_hr=sku.approx_usd_per_hr,
        total_usd_per_hr=round(sku.approx_usd_per_hr * gpu_count, 4),
        container_disk_gb=max(50, sku.vram_gb),
        volume_gb=max(20, sku.vram_gb // 4),
        source=source,
        requirements=requirements,
        ladder_remaining=ladder,
        resolved_at=now_iso,
    )


__all__ = ["GpuResolutionError", "resolve"]
