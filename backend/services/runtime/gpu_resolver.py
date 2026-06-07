"""Pure-logic GPU resolver. No I/O. No network. No imports beyond stdlib + schemas + catalog.

Given `GpuRequirements` + settings + budget, returns a `GpuPlan` with the cheapest
SKU that meets the VRAM target (after headroom multiplier + tier-up), respecting
the per-GPU $/hr cap and the force_single_gpu invariant.

For RunPod (provider="runpod", default): selects by per-GPU vram_gb; all catalog rows
have gpu_count=1 so effective_vram_gb == vram_gb.

For Azure (provider="azure"): selects by effective capacity (vram_gb * gpu_count) so
that multi-GPU VM sizes (NC48/NC96) are eligible when a paper needs >80 GB aggregated.
The returned GpuPlan.runpod_id carries the Azure VM size string (the opaque provider
identifier reused in that field) rather than a RunPod gpu_type string.

See `docs/superpowers/specs/2026-05-23-dynamic-gpu-selection-design.md` §Resolver.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from backend.agents.schemas import GpuPlan, GpuRequirements
from backend.services.runtime.gpu_catalog import GpuSku, effective_vram_gb, find_ladder

# Confidence threshold below which we treat the LLM estimate as unusable.
_CONFIDENCE_FLOOR: float = 0.4

# When estimate is unusable, we use this SKU regardless of paper.
_FALLBACK_SHORT_NAME: str = "rtx4090"

# Azure cloud_types passed into find_ladder for azure provider.
_AZURE_CLOUD_TYPES: tuple[str, ...] = ("ONDEMAND",)
# Azure fallback short_name: cheapest azure SKU (A10 24GB).
_AZURE_FALLBACK_SHORT_NAME: str = "azure_a10_24"


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
    provider: str = "runpod",
) -> GpuPlan:
    """Resolve requirements to a GpuPlan. Pure function — no I/O.

    Args:
        provider: Cloud provider to resolve against.  "runpod" (default) uses the
            existing RunPod COMMUNITY/SECURE catalog rows and is byte-for-byte
            identical to the pre-azure behaviour.  "azure" uses Azure ONDEMAND rows
            and selects by *effective* capacity (vram_gb * gpu_count) so that
            multi-GPU VM sizes (NC48/NC96) are considered when needed.

    Returns:
        GpuPlan with source in {"paper", "fallback", "manual", "informational"}.
        For azure plans, GpuPlan.runpod_id carries the Azure VM size string (the
        opaque provider identifier), GpuPlan.cloud_type is "ONDEMAND", and
        GpuPlan.gpu_count reflects the SKU's physical gpu_count.

    Raises:
        GpuResolutionError when constraints are infeasible (e.g., VRAM > largest catalog
        SKU, or required SKU exceeds the per-GPU $/hr cap). The error message names the
        cheapest SKU that would have satisfied VRAM if the cap were lifted.
    """
    if provider == "azure":
        return _resolve_azure(
            requirements=requirements,
            dynamic_gpu_enabled=dynamic_gpu_enabled,
            force_single_gpu=force_single_gpu,
            max_gpu_usd_per_hour=max_gpu_usd_per_hour,
            headroom_multiplier=headroom_multiplier,
        )
    return _resolve_runpod(
        requirements=requirements,
        dynamic_gpu_enabled=dynamic_gpu_enabled,
        force_single_gpu=force_single_gpu,
        max_gpu_usd_per_hour=max_gpu_usd_per_hour,
        headroom_multiplier=headroom_multiplier,
        fallback_vram_gb=fallback_vram_gb,
        cloud_types=cloud_types,
    )


def _resolve_runpod(
    requirements: GpuRequirements,
    *,
    dynamic_gpu_enabled: bool,
    force_single_gpu: bool,
    max_gpu_usd_per_hour: float | None,
    headroom_multiplier: float,
    fallback_vram_gb: int,
    cloud_types: tuple[str, ...],
) -> GpuPlan:
    """RunPod resolution path — identical to the pre-azure resolve() implementation."""
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


def _resolve_azure(
    requirements: GpuRequirements,
    *,
    dynamic_gpu_enabled: bool,
    force_single_gpu: bool,
    max_gpu_usd_per_hour: float | None,
    headroom_multiplier: float,
) -> GpuPlan:
    """Azure resolution path.

    Selects from Azure ONDEMAND rows by *effective* capacity (vram_gb * gpu_count)
    so multi-GPU VM sizes (NC48/NC96) are eligible when a paper needs >80 GB total.

    force_single_gpu: when True, only single-GPU SKUs (gpu_count == 1) are
    considered — same semantic as the RunPod path.

    The returned GpuPlan.runpod_id carries the Azure VM size string (e.g.
    "Standard_NC24ads_A100_v4") — the opaque provider identifier in both providers.
    GpuPlan.gpu_count reflects the physical gpu_count of the chosen SKU (1, 2, or 4).
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # Dynamic disabled → informational from cheapest azure SKU.
    if not dynamic_gpu_enabled:
        sku = _by_short_name(_AZURE_FALLBACK_SHORT_NAME, provider="azure")
        return _build_plan(sku, gpu_count=sku.gpu_count, source="informational",
                           requirements=requirements, ladder=(), now_iso=now_iso)

    estimate = requirements.estimated_vram_gb
    confidence = requirements.confidence

    # Fallback path: no estimate OR confidence too low → cheapest azure SKU.
    if estimate is None or confidence < _CONFIDENCE_FLOOR:
        sku = _by_short_name(_AZURE_FALLBACK_SHORT_NAME, provider="azure")
        full_ladder = _azure_ladder(min_effective_vram=sku.vram_gb * sku.gpu_count,
                                    max_per_gpu_usd=None,
                                    force_single_gpu=force_single_gpu)
        remaining = tuple(s.short_name for s in full_ladder
                          if s.short_name != sku.short_name)
        return _build_plan(sku, gpu_count=sku.gpu_count, source="fallback",
                           requirements=requirements, ladder=remaining, now_iso=now_iso)

    # Apply headroom multiplier (against per-GPU estimate; same logic as RunPod).
    needed_vram = math.ceil(estimate * max(headroom_multiplier, 1.0))

    # Build the azure ladder filtered by effective capacity and optionally single-GPU.
    ladder = _azure_ladder(min_effective_vram=needed_vram,
                           max_per_gpu_usd=max_gpu_usd_per_hour,
                           force_single_gpu=force_single_gpu)

    if not ladder:
        # Diagnose: would lifting the cap help?
        unconstrained = _azure_ladder(min_effective_vram=needed_vram,
                                      max_per_gpu_usd=None,
                                      force_single_gpu=force_single_gpu)
        if unconstrained:
            cheapest = unconstrained[0]
            raise GpuResolutionError(
                f"Paper requires >= {needed_vram} GB effective VRAM (after {headroom_multiplier}x headroom on "
                f"estimate={estimate}). Cheapest Azure SKU is {cheapest.short_name} at "
                f"${cheapest.approx_usd_per_hr:.2f}/hr/GPU, but `max_gpu_usd_per_hour` cap is "
                f"${max_gpu_usd_per_hour}. Raise the cap or set --vram-gb to a lower override."
            )
        raise GpuResolutionError(
            f"Paper requires >= {needed_vram} GB effective VRAM (estimate={estimate}, "
            f"multiplier={headroom_multiplier}x) but no Azure catalog SKU has that much. "
            f"Largest available Azure effective VRAM is {_largest_azure_vram(force_single_gpu)} GB. "
            f"Consider scoping down the experiment."
        )

    pick = ladder[0]
    # For azure, gpu_count comes from the SKU itself (not paper_gpu_count / cap math),
    # because the VM size already determines how many physical GPUs are attached.
    gpu_count = 1 if force_single_gpu else pick.gpu_count

    remaining = tuple(s.short_name for s in ladder[1:])
    return _build_plan(pick, gpu_count=gpu_count, source="paper",
                       requirements=requirements, ladder=remaining, now_iso=now_iso)


def _by_short_name(short_name: str, *, provider: str = "runpod") -> GpuSku:
    from backend.services.runtime.gpu_catalog import CATALOG
    for sku in CATALOG:
        if sku.short_name == short_name and sku.provider == provider:
            return sku
    raise GpuResolutionError(
        f"Catalog has no SKU with short_name={short_name!r}, provider={provider!r} (programmer error)"
    )


def _largest_vram(cloud_types: tuple[str, ...]) -> int:
    from backend.services.runtime.gpu_catalog import CATALOG
    candidates = [s.vram_gb for s in CATALOG if s.cloud_type in cloud_types]
    return max(candidates) if candidates else 0


def _azure_ladder(
    *,
    min_effective_vram: int,
    max_per_gpu_usd: float | None,
    force_single_gpu: bool,
) -> list[GpuSku]:
    """Return Azure SKUs sorted by (effective_vram_gb ASC, approx_usd_per_hr ASC).

    Filters:
        - provider == "azure"
        - cloud_type == "ONDEMAND"
        - effective_vram_gb(sku) >= min_effective_vram
        - if force_single_gpu: only gpu_count == 1 rows
        - if max_per_gpu_usd: per-GPU rate <= cap
    """
    from backend.services.runtime.gpu_catalog import CATALOG
    cap = max_per_gpu_usd if max_per_gpu_usd and max_per_gpu_usd > 0 else None
    filtered = [
        sku for sku in CATALOG
        if sku.provider == "azure"
        and sku.cloud_type == "ONDEMAND"
        and effective_vram_gb(sku) >= min_effective_vram
        and (not force_single_gpu or sku.gpu_count == 1)
        and (cap is None or sku.approx_usd_per_hr <= cap)
    ]
    return sorted(filtered, key=lambda s: (effective_vram_gb(s), s.approx_usd_per_hr))


def _largest_azure_vram(force_single_gpu: bool) -> int:
    """Largest effective VRAM among azure SKUs, optionally restricted to single-GPU."""
    from backend.services.runtime.gpu_catalog import CATALOG
    candidates = [
        effective_vram_gb(s) for s in CATALOG
        if s.provider == "azure"
        and (not force_single_gpu or s.gpu_count == 1)
    ]
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
