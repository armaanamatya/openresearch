"""Static GPU SKU catalog for RunPod dynamic GPU selection.

Vendored prices are approximate snapshots refreshed quarterly. The resolver's
ranking between SKUs is what matters; absolute prices may drift ±20%.

See `docs/superpowers/specs/2026-05-23-dynamic-gpu-selection-design.md` §Catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GpuSku:
    """One row of the catalog.

    `runpod_id` is the literal string RunPod's API accepts for `gpu_type`.
    `aliases` is a tuple of lowercase substrings that the alias resolver matches
    against paper text — order does not matter, longest first wins on tie.
    """
    runpod_id: str
    short_name: str
    vram_gb: int
    cloud_type: str
    approx_usd_per_hr: float
    aliases: tuple[str, ...] = field(default_factory=tuple)


CATALOG: tuple[GpuSku, ...] = (
    # Sorted by (vram_gb ASC, approx_usd_per_hr ASC) for human readability.
    # find_ladder() re-sorts the filtered result by price for selection.
    GpuSku("NVIDIA GeForce RTX 4090",     "rtx4090",   24, "COMMUNITY", 0.34,
           aliases=("rtx 4090", "geforce 4090", "rtx4090", "4090")),
    GpuSku("NVIDIA RTX A5000",            "a5000",     24, "COMMUNITY", 0.36,
           aliases=("a5000", "rtx a5000")),
    GpuSku("NVIDIA A100 40GB PCIe",       "a100_40",   40, "COMMUNITY", 1.19,
           aliases=("a100 40", "a100 40gb", "a100-40", "a100 40 gb")),
    GpuSku("NVIDIA RTX A6000",            "a6000",     48, "COMMUNITY", 0.49,
           aliases=("a6000", "rtx a6000")),
    GpuSku("NVIDIA L40S",                 "l40s",      48, "COMMUNITY", 0.86,
           aliases=("l40s", "l40 s")),
    GpuSku("NVIDIA A100 80GB PCIe",       "a100_80",   80, "COMMUNITY", 1.89,
           aliases=("a100 80", "a100 80gb", "a100-80", "a100")),
    GpuSku("NVIDIA H100 80GB HBM3",       "h100_80",   80, "COMMUNITY", 4.39,
           aliases=("h100", "h100 80", "h100 80gb", "h100-80")),
    GpuSku("NVIDIA H200",                 "h200",     141, "SECURE",    7.99,
           aliases=("h200",)),
)


def find_ladder(
    min_vram_gb: int,
    max_per_gpu_usd_per_hr: float | None,
    cloud_types: tuple[str, ...] = ("COMMUNITY",),
) -> list[GpuSku]:
    """Return SKUs meeting all filters, sorted by ascending price.

    Args:
        min_vram_gb: minimum required VRAM in GB; SKU must have vram_gb >= this
        max_per_gpu_usd_per_hr: per-GPU $/hr cap; None or 0 means no cap
        cloud_types: which cloud types are acceptable

    Returns:
        Filtered list sorted by approx_usd_per_hr ASC; empty list when no SKU qualifies.
    """
    cap = max_per_gpu_usd_per_hr if max_per_gpu_usd_per_hr and max_per_gpu_usd_per_hr > 0 else None
    filtered = [
        sku for sku in CATALOG
        if sku.vram_gb >= min_vram_gb
        and sku.cloud_type in cloud_types
        and (cap is None or sku.approx_usd_per_hr <= cap)
    ]
    return sorted(filtered, key=lambda s: s.approx_usd_per_hr)


def find_by_alias(phrase: str) -> GpuSku | None:
    """Lookup the first SKU whose alias appears as a substring of `phrase` (case-insensitive).

    Longest alias wins on tie to avoid 'a100' matching when 'a100 80gb' is present.
    Returns None when no alias matches.
    """
    needle = phrase.lower()
    best: tuple[int, GpuSku] | None = None
    for sku in CATALOG:
        for alias in sku.aliases:
            if alias in needle:
                if best is None or len(alias) > best[0]:
                    best = (len(alias), sku)
    return best[1] if best else None


__all__ = ["GpuSku", "CATALOG", "find_ladder", "find_by_alias"]
