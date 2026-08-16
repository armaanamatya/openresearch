"""Static GPU SKU catalog for GCP, Azure, and AWS dynamic GPU selection.

Vendored prices are approximate snapshots refreshed quarterly. The resolver's
ranking between SKUs is what matters; absolute prices may drift ±20%.

See `docs/history/specs/2026-05-23-dynamic-gpu-selection-design.md` §Catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GpuSku:
    """One row of the catalog.

    `runpod_id` is the opaque provider SKU identifier — the literal string the
    provider's API accepts (a GCE machine type for GCP, e.g. "a2-highgpu-1g";
    an Azure VM size for Azure, e.g. "Standard_NC24ads_A100_v4"). The field name
    is retained for schema stability; callers that need to distinguish providers
    must check `provider`.

    `aliases` is a tuple of lowercase substrings that the alias resolver matches
    against paper text — order does not matter, longest first wins on tie.

    `provider` identifies the cloud backend: "gcp" (default), "azure", or "aws".
    Note: `cloud_type` is a *tier* within a provider (ONDEMAND for GCP/Azure) —
    it is distinct from `provider`.

    `gpu_count` is the number of GPUs per node for this SKU. Single-GPU rows use
    1; multi-GPU VM sizes (e.g. NC48/NC96, a2-highgpu-8g) use 2/4/8.
    """
    runpod_id: str
    short_name: str
    vram_gb: int
    cloud_type: str
    approx_usd_per_hr: float
    aliases: tuple[str, ...] = field(default_factory=tuple)
    provider: str = "gcp"
    gpu_count: int = 1


def effective_vram_gb(sku: GpuSku) -> int:
    """Total VRAM across all GPUs in this SKU (vram_gb * gpu_count)."""
    return sku.vram_gb * sku.gpu_count


def usd_per_gpu_hr(sku: GpuSku) -> float:
    """Per-GPU $/hr for this SKU (``approx_usd_per_hr / gpu_count``).

    ``approx_usd_per_hr`` is the **whole-machine** on-demand rate (an
    ``a2-ultragpu-8g`` is $31.44/hr for all 8 A100s, not $31.44 per GPU) —
    ``feasibility_triage`` depends on that meaning, so it stays as-is. But the
    ``max_gpu_usd_per_hour`` cap and ``GpuPlan.sku_usd_per_hr`` are both defined
    PER-GPU (see the GpuPlan schema: "Per-GPU rate from catalog"). Comparing a
    whole-machine price against a per-GPU cap wrongly excluded every multi-GPU
    node — an 8×A100-80 machine ($31.44 total = $3.93/GPU) failed the default
    $10/GPU cap and a genuine multi-GPU paper could not resolve at all.

    For every ``gpu_count == 1`` row — the single-GPU azure/gcp rows — this
    returns ``approx_usd_per_hr`` unchanged.
    """
    return sku.approx_usd_per_hr / max(1, sku.gpu_count)


CATALOG: tuple[GpuSku, ...] = (
    # Azure rows — sorted by effective capacity (vram_gb * gpu_count)
    # then price within each tier.  Azure VM sizes serve as the provider SKU id
    # (reused in the runpod_id field).  Prices are eastus on-demand list rates;
    # refresh quarterly.
    GpuSku("Standard_NV36ads_A10_v5",   "azure_a10_24",    24, "ONDEMAND",  1.20,
           aliases=("a10", "a10 24"),         provider="azure", gpu_count=1),
    GpuSku("Standard_NC24ads_A100_v4",  "azure_a100_80",   80, "ONDEMAND",  3.67,
           aliases=("a100 80", "a100"),        provider="azure", gpu_count=1),
    GpuSku("Standard_NC48ads_A100_v4",  "azure_a100_80x2", 80, "ONDEMAND",  7.35,
           aliases=("2x a100",),              provider="azure", gpu_count=2),
    GpuSku("Standard_NC96ads_A100_v4",  "azure_a100_80x4", 80, "ONDEMAND", 14.69,
           aliases=("4x a100",),              provider="azure", gpu_count=4),
    # Standard_ND96asr_v4 = 8×A100-40GB SXM4 (NVLink).  This is the Azure mirror
    # of the GCP a2-highgpu-8g node (catalog: gcp_a100_40x8).  Sized so the 7B
    # 8-GPU SDAR cell fits one node; scale-to-zero ⇒ idle=$0.
    # approx_usd_per_hr is the eastus on-demand list rate; refresh quarterly.
    GpuSku("Standard_ND96asr_v4",      "azure_a100_40x8", 40, "ONDEMAND", 27.20,
           aliases=("8x a100 40", "nd96"),    provider="azure", gpu_count=8),

    # GCP rows — Google Compute Engine A2 machine types on GKE. The provider SKU
    # id is the GCE machine type (e.g. "a2-highgpu-8g"); the orchestrator maps it
    # to a GKE node pool. a2-highgpu = A100 40GB; a2-ultragpu = A100 80GB. The
    # 4g/8g rows are the dynamically-selectable 4×A100 / 8×A100 clusters.
    # approx_usd_per_hr is the TOTAL on-demand rate for the machine (us-central1,
    # ~$2.93/GPU for 40GB, ~$3.93/GPU for 80GB — refresh quarterly).
    GpuSku("a2-highgpu-1g",  "gcp_a100_40",    40, "ONDEMAND",  2.93,
           aliases=("a100 40", "a100 40gb", "a100-40"), provider="gcp", gpu_count=1),
    GpuSku("a2-highgpu-2g",  "gcp_a100_40x2",  40, "ONDEMAND",  5.86,
           aliases=("2x a100", "2x a100 40"),           provider="gcp", gpu_count=2),
    GpuSku("a2-highgpu-4g",  "gcp_a100_40x4",  40, "ONDEMAND", 11.72,
           aliases=("4x a100", "4x a100 40"),           provider="gcp", gpu_count=4),
    GpuSku("a2-highgpu-8g",  "gcp_a100_40x8",  40, "ONDEMAND", 23.44,
           aliases=("8x a100", "8x a100 40"),           provider="gcp", gpu_count=8),
    GpuSku("a2-ultragpu-1g", "gcp_a100_80",    80, "ONDEMAND",  3.93,
           aliases=("a100 80", "a100 80gb", "a100"),    provider="gcp", gpu_count=1),
    GpuSku("a2-ultragpu-2g", "gcp_a100_80x2",  80, "ONDEMAND",  7.86,
           aliases=("2x a100 80",),                     provider="gcp", gpu_count=2),
    GpuSku("a2-ultragpu-4g", "gcp_a100_80x4",  80, "ONDEMAND", 15.72,
           aliases=("4x a100 80",),                     provider="gcp", gpu_count=4),
    GpuSku("a2-ultragpu-8g", "gcp_a100_80x8",  80, "ONDEMAND", 31.44,
           aliases=("8x a100 80",),                     provider="gcp", gpu_count=8),
    # GCP L4 + H100 (opt-in step-up; A100 stays the default ladder).
    # L4 = g2-standard-8 (1x L4-24GB); H100 = a3-highgpu-{1,8}g (H100-80GB SXM).
    # approx_usd_per_hr = TOTAL machine on-demand rate (us-central1; refresh quarterly).
    GpuSku("g2-standard-8", "gcp_l4_24",     24, "ONDEMAND",  0.85,
           aliases=("l4", "l4 24", "nvidia l4"),     provider="gcp", gpu_count=1),
    GpuSku("a3-highgpu-1g", "gcp_h100_80",   80, "ONDEMAND", 11.06,
           aliases=("h100", "h100 80", "h100 80gb", "h100-80"),
           provider="gcp", gpu_count=1),
    GpuSku("a3-highgpu-8g", "gcp_h100_80x8", 80, "ONDEMAND", 88.49,
           aliases=("8x h100", "8x h100 80", "a3-highgpu-8g"),
           provider="gcp", gpu_count=8),
)


def find_ladder(
    min_vram_gb: int,
    max_per_gpu_usd_per_hr: float | None,
    cloud_types: tuple[str, ...] = ("ONDEMAND",),
    *,
    provider: str = "gcp",
) -> list[GpuSku]:
    """Return SKUs meeting all filters, sorted by ascending effective capacity then price.

    Args:
        min_vram_gb: minimum required VRAM per GPU in GB; SKU must have vram_gb >= this
        max_per_gpu_usd_per_hr: per-GPU $/hr cap; None or 0 means no cap
        cloud_types: which cloud types are acceptable (e.g. ("ONDEMAND",) for
            GCP/Azure)
        provider: restrict to this provider; defaults to "gcp".  Azure callers
            pass provider="azure".

    Returns:
        Filtered list sorted by (effective_vram_gb ASC, approx_usd_per_hr ASC);
        empty list when no SKU qualifies.
    """
    cap = max_per_gpu_usd_per_hr if max_per_gpu_usd_per_hr and max_per_gpu_usd_per_hr > 0 else None
    filtered = [
        sku for sku in CATALOG
        if sku.provider == provider
        and sku.vram_gb >= min_vram_gb
        and sku.cloud_type in cloud_types
        and (cap is None or sku.approx_usd_per_hr <= cap)
    ]
    return sorted(filtered, key=lambda s: (effective_vram_gb(s), s.approx_usd_per_hr))


def find_by_alias(phrase: str, *, provider: str = "gcp") -> GpuSku | None:
    """Lookup the first SKU whose alias appears as a substring of `phrase` (case-insensitive).

    Longest alias wins on tie to avoid 'a100' matching when 'a100 80gb' is present.
    Returns None when no alias matches.

    Args:
        phrase: free-form text to search (e.g. from a paper)
        provider: restrict search to this provider; defaults to "gcp".
    """
    needle = phrase.lower()
    best: tuple[int, GpuSku] | None = None
    for sku in CATALOG:
        if sku.provider != provider:
            continue
        for alias in sku.aliases:
            if alias in needle:
                if best is None or len(alias) > best[0]:
                    best = (len(alias), sku)
    return best[1] if best else None


__all__ = [
    "GpuSku",
    "CATALOG",
    "effective_vram_gb",
    "usd_per_gpu_hr",
    "find_ladder",
    "find_by_alias",
]
