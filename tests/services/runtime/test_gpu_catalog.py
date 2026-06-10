from __future__ import annotations


from backend.services.runtime.gpu_catalog import (
    CATALOG,
    GpuSku,
    effective_vram_gb,
    find_ladder,
    find_by_alias,
)


# ---------------------------------------------------------------------------
# Catalog structure
# ---------------------------------------------------------------------------

def test_catalog_is_nonempty_tuple_of_gpusku():
    assert isinstance(CATALOG, tuple)
    assert len(CATALOG) >= 7
    assert all(isinstance(sku, GpuSku) for sku in CATALOG)


def test_runpod_section_sorted_by_vram_then_price():
    """RunPod rows (the leading section) must stay sorted by (vram_gb, price) ASC.

    Azure rows are appended after the RunPod section and have their own ordering,
    so the whole-catalog sort invariant is scoped to the RunPod provider only.
    """
    runpod_rows = [sku for sku in CATALOG if sku.provider == "runpod"]
    keys = [(sku.vram_gb, sku.approx_usd_per_hr) for sku in runpod_rows]
    assert keys == sorted(keys), "RunPod rows must be sorted by (vram_gb, price) ASC"


# ---------------------------------------------------------------------------
# GpuSku defaults — provider and gpu_count must not affect existing rows
# ---------------------------------------------------------------------------

def test_existing_sku_defaults():
    """All RunPod rows must carry provider='runpod' and gpu_count=1 via the defaults."""
    for sku in CATALOG:
        if sku.provider == "runpod":
            assert sku.gpu_count == 1, f"{sku.short_name}: expected gpu_count=1"


def test_explicit_provider_kwarg_on_new_sku():
    """Constructing a standalone SKU without provider kwarg yields provider='runpod'."""
    sku = GpuSku("SOME_ID", "test_gpu", 24, "COMMUNITY", 1.0, aliases=("test",))
    assert sku.provider == "runpod"
    assert sku.gpu_count == 1


# ---------------------------------------------------------------------------
# find_ladder — runpod (default, regression)
# ---------------------------------------------------------------------------

def test_find_ladder_returns_only_runpod_by_default():
    """Default provider='runpod' must return ONLY runpod rows; no Azure leakage."""
    ladder = find_ladder(min_vram_gb=24, max_per_gpu_usd_per_hr=None, cloud_types=("COMMUNITY", "SECURE", "ONDEMAND"))
    assert all(s.provider == "runpod" for s in ladder), "Azure rows must not appear in default ladder"


def test_find_ladder_runpod_short_names_and_order():
    """Regression: the full RunPod ladder (no cap) must contain the canonical short names
    in ascending effective-vram / price order, which for RunPod equals vram_gb / price order."""
    ladder = find_ladder(min_vram_gb=24, max_per_gpu_usd_per_hr=None, cloud_types=("COMMUNITY", "SECURE"))
    short_names = [s.short_name for s in ladder]
    # All known RunPod names must appear (order: effective_vram ASC, price ASC within tier)
    expected = ["rtx4090", "a5000", "a100_40", "a6000", "l40s", "a100_80", "h100_80", "h200"]
    assert short_names == expected, f"RunPod ladder order changed: {short_names}"


def test_find_ladder_returns_only_skus_meeting_vram_and_cap():
    ladder = find_ladder(min_vram_gb=40, max_per_gpu_usd_per_hr=2.0, cloud_types=("COMMUNITY",))
    assert all(s.vram_gb >= 40 for s in ladder)
    assert all(s.approx_usd_per_hr <= 2.0 for s in ladder)
    assert all(s.cloud_type == "COMMUNITY" for s in ladder)


def test_find_ladder_sorts_by_ascending_effective_capacity_then_price():
    """Ladder must be ordered by (effective_vram_gb ASC, price ASC).

    For RunPod where gpu_count is always 1, effective_vram_gb == vram_gb, so
    the sort is (vram_gb ASC, price ASC).  This supersedes the previous
    price-only assertion, which was correct only because the original catalog
    had no multi-GPU SKUs.
    """
    ladder = find_ladder(min_vram_gb=24, max_per_gpu_usd_per_hr=10.0, cloud_types=("COMMUNITY",))
    keys = [(effective_vram_gb(s), s.approx_usd_per_hr) for s in ladder]
    assert keys == sorted(keys), f"Ladder not in (effective_vram_gb, price) order: {keys}"


def test_find_ladder_returns_empty_when_cap_too_low():
    assert find_ladder(min_vram_gb=80, max_per_gpu_usd_per_hr=0.10, cloud_types=("COMMUNITY",)) == []


def test_find_ladder_excludes_secure_only_when_community_filter():
    ladder = find_ladder(min_vram_gb=24, max_per_gpu_usd_per_hr=100.0, cloud_types=("COMMUNITY",))
    assert not any(s.short_name == "h200" for s in ladder), "H200 is SECURE-only; must not appear under COMMUNITY filter"


def test_find_ladder_includes_secure_when_filter_permits():
    ladder = find_ladder(
        min_vram_gb=100,
        max_per_gpu_usd_per_hr=100.0,
        cloud_types=("COMMUNITY", "SECURE"),
    )
    assert any(s.short_name == "h200" for s in ladder)


def test_find_ladder_no_cap_when_cap_is_none():
    ladder = find_ladder(min_vram_gb=24, max_per_gpu_usd_per_hr=None, cloud_types=("COMMUNITY",))
    assert any(s.short_name == "h100_80" for s in ladder), "no cap must include H100"


# ---------------------------------------------------------------------------
# find_ladder — azure
# ---------------------------------------------------------------------------

def test_find_ladder_azure_returns_only_azure_rows():
    ladder = find_ladder(min_vram_gb=1, max_per_gpu_usd_per_hr=None, cloud_types=("ONDEMAND",), provider="azure")
    assert len(ladder) >= 4
    assert all(s.provider == "azure" for s in ladder), "No runpod rows must appear in azure ladder"


def test_find_ladder_azure_a100_80gb_effective_capacity_order():
    """Azure 80 GB ladder must be ordered by effective capacity: 1×80 < 2×80 < 4×80."""
    ladder = find_ladder(80, None, ("ONDEMAND",), provider="azure")
    short_names = [s.short_name for s in ladder]
    assert short_names == ["azure_a100_80", "azure_a100_80x2", "azure_a100_80x4"], (
        f"Azure A100 80 GB ladder must be sorted by effective capacity: {short_names}"
    )


def test_find_ladder_azure_effective_capacity_ascending():
    """Effective VRAM (vram_gb * gpu_count) must be non-decreasing across the full azure ladder."""
    ladder = find_ladder(1, None, ("ONDEMAND",), provider="azure")
    caps = [effective_vram_gb(s) for s in ladder]
    assert caps == sorted(caps), f"Effective VRAM must be non-decreasing: {caps}"


def test_find_ladder_azure_includes_a10():
    ladder = find_ladder(24, None, ("ONDEMAND",), provider="azure")
    assert any(s.short_name == "azure_a10_24" for s in ladder)


def test_find_ladder_azure_provider_sku_ids():
    """Verify the Azure VM size strings (stored as runpod_id) are correct."""
    azure_skus = {s.short_name: s for s in CATALOG if s.provider == "azure"}
    assert azure_skus["azure_a10_24"].runpod_id == "Standard_NV36ads_A10_v5"
    assert azure_skus["azure_a100_80"].runpod_id == "Standard_NC24ads_A100_v4"
    assert azure_skus["azure_a100_80x2"].runpod_id == "Standard_NC48ads_A100_v4"
    assert azure_skus["azure_a100_80x4"].runpod_id == "Standard_NC96ads_A100_v4"


def test_find_ladder_azure_gpu_counts():
    azure_skus = {s.short_name: s for s in CATALOG if s.provider == "azure"}
    assert azure_skus["azure_a10_24"].gpu_count == 1
    assert azure_skus["azure_a100_80"].gpu_count == 1
    assert azure_skus["azure_a100_80x2"].gpu_count == 2
    assert azure_skus["azure_a100_80x4"].gpu_count == 4


# ---------------------------------------------------------------------------
# effective_vram_gb helper
# ---------------------------------------------------------------------------

def test_effective_vram_gb_single_gpu():
    sku = GpuSku("X", "test", 80, "ONDEMAND", 3.67, provider="azure", gpu_count=1)
    assert effective_vram_gb(sku) == 80


def test_effective_vram_gb_multi_gpu():
    sku = GpuSku("X", "test", 80, "ONDEMAND", 7.35, provider="azure", gpu_count=2)
    assert effective_vram_gb(sku) == 160


# ---------------------------------------------------------------------------
# find_by_alias — runpod unchanged
# ---------------------------------------------------------------------------

def test_find_by_alias_resolves_common_phrases():
    assert find_by_alias("a100").short_name == "a100_80"
    assert find_by_alias("A100 40GB").short_name == "a100_40"
    assert find_by_alias("RTX 4090").short_name == "rtx4090"
    assert find_by_alias("H100").short_name == "h100_80"


def test_find_by_alias_returns_none_on_unknown():
    assert find_by_alias("nonexistent-gpu") is None


def test_find_by_alias_azure_only_aliases_absent_from_runpod():
    """Azure-exclusive aliases must resolve under provider='azure' but not provider='runpod'.

    'a10' is only a GpuSku alias in the Azure section; the RunPod section has no
    SKU whose alias list contains 'a10' (though 'a100' does, 'a10' alone does not).
    """
    # 'a10' in isolation has no runpod alias match
    assert find_by_alias("a10") is None, "a10 (azure-only) must not appear in default runpod lookup"
    # But under provider='azure' it resolves correctly
    assert find_by_alias("a10", provider="azure") is not None
    assert find_by_alias("a10", provider="azure").short_name == "azure_a10_24"
