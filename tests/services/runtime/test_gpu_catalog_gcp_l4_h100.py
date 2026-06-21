"""GCP L4 + H100 catalog SKUs and cost-cap behaviour (SCOPE 3). Pure, hermetic."""
from __future__ import annotations

import backend.services.runtime.gpu_catalog as cat
from backend.services.runtime.gpu_catalog import find_by_alias, find_ladder


def test_gcp_l4_sku_present():
    by_name = {s.short_name: s for s in cat.CATALOG if s.provider == "gcp"}
    l4 = by_name["gcp_l4_24"]
    assert l4.provider == "gcp" and l4.cloud_type == "ONDEMAND"
    assert l4.vram_gb == 24 and l4.gpu_count == 1 and l4.approx_usd_per_hr > 0


def test_gcp_h100_skus_present():
    by_name = {s.short_name: s for s in cat.CATALOG if s.provider == "gcp"}
    h1 = by_name["gcp_h100_80"]
    assert h1.vram_gb == 80 and h1.gpu_count == 1 and h1.cloud_type == "ONDEMAND"
    h8 = by_name["gcp_h100_80x8"]
    assert h8.vram_gb == 80 and h8.gpu_count == 8


def test_resolver_picks_l4_for_small_vram():
    ladder = find_ladder(24, None, cloud_types=("ONDEMAND",), provider="gcp")
    assert ladder
    assert ladder[0].short_name == "gcp_l4_24"


def test_resolver_picks_h100_when_alias_matches():
    sku = find_by_alias("trained on 8x h100", provider="gcp")
    assert sku is not None and sku.provider == "gcp" and "h100" in sku.short_name


def test_per_gpu_cap_excludes_expensive_h100():
    ladder = find_ladder(80, 10.0, cloud_types=("ONDEMAND",), provider="gcp")
    assert "gcp_h100_80x8" not in {s.short_name for s in ladder}


def test_per_gpu_cap_none_means_no_cap():
    ladder = find_ladder(80, None, cloud_types=("ONDEMAND",), provider="gcp")
    assert "gcp_h100_80x8" in {s.short_name for s in ladder}


def test_runpod_catalog_unchanged_by_gcp_additions():
    runpod = find_ladder(24, None)  # default provider
    assert all(s.provider == "runpod" for s in runpod)
    assert "gcp_l4_24" not in {s.short_name for s in runpod}
