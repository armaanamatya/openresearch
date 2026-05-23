from __future__ import annotations

import pytest

from backend.services.runtime.gpu_catalog import (
    CATALOG,
    GpuSku,
    find_ladder,
    find_by_alias,
)


def test_catalog_is_nonempty_tuple_of_gpusku():
    assert isinstance(CATALOG, tuple)
    assert len(CATALOG) >= 7
    assert all(isinstance(sku, GpuSku) for sku in CATALOG)


def test_catalog_sorted_by_vram_then_price():
    keys = [(sku.vram_gb, sku.approx_usd_per_hr) for sku in CATALOG]
    assert keys == sorted(keys), "CATALOG must be sorted by (vram_gb, price) ASC for readability"


def test_find_ladder_returns_only_skus_meeting_vram_and_cap():
    ladder = find_ladder(min_vram_gb=40, max_per_gpu_usd_per_hr=2.0, cloud_types=("COMMUNITY",))
    assert all(s.vram_gb >= 40 for s in ladder)
    assert all(s.approx_usd_per_hr <= 2.0 for s in ladder)
    assert all(s.cloud_type == "COMMUNITY" for s in ladder)


def test_find_ladder_sorts_by_ascending_price():
    ladder = find_ladder(min_vram_gb=24, max_per_gpu_usd_per_hr=10.0, cloud_types=("COMMUNITY",))
    prices = [s.approx_usd_per_hr for s in ladder]
    assert prices == sorted(prices)


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


def test_find_by_alias_resolves_common_phrases():
    assert find_by_alias("a100").short_name == "a100_80"
    assert find_by_alias("A100 40GB").short_name == "a100_40"
    assert find_by_alias("RTX 4090").short_name == "rtx4090"
    assert find_by_alias("H100").short_name == "h100_80"


def test_find_by_alias_returns_none_on_unknown():
    assert find_by_alias("nonexistent-gpu") is None
