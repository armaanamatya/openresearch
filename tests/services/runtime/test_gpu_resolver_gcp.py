"""GCP (GKE) GPU catalog + resolver — dynamic A100 cluster selection.

GCP is a provisioned-node-pool cloud (like Azure AKS): a fixed set of A2 machine
types (a2-highgpu-1g…8g = A100 40GB ×1/2/4/8; a2-ultragpu = A100 80GB) attached
to GKE node pools. The resolver picks by EFFECTIVE capacity (vram_gb × gpu_count)
so a paper needing multiple A100s dynamically selects a 4×/8× machine — the
headline requirement. Pure tests: no GCP, no network.
"""
from __future__ import annotations

import pytest

from backend.agents.schemas import GpuRequirements
from backend.services.runtime import gpu_catalog as cat
from backend.services.runtime import gpu_resolver as r


def _req(vram, conf=0.9):
    return GpuRequirements(estimated_vram_gb=vram, confidence=conf,
                           paper_gpu_count=1, rationale="t")


def _resolve(req, **kw):
    base = dict(dynamic_gpu_enabled=True, force_single_gpu=False,
                max_gpu_usd_per_hour=None, headroom_multiplier=1.0,
                fallback_vram_gb=24, provider="gcp")
    base.update(kw)
    return r.resolve(req, **base)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

class TestGcpCatalog:
    def test_catalog_has_the_a100_family(self):
        gcp = {s.short_name: s for s in cat.CATALOG if s.provider == "gcp"}
        # the 40GB highgpu family (1/2/4/8 GPU) + the 80GB ultragpu family
        for name in ("gcp_a100_40", "gcp_a100_40x4", "gcp_a100_40x8",
                     "gcp_a100_80", "gcp_a100_80x4", "gcp_a100_80x8"):
            assert name in gcp, f"missing GCP SKU {name}"

    def test_machine_types_and_gpu_counts(self):
        by_name = {s.short_name: s for s in cat.CATALOG if s.provider == "gcp"}
        assert by_name["gcp_a100_40x8"].runpod_id == "a2-highgpu-8g"
        assert by_name["gcp_a100_40x8"].gpu_count == 8
        assert by_name["gcp_a100_40x4"].gpu_count == 4
        assert by_name["gcp_a100_80x8"].runpod_id == "a2-ultragpu-8g"
        assert all(s.cloud_type == "ONDEMAND" for s in by_name.values())

    def test_effective_vram_is_per_gpu_times_count(self):
        x8 = next(s for s in cat.CATALOG if s.short_name == "gcp_a100_40x8")
        assert cat.effective_vram_gb(x8) == 40 * 8

    def test_find_ladder_gcp_is_provider_scoped(self):
        ladder = cat.find_ladder(40, None, cloud_types=("ONDEMAND",), provider="gcp")
        assert ladder and all(s.provider == "gcp" for s in ladder)
        # ascending by effective capacity
        eff = [cat.effective_vram_gb(s) for s in ladder]
        assert eff == sorted(eff)

    def test_alias_resolves_cluster_size(self):
        assert cat.find_by_alias("8x a100", provider="gcp").gpu_count == 8


# ---------------------------------------------------------------------------
# Dynamic resolution — the 4×/8× A100 cluster selection
# ---------------------------------------------------------------------------

class TestGcpResolve:
    def test_small_model_picks_single_a100(self):
        plan = _resolve(_req(30))
        assert plan.gpu_count == 1
        assert plan.runpod_id.startswith("a2-")

    def test_large_model_picks_multi_gpu_cluster(self):
        # ~140 GB effective needs >=2 A100s; resolver picks the cheapest fitting machine.
        plan = _resolve(_req(140))
        assert plan.gpu_count >= 2
        assert cat.effective_vram_gb(
            next(s for s in cat.CATALOG if s.short_name == plan.short_name)) >= 140

    def test_huge_model_picks_8_or_4_gpu(self):
        plan = _resolve(_req(300))
        assert plan.gpu_count >= 4
        assert plan.runpod_id.startswith("a2-")

    def test_force_single_gpu_never_multi(self):
        plan = _resolve(_req(30), force_single_gpu=True)
        assert plan.gpu_count == 1

    def test_dynamic_disabled_is_informational_fallback(self):
        plan = _resolve(_req(30), dynamic_gpu_enabled=False)
        assert plan.source == "informational"
        assert plan.short_name == "gcp_a100_40"

    def test_low_confidence_falls_back(self):
        plan = _resolve(_req(30, conf=0.1))
        assert plan.source == "fallback"
        assert plan.short_name == "gcp_a100_40"

    def test_provisioned_skus_restricts_selection(self):
        # Only a 4×80GB pool is provisioned → a big paper must land on it, never 8×.
        plan = _resolve(_req(200), provisioned_skus=("gcp_a100_80x4",))
        assert plan.short_name == "gcp_a100_80x4"

    def test_ladder_remaining_is_gcp_only(self):
        plan = _resolve(_req(40))
        for sn in plan.ladder_remaining:
            sku = next((s for s in cat.CATALOG if s.short_name == sn), None)
            assert sku is not None and sku.provider == "gcp"

    def test_unsatisfiable_cap_raises_named_error(self):
        with pytest.raises(r.GpuResolutionError) as ei:
            _resolve(_req(40), max_gpu_usd_per_hour=0.01)
        assert "GCP" in str(ei.value)
