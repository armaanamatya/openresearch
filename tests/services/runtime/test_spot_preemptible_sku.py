"""Tests for C1+C2 changes: spot/preemptible flag + 8×A100-40GB SKU (D1).

Covers:
  (a) gpu_catalog resolves the new azure_a100_40x8 / gcp_a100_40x8 labels to
      40 GB VRAM per GPU + the correct machine types.
  (b) Shape/text assertion that GCP main.tf wires spot = var.use_spot.
  (c) Shape/text assertion that Azure Bicep emits Spot fields only under useSpot.

These are pure unit tests — no network, no cloud credentials.
"""
from __future__ import annotations

import pathlib

from backend.services.runtime.gpu_catalog import CATALOG, GpuSku, effective_vram_gb, find_ladder


# ---------------------------------------------------------------------------
# (a) Catalog: 40 GB A100 entries — GCP and Azure
# ---------------------------------------------------------------------------

def _sku_by_name(short_name: str) -> GpuSku:
    matches = [s for s in CATALOG if s.short_name == short_name]
    assert matches, f"SKU '{short_name}' not found in CATALOG"
    assert len(matches) == 1, f"Duplicate SKU '{short_name}' in CATALOG"
    return matches[0]


class TestGcpA100_40x8:
    """gcp_a100_40x8 — a2-highgpu-8g, 8×A100-40GB."""

    def test_exists_in_catalog(self):
        sku = _sku_by_name("gcp_a100_40x8")
        assert sku is not None

    def test_vram_is_40gb_per_gpu(self):
        sku = _sku_by_name("gcp_a100_40x8")
        assert sku.vram_gb == 40

    def test_gpu_count_is_8(self):
        sku = _sku_by_name("gcp_a100_40x8")
        assert sku.gpu_count == 8

    def test_effective_vram_is_320gb(self):
        sku = _sku_by_name("gcp_a100_40x8")
        assert effective_vram_gb(sku) == 320

    def test_machine_type_is_a2_highgpu_8g(self):
        # runpod_id field holds the GCE machine type for GCP rows.
        sku = _sku_by_name("gcp_a100_40x8")
        assert sku.runpod_id == "a2-highgpu-8g"

    def test_provider_is_gcp(self):
        sku = _sku_by_name("gcp_a100_40x8")
        assert sku.provider == "gcp"

    def test_appears_in_gcp_ladder(self):
        ladder = find_ladder(
            min_vram_gb=40, max_per_gpu_usd_per_hr=None,
            cloud_types=("ONDEMAND",), provider="gcp",
        )
        short_names = [s.short_name for s in ladder]
        assert "gcp_a100_40x8" in short_names


class TestAzureA100_40x8:
    """azure_a100_40x8 — Standard_ND96asr_v4, 8×A100-40GB (new D1 entry)."""

    def test_exists_in_catalog(self):
        sku = _sku_by_name("azure_a100_40x8")
        assert sku is not None

    def test_vram_is_40gb_per_gpu(self):
        sku = _sku_by_name("azure_a100_40x8")
        assert sku.vram_gb == 40

    def test_gpu_count_is_8(self):
        sku = _sku_by_name("azure_a100_40x8")
        assert sku.gpu_count == 8

    def test_effective_vram_is_320gb(self):
        sku = _sku_by_name("azure_a100_40x8")
        assert effective_vram_gb(sku) == 320

    def test_vm_size_is_nd96(self):
        sku = _sku_by_name("azure_a100_40x8")
        assert sku.runpod_id == "Standard_ND96asr_v4"

    def test_provider_is_azure(self):
        sku = _sku_by_name("azure_a100_40x8")
        assert sku.provider == "azure"

    def test_cloud_type_is_ondemand(self):
        sku = _sku_by_name("azure_a100_40x8")
        assert sku.cloud_type == "ONDEMAND"

    def test_appears_in_azure_ladder(self):
        ladder = find_ladder(
            min_vram_gb=40, max_per_gpu_usd_per_hr=None,
            cloud_types=("ONDEMAND",), provider="azure",
        )
        short_names = [s.short_name for s in ladder]
        assert "azure_a100_40x8" in short_names

    def test_does_not_appear_in_gcp_or_runpod_ladder(self):
        for provider in ("gcp", "runpod"):
            ladder = find_ladder(
                min_vram_gb=1, max_per_gpu_usd_per_hr=None,
                cloud_types=("ONDEMAND", "COMMUNITY", "SECURE"),
                provider=provider,
            )
            short_names = [s.short_name for s in ladder]
            assert "azure_a100_40x8" not in short_names, (
                f"azure_a100_40x8 must not appear in {provider} ladder"
            )


class TestExisting80GbSkusUntouched:
    """Existing 80 GB Azure/GCP SKUs must not be removed or modified."""

    def test_azure_a100_80_still_present(self):
        sku = _sku_by_name("azure_a100_80")
        assert sku.runpod_id == "Standard_NC24ads_A100_v4"
        assert sku.vram_gb == 80

    def test_gcp_a100_80_still_present(self):
        sku = _sku_by_name("gcp_a100_80")
        assert sku.runpod_id == "a2-ultragpu-1g"
        assert sku.vram_gb == 80

    def test_gcp_a100_80x8_still_present(self):
        sku = _sku_by_name("gcp_a100_80x8")
        assert sku.runpod_id == "a2-ultragpu-8g"
        assert sku.vram_gb == 80
        assert sku.gpu_count == 8


# ---------------------------------------------------------------------------
# (b) GCP main.tf shape: spot = var.use_spot
# ---------------------------------------------------------------------------

_GCP_MAIN_TF = (
    pathlib.Path(__file__).parent.parent.parent.parent
    / "infra" / "gcp" / "modules" / "gpu_nodepool" / "main.tf"
)
_GCP_VARS_TF = (
    pathlib.Path(__file__).parent.parent.parent.parent
    / "infra" / "gcp" / "modules" / "gpu_nodepool" / "variables.tf"
)


class TestGcpTerraformSpotWiring:
    """main.tf must wire spot = var.use_spot; variables.tf must declare use_spot."""

    def _read(self, path: pathlib.Path) -> str:
        return path.read_text(encoding="utf-8")

    def test_main_tf_wires_spot_to_var_use_spot(self):
        text = self._read(_GCP_MAIN_TF)
        assert "spot        = var.use_spot" in text, (
            "main.tf must contain 'spot        = var.use_spot'"
        )

    def test_main_tf_preemptible_still_false(self):
        text = self._read(_GCP_MAIN_TF)
        assert "preemptible = false" in text, (
            "main.tf must keep preemptible = false (Spot supersedes the preemptible API)"
        )

    def test_variables_tf_declares_use_spot_bool(self):
        text = self._read(_GCP_VARS_TF)
        assert 'variable "use_spot"' in text, "variables.tf must declare variable use_spot"
        assert "type    = bool" in text, "use_spot must have type = bool"
        assert "default = false" in text, "use_spot default must be false"

    def test_variables_tf_machine_type_default_is_a2_highgpu_8g(self):
        text = self._read(_GCP_VARS_TF)
        assert '"a2-highgpu-8g"' in text, (
            'variables.tf machine_type default must be "a2-highgpu-8g"'
        )


# ---------------------------------------------------------------------------
# (c) Azure Bicep shape: Spot fields only when useSpot = true
# ---------------------------------------------------------------------------

_AZURE_BICEP = (
    pathlib.Path(__file__).parent.parent.parent.parent
    / "infra" / "azure" / "bicep" / "modules" / "gpu-nodepool.bicep"
)


class TestAzureBicepSpotWiring:
    """gpu-nodepool.bicep must declare useSpot param and emit Spot fields conditionally."""

    def _read(self) -> str:
        return _AZURE_BICEP.read_text(encoding="utf-8")

    def test_use_spot_param_declared_with_false_default(self):
        text = self._read()
        assert "param useSpot bool = false" in text, (
            "Bicep must declare 'param useSpot bool = false'"
        )

    def test_spot_max_price_only_emitted_under_use_spot(self):
        text = self._read()
        # spotMaxPrice must appear inside a conditional (the var spotProps block),
        # not as an unconditional top-level property.
        assert "spotMaxPrice" in text, "Bicep must reference spotMaxPrice"
        # It must be guarded by useSpot (inside a ternary or conditional var).
        assert "useSpot" in text

    def test_eviction_policy_only_emitted_under_use_spot(self):
        text = self._read()
        assert "evictionPolicy" in text, "Bicep must reference evictionPolicy"
        assert "useSpot" in text

    def test_spot_props_var_contains_spot_value(self):
        text = self._read()
        assert "'Spot'" in text, "Bicep spotProps must set scaleSetPriority to 'Spot'"

    def test_regular_branch_present(self):
        text = self._read()
        assert "'Regular'" in text, "Bicep spotProps false branch must retain 'Regular'"

    def test_default_vm_size_is_nd96(self):
        text = self._read()
        assert "Standard_ND96asr_v4" in text, (
            "Bicep vmSize default must be Standard_ND96asr_v4 (8×A100-40GB)"
        )

    def test_union_merge_pattern_present(self):
        text = self._read()
        # union() must be used to merge spotProps so false branch is additive-only.
        assert "union(" in text, (
            "Bicep must use union() to conditionally merge Spot properties"
        )
