"""Tests for the multi-cloud hardware-brief resolver (Lane R).

The agent's implement_baseline prompt needs to know what hardware it
will actually run against — GPU model, VRAM, image, disk — so it can
pick batch sizes without probing. This suite pins the Azure ML resolver
(OPENRESEARCH_AZURE_VM_SIZE → SKU catalog from Microsoft Learn
/azure/virtual-machines/sizes/gpu-accelerated, May 2026).

Pinned invariants:

  * Azure SKU catalog covers the modern lineup (NCads_A100_v4,
    NCads_H100_v5, ND_H100_v5, ND_H200_v5, NV*ads_A10_v5, NC*as_T4_v3).
  * OPENRESEARCH_VRAM_OVERRIDE_GB beats any catalog lookup across all
    providers — manual override stays operator-controllable.
  * Unknown Azure SKU → falls through to {gpu_count: 1, vram_gb: None,
    vram_known: False} so the brief still renders with a "VRAM: unknown"
    line rather than crashing.
"""

from __future__ import annotations

import pytest

from backend.agents.baseline_implementation import (
    _AZURE_VM_SKU_CATALOG,
    _hardware_specs_block,
    _resolve_cloud_hardware,
)


@pytest.fixture(autouse=True)
def _clear_cloud_env(monkeypatch):
    """Strip every OPENRESEARCH_*_VM_SIZE / VRAM env so tests don't pollute each other."""
    for key in (
        "OPENRESEARCH_AZURE_VM_SIZE", "OPENRESEARCH_AZURE_REGION",
        "OPENRESEARCH_AZURE_IMAGE", "OPENRESEARCH_AZURE_DATA_DISK_GB",
        "OPENRESEARCH_AZURE_DATASTORE_GB", "OPENRESEARCH_AZURE_DATASTORE_MOUNT",
        "OPENRESEARCH_VRAM_OVERRIDE_GB",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Azure ML — new
# ---------------------------------------------------------------------------


def test_azure_a100_80gb_single_vm(monkeypatch):
    """Standard_NC24ads_A100_v4 → 1× A100 80GB.
    Verified against Microsoft Learn /azure/machine-learning/reference-managed-online-endpoints-vm-sku-list."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_NC24ads_A100_v4")
    monkeypatch.setenv("OPENRESEARCH_AZURE_REGION", "eastus")
    spec = _resolve_cloud_hardware("azure")
    assert spec is not None
    assert spec["cloud"] == "Azure ML"
    assert spec["gpu"] == "NVIDIA A100 80GB"
    assert spec["gpu_count"] == 1
    assert spec["vram_gb"] == 80
    assert spec["tier"] == "eastus"


def test_azure_a100_2gpu(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_NC48ads_A100_v4")
    spec = _resolve_cloud_hardware("azure")
    assert spec["gpu_count"] == 2
    assert spec["vram_gb"] == 80


def test_azure_a100_4gpu(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_NC96ads_A100_v4")
    spec = _resolve_cloud_hardware("azure")
    assert spec["gpu_count"] == 4


def test_azure_a100_ndm_8gpu_nvlink(monkeypatch):
    """Standard_ND96amsr_A100_v4 — 8-GPU NVLink for paper-scale training."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_ND96amsr_A100_v4")
    spec = _resolve_cloud_hardware("azure")
    assert spec["gpu_count"] == 8
    assert spec["gpu"] == "NVIDIA A100 80GB"


def test_azure_h100_nvl_94gb(monkeypatch):
    """Standard_NC40ads_H100_v5 — 1× H100 NVL with 94 GB.
    The H100 NVL bumps VRAM above the SXM5 80GB — important for the
    agent's batch-sizing math."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_NC40ads_H100_v5")
    spec = _resolve_cloud_hardware("azure")
    assert spec["gpu"] == "NVIDIA H100 NVL"
    assert spec["gpu_count"] == 1
    assert spec["vram_gb"] == 94


def test_azure_h100_sxm_8gpu(monkeypatch):
    """Standard_ND96isr_H100_v5 — 8× H100 SXM, 80 GB each."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_ND96isr_H100_v5")
    spec = _resolve_cloud_hardware("azure")
    assert spec["gpu_count"] == 8
    assert spec["vram_gb"] == 80


def test_azure_h200(monkeypatch):
    """Standard_ND96isr_H200_v5 — 8× H200, 141 GB each."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_ND96isr_H200_v5")
    spec = _resolve_cloud_hardware("azure")
    assert spec["gpu"] == "NVIDIA H200"
    assert spec["gpu_count"] == 8
    assert spec["vram_gb"] == 141


def test_azure_t4_single_card(monkeypatch):
    """Standard_NC4as_T4_v3 — cheapest T4 for dev / smoke runs."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_NC4as_T4_v3")
    spec = _resolve_cloud_hardware("azure")
    assert spec["gpu"] == "NVIDIA T4"
    assert spec["vram_gb"] == 16


def test_azure_unknown_sku_falls_through(monkeypatch):
    """Unknown SKU strings render with vram_known=False — brief still
    emits with a 'VRAM: unknown' line rather than crashing the prompt."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_NotARealSKU")
    spec = _resolve_cloud_hardware("azure")
    assert spec is not None
    assert spec["gpu"] == "Standard_NotARealSKU"
    assert spec["vram_known"] is False


def test_azure_default_image_is_curated(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_NC24ads_A100_v4")
    spec = _resolve_cloud_hardware("azure")
    assert "mcr.microsoft.com/azureml/curated/acpt-pytorch" in spec["image"]


def test_azure_image_override(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_NC24ads_A100_v4")
    monkeypatch.setenv(
        "OPENRESEARCH_AZURE_IMAGE",
        "mcr.microsoft.com/azureml/curated/acpt-pytorch-2.3-cuda12.4:latest",
    )
    spec = _resolve_cloud_hardware("azure")
    assert "2.3-cuda12.4" in spec["image"]


# ---------------------------------------------------------------------------
# VRAM override — operator-controlled across all clouds
# ---------------------------------------------------------------------------


def test_vram_override_beats_azure_catalog(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_NC24ads_A100_v4")  # would map to 80
    monkeypatch.setenv("OPENRESEARCH_VRAM_OVERRIDE_GB", "60")
    spec = _resolve_cloud_hardware("azure")
    assert spec["vram_gb"] == 60


# ---------------------------------------------------------------------------
# Prompt block emission
# ---------------------------------------------------------------------------


def test_block_emits_for_azure(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_NC40ads_H100_v5")
    monkeypatch.setenv("OPENRESEARCH_AZURE_REGION", "westus2")
    block = _hardware_specs_block("azure")
    assert "Cloud: Azure ML" in block
    assert "H100 NVL" in block
    assert "94 GB" in block
    assert "westus2" in block
    # Per-cloud image guidance — Azure version says ACPT, not runpod/pytorch.
    assert "mcr.microsoft.com/azureml" in block
    assert "do NOT re-install torch" in block


def test_block_omits_when_no_cloud_env_set():
    """Local-docker / local-process runs: no hardware brief."""
    assert _hardware_specs_block("docker") == ""
    assert _hardware_specs_block("local") == ""
    assert _hardware_specs_block(None) == ""


def test_constraint_guidance_wires_hardware_brief_for_azure(monkeypatch):
    """The brief must actually reach implement_baseline guidance — its only
    production call site was dropped with the RunPod prompt branch, leaving
    _resolve_cloud_hardware's Azure support unwired."""
    from backend.agents.baseline_implementation import _compute_constraint_guidance

    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_NC40ads_H100_v5")
    monkeypatch.setenv("OPENRESEARCH_AZURE_REGION", "westus2")
    g = _compute_constraint_guidance(sandbox_mode="azure", gpu_mode=None)
    assert "Cloud: Azure ML" in g
    assert "H100 NVL" in g


def test_constraint_guidance_no_hardware_brief_on_local(monkeypatch):
    from backend.agents.baseline_implementation import _compute_constraint_guidance

    g = _compute_constraint_guidance(sandbox_mode="local", gpu_mode=None)
    assert "Cloud: Azure ML" not in g


def test_block_includes_scope_reduction_guidance(monkeypatch):
    """Whichever cloud, the brief must point the agent at scope-adjusted
    rubric / scope.declared_reductions instead of mocks."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_NC24ads_A100_v4")
    block = _hardware_specs_block("azure")
    assert "scope-adjusted" in block.lower() or "declared_reductions" in block.lower() \
        or "scope reduction" in block.lower()
    assert "NEVER use" in block  # the mocks/surrogates anti-rule


# ---------------------------------------------------------------------------
# Catalog completeness
# ---------------------------------------------------------------------------


def test_azure_catalog_covers_modern_lineup():
    """Every catalog entry maps to a recognisable GPU model with non-zero VRAM."""
    must_cover = [
        "Standard_NC24ads_A100_v4",
        "Standard_NC40ads_H100_v5",
        "Standard_ND96isr_H100_v5",
        "Standard_ND96isr_H200_v5",
        "Standard_NC4as_T4_v3",
    ]
    for sku in must_cover:
        assert sku in _AZURE_VM_SKU_CATALOG, f"missing canonical SKU: {sku}"
        gpu, count, vram = _AZURE_VM_SKU_CATALOG[sku]
        assert gpu, f"{sku}: blank GPU"
        assert count > 0, f"{sku}: zero GPU count"
        assert vram > 0, f"{sku}: zero VRAM"
