"""Tests for the multi-cloud hardware-brief resolver (Lane R).

The agent's implement_baseline prompt needs to know what hardware it
will actually run against — GPU model, VRAM, image, disk — so it can
pick batch sizes without probing. Originally RunPod-only via the
OPENRESEARCH_RUNPOD_* env vars; this suite pins the generalisation that
also handles Azure ML (OPENRESEARCH_AZURE_VM_SIZE → SKU catalog from
Microsoft Learn /azure/virtual-machines/sizes/gpu-accelerated, May 2026)
and Brev.

Pinned invariants:

  * RunPod env unchanged from earlier ship — OPENRESEARCH_RUNPOD_GPU_TYPE
    drives the brief when sandbox contains "runpod".
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
    _GPU_VRAM_ESTIMATE_GB,
    _hardware_specs_block,
    _resolve_cloud_hardware,
)


@pytest.fixture(autouse=True)
def _clear_cloud_env(monkeypatch):
    """Strip every REPROLAB_*_GPU / VM_SIZE env so tests don't pollute each other."""
    for key in (
        "OPENRESEARCH_RUNPOD_GPU_TYPE", "OPENRESEARCH_RUNPOD_GPU_COUNT",
        "OPENRESEARCH_RUNPOD_CLOUD_TYPE", "OPENRESEARCH_RUNPOD_IMAGE",
        "OPENRESEARCH_RUNPOD_CONTAINER_DISK_GB", "OPENRESEARCH_RUNPOD_VOLUME_GB",
        "OPENRESEARCH_RUNPOD_VOLUME_MOUNT_PATH",
        "OPENRESEARCH_AZURE_VM_SIZE", "OPENRESEARCH_AZURE_REGION",
        "OPENRESEARCH_AZURE_IMAGE", "OPENRESEARCH_AZURE_DATA_DISK_GB",
        "OPENRESEARCH_AZURE_DATASTORE_GB", "OPENRESEARCH_AZURE_DATASTORE_MOUNT",
        "OPENRESEARCH_BREV_GPU_TYPE", "OPENRESEARCH_BREV_GPU_COUNT",
        "OPENRESEARCH_BREV_REGION", "OPENRESEARCH_BREV_IMAGE",
        "OPENRESEARCH_BREV_CONTAINER_DISK_GB",
        "OPENRESEARCH_VRAM_OVERRIDE_GB",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# RunPod — back-compat
# ---------------------------------------------------------------------------


def test_runpod_resolves_to_l40s(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_RUNPOD_GPU_TYPE", "NVIDIA L40S")
    monkeypatch.setenv("OPENRESEARCH_RUNPOD_CLOUD_TYPE", "SECURE")
    monkeypatch.setenv("OPENRESEARCH_RUNPOD_IMAGE", "runpod/pytorch:2.1.0")
    spec = _resolve_cloud_hardware("runpod")
    assert spec is not None
    assert spec["cloud"] == "RunPod"
    assert spec["gpu"] == "NVIDIA L40S"
    assert spec["vram_gb"] == 48
    assert spec["tier"] == "SECURE"
    assert "runpod/pytorch" in spec["image"]


def test_runpod_unknown_gpu_falls_through(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_RUNPOD_GPU_TYPE", "NVIDIA Made-Up 9999")
    spec = _resolve_cloud_hardware("runpod")
    assert spec is not None
    assert spec["vram_known"] is False
    assert spec["vram_gb"] is None


def test_runpod_no_env_returns_none():
    """No RunPod env set + sandbox=runpod → resolver returns None (nothing to brief)."""
    assert _resolve_cloud_hardware("runpod") is None


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


def test_vram_override_beats_runpod_catalog(monkeypatch):
    """Operator override bypasses the catalog (--vram-gb / OPENRESEARCH_VRAM_OVERRIDE_GB)."""
    monkeypatch.setenv("OPENRESEARCH_RUNPOD_GPU_TYPE", "NVIDIA L40S")  # would map to 48
    monkeypatch.setenv("OPENRESEARCH_VRAM_OVERRIDE_GB", "40")  # override
    spec = _resolve_cloud_hardware("runpod")
    assert spec["vram_gb"] == 40
    assert spec["vram_known"] is True


def test_vram_override_beats_azure_catalog(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_AZURE_VM_SIZE", "Standard_NC24ads_A100_v4")  # would map to 80
    monkeypatch.setenv("OPENRESEARCH_VRAM_OVERRIDE_GB", "60")
    spec = _resolve_cloud_hardware("azure")
    assert spec["vram_gb"] == 60


# ---------------------------------------------------------------------------
# Prompt block emission
# ---------------------------------------------------------------------------


def test_block_emits_for_runpod(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_RUNPOD_GPU_TYPE", "NVIDIA L40S")
    block = _hardware_specs_block("runpod")
    assert "Cloud: RunPod" in block
    assert "NVIDIA L40S" in block
    assert "48 GB" in block


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


def test_runpod_strings_have_vram_entries():
    """Every common RunPod GPU string the lab UI offers is in the VRAM map."""
    for gpu in ("NVIDIA L40S", "NVIDIA RTX 4090", "NVIDIA A100 80GB", "NVIDIA H100"):
        assert gpu in _GPU_VRAM_ESTIMATE_GB
