"""When configured gcp_gpu_skus match no available cluster SKU label, the
resolver must raise a clear, actionable error naming configured vs available
and the exact OPENRESEARCH_GCP_GPU_SKUS fix - not a bare GpuResolutionError."""

from __future__ import annotations

import pytest

from backend.services.runtime.gpu_resolver import (
    validate_configured_skus,
    GpuSkuConfigError,
)


def test_mismatch_raises_actionable_error():
    with pytest.raises(GpuSkuConfigError) as exc:
        validate_configured_skus(
            configured=["gcp_a100_80x8"],
            available=["gcp_a100_80", "gcp_a100_80x2"],
        )
    msg = str(exc.value)
    assert "gcp_a100_80x8" in msg
    assert "gcp_a100_80" in msg
    assert "OPENRESEARCH_GCP_GPU_SKUS" in msg


def test_overlap_passes():
    validate_configured_skus(
        configured=["gcp_a100_80"],
        available=["gcp_a100_80", "gcp_a100_80x2"],
    )


def test_empty_available_raises():
    with pytest.raises(GpuSkuConfigError):
        validate_configured_skus(configured=["gcp_a100_80"], available=[])
