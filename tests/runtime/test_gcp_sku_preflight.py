"""ensure_gcp_available's SKU check: fail loud on a CONFIRMED mismatch between
configured gcp_gpu_skus and the reprolab/sku labels on live cluster nodes; but
be best-effort about the QUERY - if nodes can't be enumerated, warn and pass."""

from __future__ import annotations

import pytest

from backend.services.runtime import gke_job_backend as gjb
from backend.services.runtime.gpu_resolver import GpuSkuConfigError


class _FakeNode:
    def __init__(self, sku):
        self.metadata = type("M", (), {"labels": {"reprolab/sku": sku} if sku else {}})()


class _FakeCoreApi:
    def __init__(self, skus, raises=False):
        self._skus = skus
        self._raises = raises

    def list_node(self, **kw):
        if self._raises:
            raise RuntimeError("boom")
        return type("L", (), {"items": [_FakeNode(s) for s in self._skus]})()


def test_confirmed_mismatch_raises():
    with pytest.raises(GpuSkuConfigError):
        gjb.validate_gcp_skus_against_cluster(
            configured=["gcp_a100_80x8"],
            core_api=_FakeCoreApi(["gcp_a100_80", "gcp_a100_80x2"]),
        )


def test_overlap_passes():
    gjb.validate_gcp_skus_against_cluster(
        configured=["gcp_a100_80"],
        core_api=_FakeCoreApi(["gcp_a100_80"]),
    )


def test_node_query_failure_is_best_effort(caplog):
    # Can't enumerate -> warn and pass (do NOT raise).
    gjb.validate_gcp_skus_against_cluster(
        configured=["gcp_a100_80x8"],
        core_api=_FakeCoreApi([], raises=True),
    )


def test_no_labeled_nodes_is_best_effort():
    # Cluster reachable but zero reprolab/sku labels discovered -> can't confirm
    # a mismatch -> pass (best-effort).
    gjb.validate_gcp_skus_against_cluster(
        configured=["gcp_a100_80x8"],
        core_api=_FakeCoreApi([None, None]),
    )
