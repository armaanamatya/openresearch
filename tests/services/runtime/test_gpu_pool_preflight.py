"""Tests for gpu_pool_preflight — GPU SKU <-> node-pool drift guard.

Two evidence tiers, in precedence order:
  tier 1  node-pool API  (AUTHORITATIVE — lists a pool even at ZERO nodes)
  tier 2  live Nodes     (heuristic — blind to a scaled-to-zero pool)
  tier 3  cannot_verify  (warn, proceed — NEVER a hard block)

Covers:
  * COLD cluster + drift -> HARD BLOCK via the pool API (the case the live-Node
    heuristic structurally cannot see: a scale-to-zero pool has no Node object).
  * warm cluster + drift -> hard block (via whichever tier answers).
  * all SKUs provisioned -> passes; tier 2 is not even queried when tier 1 answers.
  * pool API unreachable -> falls back to the node heuristic (no hard block from
    the API failure itself).
  * pool API PERMISSION DENIED (403, the real in-cluster GSA state) + cold
    cluster -> cannot_verify, warn only, run proceeds.
  * corroboration invariant: an observation that sees the label on NOTHING is
    cannot_verify, never "every configured SKU is absent" (guards both the
    scale-to-zero blind spot AND a hypothetical parse bug's blast radius).
  * exactly one query per tier, regardless of how many SKUs/cells.
  * nothing configured -> no query at all.
  * both pool shapes (typed container_v1 object, raw REST dict) normalize.

All fakes are plain objects — no ``kubernetes`` / ``google-cloud-*`` package and
no network (the suite is socket-hermetic).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from backend.services.runtime.gpu_pool_preflight import (
    GpuPoolPreflightResult,
    check_gpu_pools_provisioned,
    enforce_gpu_pool_preflight,
    make_gke_pool_lister,
)
from backend.services.runtime.interface import RuntimeCauseKind, SandboxRuntimeError


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _node(sku: str | None) -> SimpleNamespace:
    labels = {"reprolab/sku": sku} if sku else {}
    return SimpleNamespace(metadata=SimpleNamespace(labels=labels))


class _CountingCoreApi:
    """Injectable fake CoreV1Api — counts list_node() calls (tier 2)."""

    def __init__(
        self,
        *,
        node_skus: list[str] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self._node_skus = node_skus if node_skus is not None else []
        self._raise_exc = raise_exc
        self.list_node_call_count = 0

    def list_node(self) -> Any:
        self.list_node_call_count += 1
        if self._raise_exc is not None:
            raise self._raise_exc
        return SimpleNamespace(items=[_node(s) for s in self._node_skus])


class _CountingPoolLister:
    """Injectable fake tier-1 node-pool lister — counts calls.

    ``pool_skus`` entries become pools carrying that ``reprolab/sku`` label; a
    ``None`` entry becomes an unlabelled pool (e.g. the CPU system pool).
    """

    def __init__(
        self,
        *,
        pool_skus: list[str | None] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self._pool_skus = pool_skus if pool_skus is not None else []
        self._raise_exc = raise_exc
        self.call_count = 0

    def __call__(self) -> list[dict[str, Any]]:
        self.call_count += 1
        if self._raise_exc is not None:
            raise self._raise_exc
        pools: list[dict[str, Any]] = []
        for i, sku in enumerate(self._pool_skus):
            labels = {"reprolab/sku": sku, "reprolab/node-type": "gpu"} if sku else {}
            pools.append({"name": sku or f"system-pool-{i}", "config": {"labels": labels}})
        return pools


class _PermissionDenied(Exception):
    """Stand-in for the GKE API's 403 when container.clusters.get is missing."""


_COLD = []  # zero live nodes — every GPU pool scaled to zero (the common idle state)


# ---------------------------------------------------------------------------
# THE GAP THIS CLOSES: cold cluster + drift
# ---------------------------------------------------------------------------


def test_cold_cluster_with_drift_hard_blocks_via_pool_api():
    """A scale-to-zero cluster has ZERO Node objects, so the live-Node heuristic is
    structurally blind. The node-pool API still sees the pools -> HARD BLOCK."""
    core = _CountingCoreApi(node_skus=_COLD)  # cold: no nodes at all
    lister = _CountingPoolLister(pool_skus=[None, "gcp_a100_80", "gcp_a100_80x8"])

    with pytest.raises(SandboxRuntimeError) as exc_info:
        enforce_gpu_pool_preflight(
            ["gcp_a100_80", "gcp_a100_80x2", "gcp_a100_80x4", "gcp_a100_80x8"],
            core_api=core,
            pool_lister=lister,
        )

    message = str(exc_info.value)
    assert exc_info.value.cause_kind == RuntimeCauseKind.backend_unavailable
    assert "gcp_a100_80x2" in message and "gcp_a100_80x4" in message
    # The verdict must be attributed to the authoritative tier...
    assert "authoritative" in message
    # ...and tier 2 must never even be consulted once tier 1 answers.
    assert lister.call_count == 1
    assert core.list_node_call_count == 0


def test_cold_cluster_all_pools_present_passes_without_touching_nodes():
    """Cold cluster, no drift: authoritative OK, and the heuristic is not queried."""
    core = _CountingCoreApi(node_skus=_COLD)
    lister = _CountingPoolLister(pool_skus=["gcp_a100_80", "gcp_a100_80x8"])

    result = enforce_gpu_pool_preflight(
        ["gcp_a100_80", "gcp_a100_80x8"], core_api=core, pool_lister=lister
    )

    assert result.status == "ok"
    assert result.source == "node_pool_api"
    assert lister.call_count == 1
    assert core.list_node_call_count == 0


# ---------------------------------------------------------------------------
# Tier-1 failure -> falls back to the tier-2 heuristic (never a hard block itself)
# ---------------------------------------------------------------------------


def test_pool_api_unreachable_falls_back_to_node_heuristic():
    """The API failing is not evidence of anything — fall through to live Nodes.
    Here the (warm) nodes DO show drift, so we still hard-block, via tier 2."""
    core = _CountingCoreApi(node_skus=["gcp_a100_80", "gcp_a100_80x8"])
    lister = _CountingPoolLister(raise_exc=ConnectionError("no route to host"))

    with pytest.raises(SandboxRuntimeError) as exc_info:
        enforce_gpu_pool_preflight(
            ["gcp_a100_80x2"], core_api=core, pool_lister=lister
        )

    assert "gcp_a100_80x2" in str(exc_info.value)
    assert lister.call_count == 1
    assert core.list_node_call_count == 1  # fell back to the heuristic


def test_pool_api_unreachable_and_no_drift_is_ok_via_nodes():
    core = _CountingCoreApi(node_skus=["gcp_a100_80"])
    lister = _CountingPoolLister(raise_exc=RuntimeError("transient 500"))

    result = enforce_gpu_pool_preflight(
        ["gcp_a100_80"], core_api=core, pool_lister=lister
    )

    assert result.status == "ok"
    assert result.source == "live_nodes"


def test_pool_api_unreachable_on_cold_cluster_is_cannot_verify_not_a_block():
    """Both tiers blind -> warn and proceed. Never turn an API blip into a blocker."""
    core = _CountingCoreApi(node_skus=_COLD)
    lister = _CountingPoolLister(raise_exc=ConnectionError("no route to host"))

    result = enforce_gpu_pool_preflight(
        ["gcp_a100_80", "gcp_a100_80x2"], core_api=core, pool_lister=lister
    )

    assert result.status == "cannot_verify"  # no raise
    assert result.source == "none"
    assert "node-pool API unavailable" in result.reason
    assert "scaled to zero" in result.reason


# ---------------------------------------------------------------------------
# Permission denied (the real in-cluster orchestrator GSA state today)
# ---------------------------------------------------------------------------


def test_permission_denied_plus_cold_cluster_is_cannot_verify_warn_only():
    """The orchestrator GSA has no container.clusters.get today -> 403. That must
    degrade to a warning and let the run proceed, never error it."""
    core = _CountingCoreApi(node_skus=_COLD)
    lister = _CountingPoolLister(
        raise_exc=_PermissionDenied("403 Permission 'container.clusters.get' denied")
    )

    result = enforce_gpu_pool_preflight(
        ["gcp_a100_80", "gcp_a100_80x2"], core_api=core, pool_lister=lister
    )

    assert result.status == "cannot_verify"  # warn only, run proceeds
    assert "container.clusters.get" in result.reason


def test_permission_denied_still_hard_blocks_when_nodes_corroborate_drift():
    """Losing tier 1 does not lose the guard entirely — the heuristic still bites on
    a warm cluster. (Strictly better than nothing while the IAM role is pending.)"""
    core = _CountingCoreApi(node_skus=["gcp_a100_80", "gcp_a100_80x8"])
    lister = _CountingPoolLister(raise_exc=_PermissionDenied("403 denied"))

    with pytest.raises(SandboxRuntimeError):
        enforce_gpu_pool_preflight(["gcp_a100_80x4"], core_api=core, pool_lister=lister)


# ---------------------------------------------------------------------------
# The corroboration invariant (blast-radius bound at BOTH tiers)
# ---------------------------------------------------------------------------


def test_pools_exist_but_none_labelled_is_cannot_verify_not_mass_absence():
    """Pools returned, but not one carries reprolab/sku (no GPU pool applied, wrong
    cluster, or a parse/shape surprise). Loud — but NOT corroborated, so it must not
    hard-block every GCP run. Falls through to the heuristic, then cannot_verify."""
    core = _CountingCoreApi(node_skus=_COLD)
    lister = _CountingPoolLister(pool_skus=[None, None])  # system pools only

    result = enforce_gpu_pool_preflight(
        ["gcp_a100_80", "gcp_a100_80x8"], core_api=core, pool_lister=lister
    )

    assert result.status == "cannot_verify"
    assert result.status != "verified_absent"
    assert "NOT ONE carries" in result.reason
    assert core.list_node_call_count == 1  # fell through to tier 2


def test_empty_node_observation_is_cannot_verify_not_verified_absent():
    """Tier 2 alone (no pool lister): an idle scale-to-zero cluster must not be read
    as 'every configured SKU is absent'."""
    core = _CountingCoreApi(node_skus=_COLD)
    result = check_gpu_pools_provisioned(
        ["gcp_a100_80", "gcp_a100_80x8"], core_api=core, pool_lister=None
    )
    assert result.status == "cannot_verify"
    assert result.status != "verified_absent"


# ---------------------------------------------------------------------------
# Tier-2-only behavior (unchanged: pool_lister defaults to None on check_*)
# ---------------------------------------------------------------------------


def test_all_configured_skus_observed_is_ok():
    core = _CountingCoreApi(node_skus=["gcp_a100_80", "gcp_a100_80x8"])
    result = check_gpu_pools_provisioned(["gcp_a100_80", "gcp_a100_80x8"], core_api=core)
    assert result.status == "ok"
    assert result.source == "live_nodes"
    assert result.missing_skus == ()
    assert core.list_node_call_count == 1


def test_drifted_sku_is_verified_absent_via_warm_nodes():
    core = _CountingCoreApi(node_skus=["gcp_a100_80", "gcp_a100_80x8"])
    result = check_gpu_pools_provisioned(
        ["gcp_a100_80", "gcp_a100_80x2", "gcp_a100_80x4", "gcp_a100_80x8"],
        core_api=core,
    )
    assert result.status == "verified_absent"
    assert result.source == "live_nodes"
    assert set(result.missing_skus) == {"gcp_a100_80x2", "gcp_a100_80x4"}


def test_node_query_exception_is_cannot_verify():
    core = _CountingCoreApi(raise_exc=RuntimeError("token expired"))
    result = check_gpu_pools_provisioned(["gcp_a100_80"], core_api=core)
    assert result.status == "cannot_verify"
    assert "token expired" in result.reason


def test_nodes_without_the_label_are_ignored_not_crashing():
    """Non-GPU (system-pool) nodes carry no reprolab/sku label at all."""
    core = _CountingCoreApi(node_skus=[None, None, "gcp_a100_80"])
    result = check_gpu_pools_provisioned(["gcp_a100_80"], core_api=core)
    assert result.status == "ok"


def test_no_configured_skus_never_queries_anything():
    core = _CountingCoreApi(node_skus=["gcp_a100_80"])
    lister = _CountingPoolLister(pool_skus=["gcp_a100_80"])
    result = check_gpu_pools_provisioned([], core_api=core, pool_lister=lister)
    assert result.status == "ok"
    assert core.list_node_call_count == 0
    assert lister.call_count == 0


def test_exactly_one_query_per_tier_regardless_of_configured_count():
    core = _CountingCoreApi(node_skus=["gcp_a100_80", "gcp_a100_80x8", "gcp_l4_24"])
    check_gpu_pools_provisioned(
        ["gcp_a100_80", "gcp_a100_80x8", "gcp_l4_24", "gcp_h100_80"], core_api=core
    )
    assert core.list_node_call_count == 1


# ---------------------------------------------------------------------------
# Shape normalization: typed container_v1 object AND raw REST dict
# ---------------------------------------------------------------------------


def test_typed_container_v1_pool_shape_is_parsed():
    """google.cloud.container_v1 returns objects with .name / .config.labels."""
    typed_pools = [
        SimpleNamespace(
            name="repro-gcp-a100-80",
            config=SimpleNamespace(labels={"reprolab/sku": "gcp_a100_80"}),
        ),
        SimpleNamespace(name="default-pool", config=SimpleNamespace(labels={})),
    ]
    result = check_gpu_pools_provisioned(
        ["gcp_a100_80"],
        core_api=_CountingCoreApi(node_skus=_COLD),
        pool_lister=lambda: typed_pools,
    )
    assert result.status == "ok"
    assert result.source == "node_pool_api"


def test_rest_json_pool_shape_is_parsed():
    """The REST transport returns {"config": {"labels": {...}}} dicts."""
    rest_pools = [
        {"name": "repro-gcp-a100-80x8", "config": {"labels": {"reprolab/sku": "gcp_a100_80x8"}}},
        {"name": "default-pool", "config": {}},
    ]
    result = check_gpu_pools_provisioned(
        ["gcp_a100_80x8"],
        core_api=_CountingCoreApi(node_skus=_COLD),
        pool_lister=lambda: rest_pools,
    )
    assert result.status == "ok"
    assert result.source == "node_pool_api"


def test_dict_shaped_nodes_are_accepted():
    """A plain-dict CoreV1Api double must also work (dict.items() bound-method trap)."""

    class _DictCoreApi:
        def list_node(self) -> dict:
            return {
                "items": [
                    {"metadata": {"labels": {"reprolab/sku": "gcp_a100_80"}}},
                    {"metadata": {"labels": {}}},
                ]
            }

    result = check_gpu_pools_provisioned(["gcp_a100_80"], core_api=_DictCoreApi())
    assert result.status == "ok"


# ---------------------------------------------------------------------------
# make_gke_pool_lister — tier 1 is only addressable with project+region+cluster
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "project,region,cluster",
    [
        ("", "us-central1", "repro-gke"),   # no project
        ("proj", "", "repro-gke"),          # no region
        ("proj", "us-central1", ""),        # no cluster name (the shipped default)
        ("", "", ""),
    ],
)
def test_pool_lister_is_none_when_cluster_cannot_be_addressed(project, region, cluster):
    settings = SimpleNamespace(
        gcp_project=project, gcp_region=region, gcp_gke_cluster=cluster
    )
    assert make_gke_pool_lister(settings) is None


def test_pool_lister_is_built_when_fully_addressed(monkeypatch):
    import backend.services.runtime.gpu_pool_preflight as mod

    seen: dict[str, str] = {}

    def _fake_list(*, project: str, location: str, cluster: str):
        seen.update(project=project, location=location, cluster=cluster)
        return [{"name": "p", "config": {"labels": {"reprolab/sku": "gcp_a100_80"}}}]

    monkeypatch.setattr(mod, "list_gke_node_pools", _fake_list)
    settings = SimpleNamespace(
        gcp_project="my-proj", gcp_region="us-central1", gcp_gke_cluster="repro-gke"
    )
    lister = make_gke_pool_lister(settings)
    assert lister is not None
    # The lister passes the transport's payload straight through; check_* normalizes
    # either shape (typed object or REST dict), so no normalization happens here.
    assert lister() == [
        {"name": "p", "config": {"labels": {"reprolab/sku": "gcp_a100_80"}}}
    ]
    # The point of this test: the cluster is addressed by project + region + name —
    # no wildcard guessing, no second source of truth for the location.
    assert seen == {
        "project": "my-proj",
        "location": "us-central1",
        "cluster": "repro-gke",
    }


def test_enforce_auto_builds_lister_from_injected_settings(monkeypatch):
    """enforce(settings=...) must build tier 1 from THOSE settings (not a global)."""
    import backend.services.runtime.gpu_pool_preflight as mod

    monkeypatch.setattr(
        mod,
        "list_gke_node_pools",
        lambda **kw: [
            {"name": "p1", "config": {"labels": {"reprolab/sku": "gcp_a100_80"}}}
        ],
    )
    settings = SimpleNamespace(
        gcp_project="p", gcp_region="us-central1", gcp_gke_cluster="c"
    )
    core = _CountingCoreApi(node_skus=_COLD)  # cold — only tier 1 can answer

    with pytest.raises(SandboxRuntimeError) as exc_info:
        enforce_gpu_pool_preflight(
            ["gcp_a100_80", "gcp_a100_80x2"], core_api=core, settings=settings
        )

    assert "gcp_a100_80x2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# enforce_* policy surface
# ---------------------------------------------------------------------------


def test_enforce_raises_actionable_error_naming_the_missing_sku():
    core = _CountingCoreApi(node_skus=["gcp_a100_80", "gcp_a100_80x8"])
    with pytest.raises(SandboxRuntimeError) as exc_info:
        enforce_gpu_pool_preflight(
            ["gcp_a100_80", "gcp_a100_80x2"],
            core_api=core,
            pool_lister=None,  # tier 1 off
            provider_label="GCP",
            settings_var_name="OPENRESEARCH_GCP_GPU_SKUS",
        )
    err = exc_info.value
    assert err.cause_kind == RuntimeCauseKind.backend_unavailable
    message = str(err)
    assert "gcp_a100_80x2" in message                 # names the exact drifting SKU
    assert "OPENRESEARCH_GCP_GPU_SKUS" in message     # remediation (2): drop it
    assert "Terraform" in message                     # remediation (1): provision it
    assert "capacity_exhausted" in message            # the cost of NOT fixing it


def test_enforce_passes_silently_when_all_provisioned():
    core = _CountingCoreApi(node_skus=["gcp_a100_80", "gcp_a100_80x8"])
    result = enforce_gpu_pool_preflight(
        ["gcp_a100_80", "gcp_a100_80x8"], core_api=core, pool_lister=None
    )
    assert result.status == "ok"


def test_enforce_does_not_raise_when_cluster_unreachable():
    core = _CountingCoreApi(raise_exc=ConnectionError("no route to host"))
    result = enforce_gpu_pool_preflight(
        ["gcp_a100_80"], core_api=core, pool_lister=None
    )
    assert result.status == "cannot_verify"  # no raise


def test_enforce_queries_cluster_exactly_once():
    core = _CountingCoreApi(node_skus=["gcp_a100_80"])
    enforce_gpu_pool_preflight(["gcp_a100_80"], core_api=core, pool_lister=None)
    assert core.list_node_call_count == 1


def test_result_is_frozen_dataclass():
    result = GpuPoolPreflightResult(status="ok")
    with pytest.raises(Exception):
        result.status = "verified_absent"  # type: ignore[misc]
