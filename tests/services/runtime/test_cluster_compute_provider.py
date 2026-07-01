"""Unit tests for ClusterComputeProvider (Phase 1c, Unit E).

Hermetic: ``availability``/``backend_factory`` are always injected fakes (the
same injection points Unit A's ``select_backend_with_failover`` tests use),
so no real GKE/AKS/kubectl call is ever made. Confirms the cluster-topology
contract: failover happens at ``provision_cpu`` (through the shared Unit-A
selector, never reimplemented here), ``preflight`` degrades to
``available=False`` rather than raising when every cloud is down, and
``acquire_gpu`` is a true no-op (returns the SAME lease object) because the
cluster path has no VM GPU bracket -- GPU is acquired per-cell-Job by the K8s
autoscaler.
"""

from backend.services.runtime.cluster_compute_provider import ClusterComputeProvider
from backend.services.runtime.compute_provider import ComputeLease
from backend.services.runtime.interface import RuntimeCauseKind, SandboxRuntimeError
from backend.services.runtime.run_plan import RunPlan


def _down():
    def _r():
        raise SandboxRuntimeError(RuntimeCauseKind.backend_unavailable, "down")

    return _r


def _run_plan() -> RunPlan:
    # plan content is not consulted by ClusterComputeProvider in Phase 1c
    # (see cluster_compute_provider.py's preflight/provision_cpu docstrings)
    # -- an empty plan is enough to exercise the contract.
    return RunPlan()


def test_provision_fails_over_gcp_to_azure():
    prov = ClusterComputeProvider(
        preference=["gcp", "azure"],
        availability={"gcp": _down(), "azure": lambda: None},
        backend_factory=lambda cloud, **_: f"backend:{cloud}",
    )
    lease = prov.provision_cpu(_run_plan())
    assert isinstance(lease, ComputeLease) and lease.cloud == "azure"


def test_preflight_reports_unavailable_when_all_down():
    prov = ClusterComputeProvider(
        preference=["gcp", "azure"],
        availability={"gcp": _down(), "azure": _down()},
        backend_factory=lambda cloud, **_: f"backend:{cloud}",
    )
    assert prov.preflight(_run_plan()).available is False


def test_acquire_gpu_is_noop_no_vm_bracket():
    prov = ClusterComputeProvider(
        preference=["gcp"],
        availability={"gcp": lambda: None},
        backend_factory=lambda cloud, **_: f"backend:{cloud}",
    )
    lease = prov.provision_cpu(_run_plan())
    assert prov.acquire_gpu(lease) is lease  # no VM GPU bracket on the cluster path
