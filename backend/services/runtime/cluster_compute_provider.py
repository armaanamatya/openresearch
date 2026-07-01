"""Cluster-topology ComputeProvider wrapping the existing GKE/AKS Job backends
(Phase 1c, Unit E).

The cluster path has NO VM bracket: the orchestrator already runs in-cluster
(no SSH), and GPU is acquired per-cell-Job by the K8s autoscaler rather than
via a VM-style "arm a GPU machine type" step. So most of the whole-run
lifecycle ``ComputeProvider`` (Unit B) encodes for a VM -- ``acquire_gpu``,
``release_gpu``, ``teardown`` -- collapses to a no-op here; the real per-cell
Job dispatch, streaming, and OOM handling already live on the existing Job
path (``gpu_cell_runner.py`` / ``cell_scheduler.py`` / ``run_experiment``)
and are NOT re-implemented by this provider in Phase 1c.

The one thing THIS unit actually delivers is **GCP<->Azure failover** behind
the ``ComputeProvider`` shape: ``provision_cpu`` is where a
``ClusterComputeProvider`` resolves which cloud's Job backend
(``GkeJobBackend``/``AksJobBackend``) to use, by walking ``preference``
through the shared Unit-A selector
(:func:`~backend.services.runtime.cloud_failover.select_backend_with_failover`)
-- never reimplemented here. ``preflight`` runs the same walk in probe mode
(no caching, no lease) so a caller can check reachability before committing
to a lease.

Hermetic by construction: ``availability``/``backend_factory`` are threaded
straight through to ``select_backend_with_failover`` unchanged (``None``
means "use its real, lazily-imported GKE/AKS checks" -- this module never
imports a cloud SDK itself). Every test in this module injects fakes for
both, so nothing here ever touches ``kubectl``/a real cluster.

Honest stubs, not yet wired (documented per-method below rather than left to
silently inherited defaults): ``launch`` records a handle but does not
dispatch a real cell Job (the existing Job path already does that once a
backend is resolved); ``watch`` yields a single terminal status rather than
re-implementing the existing Job-path streaming; ``collect``/``recover`` do
not perform real object-store I/O yet -- Phase 1c's job is the provider
*shape* + failover, not full cluster-run integration.
"""

from __future__ import annotations

from typing import Callable, Iterator

from backend.services.runtime.cloud_failover import (
    BackendSelection,
    select_backend_with_failover,
)
from backend.services.runtime.compute_provider import (
    CapacityReport,
    ComputeLease,
    ComputeProvider,
    ReportBundle,
    RunHandle,
    RunStatus,
)
from backend.services.runtime.interface import SandboxRuntimeError


class ClusterComputeProvider(ComputeProvider):
    """``ComputeProvider`` over the existing in-cluster GKE/AKS Job backends.

    No SSH, no VM: the orchestrator already runs in-cluster, so this provider
    is a thin adapter that (a) picks a cloud via GCP<->Azure failover and
    (b) satisfies the ``ComputeProvider`` contract with honest no-ops/stubs
    everywhere a VM-shaped step (arm GPU, stop, delete) has no cluster
    analogue. See the module docstring for what is, and is not, wired yet.
    """

    def __init__(
        self,
        preference: "list[str]",
        *,
        availability: "dict[str, Callable[[], None]] | None" = None,
        backend_factory: "Callable[..., object] | None" = None,
        run_budget: object = None,
        gpu_plan: object = None,
    ) -> None:
        super().__init__()
        self._preference = list(preference)
        self._availability = availability
        self._backend_factory = backend_factory
        self._run_budget = run_budget
        self._gpu_plan = gpu_plan
        self._backend: object | None = None  # cached winner from the last provision_cpu

    def _select(self) -> BackendSelection:
        """Run one failover walk through the shared Unit-A selector.

        Never reimplements the try-each-cloud loop; every kwarg here is
        exactly what was passed to ``__init__`` (``None`` defers to
        ``select_backend_with_failover``'s own real-path defaults).
        """
        return select_backend_with_failover(
            self._preference,
            run_budget=self._run_budget,
            gpu_plan=self._gpu_plan,
            availability=self._availability,
            backend_factory=self._backend_factory,
        )

    def preflight(self, plan) -> CapacityReport:
        """Billing-free reachability probe.

        ``plan`` is accepted for ``ComputeProvider`` parity but not consulted
        here: unlike the VM path, K8s autoscales GPU nodes per-Job, so there
        is no pre-sized quota tied to ``plan`` to check up front -- the only
        question worth asking before committing is "is any cloud in
        ``preference`` reachable at all," which is exactly what
        ``select_backend_with_failover`` answers. This probe does NOT cache
        its result; ``provision_cpu`` (below) is where failover "happens" for
        the lease a caller actually uses, and it always re-walks
        ``preference`` from scratch.
        """
        try:
            selection = self._select()
        except SandboxRuntimeError as exc:
            return CapacityReport(available=False, reason=str(exc))
        return CapacityReport(available=True, reason=selection.cloud)

    def provision_cpu(self, plan) -> ComputeLease:
        """Resolve + cache the winning cloud's Job backend.

        This is where failover actually happens for the lease a caller goes
        on to use. No GPU billing starts here: constructing a
        ``GkeJobBackend``/``AksJobBackend`` wrapper touches no cloud resource
        by itself, and the cluster has no separate "cheap CPU tier" distinct
        from the eventual GPU tier the way a VM does -- GPU nodes autoscale
        per-Job only once a cell Job is actually submitted (on the existing
        Job path, not here). Propagates ``SandboxRuntimeError`` uncaught when
        every cloud in ``preference`` is down; a caller that already ran
        ``preflight`` and saw ``available=False`` should not call this.
        """
        selection = self._select()
        self._backend = selection.backend
        return ComputeLease(cloud=selection.cloud, cpu=selection.backend, ref=selection.cloud)

    def acquire_gpu(self, lease: ComputeLease) -> ComputeLease:
        """No-op: returns the SAME lease, unchanged.

        The cluster topology has no VM GPU bracket to arm -- GPU is acquired
        per-cell-Job by the K8s autoscaler when a Job is actually submitted
        on the existing Job path, not by a discrete "arm the GPU" step a
        provider performs up front. Object identity (not just equality) is
        preserved deliberately so a caller can tell no new lease was minted.
        """
        return lease

    def launch(self, lease: ComputeLease, run_spec) -> RunHandle:
        """Record a handle for the lease; does NOT dispatch a real cell Job.

        The existing ``run_experiment`` / ``gpu_cell_runner`` /
        ``cell_scheduler`` path already submits Jobs against whichever
        backend ``provision_cpu`` resolved -- Phase 1c does not duplicate
        that dispatch here. This stub exists so ``ReproductionRun`` (Unit C)
        can drive a cluster-backed run through the SAME state machine as a
        VM-backed one without special-casing the cluster provider; wiring a
        real dispatch call through ``launch`` is later work.
        """
        return RunHandle(id=lease.ref or lease.cloud, lease=lease)

    def watch(self, handle: RunHandle) -> Iterator[RunStatus]:
        """Yield a single terminal status.

        Full cluster-run status streaming (the SSE event stream / Job
        polling) already exists on the Job path and is not duplicated here
        -- a caller that needs live progress should watch that path
        directly. This stub only satisfies the ``ComputeProvider`` contract
        so the state machine driving it does not need a cluster special
        case.
        """
        yield RunStatus(state="terminal", synced=True)

    def collect(self, handle: RunHandle) -> ReportBundle:
        """Honest not-yet-wired stub.

        Real report retrieval is object-store (GCS/Azure blob) I/O, which
        this unit deliberately does not add (Phase 1c's hard constraint is
        no real cloud SDK calls in this provider -- see the module
        docstring). Returns ``ok=False`` naming the gap rather than
        fabricating a bundle; a later phase wires this to the existing
        blob-pull helpers (``gcs_blob.py`` / ``azure_blob.py``).
        """
        return ReportBundle(ok=False, reason="cluster report collection is not wired in Phase 1c")

    def release_gpu(self, lease: ComputeLease) -> None:
        """No-op: a K8s Job releases its GPU node claim on completion; there
        is no separate VM-style "stop" call on the cluster path.
        """
        return None

    def teardown(self, lease: ComputeLease, *, reason: str) -> None:
        """No-op: Jobs self-clean via ``ttl_seconds`` (the existing
        ``k8s_job_backend`` TTL-after-finished mechanism) -- there is no VM
        to stop/delete here.
        """
        return None

    def recover(self, lease_ref: str) -> "ReportBundle | None":
        """Best-effort re-pull after an orchestrator crash.

        Phase 1c does not implement the object-store re-pull yet (see
        ``collect``) -- returns ``None`` (unrecoverable), the same as the
        ``ComputeProvider`` default. Overridden here (rather than left to
        the inherited default silently) because the eventual design intent
        IS "re-pull the report from the object store"; documenting that
        intent next to the stub makes the gap discoverable.
        """
        return None


__all__ = ["ClusterComputeProvider"]
