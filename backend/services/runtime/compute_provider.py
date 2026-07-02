"""Cloud-agnostic whole-run compute lifecycle contract (Phase 1c, Unit B).

``ComputeProvider`` is the single abstraction a :class:`ReproductionRun` state
machine (Phase 1c, Unit C) drives through the whole-run lifecycle -- from a
billing-free capacity probe, through a cheap CPU-only staging tier, to a
GPU-armed training run, to artifact collection and teardown -- regardless of
whether the concrete backend is a single GCP VM (``VmComputeProvider``, Unit
D) or an existing GKE/AKS Job cluster (``ClusterComputeProvider``, Unit E).

Two invariants this contract exists to encode (enforced by the state machine
that drives it, not by this module):

1. ``acquire_gpu`` is the ONLY method that starts GPU billing. Everything
   before it (``preflight``, ``provision_cpu``, ``stage``) is cheap-tier or
   free, so a paper that fails triage/feasibility never touches a GPU.
2. A lease survives a crash via ``ComputeLease.ref`` -- a stable id
   ``recover()`` can use to re-attach to (or salvage artifacts from) a run
   whose orchestrator process died mid-flight, without re-provisioning.

Pure interfaces only: this module has no side effects, no I/O, and no
knowledge of gcloud/kubectl/any cloud SDK. All dataclasses are frozen value
objects; ``ComputeProvider`` is an ``abc.ABC`` with two concrete "safe
default" hooks (``stage`` = no-op, ``recover`` = unrecoverable) and the rest
abstract, so every concrete provider must implement the billing-relevant tier
ops explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class ComputeLease:
    """An opaque handle to provisioned compute, cheap-tier through GPU-armed.

    ``cpu``/``disk``/``gpu`` are intentionally untyped (``object | None``): a
    VM-based provider stores instance/disk names there, a cluster provider
    stores a resolved backend/context. ``ref`` is the stable identifier
    ``recover()`` uses to re-attach after an orchestrator crash -- it must
    survive the orchestrator process dying, so it is never derived from
    in-memory-only state.
    """

    cloud: str = ""  # "gcp" | "azure"
    cpu: object | None = None  # opaque handle (VM name / cluster ctx)
    disk: object | None = None
    gpu: object | None = None
    ref: str = ""  # stable id for recover() after a crash
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CapacityReport:
    """Result of a billing-free ``preflight`` capacity/quota probe."""

    available: bool
    reason: str = ""
    est_wait_s: float | None = None


@dataclass(frozen=True)
class RunHandle:
    """A launched run's identity, returned by ``launch`` and consumed by
    ``watch``/``collect``."""

    id: str
    lease: ComputeLease
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RunStatus:
    """One ``watch`` poll result.

    ``state`` is one of ``"running"`` | ``"terminal"`` | ``"stalled"`` |
    ``"stopped_uncollected"`` (the last meaning the compute was reclaimed --
    by a run-duration ceiling, a watchdog, or an external actor -- before
    ``collect`` ran; the driving state machine reacts by calling
    ``recover()``). ``synced`` records whether artifacts were off-box as of
    THIS poll, so an emergency stop can degrade to "last synced" instead of
    "stranded."
    """

    state: str  # "running" | "terminal" | "stalled" | "stopped_uncollected"
    detail: str = ""
    synced: bool = False  # True once artifacts are off-box for this poll


@dataclass(frozen=True)
class ReportBundle:
    """The terminal artifact of a run: a report path plus any side artifacts."""

    ok: bool
    report_path: str | None = None
    artifacts: dict = field(default_factory=dict)
    reason: str = ""


class ComputeProvider(ABC):
    """Whole-run compute lifecycle: preflight -> cheap CPU tier -> GPU-armed
    run -> collect -> teardown.

    Concrete providers (``VmComputeProvider``, ``ClusterComputeProvider``)
    must implement every method below except ``stage``/``recover``, which
    have safe defaults (no-op / unrecoverable) because not every backend has
    a meaningful "ship a bundle" or "re-attach after crash" story.
    """

    @abstractmethod
    def preflight(self, plan) -> CapacityReport:
        """Quota/capacity probe. NO lease is created, NO billing starts."""
        ...

    @abstractmethod
    def provision_cpu(self, plan) -> ComputeLease:
        """Acquire the cheap CPU tier (plus any cache disk). NO gpu."""
        ...

    def stage(self, lease: ComputeLease, bundle, run_spec) -> None:
        """Ship code + run spec onto the leased compute. Default: no-op."""
        return None

    @abstractmethod
    def acquire_gpu(self, lease: ComputeLease) -> ComputeLease:
        """Arm the GPU-billing ceiling. This is where billing starts."""
        ...

    @abstractmethod
    def launch(self, lease: ComputeLease, run_spec) -> RunHandle:
        """Start the run and return a handle for ``watch``/``collect``."""
        ...

    @abstractmethod
    def watch(self, handle: RunHandle) -> Iterator[RunStatus]:
        """Stream run status, periodically syncing artifacts off-box."""
        ...

    @abstractmethod
    def collect(self, handle: RunHandle) -> ReportBundle:
        """Pull the final report + artifacts."""
        ...

    @abstractmethod
    def release_gpu(self, lease: ComputeLease) -> None:
        """Stop GPU billing (release/stop, not necessarily destroy)."""
        ...

    @abstractmethod
    def teardown(self, lease: ComputeLease, *, reason: str) -> None:
        """Fully release the lease (delete/cleanup)."""
        ...

    def recover(self, lease_ref: str) -> "ReportBundle | None":
        """Best-effort re-attach after an orchestrator crash.

        Default: unrecoverable (``None``). Concrete providers that can
        salvage artifacts from a surviving disk/bucket override this.
        """
        return None


__all__ = [
    "CapacityReport",
    "ComputeLease",
    "ComputeProvider",
    "ReportBundle",
    "RunHandle",
    "RunStatus",
]
