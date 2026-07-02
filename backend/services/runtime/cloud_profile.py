"""Neutral per-cloud configuration profile (Phase 1c, Unit B).

``CloudProfile`` is the seam a :class:`~backend.services.runtime.compute_provider.ComputeProvider`
reads to find its cloud-specific configuration, WITHOUT this module (or any
Unit-B consumer) needing to import a specific cloud SDK or the existing
Kubernetes job-backend module. It deliberately WRAPS -- never extends -- the
existing ``backend.services.runtime.k8s_job_backend.CloudSpec`` (the
descriptor that already parameterises ``_KubernetesJobBackend`` for
GCP/Azure): a ``ClusterComputeProvider`` (Unit E) reads ``profile.k8s`` and
hands it straight to the existing ``GkeJobBackend``/``AksJobBackend``
machinery unchanged, while a ``VmComputeProvider`` (Unit D) reads
``profile.vm`` (a new, VM-only shape -- the cluster path has no VM bracket at
all).

``k8s`` is typed as ``object | None`` rather than ``CloudSpec | None`` so this
module never imports ``k8s_job_backend`` (or transitively the ``kubernetes``
SDK) at module scope -- a real ``CloudSpec`` instance still satisfies the slot
at runtime, proven by ``test_cloud_profile.py`` importing the real
``gke_job_backend._GCP_CLOUD``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VmSpec:
    """Pure configuration for the single-VM compute tier.

    Command construction (the actual ``gcloud``/``ssh``/``scp`` argv) lives in
    the concrete provider (``VmComputeProvider``, Phase 1c Unit D) -- this is
    config only, no behavior.
    """

    zone: str = ""
    cpu_machine_type: str = ""
    gpu_machine_type: str = ""
    accelerator_type: str = ""
    accelerator_count: int = 0
    image: str = ""  # or machine_image
    machine_image: str = ""
    cache_disk_name: str = ""
    tiering_strategy: str = "stage_on_gpu"  # | "machine_type_flip" | "cpu_warm_disk_then_gpu_attach"
    max_run_duration_s: int | None = None  # control-plane cost ceiling (gcloud max-run-duration=STOP)
    capacity_signatures: tuple[str, ...] = ()  # stderr substrings meaning STOCKOUT / ZONE_RESOURCE_POOL_EXHAUSTED


@dataclass(frozen=True)
class CloudProfile:
    """Neutral per-cloud config: at most one of ``k8s``/``vm`` is populated,
    depending on which ``ComputeProvider`` consumes it."""

    cloud: str  # "gcp" | "azure"
    k8s: object | None = None  # the EXISTING k8s_job_backend.CloudSpec, unchanged (typed loosely to avoid import coupling)
    vm: VmSpec | None = None


__all__ = ["CloudProfile", "VmSpec"]
