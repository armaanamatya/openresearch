"""WS3 durable-controller cluster seam + takeover-safe submit orchestration.

The correctness-critical ordering lives here as pure logic over an injected
``ControllerCluster`` (duck-typed), so every race and failure mode is unit
testable with a fake — no GCS/K8s/network. ``live_runs`` builds the manifest
and does the file-backed I/O; this module owns *when* to acquire, submit, wait,
reap, and when it is safe to fall back.

Design: ``docs/history/specs/2026-07-12-cloud-native-durable-and-cpu-lane-design.md`` §3.2, §7.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Protocol
from uuid import uuid4

logger = logging.getLogger(__name__)

__all__ = [
    "ControllerCluster",
    "ControllerNotReady",
    "ControllerStuck",
    "submit_controller",
    "sweep_orphaned_controllers",
    "K8sControllerCluster",
    "GcsK8sControllerCluster",
    "AzureK8sControllerCluster",
    "build_default_cluster",
    "build_default_lease",
]


class ControllerNotReady(Exception):
    """The controller Job was confirmed absent after a launch failure.

    A caller may retry the remote launch. Durable mode must not silently move
    execution back to the initiating laptop.
    """


class ControllerStuck(Exception):
    """The submitted controller Job never became ready AND could not be
    confirmed deleted. A remote controller may still be live, so the caller MUST
    NOT fall back to a local run (local + remote on one run dir = split-brain).
    Fail loud instead."""


class ControllerCluster(Protocol):
    """The injected I/O surface. ``acquire``/``is_current`` are lease ops
    (``BlobLease``-shaped); ``submit``/``wait_ready``/``delete_confirmed``/``reap``
    are K8s ops. The token carries ``fence_epoch``/``acquired_epoch``."""

    def now(self) -> float: ...
    def acquire(self, project_id: str, owner_id: str, now_epoch: float) -> Any: ...
    def is_current(self, token: Any) -> bool: ...
    def submit(self, manifest: dict) -> None: ...
    def wait_ready(self, job_name: str, *, timeout_s: float) -> bool: ...
    def delete_confirmed(self, job_name: str) -> bool: ...
    def reap(self, project_id: str, token: Any) -> int: ...


def submit_controller(
    cluster: ControllerCluster,
    *,
    build_manifest: Callable[[int], dict],
    project_id: str,
    owner_id: str,
    ready_timeout_s: float,
) -> dict | None:
    """Takeover-safe durable-controller submit.

    Ordering (Codex-hardened): acquire → is_current → build+submit → wait_ready
    → (is_current) reap → return handle. The manifest is built only AFTER acquire,
    from the lease's stable ``fence_epoch`` (``build_manifest`` is a factory of it),
    so Jobs never carry a fence the lease didn't grant. Reap runs only after the
    successor is ready, so a failed submit never deletes the predecessor's work.

    Returns a handle dict ``{jobName, fenceEpoch, submittedEpoch}`` on success, or
    ``None`` when the run is already owned by another driver (adopt — never submit).
    Raises :class:`ControllerNotReady` (safe to retry remotely) or
    :class:`ControllerStuck` (remote liveness is ambiguous). Acquire errors
    propagate; submit errors are confirmed absent or promoted to ``Stuck``.
    """
    token = cluster.acquire(project_id, owner_id, cluster.now())
    if token is None or not cluster.is_current(token):
        # Contended: another driver legitimately owns this run (or we were
        # superseded between acquire and use). Adopt — never submit a second
        # controller, never fall back to a local run.
        return None

    fence_epoch = int(token.fence_epoch)
    manifest = build_manifest(fence_epoch)
    job_name = manifest["metadata"]["name"]

    try:
        cluster.submit(manifest)
    except Exception as exc:
        # A create timeout is ambiguous: the API server may have persisted the
        # Job before the client lost the response. Confirm it is absent before
        # reporting a retryable launch failure; otherwise fail closed.
        if cluster.delete_confirmed(job_name):
            raise ControllerNotReady(job_name) from exc
        raise ControllerStuck(job_name) from exc

    if not cluster.wait_ready(job_name, timeout_s=ready_timeout_s):
        if cluster.delete_confirmed(job_name):
            raise ControllerNotReady(job_name)
        raise ControllerStuck(job_name)

    # Reap the predecessor's older-fence Jobs only now that the successor is up
    # and we still hold the lease.
    if cluster.is_current(token):
        cluster.reap(project_id, token)

    return {
        "jobName": job_name,
        "fenceEpoch": fence_epoch,
        "submittedEpoch": float(token.acquired_epoch),
    }


def sweep_orphaned_controllers(
    cluster: ControllerCluster,
    *,
    list_durable_runs: Callable[[], list[str]],
    build_manifest_for: Callable[[str, str], Callable[[int], dict]],
    ready_timeout_s: float,
    sweeper_owner: str = "sweeper",
) -> list[str]:
    """One deterministic sweep pass: resubmit controllers whose lease has lapsed.

    Split-brain safe by construction: each run gets a fresh owner derived from
    ``sweeper_owner`` and that exact owner is passed to ``build_manifest_for``
    for the replacement pod. ``acquire`` succeeds only when the lease is free
    or TTL-expired, so a healthy heartbeat is never taken over. A takeover bumps
    the fence and the successor reaps the dead predecessor's Jobs. Per-run
    fail-soft; returns the project_ids resubmitted.

    K8s ``backoffLimit`` remains the primary self-healing for controller Pod
    crashes; this covers the rarer Job-deleted / backoff-exhausted case.
    """
    resubmitted: list[str] = []
    for project_id in list_durable_runs():
        owner_id = f"{sweeper_owner}-{uuid4().hex}"
        try:
            handle = submit_controller(
                cluster,
                build_manifest=build_manifest_for(project_id, owner_id),
                project_id=project_id,
                owner_id=owner_id,
                ready_timeout_s=ready_timeout_s,
            )
        except (ControllerNotReady, ControllerStuck) as exc:
            logger.warning("sweep: controller resubmit for %s failed: %s", project_id, exc)
            continue
        except Exception as exc:  # cluster I/O error — never abort the whole sweep
            logger.warning("sweep: controller resubmit for %s errored: %s", project_id, exc)
            continue
        if handle is not None:
            resubmitted.append(project_id)
    return resubmitted


class K8sControllerCluster:
    """Cloud-neutral lease + Kubernetes implementation of ``ControllerCluster``.

    Every method fails fast on a missing SDK/config/API error; durable callers
    propagate the failure and never move execution back to the laptop. The
    Kubernetes adapter and the pure submit/sweep ordering are hermetically
    exercised with injected API and lease doubles.
    """

    def __init__(
        self,
        *,
        lease: Any,
        namespace: str,
        batch_api: Any | None = None,
        core_api: Any | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._lease = lease
        self._namespace = namespace
        self._batch_api = batch_api
        self._core_api = core_api
        self._clock = clock
        self._sleep = sleep

    def now(self) -> float:
        return self._clock()

    def acquire(self, project_id: str, owner_id: str, now_epoch: float) -> Any:
        return self._lease.acquire(project_id, owner_id, now_epoch)

    def is_current(self, token: Any) -> bool:
        return self._lease.is_current(token)

    def renew(self, token: Any, now_epoch: float) -> Any:
        return self._lease.renew(token, now_epoch)

    def _batch(self) -> Any:
        if self._batch_api is not None:
            return self._batch_api
        from backend.services.runtime import k8s_job_backend as kb

        self._batch_api = kb._load_kubernetes_batch_api()
        return self._batch_api

    def _core(self) -> Any:
        if self._core_api is not None:
            return self._core_api
        from backend.services.runtime import k8s_job_backend as kb

        self._core_api = kb._load_kubernetes_core_api()
        return self._core_api

    def submit(self, manifest: dict) -> None:
        self._batch().create_namespaced_job(namespace=self._namespace, body=manifest)

    def wait_ready(self, job_name: str, *, timeout_s: float) -> bool:
        batch = self._batch()
        core = self._core()
        deadline = self._clock() + timeout_s
        while self._clock() < deadline:
            job = batch.read_namespaced_job_status(name=job_name, namespace=self._namespace)
            status = getattr(job, "status", None)
            if status is not None:
                if (getattr(status, "succeeded", 0) or 0) >= 1:
                    return True  # already ran to completion — treat as ready
                terminal_failure = any(
                    getattr(condition, "type", "") == "Failed"
                    and str(getattr(condition, "status", "")).lower() == "true"
                    for condition in (getattr(status, "conditions", None) or [])
                )
                if terminal_failure:
                    return False
            pods = core.list_namespaced_pod(
                namespace=self._namespace,
                label_selector=f"job-name={job_name}",
            )
            for pod in getattr(pods, "items", []) or []:
                phase = str(getattr(getattr(pod, "status", None), "phase", ""))
                if phase in {"Running", "Succeeded"}:
                    return True
            self._sleep(2.0)
        return False

    def delete_confirmed(self, job_name: str) -> bool:
        from kubernetes.client.rest import ApiException  # type: ignore[import]

        batch = self._batch()
        try:
            batch.delete_namespaced_job(
                name=job_name, namespace=self._namespace, propagation_policy="Background"
            )
        except ApiException as exc:
            return exc.status == 404  # already gone ⇒ confirmed; else unconfirmed
        for _ in range(10):
            try:
                batch.read_namespaced_job(name=job_name, namespace=self._namespace)
            except ApiException as exc:
                if exc.status == 404:
                    return True
            self._sleep(1.0)
        return False

    def reap(self, project_id: str, token: Any) -> int:
        batch = self._batch()

        def _list(run_id: str) -> list[tuple[str, int]]:
            jobs = batch.list_namespaced_job(
                namespace=self._namespace, label_selector=f"reprolab/project={run_id}"
            )
            out: list[tuple[str, int]] = []
            for job in getattr(jobs, "items", []) or []:
                labels = getattr(job.metadata, "labels", None) or {}
                raw = labels.get("reprolab-generation") or labels.get(
                    "reprolab/fence-epoch"
                )
                if raw is None:
                    continue
                try:
                    out.append((job.metadata.name, int(raw)))
                except (TypeError, ValueError):
                    continue
            return out

        def _delete(name: str) -> None:
            batch.delete_namespaced_job(
                name=name, namespace=self._namespace, propagation_policy="Background"
            )

        return self._lease.reap_stale_fence_epochs(
            project_id, token, list_jobs=_list, delete_job=_delete
        )


class GcsK8sControllerCluster(K8sControllerCluster):
    """GCS generation-CAS lease plus the shared Kubernetes Job lifecycle."""

    def __init__(
        self,
        *,
        bucket: str,
        project: str | None,
        namespace: str,
        batch_api: Any | None = None,
        core_api: Any | None = None,
    ) -> None:
        from backend.services.runtime.blob_lease import BlobLease

        super().__init__(
            lease=BlobLease(bucket=bucket, project=project),
            namespace=namespace,
            batch_api=batch_api,
            core_api=core_api,
        )


class AzureK8sControllerCluster(K8sControllerCluster):
    """Azure Blob ETag-CAS lease plus the shared Kubernetes Job lifecycle."""

    def __init__(
        self,
        *,
        account_name: str,
        container_name: str,
        namespace: str,
        batch_api: Any | None = None,
        core_api: Any | None = None,
    ) -> None:
        from backend.services.runtime.azure_blob_lease import AzureBlobLease

        super().__init__(
            lease=AzureBlobLease(
                account_name=account_name,
                container_name=container_name,
            ),
            namespace=namespace,
            batch_api=batch_api,
            core_api=core_api,
        )


def build_default_lease(cloud: str) -> Any:
    """Build the configured cloud's CAS lease without constructing K8s I/O."""
    from backend.config import get_settings

    settings = get_settings()
    if cloud == "gcp":
        from backend.services.runtime.blob_lease import BlobLease

        bucket = getattr(settings, "gcp_gcs_bucket", "") or ""
        if not bucket:
            raise RuntimeError(
                "durable controller: OPENRESEARCH_GCP_GCS_BUCKET is unset"
            )
        return BlobLease(
            bucket=bucket,
            project=getattr(settings, "gcp_project", None) or None,
        )
    if cloud == "azure":
        from backend.services.runtime.azure_blob_lease import AzureBlobLease

        account = getattr(settings, "azure_storage_account", "") or ""
        container = (
            getattr(settings, "azure_blob_container", "")
            or "reprolab-artifacts"
        )
        if not account:
            raise RuntimeError(
                "durable controller: OPENRESEARCH_AZURE_STORAGE_ACCOUNT is unset"
            )
        return AzureBlobLease(account_name=account, container_name=container)
    raise ValueError(f"unsupported durable-controller cloud: {cloud!r}")


def build_default_cluster(cloud: str = "gcp") -> K8sControllerCluster:
    """Build a GCP or Azure controller cluster from typed settings."""
    from backend.config import get_settings

    settings = get_settings()
    lease = build_default_lease(cloud)
    namespace = (
        getattr(settings, f"{cloud}_namespace", "reprolab") or "reprolab"
    )
    if cloud == "gcp":
        return K8sControllerCluster(lease=lease, namespace=namespace)
    if cloud == "azure":
        return K8sControllerCluster(lease=lease, namespace=namespace)
    # ``build_default_lease`` already rejects this, but keep the branch local
    # for type checkers and future refactors.
    raise ValueError(f"unsupported durable-controller cloud: {cloud!r}")
