"""WS3 durable-controller cluster seam + takeover-safe submit orchestration.

The correctness-critical ordering lives here as pure logic over an injected
``ControllerCluster`` (duck-typed), so every race and failure mode is unit
testable with a fake — no GCS/K8s/network. ``live_runs`` builds the manifest
and does the file-backed I/O; this module owns *when* to acquire, submit, wait,
reap, and when it is safe to fall back.

Design: ``docs/superpowers/specs/2026-07-12-cloud-native-durable-and-cpu-lane-design.md`` §3.2, §7.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "ControllerCluster",
    "ControllerNotReady",
    "ControllerStuck",
    "submit_controller",
    "sweep_orphaned_controllers",
    "GcsK8sControllerCluster",
    "build_default_cluster",
]


class ControllerNotReady(Exception):
    """The submitted controller Job never became ready, but was CONFIRMED
    deleted — no remote controller is live, so the caller MAY safely fall back
    to a local run."""


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
    Raises :class:`ControllerNotReady` (safe to fall back) or :class:`ControllerStuck`
    (must not fall back). Pre-submit exceptions (acquire/submit) propagate to the caller.
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

    cluster.submit(manifest)  # pre-ready failure propagates → caller's fallback

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
    build_manifest_for: Callable[[str], Callable[[int], dict]],
    ready_timeout_s: float,
    sweeper_owner: str = "sweeper",
) -> list[str]:
    """One deterministic sweep pass: resubmit controllers whose lease has lapsed.

    Split-brain safe by construction: the sweep acquires with a DISTINCT
    ``sweeper_owner`` (never a run's own ``owner_id``), so ``acquire`` returns a
    token ONLY when the lease is free or TTL-expired — a healthy controller that
    heartbeats its lease is never taken over (``acquire`` → ``None`` → skipped).
    A taken-over (expired) lease bumps the fence, so the successor reaps the dead
    predecessor's Jobs. Per-run fail-soft; returns the project_ids resubmitted.

    K8s ``backoffLimit`` remains the primary self-healing for controller Pod
    crashes; this covers the rarer Job-deleted / backoff-exhausted case.
    """
    resubmitted: list[str] = []
    for project_id in list_durable_runs():
        try:
            handle = submit_controller(
                cluster,
                build_manifest=build_manifest_for(project_id),
                project_id=project_id,
                owner_id=sweeper_owner,
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


class GcsK8sControllerCluster:
    """Real ``BlobLease`` (GCS CAS) + K8s ``BatchV1Api`` implementation of the
    :class:`ControllerCluster` seam.

    Exercised at drill time only — every method fail-fast raises on a missing
    SDK / config / API error so ``_submit_durable_controller`` degrades to a
    local run (never hangs, never silently no-ops). Not hermetically tested
    (socket-hermetic): the correctness logic lives in the pure
    :func:`submit_controller`/:func:`sweep_orphaned_controllers` above, which
    ARE tested against a fake of this exact shape.
    """

    def __init__(self, *, bucket: str, project: str | None, namespace: str) -> None:
        from backend.services.runtime.blob_lease import BlobLease

        self._lease = BlobLease(bucket=bucket, project=project)
        self._namespace = namespace

    def now(self) -> float:
        import time

        return time.time()

    def acquire(self, project_id: str, owner_id: str, now_epoch: float) -> Any:
        return self._lease.acquire(project_id, owner_id, now_epoch)

    def is_current(self, token: Any) -> bool:
        return self._lease.is_current(token)

    def _batch(self) -> Any:
        from backend.services.runtime import k8s_job_backend as kb

        kb._configure_kubernetes_client()
        return kb._load_kubernetes_batch_api()

    def submit(self, manifest: dict) -> None:
        self._batch().create_namespaced_job(namespace=self._namespace, body=manifest)

    def wait_ready(self, job_name: str, *, timeout_s: float) -> bool:
        import time

        batch = self._batch()
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            job = batch.read_namespaced_job_status(name=job_name, namespace=self._namespace)
            status = getattr(job, "status", None)
            if status is not None:
                if (getattr(status, "active", 0) or 0) >= 1:
                    return True
                if (getattr(status, "succeeded", 0) or 0) >= 1:
                    return True  # already ran to completion — treat as ready
                if (getattr(status, "failed", 0) or 0) >= 1:
                    return False
            time.sleep(2.0)
        return False

    def delete_confirmed(self, job_name: str) -> bool:
        import time

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
            time.sleep(1.0)
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
                raw = labels.get("reprolab-generation")
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


def build_default_cluster() -> "GcsK8sControllerCluster":
    """Build the real cluster from settings. Raises when gcp isn't configured —
    the caller catches it and falls back to a local run (fail-soft)."""
    from backend.config import get_settings

    settings = get_settings()
    bucket = getattr(settings, "gcp_gcs_bucket", "") or ""
    if not bucket:
        raise RuntimeError(
            "durable controller: OPENRESEARCH_GCP_GCS_BUCKET is unset; "
            "cannot build the GCS-backed controller cluster"
        )
    return GcsK8sControllerCluster(
        bucket=bucket,
        project=getattr(settings, "gcp_project", None) or None,
        namespace=getattr(settings, "gcp_namespace", "reprolab") or "reprolab",
    )
