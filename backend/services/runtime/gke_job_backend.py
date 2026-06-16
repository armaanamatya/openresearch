"""GCP GKE Job-native RuntimeBackend — thin adapter over _KubernetesJobBackend.

Each ``exec`` call submits a short Kubernetes Job running the command via the
pre-baked GCR/Artifact Registry base image's exec mode, waits for it to
complete, captures logs, and returns a fully-populated ``ExecResult``.
``create_sandbox`` uploads the project to a run-scoped GCS prefix and returns a
logical (no-persistent-pod) ``Sandbox``.  ``destroy`` deletes any tracked active
Jobs but intentionally **preserves** GCS artifacts for post-run review.

All Kubernetes and GCP SDK imports are **lazy** (behind factory functions) so
the module can be imported and the full test suite can run without those optional
packages installed.  Tests inject fake ``batch_api``, ``core_api``, and
``blob_client`` via the constructor.

Auth posture:
  - In-cluster Jobs use GKE Workload Identity (Kubernetes ServiceAccount bound
    to a GCP IAM Service Account) to reach GCS — no secrets in-cluster.
  - The local orchestrator uses Application Default Credentials
    (``gcloud auth application-default login``) to upload code and download
    artifacts, and the operator's kubeconfig to submit and watch Jobs.

Design reference: ``docs/superpowers/specs/2026-06-16-gcp-gke-execution-backend-design.md``
"""

from __future__ import annotations

import logging
from typing import Any

from backend.services.runtime.interface import (
    RuntimeCauseKind,
    SandboxRuntimeError,
)
from backend.services.runtime.k8s_job_backend import (
    _KubernetesJobBackend,
    _settings_get,
    _load_kubeconfig,
    GcsStore,
    CloudSpec,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GCP CloudSpec + store factory
# ---------------------------------------------------------------------------


def _make_gcs_store(settings: Any, client: Any) -> GcsStore:
    """Build a GcsStore from the current settings."""
    bucket = _settings_get(settings, "gcp_gcs_bucket", "") or ""
    project = _settings_get(settings, "gcp_project", None) or None
    return GcsStore(bucket, project, client)


# ---------------------------------------------------------------------------
# Availability gate
# ---------------------------------------------------------------------------


def ensure_gcp_available() -> None:
    """Fail fast when GCP/GKE sandbox is selected but prerequisites are missing.

    Verifies (in order):
      1. ``google.cloud.storage`` is importable.
      2. ``google.auth`` is importable.
      3. ``kubernetes`` is importable.
      4. Required settings (GCS bucket, GCP project) are non-empty.
      5. Application Default Credentials are available.
      6. A kubeconfig can be loaded (i.e. ``gcloud container clusters get-credentials``
         was run).

    Raises ``SandboxRuntimeError(backend_unavailable, <actionable message>)``
    on any failure.
    """
    # 1. google.cloud.storage
    try:
        import google.cloud.storage  # type: ignore[import]  # noqa: F401
    except ImportError:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "GCP sandbox requires 'google-cloud-storage'. "
            "Install it with: pip install google-cloud-storage",
        )

    # 2. google.auth
    try:
        import google.auth  # type: ignore[import]  # noqa: F401
    except ImportError:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "GCP sandbox requires 'google-auth'. "
            "Install it with: pip install google-auth",
        )

    # 3. kubernetes
    try:
        import kubernetes  # type: ignore[import]  # noqa: F401
    except ImportError:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "GCP sandbox requires 'kubernetes' (the Python client). "
            "Install it with: pip install kubernetes",
        )

    # 4. Required settings.
    try:
        from backend.config import get_settings

        s = get_settings()
    except Exception:
        s = None

    gcs_bucket = _settings_get(s, "gcp_gcs_bucket", "") or ""
    if not gcs_bucket.strip():
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "GCP sandbox requires OPENRESEARCH_GCP_GCS_BUCKET (or gcp_gcs_bucket setting) "
            "to be set. Add it to .env and re-run.",
        )

    gcp_project = _settings_get(s, "gcp_project", "") or ""
    if not gcp_project.strip():
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "GCP sandbox requires OPENRESEARCH_GCP_PROJECT (or gcp_project setting) "
            "to be set. Add it to .env and re-run.",
        )

    # 5. Application Default Credentials probe.
    try:
        import google.auth  # type: ignore[import]

        google.auth.default(
            scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
        )
    except Exception as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            f"GCP sandbox: Application Default Credentials not available. "
            f"Run 'gcloud auth application-default login'. Details: {exc}",
        ) from exc

    # 6. kubeconfig — verify that a kubeconfig can be loaded.
    try:
        _load_kubeconfig()
    except SandboxRuntimeError:
        raise
    except Exception as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            f"GCP sandbox: could not load kubeconfig. "
            f"Run 'gcloud container clusters get-credentials <cluster> --region <region>'. "
            f"Details: {exc}",
        ) from exc


_GCP_CLOUD = CloudSpec(
    provider="gcp",
    settings_prefix="gcp",
    sandbox_prefix="gke",
    sandbox_label="gke",
    pod_template_extra_labels={},
    base_image_setting="gcp_base_image",
    make_object_store=_make_gcs_store,
    ensure_available=ensure_gcp_available,
)


# ---------------------------------------------------------------------------
# GkeJobBackend
# ---------------------------------------------------------------------------


class GkeJobBackend(_KubernetesJobBackend):
    """RuntimeBackend backed by GCP GKE short-lived Kubernetes Jobs.

    The backend is **pod-free**: there is no persistent pod.  ``create_sandbox``
    uploads the project to GCS and returns a logical ``Sandbox`` token.
    ``exec`` submits a K8s Job, polls until completion, captures logs, and
    returns an ``ExecResult``.  ``destroy`` cleans up any Jobs submitted during
    the sandbox's lifetime.

    All expensive I/O operations are run in an executor thread so the async
    interface stays non-blocking even though the Kubernetes client library is
    synchronous.

    Constructor keyword arguments (all optional; test doubles injected here):
      run_budget   – ``RunBudget`` for pod-second / GPU-USD caps.
      gpu_plan     – ``GpuPlan`` (or a plain dict with the same keys) resolved by
                     ``resolve_gpu_requirements``; when set, Jobs are scheduled on
                     the matching node pool via ``nodeSelector={"reprolab/sku": plan.short_name}``
                     and request ``nvidia.com/gpu=plan.gpu_count`` resources.  When
                     ``None`` (default) the backend falls back to the settings node
                     pool and requests 1 GPU.
      batch_api    – pre-built ``kubernetes.client.BatchV1Api`` (or fake).
      core_api     – pre-built ``kubernetes.client.CoreV1Api`` (or fake).
      blob_client  – pre-built GCS ``Bucket``-like client (or fake).
      settings     – settings object; defaults to ``get_settings()`` lazily.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(_GCP_CLOUD, **kw)


__all__ = ["GkeJobBackend", "ensure_gcp_available"]
