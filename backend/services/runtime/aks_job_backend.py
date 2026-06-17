"""Azure AKS Job-native RuntimeBackend — thin adapter over _KubernetesJobBackend.

Each ``exec`` call submits a short Kubernetes Job running the command via the
pre-baked ACR base image's exec mode, waits for it to complete, captures logs,
and returns a fully-populated ``ExecResult``.  ``create_sandbox`` uploads the
project to a run-scoped Azure Blob prefix and returns a logical
(no-persistent-pod) ``Sandbox``.  ``destroy`` deletes any tracked active Jobs
but intentionally **preserves** Blob artifacts for post-run review.

All Kubernetes and Azure SDK imports are **lazy** (behind factory functions) so
the module can be imported and the full test suite can run without those optional
packages installed.  Tests inject fake ``batch_api``, ``core_api``, and
``blob_client`` via the constructor.

Auth posture:
  - In-cluster Jobs use Workload Identity (annotated ServiceAccount + federated
    MI) to reach Azure Blob — no secrets in-cluster.
  - The local orchestrator uses ``DefaultAzureCredential`` (``az login``) to
    upload code and download artifacts, and the operator's kubeconfig to submit
    and watch Jobs.

Design reference: ``docs/superpowers/specs/2026-06-03-azure-aks-gpu-backend-design.md``
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
    _build_job_manifest,
    _safe_name,  # noqa: F401  (re-exported for back-compat)
    _job_name,  # noqa: F401  (re-exported for back-compat)
    _extract_exit_code,  # noqa: F401  (re-exported for back-compat)
    _DEFAULT_NAMESPACE,  # noqa: F401  (re-exported for back-compat)
    _DEFAULT_PENDING_TIMEOUT_S,  # noqa: F401  (re-exported for back-compat)
    _DEFAULT_TTL_AFTER_FINISHED_S,  # noqa: F401  (re-exported for back-compat)
    _DEFAULT_BACKOFF_LIMIT,  # noqa: F401  (re-exported for back-compat)
    _DEFAULT_BASE_IMAGE,  # noqa: F401  (re-exported for back-compat)
    _blob_code_prefix,  # noqa: F401  (re-exported for back-compat)
    _blob_artifact_key,  # noqa: F401  (re-exported for back-compat)
    _settings_get,
    _load_kubeconfig,
    AzureBlobStore,
    CloudSpec,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Azure-specific manifest builder (public for test backward-compatibility)
# ---------------------------------------------------------------------------


def _build_exec_job_manifest(
    *,
    job_name: str,
    namespace: str,
    image: str,
    service_account: str,
    command: str,
    environment: dict[str, str],
    active_deadline_seconds: int,
    ttl_seconds: int,
    backoff_limit: int,
    gpu_sku: str | None = None,
    gpu_count: int = 1,
) -> dict[str, Any]:
    """Azure-specific wrapper: adds Workload Identity pod label + aks sandbox label."""
    return _build_job_manifest(
        job_name=job_name,
        namespace=namespace,
        image=image,
        service_account=service_account,
        command=command,
        environment=environment,
        active_deadline_seconds=active_deadline_seconds,
        ttl_seconds=ttl_seconds,
        backoff_limit=backoff_limit,
        gpu_sku=gpu_sku,
        gpu_count=gpu_count,
        pod_template_extra_labels={"azure.workload.identity/use": "true"},
        sandbox_label="aks",
    )


# ---------------------------------------------------------------------------
# Azure CloudSpec + store factory
# ---------------------------------------------------------------------------


def _make_azure_store(settings: Any, client: Any) -> AzureBlobStore:
    """Build an AzureBlobStore from the current settings."""
    account = _settings_get(settings, "azure_storage_account", "") or ""
    container = (
        _settings_get(settings, "azure_blob_container", "reprolab-artifacts")
        or "reprolab-artifacts"
    )
    return AzureBlobStore(account, container, client)


# ---------------------------------------------------------------------------
# Availability gate
# ---------------------------------------------------------------------------


def ensure_azure_available() -> None:
    """Fail fast when Azure sandbox is selected but prerequisites are missing.

    Verifies (in order):
      1. ``azure.identity`` is importable.
      2. ``azure.storage.blob`` is importable.
      3. ``kubernetes`` is importable.
      4. Required settings (storage account) are non-empty.
      5. ``DefaultAzureCredential`` can be constructed (i.e. ``az login`` was run).
      6. A kubeconfig can be loaded (i.e. ``az aks get-credentials`` was run).

    Raises ``SandboxRuntimeError(backend_unavailable, <actionable message>)``
    on any failure.
    """
    # 1. azure.identity
    try:
        from azure.identity import DefaultAzureCredential  # type: ignore[import]  # noqa: F401
    except ImportError:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "Azure sandbox requires 'azure-identity'. "
            "Install it with: pip install azure-identity",
        )

    # 2. azure.storage.blob
    try:
        from azure.storage.blob import ContainerClient  # type: ignore[import]  # noqa: F401
    except ImportError:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "Azure sandbox requires 'azure-storage-blob'. "
            "Install it with: pip install azure-storage-blob",
        )

    # 3. kubernetes
    try:
        import kubernetes  # type: ignore[import]  # noqa: F401
    except ImportError:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "Azure sandbox requires 'kubernetes' (the Python client). "
            "Install it with: pip install kubernetes",
        )

    # 4. Required settings.
    try:
        from backend.config import get_settings

        s = get_settings()
    except Exception:
        s = None

    storage_account = _settings_get(s, "azure_storage_account", "") or ""
    if not storage_account.strip():
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "Azure sandbox requires AZURE_STORAGE_ACCOUNT (or azure_storage_account setting) "
            "to be set. Add it to .env and re-run.",
        )

    # 5. DefaultAzureCredential — verify it can be instantiated and has a token.
    try:
        from azure.identity import DefaultAzureCredential  # type: ignore[import]

        cred = DefaultAzureCredential()
        # A quick token probe to confirm the credential chain is functional.
        cred.get_token("https://storage.azure.com/.default")
    except Exception as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            f"Azure sandbox: DefaultAzureCredential could not acquire a token. "
            f"Run 'az login' (or set AZURE_CLIENT_ID/AZURE_CLIENT_SECRET/AZURE_TENANT_ID). "
            f"Details: {exc}",
        ) from exc

    # 6. kubeconfig — verify that a kubeconfig can be loaded.
    try:
        _load_kubeconfig()
    except SandboxRuntimeError:
        raise
    except Exception as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            f"Azure sandbox: could not load kubeconfig. "
            f"Run 'az aks get-credentials --resource-group <rg> --name <cluster>'. "
            f"Details: {exc}",
        ) from exc


_AZURE_CLOUD = CloudSpec(
    provider="azure",
    settings_prefix="azure",
    sandbox_prefix="aks",
    sandbox_label="aks",
    pod_template_extra_labels={"azure.workload.identity/use": "true"},
    base_image_setting="azure_base_image",
    make_object_store=_make_azure_store,
    ensure_available=ensure_azure_available,
)


# ---------------------------------------------------------------------------
# AksJobBackend
# ---------------------------------------------------------------------------


class AksJobBackend(_KubernetesJobBackend):
    """RuntimeBackend backed by Azure AKS short-lived Kubernetes Jobs.

    The backend is **pod-free**: there is no persistent pod.  ``create_sandbox``
    uploads the project to Blob and returns a logical ``Sandbox`` token.
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
      blob_client  – pre-built Blob ``ContainerClient`` (or fake).
      settings     – settings object; defaults to ``get_settings()`` lazily.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(_AZURE_CLOUD, **kw)


__all__ = ["AksJobBackend", "ensure_azure_available"]
