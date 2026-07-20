"""AWS EKS Job-native ``RuntimeBackend`` over the shared Kubernetes backend.

This is deliberately a runtime adapter only: it creates neither an EKS
cluster nor an S3 bucket.  Optional SDK imports are lazy and construction is
side-effect free, so choosing another sandbox remains unchanged and tests can
inject Kubernetes/S3 fakes without opening sockets.

EKS workloads should use IRSA for S3 access.  The local orchestrator uses the
normal boto3 credential chain and an already-configured kubeconfig context.
"""

from __future__ import annotations

from typing import Any

from backend.services.runtime.interface import RuntimeCauseKind, SandboxRuntimeError
from backend.services.runtime.k8s_job_backend import (
    CloudSpec,
    S3Store,
    _KubernetesJobBackend,
    _load_kubeconfig,
    _settings_get,
)


def _make_s3_store(settings: Any, client: Any) -> S3Store:
    """Build the S3-backed object store; no boto3 client is made yet."""
    return S3Store(
        _settings_get(settings, "aws_s3_bucket", "") or "",
        _settings_get(settings, "aws_region", "") or None,
        client,
    )


def ensure_aws_available() -> None:
    """Validate local prerequisites for an explicitly selected EKS sandbox.

    The gate makes no AWS API call and does not create or modify cloud
    resources.  Authentication is intentionally delegated to the normal boto3
    chain when S3 I/O is first requested, while Kubernetes authentication is
    validated by loading the operator's existing kubeconfig.
    """
    try:
        import boto3  # type: ignore[import]  # noqa: F401
    except ImportError as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "AWS sandbox requires 'boto3'. Install it with: pip install boto3",
        ) from exc

    try:
        import kubernetes  # type: ignore[import]  # noqa: F401
    except ImportError as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "AWS sandbox requires 'kubernetes' (the Python client). "
            "Install it with: pip install kubernetes",
        ) from exc

    try:
        from backend.config import get_settings

        settings = get_settings()
    except Exception:
        settings = None

    bucket = _settings_get(settings, "aws_s3_bucket", "") or ""
    if not bucket.strip():
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "AWS sandbox requires OPENRESEARCH_AWS_S3_BUCKET (or aws_s3_bucket "
            "setting) to be set. Add it to .env and re-run.",
        )
    cluster = _settings_get(settings, "aws_eks_cluster", "") or ""
    if not cluster.strip():
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "AWS sandbox requires OPENRESEARCH_AWS_EKS_CLUSTER (or aws_eks_cluster "
            "setting) to be set. Add it to .env and re-run.",
        )

    try:
        _load_kubeconfig()
    except SandboxRuntimeError:
        raise
    except Exception as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "AWS sandbox: could not load kubeconfig. Run 'aws eks update-kubeconfig "
            "--name <cluster> --region <region>' for the configured cluster. "
            f"Details: {exc}",
        ) from exc


_AWS_CLOUD = CloudSpec(
    provider="aws",
    settings_prefix="aws",
    sandbox_prefix="eks",
    sandbox_label="eks",
    pod_template_extra_labels={},
    base_image_setting="aws_base_image",
    make_object_store=_make_s3_store,
    ensure_available=ensure_aws_available,
)


class EksJobBackend(_KubernetesJobBackend):
    """Runtime backend that dispatches short-lived training Jobs to EKS.

    Construction is inert.  A sandbox upload uses S3 and each ``exec`` creates
    one short-lived Kubernetes Job in the already-configured EKS cluster.
    ``destroy`` deletes only Jobs created by this backend and preserves S3
    artifacts for review.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(_AWS_CLOUD, **kw)

    def _gpu_plan_short_name(self) -> str | None:
        """Use a plan label only when the operator declared it for this EKS pool.

        The existing GPU resolver has GCP/Azure catalogs but intentionally has
        no AWS catalog in this foundation.  Without this guard an unrelated
        RunPod plan (for example ``rtx4090``) could become an impossible EKS
        node selector.  Empty ``aws_gpu_skus`` therefore means no inferred
        selector; a configured list is the sole source of truth.
        """
        short_name = super()._gpu_plan_short_name()
        allowed = _settings_get(self._get_settings(), "aws_gpu_skus", ()) or ()
        try:
            return short_name if short_name in allowed else None
        except TypeError:
            return None


__all__ = ["EksJobBackend", "ensure_aws_available"]
