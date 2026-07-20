"""AWS EKS Job-native ``RuntimeBackend`` over the shared Kubernetes backend.

This is deliberately a runtime adapter only: it creates neither an EKS
cluster nor an S3 bucket.  Optional SDK imports are lazy and construction is
side-effect free, so choosing another sandbox remains unchanged and tests can
inject Kubernetes/S3 fakes without opening sockets.

EKS workloads should use IRSA for S3 access.  The local orchestrator uses the
normal boto3 credential chain and an already-configured kubeconfig context.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import time
from typing import Any
from uuid import uuid4

from backend.services.runtime.interface import (
    ExecResult,
    RuntimeCauseKind,
    Sandbox,
    SandboxConfig,
    SandboxRuntimeError,
)
from backend.services.runtime.k8s_job_backend import (
    CloudSpec,
    S3Store,
    _KubernetesJobBackend,
    _load_kubeconfig,
    _safe_name,
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

    # EKS labels, VRAM, and effective cost are deployment-specific; never
    # invent them from an instance family or borrow a public catalog price.
    # Without this metadata a user-provided run-USD ceiling would be blind.
    skus = _settings_get(settings, "aws_gpu_skus", ()) or ()
    try:
        max_nodes = int(_settings_get(settings, "aws_max_nodes", 0) or 0)
        gpus_per_node = int(_settings_get(settings, "aws_gpus_per_node", 0) or 0)
        vram = float(_settings_get(settings, "aws_per_gpu_vram_gb", 0.0) or 0.0)
        rate = float(_settings_get(settings, "aws_gpu_usd_per_hour", 0.0) or 0.0)
    except (TypeError, ValueError):
        max_nodes, gpus_per_node, vram, rate = 0, 0, 0.0, 0.0
    if not skus or max_nodes <= 0 or gpus_per_node != 1 or vram <= 0 or rate <= 0:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "AWS sandbox requires declared GPU pool metadata: non-empty "
            "OPENRESEARCH_AWS_GPU_SKUS plus positive AWS_MAX_NODES, "
            "AWS_PER_GPU_VRAM_GB, and AWS_GPU_USD_PER_HOUR; "
            "OPENRESEARCH_AWS_GPUS_PER_NODE must equal 1. This v1 EKS path "
            "meters whole nodes, so multi-GPU nodes would understate idle GPU cost.",
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

    _verify_kube_context_cluster(cluster)


def _verify_kube_context_cluster(cluster_name: str) -> None:
    """Fail closed when the active kubeconfig context is not the configured EKS cluster.

    This reads only local kubeconfig state.  It intentionally happens before
    any remote STS/S3 or Kubernetes API probe, preventing an operator from
    checking or submitting a workload to a similarly named wrong cluster.
    """
    try:
        from kubernetes import config as k8s_config  # type: ignore[import]

        _contexts, active = k8s_config.list_kube_config_contexts()
        active = active or {}
        context = active.get("context") or {}
        bound = str(context.get("cluster") or "")
    except Exception as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "AWS sandbox: cannot inspect the active kubeconfig context after loading it. "
            f"Run 'aws eks update-kubeconfig --name {cluster_name}' and retry. Details: {exc}",
        ) from exc
    expected_arn_suffix = f"/cluster/{cluster_name}"
    if not bound or (bound != cluster_name and not bound.endswith(expected_arn_suffix)):
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "AWS sandbox: active kubeconfig context is not bound to configured EKS "
            f"cluster {cluster_name!r} (active cluster={bound!r}). Run 'aws eks "
            f"update-kubeconfig --name {cluster_name}' and select that context.",
        )


def verify_aws_remote_readiness(
    *,
    settings: Any | None = None,
    core_api: Any | None = None,
    sts_client: Any | None = None,
    s3_client: Any | None = None,
) -> dict[str, str]:
    """Verify the controller identity and declared EKS ServiceAccount.

    This is deliberately separate from :func:`ensure_aws_available`: selecting
    the sandbox remains side-effect free, while a preflight command can opt in
    to these remote reads before a capped campaign starts.  The checks are:

    1. local active kubeconfig is bound to ``aws_eks_cluster``;
    2. the configured K8s ServiceAccount has an IRSA role annotation;
    3. STS returns the controller's authenticated caller identity; and
    4. S3 ``HeadBucket`` succeeds for the configured artifact bucket.

    Calls are read-only and all clients are injectable for socket-hermetic
    tests.  The return intentionally contains only non-secret identifiers.
    This deliberately does *not* claim that the workload IRSA role works.  Use
    :func:`verify_aws_pod_readiness` to execute the bounded, no-GPU probe from
    the actual ServiceAccount before a paid campaign.
    """
    if settings is None:
        from backend.config import get_settings

        settings = get_settings()
    cluster = _settings_get(settings, "aws_eks_cluster", "") or ""
    bucket = _settings_get(settings, "aws_s3_bucket", "") or ""
    namespace = _settings_get(settings, "aws_namespace", "reprolab") or "reprolab"
    service_account = _settings_get(settings, "aws_service_account", "reprolab-sa") or "reprolab-sa"
    region = _settings_get(settings, "aws_region", "") or None
    if not cluster or not bucket:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "AWS readiness probe requires configured aws_eks_cluster and aws_s3_bucket",
        )
    _verify_kube_context_cluster(cluster)

    try:
        if core_api is None:
            from backend.services.runtime.k8s_job_backend import _load_kubernetes_core_api

            core_api = _load_kubernetes_core_api()
        service = core_api.read_namespaced_service_account(
            name=service_account, namespace=namespace
        )
        annotations = getattr(getattr(service, "metadata", None), "annotations", None) or {}
        role_arn = str(annotations.get("eks.amazonaws.com/role-arn") or "")
        if not role_arn.startswith("arn:aws:iam::"):
            raise ValueError("missing eks.amazonaws.com/role-arn IRSA annotation")
    except Exception as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "AWS readiness probe: EKS ServiceAccount "
            f"{namespace}/{service_account} is not IRSA-configured: {exc}",
        ) from exc

    try:
        if sts_client is None or s3_client is None:
            import boto3  # type: ignore[import]

            sts_client = sts_client or boto3.client("sts", region_name=region)
            s3_client = s3_client or boto3.client("s3", region_name=region)
        identity = sts_client.get_caller_identity()
        account = str(identity.get("Account") or "")
        arn = str(identity.get("Arn") or "")
        if not account or not arn:
            raise ValueError("STS returned no Account/Arn")
        s3_client.head_bucket(Bucket=bucket)
    except Exception as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "AWS readiness probe: STS/S3 read-only verification failed for "
            f"bucket {bucket!r}: {exc}",
        ) from exc

    return {
        "cluster": str(cluster),
        "namespace": str(namespace),
        "service_account": str(service_account),
        "irsa_role_arn": role_arn,
        "sts_account": account,
        "sts_arn": arn,
        "s3_bucket": str(bucket),
    }


_POD_PROBE_SCRIPT = r'''import boto3
import json
import os

bucket = os.environ["OPENRESEARCH_AWS_PROBE_BUCKET"]
key = os.environ["OPENRESEARCH_AWS_PROBE_KEY"]
role_arn = os.environ.get("AWS_ROLE_ARN", "")
token_path = os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE", "")
if not role_arn.startswith("arn:aws:iam::"):
    raise RuntimeError("IRSA did not inject AWS_ROLE_ARN")
if not token_path or not os.path.isfile(token_path):
    raise RuntimeError("IRSA did not inject a readable AWS_WEB_IDENTITY_TOKEN_FILE")
if os.environ.get("AWS_EC2_METADATA_DISABLED", "").strip().lower() != "true":
    raise RuntimeError("IMDS must be disabled for the IRSA probe")
payload = b'{"reprolab_irsa_probe":true}'
s3 = boto3.client("s3")
identity = boto3.client("sts").get_caller_identity()
try:
    s3.put_object(Bucket=bucket, Key=key, Body=payload)
    observed = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    if observed != payload:
        raise RuntimeError("S3 probe object contents did not round-trip")
    listed = s3.list_objects_v2(Bucket=bucket, Prefix=key).get("Contents", [])
    if not any(str(item.get("Key", "")) == key for item in listed):
        raise RuntimeError("S3 ListObjectsV2 did not return the scoped probe key")
    print("REPROLAB_IRSA_PROBE=" + json.dumps({
        "account": identity.get("Account"), "arn": identity.get("Arn"), "key": key,
    }, sort_keys=True))
finally:
    s3.delete_object(Bucket=bucket, Key=key)
'''


def _pod_probe_manifest(
    *,
    job_name: str,
    namespace: str,
    service_account: str,
    image: str,
    bucket: str,
    object_key: str,
    region: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Build a no-GPU, no-static-credential IRSA probe Job manifest."""
    labels = {
        "app.kubernetes.io/name": "reprolab-irsa-probe",
        "reprolab/preflight": "irsa",
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "namespace": namespace, "labels": labels},
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": timeout_seconds,
            "ttlSecondsAfterFinished": 60,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "serviceAccountName": service_account,
                    "automountServiceAccountToken": True,
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "irsa-probe",
                        "image": image,
                        "command": ["python", "-c", _POD_PROBE_SCRIPT],
                        "env": [
                            {"name": "OPENRESEARCH_AWS_PROBE_BUCKET", "value": bucket},
                            {"name": "OPENRESEARCH_AWS_PROBE_KEY", "value": object_key},
                            # Do not permit a successful readiness probe to fall back to
                            # the node instance profile.  IRSA must inject its projected
                            # web-identity variables or the script fails closed.
                            {"name": "AWS_EC2_METADATA_DISABLED", "value": "true"},
                            {"name": "AWS_REGION", "value": region},
                            {"name": "AWS_DEFAULT_REGION", "value": region},
                        ],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "128Mi"},
                            "limits": {"cpu": "250m", "memory": "256Mi"},
                        },
                    }],
                },
            },
        },
    }


def _job_terminal_state(job: Any) -> str | None:
    """Return ``complete``/``failed`` for a Job status, else ``None``."""
    conditions = getattr(getattr(job, "status", None), "conditions", None) or ()
    for condition in conditions:
        if str(getattr(condition, "status", "")) != "True":
            continue
        kind = str(getattr(condition, "type", ""))
        if kind == "Complete":
            return "complete"
        if kind == "Failed":
            return "failed"
    return None


def _probe_log(core_api: Any, *, namespace: str, job_name: str) -> str:
    """Return a bounded best-effort probe log without masking probe failure."""
    try:
        pods = core_api.list_namespaced_pod(
            namespace=namespace, label_selector=f"job-name={job_name}"
        )
        items = getattr(pods, "items", None) or ()
        if not items:
            return ""
        name = str(getattr(getattr(items[0], "metadata", None), "name", ""))
        if not name:
            return ""
        return str(core_api.read_namespaced_pod_log(name=name, namespace=namespace))[-4_000:]
    except Exception:
        return ""


def _verified_irsa_probe_arn(
    probe_log: str,
    *,
    role_arn: str,
    object_key: str,
) -> str:
    """Extract and validate the bounded IRSA proof emitted by the probe Pod.

    A Completed Job alone proves neither that logs were retrievable nor that
    the Pod used the intended ServiceAccount role.  The marker makes this
    independent of incidental application logs and binds the proof to both the
    configured IAM role and the exact S3 prefix exercised by this invocation.
    """
    marker = "REPROLAB_IRSA_PROBE="
    payload: dict[str, Any] | None = None
    for line in reversed(probe_log.splitlines()):
        if marker not in line:
            continue
        try:
            candidate = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError as exc:
            raise ValueError("IRSA probe emitted malformed JSON evidence") from exc
        if not isinstance(candidate, dict):
            raise ValueError("IRSA probe emitted non-object JSON evidence")
        payload = candidate
        break
    if payload is None:
        raise ValueError("IRSA probe emitted no verifiable STS evidence")

    account = str(payload.get("account") or "")
    arn = str(payload.get("arn") or "")
    observed_key = str(payload.get("key") or "")
    try:
        expected_account = role_arn.split(":", 5)[4]
        role_resource = role_arn.split(":role/", 1)[1]
        role_name = role_resource.rsplit("/", 1)[-1]
    except (IndexError, ValueError) as exc:
        raise ValueError("configured IRSA role ARN is malformed") from exc
    expected_arn_prefix = f"arn:aws:sts::{expected_account}:assumed-role/{role_name}/"
    if account != expected_account or not arn.startswith(expected_arn_prefix):
        raise ValueError(
            "IRSA probe STS identity does not match the ServiceAccount role annotation"
        )
    if observed_key != object_key:
        raise ValueError("IRSA probe evidence did not bind the submitted S3 key")
    return arn


def verify_aws_pod_readiness(
    *,
    project_id: str,
    run_id: str,
    timeout_seconds: int = 120,
    settings: Any | None = None,
    batch_api: Any | None = None,
    core_api: Any | None = None,
    poll_seconds: float = 2.0,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> dict[str, str]:
    """Run a bounded, no-GPU proof of the *pod* IRSA and scoped S3 policy.

    The probe runs under the configured ServiceAccount, calls STS, and performs
    S3 Put/Get/List/Delete on an ephemeral key under the exact project/run
    prefix a campaign would use.  It never receives static AWS credentials and
    is foreground-deleted even after a successful probe.  This is deliberately
    an explicit preflight operation (not part of sandbox selection), because it
    creates one short-lived Kubernetes Job and a transient object.
    """
    if not project_id.strip() or not run_id.strip():
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "AWS pod readiness requires non-empty project_id and run_id to verify the exact S3 scope",
        )
    if not 30 <= int(timeout_seconds) <= 300:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "AWS pod readiness timeout_seconds must be between 30 and 300",
        )
    if settings is None:
        from backend.config import get_settings

        settings = get_settings()
    cluster = str(_settings_get(settings, "aws_eks_cluster", "") or "")
    bucket = str(_settings_get(settings, "aws_s3_bucket", "") or "")
    region = str(_settings_get(settings, "aws_region", "") or "")
    namespace = str(_settings_get(settings, "aws_namespace", "reprolab") or "reprolab")
    service_account = str(_settings_get(settings, "aws_service_account", "reprolab-sa") or "reprolab-sa")
    image = str(_settings_get(settings, "aws_base_image", "") or "")
    if not cluster or not bucket or not image or not region:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "AWS pod readiness requires configured aws_eks_cluster, aws_s3_bucket, aws_base_image, and aws_region",
        )
    _verify_kube_context_cluster(cluster)
    try:
        if batch_api is None or core_api is None:
            from backend.services.runtime.k8s_job_backend import (
                _load_kubernetes_batch_api,
                _load_kubernetes_core_api,
            )

            batch_api = batch_api or _load_kubernetes_batch_api()
            core_api = core_api or _load_kubernetes_core_api()
        service = core_api.read_namespaced_service_account(name=service_account, namespace=namespace)
        annotations = getattr(getattr(service, "metadata", None), "annotations", None) or {}
        role_arn = str(annotations.get("eks.amazonaws.com/role-arn") or "")
        if not role_arn.startswith("arn:aws:iam::"):
            raise ValueError("missing eks.amazonaws.com/role-arn IRSA annotation")
    except Exception as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            f"AWS pod readiness: ServiceAccount {namespace}/{service_account} is not IRSA-configured: {exc}",
        ) from exc

    suffix = uuid4().hex[:12]
    job_name = f"reprolab-irsa-{suffix}"
    object_key = (
        f"projects/{_safe_name(project_id)}/runs/{_safe_name(run_id)}"
        f"/preflight/irsa-{suffix}.json"
    )
    manifest = _pod_probe_manifest(
        job_name=job_name,
        namespace=namespace,
        service_account=service_account,
        image=image,
        bucket=bucket,
        object_key=object_key,
        region=region,
        timeout_seconds=int(timeout_seconds),
    )
    submitted = False
    outcome: str | None = None
    probe_log = ""
    cleanup_error: Exception | None = None
    try:
        try:
            batch_api.create_namespaced_job(namespace=namespace, body=manifest)
            submitted = True
        except Exception as exc:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"AWS pod readiness: failed to submit no-GPU IRSA probe Job {job_name!r}: {exc}",
            ) from exc
        deadline = monotonic() + int(timeout_seconds)
        while monotonic() < deadline:
            try:
                outcome = _job_terminal_state(
                    batch_api.read_namespaced_job_status(name=job_name, namespace=namespace)
                )
            except Exception as exc:
                raise SandboxRuntimeError(
                    RuntimeCauseKind.backend_unavailable,
                    f"AWS pod readiness: could not observe probe Job {job_name!r}: {exc}",
                ) from exc
            if outcome is not None:
                break
            sleep(max(0.1, float(poll_seconds)))
        probe_log = _probe_log(core_api, namespace=namespace, job_name=job_name)
        if outcome != "complete":
            state = outcome or "timed_out"
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"AWS pod readiness: no-GPU IRSA probe Job {job_name!r} {state}; log={probe_log!r}",
            )
    finally:
        if submitted:
            try:
                batch_api.delete_namespaced_job(
                    name=job_name,
                    namespace=namespace,
                    body={"propagationPolicy": "Foreground"},
                )
            except Exception as exc:  # foreground cleanup is a safety condition, not best effort
                cleanup_error = exc
        if cleanup_error is not None:
            raise SandboxRuntimeError(
                RuntimeCauseKind.backend_unavailable,
                f"AWS pod readiness: failed to foreground-delete probe Job {job_name!r}: {cleanup_error}",
            ) from cleanup_error

    try:
        identity = _verified_irsa_probe_arn(
            probe_log, role_arn=role_arn, object_key=object_key,
        )
    except ValueError as exc:
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            f"AWS pod readiness: completed no-GPU IRSA probe Job {job_name!r} "
            f"without a valid IRSA identity proof: {exc}",
        ) from exc
    return {
        "cluster": cluster,
        "namespace": namespace,
        "service_account": service_account,
        "s3_bucket": bucket,
        "s3_probe_key": object_key,
        "pod_sts_arn": str(identity),
    }


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

    def _blob_code_prefix(self, project_id: str, run_id: str) -> str:
        """Keep EKS uploads project-isolated without rewriting GKE/AKS keys."""
        return f"projects/{_safe_name(project_id)}/runs/{_safe_name(run_id)}/code/"

    def _blob_artifact_key(self, project_id: str, run_id: str, path: str) -> str:
        """Keep EKS artifacts project-isolated without rewriting GKE/AKS keys."""
        return (
            f"projects/{_safe_name(project_id)}/runs/{_safe_name(run_id)}"
            f"/artifacts/{path.lstrip('/')}"
        )

    async def create_sandbox(self, config: SandboxConfig) -> Sandbox:
        """Reject the generic sandbox protocol before it can upload a code bundle.

        EKS cells are submitted directly by ``k8s_job_cell_runner``.  That path
        makes the S3/IRSA entrypoint, immutable prefixes, and cap checks part of
        the Job contract.  The shared sandbox protocol cannot prove those
        properties, so even staging it is refused rather than leaving a usable
        generic EKS execution route behind.
        """
        del config
        raise SandboxRuntimeError(
            RuntimeCauseKind.backend_unavailable,
            "AWS EKS generic sandbox creation is disabled: use the EKS cell-matrix "
            "route with the S3/IRSA entrypoint.",
        )

    async def exec(self, sandbox: Sandbox, command: str, timeout: int) -> ExecResult:
        """Reject generic exec: EKS reproduction is cell-matrix-only.

        The shared generic Kubernetes exec protocol overrides the image entrypoint
        with ``/bin/sh`` and does not materialize the S3 code prefix.  Allowing
        it for EKS would bypass the IRSA/S3 cell contract, so a caller must route
        through ``k8s_job_cell_runner`` instead.  Returning a typed failed result
        preserves RuntimeBackend's ordinary command-failure handling while making
        the zero-Job guarantee easy to assert hermetically.
        """
        now = datetime.now(timezone.utc)
        return ExecResult(
            command=command,
            exit_code=1,
            stderr=(
                "AWS EKS generic exec is disabled: use the EKS cell-matrix route "
                "with the S3/IRSA entrypoint."
            ),
            started_at=now,
            finished_at=now,
            duration_seconds=0.0,
            cause_kind=RuntimeCauseKind.command_failed,
        )

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

    def _gpu_plan_total_usd_per_hr(self) -> float:
        """Meter EKS with the configured pool rate, never a foreign plan's price."""
        allowed = _settings_get(self._get_settings(), "aws_gpu_skus", ()) or ()
        short_name = self._gpu_plan_short_name()
        if short_name is None or short_name not in allowed:
            return 0.0
        try:
            rate = float(_settings_get(self._get_settings(), "aws_gpu_usd_per_hour", 0.0) or 0.0)
            count = self._gpu_plan_gpu_count()
            return rate * max(1, count) if rate > 0 else 0.0
        except (TypeError, ValueError):
            return 0.0


__all__ = ["EksJobBackend", "ensure_aws_available", "verify_aws_remote_readiness"]
