"""Hermetic EKS adapter tests: no boto3, Kubernetes, or network required."""

from __future__ import annotations

import io
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.services.runtime.eks_job_backend import (
    EksJobBackend,
    ensure_aws_available,
    verify_aws_pod_readiness,
    verify_aws_remote_readiness,
)
from backend.services.runtime.interface import Sandbox, SandboxConfig, SandboxRuntimeError


class _Batch:
    def __init__(self) -> None:
        self.created: list[dict] = []

    def create_namespaced_job(self, *, namespace: str, body: dict):
        self.created.append({"namespace": namespace, "body": body})

    def read_namespaced_job_status(self, *, name: str, namespace: str):
        condition = SimpleNamespace(type="Complete", status="True")
        return SimpleNamespace(status=SimpleNamespace(conditions=[condition]))

    def delete_namespaced_job(self, *, name: str, namespace: str, body=None):
        return None


class _Core:
    def list_namespaced_pod(self, *, namespace: str, label_selector: str):
        pod = SimpleNamespace(metadata=SimpleNamespace(name="done"), status=SimpleNamespace(phase="Succeeded"))
        return SimpleNamespace(items=[pod])

    def read_namespaced_pod_log(self, *, name: str, namespace: str, _preload_content=True):
        return "done"


class _S3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes):  # noqa: N803
        self.objects[(Bucket, Key)] = Body
        return {}

    def get_object(self, *, Bucket: str, Key: str):  # noqa: N803
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


def _settings(**overrides):
    values = dict(
        aws_region="us-west-2",
        aws_s3_bucket="reprolab-artifacts",
        aws_eks_cluster="reprolab-eks",
        aws_namespace="reprolab",
        aws_base_image="123456789012.dkr.ecr.us-west-2.amazonaws.com/reprolab:sha-1",
        aws_service_account="reprolab-sa",
        aws_pending_timeout_seconds=900,
        aws_ttl_seconds_after_finished=3600,
        aws_job_backoff_limit=0,
        aws_gpu_skus=["eks-a100"],
        aws_max_nodes=1,
        aws_gpus_per_node=1,
        aws_per_gpu_vram_gb=80.0,
        aws_gpu_usd_per_hour=3.25,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _sandbox(tmp_path: Path) -> Sandbox:
    config = SandboxConfig(project_id="proj", run_id="run", project_root=tmp_path, image="unused")
    return Sandbox(
        sandbox_id="eks-proj-run",
        name="reprolab-proj-run",
        image="unused",
        config=config,
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_generic_exec_is_blocked_before_any_eks_job_or_env_serialization(tmp_path: Path):
    batch = _Batch()
    backend = EksJobBackend(batch_api=batch, core_api=_Core(), blob_client=object(), settings=_settings())

    sandbox = _sandbox(tmp_path).model_copy(update={
        "config": _sandbox(tmp_path).config.model_copy(update={
            "environment": {"AWS_SECRET_ACCESS_KEY": "must-not-reach-pod"},
        }),
    })
    result = await backend.exec(sandbox, "echo hello", timeout=10)

    assert result.exit_code == 1
    assert result.cause_kind.value == "command_failed"
    assert "cell-matrix" in result.stderr
    assert batch.created == []


@pytest.mark.asyncio
async def test_generic_sandbox_creation_is_blocked_before_any_s3_upload(tmp_path: Path):
    s3 = _S3()
    backend = EksJobBackend(batch_api=_Batch(), core_api=_Core(), blob_client=s3, settings=_settings())

    with pytest.raises(SandboxRuntimeError, match="generic sandbox creation is disabled"):
        await backend.create_sandbox(
            SandboxConfig(project_id="proj", run_id="run", project_root=tmp_path, image="unused")
        )

    assert s3.objects == {}


@pytest.mark.asyncio
async def test_generic_exec_blocks_foreign_gpu_plan_before_any_job(tmp_path: Path):
    batch = _Batch()
    backend = EksJobBackend(
        batch_api=batch,
        core_api=_Core(),
        blob_client=object(),
        settings=_settings(aws_gpu_skus=["eks-a100"]),
        gpu_plan={"short_name": "rtx4090", "gpu_count": 1},
    )

    result = await backend.exec(_sandbox(tmp_path), "echo hello", timeout=10)

    assert result.exit_code == 1
    assert batch.created == []


@pytest.mark.asyncio
async def test_copy_in_and_out_use_injected_s3_without_a_socket(tmp_path: Path):
    s3 = _S3()
    backend = EksJobBackend(batch_api=_Batch(), core_api=_Core(), blob_client=s3, settings=_settings())
    sandbox = _sandbox(tmp_path)

    await backend.copy_in(sandbox, "/metrics.json", b'{"loss": 1}')

    assert await backend.copy_out(sandbox, "/metrics.json") == b'{"loss": 1}'
    assert ("reprolab-artifacts", "projects/proj/runs/run/artifacts/metrics.json") in s3.objects


def test_aws_sandbox_selects_eks_backend_without_contacting_aws():
    with patch("backend.services.runtime.ensure_aws_available", lambda: None):
        from backend.agents.execution import SandboxMode
        from backend.agents.rlm.primitives import _backend_for_sandbox_mode

        backend = _backend_for_sandbox_mode(SandboxMode.aws, run_budget=None)
    assert isinstance(backend, EksJobBackend)


def test_preflight_reports_missing_boto3_without_network(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)  # type: ignore[arg-type]
    with pytest.raises(SandboxRuntimeError, match="boto3"):
        ensure_aws_available()


def test_preflight_rejects_empty_default_settings_without_loading_kubeconfig():
    fake_boto3 = types.ModuleType("boto3")
    fake_kubernetes = types.ModuleType("kubernetes")
    with patch.dict(sys.modules, {"boto3": fake_boto3, "kubernetes": fake_kubernetes}):
        with patch("backend.config.get_settings", return_value=_settings(aws_s3_bucket="")):
            with pytest.raises(SandboxRuntimeError, match="AWS_S3_BUCKET"):
                ensure_aws_available()


def test_preflight_rejects_multi_gpu_nodes_before_loading_kubeconfig():
    fake_boto3 = types.ModuleType("boto3")
    fake_kubernetes = types.ModuleType("kubernetes")
    with patch.dict(sys.modules, {"boto3": fake_boto3, "kubernetes": fake_kubernetes}):
        with patch("backend.config.get_settings", return_value=_settings(aws_gpus_per_node=2)):
            with pytest.raises(SandboxRuntimeError, match="must equal 1"):
                ensure_aws_available()


def test_remote_readiness_uses_injected_read_only_clients_and_checks_irsa(monkeypatch):
    class _Core:
        def read_namespaced_service_account(self, *, name: str, namespace: str):
            assert (namespace, name) == ("reprolab", "reprolab-sa")
            return SimpleNamespace(metadata=SimpleNamespace(annotations={
                "eks.amazonaws.com/role-arn": "arn:aws:iam::123456789012:role/reprolab-cell",
            }))

    class _Sts:
        def get_caller_identity(self):
            return {"Account": "123456789012", "Arn": "arn:aws:sts::123456789012:assumed-role/controller"}

    class _S3Head:
        def head_bucket(self, *, Bucket: str):  # noqa: N803
            assert Bucket == "reprolab-artifacts"

    monkeypatch.setattr(
        "backend.services.runtime.eks_job_backend._verify_kube_context_cluster", lambda cluster: cluster
    )
    result = verify_aws_remote_readiness(
        settings=_settings(), core_api=_Core(), sts_client=_Sts(), s3_client=_S3Head(),
    )

    assert result["irsa_role_arn"].endswith(":role/reprolab-cell")
    assert result["sts_account"] == "123456789012"


def test_pod_readiness_proves_irsa_scope_with_no_gpu_or_static_credentials(monkeypatch):
    class _ProbeBatch(_Batch):
        def __init__(self) -> None:
            super().__init__()
            self.deleted: list[dict] = []

        def delete_namespaced_job(self, *, name: str, namespace: str, body=None):
            self.deleted.append({"name": name, "namespace": namespace, "body": body})

    class _ProbeCore(_Core):
        def read_namespaced_service_account(self, *, name: str, namespace: str):
            assert (namespace, name) == ("reprolab", "reprolab-sa")
            return SimpleNamespace(metadata=SimpleNamespace(annotations={
                "eks.amazonaws.com/role-arn": "arn:aws:iam::123456789012:role/reprolab-cell",
            }))

        def read_namespaced_pod_log(self, *, name: str, namespace: str, _preload_content=True):
            return (
                'some application output\nREPROLAB_IRSA_PROBE='
                '{"account":"123456789012","arn":"arn:aws:sts::123456789012:'
                'assumed-role/reprolab-cell/probe","key":"'
                'projects/project-1/runs/run-1/preflight/placeholder"}'
            )

    batch = _ProbeBatch()
    monkeypatch.setattr(
        "backend.services.runtime.eks_job_backend._verify_kube_context_cluster", lambda cluster: cluster
    )
    # Bind the random preflight key in the fake Pod log to the actual submitted
    # manifest, just as the in-cluster script does.
    class _BoundProbeCore(_ProbeCore):
        def read_namespaced_pod_log(self, *, name: str, namespace: str, _preload_content=True):
            key = batch.created[0]["body"]["spec"]["template"]["spec"]["containers"][0]["env"][1]["value"]
            return (
                "REPROLAB_IRSA_PROBE="
                + '{"account":"123456789012","arn":"arn:aws:sts::123456789012:'
                + 'assumed-role/reprolab-cell/probe","key":"' + key + '"}'
            )

    result = verify_aws_pod_readiness(
        project_id="project-1",
        run_id="run-1",
        settings=_settings(),
        batch_api=batch,
        core_api=_BoundProbeCore(),
    )

    assert result["s3_probe_key"].startswith("projects/project-1/runs/run-1/preflight/")
    assert result["pod_sts_arn"].endswith("/probe")
    manifest = batch.created[0]["body"]
    pod_spec = manifest["spec"]["template"]["spec"]
    assert pod_spec["serviceAccountName"] == "reprolab-sa"
    container = pod_spec["containers"][0]
    assert "nvidia.com/gpu" not in container["resources"]["requests"]
    assert "nvidia.com/gpu" not in container["resources"]["limits"]
    assert {item["name"] for item in container["env"]} == {
        "OPENRESEARCH_AWS_PROBE_BUCKET", "OPENRESEARCH_AWS_PROBE_KEY",
        "AWS_EC2_METADATA_DISABLED", "AWS_REGION", "AWS_DEFAULT_REGION",
    }
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["AWS_EC2_METADATA_DISABLED"] == "true"
    assert env["AWS_REGION"] == env["AWS_DEFAULT_REGION"] == "us-west-2"
    assert batch.deleted == [{
        "name": manifest["metadata"]["name"],
        "namespace": "reprolab",
        "body": {"propagationPolicy": "Foreground"},
    }]


def test_pod_readiness_fails_closed_when_the_probe_job_fails(monkeypatch):
    class _FailedBatch(_Batch):
        def read_namespaced_job_status(self, *, name: str, namespace: str):
            condition = SimpleNamespace(type="Failed", status="True")
            return SimpleNamespace(status=SimpleNamespace(conditions=[condition]))

    class _ProbeCore(_Core):
        def read_namespaced_service_account(self, *, name: str, namespace: str):
            return SimpleNamespace(metadata=SimpleNamespace(annotations={
                "eks.amazonaws.com/role-arn": "arn:aws:iam::123456789012:role/reprolab-cell",
            }))

    monkeypatch.setattr(
        "backend.services.runtime.eks_job_backend._verify_kube_context_cluster", lambda cluster: cluster
    )
    with pytest.raises(SandboxRuntimeError, match="IRSA probe Job.*failed"):
        verify_aws_pod_readiness(
            project_id="project-1",
            run_id="run-1",
            settings=_settings(),
            batch_api=_FailedBatch(),
            core_api=_ProbeCore(),
        )


def test_pod_readiness_rejects_completed_job_without_bound_irsa_log(monkeypatch):
    class _ProbeCore(_Core):
        def read_namespaced_service_account(self, *, name: str, namespace: str):
            return SimpleNamespace(metadata=SimpleNamespace(annotations={
                "eks.amazonaws.com/role-arn": "arn:aws:iam::123456789012:role/reprolab-cell",
            }))

    monkeypatch.setattr(
        "backend.services.runtime.eks_job_backend._verify_kube_context_cluster", lambda cluster: cluster
    )
    with pytest.raises(SandboxRuntimeError, match="without a valid IRSA identity proof"):
        verify_aws_pod_readiness(
            project_id="project-1",
            run_id="run-1",
            settings=_settings(),
            batch_api=_Batch(),
            core_api=_ProbeCore(),
        )
