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

from backend.services.runtime.eks_job_backend import EksJobBackend, ensure_aws_available
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
        aws_gpu_skus=[],
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
async def test_exec_uses_eks_label_and_injected_clients(tmp_path: Path):
    batch = _Batch()
    backend = EksJobBackend(batch_api=batch, core_api=_Core(), blob_client=object(), settings=_settings())

    result = await backend.exec(_sandbox(tmp_path), "echo hello", timeout=10)

    assert result.exit_code == 0
    manifest = batch.created[0]["body"]
    assert manifest["metadata"]["labels"]["reprolab/sandbox"] == "eks"
    assert manifest["spec"]["template"]["spec"]["serviceAccountName"] == "reprolab-sa"


@pytest.mark.asyncio
async def test_exec_never_uses_an_unconfigured_foreign_gpu_plan_label(tmp_path: Path):
    batch = _Batch()
    backend = EksJobBackend(
        batch_api=batch,
        core_api=_Core(),
        blob_client=object(),
        settings=_settings(aws_gpu_skus=["eks-a100"]),
        gpu_plan={"short_name": "rtx4090", "gpu_count": 1},
    )

    await backend.exec(_sandbox(tmp_path), "echo hello", timeout=10)

    selector = batch.created[0]["body"]["spec"]["template"]["spec"]["nodeSelector"]
    assert selector == {"reprolab/sku": "eks-a100"}


@pytest.mark.asyncio
async def test_copy_in_and_out_use_injected_s3_without_a_socket(tmp_path: Path):
    s3 = _S3()
    backend = EksJobBackend(batch_api=_Batch(), core_api=_Core(), blob_client=s3, settings=_settings())
    sandbox = _sandbox(tmp_path)

    await backend.copy_in(sandbox, "/metrics.json", b'{"loss": 1}')

    assert await backend.copy_out(sandbox, "/metrics.json") == b'{"loss": 1}'
    assert ("reprolab-artifacts", "runs/run/artifacts/metrics.json") in s3.objects


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
