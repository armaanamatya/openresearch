"""Hermetic EKS cell-matrix plumbing tests: S3, IRSA env boundary, and caps."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import backend.agents.rlm.k8s_job_cell_runner as runner


def _settings(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "aws_namespace": "reprolab",
        "aws_service_account": "reprolab-eks-cell",
        "aws_base_image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/cell@sha256:abc",
        "aws_s3_bucket": "reprolab-artifacts",
        "aws_region": "us-west-2",
        "aws_max_nodes": 2,
        "aws_gpus_per_node": 1,
        "aws_per_gpu_vram_gb": 80.0,
        "aws_gpu_usd_per_hour": 3.25,
        "aws_pending_timeout_seconds": 1,
        "aws_gpu_skus": ["eks-a100-80"],
        "aws_ttl_seconds_after_finished": 60,
        "aws_job_backoff_limit": 0,
        "aws_files_cache_enabled": False,
        "aws_files_share": "",
        "aws_cache_mount_path": "/mnt/reprolab-cache",
        "aws_watch_poll_interval_s": 0.001,
        "aws_cell_oom_batch_scale_step1": 0.5,
        "aws_cell_oom_batch_scale_floor": 0.25,
        "aws_bootstrap_pip_timeout_s": 60,
        "dynamic_gpu_max_escalations": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _manifest(monkeypatch, **overrides: Any) -> dict[str, Any]:
    monkeypatch.setattr(runner, "_get_settings", lambda: _settings(**overrides))
    with runner._bind_settings_prefix("aws"):
        return runner._build_job_manifest(
            job_name="eks-cell-a",
            namespace="reprolab",
            service_account="reprolab-eks-cell",
            node_pool_name="ignored-label-is-authority",
            base_image="registry/cell@sha256:abc",
            storage_account="reprolab-artifacts",
            blob_container="",
            files_share="",
            cell_id="cell-a",
            cell_params_json="{}",
            output_blob_prefix="projects/project-a/runs/run-a/cells",
            code_blob_prefix="projects/project-a/runs/run-a/code",
            active_deadline_seconds=120,
            max_oom_retries=0,
            fingerprint=None,
            now_iso=None,
            gpu_plan=SimpleNamespace(short_name="eks-a100-80", gpu_count=1),
            pod_template_extra_labels={},
            files_cache_enabled=False,
        )


def test_eks_manifest_uses_s3_irsa_boundary_and_never_leaks_static_env(monkeypatch):
    # These deliberately mimic a developer shell. The explicit EKS allow-list
    # must keep them out of the pod even when they are set locally.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "do-not-copy")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "do-not-copy")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "do-not-copy")
    manifest = _manifest(monkeypatch)
    pod = manifest["spec"]["template"]["spec"]
    env = {item["name"]: item["value"] for item in pod["containers"][0]["env"]}

    assert env["OPENRESEARCH_AWS_S3_BUCKET"] == "reprolab-artifacts"
    assert env["AWS_REGION"] == "us-west-2"
    assert env["AWS_DEFAULT_REGION"] == "us-west-2"
    assert env["AWS_EC2_METADATA_DISABLED"] == "true"
    assert {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}.isdisjoint(env)
    assert "OPENRESEARCH_GCP_GCS_BUCKET" not in env
    assert "OPENRESEARCH_AZURE_STORAGE_ACCOUNT" not in env
    assert pod["serviceAccountName"] == "reprolab-eks-cell"
    assert pod["nodeSelector"] == {"reprolab/sku": "eks-a100-80"}
    assert pod["volumes"] == [{"name": "reprolab-cache", "emptyDir": {}}]
    assert "azure.workload.identity/use" not in manifest["spec"]["template"]["metadata"]["labels"]


def test_eks_manifest_rejects_selector_outside_declared_pool(monkeypatch):
    monkeypatch.setattr(runner, "_get_settings", lambda: _settings())
    with runner._bind_settings_prefix("aws"):
        try:
            runner._build_job_manifest(
                job_name="eks-cell-b", namespace="reprolab", service_account="sa",
                node_pool_name="unused", base_image="registry/cell@sha256:abc",
                storage_account="bucket", blob_container="", files_share="", cell_id="c",
                cell_params_json="{}", output_blob_prefix="o", code_blob_prefix="c",
                active_deadline_seconds=1, max_oom_retries=0, fingerprint=None, now_iso=None,
                gpu_plan=SimpleNamespace(short_name="rtx4090", gpu_count=1),
                pod_template_extra_labels={}, files_cache_enabled=False,
            )
        except ValueError as exc:
            assert "aws_gpu_skus" in str(exc)
        else:  # pragma: no cover - assertion form keeps failure message useful
            raise AssertionError("foreign GPU selector must be rejected")


class _Batch:
    def __init__(self) -> None:
        self.manifests: list[dict[str, Any]] = []

    def create_namespaced_job(self, namespace: str, body: dict[str, Any]) -> None:
        self.manifests.append(body)

    def read_namespaced_job_status(self, _name: str, _namespace: str) -> Any:
        return SimpleNamespace(status=SimpleNamespace(
            conditions=[SimpleNamespace(type="Complete", status="True")], succeeded=1, failed=0, active=0,
        ))


class _Core:
    def list_namespaced_pod(self, _namespace: str, label_selector: str) -> Any:
        term = SimpleNamespace(exit_code=0)
        pod = SimpleNamespace(
            metadata=SimpleNamespace(name="done"),
            spec=SimpleNamespace(node_name="ip-10-0-0-1"),
            status=SimpleNamespace(phase="Succeeded", container_statuses=[
                SimpleNamespace(state=SimpleNamespace(terminated=term))
            ]),
        )
        return SimpleNamespace(items=[pod])

    def read_namespaced_pod_log(self, _name: str, _namespace: str, container: str) -> str:
        return "done"


class _Store:
    def __init__(self) -> None:
        self.upload_prefixes: list[str] = []

    def upload_prefix(self, _root: Path, *, blob_prefix: str) -> list[str]:
        self.upload_prefixes.append(blob_prefix)
        return [f"{blob_prefix}/train_cell.py"]

    def download_bytes(self, name: str) -> bytes:
        return b'{"status":"ok"}' if name.endswith("metrics.json") else b"done"

    def download_artifact(self, _name: str, destination: Path) -> Path:
        return destination


def test_eks_matrix_uses_project_and_run_scoped_s3_prefix(monkeypatch, tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    script = code / "train_cell.py"
    script.write_text("# trainer\n")
    store = _Store()
    batch = _Batch()
    monkeypatch.setattr(runner, "_get_settings", lambda: _settings())
    monkeypatch.setattr(runner, "_object_store", lambda: store)
    monkeypatch.setattr(runner, "_k8s_factory", lambda: runner._K8sClients(batch, _Core(), object))

    with runner._bind_settings_prefix("aws"), runner._bind_project_id("project-a"), runner.bind_run_context(
        gpu_plan=SimpleNamespace(short_name="eks-a100-80", gpu_count=1),
    ):
        result = runner.run_matrix(
            [{"id": "cell-a"}], script, output_root=tmp_path / "outputs" / "run-a",
            per_cell_timeout_s=5, overall_timeout_s=5,
        )

    assert result["cell-a"]["status"] == "ok"
    assert store.upload_prefixes == ["projects/project-a/runs/run-a/code"]
    env = {e["name"]: e["value"] for e in batch.manifests[0]["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["OPENRESEARCH_BLOB_OUTPUT_PREFIX"] == "projects/project-a/runs/run-a/cells"
