"""Hermetic EKS S3/IRSA cell-entrypoint tests (no boto3, GPU, or network)."""
from __future__ import annotations

import importlib.util
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any


_ENTRYPOINT = (
    Path(__file__).parent.parent.parent.parent / "docker" / "eks-cell-base" / "eks_cell_entrypoint.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("eks_cell_entrypoint_test", _ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _S3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.list_requests: list[dict[str, str]] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict[str, str]:  # noqa: N803
        assert Bucket == "bucket"
        self.objects[Key] = Body
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, BytesIO]:  # noqa: N803
        assert Bucket == "bucket"
        return {"Body": BytesIO(self.objects[Key])}

    def list_objects_v2(self, **request: str) -> dict[str, Any]:
        self.list_requests.append(request)
        prefix = request["Prefix"]
        if request.get("ContinuationToken") is None:
            keys = sorted(k for k in self.objects if k.startswith(prefix))[:1]
            remaining = sorted(k for k in self.objects if k.startswith(prefix))[1:]
            return {
                "Contents": [{"Key": k} for k in keys],
                "IsTruncated": bool(remaining),
                "NextContinuationToken": "next" if remaining else None,
            }
        keys = sorted(k for k in self.objects if k.startswith(prefix))[1:]
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}


def test_s3_bucket_round_trip_and_pagination_without_boto3():
    ep = _load()
    fake = _S3()
    bucket = ep.S3Bucket("bucket", client=fake)
    bucket.blob("projects/p/runs/a/code/a.py").upload_from_string(b"a")
    bucket.blob("projects/p/runs/a/code/b.py").upload_from_string("b")

    found = bucket.list_blobs(prefix="projects/p/runs/a/code/")

    assert [blob.name for blob in found] == [
        "projects/p/runs/a/code/a.py", "projects/p/runs/a/code/b.py",
    ]
    assert found[1].download_as_bytes() == b"b"
    assert len(fake.list_requests) == 2


def test_missing_bucket_fails_before_loading_boto3(monkeypatch):
    ep = _load()
    monkeypatch.delenv("OPENRESEARCH_AWS_S3_BUCKET", raising=False)
    assert ep.main(s3_client=object()) == 40


def test_main_stages_s3_code_and_uploads_artifacts_with_injected_client(monkeypatch, tmp_path: Path):
    ep = _load()
    fake = _S3()
    code_prefix = "projects/project-a/runs/run-a/code"
    output_prefix = "projects/project-a/runs/run-a/cells"
    fake.objects[f"{code_prefix}/train_cell.py"] = b"# staged trainer\n"
    monkeypatch.setenv("OPENRESEARCH_AWS_S3_BUCKET", "bucket")
    monkeypatch.delenv("OPENRESEARCH_GCP_GCS_BUCKET", raising=False)
    monkeypatch.setenv("OPENRESEARCH_BLOB_CODE_PREFIX", code_prefix)
    monkeypatch.setenv("OPENRESEARCH_BLOB_OUTPUT_PREFIX", output_prefix)
    monkeypatch.setenv("OPENRESEARCH_CELL_ID", "cell-a")
    monkeypatch.setenv("OPENRESEARCH_CACHE_MOUNT", str(tmp_path / "cache"))

    def runner(_train, output_dir, _env, log_path):
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir.joinpath("metrics.json").write_text(json.dumps({"status": "ok"}))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("trained")
        return 0, "trained"

    assert ep.main(s3_client=fake, subprocess_runner=runner) == 0
    assert json.loads(fake.objects[f"{output_prefix}/cell-a/metrics.json"])["status"] == "ok"
    assert json.loads(fake.objects[f"{output_prefix}/cell-a/status.json"])["outcome"] == "ok"
    assert fake.objects[f"{output_prefix}/cell-a/logs/attempt-0.log"] == b"trained"
    assert "OPENRESEARCH_GCP_GCS_BUCKET" not in os.environ
