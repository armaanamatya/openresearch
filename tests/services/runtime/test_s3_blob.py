"""Hermetic tests for the lazy S3 transfer helpers."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from backend.services.runtime import s3_blob


class FakeS3:
    """In-memory duck type for the exact boto3 operations used by s3_blob."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes):  # noqa: N803
        self.objects[(Bucket, Key)] = Body
        return {}

    def get_object(self, *, Bucket: str, Key: str):  # noqa: N803
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


def test_import_needs_no_boto3(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)  # type: ignore[arg-type]
    monkeypatch.delitem(sys.modules, "backend.services.runtime.s3_blob", raising=False)
    import importlib

    module = importlib.import_module("backend.services.runtime.s3_blob")
    assert hasattr(module, "upload_prefix")


def test_prefix_upload_filters_generated_and_escaping_paths(tmp_path: Path):
    (tmp_path / "train.py").write_text("print('ok')")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "metric.json").write_text("{}")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "secret").write_text("nope")
    outside = tmp_path.parent / "s3-blob-outside-secret"
    outside.write_text("secret")
    (tmp_path / "escape").symlink_to(outside)

    fake = FakeS3()
    keys = s3_blob.upload_prefix(
        tmp_path, blob_prefix="runs/one/code", bucket="bucket", client=fake
    )

    assert keys == ["runs/one/code/train.py"]
    assert fake.objects[("bucket", "runs/one/code/train.py")] == b"print('ok')"


def test_bytes_round_trip_and_directory_destination(tmp_path: Path):
    fake = FakeS3()
    s3_blob.upload_bytes(b"artifact", blob_name="runs/a/artifacts/out.txt", bucket="b", client=fake)

    assert s3_blob.download_bytes("runs/a/artifacts/out.txt", bucket="b", client=fake) == b"artifact"
    path = s3_blob.download_artifact("runs/a/artifacts/out.txt", tmp_path, bucket="b", client=fake)
    assert path == tmp_path / "out.txt"
    assert path.read_bytes() == b"artifact"


@pytest.mark.parametrize("key", ["", "/absolute", "runs/../secret"])
def test_rejects_unsafe_keys(key: str):
    with pytest.raises(ValueError):
        s3_blob.upload_bytes(b"x", blob_name=key, bucket="b", client=FakeS3())
