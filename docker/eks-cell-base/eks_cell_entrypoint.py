"""EKS single-cell entrypoint using S3 + IRSA, with the GKE cell contract.

The training/OOM/status contract is intentionally shared with the well-tested
GKE wrapper.  This thin adapter supplies a GCS-shaped, in-memory-free S3 object
store facade so the shared wrapper can stage code and upload metrics without
duplicating its safety-critical retry and preemption logic.

Only the normal boto3 default credential chain is used.  In EKS that means the
Kubernetes ServiceAccount's IRSA web identity; this file never reads or injects
static AWS credential values.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)


class _S3Blob:
    """The small GCS Blob-shaped interface consumed by gke_cell_entrypoint."""

    def __init__(self, client: Any, bucket: str, name: str) -> None:
        self._client = client
        self._bucket = bucket
        self.name = name

    def upload_from_string(self, data: bytes | str) -> None:
        body = data.encode("utf-8") if isinstance(data, str) else data
        self._client.put_object(Bucket=self._bucket, Key=self.name, Body=body)

    def download_as_bytes(self) -> bytes:
        return self._client.get_object(Bucket=self._bucket, Key=self.name)["Body"].read()


class S3Bucket:
    """Minimal GCS Bucket facade over an injected or lazily-created boto3 client."""

    def __init__(self, bucket_name: str, client: Any | None = None, region: str | None = None) -> None:
        if not bucket_name.strip():
            raise ValueError("OPENRESEARCH_AWS_S3_BUCKET must be non-empty")
        self._bucket_name = bucket_name
        self._client = client if client is not None else self._make_client(region)

    @staticmethod
    def _make_client(region: str | None) -> Any:
        try:
            import boto3  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("boto3 is required in the EKS cell image") from exc
        return boto3.client("s3", region_name=region or None)

    def blob(self, name: str) -> _S3Blob:
        return _S3Blob(self._client, self._bucket_name, name)

    def list_blobs(self, *, prefix: str) -> list[_S3Blob]:
        """Return every key under *prefix*, correctly handling S3 pagination."""
        result: list[_S3Blob] = []
        token: str | None = None
        while True:
            request: dict[str, str] = {"Bucket": self._bucket_name, "Prefix": prefix}
            if token:
                request["ContinuationToken"] = token
            page = self._client.list_objects_v2(**request)
            for item in page.get("Contents", []) or []:
                key = item.get("Key")
                if isinstance(key, str):
                    result.append(self.blob(key))
            if not page.get("IsTruncated"):
                return result
            token = page.get("NextContinuationToken")
            if not token:
                raise RuntimeError("S3 list_objects_v2 returned truncated response without continuation token")


def _load_shared_entrypoint() -> ModuleType:
    path = Path(__file__).with_name("gke_cell_entrypoint.py")
    # In the built image both entrypoints live together under /opt/reprolab.
    # In the source tree the reusable entrypoint stays in its GKE image folder,
    # so tests can import this adapter without first building an image.
    if not path.is_file():
        path = Path(__file__).resolve().parent.parent / "gke-cell-base" / "gke_cell_entrypoint.py"
    spec = importlib.util.spec_from_file_location("reprolab_shared_cell_entrypoint", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared cell entrypoint at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(
    *,
    s3_client: Any | None = None,
    subprocess_runner: Any | None = None,
    _preempt_upload_fn: Any | None = None,
) -> int:
    """Run the shared cell protocol using the EKS S3/IRSA artifact bus.

    Injectable arguments are deliberately test-only seams: passing ``s3_client``
    avoids importing boto3 and opening sockets in hermetic tests.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [eks-cell] %(levelname)s %(message)s")
    bucket_name = os.environ.get("OPENRESEARCH_AWS_S3_BUCKET", "").strip()
    if not bucket_name:
        logger.error("EKS cell bootstrap blocked: OPENRESEARCH_AWS_S3_BUCKET is empty")
        return 40

    shared = _load_shared_entrypoint()
    bucket = S3Bucket(
        bucket_name,
        client=s3_client,
        region=os.environ.get("AWS_REGION") or None,
    )

    # The shared GKE wrapper routes every object operation through this factory.
    # Patching it also covers its SIGTERM preemption flush, whose call site uses
    # ``client=None`` by design.
    shared._gcs_bucket_client = lambda *, bucket_name, project, client: client or bucket  # type: ignore[attr-defined]

    return int(shared.main(
        gcs_client=bucket,
        subprocess_runner=subprocess_runner,
        _preempt_upload_fn=_preempt_upload_fn,
        bucket_name_override=bucket_name,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
