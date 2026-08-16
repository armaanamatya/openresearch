"""Lazy, path-safe S3 transfer helpers for the EKS runtime backend.

The module is deliberately dependency-free at import time.  ``boto3`` is
imported only when a caller has not injected an S3-compatible client, keeping
unit tests socket-hermetic and making the AWS backend opt-in.

Injected clients implement the small subset used here::

    client.put_object(Bucket: str, Key: str, Body: bytes) -> object
    client.get_object(Bucket: str, Key: str) -> {"Body": stream}

where ``stream.read()`` returns bytes.  This shape is also compatible with
botocore's normal S3 client.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

__all__ = ["upload_prefix", "upload_bytes", "download_artifact", "download_bytes"]

logger = logging.getLogger(__name__)

# Keep code uploads free of generated output and local environment state.  The
# list matches GCS/Azure helpers so a provider switch cannot change what code is
# executed remotely.
_EXCLUDED_DIR_PARTS: frozenset[str] = frozenset(
    {"outputs", ".git", "__pycache__", ".venv", "repo"}
)


def _validate_blob_name(blob_name: str) -> str:
    """Return a safe S3 key or raise ``ValueError`` for traversal-like keys."""
    if not blob_name:
        raise ValueError("blob_name must not be empty")
    if blob_name.startswith("/"):
        raise ValueError(f"blob_name must not start with '/': {blob_name!r}")
    if ".." in blob_name.replace("\\", "/").split("/"):
        raise ValueError(f"blob_name must not contain '..': {blob_name!r}")
    return blob_name


def _make_s3_client(region: str | None = None) -> Any:
    """Construct a boto3 S3 client lazily, without doing data-plane I/O."""
    try:
        import boto3  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "boto3 must be installed to use the S3 helpers without an injected "
            "client. Run: pip install boto3"
        ) from exc
    return boto3.client("s3", region_name=region or None)


def _client_or_new(client: Any | None, region: str | None) -> Any:
    return client if client is not None else _make_s3_client(region)


def _is_excluded(rel_parts: tuple[str, ...]) -> bool:
    return any(part in _EXCLUDED_DIR_PARTS for part in rel_parts[:-1])


def _symlink_escapes(path: Path, local_root: Path) -> bool:
    if not path.is_symlink():
        return False
    try:
        path.resolve().relative_to(local_root.resolve())
        return False
    except ValueError:
        return True


def upload_prefix(
    local_root: str | Path,
    *,
    blob_prefix: str,
    bucket: str,
    region: str | None = None,
    client: Any | None = None,
) -> list[str]:
    """Upload eligible files below *local_root* to ``bucket/blob_prefix``.

    The returned key list is stable and sorted.  It excludes generated outputs,
    VCS/virtualenv state, bytecode, and symlinks escaping *local_root*.
    """
    _validate_blob_name(blob_prefix)
    root = Path(local_root).resolve()
    if not root.is_dir():
        raise ValueError(f"local_root is not a directory: {root}")
    s3 = _client_or_new(client, region)

    eligible: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        if _symlink_escapes(path, root) or not path.exists():
            continue
        relative = path.relative_to(root)
        if relative.suffix == ".pyc" or _is_excluded(relative.parts):
            continue
        eligible.append((path, f"{blob_prefix}/{relative.as_posix()}"))

    def _upload(item: tuple[Path, str]) -> str:
        path, key = item
        logger.debug("Uploading %s -> s3://%s/%s", path, bucket, key)
        s3.put_object(Bucket=bucket, Key=key, Body=path.read_bytes())
        return key

    if not eligible:
        return []
    with ThreadPoolExecutor(max_workers=min(16, len(eligible))) as executor:
        uploaded = list(executor.map(_upload, eligible))
    return sorted(uploaded)


def upload_bytes(
    data: bytes,
    *,
    blob_name: str,
    bucket: str,
    region: str | None = None,
    client: Any | None = None,
) -> None:
    """Idempotently write raw bytes to a validated key in S3."""
    _validate_blob_name(blob_name)
    _client_or_new(client, region).put_object(Bucket=bucket, Key=blob_name, Body=data)


def download_bytes(
    blob_name: str,
    *,
    bucket: str,
    region: str | None = None,
    client: Any | None = None,
) -> bytes:
    """Read one validated S3 object fully into bytes."""
    _validate_blob_name(blob_name)
    response = _client_or_new(client, region).get_object(Bucket=bucket, Key=blob_name)
    body = response["Body"]
    return body.read()


def download_artifact(
    blob_name: str,
    destination: str | Path,
    *,
    bucket: str,
    region: str | None = None,
    client: Any | None = None,
) -> Path:
    """Download one artifact into a file or an existing destination directory."""
    _validate_blob_name(blob_name)
    target = Path(destination)
    if target.exists() and target.is_dir():
        target = target / Path(blob_name.replace("\\", "/")).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(
        download_bytes(blob_name, bucket=bucket, region=region, client=client)
    )
    return target
