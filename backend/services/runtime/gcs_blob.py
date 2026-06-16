"""Thin Google Cloud Storage helpers for the GKE GPU backend.

Provides four path-safe, authenticated transfer helpers used by both the local
orchestrator (upload code, download artifacts) and the in-Job entrypoint
wrapper (push metrics/logs, pull code).

Auth model
----------
When the caller supplies no ``client``, a ``Bucket`` handle is constructed
lazily from Application Default Credentials — workload-identity inside the GKE
pod, ``gcloud auth application-default login`` on the operator's laptop.  In
tests, pass a ``FakeBucketClient`` (or any duck-typed object matching the shape
below) to avoid importing the real google-cloud-storage SDK at all.

Duck-type shape expected of an injected ``client``
---------------------------------------------------
The injected object must implement::

    client.blob(name: str) -> object  # object has:
        .upload_from_string(data: bytes) -> None
        .download_as_bytes() -> bytes

Both ``upload_from_string`` and ``download_as_bytes`` operate on the full blob
*name* (i.e. the path within the bucket).

Exclusions applied by ``upload_prefix``
----------------------------------------
The following are **silently skipped** — never uploaded:

* Anything under an ``outputs/`` directory component.
* Anything under ``.git/``.
* Anything under ``__pycache__/``.
* Files with a ``.pyc`` suffix.
* Anything under a ``.venv/`` directory component.
* Symlinks whose resolved target lies outside ``local_root`` (path-safety).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

__all__ = [
    "upload_prefix",
    "upload_bytes",
    "download_artifact",
    "download_bytes",
]

logger = logging.getLogger(__name__)

# Directory-name components that are always excluded from uploads.
_EXCLUDED_DIR_PARTS: frozenset[str] = frozenset(
    {"outputs", ".git", "__pycache__", ".venv"}
)


# ---------------------------------------------------------------------------
# Blob-name validation
# ---------------------------------------------------------------------------

def _validate_blob_name(blob_name: str) -> str:
    """Return *blob_name* unchanged, or raise ``ValueError`` if it is unsafe.

    Rejects names that:
    - start with ``/`` (absolute-path confusion),
    - contain a ``..`` path component (traversal),
    - are empty.
    """
    if not blob_name:
        raise ValueError("blob_name must not be empty")
    if blob_name.startswith("/"):
        raise ValueError(f"blob_name must not start with '/': {blob_name!r}")
    parts = blob_name.replace("\\", "/").split("/")
    if ".." in parts:
        raise ValueError(f"blob_name must not contain '..': {blob_name!r}")
    return blob_name


# ---------------------------------------------------------------------------
# Client factory (lazy import — google-cloud-storage is optional at import time)
# ---------------------------------------------------------------------------

def _make_bucket_client(bucket: str, project: str | None = None) -> Any:
    """Build a GCS ``Bucket`` handle using Application Default Credentials.

    The google-cloud-storage package is imported *here*, inside the function,
    so that module import succeeds even when it is not installed.  Tests never
    call this function (they supply a fake client instead).
    """
    try:
        from google.cloud import storage  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-storage must be installed to use "
            "the GCS helpers without an injected client.  "
            "Run: pip install google-cloud-storage"
        ) from exc

    logger.debug("Building GCS Bucket client for bucket=%s project=%s", bucket, project)
    return storage.Client(project=project).bucket(bucket)


def _client_or_new(
    client: Any | None, bucket: str, project: str | None
) -> Any:
    """Return *client* if provided, otherwise build one lazily."""
    if client is not None:
        return client
    return _make_bucket_client(bucket, project)


# ---------------------------------------------------------------------------
# Path-safety helper for upload_prefix
# ---------------------------------------------------------------------------

def _is_excluded(rel_parts: tuple[str, ...]) -> bool:
    """Return True if any directory component of *rel_parts* is excluded."""
    # All but the last element are directory components.
    for part in rel_parts[:-1]:
        if part in _EXCLUDED_DIR_PARTS:
            return True
    return False


def _symlink_escapes(path: Path, local_root: Path) -> bool:
    """Return True if *path* is a symlink pointing outside *local_root*."""
    if not path.is_symlink():
        return False
    try:
        target = path.resolve()
        local_root_resolved = local_root.resolve()
        # Check that the resolved target is inside local_root.
        target.relative_to(local_root_resolved)
        return False
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_prefix(
    local_root: str | Path,
    *,
    blob_prefix: str,
    bucket: str,
    project: str | None = None,
    client: Any | None = None,
) -> list[str]:
    """Walk *local_root* recursively and upload each eligible file to GCS.

    The blob name for each file is ``<blob_prefix>/<relative-posix-path>``.
    All blob names use forward slashes regardless of the host OS.

    Files are **excluded** when any of the following applies:

    - A directory component of the relative path is in
      ``{outputs, .git, __pycache__, .venv}``.
    - The file has a ``.pyc`` suffix.
    - The path is a symlink whose resolved target escapes *local_root*.

    Parameters
    ----------
    local_root:
        The directory to walk.  Must exist.
    blob_prefix:
        Prefix prepended to every blob name (e.g. ``"runs/abc123/code"``).
        Must be a valid, sanitized blob path component (validated via
        :func:`_validate_blob_name`).
    bucket:
        GCS bucket name (used when *client* is ``None``).
    project:
        GCP project ID (used when *client* is ``None``; optional).
    client:
        Optional pre-built duck-typed ``Bucket``-like object.  When ``None`` a
        real client is constructed via Application Default Credentials.

    Returns
    -------
    list[str]
        Sorted list of blob names that were actually uploaded.
    """
    _validate_blob_name(blob_prefix)
    local_root = Path(local_root).resolve()
    if not local_root.is_dir():
        raise ValueError(f"local_root is not a directory: {local_root}")

    bucket_client = _client_or_new(client, bucket, project)

    # Collect eligible (abs_path, blob_name) pairs up-front before spawning
    # threads, so filtering logic stays serial and deterministic.
    eligible: list[tuple[Path, str]] = []
    for abs_path in sorted(local_root.rglob("*")):
        if not abs_path.is_file() and not abs_path.is_symlink():
            continue  # skip directories themselves

        # Path-safety: skip symlinks escaping local_root.
        if _symlink_escapes(abs_path, local_root):
            logger.debug("Skipping symlink escaping root: %s", abs_path)
            continue

        # Only dereference real files from here; skip broken symlinks.
        if not abs_path.exists():
            continue

        rel = abs_path.relative_to(local_root)
        rel_parts = rel.parts  # tuple of path components

        # Exclude .pyc files.
        if rel.suffix == ".pyc":
            continue

        # Exclude forbidden directory components.
        if _is_excluded(rel_parts):
            continue

        # Build a forward-slash blob name.
        blob_name = f"{blob_prefix}/{rel.as_posix()}"
        eligible.append((abs_path, blob_name))

    if not eligible:
        return []

    def _upload_one(args: tuple[Path, str]) -> str:
        abs_path, blob_name = args
        logger.debug("Uploading %s -> %s", abs_path, blob_name)
        data = abs_path.read_bytes()
        bucket_client.blob(blob_name).upload_from_string(data)
        return blob_name

    # Fan out uploads with a bounded thread pool.  GCS Bucket is thread-safe
    # for independent blob uploads; FakeBucketClient dict writes are
    # GIL-protected and keyed independently.
    # executor.map preserves submission order and re-raises the first exception.
    max_workers = min(16, len(eligible))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        uploaded = list(executor.map(_upload_one, eligible))

    uploaded.sort()
    return uploaded


def upload_bytes(
    data: bytes,
    *,
    blob_name: str,
    bucket: str,
    project: str | None = None,
    client: Any | None = None,
) -> None:
    """Upload raw *data* to a single blob.

    Uses ``upload_from_string`` so repeated calls are idempotent (GCS always
    overwrites on upload).

    Parameters
    ----------
    data:
        The bytes to upload.
    blob_name:
        Destination blob path within the bucket.  Must not start with ``/``
        or contain ``..`` components.
    bucket:
        GCS bucket name (used when *client* is ``None``).
    project:
        GCP project ID (used when *client* is ``None``; optional).
    client:
        Optional pre-built duck-typed ``Bucket``-like object.
    """
    _validate_blob_name(blob_name)
    bucket_client = _client_or_new(client, bucket, project)
    logger.debug("upload_bytes -> %s (%d bytes)", blob_name, len(data))
    bucket_client.blob(blob_name).upload_from_string(data)


def download_artifact(
    blob_name: str,
    destination: str | Path,
    *,
    bucket: str,
    project: str | None = None,
    client: Any | None = None,
) -> Path:
    """Download a single blob to a local *destination* path.

    Parent directories are created if they do not exist.

    Parameters
    ----------
    blob_name:
        Source blob path within the bucket.
    destination:
        Local filesystem path to write.  If a directory is passed the file is
        written **into** that directory using the blob's filename component.
    bucket:
        GCS bucket name (used when *client* is ``None``).
    project:
        GCP project ID (used when *client* is ``None``; optional).
    client:
        Optional pre-built duck-typed ``Bucket``-like object.

    Returns
    -------
    Path
        Absolute path of the file that was written.
    """
    _validate_blob_name(blob_name)
    destination = Path(destination)
    if destination.is_dir():
        # Derive filename from the last component of the blob name.
        destination = destination / Path(blob_name.replace("\\", "/")).name

    destination.parent.mkdir(parents=True, exist_ok=True)

    bucket_client = _client_or_new(client, bucket, project)
    logger.debug("download_artifact %s -> %s", blob_name, destination)
    raw = bucket_client.blob(blob_name).download_as_bytes()
    destination.write_bytes(raw)
    return destination.resolve()


def download_bytes(
    blob_name: str,
    *,
    bucket: str,
    project: str | None = None,
    client: Any | None = None,
) -> bytes:
    """Download a blob and return its contents as bytes.

    Parameters
    ----------
    blob_name:
        Source blob path within the bucket.
    bucket:
        GCS bucket name (used when *client* is ``None``).
    project:
        GCP project ID (used when *client* is ``None``; optional).
    client:
        Optional pre-built duck-typed ``Bucket``-like object.

    Returns
    -------
    bytes
        Raw blob contents.
    """
    _validate_blob_name(blob_name)
    bucket_client = _client_or_new(client, bucket, project)
    logger.debug("download_bytes <- %s", blob_name)
    return bucket_client.blob(blob_name).download_as_bytes()
