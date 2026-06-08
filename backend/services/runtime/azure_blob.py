"""Thin Azure Blob Storage helpers for the AKS GPU backend.

Provides four path-safe, authenticated transfer helpers used by both the local
orchestrator (upload code, download artifacts) and the in-Job entrypoint
wrapper (push metrics/logs, pull code).

Auth model
----------
When the caller supplies no ``client``, a ``ContainerClient`` is constructed
lazily from ``DefaultAzureCredential`` — workload-identity inside the AKS pod,
``az login`` on the operator's laptop.  In tests, pass a ``FakeContainerClient``
(or any duck-typed object matching the shape below) to avoid importing the real
Azure SDK at all.

Duck-type shape expected of an injected ``client``
---------------------------------------------------
The injected object must implement::

    client.upload_blob(name: str, data: bytes, overwrite: bool) -> None
    client.download_blob(name: str) -> object  # object has .readall() -> bytes
    client.get_blob_client(name: str) -> object  # optional — not used internally

Both ``upload_blob`` and ``download_blob`` accept the full blob *name* (i.e.
the path within the container) as the first positional argument.

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
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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
# Client factory (lazy import — azure SDK is optional at import time)
# ---------------------------------------------------------------------------

def _make_container_client(account_name: str, container_name: str) -> Any:
    """Build a ``ContainerClient`` using ``DefaultAzureCredential``.

    The azure-identity and azure-storage-blob packages are imported *here*,
    inside the function, so that module import succeeds even when neither
    package is installed.  Tests never call this function (they supply a fake
    client instead).
    """
    try:
        from azure.identity import DefaultAzureCredential  # type: ignore[import]
        from azure.storage.blob import ContainerClient  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "azure-identity and azure-storage-blob must be installed to use "
            "the Azure Blob helpers without an injected client.  "
            "Run: pip install azure-identity azure-storage-blob"
        ) from exc

    account_url = f"https://{account_name}.blob.core.windows.net"
    credential = DefaultAzureCredential()
    logger.debug(
        "Building ContainerClient for %s/%s", account_name, container_name
    )
    return ContainerClient(
        account_url=account_url,
        container_name=container_name,
        credential=credential,
    )


def _client_or_new(
    client: Any | None, account_name: str, container_name: str
) -> Any:
    """Return *client* if provided, otherwise build one lazily."""
    if client is not None:
        return client
    return _make_container_client(account_name, container_name)


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
    account_name: str,
    container_name: str,
    client: Any | None = None,
) -> list[str]:
    """Walk *local_root* recursively and upload each eligible file to Blob.

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
    account_name:
        Azure storage account name (used when *client* is ``None``).
    container_name:
        Blob container name (used when *client* is ``None``).
    client:
        Optional pre-built duck-typed ``ContainerClient``.  When ``None`` a
        real client is constructed via ``DefaultAzureCredential``.

    Returns
    -------
    list[str]
        Sorted list of blob names that were actually uploaded.
    """
    _validate_blob_name(blob_prefix)
    local_root = Path(local_root).resolve()
    if not local_root.is_dir():
        raise ValueError(f"local_root is not a directory: {local_root}")

    container = _client_or_new(client, account_name, container_name)

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
        container.upload_blob(blob_name, data, overwrite=True)
        return blob_name

    # Fan out uploads with a bounded thread pool.  azure ContainerClient is
    # thread-safe for independent blob uploads; FakeContainerClient dict writes
    # are GIL-protected and keyed independently.
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
    account_name: str,
    container_name: str,
    client: Any | None = None,
) -> None:
    """Upload raw *data* to a single blob.

    Uses ``upload_blob(..., overwrite=True)`` so repeated calls are idempotent.

    Parameters
    ----------
    data:
        The bytes to upload.
    blob_name:
        Destination blob path within the container.  Must not start with ``/``
        or contain ``..`` components.
    account_name:
        Azure storage account name (used when *client* is ``None``).
    container_name:
        Blob container name (used when *client* is ``None``).
    client:
        Optional pre-built duck-typed ``ContainerClient``.
    """
    _validate_blob_name(blob_name)
    container = _client_or_new(client, account_name, container_name)
    logger.debug("upload_bytes -> %s (%d bytes)", blob_name, len(data))
    container.upload_blob(blob_name, data, overwrite=True)


def download_artifact(
    blob_name: str,
    destination: str | Path,
    *,
    account_name: str,
    container_name: str,
    client: Any | None = None,
) -> Path:
    """Download a single blob to a local *destination* path.

    Parent directories are created if they do not exist.

    Parameters
    ----------
    blob_name:
        Source blob path within the container.
    destination:
        Local filesystem path to write.  If a directory is passed the file is
        written **into** that directory using the blob's filename component.
    account_name:
        Azure storage account name (used when *client* is ``None``).
    container_name:
        Blob container name (used when *client* is ``None``).
    client:
        Optional pre-built duck-typed ``ContainerClient``.

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

    container = _client_or_new(client, account_name, container_name)
    logger.debug("download_artifact %s -> %s", blob_name, destination)
    raw = container.download_blob(blob_name).readall()
    destination.write_bytes(raw)
    return destination.resolve()


def download_bytes(
    blob_name: str,
    *,
    account_name: str,
    container_name: str,
    client: Any | None = None,
) -> bytes:
    """Download a blob and return its contents as bytes.

    Parameters
    ----------
    blob_name:
        Source blob path within the container.
    account_name:
        Azure storage account name (used when *client* is ``None``).
    container_name:
        Blob container name (used when *client* is ``None``).
    client:
        Optional pre-built duck-typed ``ContainerClient``.

    Returns
    -------
    bytes
        Raw blob contents.
    """
    _validate_blob_name(blob_name)
    container = _client_or_new(client, account_name, container_name)
    logger.debug("download_bytes <- %s", blob_name)
    return container.download_blob(blob_name).readall()
