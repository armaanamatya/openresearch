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
        .upload_from_string(data: bytes, if_generation_match: int | None = None) -> None
        .download_as_bytes() -> bytes
        .generation: int | None  # populated after upload_from_string/download_as_bytes

Both ``upload_from_string`` and ``download_as_bytes`` operate on the full blob
*name* (i.e. the path within the bucket).

Compare-and-swap (generation preconditions)
--------------------------------------------
``upload_bytes`` accepts an optional ``if_generation_match`` forwarded
verbatim to ``Blob.upload_from_string`` (``0`` = create-only-if-absent, a
positive int = overwrite-only-if-still-at-that-generation).  A failed
precondition — the real SDK raises ``google.api_core.exceptions
.PreconditionFailed`` (HTTP 412) — is caught at this module's boundary and
re-raised as the typed :class:`PreconditionFailedError`, so callers (e.g.
``backend.services.runtime.blob_lease.BlobLease``) never need to import the
google exception hierarchy themselves.  A duck-typed test double must raise
the real ``google.api_core.exceptions.PreconditionFailed``/``NotFound`` (both
lightweight, dependency-free exception classes) to exercise this path — see
``tests/services/runtime/test_gcs_blob.py``'s ``FakeBucketClient`` for the
reference implementation.

``read_bytes_with_generation`` is the CAS-aware counterpart of
``download_bytes``: it returns ``(data, generation)`` for an existing blob,
or ``None`` (never raises) when the blob is absent.

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
    "read_bytes_with_generation",
    "PreconditionFailedError",
]

logger = logging.getLogger(__name__)

# Directory-name components that are always excluded from uploads.
_EXCLUDED_DIR_PARTS: frozenset[str] = frozenset(
    {"outputs", ".git", "__pycache__", ".venv", "repo"}
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
# Compare-and-swap (generation precondition) support
# ---------------------------------------------------------------------------

class PreconditionFailedError(Exception):
    """Raised when a ``if_generation_match`` compare-and-swap write loses a race.

    Normalizes ``google.api_core.exceptions.PreconditionFailed`` (HTTP 412)
    into a type this module's callers can catch without importing the google
    SDK's exception hierarchy — e.g. ``blob_lease.BlobLease`` branches on this
    to return ``None`` (superseded) instead of propagating a raw google
    exception.
    """


def _precondition_failed_exc_types() -> tuple[type[BaseException], ...]:
    """Return the real SDK exception class(es) meaning "CAS precondition failed".

    Imported lazily (same pattern as :func:`_make_bucket_client`) so this
    module stays importable without google-cloud-storage installed.  If the
    import itself fails there is nothing meaningful to catch as a precondition
    failure, so an empty tuple is returned; ``except ():`` matches nothing and
    whatever the caller's client raised propagates unmodified.
    """
    try:
        from google.api_core import exceptions as gcs_exceptions  # type: ignore[import]
    except ImportError:
        return ()
    return (gcs_exceptions.PreconditionFailed,)


def _not_found_exc_types() -> tuple[type[BaseException], ...]:
    """Return the real SDK exception class(es) meaning "blob does not exist".

    Same lazy-import rationale as :func:`_precondition_failed_exc_types`.
    """
    try:
        from google.api_core import exceptions as gcs_exceptions  # type: ignore[import]
    except ImportError:
        return ()
    return (gcs_exceptions.NotFound,)


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
    if_generation_match: int | None = None,
) -> int:
    """Upload raw *data* to a single blob, returning the written generation.

    Uses ``upload_from_string`` so a call with ``if_generation_match=None``
    (the default) always overwrites unconditionally — byte-identical to this
    function's behavior before the CAS parameter existed.

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
    if_generation_match:
        Optional compare-and-swap precondition forwarded to
        ``Blob.upload_from_string`` **only when not None** — ``0`` means
        "create only if the blob is currently absent"; a positive int means
        "overwrite only if the blob's live generation still equals this".
        ``None`` omits the precondition kwarg entirely, so the call and its
        on-the-wire semantics are unchanged from before this parameter
        existed.

    Returns
    -------
    int
        The object's generation as it now exists in the bucket — read back
        off the blob handle (``blob.generation``) immediately after the
        upload call returns.

    Raises
    ------
    PreconditionFailedError
        If *if_generation_match* is given and the blob's live generation no
        longer matches it (someone else won the race).  The underlying
        ``google.api_core.exceptions.PreconditionFailed`` is chained as
        ``__cause__``.
    """
    _validate_blob_name(blob_name)
    bucket_client = _client_or_new(client, bucket, project)
    blob = bucket_client.blob(blob_name)
    logger.debug(
        "upload_bytes -> %s (%d bytes, if_generation_match=%r)",
        blob_name, len(data), if_generation_match,
    )
    try:
        if if_generation_match is None:
            blob.upload_from_string(data)
        else:
            blob.upload_from_string(data, if_generation_match=if_generation_match)
    except _precondition_failed_exc_types() as exc:
        raise PreconditionFailedError(
            f"upload_bytes: generation precondition failed for "
            f"{blob_name!r} (if_generation_match={if_generation_match!r})"
        ) from exc
    return blob.generation


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


def read_bytes_with_generation(
    *,
    blob_name: str,
    bucket: str,
    project: str | None = None,
    client: Any | None = None,
) -> tuple[bytes, int] | None:
    """Download a blob's bytes together with its current generation.

    This is the CAS-aware counterpart of :func:`download_bytes`, used by
    ``blob_lease.BlobLease`` to read the current fence token before deciding
    whether/how to write.  Never raises on a missing blob — that is the
    "no lease yet" / "first write" case callers must distinguish from a real
    error, so it is signalled by returning ``None`` rather than an exception.

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
    tuple[bytes, int] | None
        ``(data, generation)`` if the blob exists, else ``None``.
    """
    _validate_blob_name(blob_name)
    bucket_client = _client_or_new(client, bucket, project)
    blob = bucket_client.blob(blob_name)
    try:
        data = blob.download_as_bytes()
    except _not_found_exc_types():
        return None
    logger.debug(
        "read_bytes_with_generation <- %s (generation=%r)", blob_name, blob.generation
    )
    return data, blob.generation
