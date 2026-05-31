"""Shared helpers for remote PDF fetchers."""

from __future__ import annotations

import hashlib
import logging
import os
import socket
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.services.ingestion.intake.fetchers.interface import (
    FetchResult,
    IntakeFetchError,
)


_PDF_MAGIC = b"%PDF-"
_MAX_PDF_BYTES = 100 * 1024 * 1024
_CHUNK_BYTES = 64 * 1024

# Retry policy: 3 total attempts, exponential backoff with cap.
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = [1.0, 2.0]  # sleep before attempt 2, then attempt 3
_RETRY_BACKOFF_CAP = 4.0

# Module-level sleep hook — replace in tests to avoid real sleeps.
_sleep_fn: Callable[[float], None] = time.sleep

logger = logging.getLogger(__name__)


class UrlOpen(Protocol):
    def __call__(self, request: Request, *, timeout: float): ...


def _is_transient_http_error(exc: HTTPError) -> bool:
    """Return True when the HTTPError status warrants a retry (429, 5xx)."""
    code = int(exc.code)
    return code == 429 or 500 <= code < 600


def _attempt_download(
    *,
    request: Request,
    tmp_path: Path,
    url: str,
    fetched_via: str,
    timeout_seconds: float,
    urlopen_fn: UrlOpen,
) -> tuple[bytes, hashlib._Hash, int]:
    """Single attempt: open URL, stream to tmp_path.

    Returns (first_bytes, digest, total_bytes) on success.
    Raises IntakeFetchError (retryable=True/False) on any failure.
    """
    try:
        with urlopen_fn(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", getattr(response, "code", 200))
            if status is not None and int(status) >= 400:
                raise IntakeFetchError(
                    f"{fetched_via} fetch returned HTTP {status} for {url}",
                    cause_kind="http_error",
                    retryable=500 <= int(status) < 600,
                )

            digest = hashlib.sha256()
            total = 0
            first = b""
            with tmp_path.open("wb") as dst_f:
                while True:
                    chunk = response.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    if not first:
                        first = chunk[: len(_PDF_MAGIC)]
                    total += len(chunk)
                    if total > _MAX_PDF_BYTES:
                        raise IntakeFetchError(
                            f"Remote PDF size exceeds limit {_MAX_PDF_BYTES}",
                            cause_kind="pdf_too_large",
                            retryable=False,
                        )
                    digest.update(chunk)
                    dst_f.write(chunk)
    except HTTPError as exc:
        raise IntakeFetchError(
            f"{fetched_via} fetch returned HTTP {exc.code} for {url}",
            cause_kind="http_error",
            retryable=_is_transient_http_error(exc),
        ) from exc
    except socket.timeout as exc:
        raise IntakeFetchError(
            f"{fetched_via} fetch timed out for {url}",
            cause_kind="timeout",
            retryable=True,
        ) from exc
    except URLError as exc:
        # URLError wraps socket.timeout on some Python versions — unwrap it.
        if isinstance(exc.reason, socket.timeout):
            raise IntakeFetchError(
                f"{fetched_via} fetch timed out for {url}",
                cause_kind="timeout",
                retryable=True,
            ) from exc
        raise IntakeFetchError(
            f"{fetched_via} fetch failed for {url}: {exc.reason}",
            cause_kind="network_error",
            retryable=True,
        ) from exc
    except TimeoutError as exc:
        raise IntakeFetchError(
            f"{fetched_via} fetch timed out for {url}",
            cause_kind="timeout",
            retryable=True,
        ) from exc
    except IntakeFetchError:
        raise
    except OSError as exc:
        raise IntakeFetchError(
            f"Failed to store downloaded PDF for request: {exc}",
            cause_kind="storage_error",
            retryable=True,
        ) from exc

    return first, digest, total


def download_pdf(
    *,
    url: str,
    runs_root: Path,
    project_id: str,
    fetched_via: str,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 30.0,
    urlopen_fn: UrlOpen = urlopen,
) -> FetchResult:
    """Download a remote PDF with bounded size, magic-byte validation, and retry.

    Transient failures (socket/network timeout, URLError without HTTP status,
    HTTP 429 or 5xx) are retried up to ``_MAX_ATTEMPTS`` total attempts with
    exponential backoff capped at ``_RETRY_BACKOFF_CAP`` seconds.  Non-transient
    failures (e.g. HTTP 404) propagate immediately on the first attempt.
    """
    dst_dir = runs_root / project_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / "raw_paper.pdf"
    tmp_path = dst_dir / "raw_paper.pdf.tmp"

    request = Request(
        url,
        headers={
            "User-Agent": "openresearch-ingestion/0.1",
            "Accept": "application/pdf,*/*;q=0.5",
            **dict(headers or {}),
        },
    )

    last_exc: IntakeFetchError | None = None
    for attempt in range(_MAX_ATTEMPTS):
        if attempt > 0:
            sleep_s = min(_RETRY_BACKOFF_SECONDS[attempt - 1], _RETRY_BACKOFF_CAP)
            logger.warning(
                "%s PDF fetch attempt %d/%d failed (%s); retrying in %.1fs",
                fetched_via,
                attempt,
                _MAX_ATTEMPTS,
                last_exc.cause_kind if last_exc else "unknown",
                sleep_s,
            )
            _sleep_fn(sleep_s)

        try:
            first, digest, total = _attempt_download(
                request=request,
                tmp_path=tmp_path,
                url=url,
                fetched_via=fetched_via,
                timeout_seconds=timeout_seconds,
                urlopen_fn=urlopen_fn,
            )
        except IntakeFetchError as exc:
            last_exc = exc
            if not exc.retryable:
                # Non-transient — fail immediately, no retry.
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            # Transient — loop to next attempt.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            continue

        # Successful download — validate magic bytes and atomically commit.
        try:
            if first != _PDF_MAGIC:
                raise IntakeFetchError(
                    f"Downloaded content from {url} does not start with PDF magic",
                    cause_kind="not_a_pdf",
                    retryable=False,
                )
            os.replace(tmp_path, dst_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            raise

        return FetchResult(
            raw_paper_path=str(dst_path.resolve()),
            pdf_sha256=digest.hexdigest(),
            pdf_size_bytes=total,
        )

    # Exhausted all attempts.
    assert last_exc is not None
    raise last_exc


__all__ = ["download_pdf"]
