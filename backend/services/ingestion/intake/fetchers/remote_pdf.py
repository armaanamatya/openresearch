"""Shared helpers for remote PDF fetchers."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.services.ingestion.intake.fetchers.interface import (
    FetchResult,
    IntakeFetchError,
)


_PDF_MAGIC = b"%PDF-"
_MAX_PDF_BYTES = 100 * 1024 * 1024
_CHUNK_BYTES = 64 * 1024


class UrlOpen(Protocol):
    def __call__(self, request: Request, *, timeout: float): ...


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
    """Download a remote PDF with bounded size and magic-byte validation."""
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
            retryable=500 <= int(exc.code) < 600,
        ) from exc
    except URLError as exc:
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
            f"Failed to store downloaded PDF for {project_id}: {exc}",
            cause_kind="storage_error",
            retryable=True,
        ) from exc

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
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return FetchResult(
        raw_paper_path=str(dst_path.resolve()),
        pdf_sha256=digest.hexdigest(),
        pdf_size_bytes=total,
    )


__all__ = ["download_pdf"]
