"""PdfPathFetcher: copies a local PDF into per-project storage."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from backend.services.ingestion.intake.fetchers.interface import (
    FetchResult,
    IntakeFetchError,
    IntakeFetcher,
)
from backend.services.ingestion.intake.sources import PaperSource, PdfPath
from backend.services.paths import normalize_path_input


_PDF_MAGIC = b"%PDF-"
_MAX_PDF_BYTES = 100 * 1024 * 1024  # 100 MB hard cap; bigger is suspicious


class PdfPathFetcher(IntakeFetcher):
    """Copies a local PDF into ``runs/{project_id}/raw_paper.pdf``,
    verifies it starts with the PDF magic bytes, and returns the SHA256
    + size. Bytes-level integrity (not signature trust) is what we need
    here — the spec's heavy security hardening (subprocess isolation,
    seccomp) lives in the parser worker, not the fetcher.

    `runs_root` defaults to ``./runs`` so the same PDF re-fetched in a
    different cwd lands in different paths, which is fine — the file
    on disk is keyed by project_id (content-addressed, so the same
    PDF always gets the same project_id).
    """

    def __init__(self, runs_root: Path = Path("runs")) -> None:
        self._runs_root = runs_root

    @property
    def kind(self) -> str:
        return "pdf_path"

    def fetch(self, source: PaperSource, *, project_id: str) -> FetchResult:
        if not isinstance(source, PdfPath):
            raise IntakeFetchError(
                f"PdfPathFetcher cannot fetch source kind {source.kind!r}",
                cause_kind="unsupported_source_kind",
                retryable=False,
            )

        src_path = Path(normalize_path_input(source.path)).expanduser()
        if not src_path.exists():
            raise IntakeFetchError(
                f"PDF not found at {src_path!s}",
                cause_kind="file_not_found",
                retryable=True,  # may appear later; not a structural failure
            )
        if not src_path.is_file():
            raise IntakeFetchError(
                f"Path {src_path!s} is not a regular file",
                cause_kind="not_a_file",
                retryable=False,
            )

        # Validate magic + size before anything else.
        size = src_path.stat().st_size
        if size > _MAX_PDF_BYTES:
            raise IntakeFetchError(
                f"PDF size {size} exceeds limit {_MAX_PDF_BYTES}",
                cause_kind="pdf_too_large",
                retryable=False,
            )
        with src_path.open("rb") as f:
            head = f.read(len(_PDF_MAGIC))
        if head != _PDF_MAGIC:
            raise IntakeFetchError(
                f"File at {src_path!s} does not start with PDF magic ({head!r})",
                cause_kind="not_a_pdf",
                retryable=False,
            )

        # Hash + copy in one streamed pass.
        dst_dir = self._runs_root / project_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / "raw_paper.pdf"

        digest = hashlib.sha256()
        with src_path.open("rb") as src_f, dst_path.open("wb") as dst_f:
            while True:
                chunk = src_f.read(64 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                dst_f.write(chunk)

        return FetchResult(
            raw_paper_path=str(dst_path.resolve()),
            pdf_sha256=digest.hexdigest(),
            pdf_size_bytes=size,
        )


__all__ = ["PdfPathFetcher"]
