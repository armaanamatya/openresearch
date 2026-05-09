"""ArxivFetcher: downloads PDFs from arXiv."""

from __future__ import annotations

from pathlib import Path

from backend.services.ingestion.intake.fetchers.interface import (
    FetchResult,
    IntakeFetchError,
    IntakeFetcher,
)
from backend.services.ingestion.intake.fetchers.remote_pdf import UrlOpen, download_pdf
from backend.services.ingestion.intake.sources import ArxivId, PaperSource


class ArxivFetcher(IntakeFetcher):
    def __init__(
        self,
        runs_root: Path = Path("runs"),
        *,
        base_url: str = "https://arxiv.org/pdf",
        urlopen_fn: UrlOpen | None = None,
    ) -> None:
        self._runs_root = runs_root
        self._base_url = base_url.rstrip("/")
        self._urlopen_fn = urlopen_fn

    @property
    def kind(self) -> str:
        return "arxiv"

    def fetch(self, source: PaperSource, *, project_id: str) -> FetchResult:
        if not isinstance(source, ArxivId):
            raise IntakeFetchError(
                f"ArxivFetcher cannot fetch source kind {source.kind!r}",
                cause_kind="unsupported_source_kind",
                retryable=False,
            )
        url = f"{self._base_url}/{source.arxiv_id}"
        kwargs = {"urlopen_fn": self._urlopen_fn} if self._urlopen_fn else {}
        return download_pdf(
            url=url,
            runs_root=self._runs_root,
            project_id=project_id,
            fetched_via=self.kind,
            **kwargs,
        )


__all__ = ["ArxivFetcher"]
