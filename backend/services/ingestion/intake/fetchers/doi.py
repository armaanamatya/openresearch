"""DoiFetcher: resolves a DOI through doi.org and stores the resulting PDF."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from backend.services.ingestion.intake.fetchers.interface import (
    FetchResult,
    IntakeFetchError,
    IntakeFetcher,
)
from backend.services.ingestion.intake.fetchers.remote_pdf import UrlOpen, download_pdf
from backend.services.ingestion.intake.sources import DoiRef, PaperSource


class DoiFetcher(IntakeFetcher):
    def __init__(
        self,
        runs_root: Path = Path("runs"),
        *,
        resolver_base_url: str = "https://doi.org",
        urlopen_fn: UrlOpen | None = None,
    ) -> None:
        self._runs_root = runs_root
        self._resolver_base_url = resolver_base_url.rstrip("/")
        self._urlopen_fn = urlopen_fn

    @property
    def kind(self) -> str:
        return "doi"

    def fetch(self, source: PaperSource, *, project_id: str) -> FetchResult:
        if not isinstance(source, DoiRef):
            raise IntakeFetchError(
                f"DoiFetcher cannot fetch source kind {source.kind!r}",
                cause_kind="unsupported_source_kind",
                retryable=False,
            )
        encoded = quote(source.doi, safe="/")
        url = f"{self._resolver_base_url}/{encoded}"
        kwargs = {"urlopen_fn": self._urlopen_fn} if self._urlopen_fn else {}
        return download_pdf(
            url=url,
            runs_root=self._runs_root,
            project_id=project_id,
            fetched_via=self.kind,
            headers={"Accept": "application/pdf"},
            **kwargs,
        )


__all__ = ["DoiFetcher"]
