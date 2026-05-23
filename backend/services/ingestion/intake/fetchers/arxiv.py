"""ArxivFetcher: downloads PDFs (and HTML when available) from arXiv."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.request import Request, urlopen as _stdlib_urlopen

from backend.services.ingestion.intake.fetchers.interface import (
    FetchResult,
    IntakeFetchError,
    IntakeFetcher,
)
from backend.services.ingestion.intake.fetchers.remote_pdf import UrlOpen, download_pdf
from backend.services.ingestion.intake.sources import ArxivId, PaperSource

logger = logging.getLogger(__name__)

_HTML_BASE_URL = "https://arxiv.org/html"
_HTML_TIMEOUT = 30.0
_HTML_MIN_BYTES = 5000
_HTML_MAX_BYTES = 50 * 1024 * 1024  # 50 MB — half the PDF cap; guard against DoS (T17, review I5)
_USER_AGENT = "Mozilla/5.0 (compatible; openresearch-ingestion/0.1)"


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
        result = download_pdf(
            url=url,
            runs_root=self._runs_root,
            project_id=project_id,
            fetched_via=self.kind,
            **kwargs,
        )

        # Opportunistically fetch the arXiv HTML version — fail-soft.
        self._fetch_html(source.arxiv_id, project_id=project_id)

        return result

    def _fetch_html(self, arxiv_id: str, *, project_id: str) -> None:
        """Fetch the arXiv HTML rendering and write it as raw_paper.html.

        Silently skips if the HTML is unavailable or too small.
        Never raises — the PDF fetch result is the authoritative contract.
        """
        try:
            html_url = f"{_HTML_BASE_URL}/{arxiv_id}"
            _open = self._urlopen_fn if self._urlopen_fn is not None else _stdlib_urlopen
            request = Request(
                html_url,
                headers={"User-Agent": _USER_AGENT},
            )
            with _open(request, timeout=_HTML_TIMEOUT) as response:
                status = getattr(response, "status", getattr(response, "code", 200))
                if status is not None and int(status) != 200:
                    logger.debug("arXiv HTML fetch returned HTTP %s for %s — skipping", status, html_url)
                    return

                # Chunked-read with a 50 MB cap — unbounded read() OOM-kills
                # ingestion on malicious/buggy endpoints (T17, review I5).
                body = bytearray()
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    body.extend(chunk)
                    if len(body) > _HTML_MAX_BYTES:
                        logger.warning(
                            "arXiv HTML body for %s exceeded %d bytes — aborting (possible DoS)",
                            arxiv_id,
                            _HTML_MAX_BYTES,
                        )
                        return
                body = bytes(body)

            content_type = ""
            # urllib responses expose headers via info() or headers attribute.
            try:
                content_type = response.info().get("Content-Type", "")  # type: ignore[union-attr]
            except Exception:
                pass

            is_html = (
                "html" in content_type.lower()
                or body[:9].lower().startswith(b"<!doctype")
                or body[:5].lower().startswith(b"<html")
            )
            if not is_html:
                logger.debug("arXiv HTML body for %s is not HTML (content-type=%r) — skipping", arxiv_id, content_type)
                return

            if len(body) < _HTML_MIN_BYTES:
                logger.debug("arXiv HTML for %s is only %d bytes (min %d) — skipping", arxiv_id, len(body), _HTML_MIN_BYTES)
                return

            # Require a structural arXiv-paper marker to reject error/interstitial pages.
            has_article_marker = b"<article" in body or b"ltx_document" in body
            if not has_article_marker:
                logger.debug(
                    "arXiv HTML for %s lacks <article> / ltx_document marker — likely an interstitial, skipping",
                    arxiv_id,
                )
                return

            dst_dir = self._runs_root / project_id
            dst_dir.mkdir(parents=True, exist_ok=True)
            html_path = dst_dir / "raw_paper.html"
            html_path.write_bytes(body)
            logger.info("arXiv HTML saved to %s (%d bytes)", html_path, len(body))

        except Exception as exc:
            logger.warning("arXiv HTML fetch failed for %s (project %s): %s", arxiv_id, project_id, exc)


__all__ = ["ArxivFetcher"]
