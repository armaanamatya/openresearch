"""ArxivFetcher: downloads PDFs (and HTML when available) from arXiv."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
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
# ar5iv mirror — freshly-posted papers often lack an arxiv.org/html rendering
# for hours but are already on ar5iv. Tried as a fail-soft 2nd source after the
# native endpoint. Pattern (native-then-ar5iv source order) adapted from
# huggingface/ml-intern papers_tool.py (Apache-2.0).
_AR5IV_BASE_URL = "https://ar5iv.labs.arxiv.org/html"
_HTML_TIMEOUT = 30.0
_HTML_MIN_BYTES = 5000
_HTML_MAX_BYTES = 50 * 1024 * 1024  # 50 MB — half the PDF cap; guard against DoS (T17, review I5)
_USER_AGENT = "Mozilla/5.0 (compatible; openresearch-ingestion/0.1)"

# Within-source transient-failure retry for the HTML fetch (F-28) — mirrors the
# PDF fetch retry (remote_pdf.download_pdf). The native endpoint and the ar5iv
# mirror are each retried up to _HTML_MAX_ATTEMPTS times on a transient signal
# (timeout, URLError, HTTP 429/5xx) before falling through to the next source;
# a non-transient miss (404 / non-HTML / too-small / no marker) is not retried.
_HTML_MAX_ATTEMPTS = 2
_HTML_RETRY_BACKOFF_SECONDS = [1.0]  # sleep before attempt 2
_html_sleep_fn = time.sleep  # module-level hook — patched in tests to avoid real sleeps


def _cached_html_is_valid(body: bytes) -> bool:
    """Body-based validation for reusing a cached raw_paper.html (F-31).

    Mirrors the structural checks ``_try_fetch_html`` applies (min size + an
    HTML-looking prefix + the arXiv ``<article>``/``ltx_document`` marker). It
    is intentionally conservative: a miss only triggers a fresh fetch, never a
    false reuse.
    """
    if len(body) < _HTML_MIN_BYTES:
        return False
    if not (body[:9].lower().startswith(b"<!doctype") or body[:5].lower().startswith(b"<html")):
        return False
    return b"<article" in body or b"ltx_document" in body


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

    def _fetch_html(self, arxiv_id: str, *, project_id: str, force_refetch: bool = False) -> None:
        """Fetch the arXiv HTML rendering and write it as raw_paper.html.

        Tries the native arxiv.org/html endpoint first, then the ar5iv mirror as
        a fallback — freshly-posted papers often lack a native rendering for
        hours but are already on ar5iv. Silently skips if neither source yields a
        valid body. Never raises — the PDF fetch result is the authoritative
        contract.

        A valid cached raw_paper.html is reused instead of re-downloading
        (F-31; arXiv paper versions are immutable). ``force_refetch=True``
        bypasses the cache for a genuine re-ingest.

        Source order (native-then-ar5iv) adapted from huggingface/ml-intern
        papers_tool.py (Apache-2.0).
        """
        html_path = self._runs_root / project_id / "raw_paper.html"
        if not force_refetch and html_path.exists():
            try:
                cached = html_path.read_bytes()
            except OSError:
                cached = b""
            if _cached_html_is_valid(cached):
                logger.info("arXiv HTML reused from cache: %s (%d bytes)", html_path, len(cached))
                return

        for base_url in (_HTML_BASE_URL, _AR5IV_BASE_URL):
            body = self._try_fetch_html(f"{base_url}/{arxiv_id}", arxiv_id)
            if body is None:
                continue  # source unavailable/invalid — fall through to the next
            dst_dir = self._runs_root / project_id
            dst_dir.mkdir(parents=True, exist_ok=True)
            html_path = dst_dir / "raw_paper.html"
            html_path.write_bytes(body)
            logger.info("arXiv HTML saved to %s (%d bytes, source=%s)", html_path, len(body), base_url)
            return  # first valid source wins — a good native body short-circuits ar5iv

    def _try_fetch_html(self, html_url: str, arxiv_id: str) -> bytes | None:
        """Fetch + validate a single HTML source. Returns the validated body or None.

        Transient failures (timeout, URLError, HTTP 429/5xx) are retried up to
        ``_HTML_MAX_ATTEMPTS`` times within this source before falling through
        (F-28) — mirroring the PDF fetch retry. A non-transient miss (404 /
        non-HTML / too-small / missing arXiv marker / size-cap) returns None
        immediately. Never raises — any error yields None so the caller can fall
        through to the next source.
        """
        _open = self._urlopen_fn if self._urlopen_fn is not None else _stdlib_urlopen
        request = Request(
            html_url,
            headers={"User-Agent": _USER_AGENT},
        )

        for attempt in range(_HTML_MAX_ATTEMPTS):
            if attempt > 0:
                idx = min(attempt - 1, len(_HTML_RETRY_BACKOFF_SECONDS) - 1)
                sleep_s = _HTML_RETRY_BACKOFF_SECONDS[idx]
                logger.warning(
                    "arXiv HTML fetch for %s failed transiently (attempt %d/%d); retrying in %.1fs",
                    html_url, attempt, _HTML_MAX_ATTEMPTS, sleep_s,
                )
                _html_sleep_fn(sleep_s)

            try:
                with _open(request, timeout=_HTML_TIMEOUT) as response:
                    status = getattr(response, "status", getattr(response, "code", 200))
                    if status is not None and int(status) != 200:
                        code = int(status)
                        if code == 429 or 500 <= code < 600:
                            continue  # transient HTTP — retry within source
                        logger.debug("arXiv HTML fetch returned HTTP %s for %s — skipping", status, html_url)
                        return None

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
                            return None
                    body = bytes(body)

                    # Read headers inside the `with` block — response.info() is
                    # only valid while the connection is open (review M5 / T28).
                    content_type = response.info().get("Content-Type", "") or ""  # type: ignore[union-attr]
            except HTTPError as exc:
                code = int(getattr(exc, "code", 0) or 0)
                if code == 429 or 500 <= code < 600:
                    continue  # transient — retry within source
                logger.warning("arXiv HTML fetch failed for %s at %s: %s", arxiv_id, html_url, exc)
                return None
            except (URLError, TimeoutError) as exc:
                # timeout / connection error (socket.timeout aliases TimeoutError) — transient
                logger.warning("arXiv HTML fetch transient error for %s at %s: %s", arxiv_id, html_url, exc)
                continue
            except Exception as exc:
                logger.warning("arXiv HTML fetch failed for %s at %s: %s", arxiv_id, html_url, exc)
                return None

            # --- Validation (non-transient — never retried) ---
            is_html = (
                "html" in content_type.lower()
                or body[:9].lower().startswith(b"<!doctype")
                or body[:5].lower().startswith(b"<html")
            )
            if not is_html:
                logger.debug("arXiv HTML body for %s is not HTML (content-type=%r) — skipping", arxiv_id, content_type)
                return None

            if len(body) < _HTML_MIN_BYTES:
                logger.debug("arXiv HTML for %s is only %d bytes (min %d) — skipping", arxiv_id, len(body), _HTML_MIN_BYTES)
                return None

            # Require a structural arXiv-paper marker to reject error/interstitial pages.
            has_article_marker = b"<article" in body or b"ltx_document" in body
            if not has_article_marker:
                logger.debug(
                    "arXiv HTML for %s lacks <article> / ltx_document marker — likely an interstitial, skipping",
                    arxiv_id,
                )
                return None

            return body

        # All attempts exhausted on transient failures.
        return None


__all__ = ["ArxivFetcher"]
