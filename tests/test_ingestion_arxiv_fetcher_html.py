"""Tests for ArxivFetcher HTML fetch — fail-soft HTML sibling download."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from urllib.request import Request


from backend.services.ingestion.intake.fetchers import arxiv as arxiv_mod
from backend.services.ingestion.intake.fetchers.arxiv import ArxivFetcher
from backend.services.ingestion.intake.sources import ArxivId


# ---------------------------------------------------------------------------
# Helpers — fake urlopen that returns HTML or a non-200/non-HTML response.
# ---------------------------------------------------------------------------

_ARXIV_ID = "2512.24601"
_GOOD_HTML = b"<!DOCTYPE html><html><body><article>" + b"x" * 6000 + b"</article></body></html>"
_SMALL_HTML = b"<!DOCTYPE html><html><body>tiny</body></html>"
_PDF_BYTES = b"%PDF-1.4 fake content here " + b"0" * 200


class _FakeResponse:
    """Minimal urllib response stand-in with chunked-read support."""

    def __init__(self, body: bytes, status: int = 200, content_type: str = "text/html") -> None:
        self._body = body
        self._pos = 0
        self.status = status
        self._content_type = content_type

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            chunk = self._body[self._pos:]
            self._pos = len(self._body)
        else:
            chunk = self._body[self._pos : self._pos + n]
            self._pos += len(chunk)
        return chunk

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def info(self) -> MagicMock:
        m = MagicMock()
        m.get.return_value = self._content_type
        return m


def _make_urlopen(pdf_response: _FakeResponse, html_response: _FakeResponse):
    """Return a urlopen callable that dispatches based on URL."""
    def _urlopen(request: Request, *, timeout: float) -> _FakeResponse:
        url = request.full_url
        if "html" in url:
            return html_response
        return pdf_response
    return _urlopen


def _setup_fetcher(tmp_path: Path, pdf_response: _FakeResponse, html_response: _FakeResponse) -> tuple[ArxivFetcher, ArxivId]:
    """Build a fetcher + source pair with injected urlopen."""
    # Write a valid-looking PDF to satisfy download_pdf's magic-byte check.
    # We patch the urlopen so the actual fetch returns our fake response.

    # download_pdf checks the first 5 bytes for %PDF-; we mock the response
    # to return that. But we need to route correctly: pdf URL → pdf_response.
    fetcher = ArxivFetcher(
        runs_root=tmp_path,
        urlopen_fn=_make_urlopen(pdf_response, html_response),
    )
    source = ArxivId(arxiv_id=_ARXIV_ID)
    return fetcher, source


def test_html_sibling_written_on_200_html_response(tmp_path: Path):
    """A 200 HTML response with >5000 bytes causes raw_paper.html to be written."""
    project_id = "prj_html_test"

    pdf_resp = _FakeResponse(_PDF_BYTES, status=200, content_type="application/pdf")
    html_resp = _FakeResponse(_GOOD_HTML, status=200, content_type="text/html")
    fetcher, source = _setup_fetcher(tmp_path, pdf_resp, html_resp)

    fetcher.fetch(source, project_id=project_id)

    html_path = tmp_path / project_id / "raw_paper.html"
    assert html_path.exists(), "raw_paper.html should have been written"
    assert html_path.read_bytes() == _GOOD_HTML


def test_no_html_sibling_on_404(tmp_path: Path):
    """A 404 response causes no raw_paper.html to be written and no exception."""
    project_id = "prj_html_404"

    pdf_resp = _FakeResponse(_PDF_BYTES, status=200, content_type="application/pdf")
    html_resp = _FakeResponse(b"not found", status=404, content_type="text/html")
    fetcher, source = _setup_fetcher(tmp_path, pdf_resp, html_resp)

    # Should not raise even though HTML is 404.
    fetcher.fetch(source, project_id=project_id)

    html_path = tmp_path / project_id / "raw_paper.html"
    assert not html_path.exists(), "raw_paper.html should NOT be written on 404"


def test_no_html_sibling_when_body_too_small(tmp_path: Path):
    """An HTML response smaller than 5000 bytes causes no raw_paper.html."""
    project_id = "prj_html_small"

    pdf_resp = _FakeResponse(_PDF_BYTES, status=200, content_type="application/pdf")
    html_resp = _FakeResponse(_SMALL_HTML, status=200, content_type="text/html")
    fetcher, source = _setup_fetcher(tmp_path, pdf_resp, html_resp)

    fetcher.fetch(source, project_id=project_id)

    html_path = tmp_path / project_id / "raw_paper.html"
    assert not html_path.exists(), "raw_paper.html should NOT be written when body < 5000 bytes"


def test_no_html_sibling_when_no_arxiv_marker(tmp_path: Path):
    """A 200 HTML body without <article> or ltx_document is rejected and raw_paper.html is NOT written."""
    project_id = "prj_html_no_marker"

    no_marker_html = b"<!DOCTYPE html><html><body>" + b"x" * 6000 + b"</body></html>"
    pdf_resp = _FakeResponse(_PDF_BYTES, status=200, content_type="application/pdf")
    html_resp = _FakeResponse(no_marker_html, status=200, content_type="text/html")
    fetcher, source = _setup_fetcher(tmp_path, pdf_resp, html_resp)

    fetcher.fetch(source, project_id=project_id)

    html_path = tmp_path / project_id / "raw_paper.html"
    assert not html_path.exists(), "raw_paper.html must NOT be written when arXiv structural marker is absent"


def test_html_sibling_written_when_arxiv_marker_present(tmp_path: Path):
    """A 200 HTML body containing <article> and >5000 bytes causes raw_paper.html to be written."""
    project_id = "prj_html_with_marker"

    marker_html = b"<!DOCTYPE html><html><body><article>" + b"x" * 6000 + b"</article></body></html>"
    pdf_resp = _FakeResponse(_PDF_BYTES, status=200, content_type="application/pdf")
    html_resp = _FakeResponse(marker_html, status=200, content_type="text/html")
    fetcher, source = _setup_fetcher(tmp_path, pdf_resp, html_resp)

    fetcher.fetch(source, project_id=project_id)

    html_path = tmp_path / project_id / "raw_paper.html"
    assert html_path.exists(), "raw_paper.html must be written when arXiv <article> marker is present"
    assert html_path.read_bytes() == marker_html


# ---------------------------------------------------------------------------
# T17 guard test — unbounded HTML fetch must be capped (review I5)
# ---------------------------------------------------------------------------

def test_arxiv_html_fetch_caps_at_50mb(tmp_path: Path):
    """Symptom: a malicious arXiv-HTML body OOM-kills ingestion.

    The HTML fetch used response.read() unbounded; PDF enforced a 100 MB cap
    but HTML had none (review I5 / T17). Verify: a 60 MB simulated body causes
    the fetch to abort gracefully (no raw_paper.html written, no OOM).
    """
    project_id = "prj_html_50mb_cap"

    _60MB = 60 * 1024 * 1024
    big_html_prefix = b"<!DOCTYPE html><html><body><article>"
    big_html_body = big_html_prefix + b"x" * (_60MB - len(big_html_prefix))

    pdf_resp = _FakeResponse(_PDF_BYTES, status=200, content_type="application/pdf")
    html_resp = _FakeResponse(big_html_body, status=200, content_type="text/html")
    fetcher, source = _setup_fetcher(tmp_path, pdf_resp, html_resp)

    # fetch() is fail-soft — HTML errors are swallowed.
    result = fetcher.fetch(source, project_id=project_id)

    # The PDF result must still be valid.
    assert result.raw_paper_path.endswith("raw_paper.pdf")
    # HTML must NOT have been written — the cap aborted the read.
    html_path = tmp_path / project_id / "raw_paper.html"
    assert not html_path.exists(), "raw_paper.html must NOT be written when body exceeds 50 MB cap"


def test_fetch_result_unchanged_when_html_fails(tmp_path: Path):
    """HTML fetch failure does not affect the returned FetchResult (PDF path still set)."""
    project_id = "prj_html_err"

    pdf_resp = _FakeResponse(_PDF_BYTES, status=200, content_type="application/pdf")
    # html_resp raises to simulate a network failure
    def _failing_urlopen(request: Request, *, timeout: float) -> _FakeResponse:
        url = request.full_url
        if "html" in url:
            raise OSError("simulated network failure")
        return pdf_resp

    fetcher = ArxivFetcher(
        runs_root=tmp_path,
        urlopen_fn=_failing_urlopen,
    )
    source = ArxivId(arxiv_id=_ARXIV_ID)

    result = fetcher.fetch(source, project_id=project_id)

    assert result.raw_paper_path.endswith("raw_paper.pdf")
    html_path = tmp_path / project_id / "raw_paper.html"
    assert not html_path.exists()


# ---------------------------------------------------------------------------
# ar5iv fallback (P0 / borrow #2) — native-then-ar5iv source order.
# Native and ar5iv are distinguished by "ar5iv" in the request URL.
# ---------------------------------------------------------------------------

def test_html_falls_back_to_ar5iv_when_native_unavailable(tmp_path: Path):
    """When arxiv.org/html 404s, the ar5iv mirror is tried and its body is used.

    Freshly-posted papers often lack a native arxiv.org/html rendering for hours
    but are already on ar5iv — the fallback recovers those.
    """
    project_id = "prj_html_ar5iv_fallback"

    pdf_resp = _FakeResponse(_PDF_BYTES, status=200, content_type="application/pdf")
    native_404 = _FakeResponse(b"not found", status=404, content_type="text/html")
    ar5iv_good = _FakeResponse(_GOOD_HTML, status=200, content_type="text/html")

    def _urlopen(request: Request, *, timeout: float) -> _FakeResponse:
        url = request.full_url
        if "ar5iv" in url:
            return ar5iv_good
        if "html" in url:
            return native_404
        return pdf_resp

    fetcher = ArxivFetcher(runs_root=tmp_path, urlopen_fn=_urlopen)
    source = ArxivId(arxiv_id=_ARXIV_ID)

    fetcher.fetch(source, project_id=project_id)

    html_path = tmp_path / project_id / "raw_paper.html"
    assert html_path.exists(), "raw_paper.html should be written from the ar5iv fallback"
    assert html_path.read_bytes() == _GOOD_HTML


def test_html_prefers_native_arxiv_and_skips_ar5iv(tmp_path: Path):
    """A valid native arxiv.org/html body short-circuits — ar5iv is never queried."""
    project_id = "prj_html_prefer_native"

    pdf_resp = _FakeResponse(_PDF_BYTES, status=200, content_type="application/pdf")
    native_good = _FakeResponse(_GOOD_HTML, status=200, content_type="text/html")

    requested_html_urls: list[str] = []

    def _urlopen(request: Request, *, timeout: float) -> _FakeResponse:
        url = request.full_url
        if "html" in url:
            requested_html_urls.append(url)
            return native_good
        return pdf_resp

    fetcher = ArxivFetcher(runs_root=tmp_path, urlopen_fn=_urlopen)
    source = ArxivId(arxiv_id=_ARXIV_ID)

    fetcher.fetch(source, project_id=project_id)

    html_path = tmp_path / project_id / "raw_paper.html"
    assert html_path.exists()
    assert html_path.read_bytes() == _GOOD_HTML
    # Exactly one HTML fetch — the native endpoint — and ar5iv was never queried.
    assert len(requested_html_urls) == 1, f"expected 1 HTML fetch, got {requested_html_urls}"
    assert "ar5iv" not in requested_html_urls[0]
    assert "arxiv.org/html" in requested_html_urls[0]


# ---------------------------------------------------------------------------
# F-28 — within-source transient retry (the PDF fetch got a 3-attempt retry;
# the HTML fetch was single-shot, so a transient 503/429/timeout silently
# degraded the run to figure-noisy PDF).
# ---------------------------------------------------------------------------

def test_html_fetch_retries_transient_then_succeeds(tmp_path: Path, monkeypatch):
    """A transient 503 on the native HTML endpoint is retried within-source and
    then succeeds — without degrading to ar5iv/PDF (F-28)."""
    project_id = "prj_html_retry"
    monkeypatch.setattr(arxiv_mod, "_html_sleep_fn", lambda _s: None)

    pdf_resp = _FakeResponse(_PDF_BYTES, status=200, content_type="application/pdf")
    native_calls = {"n": 0}

    def _urlopen(request: Request, *, timeout: float) -> _FakeResponse:
        url = request.full_url
        if "ar5iv" in url:
            # ar5iv must NOT be needed — the native retry should succeed first.
            return _FakeResponse(b"ar5iv should not be used", status=500, content_type="text/html")
        if "html" in url:  # native arxiv.org/html
            native_calls["n"] += 1
            if native_calls["n"] == 1:
                return _FakeResponse(b"upstream busy", status=503, content_type="text/html")
            return _FakeResponse(_GOOD_HTML, status=200, content_type="text/html")
        return pdf_resp

    fetcher = ArxivFetcher(runs_root=tmp_path, urlopen_fn=_urlopen)
    source = ArxivId(arxiv_id=_ARXIV_ID)
    fetcher.fetch(source, project_id=project_id)

    html_path = tmp_path / project_id / "raw_paper.html"
    assert html_path.exists(), "raw_paper.html should be written after a within-source transient retry"
    assert html_path.read_bytes() == _GOOD_HTML
    assert native_calls["n"] == 2, "native HTML must be retried once (2 attempts), not single-shot"


def test_html_fetch_does_not_retry_on_404(tmp_path: Path, monkeypatch):
    """A 404 (no HTML rendering yet) is permanent — tried once, then falls
    straight through to the ar5iv mirror; F-28's retry must not over-retry."""
    project_id = "prj_html_404_noretry"
    monkeypatch.setattr(arxiv_mod, "_html_sleep_fn", lambda _s: None)

    pdf_resp = _FakeResponse(_PDF_BYTES, status=200, content_type="application/pdf")
    native_calls = {"n": 0}
    ar5iv_good = _FakeResponse(_GOOD_HTML, status=200, content_type="text/html")

    def _urlopen(request: Request, *, timeout: float) -> _FakeResponse:
        url = request.full_url
        if "ar5iv" in url:
            return ar5iv_good
        if "html" in url:
            native_calls["n"] += 1
            return _FakeResponse(b"not found", status=404, content_type="text/html")
        return pdf_resp

    fetcher = ArxivFetcher(runs_root=tmp_path, urlopen_fn=_urlopen)
    source = ArxivId(arxiv_id=_ARXIV_ID)
    fetcher.fetch(source, project_id=project_id)

    html_path = tmp_path / project_id / "raw_paper.html"
    assert html_path.exists(), "should recover via ar5iv"
    assert html_path.read_bytes() == _GOOD_HTML
    assert native_calls["n"] == 1, "404 is permanent — native tried exactly once, then ar5iv"


# ---------------------------------------------------------------------------
# F-31 — validated-cache reuse: a valid cached raw_paper.html is reused instead
# of re-downloading (arXiv paper versions are immutable).
# ---------------------------------------------------------------------------

def test_html_fetch_reuses_validated_cache(tmp_path: Path):
    """A valid cached raw_paper.html is reused without re-querying the network (F-31).

    Asserts on the urlopen call count (not a raised error — _try_fetch_html
    swallows exceptions), so a cache miss is detectable.
    """
    project_id = "prj_html_cache"
    dst = tmp_path / project_id
    dst.mkdir(parents=True)
    (dst / "raw_paper.html").write_bytes(_GOOD_HTML)  # seed a valid cached body

    calls = {"n": 0}

    def _count(request: Request, *, timeout: float) -> _FakeResponse:
        calls["n"] += 1
        return _FakeResponse(_GOOD_HTML, status=200, content_type="text/html")

    fetcher = ArxivFetcher(runs_root=tmp_path, urlopen_fn=_count)
    fetcher._fetch_html(_ARXIV_ID, project_id=project_id)
    assert calls["n"] == 0, "a valid cached raw_paper.html must be reused, not re-fetched"
    assert (dst / "raw_paper.html").read_bytes() == _GOOD_HTML


def test_html_fetch_force_refetch_bypasses_cache(tmp_path: Path):
    """force_refetch=True re-fetches even when a valid cached HTML body exists (F-31)."""
    project_id = "prj_html_force"
    dst = tmp_path / project_id
    dst.mkdir(parents=True)
    (dst / "raw_paper.html").write_bytes(_GOOD_HTML)

    calls = {"n": 0}
    fresh = b"<!DOCTYPE html><html><body><article>" + b"y" * 6000 + b"</article></body></html>"

    def _urlopen(request: Request, *, timeout: float) -> _FakeResponse:
        calls["n"] += 1
        return _FakeResponse(fresh, status=200, content_type="text/html")

    fetcher = ArxivFetcher(runs_root=tmp_path, urlopen_fn=_urlopen)
    fetcher._fetch_html(_ARXIV_ID, project_id=project_id, force_refetch=True)
    assert calls["n"] >= 1, "force_refetch must re-fetch"
    assert (dst / "raw_paper.html").read_bytes() == fresh
