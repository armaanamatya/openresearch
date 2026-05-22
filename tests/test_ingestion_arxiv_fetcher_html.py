"""Tests for ArxivFetcher HTML fetch — fail-soft HTML sibling download."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock
from urllib.request import Request

import pytest

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
