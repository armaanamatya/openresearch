"""Tests for download_pdf retry logic in remote_pdf.py.

Covers:
- succeeds after a transient 503 on attempt 1,
- succeeds after a socket.timeout on attempt 1,
- succeeds after a URLError (network) on attempt 1,
- raises IntakeFetchError after exhausting all attempts,
- does NOT retry on HTTP 404 (non-transient),
- sleep is called with expected backoff values (not real sleeps).
"""

from __future__ import annotations

import socket
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

import backend.services.ingestion.intake.fetchers.remote_pdf as _module
from backend.services.ingestion.intake.fetchers.interface import IntakeFetchError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PDF_BYTES = b"%PDF-1.4 " + b"x" * 300  # >= 5 bytes; passes magic check


class _FakeResponse:
    """Minimal urllib response stand-in."""

    def __init__(self, body: bytes = _PDF_BYTES, status: int = 200) -> None:
        self._body = body
        self._pos = 0
        self.status = status

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            chunk = self._body[self._pos :]
            self._pos = len(self._body)
        else:
            chunk = self._body[self._pos : self._pos + n]
            self._pos += len(chunk)
        return chunk

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        pass


def _make_urlopen(*side_effects):
    """Return a urlopen callable that raises/returns items from side_effects in order."""
    calls: list[int] = [0]

    def _urlopen(request: Request, *, timeout: float):
        idx = calls[0]
        calls[0] += 1
        effect = side_effects[idx] if idx < len(side_effects) else side_effects[-1]
        if isinstance(effect, BaseException):
            raise effect
        if isinstance(effect, type) and issubclass(effect, BaseException):
            raise effect()
        return effect

    return _urlopen


def _make_http_error(code: int) -> HTTPError:
    return HTTPError(url="http://x.com/paper.pdf", code=code, msg="err", hdrs={}, fp=None)  # type: ignore[arg-type]


def _run_download(tmp_path: Path, urlopen_fn, *, sleep_calls=None):
    """Run download_pdf with a no-op sleep hook and return FetchResult.

    When the call raises, the exception propagates normally.  The sleep_calls
    list (if provided) is populated *inside* the finally block so it is always
    updated even when an exception is raised.
    """
    recorded_sleeps: list[float] = []

    def _fake_sleep(s: float) -> None:
        recorded_sleeps.append(s)

    old_sleep = _module._sleep_fn
    _module._sleep_fn = _fake_sleep
    try:
        result = _module.download_pdf(
            url="http://arxiv.org/pdf/2605.15155",
            runs_root=tmp_path,
            project_id="prj_test",
            fetched_via="test_fetcher",
            urlopen_fn=urlopen_fn,
            timeout_seconds=5.0,
        )
    finally:
        _module._sleep_fn = old_sleep
        if sleep_calls is not None:
            sleep_calls.extend(recorded_sleeps)

    return result


# ---------------------------------------------------------------------------
# Retry-success cases
# ---------------------------------------------------------------------------


def test_succeeds_after_transient_503(tmp_path: Path):
    """Attempt 1 → 503, attempt 2 → 200 OK."""
    err_503 = _make_http_error(503)
    ok_resp = _FakeResponse(_PDF_BYTES)
    urlopen = _make_urlopen(err_503, ok_resp)

    sleeps: list[float] = []
    result = _run_download(tmp_path, urlopen, sleep_calls=sleeps)

    assert result.raw_paper_path.endswith("raw_paper.pdf")
    assert Path(result.raw_paper_path).exists()
    # One backoff sleep before attempt 2.
    assert len(sleeps) == 1
    assert sleeps[0] == _module._RETRY_BACKOFF_SECONDS[0]


def test_succeeds_after_socket_timeout(tmp_path: Path):
    """Attempt 1 → socket.timeout, attempt 2 → 200 OK."""
    urlopen = _make_urlopen(socket.timeout("timed out"), _FakeResponse(_PDF_BYTES))

    sleeps: list[float] = []
    result = _run_download(tmp_path, urlopen, sleep_calls=sleeps)

    assert Path(result.raw_paper_path).exists()
    assert len(sleeps) == 1


def test_succeeds_after_urlerror_network(tmp_path: Path):
    """Attempt 1 → URLError (no HTTP code), attempt 2 → 200 OK."""
    net_err = URLError(reason="Connection refused")
    urlopen = _make_urlopen(net_err, _FakeResponse(_PDF_BYTES))

    sleeps: list[float] = []
    result = _run_download(tmp_path, urlopen, sleep_calls=sleeps)

    assert Path(result.raw_paper_path).exists()
    assert len(sleeps) == 1


def test_succeeds_after_429(tmp_path: Path):
    """Attempt 1 → 429 Too Many Requests, attempt 2 → 200 OK."""
    err_429 = _make_http_error(429)
    urlopen = _make_urlopen(err_429, _FakeResponse(_PDF_BYTES))

    sleeps: list[float] = []
    result = _run_download(tmp_path, urlopen, sleep_calls=sleeps)

    assert Path(result.raw_paper_path).exists()
    assert len(sleeps) == 1


def test_succeeds_on_third_attempt(tmp_path: Path):
    """Two transient failures before success: expect 2 backoff sleeps."""
    err_503 = _make_http_error(503)
    err_503b = _make_http_error(503)
    urlopen = _make_urlopen(err_503, err_503b, _FakeResponse(_PDF_BYTES))

    sleeps: list[float] = []
    result = _run_download(tmp_path, urlopen, sleep_calls=sleeps)

    assert Path(result.raw_paper_path).exists()
    assert len(sleeps) == 2
    assert sleeps[0] == _module._RETRY_BACKOFF_SECONDS[0]
    assert sleeps[1] == _module._RETRY_BACKOFF_SECONDS[1]


# ---------------------------------------------------------------------------
# Exhausted-all-attempts case
# ---------------------------------------------------------------------------


def test_raises_after_all_attempts_exhausted(tmp_path: Path):
    """Three transient 503s → raises IntakeFetchError after 3 attempts."""
    errs = [_make_http_error(503)] * _module._MAX_ATTEMPTS
    urlopen = _make_urlopen(*errs)

    sleeps: list[float] = []
    with pytest.raises(IntakeFetchError) as exc_info:
        _run_download(tmp_path, urlopen, sleep_calls=sleeps)

    assert exc_info.value.cause_kind == "http_error"
    # MAX_ATTEMPTS - 1 sleeps (before attempts 2 and 3).
    assert len(sleeps) == _module._MAX_ATTEMPTS - 1


# ---------------------------------------------------------------------------
# Non-transient: no retry on 404
# ---------------------------------------------------------------------------


def test_no_retry_on_404(tmp_path: Path):
    """HTTP 404 is non-retryable: must fail immediately, no sleep."""
    err_404 = _make_http_error(404)
    # Second entry would be the retry — we want to confirm it is never reached.
    urlopen = _make_urlopen(err_404, _FakeResponse(_PDF_BYTES))

    sleeps: list[float] = []
    with pytest.raises(IntakeFetchError) as exc_info:
        _run_download(tmp_path, urlopen, sleep_calls=sleeps)

    assert exc_info.value.cause_kind == "http_error"
    assert not exc_info.value.retryable
    # No sleep at all — failed fast on first attempt.
    assert sleeps == []


# ---------------------------------------------------------------------------
# Verify existing happy-path behaviour is unchanged
# ---------------------------------------------------------------------------


def test_happy_path_no_sleep(tmp_path: Path):
    """Clean 200 response → success on first attempt, no sleep called."""
    urlopen = _make_urlopen(_FakeResponse(_PDF_BYTES))

    sleeps: list[float] = []
    result = _run_download(tmp_path, urlopen, sleep_calls=sleeps)

    assert Path(result.raw_paper_path).exists()
    assert result.pdf_size_bytes == len(_PDF_BYTES)
    assert sleeps == []


def test_not_a_pdf_raises_immediately(tmp_path: Path):
    """Content without PDF magic bytes → not_a_pdf, no retry."""
    not_pdf = b"NOTPDF " + b"x" * 200
    urlopen = _make_urlopen(_FakeResponse(not_pdf))

    sleeps: list[float] = []
    with pytest.raises(IntakeFetchError) as exc_info:
        _run_download(tmp_path, urlopen, sleep_calls=sleeps)

    assert exc_info.value.cause_kind == "not_a_pdf"
    assert sleeps == []


# ---------------------------------------------------------------------------
# F-31 — validated-cache reuse (raw_paper.pdf is re-fetched on every ingest;
# a committed PDF is always a complete magic-validated download, and arXiv
# paper versions are immutable, so it should be reused).
# ---------------------------------------------------------------------------

def test_download_pdf_reuses_validated_cache(tmp_path: Path):
    """A committed raw_paper.pdf is reused on the next call with no network
    fetch (F-31); the returned FetchResult matches the cached file."""
    first = _run_download(tmp_path, _make_urlopen(_FakeResponse()))

    def _explode(request: Request, *, timeout: float):
        raise AssertionError("network fetch must not happen on a cache hit")

    second = _module.download_pdf(
        url="http://arxiv.org/pdf/2605.15155",
        runs_root=tmp_path,
        project_id="prj_test",
        fetched_via="test_fetcher",
        urlopen_fn=_explode,
        timeout_seconds=5.0,
    )
    assert second.raw_paper_path == first.raw_paper_path
    assert second.pdf_sha256 == first.pdf_sha256
    assert second.pdf_size_bytes == first.pdf_size_bytes


def test_download_pdf_force_refetch_bypasses_cache(tmp_path: Path):
    """force_refetch=True re-downloads even when a valid cached PDF exists (F-31)."""
    _run_download(tmp_path, _make_urlopen(_FakeResponse()))

    calls = {"n": 0}

    def _count(request: Request, *, timeout: float):
        calls["n"] += 1
        return _FakeResponse()

    _module.download_pdf(
        url="http://arxiv.org/pdf/2605.15155",
        runs_root=tmp_path,
        project_id="prj_test",
        fetched_via="test_fetcher",
        urlopen_fn=_count,
        timeout_seconds=5.0,
        force_refetch=True,
    )
    assert calls["n"] == 1, "force_refetch must re-download"
