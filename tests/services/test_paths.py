"""Tests for backend.services.paths.normalize_path_input."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from backend.services.paths import normalize_path_input


# ---------------------------------------------------------------------------
# Helper — force posix_host = True in the function under test regardless of
# the actual test-runner platform. On any POSIX system (Linux/macOS/WSL) this
# is a no-op; on native Windows it lets the POSIX-branch tests still run.
# ---------------------------------------------------------------------------
def _posix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure os.name appears as 'posix' for the paths module."""
    monkeypatch.setattr("backend.services.paths.os.name", "posix")


# ---------------------------------------------------------------------------
# 1. Windows absolute path with backslash → WSL mount path
# ---------------------------------------------------------------------------
def test_windows_abs_path_to_wsl_mount(monkeypatch):
    _posix(monkeypatch)
    # No wslpath binary on CI runners — ensure pure-Python fallback fires.
    monkeypatch.setattr("backend.services.paths.shutil.which", lambda _: None)
    result = normalize_path_input(r"C:\Users\Foo\paper.pdf")
    assert result == "/mnt/c/Users/Foo/paper.pdf"


# ---------------------------------------------------------------------------
# 2. Windows absolute path with forward slash → WSL mount path
# ---------------------------------------------------------------------------
def test_windows_abs_path_forward_slash(monkeypatch):
    _posix(monkeypatch)
    monkeypatch.setattr("backend.services.paths.shutil.which", lambda _: None)
    result = normalize_path_input("C:/Users/Foo/paper.pdf")
    assert result == "/mnt/c/Users/Foo/paper.pdf"


# ---------------------------------------------------------------------------
# 3. Double-quoted path (cmd / Explorer paste) — quotes stripped then normalized
# ---------------------------------------------------------------------------
def test_quoted_path_unquoted(monkeypatch):
    _posix(monkeypatch)
    monkeypatch.setattr("backend.services.paths.shutil.which", lambda _: None)
    result = normalize_path_input(r'"C:\Users\Foo paper.pdf"')
    assert result == "/mnt/c/Users/Foo paper.pdf"


# ---------------------------------------------------------------------------
# 4. Single-quoted path
# ---------------------------------------------------------------------------
def test_single_quoted_path(monkeypatch):
    _posix(monkeypatch)
    monkeypatch.setattr("backend.services.paths.shutil.which", lambda _: None)
    result = normalize_path_input(r"'C:\Users\Foo\paper.pdf'")
    assert result == "/mnt/c/Users/Foo/paper.pdf"


# ---------------------------------------------------------------------------
# 5. WSL UNC path with dollar sign
# ---------------------------------------------------------------------------
def test_wsl_unc_dollar(monkeypatch):
    _posix(monkeypatch)
    result = normalize_path_input(r"\\wsl$\Ubuntu\home\foo\paper.pdf")
    assert result == "/home/foo/paper.pdf"


# ---------------------------------------------------------------------------
# 6. WSL UNC path with .localhost
# ---------------------------------------------------------------------------
def test_wsl_unc_localhost(monkeypatch):
    _posix(monkeypatch)
    result = normalize_path_input(r"\\wsl.localhost\Ubuntu\home\foo\paper.pdf")
    assert result == "/home/foo/paper.pdf"


# ---------------------------------------------------------------------------
# 7. Already-canonical POSIX path — unchanged
# ---------------------------------------------------------------------------
def test_posix_path_unchanged(monkeypatch):
    _posix(monkeypatch)
    path = "/mnt/c/Users/Foo/paper.pdf"
    assert normalize_path_input(path) == path


# ---------------------------------------------------------------------------
# 8. Relative path — unchanged
# ---------------------------------------------------------------------------
def test_relative_path_unchanged(monkeypatch):
    _posix(monkeypatch)
    path = "./paper.pdf"
    assert normalize_path_input(path) == path


# ---------------------------------------------------------------------------
# 9. Tilde-home expanded on POSIX
# ---------------------------------------------------------------------------
def test_tilde_home_expanded(monkeypatch):
    _posix(monkeypatch)
    result = normalize_path_input("~/paper.pdf")
    expected = os.path.expanduser("~/paper.pdf")
    assert result == expected
    assert not result.startswith("~")


# ---------------------------------------------------------------------------
# 10. arXiv ID — unchanged
# ---------------------------------------------------------------------------
def test_arxiv_id_unchanged(monkeypatch):
    _posix(monkeypatch)
    arxiv_id = "2512.24601"
    assert normalize_path_input(arxiv_id) == arxiv_id


# ---------------------------------------------------------------------------
# 11. arXiv URL — unchanged
# ---------------------------------------------------------------------------
def test_arxiv_url_unchanged(monkeypatch):
    _posix(monkeypatch)
    url = "https://arxiv.org/abs/2512.24601"
    assert normalize_path_input(url) == url


# ---------------------------------------------------------------------------
# 12. DOI — unchanged
# ---------------------------------------------------------------------------
def test_doi_unchanged(monkeypatch):
    _posix(monkeypatch)
    doi = "10.1234/abc.def"
    assert normalize_path_input(doi) == doi


# ---------------------------------------------------------------------------
# 13. Empty string → empty string
# ---------------------------------------------------------------------------
def test_empty_returns_empty():
    assert normalize_path_input("") == ""


# ---------------------------------------------------------------------------
# 14. None passes through (non-string handled gracefully)
# ---------------------------------------------------------------------------
def test_none_passes_through():
    # The function signature says str but the guard handles None.
    result = normalize_path_input(None)  # type: ignore[arg-type]
    assert result is None


# ---------------------------------------------------------------------------
# 15. Idempotency — applying twice yields same result as once
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [
    r"C:\Users\Foo\paper.pdf",
    "C:/Users/Foo/paper.pdf",
    "/mnt/c/Users/Foo/paper.pdf",
    "/home/foo/paper.pdf",
    "2512.24601",
    "https://arxiv.org/abs/2512.24601",
    "10.1234/abc.def",
    "",
])
def test_idempotent(monkeypatch, value):
    _posix(monkeypatch)
    monkeypatch.setattr("backend.services.paths.shutil.which", lambda _: None)
    once = normalize_path_input(value)
    twice = normalize_path_input(once)
    assert once == twice, f"Not idempotent for {value!r}: {once!r} → {twice!r}"


# ---------------------------------------------------------------------------
# Bonus: wslpath binary path — mock subprocess so CI doesn't need wslpath
# ---------------------------------------------------------------------------
def test_windows_path_via_wslpath_mock(monkeypatch):
    """When wslpath is present and succeeds, its output is used."""
    _posix(monkeypatch)
    monkeypatch.setattr("backend.services.paths.shutil.which", lambda _: "/usr/bin/wslpath")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "/mnt/c/Users/Foo/paper.pdf\n"
    with patch("backend.services.paths.subprocess.run", return_value=mock_result) as mock_run:
        result = normalize_path_input(r"C:\Users\Foo\paper.pdf")
    assert result == "/mnt/c/Users/Foo/paper.pdf"
    mock_run.assert_called_once()


def test_wslpath_failure_falls_back_to_python(monkeypatch):
    """When wslpath fails, pure-Python conversion is used."""
    _posix(monkeypatch)
    monkeypatch.setattr("backend.services.paths.shutil.which", lambda _: "/usr/bin/wslpath")
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    with patch("backend.services.paths.subprocess.run", return_value=mock_result):
        result = normalize_path_input(r"C:\Users\Foo\paper.pdf")
    assert result == "/mnt/c/Users/Foo/paper.pdf"
