"""Unit tests for WSL-aware Claude OAuth credential discovery.

Tests cover:
- POSIX home detection
- macOS Keychain detection
- WSL diagnostic warning when credentials exist on Windows side
- _scan_wsl_windows_credentials path scanning logic
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


from backend.agents.runtime.factory import (
    _has_claude_subscription_oauth,
    _scan_wsl_windows_credentials,
)


# ---------------------------------------------------------------------------
# _has_claude_subscription_oauth — POSIX home
# ---------------------------------------------------------------------------


def test_oauth_detected_in_posix_home(tmp_path):
    """Returns True when ~/.claude/.credentials.json exists."""
    creds = tmp_path / ".credentials.json"
    creds.write_text("{}")
    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("os.path.isfile", return_value=True),
    ):
        assert _has_claude_subscription_oauth() is True


def test_oauth_missing_in_posix_home():
    """Returns False when no credentials exist anywhere."""
    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("os.path.isfile", return_value=False),
        patch(
            "backend.agents.runtime.factory._scan_wsl_windows_credentials",
            return_value=None,
        ),
        # Ensure we don't hit darwin path on non-darwin
        patch.object(sys, "platform", "linux"),
    ):
        assert _has_claude_subscription_oauth() is False


# ---------------------------------------------------------------------------
# _has_claude_subscription_oauth — macOS Keychain
# ---------------------------------------------------------------------------


def test_macos_keychain_check_when_on_darwin():
    """Returns True on darwin when Keychain has the credential entry (exit 0)."""
    mock_result = MagicMock()
    mock_result.returncode = 0

    with (
        patch("shutil.which", return_value="/usr/local/bin/claude"),
        patch("os.path.isfile", return_value=False),
        patch.object(sys, "platform", "darwin"),
        patch(
            "subprocess.run",
            return_value=mock_result,
        ),
    ):
        assert _has_claude_subscription_oauth() is True


def test_macos_keychain_returns_false_on_nonzero():
    """Returns False on darwin when Keychain lookup exits non-zero."""
    mock_result = MagicMock()
    mock_result.returncode = 44

    with (
        patch("shutil.which", return_value="/usr/local/bin/claude"),
        patch("os.path.isfile", return_value=False),
        patch.object(sys, "platform", "darwin"),
        patch("subprocess.run", return_value=mock_result),
    ):
        assert _has_claude_subscription_oauth() is False


# ---------------------------------------------------------------------------
# _has_claude_subscription_oauth — WSL diagnostic
# ---------------------------------------------------------------------------


def test_wsl_diagnostic_warning_when_windows_creds_exist(caplog):
    """Returns False (not usable) but logs a WARNING with symlink command when
    Windows-side creds are found."""
    import logging

    windows_path = "/mnt/c/Users/foo/.claude/.credentials.json"

    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("os.path.isfile", return_value=False),
        patch.object(sys, "platform", "linux"),
        patch(
            "backend.agents.runtime.factory._scan_wsl_windows_credentials",
            return_value=windows_path,
        ),
    ):
        with caplog.at_level(logging.WARNING, logger="backend.agents.runtime.factory"):
            result = _has_claude_subscription_oauth()

    assert result is False
    assert windows_path in caplog.text
    # The actionable symlink command must be present.
    assert "ln -s" in caplog.text


def test_wsl_returns_false_when_no_creds_anywhere(caplog):
    """Returns False and logs nothing when _scan_wsl_windows_credentials returns None."""
    import logging

    with (
        patch("shutil.which", return_value="/usr/bin/claude"),
        patch("os.path.isfile", return_value=False),
        patch.object(sys, "platform", "linux"),
        patch(
            "backend.agents.runtime.factory._scan_wsl_windows_credentials",
            return_value=None,
        ),
    ):
        with caplog.at_level(logging.WARNING, logger="backend.agents.runtime.factory"):
            result = _has_claude_subscription_oauth()

    assert result is False
    # No warning logged — nothing to point the user at.
    assert "ln -s" not in caplog.text
    assert "mnt/c" not in caplog.text


# ---------------------------------------------------------------------------
# _scan_wsl_windows_credentials
# ---------------------------------------------------------------------------


def test_scan_wsl_credentials_finds_user_dir():
    """Returns the first matching Windows-side credentials path."""
    with (
        patch(
            "backend.agents.runtime.factory._scan_wsl_windows_credentials.__wrapped__"
            if hasattr(_scan_wsl_windows_credentials, "__wrapped__") else
            "backend.agents.execution._is_wsl",
            return_value=True,
        ),
        patch("os.path.isdir", return_value=True),
        patch("os.listdir", return_value=["foo", "Public"]),
        patch(
            "os.path.isfile",
            side_effect=lambda p: p == "/mnt/c/Users/foo/.claude/.credentials.json",
        ),
    ):
        # We need _is_wsl to return True inside _scan_wsl_windows_credentials.
        with patch("backend.agents.execution._is_wsl", return_value=True):
            result = _scan_wsl_windows_credentials()

    assert result == "/mnt/c/Users/foo/.claude/.credentials.json"


def test_scan_skips_system_users():
    """Returns None when only Windows pseudo-users exist (Public, Default, etc.)."""
    with (
        patch("backend.agents.execution._is_wsl", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch("os.listdir", return_value=["Public", "Default", "All Users"]),
        patch("os.path.isfile", return_value=False),
    ):
        result = _scan_wsl_windows_credentials()

    assert result is None


def test_scan_returns_none_when_mnt_c_users_missing():
    """Returns None when /mnt/c/Users does not exist (not on WSL or no C drive)."""
    with (
        patch("backend.agents.execution._is_wsl", return_value=True),
        patch("os.path.isdir", return_value=False),
    ):
        result = _scan_wsl_windows_credentials()

    assert result is None


def test_scan_returns_none_on_oserror():
    """Returns None gracefully when os.listdir raises OSError."""
    with (
        patch("backend.agents.execution._is_wsl", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch("os.listdir", side_effect=OSError("permission denied")),
    ):
        result = _scan_wsl_windows_credentials()

    assert result is None


def test_scan_returns_none_when_not_wsl():
    """Returns None immediately on non-WSL systems."""
    with patch("backend.agents.execution._is_wsl", return_value=False):
        result = _scan_wsl_windows_credentials()

    assert result is None
