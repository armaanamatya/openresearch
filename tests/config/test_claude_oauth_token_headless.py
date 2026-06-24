"""
Tests for the Stream B headless claude-oauth credential path.

Verifies that:
- CLAUDE_CODE_OAUTH_TOKEN being set in the environment satisfies
  _has_claude_subscription_oauth() (i.e. the token alone is sufficient
  to resolve as a valid unattended root — no ~/.claude/.credentials.json
  is needed).
- The _warn_on_shell_env_override validator does NOT warn when
  CLAUDE_CODE_OAUTH_TOKEN is set (it is not in _SUSPECT_KEYS).
- Unattended-pod credential check works even when the credentials file
  is absent.
"""

from __future__ import annotations

import io
import os
import sys
from unittest.mock import patch

import pytest


def test_token_env_satisfies_oauth_check(monkeypatch):
    """CLAUDE_CODE_OAUTH_TOKEN being set must cause _has_claude_subscription_oauth to return True.

    This is the core Stream B invariant: an unattended pod with the long-lived
    token does not need ~/.claude/.credentials.json.
    """
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok_test_headless_abc123")
    # Ensure no credentials file shadows the env-var fast-path.
    with patch("os.path.isfile", return_value=False):
        from backend.agents.runtime import factory
        # Reload to pick up the monkeypatched env (the function reads os.environ live).
        result = factory._has_claude_subscription_oauth()
    assert result is True, (
        "_has_claude_subscription_oauth() should return True when "
        "CLAUDE_CODE_OAUTH_TOKEN is set, even without a credentials file."
    )


def test_no_token_no_creds_returns_false(monkeypatch):
    """When neither the token env nor the credentials file is present, return False."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    with patch("os.path.isfile", return_value=False):
        from backend.agents.runtime import factory
        # On non-macOS, non-Windows, non-WSL the function reaches the final return.
        # Patch sys.platform to a neutral value so no other branch fires.
        with patch.object(sys, "platform", "linux"):
            with patch.object(factory, "_scan_wsl_windows_credentials", return_value=None):
                result = factory._has_claude_subscription_oauth()
    assert result is False


def test_token_not_in_suspect_keys(monkeypatch):
    """CLAUDE_CODE_OAUTH_TOKEN must NOT trigger a shell-override warning.

    The key is intentionally set by operators in-cluster and must not emit
    a spurious 'shell value differs from .env' warning.

    We verify this by checking that CLAUDE_CODE_OAUTH_TOKEN is not in the
    _SUSPECT_KEYS tuple used by _warn_on_shell_env_override in cli.py.
    """
    import backend.cli as cli_module
    import inspect

    source = inspect.getsource(cli_module._warn_on_shell_env_override)
    # The function defines _SUSPECT_KEYS as a tuple literal — confirm the token
    # name is not present.
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in source, (
        "CLAUDE_CODE_OAUTH_TOKEN must not be listed in _SUSPECT_KEYS inside "
        "_warn_on_shell_env_override — it is intentionally operator-supplied "
        "in production (injected from the secret store) and must never trigger "
        "a spurious 'shell shadows .env' warning."
    )


def test_no_shell_warning_emitted_when_token_set(monkeypatch, tmp_path):
    """Even when CLAUDE_CODE_OAUTH_TOKEN differs between shell and .env,
    _warn_on_shell_env_override emits nothing to stderr about it."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok_in_shell_abc")

    import backend.cli as cli_module
    captured = io.StringIO()
    original_stderr = sys.stderr
    sys.stderr = captured
    try:
        # Use a .env that also has the token (different value) — a worst-case
        # scenario where both are set.  The validator must stay silent.
        with patch("dotenv.dotenv_values", return_value={"CLAUDE_CODE_OAUTH_TOKEN": "tok_in_dotenv_xyz"}):
            cli_module._warn_on_shell_env_override()
    finally:
        sys.stderr = original_stderr

    output = captured.getvalue()
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in output, (
        "CLAUDE_CODE_OAUTH_TOKEN must not appear in shell-override warnings — "
        "it is intentionally operator-supplied and is not a stale key."
    )


def test_credentials_file_still_works_without_token(monkeypatch, tmp_path):
    """Existing credentials-file path is unaffected by the Stream B change."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    creds_path = tmp_path / ".claude" / ".credentials.json"
    creds_path.parent.mkdir(parents=True)
    creds_path.write_text('{"token": "stub"}')

    from backend.agents.runtime import factory

    with patch.object(factory, "_scan_wsl_windows_credentials", return_value=None), \
         patch.object(sys, "platform", "linux"), \
         patch("shutil.which", return_value="/usr/local/bin/claude"), \
         patch("os.path.isfile", side_effect=lambda p: p == str(tmp_path / ".claude" / ".credentials.json")), \
         patch("os.path.expanduser", side_effect=lambda p: p.replace("~", str(tmp_path))):
        result = factory._has_claude_subscription_oauth()
    assert result is True, (
        "The credentials-file detection path must still work when "
        "CLAUDE_CODE_OAUTH_TOKEN is absent."
    )
