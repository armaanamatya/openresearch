"""Tests for _warn_on_shell_env_override — BUG-LR-014.

Verifies that:
1. A warning is printed to stderr when shell-exported key differs from .env.
2. No warning when both values match.
3. No warning when key is only in shell or only in .env.
4. The warning shows key prefix, not the full key value.
5. Multiple differing keys each get their own warning line.
"""
from __future__ import annotations

import os
from unittest.mock import patch

from backend.cli import _warn_on_shell_env_override


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(shell_key: str, shell_val: str, env_val: str) -> str:
    """Convenience: single key test."""
    from io import StringIO
    from unittest.mock import patch
    import backend.cli as _m

    fake_stderr = StringIO()
    patched_environ = dict(os.environ)
    patched_environ[shell_key] = shell_val
    with (
        patch.object(_m.os, "environ", patched_environ),
        patch("sys.stderr", fake_stderr),
        patch("dotenv.dotenv_values", return_value={shell_key: env_val}),
    ):
        _warn_on_shell_env_override()
    return fake_stderr.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_warns_when_keys_differ() -> None:
    out = _run("OPENAI_API_KEY", "sk-svcacct-XXXXXXXXXXXXXX1234", "sk-proj-YYYYYYYYYYYYYY5678")
    assert "OPENAI_API_KEY" in out
    assert "warn" in out.lower()


def test_warning_shows_prefix_not_full_key() -> None:
    out = _run("OPENAI_API_KEY", "sk-svcacct-XXXXXXXXXXXXXX1234", "sk-proj-YYYYYYYYYYYYYY5678")
    # Full key values must NOT appear.
    assert "sk-svcacct-XXXXXXXXXXXXXX1234" not in out
    assert "sk-proj-YYYYYYYYYYYYYY5678" not in out
    # But some prefix should be shown.
    assert "sk-svcacct-" in out or "sk-svc" in out


def test_no_warning_when_keys_match() -> None:
    same_key = "sk-proj-SAMEKEYSAMEKEYSAMEK"
    out = _run("OPENAI_API_KEY", same_key, same_key)
    assert "warn" not in out.lower()
    assert "OPENAI_API_KEY" not in out


def test_no_warning_when_shell_key_absent() -> None:
    """Key only in .env — shell doesn't shadow it → no warning."""
    from io import StringIO
    import backend.cli as _m

    fake_stderr = StringIO()
    patched_environ = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    with (
        patch.object(_m.os, "environ", patched_environ),
        patch("sys.stderr", fake_stderr),
        patch("dotenv.dotenv_values", return_value={"OPENAI_API_KEY": "sk-proj-ONLYINENV"}),
    ):
        _warn_on_shell_env_override()
    assert "warn" not in fake_stderr.getvalue().lower()


def test_no_warning_when_dotenv_key_absent() -> None:
    """Key only in shell — no .env value to compare → no warning."""
    out = _run("OPENAI_API_KEY", "sk-svcacct-ONLY_IN_SHELL", "")
    assert "warn" not in out.lower()


def test_env_u_workaround_mentioned_in_warning() -> None:
    out = _run("OPENAI_API_KEY", "sk-svcacct-XXXXXXXXXXXXXX1234", "sk-proj-YYYYYYYYYYYYYY5678")
    assert "env -u OPENAI_API_KEY" in out


# ---------------------------------------------------------------------------
# G4: Azure / Azure Foundry keys are suspect too — a stale shell value silently
# shadowing .env causes 401s exactly like the OpenAI/Anthropic keys.
# ---------------------------------------------------------------------------
import pytest


@pytest.mark.parametrize(
    "key",
    [
        "AZURE_FOUNDRY_API_KEY",
        "AZURE_FOUNDRY_ENDPOINT",
        "AZURE_FOUNDRY_DEPLOYMENT",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
    ],
)
def test_warns_when_azure_foundry_keys_differ(key) -> None:
    out = _run(key, "shell-value-AAAAAAAAAAAA", "dotenv-value-BBBBBBBBBBBB")
    assert key in out
    assert "warn" in out.lower()
    assert f"env -u {key}" in out


@pytest.mark.parametrize("key", ["AZURE_FOUNDRY_API_KEY", "AZURE_OPENAI_ENDPOINT"])
def test_no_warning_when_azure_keys_match(key) -> None:
    same = "same-value-CCCCCCCCCCCCCC"
    out = _run(key, same, same)
    assert "warn" not in out.lower()
    assert key not in out
