"""Tests for agent runtime provider resolution."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from backend.agents.runtime import (
    configure_openai_agents_sdk_credentials,
    has_provider_credentials,
    ProviderConfigurationError,
    make_runtime,
    selected_provider,
    validate_provider_credentials,
)


def test_selected_provider_accepts_aliases(monkeypatch) -> None:
    monkeypatch.delenv("OPENRESEARCH_LLM_PROVIDER", raising=False)

    assert selected_provider("anthropic") == "anthropic"
    assert selected_provider("claude") == "anthropic"
    assert selected_provider("openai") == "openai"
    assert selected_provider("oai") == "openai"


def test_selected_provider_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_LLM_PROVIDER", "openai")

    assert selected_provider() == "openai"


def test_selected_provider_rejects_unknown() -> None:
    with pytest.raises(ProviderConfigurationError):
        selected_provider("local")


def test_validate_provider_credentials_checks_matching_env(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    # The Anthropic check also passes when the `claude` CLI is on PATH (it
    # inherits subscription auth); pin it absent so this test deterministically
    # exercises the missing-credentials path regardless of the host environment.
    monkeypatch.setattr("shutil.which", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(
            anthropic_api_key="",
            openai_api_key="",
            openai_admin_key="",
        ),
    )
    with pytest.raises(ProviderConfigurationError):
        validate_provider_credentials("anthropic")
    with pytest.raises(ProviderConfigurationError):
        validate_provider_credentials("openai")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert validate_provider_credentials("anthropic") == "anthropic"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert validate_provider_credentials("openai") == "openai"


def test_validate_provider_credentials_accepts_reprolab_openai_key(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(openai_api_key="sk-reprolab", openai_admin_key=""),
    )

    assert validate_provider_credentials("openai") == "openai"


def test_configure_openai_agents_sdk_credentials_sets_default_key(
    monkeypatch,
) -> None:
    configured: list[tuple[str, bool]] = []
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(openai_api_key="sk-reprolab", openai_admin_key=""),
    )

    configure_openai_agents_sdk_credentials(
        lambda key, *, use_for_tracing=True: configured.append(
            (key, use_for_tracing)
        )
    )

    assert configured == [("sk-reprolab", True)]
    assert os.environ["OPENAI_API_KEY"] == "sk-reprolab"


def test_make_runtime_instantiates_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)

    assert make_runtime("anthropic").provider_name == "anthropic"
    assert make_runtime("openai").provider_name == "openai"


# ---------------------------------------------------------------------------
# has_provider_credentials — predicate sibling of validate_provider_credentials.
# Used by the orchestrator to filter the fallback chain so a run with only
# anthropic credentials never tries to fall over to openai (the symptom: a
# misleading 401 from openai that kills the run on an unrelated anthropic
# blip).
# ---------------------------------------------------------------------------

def test_has_provider_credentials_anthropic_via_env(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xyz")
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(anthropic_api_key="", openai_api_key="", openai_admin_key=""),
    )
    assert has_provider_credentials("anthropic") is True


def test_has_provider_credentials_anthropic_via_settings(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(anthropic_api_key="sk-from-env-file", openai_api_key="", openai_admin_key=""),
    )
    # Without claude CLI in our isolated path the only signal is the settings key.
    monkeypatch.setattr("backend.agents.runtime.factory.shutil.which", lambda _: None)
    assert has_provider_credentials("anthropic") is True


def test_has_provider_credentials_anthropic_via_claude_cli(monkeypatch) -> None:
    """Symptom: CLI presence alone returned True even when the OAuth session was absent.

    has_provider_credentials must check that the credentials file written by
    `claude login` exists, not just that the binary is on PATH (handoff P2-I11 / T23).
    Verify: with both the binary and the credentials file present, returns True.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(anthropic_api_key="", openai_api_key="", openai_admin_key=""),
    )
    monkeypatch.setattr("backend.agents.runtime.factory.shutil.which", lambda name: "/opt/bin/claude" if name == "claude" else None)
    # Also mock the credentials file check — `claude login` writes this file.
    monkeypatch.setattr("backend.agents.runtime.factory.os.path.isfile", lambda _: True)
    # Subscription login via the CLI counts as credentials when creds file exists.
    assert has_provider_credentials("anthropic") is True


def test_has_provider_credentials_anthropic_rejects_logged_out_cli(monkeypatch) -> None:
    """Symptom: presence of `claude` CLI on PATH passes the credential check even when logged out.

    has_provider_credentials returned True on CLI presence alone (handoff P2-I11 / T23).
    Verify: with no ANTHROPIC_API_KEY and a CLI binary present but no credentials
    file AND no macOS Keychain entry (i.e. logged-out session), returns False.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(anthropic_api_key="", openai_api_key="", openai_admin_key=""),
    )
    monkeypatch.setattr("backend.agents.runtime.factory.shutil.which", lambda name: "/opt/bin/claude" if name == "claude" else None)
    # Credentials file absent — not logged in.
    monkeypatch.setattr("backend.agents.runtime.factory.os.path.isfile", lambda _: False)
    # And no Keychain entry either (macOS path). Forcing a non-darwin platform
    # short-circuits the Keychain branch entirely; this test is platform-agnostic.
    monkeypatch.setattr("backend.agents.runtime.factory.sys.platform", "linux")
    assert has_provider_credentials("anthropic") is False


# ---------------------------------------------------------------------------
# 2026-05-23 macOS Keychain OAuth detection — modern `claude login` on macOS
# stores credentials in the Keychain, not in ~/.claude/.credentials.json.
# Before this fix, has_provider_credentials returned False on every macOS dev
# machine with Claude Code logged in, so the sub-agent runtime resolved as
# `unresolved` and every implement_baseline call died with a credential error.
# These tests pin the Keychain probe contract.
# ---------------------------------------------------------------------------

def test_has_provider_credentials_anthropic_via_macos_keychain(monkeypatch) -> None:
    """macOS path: claude CLI present + no creds file + Keychain entry exists → True.

    `claude login` on modern macOS writes a `genp` keychain item with service
    name "Claude Code-credentials". We probe via `security find-generic-password
    -s "Claude Code-credentials"` — exit code 0 means the entry exists.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(anthropic_api_key="", openai_api_key="", openai_admin_key=""),
    )
    monkeypatch.setattr("backend.agents.runtime.factory.shutil.which", lambda name: "/opt/bin/claude" if name == "claude" else None)
    # No credentials file on disk — modern macOS Keychain storage only.
    monkeypatch.setattr("backend.agents.runtime.factory.os.path.isfile", lambda _: False)
    monkeypatch.setattr("backend.agents.runtime.factory.sys.platform", "darwin")
    # Fake `security` exit 0 — the keychain entry exists.
    monkeypatch.setattr(
        "backend.agents.runtime.factory.subprocess.run",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
    )
    assert has_provider_credentials("anthropic") is True


def test_has_provider_credentials_anthropic_rejects_missing_keychain_entry(monkeypatch) -> None:
    """macOS path: claude CLI present + no creds file + no Keychain entry → False.

    `security find-generic-password` exits 44 when the requested item is not
    found in the Keychain; any non-zero exit code must read as "no session".
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(anthropic_api_key="", openai_api_key="", openai_admin_key=""),
    )
    monkeypatch.setattr("backend.agents.runtime.factory.shutil.which", lambda name: "/opt/bin/claude" if name == "claude" else None)
    monkeypatch.setattr("backend.agents.runtime.factory.os.path.isfile", lambda _: False)
    monkeypatch.setattr("backend.agents.runtime.factory.sys.platform", "darwin")
    monkeypatch.setattr(
        "backend.agents.runtime.factory.subprocess.run",
        lambda *_a, **_kw: SimpleNamespace(returncode=44, stdout=b"", stderr=b"item not found"),
    )
    assert has_provider_credentials("anthropic") is False


def test_has_provider_credentials_linux_skips_keychain_probe(monkeypatch) -> None:
    """Linux path: the Keychain branch is darwin-only; never invoke `security` on Linux.

    Guards against accidentally running the `security` subprocess on a Linux
    container — the binary doesn't exist there and even a FileNotFoundError
    would slow CI down. With no creds file and platform=linux, the helper
    must short-circuit to False without touching subprocess.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(anthropic_api_key="", openai_api_key="", openai_admin_key=""),
    )
    monkeypatch.setattr("backend.agents.runtime.factory.shutil.which", lambda name: "/opt/bin/claude" if name == "claude" else None)
    monkeypatch.setattr("backend.agents.runtime.factory.os.path.isfile", lambda _: False)
    monkeypatch.setattr("backend.agents.runtime.factory.sys.platform", "linux")

    # If anything reaches subprocess.run, this raises and the test fails.
    def _boom(*_a, **_kw):
        raise AssertionError("subprocess.run must not be called on non-darwin")
    monkeypatch.setattr("backend.agents.runtime.factory.subprocess.run", _boom)

    assert has_provider_credentials("anthropic") is False


def test_has_provider_credentials_anthropic_keychain_probe_timeout(monkeypatch) -> None:
    """A hung Keychain daemon must not block pipeline startup.

    ``subprocess.TimeoutExpired`` raised by the security probe is caught and
    treated as "no session" — never propagates.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(anthropic_api_key="", openai_api_key="", openai_admin_key=""),
    )
    monkeypatch.setattr("backend.agents.runtime.factory.shutil.which", lambda name: "/opt/bin/claude" if name == "claude" else None)
    monkeypatch.setattr("backend.agents.runtime.factory.os.path.isfile", lambda _: False)
    monkeypatch.setattr("backend.agents.runtime.factory.sys.platform", "darwin")

    def _hang(*_a, **kw):
        # subprocess.run raises TimeoutExpired when timeout= elapses.
        raise subprocess_module.TimeoutExpired(cmd=["security"], timeout=kw.get("timeout", 5))
    import subprocess as subprocess_module
    monkeypatch.setattr("backend.agents.runtime.factory.subprocess.run", _hang)

    assert has_provider_credentials("anthropic") is False


def test_has_provider_credentials_anthropic_returns_false_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(anthropic_api_key="", openai_api_key="", openai_admin_key=""),
    )
    monkeypatch.setattr("backend.agents.runtime.factory.shutil.which", lambda _: None)
    assert has_provider_credentials("anthropic") is False


def test_has_provider_credentials_openai_via_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(anthropic_api_key="", openai_api_key="", openai_admin_key=""),
    )
    assert has_provider_credentials("openai") is True


def test_has_provider_credentials_openai_via_admin_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-admin")
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(anthropic_api_key="", openai_api_key="", openai_admin_key=""),
    )
    assert has_provider_credentials("openai") is True


def test_has_provider_credentials_openai_returns_false_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(anthropic_api_key="", openai_api_key="", openai_admin_key=""),
    )
    assert has_provider_credentials("openai") is False


def test_has_provider_credentials_invalid_provider_returns_false() -> None:
    assert has_provider_credentials("local") is False
