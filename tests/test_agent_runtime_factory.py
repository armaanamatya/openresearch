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
    monkeypatch.delenv("REPROLAB_LLM_PROVIDER", raising=False)

    assert selected_provider("anthropic") == "anthropic"
    assert selected_provider("claude") == "anthropic"
    assert selected_provider("openai") == "openai"
    assert selected_provider("oai") == "openai"


def test_selected_provider_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("REPROLAB_LLM_PROVIDER", "openai")

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
    file (i.e. logged-out session), returns False.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda **_: SimpleNamespace(anthropic_api_key="", openai_api_key="", openai_admin_key=""),
    )
    monkeypatch.setattr("backend.agents.runtime.factory.shutil.which", lambda name: "/opt/bin/claude" if name == "claude" else None)
    # Credentials file absent — not logged in.
    monkeypatch.setattr("backend.agents.runtime.factory.os.path.isfile", lambda _: False)
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
