"""Tests for agent runtime provider resolution."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from backend.agents.runtime import (
    configure_openai_agents_sdk_credentials,
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
