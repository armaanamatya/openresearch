"""Runtime provider factory."""

from __future__ import annotations

import os
from typing import cast

from backend.agents.runtime.base import (
    AgentRuntime,
    ProviderConfigurationError,
    ProviderName,
)
from backend.config import get_settings


def selected_provider(provider: ProviderName | str | None = None) -> ProviderName:
    """Resolve the requested provider from argument, env, or settings."""
    raw = (
        provider
        or os.getenv("REPROLAB_LLM_PROVIDER")
        or get_settings().llm_provider
        or "anthropic"
    )
    normalized = str(raw).strip().lower()
    if normalized in {"anthropic", "claude"}:
        return "anthropic"
    if normalized in {"openai", "oai"}:
        return "openai"
    raise ProviderConfigurationError(
        provider=normalized,
        reason="expected 'anthropic' or 'openai'",
    )


def validate_provider_credentials(provider: ProviderName | str | None = None) -> ProviderName:
    """Validate API credentials for explicit SDK-mode invocations."""
    resolved = selected_provider(provider)
    env_name = "ANTHROPIC_API_KEY" if resolved == "anthropic" else "OPENAI_API_KEY"
    if not os.getenv(env_name):
        raise ProviderConfigurationError(
            provider=resolved,
            reason=f"{env_name} is not set",
        )
    return resolved


def make_runtime(
    provider: ProviderName | str | None = None,
    *,
    require_api_key: bool = False,
) -> AgentRuntime:
    """Create a provider runtime.

    Credentials are validated only when requested so tests and offline import
    paths can instantiate the orchestrator without requiring local secrets.
    """
    resolved = (
        validate_provider_credentials(provider)
        if require_api_key
        else selected_provider(provider)
    )
    if resolved == "anthropic":
        from backend.agents.runtime.claude_runtime import ClaudeAgentRuntime

        return ClaudeAgentRuntime()
    if resolved == "openai":
        from backend.agents.runtime.openai_runtime import OpenAiAgentRuntime

        return OpenAiAgentRuntime()
    raise ProviderConfigurationError(
        provider=cast(str, resolved),
        reason="unsupported provider",
    )


__all__ = ["make_runtime", "selected_provider", "validate_provider_credentials"]
