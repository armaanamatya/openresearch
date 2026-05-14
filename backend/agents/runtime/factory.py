"""Runtime provider factory."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from typing import Any, cast

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
    settings = get_settings(_force_reload=True)
    if resolved == "anthropic":
        has_api_key = bool(
            os.getenv("ANTHROPIC_API_KEY")
            or getattr(settings, "anthropic_api_key", "")
        )
        # claude-agent-sdk spawns the `claude` CLI as a subprocess and inherits
        # its subscription login — no API key needed when the CLI is on PATH.
        has_claude_cli = shutil.which("claude") is not None
        if not (has_api_key or has_claude_cli):
            raise ProviderConfigurationError(
                provider=resolved,
                reason=(
                    "Anthropic credentials are missing; set ANTHROPIC_API_KEY "
                    "or REPROLAB_ANTHROPIC_API_KEY, or install and log in to "
                    "the Claude Code CLI (`claude login`)."
                ),
            )
        return resolved
    if not (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_ADMIN_KEY")
        or getattr(settings, "openai_api_key", "")
        or getattr(settings, "openai_admin_key", "")
    ):
        raise ProviderConfigurationError(
            provider=resolved,
            reason=(
                "OpenAI credentials are missing; set OPENAI_API_KEY "
                "or REPROLAB_OPENAI_API_KEY, or set OPENAI_ADMIN_KEY "
                "or REPROLAB_OPENAI_ADMIN_KEY"
            ),
        )
    return resolved


def configure_openai_agents_sdk_credentials(
    set_default_openai_key: Callable[..., Any] | None,
) -> None:
    """Bridge ReproLab settings into the OpenAI Agents SDK credential hooks.

    The OpenAI Agents SDK can read ``OPENAI_API_KEY`` / ``OPENAI_ADMIN_KEY``
    directly. ReproLab also supports ``REPROLAB_*`` aliases and ``.env`` via
    Settings, so this function performs the provider-specific bridge once,
    before any Agents SDK objects are constructed.
    """
    validate_provider_credentials("openai")
    settings = get_settings(_force_reload=True)
    api_key = (settings.openai_api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        if set_default_openai_key is not None:
            try:
                set_default_openai_key(api_key, use_for_tracing=True)
            except TypeError:
                set_default_openai_key(api_key)
    admin_key = (
        settings.openai_admin_key
        or os.getenv("OPENAI_ADMIN_KEY")
        or ""
    ).strip()
    if admin_key:
        os.environ["OPENAI_ADMIN_KEY"] = admin_key


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


__all__ = [
    "configure_openai_agents_sdk_credentials",
    "make_runtime",
    "selected_provider",
    "validate_provider_credentials",
]
