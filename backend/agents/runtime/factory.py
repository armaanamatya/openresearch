"""Runtime provider factory."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from typing import Any, cast

from backend.agents.runtime.base import (
    AgentRuntime,
    ProviderConfigurationError,
    ProviderName,
)
from backend.config import get_settings


def _has_claude_subscription_oauth() -> bool:
    """Detect a working Claude Code subscription that ``claude-agent-sdk`` can use.

    ``claude-agent-sdk`` spawns the ``claude`` CLI as a subprocess and inherits
    its OAuth session — so the predicate is "is there a valid stored session
    on this machine that the CLI can read?"  There are two storage backends:

    - **File-based** (Linux, older macOS, CI containers): ``claude login`` writes
      ``~/.claude/.credentials.json``. Existence of that file is sufficient proof.
    - **macOS Keychain** (modern Claude Code on macOS): the credentials are
      stored as a ``genp`` keychain item with service name
      ``"Claude Code-credentials"``. We probe via ``security
      find-generic-password -s "Claude Code-credentials"`` — exit code 0 means
      the entry exists.

    Before 2026-05-23 the file-only check returned False on every modern macOS
    install with Claude Code logged in via Keychain, so the SDK runtime
    resolved as ``unresolved`` and every ``implement_baseline`` sub-call died
    with a credential error. This helper closes that gap.

    Returns True iff the ``claude`` binary is on ``PATH`` AND a stored session
    is detectable. Times out at 5 s on the Keychain probe to prevent a hung
    Keychain Access daemon from blocking pipeline startup.
    """
    if shutil.which("claude") is None:
        return False
    if os.path.isfile(os.path.expanduser("~/.claude/.credentials.json")):
        return True
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-s", "Claude Code-credentials"],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0
    return False


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
        # its OAuth session. ``_has_claude_subscription_oauth`` handles both
        # storage backends (file on Linux/older macOS, Keychain on modern macOS).
        has_claude_cli = _has_claude_subscription_oauth()
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


def has_provider_credentials(provider: ProviderName | str | None = None) -> bool:
    """Non-raising counterpart to validate_provider_credentials.

    Returns True iff the resolved provider has working credentials in env or
    settings. Used by the orchestrator to filter the fallback chain so we
    don't surface a misleading 401 from a provider the operator never
    configured. ``validate_provider_credentials`` raises on missing creds
    which is the wrong shape for predicate use.
    """
    try:
        resolved = selected_provider(provider)
    except ProviderConfigurationError:
        return False
    settings = get_settings()
    if resolved == "anthropic":
        if os.getenv("ANTHROPIC_API_KEY") or getattr(settings, "anthropic_api_key", ""):
            return True
        # claude-agent-sdk inherits a subscription login from the `claude` CLI
        # subprocess. ``_has_claude_subscription_oauth`` handles both storage
        # backends (file on Linux/older macOS, Keychain on modern macOS).
        return _has_claude_subscription_oauth()
    if resolved == "openai":
        return bool(
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAI_ADMIN_KEY")
            or getattr(settings, "openai_api_key", "")
            or getattr(settings, "openai_admin_key", "")
        )
    return False


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
    "has_provider_credentials",
    "make_runtime",
    "selected_provider",
    "validate_provider_credentials",
]
