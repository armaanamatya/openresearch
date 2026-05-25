"""Runtime provider factory."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


def _scan_wsl_windows_credentials() -> str | None:
    """On WSL, find a Windows-side ``~/.claude/.credentials.json`` by scanning
    ``/mnt/c/Users/<*>/.claude/.credentials.json``.

    Returns the matching path or None if no candidate exists. Bounded to
    common user directory shapes — does not recurse beyond depth 1 under
    ``/mnt/c/Users/``.
    """
    from backend.agents.execution import _is_wsl  # local import — avoids circular at module level

    if not _is_wsl():
        return None
    users_root = "/mnt/c/Users"
    if not os.path.isdir(users_root):
        return None
    try:
        for entry in os.listdir(users_root):
            # Skip Windows-system pseudo-users.
            if entry.lower() in {"public", "default", "default user", "all users", "defaultaccount"}:
                continue
            candidate = os.path.join(users_root, entry, ".claude", ".credentials.json")
            if os.path.isfile(candidate):
                return candidate
    except OSError:
        return None
    return None


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

    On WSL, users frequently run ``claude login`` from Windows, placing
    credentials at ``C:\\Users\\<x>\\.claude\\.credentials.json``. The WSL
    home directory (``~``) is a different filesystem, so the SDK cannot read
    those credentials directly. When WSL-side credentials are absent, this
    function scans the Windows-mounted user profiles and emits an actionable
    warning with the symlink command the user should run.

    Returns True iff a stored session is detectable by the SDK (i.e. under
    ``~`` or ``%USERPROFILE%``). On Windows, the ``claude`` binary may not be
    on PATH (Git Bash doesn't inherit the Windows PATH entry for
    ``~/.local/bin``), but the SDK can still read the credentials file and
    locate the binary via known install paths.  Times out at 5 s on the
    Keychain probe to prevent a hung Keychain Access daemon from blocking
    pipeline startup.
    """
    # 1. POSIX user home — the canonical location the SDK reads from.
    if os.path.isfile(os.path.expanduser("~/.claude/.credentials.json")):
        # On non-Windows, also verify claude binary is available.
        if sys.platform == "win32" or shutil.which("claude") is not None:
            return True
    # 2. Windows: check USERPROFILE path (Git Bash ~ may differ from USERPROFILE).
    if sys.platform == "win32":
        userprofile = os.environ.get("USERPROFILE", "")
        if userprofile:
            win_creds = os.path.join(userprofile, ".claude", ".credentials.json")
            if os.path.isfile(win_creds):
                return True
    # 3. macOS Keychain (modern Claude Code on macOS).
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
    # 4. WSL diagnostic — surface a helpful warning if creds exist on the Windows side
    #    but are not reachable by the SDK (which reads from $HOME, not /mnt/c).
    if shutil.which("claude") is None and sys.platform != "win32":
        wsl_creds = _scan_wsl_windows_credentials()
        if wsl_creds is not None:
            logger.warning(
                "_has_claude_subscription_oauth: Claude OAuth credentials found at %s "
                "(Windows side) but the claude-agent-sdk reads from $HOME=%s, not /mnt/c. "
                "Either symlink them — `ln -s %s ~/.claude/.credentials.json` — or run "
                "`claude login` from inside WSL.",
                wsl_creds,
                os.path.expanduser("~"),
                wsl_creds,
            )
    return False


def _has_azure_openai_credentials() -> bool:
    """Return True iff both AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set.

    Azure OpenAI requires both an API key and a resource endpoint — unlike plain
    OpenAI where only the key is needed.  Neither alone is sufficient.
    """
    return bool(
        os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")
    )


def _has_anthropic_api_credentials() -> bool:
    """Return True iff an Anthropic API key is available via env or Settings.

    Reads both ``os.environ`` and ``settings.anthropic_api_key`` — pydantic-settings
    loads ``.env`` even when the env var has been deleted, so checking both keeps
    the answer accurate. Extracted as a helper so tests can patch this single
    function rather than having to clear both surfaces independently.
    """
    settings = get_settings()
    return bool(
        os.getenv("ANTHROPIC_API_KEY")
        or getattr(settings, "anthropic_api_key", "")
    )


def _has_openai_credentials() -> bool:
    """Return True iff OpenAI credentials are available via env or Settings.

    Either OPENAI_API_KEY or OPENAI_ADMIN_KEY satisfies the check.
    Extracted as a helper so tests can patch this single function — see
    ``_has_anthropic_api_credentials`` for the rationale.
    """
    settings = get_settings()
    return bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_ADMIN_KEY")
        or getattr(settings, "openai_api_key", "")
        or getattr(settings, "openai_admin_key", "")
    )


def _has_featherless_credentials() -> bool:
    """Return True iff a Featherless API key is available via env or Settings."""
    settings = get_settings()
    return bool(
        os.getenv("FEATHERLESS_API_KEY")
        or getattr(settings, "featherless_api_key", "")
    )


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
    # Azure OpenAI is not a ProviderName literal — validate separately.
    if str(provider or "").lower() in {"azure-openai", "azure_openai", "azure"}:
        if not _has_azure_openai_credentials():
            raise ProviderConfigurationError(
                provider="azure-openai",
                reason=(
                    "Azure OpenAI credentials are missing; set both "
                    "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
                ),
            )
        return "openai"  # closest ProviderName for callers that need one
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
        # Lane 4: enforce the production llm_auth_strategy gate.  Each value
        # narrows which credential surfaces are acceptable; an unsatisfiable
        # strategy fails fast here rather than silently degrading to the
        # wrong rate-limit pool at runtime.
        strategy = (getattr(settings, "llm_auth_strategy", "auto") or "auto").lower()
        if strategy == "api_only" and not has_api_key:
            raise ProviderConfigurationError(
                provider=resolved,
                reason=(
                    "REPROLAB_LLM_AUTH_STRATEGY=api_only requires a funded "
                    "ANTHROPIC_API_KEY but none was found.  Either set the API "
                    "key, or switch the strategy to 'auto' / 'oauth_only'."
                ),
            )
        if strategy == "oauth_only" and not has_claude_cli:
            raise ProviderConfigurationError(
                provider=resolved,
                reason=(
                    "REPROLAB_LLM_AUTH_STRATEGY=oauth_only requires a logged-in "
                    "Claude Code CLI subscription but none was detected.  Run "
                    "`claude login` or switch the strategy to 'auto' / 'api_only'."
                ),
            )
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
    # Azure OpenAI is not a ProviderName literal — handle it before the
    # selected_provider() call which would raise on unknown providers.
    if str(provider or "").lower() in {"azure-openai", "azure_openai", "azure"}:
        return _has_azure_openai_credentials()
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


def aggregate_auth_status() -> dict:
    """Aggregate provider credential availability for the /auth-status endpoint.

    Returns a dict matching the D1 response shape so the UI can enable/disable
    provider radio buttons without any LLM call or subprocess.
    """
    # All provider checks route through dedicated helpers so tests can patch a
    # single function per provider rather than having to clear env + Settings
    # independently. pydantic-settings loads .env even after monkeypatch.delenv.
    anthropic_api_key = _has_anthropic_api_credentials()
    oauth_available = _has_claude_subscription_oauth()
    openai_available = _has_openai_credentials()
    azure_available = _has_azure_openai_credentials()
    featherless_available = _has_featherless_credentials()

    # Determine defaults (first available wins, in priority order)
    default_root = "anthropic_oauth" if oauth_available else (
        "anthropic_api" if anthropic_api_key else (
            "openai_api" if openai_available else (
                "azure_openai" if azure_available else (
                    "featherless" if featherless_available else "anthropic_oauth"
                )
            )
        )
    )

    # Sub-agent auth: SDK uses ANTHROPIC_API_KEY or OAuth
    subagent_api = anthropic_api_key
    subagent_oauth = oauth_available
    default_subagent = "anthropic_oauth" if subagent_oauth else (
        "anthropic_api" if subagent_api else "anthropic_oauth"
    )

    return {
        "providers": {
            "anthropic_api": {
                "available": anthropic_api_key,
                "detail": "ANTHROPIC_API_KEY set" if anthropic_api_key else "ANTHROPIC_API_KEY missing",
            },
            "anthropic_oauth": {
                "available": oauth_available,
                "detail": "claude CLI subscription" if oauth_available else "claude login required",
            },
            "openai_api": {
                "available": openai_available,
                "detail": "OPENAI_API_KEY set" if openai_available else "OPENAI_API_KEY missing",
            },
            "azure_openai": {
                "available": azure_available,
                "detail": "Azure credentials set" if azure_available else "AZURE_OPENAI_API_KEY missing",
            },
            "featherless": {
                "available": featherless_available,
                "detail": "FEATHERLESS_API_KEY set" if featherless_available else "FEATHERLESS_API_KEY missing",
            },
        },
        "subagent_auth": {
            "anthropic_api": subagent_api,
            "anthropic_oauth": subagent_oauth,
        },
        "defaults": {
            "root_provider": default_root,
            "root_model": "sonnet",
            "subagent_auth": default_subagent,
        },
    }


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
    "_has_azure_openai_credentials",
    "_has_claude_subscription_oauth",
    "_scan_wsl_windows_credentials",
    "aggregate_auth_status",
    "configure_openai_agents_sdk_credentials",
    "has_provider_credentials",
    "make_runtime",
    "selected_provider",
    "validate_provider_credentials",
]
