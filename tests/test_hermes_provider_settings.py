"""Settings-driven Hermes provider tests.

The contract these lock in:

1. ``is_available()`` returns True iff the relevant key is present in
   ``Settings`` AND the SDK module is importable. It does NOT consult
   ``os.environ`` directly.
2. ``call()`` passes the key explicitly to the SDK constructor, so the
   SDK does not fall back to its own ``os.environ`` lookup either.
3. Both behaviors hold even when ``os.environ`` has no provider keys at
   all — the values come from pydantic-settings reading ``.env`` (or
   from an explicit ``api_key=`` constructor override in tests).

This is what makes Hermes work without a docker rebuild or any shell
``source .env`` step: the .env file alone is sufficient.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from backend.config import get_settings
from backend.hermes_audit.providers import (
    ClaudeAuditProvider,
    ClaudeCodeSdkProvider,
    OpenAIAuditProvider,
)


@pytest.fixture(autouse=True)
def scrub_provider_env(monkeypatch: pytest.MonkeyPatch):
    """Strip every os.environ var that any of these tests' assertions
    depend on. Asserts the post-condition that the providers do NOT use
    os.environ — if any test only passes because the host shell happens
    to have ANTHROPIC_API_KEY set, we want it to fail loudly."""
    for name in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "REPROLAB_ANTHROPIC_API_KEY",
        "REPROLAB_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    # Force a fresh Settings construction so we don't leak the cached
    # one from any prior test that read .env.
    get_settings(_force_reload=True)
    yield
    # Restore so other test files see a clean slate.
    get_settings(_force_reload=True)


def test_openai_is_available_with_explicit_key_and_no_env(monkeypatch: pytest.MonkeyPatch):
    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **_: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    provider = OpenAIAuditProvider(api_key="sk-fake-test")

    assert provider.is_available() is True


def test_openai_is_unavailable_with_empty_key(monkeypatch: pytest.MonkeyPatch):
    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **_: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    provider = OpenAIAuditProvider(api_key="")

    assert provider.is_available() is False


def test_openai_is_unavailable_when_sdk_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "openai", None)  # pyright: ignore[reportArgumentType]

    provider = OpenAIAuditProvider(api_key="sk-fake-test")

    assert provider.is_available() is False


def test_openai_call_passes_api_key_to_sdk(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, *, api_key: str, **_: Any) -> None:
            captured["api_key"] = api_key
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content='{"status":"grounded"}')
                            )
                        ]
                    )
                )
            )

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    provider = OpenAIAuditProvider(api_key="sk-explicit-key")
    response = provider.call("audit this please")

    assert captured["api_key"] == "sk-explicit-key"
    assert response == '{"status":"grounded"}'


def test_claude_is_available_with_explicit_key_and_no_env(monkeypatch: pytest.MonkeyPatch):
    fake_anthropic = ModuleType("anthropic")
    fake_anthropic.Anthropic = lambda **_: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    provider = ClaudeAuditProvider(api_key="sk-ant-fake")

    assert provider.is_available() is True


def test_claude_call_passes_api_key_to_sdk(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, *, api_key: str, **_: Any) -> None:
            captured["api_key"] = api_key
            self.messages = SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    content=[SimpleNamespace(text='{"status":"grounded"}')]
                )
            )

    fake_anthropic = ModuleType("anthropic")
    fake_anthropic.Anthropic = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    provider = ClaudeAuditProvider(api_key="sk-ant-explicit")
    response = provider.call("audit this please")

    assert captured["api_key"] == "sk-ant-explicit"
    assert response == '{"status":"grounded"}'


def test_settings_picks_up_unprefixed_alias(monkeypatch: pytest.MonkeyPatch):
    """Critical contract for non-docker host runs: pydantic-settings
    reads ``ANTHROPIC_API_KEY``/``OPENAI_API_KEY`` out of .env (or shell
    env) directly via the AliasChoices on the Settings field. No
    REPROLAB_ prefix needed, no os.environ bootstrap needed."""

    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell-env")
    settings = get_settings(_force_reload=True)

    assert settings.openai_api_key == "sk-from-shell-env"


def test_settings_falls_back_to_reprolab_prefix(monkeypatch: pytest.MonkeyPatch):
    """Operators who have OPENAI_API_KEY reserved at the shell level for
    a different scope can use REPROLAB_OPENAI_API_KEY without conflict."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("REPROLAB_OPENAI_API_KEY", "sk-reprolab-scoped")
    settings = get_settings(_force_reload=True)

    assert settings.openai_api_key == "sk-reprolab-scoped"


# ---------------------------------------------------------------------------
# ClaudeCodeSdkProvider — uses claude_agent_sdk with Claude Code session auth
# instead of an Anthropic API key. Tests verify availability gating on the
# SDK module's presence and that ``call()`` correctly drives the async
# ``query`` iterator from a sync caller.
# ---------------------------------------------------------------------------


def test_claude_code_sdk_unavailable_when_module_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)  # pyright: ignore[reportArgumentType]

    provider = ClaudeCodeSdkProvider()

    assert provider.is_available() is False


def test_claude_code_sdk_available_when_module_present(monkeypatch: pytest.MonkeyPatch):
    fake_sdk = ModuleType("claude_agent_sdk")
    fake_sdk.ClaudeAgentOptions = lambda **_: None  # type: ignore[attr-defined]
    fake_sdk.ResultMessage = type("ResultMessage", (), {})  # type: ignore[attr-defined]
    fake_sdk.query = lambda **_: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)

    provider = ClaudeCodeSdkProvider()

    assert provider.is_available() is True


def test_claude_code_sdk_call_collects_async_iterator(monkeypatch: pytest.MonkeyPatch):
    """``call()`` is sync but the SDK's query() is an async iterator.
    Verify the provider drains the iterator and returns concatenated text."""

    captured_options: dict[str, Any] = {}

    class FakeOptions:
        def __init__(self, **kwargs: Any) -> None:
            captured_options.update(kwargs)

    class FakeResultMessage:
        def __init__(self, text: str) -> None:
            self.text = text

    async def fake_query(*, prompt: str, options: Any):
        yield FakeResultMessage('{"status":"grounded"}')

    fake_sdk = ModuleType("claude_agent_sdk")
    fake_sdk.ClaudeAgentOptions = FakeOptions  # type: ignore[attr-defined]
    fake_sdk.ResultMessage = FakeResultMessage  # type: ignore[attr-defined]
    fake_sdk.query = fake_query  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)

    provider = ClaudeCodeSdkProvider()

    response = provider.call("audit this please")

    assert response == '{"status":"grounded"}'
    assert captured_options["permission_mode"] == "bypassPermissions"
    assert captured_options["max_turns"] == 1


def test_claude_code_sdk_call_raises_on_empty_response(monkeypatch: pytest.MonkeyPatch):
    """Empty stream is treated as a failure (so the chain falls through
    to the next provider) — the alternative would be returning '' which
    extract_audit_json would surface as an opaque parse error."""

    async def fake_query(*, prompt: str, options: Any):
        if False:  # never yields
            yield None

    fake_sdk = ModuleType("claude_agent_sdk")
    fake_sdk.ClaudeAgentOptions = lambda **_: None  # type: ignore[attr-defined]
    fake_sdk.ResultMessage = type("ResultMessage", (), {})  # type: ignore[attr-defined]
    fake_sdk.query = fake_query  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)

    provider = ClaudeCodeSdkProvider()

    with pytest.raises(RuntimeError, match="empty response"):
        provider.call("audit")
