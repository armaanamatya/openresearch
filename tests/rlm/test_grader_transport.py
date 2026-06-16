"""Grader transport — sampler-capable, backwards-compatible (spec 2026-06-16 §A5).

Unit tests only — every underlying SDK client is mocked, no network/LLM.
Covers:
  * ``sample_completions`` falls back to N× ``complete`` on a client lacking
    ``complete_samples`` (complete called exactly n times).
  * ``sample_completions`` delegates to ``complete_samples`` when present.
  * ``OpenAILlmClient.complete_samples`` passes n/seed/temperature=0 to the SDK
    and returns n items; degrades to N sequential calls on a ``TypeError``.
  * ``AnthropicMessagesClient`` complete + complete_samples shape + usage.
  * ``build_grader_client`` returns the fallback unchanged when env unset, and a
    new client when ``REPROLAB_GRADER_BACKEND=anthropic``.
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.grader_transport import (
    build_grader_client,
    sample_completions,
)


# ---------------------------------------------------------------------------
# sample_completions — universal entry
# ---------------------------------------------------------------------------


class _CompleteOnlyClient:
    """A legacy client implementing ONLY the base ``complete`` protocol."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        return f"ans-{self.calls}"


class _SamplerClient:
    """A client exposing ``complete_samples`` — records what it was passed."""

    def __init__(self) -> None:
        self.seen: dict[str, object] = {}

    def complete(self, *, system: str, user: str) -> str:  # pragma: no cover
        raise AssertionError("complete must not be used when complete_samples exists")

    def complete_samples(self, *, system, user, n=1, temperature=None, seed=None):
        self.seen = dict(system=system, user=user, n=n, temperature=temperature, seed=seed)
        return [f"s{i}" for i in range(n)]


def test_sample_completions_falls_back_to_n_complete_calls():
    client = _CompleteOnlyClient()
    out = sample_completions(client, system="S", user="U", n=3)
    assert out == ["ans-1", "ans-2", "ans-3"]
    assert client.calls == 3  # exactly n calls to complete


def test_sample_completions_uses_complete_samples_when_present():
    client = _SamplerClient()
    out = sample_completions(client, system="S", user="U", n=4, temperature=0, seed=42)
    assert out == ["s0", "s1", "s2", "s3"]
    assert client.seen == dict(system="S", user="U", n=4, temperature=0, seed=42)


def test_sample_completions_n1_default_temperature():
    client = _CompleteOnlyClient()
    out = sample_completions(client, system="S", user="U", n=1)
    assert out == ["ans-1"]
    assert client.calls == 1


# ---------------------------------------------------------------------------
# OpenAILlmClient.complete_samples — native n + seed, and TypeError degrade
# ---------------------------------------------------------------------------


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)


class _Usage:
    # Chat Completions usage shape (prompt_tokens/completion_tokens).
    prompt_tokens = 100
    completion_tokens = 20
    prompt_tokens_details = None
    completion_tokens_details = None


def test_openai_complete_samples_passes_n_seed_temperature_zero(monkeypatch):
    from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

    client = OpenAILlmClient(api_key="test", model="x")
    captured: dict[str, object] = {}

    class _Resp:
        usage = _Usage()
        choices = [_Choice("a"), _Choice("b"), _Choice("c")]

    def _fake_create(**kwargs):
        captured.update(kwargs)
        # Return as many choices as requested n (here we hardcode 3 for n=3).
        return _Resp()

    monkeypatch.setattr(client._client.chat.completions, "create", _fake_create)

    out = client.complete_samples(system="s", user="u", n=3, seed=7)
    assert out == ["a", "b", "c"]
    assert len(out) == 3
    # n/seed/temperature=0 reached the SDK in ONE round-trip
    assert captured["n"] == 3
    assert captured["seed"] == 7
    assert captured["temperature"] == 0
    # usage captured from the single multi-choice response
    assert client._last_usage["input_tokens"] == 100
    assert client._last_usage["output_tokens"] == 20


def test_openai_complete_samples_honours_explicit_temperature(monkeypatch):
    from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

    client = OpenAILlmClient(api_key="test", model="x")
    captured: dict[str, object] = {}

    class _Resp:
        usage = _Usage()
        choices = [_Choice("z")]

    def _fake_create(**kwargs):
        captured.update(kwargs)
        return _Resp()

    monkeypatch.setattr(client._client.chat.completions, "create", _fake_create)
    client.complete_samples(system="s", user="u", n=1, temperature=0.7)
    assert captured["temperature"] == 0.7


def test_openai_complete_samples_degrades_on_typeerror(monkeypatch):
    from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

    client = OpenAILlmClient(api_key="test", model="x")
    seen_kwargs: list[dict] = []

    def _fake_create(**kwargs):
        seen_kwargs.append(kwargs)
        # First call (with n=/seed=) raises TypeError, simulating an SDK/provider
        # that rejects those kwargs; subsequent single-choice calls succeed.
        if "n" in kwargs or "seed" in kwargs:
            raise TypeError("unexpected keyword argument 'n'")

        class _Resp:
            usage = _Usage()
            choices = [_Choice("seq")]

        return _Resp()

    monkeypatch.setattr(client._client.chat.completions, "create", _fake_create)

    out = client.complete_samples(system="s", user="u", n=3, seed=9)
    assert out == ["seq", "seq", "seq"]
    assert len(out) == 3
    # 1 rejected native attempt + 3 sequential fallback calls
    assert len(seen_kwargs) == 4
    # the fallback calls carry no n/seed and pin temperature
    for kw in seen_kwargs[1:]:
        assert "n" not in kw and "seed" not in kw
        assert kw["temperature"] == 0


# ---------------------------------------------------------------------------
# AnthropicMessagesClient — raw Messages API path (mocked SDK)
# ---------------------------------------------------------------------------


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _AntUsage:
    input_tokens = 11
    output_tokens = 5
    cache_read_input_tokens = 3
    cache_creation_input_tokens = 0


class _AntMessage:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]
        self.usage = _AntUsage()


class _FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _AntMessage(f"reply-{len(self.calls)}")


class _FakeAnthropic:
    last_init_kwargs: dict = {}

    def __init__(self, **kwargs) -> None:
        type(self).last_init_kwargs = kwargs
        self.messages = _FakeMessages()


@pytest.fixture()
def _patch_anthropic(monkeypatch):
    """Patch ``anthropic.Anthropic`` so no key/network is needed."""
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    return _FakeAnthropic


def test_anthropic_messages_client_complete(_patch_anthropic):
    from backend.services.context.workspace.tools.anthropic_messages_client import (
        AnthropicMessagesClient,
    )

    client = AnthropicMessagesClient(api_key="sk-test", model="claude-sonnet-4-6")
    out = client.complete(system="grade this", user="evidence")
    assert out == "reply-1"
    # temperature=0 + system as a top-level param + single user message
    call = client._client.messages.calls[0]
    assert call["temperature"] == 0
    assert call["system"] == "grade this"
    assert call["model"] == "claude-sonnet-4-6"
    assert call["messages"] == [{"role": "user", "content": "evidence"}]
    # usage mirrored from the Anthropic field names
    assert client._last_usage["input_tokens"] == 11
    assert client._last_usage["output_tokens"] == 5
    assert client._last_usage["cache_read_input_tokens"] == 3


def test_anthropic_messages_client_complete_samples_sequential(_patch_anthropic):
    from backend.services.context.workspace.tools.anthropic_messages_client import (
        AnthropicMessagesClient,
    )

    client = AnthropicMessagesClient(api_key="sk-test")
    out = client.complete_samples(system="s", user="u", n=3, temperature=0, seed=1)
    assert out == ["reply-1", "reply-2", "reply-3"]
    # n sequential Messages calls, each at temperature=0 (seed ignored — no API support)
    calls = client._client.messages.calls
    assert len(calls) == 3
    for c in calls:
        assert c["temperature"] == 0
        assert "seed" not in c


# ---------------------------------------------------------------------------
# build_grader_client — default passthrough + anthropic backend
# ---------------------------------------------------------------------------


def test_build_grader_client_passthrough_when_env_unset(monkeypatch):
    monkeypatch.delenv("REPROLAB_GRADER_BACKEND", raising=False)
    monkeypatch.delenv("REPROLAB_GRADER_MODEL", raising=False)

    sentinel = object()
    client, label = build_grader_client(sentinel, "root-label")
    assert client is sentinel  # unchanged — today's behaviour
    assert label == "root-label"


def test_build_grader_client_model_only_is_passthrough(monkeypatch):
    # A model override with no backend can't pick a transport → passthrough.
    monkeypatch.delenv("REPROLAB_GRADER_BACKEND", raising=False)
    monkeypatch.setenv("REPROLAB_GRADER_MODEL", "claude-sonnet-4-6")

    sentinel = object()
    client, label = build_grader_client(sentinel, "root-label")
    assert client is sentinel
    assert label == "root-label"


def test_build_grader_client_anthropic_backend(monkeypatch, _patch_anthropic):
    from backend.services.context.workspace.tools.anthropic_messages_client import (
        AnthropicMessagesClient,
    )

    monkeypatch.setenv("REPROLAB_GRADER_BACKEND", "anthropic")
    monkeypatch.delenv("REPROLAB_GRADER_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    sentinel = object()
    client, label = build_grader_client(sentinel, "root-label")
    assert client is not sentinel
    assert isinstance(client, AnthropicMessagesClient)
    assert label == "grader:anthropic:claude-sonnet-4-6"


def test_build_grader_client_anthropic_respects_model_override(monkeypatch, _patch_anthropic):
    monkeypatch.setenv("REPROLAB_GRADER_BACKEND", "anthropic")
    monkeypatch.setenv("REPROLAB_GRADER_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    client, label = build_grader_client(object(), "root-label")
    assert label == "grader:anthropic:claude-opus-4-8"
    assert client._model == "claude-opus-4-8"


def test_build_grader_client_unknown_backend_falls_back(monkeypatch):
    monkeypatch.setenv("REPROLAB_GRADER_BACKEND", "bananas")
    monkeypatch.delenv("REPROLAB_GRADER_MODEL", raising=False)

    sentinel = object()
    client, label = build_grader_client(sentinel, "root-label")
    assert client is sentinel  # unknown backend → safe passthrough, no raise
    assert label == "root-label"


def test_build_grader_client_construction_error_falls_back(monkeypatch):
    # Force AnthropicMessagesClient construction to blow up; transport must
    # fall back to the root client rather than raise.
    import backend.services.context.workspace.tools.anthropic_messages_client as amc

    def _boom(*a, **k):
        raise RuntimeError("no SDK / no key")

    monkeypatch.setattr(amc, "AnthropicMessagesClient", _boom)
    monkeypatch.setenv("REPROLAB_GRADER_BACKEND", "anthropic")
    monkeypatch.delenv("REPROLAB_GRADER_MODEL", raising=False)

    sentinel = object()
    client, label = build_grader_client(sentinel, "root-label")
    assert client is sentinel
    assert label == "root-label"
