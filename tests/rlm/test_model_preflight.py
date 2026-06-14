"""Model pin + availability preflight — the 2026-06-14 Fable-5 wedge fix.

The bundled claude CLI's mutable default model resolved to the then-unavailable
Fable 5; every model=None SDK call returned a 'model unavailable' block instead
of JSON, and both live runs wedged ~14 min then shipped 0. These tests pin the
contract that prevents recurrence: an EXPLICIT model on every OAuth call, plus a
fail-fast preflight that aborts (with a report) instead of wedging.
"""

from __future__ import annotations

from backend.services.context.workspace.tools.rlm_query import (
    ClaudeLlmClient,
    default_oauth_model,
    is_model_unavailable_response,
    preflight_model_available,
)

_BLOCK = (
    "There's an issue with the selected model (claude-fable-5[1m]). "
    "It may not exist or you may not have access."
)


class _StubClient:
    """Minimal LlmClient stub returning a scripted response sequence."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, *, system, user):
        self.calls += 1
        r = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        if isinstance(r, Exception):
            raise r
        return r


class TestDefaultOauthModel:
    def test_defaults_to_sonnet(self, monkeypatch):
        monkeypatch.delenv("REPROLAB_OAUTH_FALLBACK_MODEL", raising=False)
        assert default_oauth_model() == "claude-sonnet-4-6"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_OAUTH_FALLBACK_MODEL", "claude-opus-4-8")
        assert default_oauth_model() == "claude-opus-4-8"

    def test_client_none_model_becomes_explicit(self, monkeypatch):
        """model=None must NEVER reach the SDK — it would defer to the CLI default."""
        monkeypatch.delenv("REPROLAB_OAUTH_FALLBACK_MODEL", raising=False)
        assert ClaudeLlmClient(model=None)._model == "claude-sonnet-4-6"

    def test_client_explicit_model_preserved(self):
        assert ClaudeLlmClient(model="claude-opus-4-7")._model == "claude-opus-4-7"


class TestIsModelUnavailable:
    def test_block_detected(self):
        assert is_model_unavailable_response(_BLOCK) is True

    def test_normal_text_not_flagged(self):
        # A normal planning response that mentions "model" is NOT a false positive.
        assert is_model_unavailable_response("The model trains for 10 epochs.") is False

    def test_empty_not_flagged(self):
        assert is_model_unavailable_response("") is False
        assert is_model_unavailable_response(None) is False


class TestPreflight:
    def test_available_when_ok(self):
        ok, _ = preflight_model_available(_StubClient(["OK"]))
        assert ok is True

    def test_unavailable_when_block_every_attempt(self):
        c = _StubClient([_BLOCK, _BLOCK])
        ok, detail = preflight_model_available(c, attempts=2)
        assert ok is False
        assert "selected model" in detail.lower()
        assert c.calls == 2

    def test_failsoft_on_transport_error(self):
        # An ambiguous transport blip must NOT abort a good run.
        ok, _ = preflight_model_available(_StubClient([RuntimeError("blip")]))
        assert ok is True

    def test_recovers_if_a_later_attempt_succeeds(self):
        # Transient block then real content → available (no false abort).
        ok, _ = preflight_model_available(_StubClient([_BLOCK, "OK"]), attempts=2)
        assert ok is True
