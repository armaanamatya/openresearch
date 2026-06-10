"""Tests for the rlm.clients monkeypatch that registers anthropic-oauth."""

from __future__ import annotations



def _reset_patch():
    """Reset the idempotency flag so tests can re-apply the patch cleanly."""
    import backend.agents.rlm._oauth_backend_patch as _mod
    _mod._APPLIED = False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestApplyIdempotent:
    """apply_oauth_backend_patch() is idempotent — second call is a no-op."""

    def test_apply_idempotent(self):
        import rlm.clients
        from backend.agents.rlm._oauth_backend_patch import apply_oauth_backend_patch

        _reset_patch()
        apply_oauth_backend_patch()
        patched_fn = rlm.clients.get_client

        # Second call should not replace the function again.
        apply_oauth_backend_patch()
        assert rlm.clients.get_client is patched_fn


class TestPatchedGetClientDispatchesToOauthClient:
    """After patching, get_client('anthropic-oauth', ...) returns a ClaudeOauthClient."""

    def test_patched_get_client_dispatches_to_oauth_client(self):
        import rlm.clients
        from backend.agents.rlm._oauth_backend_patch import apply_oauth_backend_patch
        from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient

        _reset_patch()
        apply_oauth_backend_patch()

        client = rlm.clients.get_client("anthropic-oauth", {"model_name": "X"})
        assert isinstance(client, ClaudeOauthClient)
        assert client.model_name == "X"


class TestPatchedGetClientFallsThroughForKnownBackends:
    """Patching does not break standard backends like 'openai'."""

    def test_patched_get_client_falls_through_for_known_backends(self):
        import rlm.clients
        from backend.agents.rlm._oauth_backend_patch import apply_oauth_backend_patch

        _reset_patch()
        apply_oauth_backend_patch()

        # openai backend should still dispatch correctly.
        client = rlm.clients.get_client("openai", {"api_key": "sk-fake", "model_name": "gpt-5"})
        # The rlm OpenAIClient is returned — check it's not a ClaudeOauthClient.
        from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient
        assert not isinstance(client, ClaudeOauthClient)
        # It should have a model_name attribute matching what we passed.
        assert client.model_name == "gpt-5"
