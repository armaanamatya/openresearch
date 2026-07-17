"""Tests for the claude-oauth RootModel entry and resolve_root_model OAuth logic."""

from __future__ import annotations

import sys

import pytest


def _reload_models():
    """Re-import models.py so _build_registry() re-reads env vars."""
    for mod in list(sys.modules):
        if "backend.agents.rlm.models" in mod:
            del sys.modules[mod]
    import backend.agents.rlm.models as m
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClaudeOauthInRegistry:
    """claude-oauth exists in ROOT_MODELS with the correct fields."""

    def test_claude_oauth_in_registry(self):
        from backend.agents.rlm.models import ROOT_MODELS

        assert "claude-oauth" in ROOT_MODELS
        entry = ROOT_MODELS["claude-oauth"]
        assert entry.rlm_backend == "anthropic-oauth"
        assert entry.api_key_env is None


class TestResolveClaudeOauthWithCredentials:
    """resolve_root_model('claude-oauth') succeeds when has_provider_credentials is True."""

    def test_resolve_claude_oauth_with_credentials(self, monkeypatch):
        monkeypatch.setattr(
            "backend.agents.runtime.factory.has_provider_credentials",
            lambda provider=None: True,
        )
        from backend.agents.rlm.models import resolve_root_model

        entry = resolve_root_model("claude-oauth")
        assert entry.key == "claude-oauth"
        assert entry.rlm_backend == "anthropic-oauth"


class TestResolveClaudeOauthWithoutCredentialsFailsLoud:
    """resolve_root_model('claude-oauth') raises ValueError when no creds are found."""

    def test_resolve_claude_oauth_without_credentials_fails_loud(self, monkeypatch):
        monkeypatch.setattr(
            "backend.agents.runtime.factory.has_provider_credentials",
            lambda provider=None: False,
        )
        from backend.agents.rlm.models import resolve_root_model

        with pytest.raises(ValueError) as exc_info:
            resolve_root_model("claude-oauth")

        msg = str(exc_info.value)
        assert "Claude credentials" in msg
        assert "claude login" in msg


class TestDefaultFallsBackToClaudeOauth:
    """When no API keys are set but OAuth creds are present, default picks claude-oauth."""

    def test_default_falls_back_to_claude_oauth_when_no_keys_but_oauth_present(
        self, monkeypatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENRESEARCH_RLM_ROOT_MODEL", raising=False)

        monkeypatch.setattr(
            "backend.agents.runtime.factory.has_provider_credentials",
            lambda provider=None: True,
        )

        mod = _reload_models()
        result = mod.resolve_root_model(None)
        assert result.key == "claude-oauth"


class TestDefaultFallsThroughToQwen3CoderWhenNoCredentials:
    """When all keys are unset and OAuth is False, default picks qwen3-coder which then fails."""

    def test_default_falls_through_to_qwen3coder_when_no_credentials_anywhere(
        self, monkeypatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENRESEARCH_RLM_ROOT_MODEL", raising=False)

        monkeypatch.setattr(
            "backend.agents.runtime.factory.has_provider_credentials",
            lambda provider=None: False,
        )

        mod = _reload_models()
        # qwen3-coder requires OPENROUTER_API_KEY which is unset — expect ValueError.
        with pytest.raises(ValueError):
            mod.resolve_root_model(None)


def _stub_settings_pin(monkeypatch, value: str) -> None:
    """Point models.py's Settings fallback at a stub with the given pin."""
    from types import SimpleNamespace

    import backend.config as _cfg

    monkeypatch.setattr(
        _cfg, "get_settings",
        lambda _force_reload=False: SimpleNamespace(rlm_root_model_name=value),
    )


class TestOAuthRootModelPin:
    """OPENRESEARCH_RLM_ROOT_MODEL_NAME (legacy REPROLAB_ accepted) pins the OAuth root's model id."""

    def test_pin_overrides_oauth_root_model(self, monkeypatch):
        from backend.agents.runtime import factory as _factory
        monkeypatch.setattr(_factory, "has_provider_credentials", lambda p: True)
        monkeypatch.delenv("OPENRESEARCH_RLM_ROOT_MODEL_NAME", raising=False)
        monkeypatch.setenv("REPROLAB_RLM_ROOT_MODEL_NAME", "claude-opus-4-8")
        from backend.agents.rlm.models import resolve_root_model
        entry = resolve_root_model("claude-oauth")
        assert entry.backend_kwargs["model_name"] == "claude-opus-4-8"
        # sub-backend (navigation) stays untouched
        assert entry.sub_backend_kwargs["model_name"].startswith("claude-haiku")

    def test_canonical_openresearch_prefix_pins(self, monkeypatch):
        from backend.agents.runtime import factory as _factory
        monkeypatch.setattr(_factory, "has_provider_credentials", lambda p: True)
        monkeypatch.delenv("REPROLAB_RLM_ROOT_MODEL_NAME", raising=False)
        monkeypatch.setenv("OPENRESEARCH_RLM_ROOT_MODEL_NAME", "claude-opus-4-8")
        from backend.agents.rlm.models import resolve_root_model
        entry = resolve_root_model("claude-oauth")
        assert entry.backend_kwargs["model_name"] == "claude-opus-4-8"

    def test_canonical_prefix_wins_over_legacy(self, monkeypatch):
        from backend.agents.runtime import factory as _factory
        monkeypatch.setattr(_factory, "has_provider_credentials", lambda p: True)
        monkeypatch.setenv("OPENRESEARCH_RLM_ROOT_MODEL_NAME", "claude-opus-4-8")
        monkeypatch.setenv("REPROLAB_RLM_ROOT_MODEL_NAME", "claude-sonnet-4-6")
        from backend.agents.rlm.models import resolve_root_model
        entry = resolve_root_model("claude-oauth")
        assert entry.backend_kwargs["model_name"] == "claude-opus-4-8"

    def test_settings_env_file_fallback(self, monkeypatch):
        """A .env-only pin works: the CLI never exports .env into os.environ,
        so models.py falls back to Settings when both process-env names are unset."""
        from backend.agents.runtime import factory as _factory
        monkeypatch.setattr(_factory, "has_provider_credentials", lambda p: True)
        monkeypatch.delenv("OPENRESEARCH_RLM_ROOT_MODEL_NAME", raising=False)
        monkeypatch.delenv("REPROLAB_RLM_ROOT_MODEL_NAME", raising=False)
        _stub_settings_pin(monkeypatch, "claude-opus-4-8")
        from backend.agents.rlm.models import resolve_root_model
        entry = resolve_root_model("claude-oauth")
        assert entry.backend_kwargs["model_name"] == "claude-opus-4-8"

    def test_no_pin_keeps_registry_default(self, monkeypatch):
        from backend.agents.runtime import factory as _factory
        monkeypatch.setattr(_factory, "has_provider_credentials", lambda p: True)
        monkeypatch.delenv("OPENRESEARCH_RLM_ROOT_MODEL_NAME", raising=False)
        monkeypatch.delenv("REPROLAB_RLM_ROOT_MODEL_NAME", raising=False)
        _stub_settings_pin(monkeypatch, "")  # isolate from the developer's real .env
        from backend.agents.rlm.models import resolve_root_model
        entry = resolve_root_model("claude-oauth")
        assert entry.backend_kwargs["model_name"] == "claude-sonnet-4-6"

    def test_pin_does_not_mutate_registry(self, monkeypatch):
        from backend.agents.runtime import factory as _factory
        monkeypatch.setattr(_factory, "has_provider_credentials", lambda p: True)
        monkeypatch.setenv("OPENRESEARCH_RLM_ROOT_MODEL_NAME", "claude-opus-4-8")
        from backend.agents.rlm.models import ROOT_MODELS, resolve_root_model
        resolve_root_model("claude-oauth")
        assert ROOT_MODELS["claude-oauth"].backend_kwargs["model_name"] == "claude-sonnet-4-6"


class TestOAuthRootModelPinCanonicalEnv:
    """OPENRESEARCH_RLM_ROOT_MODEL_NAME (canonical prefix) + Settings fallback.

    The pin was born under the legacy REPROLAB_ prefix and read straight from
    os.environ, so (a) the canonical OPENRESEARCH_ name silently did nothing
    when set after config import, and (b) a `.env`-file-only value never worked
    at all (pydantic-settings does not mutate os.environ).
    """

    def test_pin_via_canonical_env(self, monkeypatch):
        from backend.agents.runtime import factory as _factory
        monkeypatch.setattr(_factory, "has_provider_credentials", lambda p: True)
        monkeypatch.delenv("REPROLAB_RLM_ROOT_MODEL_NAME", raising=False)
        monkeypatch.setenv("OPENRESEARCH_RLM_ROOT_MODEL_NAME", "claude-opus-4-8")
        from backend.agents.rlm.models import resolve_root_model
        entry = resolve_root_model("claude-oauth")
        assert entry.backend_kwargs["model_name"] == "claude-opus-4-8"

    def test_pin_via_settings_dotenv_fallback(self, monkeypatch):
        from backend.agents.runtime import factory as _factory
        monkeypatch.setattr(_factory, "has_provider_credentials", lambda p: True)
        monkeypatch.delenv("REPROLAB_RLM_ROOT_MODEL_NAME", raising=False)
        monkeypatch.delenv("OPENRESEARCH_RLM_ROOT_MODEL_NAME", raising=False)

        import backend.config as _config

        class _StubSettings:
            rlm_root_model_name = "claude-opus-4-8"
            rlm_root_model = ""

        monkeypatch.setattr(_config, "get_settings", lambda: _StubSettings())
        from backend.agents.rlm.models import resolve_root_model
        entry = resolve_root_model("claude-oauth")
        assert entry.backend_kwargs["model_name"] == "claude-opus-4-8"

    def test_root_model_key_via_settings_dotenv_fallback(self, monkeypatch):
        """A .env-only OPENRESEARCH_RLM_ROOT_MODEL selects the root registry key."""
        from backend.agents.runtime import factory as _factory
        monkeypatch.setattr(_factory, "has_provider_credentials", lambda p: True)
        monkeypatch.delenv("OPENRESEARCH_RLM_ROOT_MODEL", raising=False)
        monkeypatch.delenv("REPROLAB_RLM_ROOT_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)

        import backend.config as _config

        class _StubSettings:
            rlm_root_model = "claude-oauth"
            rlm_root_model_name = ""

        monkeypatch.setattr(_config, "get_settings", lambda: _StubSettings())
        from backend.agents.rlm.models import resolve_root_model
        entry = resolve_root_model(None)
        assert entry.key == "claude-oauth"

    def test_settings_fields_exist(self):
        from backend.config import Settings
        s = Settings(
            rlm_root_model="claude-oauth",
            rlm_root_model_name="claude-opus-4-8",
            _env_file=None,
        )
        assert s.rlm_root_model == "claude-oauth"
        assert s.rlm_root_model_name == "claude-opus-4-8"


class TestPlannerStampHonesty:
    """A pinned OAuth root must stamp final_report.models.planner with the model
    that ACTUALLY drove the loop (key:model), mirroring the RoleSpec stamp
    vocabulary; unpinned runs stay byte-identical ("claude-oauth")."""

    def _resolve(self, monkeypatch, pin: str | None):
        from backend.agents.runtime import factory as _factory
        monkeypatch.setattr(_factory, "has_provider_credentials", lambda p: True)
        for var in ("OPENRESEARCH_RLM_ROOT_MODEL_NAME", "REPROLAB_RLM_ROOT_MODEL_NAME"):
            monkeypatch.delenv(var, raising=False)
        # Isolate from the developer's real .env: the rlm library load_dotenv()s
        # it into os.environ at import, and models.py's Settings fallback reads
        # it directly — either path would leak a real pin into this test.
        _stub_settings_pin(monkeypatch, pin or "")
        if pin:
            monkeypatch.setenv("OPENRESEARCH_RLM_ROOT_MODEL_NAME", pin)
        from backend.agents.rlm.models import resolve_root_model
        return resolve_root_model("claude-oauth")

    def test_pinned_root_stamps_key_and_model(self, monkeypatch):
        from backend.agents.rlm.run import _planner_stamp_for
        entry = self._resolve(monkeypatch, "claude-opus-4-8")
        assert _planner_stamp_for(entry) == "claude-oauth:claude-opus-4-8"

    def test_unpinned_root_stamps_none(self, monkeypatch):
        from backend.agents.rlm.run import _planner_stamp_for
        entry = self._resolve(monkeypatch, None)
        assert _planner_stamp_for(entry) is None

    def test_non_oauth_root_stamps_none(self, monkeypatch):
        from backend.agents.rlm.run import _planner_stamp_for
        from backend.agents.rlm.models import ROOT_MODELS
        assert _planner_stamp_for(ROOT_MODELS["gpt-5"]) is None
