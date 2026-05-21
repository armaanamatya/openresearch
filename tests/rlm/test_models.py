"""Contract tests for backend.agents.rlm.models.

Tests cover:
  - Direct name lookup in ROOT_MODELS
  - The layered default (gpt-5 when OPENAI_API_KEY set; qwen3-coder otherwise)
  - Unknown name raises ValueError listing valid keys
  - All registry entries carry valid rlm ClientBackend literals
  - qwen3-coder has a non-empty prompt_addendum
  - paper_validated flags are correct per the spec
"""

from __future__ import annotations

import importlib
import sys

import pytest

# Valid ClientBackend literals from rlm/core/types.py.
_VALID_RLM_BACKENDS = frozenset(
    {
        "openai",
        "portkey",
        "openrouter",
        "vercel",
        "vllm",
        "litellm",
        "anthropic",
        "azure_openai",
        "gemini",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_models(monkeypatch_env: dict | None = None):
    """Re-import models.py so _build_registry() re-reads env vars.

    This is needed because ROOT_MODELS is built at module-load time.
    We drop the cached module from sys.modules and re-import.
    """
    # Apply any env-var patches before re-import.
    # (The caller uses monkeypatch — this helper just does the reload.)
    for mod in list(sys.modules):
        if "backend.agents.rlm.models" in mod:
            del sys.modules[mod]
    import backend.agents.rlm.models as m
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolveByName:
    """resolve_root_model(name) returns the correct RootModel."""

    def test_gpt5_resolves(self):
        from backend.agents.rlm.models import resolve_root_model

        m = resolve_root_model("gpt-5")
        assert m.key == "gpt-5"
        assert m.rlm_backend == "openai"

    def test_qwen3_coder_resolves(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")  # A1-H1: openrouter model needs it
        from backend.agents.rlm.models import resolve_root_model

        m = resolve_root_model("qwen3-coder")
        assert m.key == "qwen3-coder"
        assert m.rlm_backend == "openrouter"

    def test_openrouter_model_without_key_raises(self, monkeypatch):
        """A1-H1: an OpenRouter-backed model fails fast when OPENROUTER_API_KEY is absent."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from backend.agents.rlm.models import resolve_root_model

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            resolve_root_model("qwen3-coder")

    def test_kimi_resolves(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")  # A1-H1: openrouter model needs it
        from backend.agents.rlm.models import resolve_root_model

        m = resolve_root_model("kimi-k2.5")
        assert m.key == "kimi-k2.5"

    def test_claude_resolves(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        from backend.agents.rlm.models import resolve_root_model

        m = resolve_root_model("claude")
        assert m.key == "claude"
        assert m.rlm_backend == "anthropic"


class TestLayeredDefault:
    """None resolves via env var then via the OPENAI_API_KEY-based fallback."""

    def test_default_with_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("REPROLAB_RLM_ROOT_MODEL", raising=False)
        # Re-import so ROOT_MODELS is fresh; resolve_root_model reads env at call time.
        mod = _reload_models()
        result = mod.resolve_root_model(None)
        assert result.key == "gpt-5"

    def test_default_without_openai_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("REPROLAB_RLM_ROOT_MODEL", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")  # qwen3-coder default needs it (A1-H1)
        mod = _reload_models()
        result = mod.resolve_root_model(None)
        assert result.key == "qwen3-coder"

    def test_env_var_overrides_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("REPROLAB_RLM_ROOT_MODEL", "claude")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        mod = _reload_models()
        result = mod.resolve_root_model(None)
        assert result.key == "claude"

    def test_explicit_name_beats_env_var(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_RLM_ROOT_MODEL", "claude")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from backend.agents.rlm.models import resolve_root_model

        result = resolve_root_model("gpt-5")
        assert result.key == "gpt-5"


class TestUnknownName:
    """An unrecognised name raises ValueError with the valid keys listed."""

    def test_raises_value_error(self):
        from backend.agents.rlm.models import resolve_root_model

        with pytest.raises(ValueError, match="Unknown root model"):
            resolve_root_model("gpt-99-super")

    def test_error_lists_valid_keys(self):
        from backend.agents.rlm.models import ROOT_MODELS, resolve_root_model

        with pytest.raises(ValueError) as exc_info:
            resolve_root_model("not-a-model")

        msg = str(exc_info.value)
        for key in ROOT_MODELS:
            assert key in msg, f"Expected {key!r} listed in error; got: {msg}"


class TestRegistryContract:
    """All ROOT_MODELS entries satisfy the §6 spec contract."""

    def test_all_registry_entries_present(self):
        from backend.agents.rlm.models import ROOT_MODELS

        assert set(ROOT_MODELS) == {"gpt-5", "qwen3-coder", "kimi-k2.5", "claude", "qwen3-coder-featherless"}

    def test_all_backends_are_valid_rlm_literals(self):
        from backend.agents.rlm.models import ROOT_MODELS

        for key, model in ROOT_MODELS.items():
            assert model.rlm_backend in _VALID_RLM_BACKENDS, (
                f"{key}: rlm_backend={model.rlm_backend!r} not a valid ClientBackend"
            )
            assert model.sub_backend in _VALID_RLM_BACKENDS, (
                f"{key}: sub_backend={model.sub_backend!r} not a valid ClientBackend"
            )

    def test_paper_validated_flags(self):
        from backend.agents.rlm.models import ROOT_MODELS

        assert ROOT_MODELS["gpt-5"].paper_validated is True
        assert ROOT_MODELS["qwen3-coder"].paper_validated is True
        assert ROOT_MODELS["kimi-k2.5"].paper_validated is False
        assert ROOT_MODELS["claude"].paper_validated is False

    def test_qwen_has_nonempty_prompt_addendum(self):
        from backend.agents.rlm.models import ROOT_MODELS

        addendum = ROOT_MODELS["qwen3-coder"].prompt_addendum
        assert addendum, "qwen3-coder must have a non-empty prompt_addendum"
        assert "llm_query" in addendum, (
            "qwen addendum should reference llm_query (anti-over-subcalling line)"
        )

    def test_other_models_have_empty_addendum(self):
        from backend.agents.rlm.models import ROOT_MODELS

        for key in ("gpt-5", "kimi-k2.5", "claude"):
            assert ROOT_MODELS[key].prompt_addendum == "", (
                f"{key} should have an empty prompt_addendum"
            )

    def test_backend_kwargs_have_model_name(self):
        from backend.agents.rlm.models import ROOT_MODELS

        for key, model in ROOT_MODELS.items():
            assert "model_name" in model.backend_kwargs, (
                f"{key}: backend_kwargs must contain 'model_name'"
            )
            assert "model_name" in model.sub_backend_kwargs, (
                f"{key}: sub_backend_kwargs must contain 'model_name'"
            )


class TestEnvVarSlugOverride:
    """OpenRouter slugs can be overridden via env vars (config-driven, not hardcoded)."""

    def test_qwen_slug_env_var(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_RLM_ROOT_SLUG_QWEN", "qwen/qwen3-custom-test")
        mod = _reload_models()
        assert mod.ROOT_MODELS["qwen3-coder"].backend_kwargs["model_name"] == (
            "qwen/qwen3-custom-test"
        )

    def test_kimi_slug_env_var(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_RLM_ROOT_SLUG_KIMI", "moonshotai/kimi-custom")
        mod = _reload_models()
        assert mod.ROOT_MODELS["kimi-k2.5"].backend_kwargs["model_name"] == (
            "moonshotai/kimi-custom"
        )


class TestMissingApiKeyFailsFast:
    """resolve_root_model raises ValueError at resolve time when a required API key is absent."""

    def test_anthropic_key_absent_raises(self, monkeypatch):
        """Missing ANTHROPIC_API_KEY raises ValueError, not a deep TypeError."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from backend.agents.rlm.models import resolve_root_model

        with pytest.raises(ValueError) as exc_info:
            resolve_root_model("claude")
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_openrouter_key_absent_raises_regression(self, monkeypatch):
        """Regression: qwen3-coder (openrouter) still raises ValueError with OPENROUTER_API_KEY in msg."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from backend.agents.rlm.models import resolve_root_model

        with pytest.raises(ValueError) as exc_info:
            resolve_root_model("qwen3-coder")
        assert "OPENROUTER_API_KEY" in str(exc_info.value)

    def test_happy_path_key_set_injects_into_kwargs(self, monkeypatch):
        """With the key set, resolve_root_model returns a RootModel with api_key in both *_kwargs."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-happy")
        from backend.agents.rlm.models import resolve_root_model

        result = resolve_root_model("claude")
        assert result.backend_kwargs.get("api_key") == "sk-ant-happy"
        assert result.sub_backend_kwargs.get("api_key") == "sk-ant-happy"


class TestFeatherlessEntry:
    """Contract tests for the qwen3-coder-featherless RootModel entry."""

    def test_resolves_with_key_set(self, monkeypatch):
        """With FEATHERLESS_API_KEY set, resolve returns correct fields."""
        monkeypatch.setenv("FEATHERLESS_API_KEY", "fl-test-key-abc")
        from backend.agents.rlm.models import (
            FEATHERLESS_BASE_URL,
            FEATHERLESS_ROOT_MODEL,
            resolve_root_model,
        )

        m = resolve_root_model("qwen3-coder-featherless")
        assert m.api_key_env == "FEATHERLESS_API_KEY"
        assert m.backend_kwargs["base_url"] == FEATHERLESS_BASE_URL
        assert m.backend_kwargs["model_name"] == FEATHERLESS_ROOT_MODEL
        assert m.backend_kwargs["api_key"] == "fl-test-key-abc"
        assert m.paper_validated is True

    def test_raises_when_key_absent(self, monkeypatch):
        """Without FEATHERLESS_API_KEY, resolve raises ValueError naming the env var."""
        monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
        from backend.agents.rlm.models import resolve_root_model

        with pytest.raises(ValueError) as exc_info:
            resolve_root_model("qwen3-coder-featherless")
        assert "FEATHERLESS_API_KEY" in str(exc_info.value)

    def test_sub_backend_kwargs_carry_base_url(self, monkeypatch):
        """sub_backend_kwargs also contain base_url and the 30B sub model."""
        monkeypatch.setenv("FEATHERLESS_API_KEY", "fl-test-key-abc")
        from backend.agents.rlm.models import (
            FEATHERLESS_BASE_URL,
            FEATHERLESS_SUBCALL_MODEL,
            resolve_root_model,
        )

        m = resolve_root_model("qwen3-coder-featherless")
        assert m.sub_backend_kwargs["base_url"] == FEATHERLESS_BASE_URL
        assert m.sub_backend_kwargs["model_name"] == FEATHERLESS_SUBCALL_MODEL
        assert m.sub_backend_kwargs.get("api_key") == "fl-test-key-abc"


class TestEnvVarFor:
    """Unit tests for _env_var_for helper."""

    def test_explicit_api_key_env_wins_over_backend_default(self):
        """When api_key_env is set explicitly, it takes priority."""
        from backend.agents.rlm.models import _env_var_for

        result = _env_var_for("openai", "FEATHERLESS_API_KEY")
        assert result == "FEATHERLESS_API_KEY"

    def test_none_api_key_env_falls_back_to_backend_default(self):
        """When api_key_env is None, the backend-type default is returned."""
        from backend.agents.rlm.models import _env_var_for

        result = _env_var_for("openai", None)
        assert result == "OPENAI_API_KEY"

    def test_anthropic_backend_default(self):
        from backend.agents.rlm.models import _env_var_for

        assert _env_var_for("anthropic", None) == "ANTHROPIC_API_KEY"

    def test_unknown_backend_returns_none_when_no_explicit(self):
        from backend.agents.rlm.models import _env_var_for

        assert _env_var_for("vllm", None) is None
