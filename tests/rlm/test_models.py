"""Contract tests for backend.agents.rlm.models.

Tests cover:
  - Direct name lookup in ROOT_MODELS
  - The layered default (gpt-5 when OPENAI_API_KEY set; qwen3-coder otherwise)
  - Unknown name raises ValueError listing valid keys
  - All four registry entries carry valid rlm ClientBackend literals
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

    def test_qwen3_coder_resolves(self):
        from backend.agents.rlm.models import resolve_root_model

        m = resolve_root_model("qwen3-coder")
        assert m.key == "qwen3-coder"
        assert m.rlm_backend == "openrouter"

    def test_kimi_resolves(self):
        from backend.agents.rlm.models import resolve_root_model

        m = resolve_root_model("kimi-k2.5")
        assert m.key == "kimi-k2.5"

    def test_claude_resolves(self):
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
        mod = _reload_models()
        result = mod.resolve_root_model(None)
        assert result.key == "qwen3-coder"

    def test_env_var_overrides_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("REPROLAB_RLM_ROOT_MODEL", "claude")
        mod = _reload_models()
        result = mod.resolve_root_model(None)
        assert result.key == "claude"

    def test_explicit_name_beats_env_var(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_RLM_ROOT_MODEL", "claude")
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
    """All four ROOT_MODELS entries satisfy the §6 spec contract."""

    def test_all_four_entries_present(self):
        from backend.agents.rlm.models import ROOT_MODELS

        assert set(ROOT_MODELS) == {"gpt-5", "qwen3-coder", "kimi-k2.5", "claude"}

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
