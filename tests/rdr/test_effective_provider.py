"""Tests for _effective_provider in backend/agents/rdr/run.py.

Guards the fix for rlm_hybrid_smoke_1779521608: a revoked/invalid
OPENAI_API_KEY in .env was silently misrouting every cluster agent to OpenAI
(27/27 clusters failed with HTTP 401). Same bug class as commit 005e3b6 which
fixed _build_llm_client in rlm/run.py.
"""

from __future__ import annotations


import pytest

from backend.agents.rdr.run import _effective_provider


def test_explicit_provider_wins() -> None:
    """Explicit provider arg overrides model implication."""
    result = _effective_provider("anthropic", model="gpt-5")
    assert result == "anthropic"


def test_claude_model_implies_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """claude-* / sonnet / opus / haiku model names → anthropic regardless of env."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    for model in [
        "claude-oauth",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-haiku-4-5-20251001",
    ]:
        result = _effective_provider(None, model=model)
        assert result == "anthropic", (
            f"Expected 'anthropic' for model={model!r}, got {result!r}"
        )


def test_gpt_model_implies_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """gpt-* model names → openai regardless of env."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    for model in ["gpt-5", "gpt-4o-mini"]:
        result = _effective_provider(None, model=model)
        assert result == "openai", (
            f"Expected 'openai' for model={model!r}, got {result!r}"
        )


def test_anthropic_creds_preferred_over_invalid_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for rlm_hybrid_smoke_1779521608.

    When OPENAI_API_KEY is set to a junk/revoked value AND valid Anthropic
    credentials exist, _effective_provider must return 'anthropic' — not
    'openai'. The old code returned 'openai' unconditionally when any
    OPENAI_API_KEY was present in env.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-svcacct-REVOKED-junk-value")

    # _effective_provider does a lazy `from backend.agents.runtime.factory import
    # has_provider_credentials` — patch it on the factory module so the import
    # inside the function picks up the fake.
    from backend.agents.runtime import factory as _fac
    original_fn = _fac.has_provider_credentials

    def _fake_hpc(provider=None):
        return provider == "anthropic"

    _fac.has_provider_credentials = _fake_hpc
    try:
        result = _effective_provider(None, model=None)
    finally:
        _fac.has_provider_credentials = original_fn

    assert result == "anthropic", (
        f"Expected 'anthropic' when valid Anthropic creds exist despite junk OPENAI_API_KEY; "
        f"got {result!r}. This is the regression guard for rlm_hybrid_smoke_1779521608."
    )


def test_falls_through_to_openai_when_no_anthropic_creds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When has_provider_credentials('anthropic') is False and OPENAI_API_KEY is set,
    fall through to openai."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-valid-looking-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY_PATH", raising=False)

    from backend.agents.runtime import factory as _fac
    original_fn = _fac.has_provider_credentials

    try:
        def _fake_hpc(provider=None):
            return False  # no Anthropic creds

        _fac.has_provider_credentials = _fake_hpc
        result = _effective_provider(None, model=None)
    finally:
        _fac.has_provider_credentials = original_fn

    assert result == "openai", (
        f"Expected 'openai' when no Anthropic creds and OPENAI_API_KEY is set; got {result!r}"
    )
