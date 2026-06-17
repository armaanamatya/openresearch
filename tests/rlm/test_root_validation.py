"""Unit tests for the pure root-validation classifier.

Covers the public contract of ``backend.agents.rlm.root_validation``:
classify a resolved RootModel into validated/risk for the root-validation
gate (oauth-root-reliability plan, P2). The module is PURE — duck-typed on
``.paper_validated`` / ``.rlm_backend`` / ``.key``; no I/O, no env.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.agents.rlm.root_validation import (
    RISK_DEGENERATE_LOOP,
    RISK_NONE,
    RISK_UNVALIDATED,
    RootValidation,
    classify_root_model,
    is_degenerate_loop_risk,
)


@dataclass(frozen=True)
class _FakeRoot:
    """Minimal duck-typed stand-in for ``RootModel``."""

    key: str
    rlm_backend: str
    paper_validated: bool


def test_gpt5_validated_risk_none() -> None:
    rv = classify_root_model(_FakeRoot("gpt-5", "openai", True))
    assert isinstance(rv, RootValidation)
    assert rv.validated is True
    assert rv.risk == RISK_NONE
    assert rv.model_key == "gpt-5"


def test_claude_oauth_unvalidated_degenerate_loop() -> None:
    rv = classify_root_model(_FakeRoot("claude-oauth", "anthropic-oauth", False))
    assert rv.validated is False
    assert rv.risk == RISK_DEGENERATE_LOOP
    assert rv.model_key == "claude-oauth"


def test_claude_api_unvalidated_plain() -> None:
    # anthropic API key path (NOT oauth) — unvalidated but not the
    # documented degenerate-loop risk.
    rv = classify_root_model(_FakeRoot("claude", "anthropic", False))
    assert rv.validated is False
    assert rv.risk == RISK_UNVALIDATED


def test_kimi_unvalidated() -> None:
    rv = classify_root_model(_FakeRoot("kimi-k2.5", "openrouter", False))
    assert rv.validated is False
    assert rv.risk == RISK_UNVALIDATED


def test_azure_unvalidated() -> None:
    rv = classify_root_model(_FakeRoot("azure-gpt-4o", "azure_openai", False))
    assert rv.validated is False
    assert rv.risk == RISK_UNVALIDATED


def test_is_degenerate_loop_risk_only_for_oauth() -> None:
    assert is_degenerate_loop_risk(_FakeRoot("claude-oauth", "anthropic-oauth", False)) is True
    assert is_degenerate_loop_risk(_FakeRoot("claude", "anthropic", False)) is False
    assert is_degenerate_loop_risk(_FakeRoot("gpt-5", "openai", True)) is False
    assert is_degenerate_loop_risk(_FakeRoot("kimi-k2.5", "openrouter", False)) is False


def test_degenerate_loop_risk_wins_even_if_validated() -> None:
    # Defensive: an oauth backend is flagged degenerate_loop regardless of the
    # paper_validated bit (the risk axis is backend-driven, not validation-driven).
    rv = classify_root_model(_FakeRoot("hypothetical", "anthropic-oauth", True))
    assert rv.validated is True
    assert rv.risk == RISK_DEGENERATE_LOOP


def test_classify_real_registry_models() -> None:
    # Hermetic: resolve_root_model on explicit names does not probe credentials.
    from backend.agents.rlm.models import resolve_root_model

    gpt5 = classify_root_model(resolve_root_model("gpt-5"))
    assert gpt5.validated is True and gpt5.risk == RISK_NONE

    oauth = classify_root_model(resolve_root_model("claude-oauth"))
    assert oauth.validated is False and oauth.risk == RISK_DEGENERATE_LOOP
