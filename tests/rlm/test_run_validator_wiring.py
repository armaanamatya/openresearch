"""Tests for the P2.3 validator wiring in run.py.

Covers:
  - _validator_separation_tier for all four tiers
  - build_validator_client fail-closed when backend is set but creds are missing
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.run import _validator_separation_tier
from backend.agents.rlm.role_models import (
    RoleSelection,
    RoleSpec,
    PROVIDER_ANTHROPIC_OAUTH,
    PROVIDER_AZURE,
    _classify_model_family,
)


# ---------------------------------------------------------------------------
# Helpers to build a minimal RoleSelection with a given executor spec
# ---------------------------------------------------------------------------


def _planner_spec(token: str = "claude-oauth") -> RoleSpec:
    return RoleSpec(
        role="planner",
        token=token,
        provider=PROVIDER_ANTHROPIC_OAUTH,
        model="claude-sonnet-4-6",
        family="claude",
    )


def _role_selection(executor: RoleSpec | None = None) -> RoleSelection:
    return RoleSelection(
        planner=_planner_spec(),
        executor=executor,
        verifier=None,
        grader=None,
        validator=None,
    )


def _azure_executor_spec(deployment: str = "gpt-4o-execA") -> RoleSpec:
    return RoleSpec(
        role="executor",
        token="azure",
        provider=PROVIDER_AZURE,
        model=deployment,
        family=_classify_model_family(PROVIDER_AZURE, deployment),
    )


def _oauth_executor_spec() -> RoleSpec:
    return RoleSpec(
        role="executor",
        token="sonnet",
        provider=PROVIDER_ANTHROPIC_OAUTH,
        model="claude-sonnet-4-6",
        family="claude",
    )


# ---------------------------------------------------------------------------
# Tier: "independent" — different families
# ---------------------------------------------------------------------------


def test_independent_tier_oauth_executor_vs_azure_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OAuth/Claude executor × Azure validator → independent (different families)."""
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "azure")
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "gpt-4o-valB")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-execA")

    sel = _role_selection(executor=_oauth_executor_spec())
    tier = _validator_separation_tier(sel)
    assert tier == "independent", f"expected independent, got {tier!r}"


# ---------------------------------------------------------------------------
# Tier: "independent" — no executor spec (validator with no peer)
# ---------------------------------------------------------------------------


def test_independent_tier_no_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """No executor spec → independent (the validator is distinct from the default path)."""
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "azure")
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "gpt-4o-valB")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-execA")

    sel = _role_selection(executor=None)
    tier = _validator_separation_tier(sel)
    assert tier == "independent", f"expected independent, got {tier!r}"


# ---------------------------------------------------------------------------
# Tier: "weak" — same family (azure × azure), DIFFERENT deployments
# ---------------------------------------------------------------------------


def test_weak_tier_azure_executor_different_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Azure executor(deployA) × Azure validator(deployB) → weak (same family, different model)."""
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "azure")
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "gpt-4o-valB")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-execA")

    # executor provider=azure, model will be replaced with AZURE_OPENAI_DEPLOYMENT=execA
    sel = _role_selection(executor=_azure_executor_spec("gpt-4o-execA"))
    tier = _validator_separation_tier(sel)
    assert tier == "weak", f"expected weak, got {tier!r}"


# ---------------------------------------------------------------------------
# Tier: "degraded" — same family AND same model/deployment
# ---------------------------------------------------------------------------


def test_degraded_tier_same_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Azure executor(deployX) × Azure validator(deployX) → degraded."""
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "azure")
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "gpt-4o-shared")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-shared")

    sel = _role_selection(executor=_azure_executor_spec("gpt-4o-shared"))
    tier = _validator_separation_tier(sel)
    assert tier == "degraded", f"expected degraded, got {tier!r}"


# ---------------------------------------------------------------------------
# Tier: "unavailable" — OPENRESEARCH_VALIDATOR_BACKEND unset
# ---------------------------------------------------------------------------


def test_unavailable_tier_when_backend_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """When OPENRESEARCH_VALIDATOR_BACKEND is unset → unavailable."""
    monkeypatch.delenv("OPENRESEARCH_VALIDATOR_BACKEND", raising=False)

    sel = _role_selection(executor=_oauth_executor_spec())
    tier = _validator_separation_tier(sel)
    assert tier == "unavailable", f"expected unavailable, got {tier!r}"


def test_unavailable_tier_when_role_selection_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When role_selection is None and backend is unset → unavailable."""
    monkeypatch.delenv("OPENRESEARCH_VALIDATOR_BACKEND", raising=False)
    tier = _validator_separation_tier(None)
    assert tier == "unavailable"


# ---------------------------------------------------------------------------
# Fail-closed: build_validator_client raises ValueError when backend is set
#              but the required credentials are absent
# ---------------------------------------------------------------------------


def test_build_validator_client_fail_closed_on_missing_creds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With OPENRESEARCH_VALIDATOR_BACKEND=azure but no AZURE_OPENAI_* creds,
    build_validator_client raises ValueError (fail-closed)."""
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "azure")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

    from backend.agents.rlm.grader_transport import build_validator_client

    # A sentinel object as the fallback client.
    fallback = object()
    with pytest.raises(ValueError, match="OPENRESEARCH_VALIDATOR_BACKEND"):
        build_validator_client(fallback_client=fallback, fallback_label="test")


# ---------------------------------------------------------------------------
# Sanity: build_validator_client returns fallback unchanged when backend unset
# ---------------------------------------------------------------------------


def test_build_validator_client_no_op_when_backend_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With OPENRESEARCH_VALIDATOR_BACKEND unset, build_validator_client is a no-op."""
    monkeypatch.delenv("OPENRESEARCH_VALIDATOR_BACKEND", raising=False)

    from backend.agents.rlm.grader_transport import build_validator_client

    fallback = object()
    client, label = build_validator_client(fallback_client=fallback, fallback_label="my-label")
    assert client is fallback
    assert label == "my-label"
