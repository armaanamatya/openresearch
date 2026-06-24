"""ADR 2026-06-21 Decision 1 — single Azure deployment resolver.

`grader_transport.build_transport_client` (azure branch) and
`run.py::_validator_separation_tier` previously each re-derived the Azure
`model -> deployment` rule ("kept in sync by comment only"). Both must now route
through the one pure helper `grader_transport.resolve_azure_deployment`, and the
helper must be byte-identical to the old inline logic:

    explicit override  ->  AZURE_OPENAI_DEPLOYMENT (stripped)  ->  None
"""
import pytest

from backend.agents.rlm.grader_transport import resolve_azure_deployment


@pytest.fixture(autouse=True)
def _clear_azure_env(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    yield


def test_override_wins_over_env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployA")
    assert resolve_azure_deployment("deployB") == "deployB"


def test_falls_back_to_env_when_no_override(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployA")
    assert resolve_azure_deployment(None) == "deployA"


def test_env_is_stripped(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "  deployA  ")
    assert resolve_azure_deployment(None) == "deployA"


def test_none_when_nothing_set(monkeypatch):
    assert resolve_azure_deployment(None) is None
    # empty env string also collapses to None (matches `.strip() or None`)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "   ")
    assert resolve_azure_deployment(None) is None


def test_validator_separation_tier_routes_through_resolver(monkeypatch):
    """azure(deployA) executor x azure(deployB) validator must read as 'weak',
    not 'degraded' — the behaviour the forked resolver protected, now via the
    shared helper."""
    from backend.agents.rlm import run as run_mod
    from backend.agents.rlm.role_models import RoleSpec, PROVIDER_AZURE

    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "azure")
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "deployB")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployA")

    exec_spec = RoleSpec(
        role="executor", token="azure", provider=PROVIDER_AZURE,
        model=None, family="gpt",
    )

    class _Sel:
        executor = exec_spec

    tier = run_mod._validator_separation_tier(_Sel())
    assert tier == "weak"
