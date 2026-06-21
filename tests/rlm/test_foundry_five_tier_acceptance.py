"""ACCEPTANCE: a single Foundry vocabulary (azure-foundry/foundry/grok/grok-4.3)
selects a Foundry model for ANY of the five tiers; with the vocabulary UNSET,
every tier is byte-for-byte today's path. No network — runtimes/clients faked or
only descriptor-resolved.

The ``grok-4.3`` parametrization is the NON-intersection alias: it exercises the
unified vocabulary end-to-end and proves the same mode alias works across all
five tiers (root, executor, verifier, grader, accelerator) — not just the three
aliases that historically intersected every site."""
from __future__ import annotations

import types
import pytest

# Every alias in the canonical FOUNDRY_MODE_ALIASES vocabulary, including the
# non-intersection ``grok-4.3`` that pre-unification only the executor/factory
# tiers accepted.
FOUNDRY_ALIASES = ["azure-foundry", "foundry", "grok", "grok-4.3"]


@pytest.fixture()
def foundry_env(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://x.services.ai.azure.com")
    monkeypatch.setenv("AZURE_FOUNDRY_DEPLOYMENT", "grok-4.3")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "foundry-key")
    yield


@pytest.fixture()
def no_foundry_env(monkeypatch):
    for k in ("AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_DEPLOYMENT", "AZURE_FOUNDRY_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("backend.config.get_settings", lambda *a, **k: types.SimpleNamespace())
    yield


@pytest.mark.parametrize("alias", FOUNDRY_ALIASES)
def test_tier_root_foundry(foundry_env, alias):
    from backend.agents.rlm.models import resolve_root_model
    entry = resolve_root_model(alias)
    assert entry.key == "azure-foundry"
    assert entry.backend_kwargs["model_name"] == "grok-4.3"


@pytest.mark.parametrize("alias", FOUNDRY_ALIASES)
def test_tier_executor_foundry(foundry_env, monkeypatch, alias):
    monkeypatch.setenv("OPENRESEARCH_EXECUTOR", alias)
    import backend.agents.runtime.azure_foundry_runtime as afr
    monkeypatch.setattr(afr, "AzureFoundryAgentRuntime", lambda: object())
    from backend.agents.rlm.executor import resolve_executor
    plan = resolve_executor()
    assert plan is not None
    assert plan.label == "azure-foundry:grok-4.3"


@pytest.mark.parametrize("alias", FOUNDRY_ALIASES)
@pytest.mark.parametrize("role", ["verifier", "grader"])
def test_tier_verifier_grader_foundry(foundry_env, monkeypatch, role, alias):
    import backend.services.context.workspace.tools.openai_client as oac

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(oac, "OpenAILlmClient", _FakeOpenAI)
    from backend.agents.rlm.grader_transport import build_transport_client
    client, label = build_transport_client(
        backend=alias, model=None,
        fallback_client=object(), fallback_label="fb", role_label=role,
    )
    assert isinstance(client, _FakeOpenAI)
    assert label == f"{role}:azure-foundry:grok-4.3"


@pytest.mark.parametrize("alias", FOUNDRY_ALIASES)
def test_tier_accelerator_foundry(foundry_env, alias):
    from backend.agents.rlm.accelerator import resolve_accelerator
    ep = resolve_accelerator(alias)
    assert ep is not None
    assert ep.kind == "foundry"
    assert ep.model == "grok-4.3"


def test_all_tiers_unset_inherit(no_foundry_env, monkeypatch):
    from backend.agents.rlm.executor import resolve_executor
    from backend.agents.rlm.accelerator import resolve_accelerator
    from backend.agents.rlm.grader_transport import build_grader_client

    monkeypatch.delenv("OPENRESEARCH_EXECUTOR", raising=False)
    assert resolve_executor() is None
    assert resolve_accelerator("off") is None

    monkeypatch.delenv("OPENRESEARCH_GRADER_BACKEND", raising=False)
    monkeypatch.delenv("OPENRESEARCH_GRADER_MODEL", raising=False)
    sentinel = object()
    client, label = build_grader_client(sentinel, "root-label")
    assert client is sentinel and label == "root-label"
