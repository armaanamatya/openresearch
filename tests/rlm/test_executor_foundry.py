"""G2: OPENRESEARCH_EXECUTOR=azure-foundry/grok routes implement_baseline onto a
Foundry deployment; missing creds GRACEFULLY fall back to None (NOT fail-fast).
No network — the foundry runtime is faked."""
from __future__ import annotations

import types
import pytest

from backend.agents.rlm.executor import ExecutorPlan, resolve_executor


def _clear_foundry(monkeypatch):
    for k in ("AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_DEPLOYMENT", "AZURE_FOUNDRY_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("backend.config.get_settings", lambda *a, **k: types.SimpleNamespace())


@pytest.mark.parametrize("mode", ["azure-foundry", "foundry", "grok"])
def test_foundry_mode_returns_plan(monkeypatch, mode):
    monkeypatch.setenv("OPENRESEARCH_EXECUTOR", mode)
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://x.services.ai.azure.com")
    monkeypatch.setenv("AZURE_FOUNDRY_DEPLOYMENT", "grok-4.3")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "foundry-key")

    import backend.agents.runtime.azure_foundry_runtime as afr

    class _FakeRT:
        def __init__(self):
            self.built = True

    monkeypatch.setattr(afr, "AzureFoundryAgentRuntime", _FakeRT)

    plan = resolve_executor()
    assert isinstance(plan, ExecutorPlan)
    assert plan.model == "grok-4.3"
    assert plan.label == "azure-foundry:grok-4.3"
    assert isinstance(plan.runtime, _FakeRT)


def test_foundry_mode_missing_creds_falls_back_gracefully(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTOR", "grok")
    _clear_foundry(monkeypatch)
    assert resolve_executor() is None  # default Sonnet executor


def test_executor_unset_is_unchanged(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EXECUTOR", raising=False)
    assert resolve_executor() is None
