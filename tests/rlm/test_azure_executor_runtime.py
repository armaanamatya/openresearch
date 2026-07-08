"""Hermetic tests for the Azure OpenAI executor runtime (Stream D).

No network calls — all external clients are stubbed via monkeypatch.
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock


import backend.agents.rlm.executor as ex
from backend.agents.rlm.executor import ExecutorPlan, resolve_executor
from backend.agents.runtime.azure_openai_runtime import AzureOpenAiAgentRuntime
from backend.services.context.workspace.tools.azure_openai_client import (
    DEFAULT_AZURE_OPENAI_API_VERSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_agents():
    """A minimal fake ``agents`` module with the attributes _configure_sdk_client touches."""
    m = types.ModuleType("agents")
    m.OpenAIChatCompletionsModel = MagicMock(name="OpenAIChatCompletionsModel")
    m.set_tracing_disabled = MagicMock(name="set_tracing_disabled")
    return m


# ---------------------------------------------------------------------------
# AzureOpenAiAgentRuntime unit tests
# ---------------------------------------------------------------------------

def test_provider_name_is_openai():
    rt = AzureOpenAiAgentRuntime(
        azure_endpoint="https://x.openai.azure.com/",
        api_key="k",
        deployment="gpt-4o",
    )
    assert rt.provider_name == "openai"


def test_model_override_returns_deployment():
    rt = AzureOpenAiAgentRuntime(
        azure_endpoint="https://x.openai.azure.com/",
        api_key="k",
        deployment="my-deploy",
    )
    assert rt._model_override() == "my-deploy"


def test_model_override_none_when_no_deployment():
    rt = AzureOpenAiAgentRuntime(
        azure_endpoint="https://x.openai.azure.com/",
        api_key="k",
        deployment="",
    )
    assert rt._model_override() is None


def test_configure_sdk_client_builds_async_azure_openai(monkeypatch):
    """_configure_sdk_client must call AsyncAzureOpenAI with the right kwargs."""
    captured: dict = {}

    class FakeAsyncAzureOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai as _openai_mod
    monkeypatch.setattr(_openai_mod, "AsyncAzureOpenAI", FakeAsyncAzureOpenAI)

    rt = AzureOpenAiAgentRuntime(
        azure_endpoint="https://myres.openai.azure.com/",
        api_key="test-key",
        api_version="2024-10-21",
        deployment="gpt-4o-dep",
    )
    fake_agents = _make_fake_agents()
    client, chat_model_cls = rt._configure_sdk_client(fake_agents)

    assert isinstance(client, FakeAsyncAzureOpenAI)
    assert captured["azure_endpoint"] == "https://myres.openai.azure.com/"
    assert captured["api_key"] == "test-key"
    assert captured["api_version"] == "2024-10-21"
    assert captured["azure_deployment"] == "gpt-4o-dep"
    # tracing must be disabled
    fake_agents.set_tracing_disabled.assert_called_once_with(True)
    # chat_model_cls must be the fake agents-module attribute
    assert chat_model_cls is fake_agents.OpenAIChatCompletionsModel


def test_configure_sdk_client_uses_default_api_version(monkeypatch):
    """When no api_version is supplied, falls back to DEFAULT_AZURE_OPENAI_API_VERSION."""
    captured: dict = {}

    class FakeAsyncAzureOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai as _openai_mod
    monkeypatch.setattr(_openai_mod, "AsyncAzureOpenAI", FakeAsyncAzureOpenAI)

    rt = AzureOpenAiAgentRuntime(
        azure_endpoint="https://x.openai.azure.com/",
        api_key="k",
        deployment="dep",
    )
    rt._configure_sdk_client(_make_fake_agents())
    assert captured["api_version"] == DEFAULT_AZURE_OPENAI_API_VERSION


def test_env_var_resolution(monkeypatch):
    """Constructor resolves all four fields from env vars when args are omitted."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env-res.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-dep")

    rt = AzureOpenAiAgentRuntime()
    assert rt._azure_endpoint == "https://env-res.openai.azure.com/"
    assert rt._api_key == "env-key"
    assert rt._api_version == "2025-01-01"
    assert rt._deployment == "env-dep"
    assert rt._model_override() == "env-dep"


# ---------------------------------------------------------------------------
# resolve_executor integration tests
# ---------------------------------------------------------------------------

def _patch_azure_creds(monkeypatch, *, has_creds: bool, deployment: str = "gpt-4o"):
    """Set up monkeypatch env so _has_azure_openai_credentials() returns has_creds."""
    if has_creds:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
    else:
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    if deployment:
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", deployment)
    else:
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)


def test_resolve_executor_azure_returns_plan(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTOR", "azure")
    _patch_azure_creds(monkeypatch, has_creds=True, deployment="my-gpt4o")

    plan = resolve_executor()

    assert isinstance(plan, ExecutorPlan)
    assert isinstance(plan.runtime, AzureOpenAiAgentRuntime)
    assert plan.model == "my-gpt4o"
    assert plan.label.startswith("azure:")


def test_resolve_executor_azure_openai_alias(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTOR", "azure-openai")
    _patch_azure_creds(monkeypatch, has_creds=True, deployment="dep2")

    plan = resolve_executor()
    assert isinstance(plan, ExecutorPlan)
    assert plan.runtime.provider_name == "openai"


def test_resolve_executor_aoai_alias(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTOR", "aoai")
    _patch_azure_creds(monkeypatch, has_creds=True, deployment="dep3")

    plan = resolve_executor()
    assert isinstance(plan, ExecutorPlan)


def test_resolve_executor_azure_missing_creds_falls_back(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTOR", "azure")
    _patch_azure_creds(monkeypatch, has_creds=False, deployment="gpt-4o")

    assert resolve_executor() is None


def test_resolve_executor_azure_missing_deployment_falls_back(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_EXECUTOR", "azure")
    _patch_azure_creds(monkeypatch, has_creds=True, deployment="")

    assert resolve_executor() is None


# ---------------------------------------------------------------------------
# Regression — existing paths unaffected
# ---------------------------------------------------------------------------

def test_default_unset_returns_none(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EXECUTOR", raising=False)
    assert resolve_executor() is None


def test_vllm_mode_path_unaffected(monkeypatch):
    """A vLLM mode that fails the probe still returns None (not an Azure plan)."""
    monkeypatch.setenv("OPENRESEARCH_EXECUTOR", "qwen")
    monkeypatch.setattr(ex, "_probe", lambda *a, **k: False)
    assert resolve_executor() is None
