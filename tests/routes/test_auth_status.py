"""Tests for GET /auth-status — provider credential probing endpoint.

Design decision D1 from the lane-h spec: the endpoint returns a structured
dict of per-provider availability without requiring any LLM call or API key
validation side-effects.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


def _reset_settings_cache():
    import backend.config as _config
    _config._settings_cache = None


@pytest.fixture(autouse=True)
def _isolate_settings_cache():
    _reset_settings_cache()
    try:
        yield
    finally:
        _reset_settings_cache()


def _fresh_app(monkeypatch, *, runs_root_tmp):
    monkeypatch.setenv("REPROLAB_RUNS_ROOT", str(runs_root_tmp))
    monkeypatch.delenv("REPROLAB_DEMO_SECRET", raising=False)
    from backend.app import create_app
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------

def test_auth_status_returns_documented_shape(tmp_path, monkeypatch):
    """Response must contain providers, subagent_auth, and defaults keys."""
    with (
        patch("backend.agents.runtime.factory._has_claude_subscription_oauth", return_value=False),
        patch("backend.agents.runtime.factory._has_azure_openai_credentials", return_value=False),
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
        monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        client = _fresh_app(monkeypatch, runs_root_tmp=tmp_path)

    response = client.get("/auth-status")
    assert response.status_code == 200
    body = response.json()

    assert "providers" in body
    assert "subagent_auth" in body
    assert "defaults" in body

    providers = body["providers"]
    for key in ("anthropic_api", "anthropic_oauth", "openai_api", "azure_openai", "featherless"):
        assert key in providers, f"missing provider key: {key}"
        assert "available" in providers[key]
        assert "detail" in providers[key]

    subagent = body["subagent_auth"]
    assert "anthropic_api" in subagent
    assert "anthropic_oauth" in subagent

    defaults = body["defaults"]
    assert "root_provider" in defaults
    assert "root_model" in defaults
    assert "subagent_auth" in defaults
    assert defaults["root_model"] == "sonnet"


# ---------------------------------------------------------------------------
# All providers unavailable
# ---------------------------------------------------------------------------

def test_all_providers_unavailable(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    client = _fresh_app(monkeypatch, runs_root_tmp=tmp_path)

    with (
        patch("backend.agents.runtime.factory._has_claude_subscription_oauth", return_value=False),
        patch("backend.agents.runtime.factory._has_azure_openai_credentials", return_value=False),
        patch("backend.agents.runtime.factory._has_anthropic_api_credentials", return_value=False),
        patch("backend.agents.runtime.factory._has_openai_credentials", return_value=False),
        patch("backend.agents.runtime.factory._has_featherless_credentials", return_value=False),
    ):
        response = client.get("/auth-status")

    body = response.json()
    providers = body["providers"]

    assert providers["anthropic_api"]["available"] is False
    assert providers["anthropic_oauth"]["available"] is False
    assert providers["openai_api"]["available"] is False
    assert providers["azure_openai"]["available"] is False
    assert providers["featherless"]["available"] is False

    assert body["subagent_auth"]["anthropic_api"] is False
    assert body["subagent_auth"]["anthropic_oauth"] is False


# ---------------------------------------------------------------------------
# anthropic_api available
# ---------------------------------------------------------------------------

def test_anthropic_api_available(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    client = _fresh_app(monkeypatch, runs_root_tmp=tmp_path)

    with (
        patch("backend.agents.runtime.factory._has_claude_subscription_oauth", return_value=False),
        patch("backend.agents.runtime.factory._has_azure_openai_credentials", return_value=False),
    ):
        response = client.get("/auth-status")

    body = response.json()
    assert body["providers"]["anthropic_api"]["available"] is True
    assert "ANTHROPIC_API_KEY" in body["providers"]["anthropic_api"]["detail"]
    assert body["subagent_auth"]["anthropic_api"] is True
    assert body["defaults"]["root_provider"] == "anthropic_api"


# ---------------------------------------------------------------------------
# anthropic_oauth available (OAuth wins over API key in default priority)
# ---------------------------------------------------------------------------

def test_anthropic_oauth_available(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    client = _fresh_app(monkeypatch, runs_root_tmp=tmp_path)

    with (
        patch("backend.agents.runtime.factory._has_claude_subscription_oauth", return_value=True),
        patch("backend.agents.runtime.factory._has_azure_openai_credentials", return_value=False),
    ):
        response = client.get("/auth-status")

    body = response.json()
    assert body["providers"]["anthropic_oauth"]["available"] is True
    assert "claude CLI" in body["providers"]["anthropic_oauth"]["detail"]
    assert body["subagent_auth"]["anthropic_oauth"] is True
    # OAuth is highest priority default
    assert body["defaults"]["root_provider"] == "anthropic_oauth"
    assert body["defaults"]["subagent_auth"] == "anthropic_oauth"


# ---------------------------------------------------------------------------
# openai_api available
# ---------------------------------------------------------------------------

def test_openai_api_available(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    client = _fresh_app(monkeypatch, runs_root_tmp=tmp_path)

    with (
        patch("backend.agents.runtime.factory._has_claude_subscription_oauth", return_value=False),
        patch("backend.agents.runtime.factory._has_azure_openai_credentials", return_value=False),
    ):
        response = client.get("/auth-status")

    body = response.json()
    assert body["providers"]["openai_api"]["available"] is True
    assert "OPENAI_API_KEY" in body["providers"]["openai_api"]["detail"]
    assert body["defaults"]["root_provider"] == "openai_api"


# ---------------------------------------------------------------------------
# azure_openai available
# ---------------------------------------------------------------------------

def test_azure_openai_available(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    client = _fresh_app(monkeypatch, runs_root_tmp=tmp_path)

    with (
        patch("backend.agents.runtime.factory._has_claude_subscription_oauth", return_value=False),
        patch("backend.agents.runtime.factory._has_azure_openai_credentials", return_value=True),
        patch("backend.agents.runtime.factory._has_anthropic_api_credentials", return_value=False),
        patch("backend.agents.runtime.factory._has_openai_credentials", return_value=False),
        patch("backend.agents.runtime.factory._has_featherless_credentials", return_value=False),
    ):
        response = client.get("/auth-status")

    body = response.json()
    assert body["providers"]["azure_openai"]["available"] is True
    assert "Azure" in body["providers"]["azure_openai"]["detail"]
    assert body["defaults"]["root_provider"] == "azure_openai"


# ---------------------------------------------------------------------------
# featherless available
# ---------------------------------------------------------------------------

def test_featherless_available(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("FEATHERLESS_API_KEY", "fl-test-key")
    client = _fresh_app(monkeypatch, runs_root_tmp=tmp_path)

    with (
        patch("backend.agents.runtime.factory._has_claude_subscription_oauth", return_value=False),
        patch("backend.agents.runtime.factory._has_azure_openai_credentials", return_value=False),
        patch("backend.agents.runtime.factory._has_anthropic_api_credentials", return_value=False),
        patch("backend.agents.runtime.factory._has_openai_credentials", return_value=False),
        patch("backend.agents.runtime.factory._has_featherless_credentials", return_value=True),
    ):
        response = client.get("/auth-status")

    body = response.json()
    assert body["providers"]["featherless"]["available"] is True
    assert "FEATHERLESS_API_KEY" in body["providers"]["featherless"]["detail"]
    assert body["defaults"]["root_provider"] == "featherless"


# ---------------------------------------------------------------------------
# StartRunRequest accepts new optional fields without breaking existing ones
# ---------------------------------------------------------------------------

def test_start_run_request_new_optional_fields():
    """New fields are optional — omitting them keeps default None."""
    from backend.services.events.live_runs import StartRunRequest

    req = StartRunRequest()
    assert req.root_provider is None
    assert req.subagent_auth is None
    assert req.dynamic_gpu is None
    assert req.force_single_gpu is None
    assert req.max_gpu_usd_per_hour is None
    assert req.vram_gb is None


def test_start_run_request_new_fields_roundtrip():
    """New fields survive model_dump and reconstruction."""
    from backend.services.events.live_runs import StartRunRequest

    req = StartRunRequest(
        root_provider="anthropic_oauth",
        subagent_auth="anthropic_api",
        dynamic_gpu=True,
        force_single_gpu=False,
        max_gpu_usd_per_hour=0.5,
        vram_gb=24,
    )
    d = req.model_dump()
    assert d["root_provider"] == "anthropic_oauth"
    assert d["subagent_auth"] == "anthropic_api"
    assert d["dynamic_gpu"] is True
    assert d["force_single_gpu"] is False
    assert d["max_gpu_usd_per_hour"] == 0.5
    assert d["vram_gb"] == 24


def test_start_run_request_preserves_existing_defaults():
    """Existing fields keep their defaults — no regression."""
    from backend.services.events.live_runs import StartRunRequest

    req = StartRunRequest()
    assert req.mode == "rlm"
    assert req.provider == "anthropic"
    assert req.executionMode == "max"  # default flipped efficient→max in 738478a
    assert req.sandbox == "runpod"
    assert req.gpuMode == "auto"
    assert req.model == "sonnet"
    assert req.paper_id is None
