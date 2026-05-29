"""Tests for backend.agents.rlm.accelerator.

All network I/O is monkeypatched — no real HTTP calls are made.
"""

from __future__ import annotations

import os
import importlib
from unittest.mock import patch, MagicMock

import pytest

from backend.agents.rlm.accelerator import (
    AcceleratorEndpoint,
    AcceleratorError,
    build_accelerator_client,
    probe_endpoint,
    resolve_accelerator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _probe_true(_url: str, **_kw) -> bool:
    return True


def _probe_false(_url: str, **_kw) -> bool:
    return False


# ---------------------------------------------------------------------------
# resolve_accelerator("off")
# ---------------------------------------------------------------------------


class TestResolveOff:
    def test_returns_none(self):
        assert resolve_accelerator("off") is None

    def test_returns_none_case_insensitive(self):
        assert resolve_accelerator("OFF") is None


# ---------------------------------------------------------------------------
# resolve_accelerator("endpoint")
# ---------------------------------------------------------------------------


class TestResolveEndpoint:
    def test_returns_endpoint_when_probe_ok(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_ACCELERATOR_BASE_URL", "http://host:8001/v1")
        monkeypatch.setenv("REPROLAB_ACCELERATOR_MODEL", "my-model")
        monkeypatch.setenv("REPROLAB_ACCELERATOR_API_KEY", "tok-123")

        with patch(
            "backend.agents.rlm.accelerator.probe_endpoint", side_effect=_probe_true
        ):
            ep = resolve_accelerator("endpoint")

        assert isinstance(ep, AcceleratorEndpoint)
        assert ep.base_url == "http://host:8001/v1"
        assert ep.model == "my-model"
        assert ep.api_key == "tok-123"
        assert ep.kind == "endpoint"
        assert ep.is_azure is False

    def test_raises_when_url_missing(self, monkeypatch):
        monkeypatch.delenv("REPROLAB_ACCELERATOR_BASE_URL", raising=False)
        with pytest.raises(AcceleratorError, match="REPROLAB_ACCELERATOR_BASE_URL"):
            resolve_accelerator("endpoint")

    def test_raises_when_model_missing(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_ACCELERATOR_BASE_URL", "http://host:8001/v1")
        monkeypatch.delenv("REPROLAB_ACCELERATOR_MODEL", raising=False)
        with pytest.raises(AcceleratorError, match="REPROLAB_ACCELERATOR_MODEL"):
            resolve_accelerator("endpoint")

    def test_raises_when_probe_fails(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_ACCELERATOR_BASE_URL", "http://host:8001/v1")
        monkeypatch.setenv("REPROLAB_ACCELERATOR_MODEL", "m")
        with patch(
            "backend.agents.rlm.accelerator.probe_endpoint", side_effect=_probe_false
        ):
            with pytest.raises(AcceleratorError, match="health probe"):
                resolve_accelerator("endpoint")

    def test_default_api_key_is_local(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_ACCELERATOR_BASE_URL", "http://host:8001/v1")
        monkeypatch.setenv("REPROLAB_ACCELERATOR_MODEL", "m")
        monkeypatch.delenv("REPROLAB_ACCELERATOR_API_KEY", raising=False)
        with patch(
            "backend.agents.rlm.accelerator.probe_endpoint", side_effect=_probe_true
        ):
            ep = resolve_accelerator("endpoint")
        assert ep.api_key == "local"


# ---------------------------------------------------------------------------
# resolve_accelerator("local")
# ---------------------------------------------------------------------------


class TestResolveLocal:
    def test_returns_endpoint_when_probe_ok(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_ACCELERATOR_BASE_URL", "http://127.0.0.1:8001/v1")
        monkeypatch.setenv("REPROLAB_ACCELERATOR_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
        with patch(
            "backend.agents.rlm.accelerator.probe_endpoint", side_effect=_probe_true
        ):
            ep = resolve_accelerator("local")
        assert ep is not None
        assert ep.kind == "local"
        assert ep.base_url == "http://127.0.0.1:8001/v1"

    def test_returns_none_when_probe_fails_explicit(self, monkeypatch):
        """Explicit 'local' with a dead server returns None (not a hard error).

        The contract is documented: the server may simply not be running yet;
        callers should fall back to the default Sonnet/OAuth path.
        """
        monkeypatch.delenv("REPROLAB_ACCELERATOR_BASE_URL", raising=False)
        with patch(
            "backend.agents.rlm.accelerator.probe_endpoint", side_effect=_probe_false
        ):
            ep = resolve_accelerator("local")
        assert ep is None

    def test_default_url_and_model_used_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("REPROLAB_ACCELERATOR_BASE_URL", raising=False)
        monkeypatch.delenv("REPROLAB_ACCELERATOR_MODEL", raising=False)
        captured = {}

        def _capture(url, **kw):
            captured["url"] = url
            return True

        with patch("backend.agents.rlm.accelerator.probe_endpoint", side_effect=_capture):
            ep = resolve_accelerator("local")

        assert "127.0.0.1:8001" in captured["url"]
        assert ep is not None
        assert "Qwen" in ep.model


# ---------------------------------------------------------------------------
# resolve_accelerator("runpod")
# ---------------------------------------------------------------------------


class TestResolveRunpod:
    def test_raises_when_no_url_set_explicit(self, monkeypatch):
        monkeypatch.delenv("REPROLAB_ACCELERATOR_BASE_URL", raising=False)
        with pytest.raises(AcceleratorError, match="auto-provisioning not yet implemented"):
            resolve_accelerator("runpod")

    def test_returns_endpoint_when_url_set_and_probe_ok(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_ACCELERATOR_BASE_URL", "http://runpod-proxy:8001/v1")
        monkeypatch.setenv("REPROLAB_ACCELERATOR_MODEL", "my-model")
        with patch(
            "backend.agents.rlm.accelerator.probe_endpoint", side_effect=_probe_true
        ):
            ep = resolve_accelerator("runpod")
        assert ep is not None
        assert ep.kind == "runpod"

    def test_raises_when_url_set_but_probe_fails_explicit(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_ACCELERATOR_BASE_URL", "http://runpod-proxy:8001/v1")
        monkeypatch.setenv("REPROLAB_ACCELERATOR_MODEL", "my-model")
        with patch(
            "backend.agents.rlm.accelerator.probe_endpoint", side_effect=_probe_false
        ):
            with pytest.raises(AcceleratorError, match="probe failed"):
                resolve_accelerator("runpod")


# ---------------------------------------------------------------------------
# resolve_accelerator("azure")
# ---------------------------------------------------------------------------


class TestResolveAzure:
    def test_returns_endpoint_when_creds_present(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myres.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-deploy")

        ep = resolve_accelerator("azure")

        assert ep is not None
        assert ep.is_azure is True
        assert ep.kind == "azure"
        assert ep.api_key == "fake-key"
        assert ep.model == "gpt-4o-deploy"

    def test_raises_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myres.openai.azure.com")
        with pytest.raises(AcceleratorError, match="AZURE_OPENAI_API_KEY"):
            resolve_accelerator("azure")

    def test_raises_when_endpoint_missing(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        with pytest.raises(AcceleratorError, match="AZURE_OPENAI_ENDPOINT"):
            resolve_accelerator("azure")

    def test_model_defaults_to_gpt4o_when_deployment_absent(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myres.openai.azure.com")
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

        ep = resolve_accelerator("azure")
        assert ep is not None
        assert ep.model == "gpt-4o"


# ---------------------------------------------------------------------------
# resolve_accelerator("auto")
# ---------------------------------------------------------------------------


class TestResolveAuto:
    def test_returns_none_when_no_providers_satisfied(self, monkeypatch):
        """No GPU, no runpod env, no azure creds → None."""
        monkeypatch.delenv("REPROLAB_ACCELERATOR_BASE_URL", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

        with patch(
            "backend.agents.rlm.accelerator.probe_endpoint", side_effect=_probe_false
        ), patch(
            "backend.services.runtime.gpu_resolution.host_supports_nvidia_gpu",
            return_value=False,
        ):
            ep = resolve_accelerator("auto")

        assert ep is None

    def test_prefers_local_when_gpu_present_and_probe_ok(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_ACCELERATOR_BASE_URL", "http://127.0.0.1:8001/v1")
        monkeypatch.setenv("REPROLAB_ACCELERATOR_MODEL", "Qwen/Qwen2.5-Coder-32B")

        with patch(
            "backend.agents.rlm.accelerator.probe_endpoint", side_effect=_probe_true
        ), patch(
            "backend.services.runtime.gpu_resolution.host_supports_nvidia_gpu",
            return_value=True,
        ):
            ep = resolve_accelerator("auto")

        assert ep is not None
        assert ep.kind == "local"

    def test_falls_back_to_azure_when_local_unavailable(self, monkeypatch):
        monkeypatch.delenv("REPROLAB_ACCELERATOR_BASE_URL", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://r.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-dep")

        with patch(
            "backend.agents.rlm.accelerator.probe_endpoint", side_effect=_probe_false
        ), patch(
            "backend.services.runtime.gpu_resolution.host_supports_nvidia_gpu",
            return_value=False,
        ):
            ep = resolve_accelerator("auto")

        assert ep is not None
        assert ep.kind == "azure"

    def test_returns_none_when_no_gpu_no_azure(self, monkeypatch):
        monkeypatch.delenv("REPROLAB_ACCELERATOR_BASE_URL", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

        with patch(
            "backend.agents.rlm.accelerator.probe_endpoint", side_effect=_probe_false
        ), patch(
            "backend.services.runtime.gpu_resolution.host_supports_nvidia_gpu",
            return_value=False,
        ):
            ep = resolve_accelerator("auto")

        assert ep is None

    def test_never_raises(self, monkeypatch):
        """auto mode must not raise even when something unexpected breaks."""
        monkeypatch.delenv("REPROLAB_ACCELERATOR_BASE_URL", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

        def _boom(_url=None, **_kw):
            raise RuntimeError("boom")

        with patch(
            "backend.agents.rlm.accelerator.probe_endpoint", side_effect=_boom
        ), patch(
            "backend.services.runtime.gpu_resolution.host_supports_nvidia_gpu",
            side_effect=_boom,
        ):
            ep = resolve_accelerator("auto")

        assert ep is None


# ---------------------------------------------------------------------------
# build_accelerator_client
# ---------------------------------------------------------------------------


class TestBuildAcceleratorClient:
    def test_non_azure_has_complete_attr(self, monkeypatch):
        ep = AcceleratorEndpoint(
            base_url="http://127.0.0.1:8001/v1",
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            api_key="local",
            kind="local",
            is_azure=False,
        )
        # Patch OpenAI constructor to avoid real network at import time.
        with patch("openai.OpenAI", return_value=MagicMock()):
            client = build_accelerator_client(ep)
        assert hasattr(client, "complete"), "client must expose .complete()"
        assert callable(client.complete)

    def test_azure_has_complete_attr(self, monkeypatch):
        ep = AcceleratorEndpoint(
            base_url="https://myres.openai.azure.com",
            model="gpt-4o-deploy",
            api_key="azure-key",
            kind="azure",
            is_azure=True,
        )
        with patch("openai.AzureOpenAI", return_value=MagicMock()):
            client = build_accelerator_client(ep)
        assert hasattr(client, "complete"), "azure client must expose .complete()"
        assert callable(client.complete)

    def test_non_azure_is_openai_client(self, monkeypatch):
        from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

        ep = AcceleratorEndpoint(
            base_url="http://127.0.0.1:8001/v1",
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            api_key="local",
        )
        with patch("openai.OpenAI", return_value=MagicMock()):
            client = build_accelerator_client(ep)
        assert isinstance(client, OpenAILlmClient)

    def test_azure_is_azure_client(self, monkeypatch):
        from backend.services.context.workspace.tools.azure_openai_client import (
            AzureOpenAILlmClient,
        )

        ep = AcceleratorEndpoint(
            base_url="https://myres.openai.azure.com",
            model="gpt-4o-deploy",
            api_key="azure-key",
            kind="azure",
            is_azure=True,
        )
        with patch("openai.AzureOpenAI", return_value=MagicMock()):
            client = build_accelerator_client(ep)
        assert isinstance(client, AzureOpenAILlmClient)


# ---------------------------------------------------------------------------
# probe_endpoint
# ---------------------------------------------------------------------------


class TestProbeEndpoint:
    def test_returns_true_on_200(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200

        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert probe_endpoint("http://host:8001/v1") is True

    def test_returns_false_on_500(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 500

        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert probe_endpoint("http://host:8001/v1") is False

    def test_returns_false_on_network_error(self):
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            assert probe_endpoint("http://host:8001/v1") is False

    def test_appends_models_to_v1_url(self):
        captured = {}

        def _fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            raise Exception("stop")

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            probe_endpoint("http://host:8001/v1")

        assert captured["url"].endswith("/models")

    def test_does_not_double_append_models(self):
        captured = {}

        def _fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            raise Exception("stop")

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            probe_endpoint("http://host:8001/v1/models")

        assert captured["url"].count("/models") == 1


# ---------------------------------------------------------------------------
# Invalid mode
# ---------------------------------------------------------------------------


class TestInvalidMode:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown accelerator mode"):
            resolve_accelerator("bogus")
