"""ACC-1 / ACC-2: accelerator endpoint route honors OPENAI_API_KEY and the
SUBRLM timeout env var (both were documented but not wired)."""
import pytest

from backend.agents.rlm import accelerator
from backend.agents.rlm.accelerator import AcceleratorEndpoint


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (
        "OPENRESEARCH_ACCELERATOR_BASE_URL",
        "OPENRESEARCH_ACCELERATOR_MODEL",
        "OPENRESEARCH_ACCELERATOR_API_KEY",
        "OPENAI_API_KEY",
        "OPENRESEARCH_SUBRLM_OPENAI_TIMEOUT_S",
    ):
        monkeypatch.delenv(k, raising=False)


def test_openai_host_falls_back_to_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ACCELERATOR_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENRESEARCH_ACCELERATOR_MODEL", "gpt-5-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
    # ACCELERATOR_API_KEY deliberately unset.
    seen = {}

    def fake_probe(base_url, *, api_key=None, timeout=3.0):
        seen["api_key"] = api_key
        return True

    monkeypatch.setattr(accelerator, "probe_endpoint", fake_probe)
    ep = accelerator._resolve_endpoint(explicit=True)
    assert ep is not None
    assert ep.api_key == "sk-real-key"  # ACC-2: not "local"
    assert seen["api_key"] == "sk-real-key"  # probe authenticated too


def test_explicit_accelerator_api_key_wins_over_openai(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ACCELERATOR_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENRESEARCH_ACCELERATOR_MODEL", "gpt-5-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
    monkeypatch.setenv("OPENRESEARCH_ACCELERATOR_API_KEY", "explicit-key")
    monkeypatch.setattr(accelerator, "probe_endpoint", lambda *a, **k: True)
    ep = accelerator._resolve_endpoint(explicit=True)
    assert ep.api_key == "explicit-key"


def test_non_openai_host_keeps_local_default(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ACCELERATOR_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("OPENRESEARCH_ACCELERATOR_MODEL", "qwen")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")  # must be ignored for local host
    monkeypatch.setattr(accelerator, "probe_endpoint", lambda *a, **k: True)
    ep = accelerator._resolve_endpoint(explicit=True)
    assert ep.api_key == "local"


def test_subrlm_timeout_is_honored(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "backend.services.context.workspace.tools.openai_client.OpenAILlmClient",
        FakeClient,
    )
    ep = AcceleratorEndpoint(base_url="http://x/v1", model="m", api_key="k", kind="endpoint")

    monkeypatch.setenv("OPENRESEARCH_SUBRLM_OPENAI_TIMEOUT_S", "120")
    accelerator.build_accelerator_client(ep)
    assert captured.get("timeout") == 120.0


def test_subrlm_timeout_unset_uses_client_default(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "backend.services.context.workspace.tools.openai_client.OpenAILlmClient",
        FakeClient,
    )
    ep = AcceleratorEndpoint(base_url="http://x/v1", model="m", api_key="k", kind="endpoint")
    accelerator.build_accelerator_client(ep)
    assert "timeout" not in captured  # falls through to OpenAILlmClient's 300s default
