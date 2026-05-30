"""Phase 4 — route nav sub-RLM to OpenAI gpt-5-mini via the endpoint accelerator.

Moves the hot-volume rlm_query/llm_query navigation off the bundled-CLI transport
(no read-idle, orphaned-child wedge) onto the openai/httpx transport (raises
ReadTimeout, retries 2x, leaks no subprocess) with a tight read timeout. Grader/
verify stay on Sonnet-OAuth (scope=navigation). User has OPENAI_API_KEY.
"""
from __future__ import annotations

from backend.agents.rlm import accelerator as acc


def test_endpoint_falls_back_to_openai_api_key(monkeypatch):
    monkeypatch.setenv("REPROLAB_ACCELERATOR", "endpoint")
    monkeypatch.setenv("REPROLAB_ACCELERATOR_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("REPROLAB_ACCELERATOR_MODEL", "gpt-5-mini")
    monkeypatch.delenv("REPROLAB_ACCELERATOR_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-xyz")
    monkeypatch.setattr(acc, "probe_endpoint", lambda *a, **k: True)  # skip network

    ep = acc.resolve_accelerator("endpoint")
    assert ep is not None
    assert ep.model == "gpt-5-mini"
    assert ep.api_key == "sk-test-xyz"  # fell back to OPENAI_API_KEY, not "local"


def test_non_openai_endpoint_keeps_local_default(monkeypatch):
    """A self-hosted vLLM endpoint must NOT pick up OPENAI_API_KEY."""
    monkeypatch.setenv("REPROLAB_ACCELERATOR_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("REPROLAB_ACCELERATOR_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
    monkeypatch.delenv("REPROLAB_ACCELERATOR_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setattr(acc, "probe_endpoint", lambda *a, **k: True)

    ep = acc.resolve_accelerator("endpoint")
    assert ep is not None
    assert ep.api_key == "local"


def test_other_backend_kwargs_carry_read_timeout(monkeypatch):
    from backend.agents.rlm.run import _build_accel_sub_backend_kwargs

    ep = acc.AcceleratorEndpoint(
        base_url="https://api.openai.com/v1", model="gpt-5-mini",
        api_key="sk-x", kind="endpoint", is_azure=False,
    )
    monkeypatch.setenv("REPROLAB_SUBRLM_OPENAI_TIMEOUT_S", "90")
    kwargs = _build_accel_sub_backend_kwargs(ep)
    assert kwargs["model_name"] == "gpt-5-mini"
    assert kwargs["base_url"] == "https://api.openai.com/v1"
    assert kwargs["api_key"] == "sk-x"
    assert kwargs["timeout"] == 90.0


def test_other_backend_kwargs_default_timeout(monkeypatch):
    from backend.agents.rlm.run import _build_accel_sub_backend_kwargs

    ep = acc.AcceleratorEndpoint(
        base_url="https://api.openai.com/v1", model="gpt-5-mini",
        api_key="sk-x", kind="endpoint", is_azure=False,
    )
    monkeypatch.delenv("REPROLAB_SUBRLM_OPENAI_TIMEOUT_S", raising=False)
    kwargs = _build_accel_sub_backend_kwargs(ep)
    assert kwargs["timeout"] == 120.0
