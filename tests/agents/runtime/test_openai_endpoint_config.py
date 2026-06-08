"""OpenAI Agents SDK → custom endpoint (executor tier: run agents on local Qwen via vLLM)."""
from __future__ import annotations

from backend.agents.runtime.factory import configure_openai_agents_sdk_for_endpoint
from backend.agents.runtime.openai_runtime import OpenAiAgentRuntime


def test_runtime_stores_endpoint():
    rt = OpenAiAgentRuntime(base_url="http://127.0.0.1:8001/v1", api_key="local")
    assert rt._base_url == "http://127.0.0.1:8001/v1"
    assert rt._api_key == "local"
    assert rt.provider_name == "openai"


def test_runtime_default_has_no_endpoint():
    rt = OpenAiAgentRuntime()
    assert rt._base_url is None
    assert rt._use_chat_completions is True


def test_configure_endpoint_installs_custom_client_and_chat_completions():
    captured: dict[str, object] = {"client": None, "api": "unset"}
    configure_openai_agents_sdk_for_endpoint(
        "http://127.0.0.1:8001/v1",
        "local",
        set_default_openai_client=lambda c, **kw: captured.__setitem__("client", c),
        set_default_openai_api=lambda a: captured.__setitem__("api", a),
    )
    assert captured["client"] is not None
    assert str(getattr(captured["client"], "base_url", "")).startswith("http://127.0.0.1:8001")
    assert captured["api"] == "chat_completions"


def test_configure_endpoint_responses_mode_skips_api_switch():
    captured: dict[str, object] = {"api": "untouched"}
    configure_openai_agents_sdk_for_endpoint(
        "http://x/v1",
        "k",
        set_default_openai_client=lambda c, **kw: None,
        set_default_openai_api=lambda a: captured.__setitem__("api", a),
        use_chat_completions=False,
    )
    assert captured["api"] == "untouched"
