"""P1 / Gap A — Claude root allowed_tools parity + hermetic SDK config.

Closes invariant 3: the Claude root previously inherited ALL default SDK tools
(no allowed_tools) while OpenAI restricted both providers. The helpers are pure so
the contract is testable without the SDK.
"""

from __future__ import annotations

from backend.agents.registry import AGENT_REGISTRY
from backend.agents.runtime.base import AgentRuntimeSpec
from backend.agents.runtime.claude_runtime import (
    _agent_options_kwargs,
    _hermetic_enabled,
    _tools_for_agent,
)
from backend.agents.runtime.openai_runtime import _TOOL_DESCRIPTIONS as _OPENAI_TOOL_VOCAB


def test_cross_provider_tool_parity():
    """For every registered agent: Claude allowed_tools == registry declaration,
    and every declared tool is within OpenAI's buildable vocabulary. Catches a
    future tool added on one provider but not honored on the other."""
    for agent_id, entry in AGENT_REGISTRY.items():
        spec = entry.to_runtime_spec("anthropic")
        names = [t.name for t in spec.tools]
        assert _tools_for_agent(spec, {}) == names, agent_id
        assert set(names).issubset(set(_OPENAI_TOOL_VOCAB)), (agent_id, names)


def test_tools_for_agent_merges_mcp_extensions():
    spec = AGENT_REGISTRY["baseline-implementation"].to_runtime_spec("anthropic")
    out = _tools_for_agent(spec, {spec.name: ["mcp__apify-arxiv"]})
    assert "mcp__apify-arxiv" in out
    assert out[: len(spec.tools)] == [t.name for t in spec.tools]  # base first, extra appended


def test_tools_for_agent_none_when_empty():
    spec = AgentRuntimeSpec(name="x", instructions="i", model="m", tools=())
    assert _tools_for_agent(spec, {}) is None


def test_root_options_restrict_allowed_tools():
    """The core Gap A fix: the root spec's allowed_tools is its declared tools."""
    spec = AGENT_REGISTRY["baseline-implementation"].to_runtime_spec("anthropic")
    kw = _agent_options_kwargs(spec, {}, {})
    assert kw["allowed_tools"] == [t.name for t in spec.tools]


def test_root_options_keep_bypass_permissions():
    spec = AGENT_REGISTRY["baseline-implementation"].to_runtime_spec("anthropic")
    assert _agent_options_kwargs(spec, {}, {})["permission_mode"] == "bypassPermissions"


def test_root_options_mcp_servers_always_explicit():
    spec = AGENT_REGISTRY["baseline-implementation"].to_runtime_spec("anthropic")
    kw = _agent_options_kwargs(spec, {}, {})
    assert "mcp_servers" in kw and kw["mcp_servers"] == {}


def test_root_options_empty_tools_omits_allowed_tools():
    """Defensive all-defaults path: a direct empty spec must NOT pass
    allowed_tools=[] (the registry guard makes this unreachable for real agents)."""
    spec = AgentRuntimeSpec(name="x", instructions="i", model="m", tools=())
    assert "allowed_tools" not in _agent_options_kwargs(spec, {}, {})


def test_hermetic_on_by_default(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_SDK_HERMETIC", raising=False)
    assert _hermetic_enabled() is True
    spec = AGENT_REGISTRY["baseline-implementation"].to_runtime_spec("anthropic")
    kw = _agent_options_kwargs(spec, {}, {})
    assert kw["setting_sources"] == [] and kw["strict_mcp_config"] is True


def test_hermetic_off_via_hatch_keeps_tool_restriction(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SDK_HERMETIC", "false")
    assert _hermetic_enabled() is False
    spec = AGENT_REGISTRY["baseline-implementation"].to_runtime_spec("anthropic")
    kw = _agent_options_kwargs(spec, {}, {})
    assert "setting_sources" not in kw and "strict_mcp_config" not in kw
    # The allowed_tools restriction stays on regardless of the hermetic hatch
    # (a hatch for it would re-open invariant 3).
    assert kw["allowed_tools"] == [t.name for t in spec.tools]


def test_hermetic_hatch_value_parsing(monkeypatch):
    for off in ("0", "false", "no", "off", "False", "OFF"):
        monkeypatch.setenv("OPENRESEARCH_SDK_HERMETIC", off)
        assert _hermetic_enabled() is False, off
    for on in ("1", "true", "yes", "anything-else"):
        monkeypatch.setenv("OPENRESEARCH_SDK_HERMETIC", on)
        assert _hermetic_enabled() is True, on
