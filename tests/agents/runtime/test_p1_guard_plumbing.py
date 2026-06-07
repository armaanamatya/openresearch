"""P1 / #7 Unit A — RuntimeGuard blocklist plumbing.

Covers the wiring (not the source resolution, which is Unit B): to_runtime_spec
seeds the spec's RuntimeGuard from blocked_terms, fails closed on an empty tool
list, and RunContext auto-loads blocked_terms from OPENRESEARCH_BLOCKED_TERMS_JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.registry import AGENT_REGISTRY, AgentSpec
from backend.agents.rlm.context import RunContext

_PAPER_REPO = "https://github.com/BartekCupial/finetuning-RL-as-CL"


def test_to_runtime_spec_sets_guard_from_blocked_terms():
    """blocked_terms seeds the spec's RuntimeGuard and the guard actually matches."""
    spec = AGENT_REGISTRY["baseline-implementation"].to_runtime_spec(
        "anthropic", blocked_terms=(_PAPER_REPO,)
    )
    assert spec.guard.blocked_terms == (_PAPER_REPO,)
    # Canonicalised match: a clone command referencing the repo is flagged.
    assert spec.guard.find_blocked_term(f"git clone {_PAPER_REPO}.git") is not None
    # A non-blocked framework repo is not flagged.
    assert spec.guard.find_blocked_term("pip install git+https://github.com/huggingface/trl") is None


def test_to_runtime_spec_default_guard_is_empty():
    """With no blocked_terms the guard is a no-op (preserves prior behavior)."""
    spec = AGENT_REGISTRY["baseline-implementation"].to_runtime_spec("anthropic")
    assert spec.guard.blocked_terms == ()
    assert spec.guard.find_blocked_term(f"git clone {_PAPER_REPO}") is None


def test_guard_applies_to_all_registry_agents():
    """The blocklist threads into EVERY registry agent uniformly (grill: all agents)."""
    for agent_id in AGENT_REGISTRY:
        spec = AGENT_REGISTRY[agent_id].to_runtime_spec("anthropic", blocked_terms=(_PAPER_REPO,))
        assert spec.guard.blocked_terms == (_PAPER_REPO,), agent_id


@pytest.mark.parametrize("tools", [[], ["Agent"]])
def test_to_runtime_spec_fails_closed_on_empty_tools(tools):
    """Fail-closed (invariant 3): an agent resolving to zero tools must raise,
    never silently fall through to the SDK 'all defaults' path."""
    spec = AgentSpec(
        agent_id="empty-agent",
        role="builder",
        description="d",
        prompt="p",
        tools=tools,
    )
    with pytest.raises(ValueError, match="empty tool list"):
        spec.to_runtime_spec("anthropic")


def test_all_registered_agents_have_nonempty_tools():
    """Registry-validation companion to the runtime assert: no shipped agent is empty."""
    for agent_id, spec in AGENT_REGISTRY.items():
        # Must not raise — every registered agent declares ≥1 non-Agent tool.
        rt = spec.to_runtime_spec("anthropic")
        assert rt.tools, agent_id


def _dummy_run_context(**overrides) -> RunContext:
    base = dict(
        project_id="prj_test",
        project_dir=Path("/tmp/prj_test"),
        runs_root=Path("/tmp"),
        dashboard=None,
        cost_ledger=None,
        llm_client=None,
        provider="anthropic",
        model="claude-opus-4-7",
    )
    base.update(overrides)
    return RunContext(**base)


def test_runcontext_autoloads_blocked_terms_from_env(monkeypatch):
    """RunContext picks up blocked_terms from OPENRESEARCH_BLOCKED_TERMS_JSON (subprocess seam)."""
    monkeypatch.setenv("OPENRESEARCH_BLOCKED_TERMS_JSON", json.dumps([_PAPER_REPO, "  ", "github.com/x/y"]))
    ctx = _dummy_run_context()
    # Whitespace-only entries are dropped.
    assert ctx.blocked_terms == (_PAPER_REPO, "github.com/x/y")


def test_runcontext_caller_blocked_terms_not_overridden_by_env(monkeypatch):
    """A caller-supplied blocked_terms wins over the env var (tests / explicit wiring)."""
    monkeypatch.setenv("OPENRESEARCH_BLOCKED_TERMS_JSON", json.dumps(["github.com/from/env"]))
    ctx = _dummy_run_context(blocked_terms=("github.com/from/caller",))
    assert ctx.blocked_terms == ("github.com/from/caller",)


def test_runcontext_no_env_means_empty_blocklist(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_BLOCKED_TERMS_JSON", raising=False)
    ctx = _dummy_run_context()
    assert ctx.blocked_terms == ()


def test_runcontext_malformed_env_never_crashes(monkeypatch):
    """A bad env value must never crash a run — it degrades to an empty blocklist."""
    monkeypatch.setenv("OPENRESEARCH_BLOCKED_TERMS_JSON", "{not valid json")
    ctx = _dummy_run_context()
    assert ctx.blocked_terms == ()
