"""Phase 1 regression tests for the GEPA prompt-override seam.

Pinned invariants:

1. With no active override, every prompt resolves byte-identically to the
   pre-GEPA module constants (production safety).
2. With an active override, AgentSpec.to_runtime_spec uses it; the source
   module global IS NOT mutated.
3. PromptOverrideContext.use is properly nested + leaves no global state.
4. Immutable build_system_prompt regions cannot be overridden (any attempt is
   silently ignored).
5. AGENT_TO_COMPONENT keys all correspond to real AGENT_REGISTRY entries +
   real REGIONS entries.
"""

from __future__ import annotations

import pytest

from backend.agents.optimization import (
    AGENT_TO_COMPONENT,
    PROMPT_OVERRIDES,
    REGIONS,
)
from backend.agents.optimization.mutable_regions import get_default
from backend.agents.prompts import (
    BASELINE_IMPLEMENTATION_PROMPT,
    IMPROVEMENT_ORCHESTRATOR_PROMPT,
)
from backend.agents.prompts.improvement import (
    ADAPTIVE_POOL_GENERATION_PROMPT,
    ADAPTIVE_RERANK_PROMPT,
    IMPROVEMENT_ORCHESTRATOR_ROUND_N_PROMPT,
)
from backend.agents.registry import AGENT_REGISTRY


# ---------------------------------------------------------------------------
# Invariant 1: registry default_text matches source module constants byte-for-byte
# ---------------------------------------------------------------------------


def test_lane_b_regions_match_source_constants() -> None:
    assert REGIONS["improvement.orchestrator.body"].default_text is IMPROVEMENT_ORCHESTRATOR_PROMPT
    assert REGIONS["improvement.pool_generation.body"].default_text is ADAPTIVE_POOL_GENERATION_PROMPT
    assert REGIONS["improvement.rerank.body"].default_text is ADAPTIVE_RERANK_PROMPT
    assert REGIONS["improvement.orchestrator_round_n.body"].default_text is IMPROVEMENT_ORCHESTRATOR_ROUND_N_PROMPT


def test_lane_c_region_matches_source_constant() -> None:
    assert REGIONS["baseline_agent.body"].default_text is BASELINE_IMPLEMENTATION_PROMPT


def test_lane_a_region_lazily_initialized() -> None:
    from backend.agents.rlm.system_prompt import _DECOMPOSITION_EXAMPLE

    # First access triggers the lazy init that resolves the system_prompt import cycle.
    text = get_default("root_system.decomposition_example")
    assert text == _DECOMPOSITION_EXAMPLE
    assert REGIONS["root_system.decomposition_example"].char_budget >= 2048


# ---------------------------------------------------------------------------
# Invariant 2: to_runtime_spec uses override; module global unaffected
# ---------------------------------------------------------------------------


def test_to_runtime_spec_no_override_uses_default() -> None:
    spec = AGENT_REGISTRY["baseline-implementation"].to_runtime_spec("anthropic")
    assert spec.instructions is BASELINE_IMPLEMENTATION_PROMPT


def test_to_runtime_spec_uses_explicit_override() -> None:
    spec = AGENT_REGISTRY["baseline-implementation"].to_runtime_spec(
        "anthropic", prompt_override="DIFFERENT TEXT"
    )
    assert spec.instructions == "DIFFERENT TEXT"
    # Source module global must remain untouched.
    assert BASELINE_IMPLEMENTATION_PROMPT.startswith("You are")  # sanity check


# ---------------------------------------------------------------------------
# Invariant 3: PromptOverrideContext nesting + cleanup
# ---------------------------------------------------------------------------


def test_override_context_empty_by_default() -> None:
    assert PROMPT_OVERRIDES.current() == {}
    assert PROMPT_OVERRIDES.get("improvement.orchestrator.body") is None
    assert not PROMPT_OVERRIDES.has_active_override()


def test_override_context_use_pushes_and_pops() -> None:
    with PROMPT_OVERRIDES.use({"improvement.orchestrator.body": "X"}):
        assert PROMPT_OVERRIDES.get("improvement.orchestrator.body") == "X"
        assert PROMPT_OVERRIDES.has_active_override()
    assert PROMPT_OVERRIDES.get("improvement.orchestrator.body") is None
    assert not PROMPT_OVERRIDES.has_active_override()


def test_override_context_nested_top_wins() -> None:
    with PROMPT_OVERRIDES.use({"a": "outer", "b": "outer-b"}):
        with PROMPT_OVERRIDES.use({"a": "inner"}):
            assert PROMPT_OVERRIDES.get("a") == "inner"
            assert PROMPT_OVERRIDES.get("b") == "outer-b"
        assert PROMPT_OVERRIDES.get("a") == "outer"
    assert PROMPT_OVERRIDES.get("a") is None


def test_override_context_cleans_up_on_exception() -> None:
    with pytest.raises(RuntimeError):
        with PROMPT_OVERRIDES.use({"a": "x"}):
            raise RuntimeError("boom")
    assert PROMPT_OVERRIDES.current() == {}


# ---------------------------------------------------------------------------
# Invariant 4: build_system_prompt — zero override = byte-identical;
# immutable regions silently ignored.
# ---------------------------------------------------------------------------


def _minimal_root_model():
    from backend.agents.rlm.models import RootModel

    return RootModel(
        key="test",
        rlm_backend="openai",
        prompt_addendum="",
    )


def test_build_system_prompt_no_override_byte_identical() -> None:
    from backend.agents.rlm.system_prompt import build_system_prompt

    ctx_meta = {"paper_text": {"type": "str", "length": 1000}}
    rm = _minimal_root_model()
    baseline = build_system_prompt(context_metadata=ctx_meta, root_model=rm)
    with_empty = build_system_prompt(
        context_metadata=ctx_meta, root_model=rm, overrides={}
    )
    with_none = build_system_prompt(
        context_metadata=ctx_meta, root_model=rm, overrides=None
    )
    assert baseline == with_empty == with_none


def test_build_system_prompt_mutable_override_applied() -> None:
    from backend.agents.rlm.system_prompt import build_system_prompt

    ctx_meta = {"paper_text": {"type": "str", "length": 1000}}
    rm = _minimal_root_model()
    baseline = build_system_prompt(context_metadata=ctx_meta, root_model=rm)
    sentinel = "GEPA-TEST-SENTINEL-DECOMPOSITION-XYZ"
    mutated = build_system_prompt(
        context_metadata=ctx_meta,
        root_model=rm,
        overrides={"root_system.decomposition_example": sentinel},
    )
    assert sentinel in mutated
    assert sentinel not in baseline


def test_build_system_prompt_immutable_override_silently_ignored() -> None:
    from backend.agents.rlm.system_prompt import build_system_prompt

    ctx_meta = {"paper_text": {"type": "str", "length": 1000}}
    rm = _minimal_root_model()
    baseline = build_system_prompt(context_metadata=ctx_meta, root_model=rm)
    # Try to override a non-existent / immutable component id.
    same = build_system_prompt(
        context_metadata=ctx_meta,
        root_model=rm,
        overrides={
            "root_system.final_var_contract": "I AM EVIL",
            "rlm.operating_model": "ALSO EVIL",
            "heartbeat.section": "STILL EVIL",
        },
    )
    assert same == baseline
    assert "I AM EVIL" not in same


# ---------------------------------------------------------------------------
# Invariant 5: AGENT_TO_COMPONENT consistency
# ---------------------------------------------------------------------------


def test_agent_to_component_maps_real_agents_and_regions() -> None:
    for agent_id, component_id in AGENT_TO_COMPONENT.items():
        assert agent_id in AGENT_REGISTRY, f"unknown agent_id: {agent_id}"
        assert component_id in REGIONS, f"unknown component_id: {component_id}"
