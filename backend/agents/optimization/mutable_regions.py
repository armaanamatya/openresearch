"""Registry of GEPA-addressable mutable prompt regions.

Each region carries a stable ``component_id``, the canonical default text
(re-using the original source-module constants — no string duplication), and a
soft ``char_budget`` that the GEPA adapter enforces on candidate mutations.

Lane B v1 (4 components, per
``docs/superpowers/specs/2026-05-26-gepa-phase0-audit.md`` §0.3):
- ``improvement.orchestrator.body``
- ``improvement.pool_generation.body``
- ``improvement.rerank.body``
- ``improvement.orchestrator_round_n.body``

Lane A scaffolded (1 component, used by ``build_system_prompt``):
- ``root_system.decomposition_example``

Lane C scaffolded (1 component, used by ``baseline-implementation`` agent):
- ``baseline_agent.body``

Immutable sections (Algorithm-2 guard, FINAL_VAR contract, heartbeat, GPU
selection, chat steering, forced-iteration policy, security text) are NOT in
this registry and cannot be selected as GEPA mutation targets.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.agents.prompts import (
    BASELINE_IMPLEMENTATION_PROMPT,
    IMPROVEMENT_ORCHESTRATOR_PROMPT,
    IMPROVEMENT_PATH_PROMPT,
)
from backend.agents.prompts.improvement import (
    ADAPTIVE_POOL_GENERATION_PROMPT,
    ADAPTIVE_RERANK_PROMPT,
    IMPROVEMENT_ORCHESTRATOR_ROUND_N_PROMPT,
)


@dataclass(frozen=True)
class MutableRegion:
    """One GEPA-addressable prompt region."""

    component_id: str
    default_text: str
    description: str
    char_budget: int  # soft cap; mutations exceeding are rejected by the adapter


# Default char budget = 2× current length (gives GEPA room to expand without
# letting it 10× the prompt). Computed at module load.
def _budget(text: str) -> int:
    return max(2048, 2 * len(text))


# Late-bound to avoid an import cycle: ``system_prompt`` imports nothing from
# this module, but referencing ``_DECOMPOSITION_EXAMPLE`` directly here would
# pull the whole system_prompt module at import time. Use a sentinel that's
# resolved on first access.
def _decomposition_example_default() -> str:
    from backend.agents.rlm.system_prompt import _DECOMPOSITION_EXAMPLE

    return _DECOMPOSITION_EXAMPLE


REGIONS: dict[str, MutableRegion] = {
    # ----- Lane B (4 components, v1) ------------------------------------
    "improvement.orchestrator.body": MutableRegion(
        component_id="improvement.orchestrator.body",
        default_text=IMPROVEMENT_ORCHESTRATOR_PROMPT,
        description=(
            "Selects which improvement hypothesis to run. Load-bearing in both "
            "fixed-N and adaptive modes."
        ),
        char_budget=_budget(IMPROVEMENT_ORCHESTRATOR_PROMPT),
    ),
    "improvement.pool_generation.body": MutableRegion(
        component_id="improvement.pool_generation.body",
        default_text=ADAPTIVE_POOL_GENERATION_PROMPT,
        description=(
            "Generates the candidate pool for adaptive mode; produces the set "
            "the orchestrator + rerank operate on."
        ),
        char_budget=_budget(ADAPTIVE_POOL_GENERATION_PROMPT),
    ),
    "improvement.rerank.body": MutableRegion(
        component_id="improvement.rerank.body",
        default_text=ADAPTIVE_RERANK_PROMPT,
        description=(
            "Re-ranks remaining adaptive candidates after observing run "
            "results."
        ),
        char_budget=_budget(ADAPTIVE_RERANK_PROMPT),
    ),
    "improvement.orchestrator_round_n.body": MutableRegion(
        component_id="improvement.orchestrator_round_n.body",
        default_text=IMPROVEMENT_ORCHESTRATOR_ROUND_N_PROMPT,
        description=(
            "Multi-round orchestrator; selects hypotheses that build on prior "
            "rounds' winners."
        ),
        char_budget=_budget(IMPROVEMENT_ORCHESTRATOR_ROUND_N_PROMPT),
    ),
    # ----- Lane A scaffold (1 component) --------------------------------
    # Resolved on access to avoid an import cycle with system_prompt.
    # Phase 3 will switch ``default_text`` to a thunk.
    "root_system.decomposition_example": MutableRegion(
        component_id="root_system.decomposition_example",
        default_text="",  # populated by _initialize_lane_a() at first access
        description=(
            "In-context decomposition example shown to the RLM root. Primary "
            "Lane A optimization target."
        ),
        char_budget=8192,  # set after init below
    ),
    # ----- Lane C scaffold (1 component) --------------------------------
    "baseline_agent.body": MutableRegion(
        component_id="baseline_agent.body",
        default_text=BASELINE_IMPLEMENTATION_PROMPT,
        description=(
            "Sonnet sub-agent prompt that writes the baseline reproduction "
            "code. Lane C optimization target."
        ),
        char_budget=_budget(BASELINE_IMPLEMENTATION_PROMPT),
    ),
    # ----- Lane B v1.1 deferred (path agent) ----------------------------
    # Documented here for traceability; commented out until v1 ships.
    # "improvement.path.body": MutableRegion(
    #     component_id="improvement.path.body",
    #     default_text=IMPROVEMENT_PATH_PROMPT,
    #     description="Sub-agent that executes one improvement hypothesis.",
    #     char_budget=_budget(IMPROVEMENT_PATH_PROMPT),
    # ),
}


def _initialize_lane_a() -> None:
    """Populate ``root_system.decomposition_example`` default_text post-import."""
    cid = "root_system.decomposition_example"
    cur = REGIONS[cid]
    if cur.default_text:
        return
    text = _decomposition_example_default()
    REGIONS[cid] = MutableRegion(
        component_id=cid,
        default_text=text,
        description=cur.description,
        char_budget=_budget(text),
    )


def get_default(component_id: str) -> str:
    """Return the canonical default text for ``component_id``.

    Lazily initializes Lane A regions on first access to avoid import cycles.
    """
    if component_id == "root_system.decomposition_example":
        _initialize_lane_a()
    return REGIONS[component_id].default_text


# ---------------------------------------------------------------------------
# Agent ID → component ID mapping (used by collect_agent_text override seam)
# ---------------------------------------------------------------------------

AGENT_TO_COMPONENT: dict[str, str] = {
    "baseline-implementation": "baseline_agent.body",
    "improvement-orchestrator": "improvement.orchestrator.body",
    # The other 3 improvement components are NOT dispatched as their own
    # AGENT_REGISTRY entries — they're rendered inline by code in
    # backend/agents/hybrid/. The override seam for those goes through
    # build_prompt(...) callers reading PROMPT_OVERRIDES directly.
}


__all__ = [
    "AGENT_TO_COMPONENT",
    "MutableRegion",
    "REGIONS",
    "get_default",
]


# Suppress unused-import warning: IMPROVEMENT_PATH_PROMPT is referenced in the
# commented-out v1.1 region above and via the audit doc. Re-export for callers.
_ = IMPROVEMENT_PATH_PROMPT
