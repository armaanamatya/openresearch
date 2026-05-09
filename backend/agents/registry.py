"""Agent registry — all 13 ReproLab agent definitions for the Claude Agent SDK.

Usage:
    from backend.agents.registry import get_agent_definitions
    opts = ClaudeAgentOptions(agents=get_agent_definitions())
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.agents.prompts import (
    ARTIFACT_DISCOVERY_PROMPT,
    ARTIFACT_DIFF_VERIFIER_PROMPT,
    BASELINE_IMPLEMENTATION_PROMPT,
    DATA_METRICS_VERIFIER_PROMPT,
    ENVIRONMENT_DETECTIVE_PROMPT,
    ENVIRONMENT_VERIFIER_PROMPT,
    EXPERIMENT_RUNNER_PROMPT,
    IMPROVEMENT_ORCHESTRATOR_PROMPT,
    IMPROVEMENT_PATH_PROMPT,
    METHOD_FIDELITY_VERIFIER_PROMPT,
    PAPER_UNDERSTANDING_PROMPT,
    REPRODUCTION_PLANNER_PROMPT,
    SUPERVISOR_VERIFIER_PROMPT,
)


@dataclass(frozen=True)
class AgentSpec:
    """Internal registry entry for a ReproLab agent."""

    agent_id: str
    role: str
    description: str
    prompt: str
    tools: list[str] = field(default_factory=list)
    spawn_permissions: bool = False
    max_turns: int | None = None


# ---------------------------------------------------------------------------
# The 13 PRD agents
# ---------------------------------------------------------------------------

AGENT_REGISTRY: dict[str, AgentSpec] = {
    # --- Builder agents (sequential pipeline) ---
    "paper-understanding": AgentSpec(
        agent_id="paper-understanding",
        role="builder",
        description="Extracts claims, datasets, metrics, and ambiguities from a research paper.",
        prompt=PAPER_UNDERSTANDING_PROMPT,
        tools=["Read", "Bash"],
    ),
    "artifact-discovery": AgentSpec(
        agent_id="artifact-discovery",
        role="builder",
        description="Finds repositories, datasets, and dependency clues for a paper.",
        prompt=ARTIFACT_DISCOVERY_PROMPT,
        tools=["WebSearch", "WebFetch", "Bash", "Read"],
    ),
    "environment-detective": AgentSpec(
        agent_id="environment-detective",
        role="builder",
        # TODO(#17 / runtime): RuntimeBackend is wired separately; this
        # description is the target PRD contract for the agent role.
        description="Infers and builds the Docker environment for a paper.",
        prompt=ENVIRONMENT_DETECTIVE_PROMPT,
        tools=["Read", "Write", "Bash", "WebSearch"],
    ),
    "reproduction-planner": AgentSpec(
        agent_id="reproduction-planner",
        role="builder",
        description="Creates the reproduction contract and execution plan.",
        prompt=REPRODUCTION_PLANNER_PROMPT,
        tools=["Read", "Write"],
    ),
    "baseline-implementation": AgentSpec(
        agent_id="baseline-implementation",
        role="builder",
        # TODO(#17 / runtime): RuntimeBackend is wired separately; this
        # description is the target PRD contract for the agent role.
        description="Implements or adapts the paper baseline inside a Docker sandbox.",
        prompt=BASELINE_IMPLEMENTATION_PROMPT,
        tools=["Read", "Write", "Edit", "Bash"],
    ),
    "experiment-runner": AgentSpec(
        agent_id="experiment-runner",
        role="builder",
        description="Executes experiments and captures all artifacts.",
        prompt=EXPERIMENT_RUNNER_PROMPT,
        tools=["Bash", "Read", "Write"],
    ),
    # --- Verifier agents ---
    "method-fidelity-verifier": AgentSpec(
        agent_id="method-fidelity-verifier",
        role="verifier",
        description="Verifies implementation matches the paper method.",
        prompt=METHOD_FIDELITY_VERIFIER_PROMPT,
        tools=["Read", "Bash"],
    ),
    "environment-verifier": AgentSpec(
        agent_id="environment-verifier",
        role="verifier",
        # TODO(#17 / runtime): RuntimeBackend is wired separately; this
        # description is the target PRD contract for the agent role.
        description="Verifies the Docker environment is reproducible.",
        prompt=ENVIRONMENT_VERIFIER_PROMPT,
        tools=["Read", "Bash"],
    ),
    "data-metrics-verifier": AgentSpec(
        agent_id="data-metrics-verifier",
        role="verifier",
        description="Verifies correct data usage and metric validity.",
        prompt=DATA_METRICS_VERIFIER_PROMPT,
        tools=["Read", "Bash"],
    ),
    "artifact-diff-verifier": AgentSpec(
        agent_id="artifact-diff-verifier",
        role="verifier",
        description="Verifies all required artifacts exist and prove the claim.",
        prompt=ARTIFACT_DIFF_VERIFIER_PROMPT,
        tools=["Read", "Bash"],
    ),
    "supervisor-verifier": AgentSpec(
        agent_id="supervisor-verifier",
        role="supervisor",
        description="Assigns verification tasks, resolves disagreements, decides final status, and generates the Research Map.",
        prompt=SUPERVISOR_VERIFIER_PROMPT,
        tools=["Read", "Agent"],
        spawn_permissions=True,
    ),
    # --- Improvement agents ---
    "improvement-orchestrator": AgentSpec(
        agent_id="improvement-orchestrator",
        role="improvement",
        description="Selects N improvement hypotheses from evidence and launches path agents.",
        prompt=IMPROVEMENT_ORCHESTRATOR_PROMPT,
        tools=["Read", "Bash", "Agent"],
        spawn_permissions=True,
    ),
    "improvement-path": AgentSpec(
        agent_id="improvement-path",
        role="improvement",
        # TODO(#17 / runtime): RuntimeBackend is wired separately; this
        # description is the target PRD contract for the agent role.
        description="Executes one improvement hypothesis in an isolated branch and sandbox.",
        prompt=IMPROVEMENT_PATH_PROMPT,
        tools=["Read", "Write", "Edit", "Bash"],
    ),
}


def get_agent_definitions() -> dict[str, dict]:
    """Convert registry to claude-agent-sdk AgentDefinition dicts.

    Returns a dict suitable for ``ClaudeAgentOptions(agents=...)``.
    """
    from claude_agent_sdk import AgentDefinition

    defs: dict[str, AgentDefinition] = {}
    for name, spec in AGENT_REGISTRY.items():
        defs[name] = AgentDefinition(
            description=spec.description,
            prompt=spec.prompt,
            tools=spec.tools if spec.tools else None,
            maxTurns=spec.max_turns,
        )
    return defs
