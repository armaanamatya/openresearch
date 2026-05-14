"""Agent system prompts for all ReproLab specialist agents."""

from backend.agents.prompts.paper_understanding import PAPER_UNDERSTANDING_PROMPT
from backend.agents.prompts.environment_detective import ENVIRONMENT_DETECTIVE_PROMPT
from backend.agents.prompts.reproduction_planner import REPRODUCTION_PLANNER_PROMPT
from backend.agents.prompts.baseline_implementation import BASELINE_IMPLEMENTATION_PROMPT
from backend.agents.prompts.experiment_runner import EXPERIMENT_RUNNER_PROMPT
from backend.agents.prompts.verifiers import (
    METHOD_FIDELITY_VERIFIER_PROMPT,
    ENVIRONMENT_VERIFIER_PROMPT,
    DATA_METRICS_VERIFIER_PROMPT,
    ARTIFACT_DIFF_VERIFIER_PROMPT,
    SUPERVISOR_VERIFIER_PROMPT,
)
from backend.agents.prompts.improvement import (
    IMPROVEMENT_ORCHESTRATOR_PROMPT,
    IMPROVEMENT_PATH_PROMPT,
)
from backend.agents.prompts.artifact_discovery import ARTIFACT_DISCOVERY_PROMPT
from backend.agents.prompts.rubric_verifier import RUBRIC_VERIFIER_PROMPT

__all__ = [
    "PAPER_UNDERSTANDING_PROMPT",
    "ARTIFACT_DISCOVERY_PROMPT",
    "ENVIRONMENT_DETECTIVE_PROMPT",
    "REPRODUCTION_PLANNER_PROMPT",
    "BASELINE_IMPLEMENTATION_PROMPT",
    "EXPERIMENT_RUNNER_PROMPT",
    "METHOD_FIDELITY_VERIFIER_PROMPT",
    "ENVIRONMENT_VERIFIER_PROMPT",
    "DATA_METRICS_VERIFIER_PROMPT",
    "ARTIFACT_DIFF_VERIFIER_PROMPT",
    "SUPERVISOR_VERIFIER_PROMPT",
    "IMPROVEMENT_ORCHESTRATOR_PROMPT",
    "IMPROVEMENT_PATH_PROMPT",
    "RUBRIC_VERIFIER_PROMPT",
]
