"""ReproLab Agent Layer — LLM-powered agents for paper reproduction."""

from backend.agents.registry import AGENT_REGISTRY, get_agent_definitions
from backend.agents.schemas import (
    AgentOutput,
    Ambiguity,
    Assumption,
    BaselineResult,
    DatasetRequirement,
    EnvironmentSpec,
    ExperimentArtifacts,
    GateDecision,
    GateStatus,
    ImprovementHypothesis,
    MetricSpec,
    PaperClaimMap,
    PathResult,
    ReproductionContract,
    ResearchMap,
    TrainingRecipe,
    VerificationReport,
)
from backend.hermes_audit import HermesAuditReport

__all__ = [
    "AGENT_REGISTRY",
    "get_agent_definitions",
    "AgentOutput",
    "Ambiguity",
    "Assumption",
    "BaselineResult",
    "DatasetRequirement",
    "EnvironmentSpec",
    "ExperimentArtifacts",
    "GateDecision",
    "GateStatus",
    "ImprovementHypothesis",
    "MetricSpec",
    "PaperClaimMap",
    "PathResult",
    "ReproductionContract",
    "ResearchMap",
    "TrainingRecipe",
    "VerificationReport",
    "HermesAuditReport",
]
