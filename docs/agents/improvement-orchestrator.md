# `improvement-orchestrator`

> **Stage:** improvement (fan-out controller).
> **Registry entry:** [`AGENT_REGISTRY["improvement-orchestrator"]`](../../backend/agents/registry.py).
> **Implementation:** [`backend/agents/improvement.py`](../../backend/agents/improvement.py).

> **STUB**

## Purpose
Generate hypotheses, select adaptive batches (using `expected_value_score` + `category` diversification), fan out to `improvement-path` agents, then compose winners.

## Accepts
TODO — baseline `ExperimentArtifacts`, `RubricVerification.weak_points`, prior `ImprovementRound`s.

## Emits
TODO — [`ImprovementRound`](../../backend/agents/schemas.py), eventually a [`CompositionPhase`](../../backend/agents/schemas.py).

## Source
- [`backend/agents/improvement.py`](../../backend/agents/improvement.py)
- Models: `ImprovementHypothesis`, `ImprovementRound`, `CompositionAttempt`, `CompositionPhase`.
