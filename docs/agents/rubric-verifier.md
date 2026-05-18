# `rubric-verifier`

> **Stage:** verifier (parallel gate, Track 3).
> **Registry entry:** [`AGENT_REGISTRY["rubric-verifier"]`](../../backend/agents/registry.py).
> **Implementation:** [`backend/agents/verification.py`](../../backend/agents/verification.py), [`backend/agents/rubric_source.py`](../../backend/agents/rubric_source.py).

> **STUB**

## Purpose
PaperBench-style weighted scoring across rubric areas. The LLM supplies per-area `score` + `weight`; `RubricVerification.from_areas` deterministically computes `overall_score` and `meets_target` (never trusted from the model).

## Accepts
TODO — rubric source (`paperbench_bundle` or `generated`), baseline + improved artifacts.

## Emits
TODO — [`RubricVerification`](../../backend/agents/schemas.py) containing a list of [`RubricAreaScore`](../../backend/agents/schemas.py). `weak_points` per area feeds the improvement orchestrator.

## Source
- Models: `RubricAreaScore`, `RubricVerification` in [`backend/agents/schemas.py`](../../backend/agents/schemas.py).
- Rubric loading: [`backend/agents/rubric_source.py`](../../backend/agents/rubric_source.py).
