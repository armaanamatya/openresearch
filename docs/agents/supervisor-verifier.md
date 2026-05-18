# `supervisor-verifier`

> **Stage:** verifier (gate aggregator).
> **Registry entry:** [`AGENT_REGISTRY["supervisor-verifier"]`](../../backend/agents/registry.py).
> **Implementation:** [`backend/agents/verification.py`](../../backend/agents/verification.py).

> **STUB**

## Purpose
Aggregate the parallel verifiers' `VerifierScore`s into a single gate decision (`gate_1`, `gate_2`, `gate_3`) with a `GateStatus`.

## Accepts
TODO — list of `VerifierScore` from upstream verifiers.

## Emits
TODO — [`VerificationReport`](../../backend/agents/schemas.py) + a simplified [`GateDecision`](../../backend/agents/schemas.py) for the orchestrator.

## Source
- [`backend/agents/verification.py`](../../backend/agents/verification.py)
- Models: `VerificationReport`, `GateDecision`, `GateStatus`.
