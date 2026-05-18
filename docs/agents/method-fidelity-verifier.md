# `method-fidelity-verifier`

> **Stage:** verifier (parallel gate).
> **Registry entry:** [`AGENT_REGISTRY["method-fidelity-verifier"]`](../../backend/agents/registry.py).
> **Implementation:** [`backend/agents/verification.py`](../../backend/agents/verification.py).

> **STUB**

## Purpose
Score how faithfully the baseline implementation matches the paper's described method (architecture, loss, training procedure).

## Accepts
TODO — `BaselineResult`, `PaperClaimMap`, code path.

## Emits
TODO — [`VerifierScore`](../../backend/agents/schemas.py) inside a [`VerificationReport`](../../backend/agents/schemas.py).

## Source
- [`backend/agents/verification.py`](../../backend/agents/verification.py)
- Models: `VerifierScore`, `VerificationReport`, `GateStatus`.
