# `environment-verifier`

> **Stage:** verifier (parallel gate).
> **Registry entry:** [`AGENT_REGISTRY["environment-verifier"]`](../../backend/agents/registry.py).
> **Implementation:** [`backend/agents/verification.py`](../../backend/agents/verification.py).

> **STUB**

## Purpose
Verify the synthesized environment (Dockerfile, pinned deps) actually builds, installs, and runs the baseline entry points.

## Accepts
TODO — `EnvironmentSpec`, build/run logs.

## Emits
TODO — `VerifierScore` inside a `VerificationReport`.

## Source
- [`backend/agents/verification.py`](../../backend/agents/verification.py)
