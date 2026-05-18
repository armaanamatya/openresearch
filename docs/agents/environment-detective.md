# `environment-detective`

> **Stage:** builder (3 of 6).
> **Registry entry:** [`AGENT_REGISTRY["environment-detective"]`](../../backend/agents/registry.py).
> **Implementation:** [`backend/agents/environment_detective.py`](../../backend/agents/environment_detective.py).

> **STUB**

## Purpose
Synthesize a Dockerfile + dependency spec (`EnvironmentSpec`) from the paper's hardware clues and any discovered artifacts.

## Accepts
TODO — `PaperClaimMap` + `artifacts.json`.

## Emits
TODO — [`EnvironmentSpec`](../../backend/agents/schemas.py), writes `environment_spec.json` + `Dockerfile`.

## Source
- [`backend/agents/environment_detective.py`](../../backend/agents/environment_detective.py)
- Model: `EnvironmentSpec` in [`backend/agents/schemas.py`](../../backend/agents/schemas.py).
