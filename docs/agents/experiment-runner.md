# `experiment-runner`

> **Stage:** builder (6 of 6).
> **Registry entry:** [`AGENT_REGISTRY["experiment-runner"]`](../../backend/agents/registry.py).
> **Implementation:** [`backend/agents/experiment_runner.py`](../../backend/agents/experiment_runner.py).

> **STUB**

## Purpose
Execute the baseline training/eval commands inside the Docker environment, collect metrics, plots, and logs.

## Accepts
TODO — `BaselineResult`, `EnvironmentSpec`.

## Emits
TODO — [`ExperimentArtifacts`](../../backend/agents/schemas.py) (`metrics`, `plots`, `log_path`, `commands_log_path`, `provenance_path`, `success`).

## Source
- [`backend/agents/experiment_runner.py`](../../backend/agents/experiment_runner.py)
- Model: `ExperimentArtifacts` in [`backend/agents/schemas.py`](../../backend/agents/schemas.py).
