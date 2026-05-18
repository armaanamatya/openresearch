# `report-generator`

> **Stage:** final synthesis (post-improvement).
> **Implementation:** [`backend/agents/report_generator.py`](../../backend/agents/report_generator.py).

> **STUB** — note: not a separately-registered `agent_id`; invoked by the orchestrator after all paths complete.

## Purpose
Produce the run's final report: paper-vs-reproduction deltas, per-path summaries, rubric scoring, statistical notes, overall verdict.

## Accepts
TODO — `PaperClaimMap`, baseline `ExperimentArtifacts`, all `PathResult`s, optional `RubricVerification` (Track 3), optional published PaperBench baseline.

## Emits
TODO — [`FinalReport`](../../backend/agents/schemas.py) (including `MetricDelta`s, `RubricArea`s, `PathSummary`s, and Track 3 verifier deltas).

## Source
- [`backend/agents/report_generator.py`](../../backend/agents/report_generator.py)
- Models: `FinalReport`, `MetricDelta`, `RubricArea`, `PathSummary` in [`backend/agents/schemas.py`](../../backend/agents/schemas.py).
