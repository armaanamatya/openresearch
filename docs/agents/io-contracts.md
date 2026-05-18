# Agent I/O Contracts

Canonical reference for what gets passed *into* every agent in the ReproLab
pipeline, what comes *out*, and which Pydantic models define the wire shapes.

This is a **single combined reference** for three audiences:

1. Devs **adding or modifying agents** — find the input/output model you must
   honor.
2. Devs **wiring agents together** (pipeline/orchestrator) — find which agent
   produces what the next one consumes.
3. Devs **consuming agent outputs externally** (frontend, SSE, dashboard) —
   find the JSON shape that crosses a process boundary.

> **Source of truth.** The Pydantic models in
> [`backend/agents/schemas.py`](../../backend/agents/schemas.py) and
> [`backend/schemas/*`](../../backend/schemas/) are authoritative. This doc
> mirrors them with field tables and examples; if there is a conflict, the
> Pydantic class wins and this file is stale — please fix it.

---

## Pipeline order

The orchestrator runs agents in stages. Builder agents are sequential;
verifiers run in parallel; improvement runs as a fan-out of `improvement-path`
agents under `improvement-orchestrator`.

```
paper-understanding
  └─> artifact-discovery
        └─> environment-detective
              └─> reproduction-planner
                    └─> baseline-implementation
                          └─> experiment-runner
                                ├─> method-fidelity-verifier  ┐
                                ├─> environment-verifier      │  parallel
                                ├─> data-metrics-verifier     │  gate
                                ├─> artifact-diff-verifier    │
                                ├─> rubric-verifier           │
                                └─> supervisor-verifier       ┘
                                      └─> improvement-orchestrator
                                            └─> N × improvement-path
                                                  └─> (loop until converged)
                                                        └─> report-generator
```

Concrete orchestration logic lives in
[`backend/agents/pipeline.py`](../../backend/agents/pipeline.py) and
[`backend/agents/orchestrator.py`](../../backend/agents/orchestrator.py).

---

## The universal envelope: `AgentOutput`

Every agent ultimately returns an `AgentOutput`. Its `structured_outputs` dict
holds the agent-specific Pydantic payload (e.g. `PaperClaimMap`,
`EnvironmentSpec`).

**Source.** [`backend/agents/schemas.py`](../../backend/agents/schemas.py)
(`class AgentOutput`).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `agent_id` | `str` | — | Registry key, e.g. `paper-understanding`. Must match an entry in [`AGENT_REGISTRY`](../../backend/agents/registry.py). |
| `status` | `str` | `"completed"` | One of `completed`, `failed`, `skipped`, `partial`. Resilience wrappers may set this. |
| `structured_outputs` | `dict[str, Any]` | `{}` | The agent's typed payload (e.g. `{"paper_claim_map": {...}}`). Keys are agent-specific; see each per-agent doc. |
| `summary` | `str` | `""` | Human-readable one-paragraph summary. Used in UI/logs, not gating. |
| `exploration_log` | `dict[str, Any]` | `{}` | Free-form trace data (model calls, tool invocations, intermediate findings). Not gating. |

**JSON example (real, redacted from `runs/prj_01a6.../`):**

```json
{
  "agent_id": "paper-understanding",
  "status": "completed",
  "structured_outputs": {
    "paper_claim_map": {
      "core_contribution": "PPO proposes a clipped surrogate objective ...",
      "claims": [ { "method": "PPO ...", "dataset": "MuJoCo", "metric": "...", "expected_result": "0.82" } ],
      "...": "..."
    }
  },
  "summary": "Extracted 4 claims, 3 datasets, 4 metrics from PPO paper.",
  "exploration_log": { "model": "claude-opus-4-7", "tool_calls": 7 }
}
```

---

## Shared primitives

These small types appear inside multiple agent payloads.

### `RiskLevel` (enum)

```python
class RiskLevel(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"
```

JSON: a bare string, one of `"low" | "medium" | "high" | "critical"`.

### `Ambiguity`

A detail missing or under-specified in the paper.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `assumption_id` | `str` | — | e.g. `"A001"`. Stable ID across the run. |
| `detail` | `str` | — | What is under-specified. |
| `chosen_value` | `str \| None` | `None` | The value the agent chose, if any. |
| `evidence` | `list[str]` | `[]` | Paper-section references backing the choice. |
| `risk` | `RiskLevel` | `medium` | Severity if the assumption is wrong. |

```json
{
  "assumption_id": "A003",
  "detail": "Paper does not specify random-seed protocol for Atari runs.",
  "chosen_value": "3 seeds, seeds 0/1/2",
  "evidence": ["Section 6.4"],
  "risk": "medium"
}
```

### `Assumption`

A concrete assumption logged in the assumption ledger (similar to `Ambiguity`,
but with `chosen_value` mandatory and a `verified_by` back-reference).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `assumption_id` | `str` | — | Stable ID. |
| `detail` | `str` | — | The assumption. |
| `chosen_value` | `str` | — | What was assumed. **Required.** |
| `evidence` | `list[str]` | `[]` | References. |
| `risk` | `RiskLevel` | `medium` | Severity. |
| `verified_by` | `str \| None` | `None` | `agent_id` (e.g. `"environment-verifier"`) that signed it off. |

### `MetricSpec`

| Field | Type | Default |
|---|---|---|
| `name` | `str` | — |
| `definition` | `str` | — |
| `target_value` | `str \| None` | `None` |
| `source_section` | `str \| None` | `None` |

### `DatasetRequirement`

| Field | Type | Default |
|---|---|---|
| `name` | `str` | — |
| `source` | `str` | `""` |
| `download_method` | `str` | `""` |
| `size_estimate` | `str` | `""` |
| `notes` | `str` | `""` |

### `TrainingRecipe`

| Field | Type | Default |
|---|---|---|
| `optimizer` | `str` | `""` |
| `learning_rate` | `str` | `""` |
| `batch_size` | `str` | `""` |
| `epochs_or_steps` | `str` | `""` |
| `scheduler` | `str` | `""` |
| `other_hparams` | `dict[str, Any]` | `{}` |

---

## Per-agent files

Each file uses the same structure: **Purpose → Accepts → Emits → Source**.
Failure-mode shapes (resilience envelopes, retry semantics) live centrally in
[`errors.md`](errors.md). Dashboard events fired during execution live in
[`events.md`](events.md).

### Builder agents (sequential)

- [paper-understanding](paper-understanding.md) — extract claims, datasets, metrics from the paper
- [artifact-discovery](artifact-discovery.md) — locate official code, weights, eval bundles
- [environment-detective](environment-detective.md) — synthesize Dockerfile + dependency spec
- [reproduction-planner](reproduction-planner.md) — define what counts as reproduction
- [baseline-implementation](baseline-implementation.md) — adapt or implement the baseline
- [experiment-runner](experiment-runner.md) — execute training/eval, collect artifacts

### Verifier agents (parallel gate)

- [method-fidelity-verifier](method-fidelity-verifier.md)
- [environment-verifier](environment-verifier.md)
- [data-metrics-verifier](data-metrics-verifier.md)
- [artifact-diff-verifier](artifact-diff-verifier.md)
- [rubric-verifier](rubric-verifier.md) — PaperBench-style weighted scoring
- [supervisor-verifier](supervisor-verifier.md) — gate decision aggregator

### Improvement agents

- [improvement-orchestrator](improvement-orchestrator.md) — hypothesis generation, batch selection, composition
- [improvement-path](improvement-path.md) — one parallel hypothesis path

---

## Non-agent infrastructure

Same template (Purpose / Accepts / Emits / Source), separate files because
these have non-trivial I/O contracts of their own.

- [orchestrator](orchestrator.md) — `ReproLabOrchestrator`, `PipelineState`, stage transitions
- [pipeline](pipeline.md) — `run_pipeline_sdk` / `run_pipeline_offline` entry points
- [registry](registry.md) — `AgentSpec` shape, `AGENT_REGISTRY` rules, `to_runtime_spec`
- [runtime](runtime.md) — `AgentRuntime`, `ProviderName`, `AgentRuntimeSpec`, `ToolSpec`
- [dashboard-emitter](dashboard-emitter.md) — events fired to the frontend
- [telemetry](telemetry.md) — observability hooks (cf. `backend/observability/`)
- [structured-output](structured-output.md) — parsing/validation helpers around Pydantic

---

## Cross-cutting

- [events.md](events.md) — every dashboard event type and its payload schema
- [errors.md](errors.md) — failure envelopes, resilience wrappers, retry semantics
- [resilience.md](resilience.md) — existing doc on retry/timeout policy

---

## Conventions

- **Filenames** are kebab-case and match the registered `agent_id` (so
  `paper-understanding.md` ↔ `agent_id="paper-understanding"`).
- **All examples** are either synthetic (clearly marked) or extracted from a
  real run under `runs/` and redacted. Real examples carry a `(real, redacted)`
  tag.
- **Encoding** for all on-disk JSON is UTF-8, written via
  `Path.write_text(..., encoding="utf-8")`. See
  [`docs/design/cross-platform-encoding-fix.md`](../design/cross-platform-encoding-fix.md).
- **Wire crossings.** When a payload leaves Python (SSE, REST, disk JSON), it
  is `model_dump_json()`-ed — enums become their `.value` strings,
  `None` becomes `null`, datetimes become ISO-8601.
