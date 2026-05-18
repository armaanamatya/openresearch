# Dashboard Events

> **STUB** — central catalog of every event the agent layer can emit to the dashboard / SSE channel.

> Emitter: [`dashboard-emitter.md`](dashboard-emitter.md). Schema home (Pydantic): [`backend/schemas/events.py`](../../backend/schemas/events.py) and [`backend/schemas/messages.py`](../../backend/schemas/messages.py).

## Event taxonomy (to be filled)

| Event type | Fired by | Payload model | Trigger |
|---|---|---|---|
| `agent.started` | every agent | TODO | when an agent begins execution |
| `agent.progress` | every agent | TODO | mid-run tool/model-call updates |
| `agent.completed` | every agent | TODO | clean exit, before envelope is returned |
| `agent.failed` | every agent | TODO | exception caught by resilience wrapper |
| `stage.transition` | orchestrator | TODO | `PipelineStage` change |
| `gate.decision` | supervisor-verifier | TODO | gate 1/2/3 pass/fail emitted |
| `improvement.round.started` | improvement-orchestrator | TODO | new round of parallel paths begins |
| `improvement.round.completed` | improvement-orchestrator | TODO | round closes with `best_path_id` |
| `composition.attempted` | improvement-orchestrator | TODO | one composition attempt finishes |
| `report.finalized` | report-generator | TODO | `FinalReport` written |

## Common envelope (to be filled)

Every event presumably carries `project_id`, `run_id`, `timestamp`, `event_type`, `payload`. Confirm against [`backend/schemas/events.py`](../../backend/schemas/events.py) and fill exact fields here.

## Wire format

JSON over SSE. UTF-8. ISO-8601 timestamps. Enums serialized as their string `.value`.
