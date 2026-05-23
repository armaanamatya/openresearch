# Frontend Integration — Backend Contract

How the frontend talks to the OpenResearch backend. The backend is the source
of truth; the UI is a **pure renderer** of run state plus a streamed event log.
This is the contract the Phase 4 frontend redesign (issue #61) builds against.

## Architecture

```
Browser  →  Next.js proxy (/api/demo/*)  →  FastAPI backend (:8000)  →  run subprocess
```

The browser never talks to the backend directly — every call goes through a
server-side Next.js route, so there is no CORS layer.

Each reproduction run is a **long-lived subprocess**. The backend HTTP layer is
**stateless**: it spawns subprocesses and reads their files. All run state lives
on disk under `runs/<project_id>/`.

## Run lifecycle

1. `POST /runs` (or `/runs/upload`, `/runs/arxiv`) — the backend spawns the run
   subprocess and returns `202` with the initial state.
2. The UI opens the SSE stream `GET /runs/<id>/events` and renders events live.
3. On the `run_complete` event, the UI reads `GET /runs/<id>/final-report`.
4. A reconnecting client re-opens the SSE stream — the event log is replayed
   from disk, so no events are lost across a refresh or a dropped connection.

## HTTP API

| Route | Purpose |
|---|---|
| `POST /runs` · `/runs/upload` · `/runs/arxiv` | start a run (`202`) |
| `POST /runs/{id}/resume` | resume a checkpointed run |
| `GET /runs` · `/runs/latest` | list runs / most-recent run |
| `GET /runs/{id}` | run state snapshot |
| `GET /runs/{id}/events` | **SSE event stream** (`text/event-stream`) |
| `GET /runs/{id}/final-report` | the final benchmark report |
| `GET /runs/{id}/source-pdf` | the source paper PDF |
| `DELETE /runs/{id}` | delete a run |
| `GET /health` · `/models` | health check, model registry |

## SSE event stream

`GET /runs/<id>/events` streams the run's append-only event log
(`dashboard_events.jsonl`). Every event is a JSON object carrying `event` and
`timestamp`. RLM-mode event types:

| `event` | Key fields | Emitted when |
|---|---|---|
| `repl_iteration` | `iteration`, `response` (≤4 KB), `code_blocks[]`, `sub_calls`, `timing` | each root-loop iteration completes |
| `primitive_call` | `primitive`, `status` (`start`/`ok`/`error`), `args_summary`, `result_summary`, `iteration` | a primitive starts / finishes |
| `sub_rlm_spawned` | `depth`, `model`, `prompt_preview` (≤200 ch) | a recursive sub-call starts |
| `sub_rlm_complete` | `depth`, `model`, `duration_ms`, `error` | a sub-call finishes |
| `run_complete` | `status`, `iterations`, `rubric_score`, `cost_usd`, `final_report_path` | the run ends |
| `candidate_proposed` | `iteration`, `round`, `candidate` (`id`, `title`, `category`, `description`, `reasoning`), `parent_id?` | one per hypothesis returned by a successful `propose_improvements` call |
| `candidate_outcome` | `iteration`, `candidate_id`, `outcome`, `rubric_delta` | root calls `record_candidate_outcome` after evaluating a candidate |
| `rubric_score` | `iteration`, `score`, `target`, `areas[]` (`area`, `score`, `weight`, `status`) | after a successful `verify_against_rubric` call (not emitted on fail-soft failure) |

`code_blocks[]` element shape:
`{code, stdout_meta:{length,prefix,has_traceback}, stderr_meta:{…}, vars:{name:{type,size}}, sub_calls}`.

A `primitive_call` with `status:"error"` marks a failed primitive — including
*fail-soft* failures where the primitive returned a failure-shaped result
instead of raising. Render these as failures.

### Corpus-free invariant — read this

The frontend **never receives the paper corpus**. `sse_bridge.sanitize_iteration`
is the single egress chokepoint: it strips REPL `locals`, bounds `response` to
4 KB, and reduces stdout/stderr to ≤200-char metadata prefixes. `vars` carries
variable **shapes** (`{type, size}`), never values. Build the UI around
summaries and shapes — full values are not, and will not be, on the wire.

## Run artifacts — `runs/<project_id>/`

| File | Contents |
|---|---|
| `dashboard_events.jsonl` | append-only event log — the SSE source |
| `final_report.{json,md}` | benchmark report — verdict, rubric, cost |
| `cost_ledger.jsonl` | authoritative primitive-call + cost ledger |
| `experiment_runs.jsonl` | every `run_experiment` result (logs, success, metrics) |
| `demo_status.json` | UI status snapshot (atomic write) |
| `rlm_state/` | per-iteration checkpoints (resume-safe) |
| `code/` | the reproduced project |

## Design properties

- **Stateless HTTP layer** — runs are subprocesses; the API only reads files.
  No shared in-memory run state; runs scale independently.
- **Append-only event log** — events are never mutated or dropped; a client can
  reconnect and replay the full history.
- **Atomic writes** — status and report files are written via temp-file +
  `os.replace`, so a reader never observes a partial file.
- **Single egress chokepoint** — corpus-leak protection lives in exactly one
  function; new events cannot widen the surface by accident.
- **Provider-agnostic** — model and runtime resolution sit behind one seam.

## Phase 4 (#61) notes

- The UI can be built **today** on `repl_iteration` + `primitive_call` +
  `run_complete`: the exploration tree, the REPL-state panel
  (`code_blocks[].vars`), the live iteration view (`response`, `code`), and the
  primitive-call history all derive from these.
- `candidate_proposed`, `candidate_outcome`, and `rubric_score` are now
  **dedicated emitted events** — the backend pushes them directly rather than
  leaving the UI to derive them from `primitive_call`. `root_reasoning` remains
  derivable (≈ `repl_iteration.response`); `variable_update` is folded into
  `repl_iteration` (`code_blocks[].vars`).
- Keep the renderer **pure** — never recompute run logic client-side. The
  backend owns verdicts, scores, and state.

## Designing the backend with the frontend in mind

When adding backend behavior, preserve these so the frontend stays simple:

1. **Emit a typed event** for anything the UI must show live — never make the
   UI poll for run progress.
2. **Route every new event** through `make_emit` / `sanitize_iteration` — never
   widen the egress surface.
3. **Keep events additive and value-free** — add optional fields, never rename
   or remove them; stream shapes and summaries, never raw corpus.
4. **Persist any run output** to `runs/<id>/` so a late or reconnecting client
   can still read it.
