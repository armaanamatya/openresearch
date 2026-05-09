# Ingestion + Context Layer — Event-Sourced Design Spec

| Field | Value |
|---|---|
| Date | 2026-05-09 |
| Owner | lolout1 |
| Status | Approved for implementation |
| Issues | #3 (umbrella), #4 (umbrella), #12, #13, #14, #15, #16 |
| Cross-team deps (slot-in) | #8 (canonical schemas, rishi-golla), #9 (SQLite repository, rishi-golla) |
| Architectural approach | **Event Sourcing + CQRS** (Approach C — full audit-grade, no cuts) |

## 1. Why event sourcing for this system

ReproLab's entire product thesis is **auditability**: every claim, every decision, every reproduction step must be traceable. Event sourcing isn't a stylistic choice here — it's the *natural* representation of the domain:

| Domain need | Event sourcing answer |
|---|---|
| "What did the agent know at decision time?" | Replay events up to that point. |
| "Why did the verifier reject this?" | The rejection event is a first-class fact in the log. |
| "Reproduce the run from scratch on a different machine" | Replay the event stream. |
| "Show provenance for this citation" | Walk back along `causation_id` to the originating event. |
| "Add a new dashboard view of agent activity" | Add a new projection; rebuild from existing events. |
| "Detect a verification disagreement" | A process manager listens for divergent verifier events. |

Approach A (Protocol-bounded services) gives you swappability but the *truth* is held in mutable service state — you reconstruct history by reading logs after the fact. Event sourcing inverts this: the event log **is** the truth; service state is a derived projection. For ReproLab specifically, this is the difference between "we kept good logs" and "the log is the system."

## 2. Goals, Non-Goals, Constraints

### Goals
1. **Single source of truth** — the event store is canonical; all state derives from it.
2. **Replay-as-debugging** — any past run can be replayed exactly (modulo external API changes, which we record into the log via captured-response events).
3. **Time travel** — answer "what did agent X believe at time T" by replaying events for project P up to T.
4. **Multiple consumers, zero coupling** — dashboard, verifiers, supervisors, future analytics all read the same events through projections of their own.
5. **Schema evolution without rewrites** — every event has a `schema_version`; upcasters migrate old payloads forward at read time.
6. **Citation invariant enforced at the event boundary** — events that carry agent claims cannot be constructed without `citations: tuple[Citation, ...]` populated.
7. **No data loss on failure** — every command attempt produces an event (success or failure). Crashes mid-process are recoverable.
8. **Production observability** — structured logs, OpenTelemetry traces with `correlation_id` and `causation_id` chains, Prometheus metrics.
9. **`mypy --strict`** clean across the stack.

### Non-Goals
- Building the agent runtime (Docker sandbox / RuntimeBackend) — separate umbrella.
- Building the agent orchestrator + spawn policy — separate umbrella.
- Verifier team and improvement agents — they consume events from this layer, but live elsewhere.
- Frontend rendering — it subscribes to the event store, but rendering is its own spec.
- Six REPL tools beyond `lookup` and `semantic_search` — they plug in via the same `WorkspaceTool` Protocol but ship later.

### Constraints
- Python 3.11+, `mypy --strict`.
- All HTTP through `httpx` with retries + rate limiting + circuit breakers.
- All external API responses *captured* as events (so replays are deterministic even if upstream changes).
- Event store starts as embedded SQLite (WAL mode) — handles tens of millions of events comfortably; can graduate to EventStoreDB or Postgres later via the same Protocol.
- Long timeline → no hackathon-driven cuts.

## 3. Architectural Overview

### 3.1 The CQRS / event-sourcing shape

```
                           ┌───────────────────────────────┐
                           │        Event Store            │
                           │  (append-only, partitioned    │
                           │   by aggregate_id)            │
                           └───────────────────────────────┘
                                  ▲              │
                                  │ events       │ subscribe / replay
                                  │              ▼
   ┌──────────┐    cmd     ┌────────────┐    ┌─────────────────┐    read     ┌──────────┐
   │  Caller  │──────────► │ Aggregate  │    │   Projections   │ ◄─────────  │ Readers  │
   │ (CLI/API)│            │  (write    │    │  (read models)  │             │(agents,  │
   └──────────┘            │   side)    │    └─────────────────┘             │ dashboard│
                           └────────────┘            ▲                       │ verifiers│
                                                     │                       └──────────┘
                                  ┌──────────────────┴─────────────────────┐
                                  │            Process Managers            │
                                  │  (event-driven workflow coordinators)  │
                                  └────────────────────────────────────────┘
```

- **Commands** carry intent (e.g., `CreateProject`, `StartParsing`, `BuildWorkspace`).
- **Aggregates** apply commands → emit events.
- **Events** are facts; immutable; appended to the event store.
- **Projections** subscribe to events → maintain read models (indexed for query).
- **Process managers** coordinate workflows by reacting to events and issuing new commands.
- **Readers** (agents, dashboard, verifiers) consume projections; never read aggregates directly.

### 3.2 Aggregates in this layer

| Aggregate | aggregate_type | Lifecycle | Owns |
|---|---|---|---|
| `Project` | `project` | Creation → metadata complete → archived | Source identity, paper PDF reference, paper metadata |
| `ParsedPaper` | `parsed_paper` | Parse start → complete (or failed) | Parser identity, sections, references, figures, full text |
| `Discovery` | `discovery` | Start → adapters report → complete | Discovered artifacts per adapter |
| `Index` | `index` | Start → sources/chunks registered → complete | SourceRefs and Chunks for the project |
| `Workspace` | `workspace` | Created → variables loaded → tool calls → ready/closed | Per-agent variable bindings, tool call history |

Each aggregate has a content-hashed or ULID-based `aggregate_id`.

### 3.3 Projections (read models)

| Projection | Subscribes to | Materializes |
|---|---|---|
| `ProjectsProjection` | `Project*` events | Current project list, status, metadata |
| `ParsedPapersProjection` | `ParsedPaper*` | Latest parsed view per project, section index |
| `ArtifactsProjection` | `Discovery*` | Discovered artifacts per project, by kind |
| `SourcesProjection` | `Index*` | SourceRef + Chunk lookup tables |
| `SemanticIndexProjection` | `Index*` | Chroma embedding index keyed by ChunkId |
| `WorkspaceProjection` | `Workspace*` | Per-workspace variables, tool history |
| `CitationGraphProjection` | `Workspace*`, `Verification*` | Edges from claims → evidence chunks |
| `EventTimelineProjection` | * (firehose) | Per-project event timeline for the dashboard |

Projections are *eventually consistent*. A projection can be torn down and rebuilt from the event log at any time — this is the testing and migration superpower of event sourcing.

### 3.4 Process managers

| Process manager | Reacts to | Issues commands |
|---|---|---|
| `IngestionFlow` | `ProjectCreated` | `StartParsing`, `StartDiscovery` |
| `IndexingFlow` | `ParsingCompleted` + `DiscoveryCompleted` (joined) | `StartIndexing` |
| `WorkspaceReadyFlow` | `IndexingCompleted` | `BuildWorkspace(agent_name)` (per agent) |
| `RetryFlow` | `*Failed` events with `retryable=True` | Re-issue original command with backoff |
| `CapturedResponseFlow` | All `ExternalApiCalled` | Persist captured responses to enable deterministic replay |

Process managers hold their own state, also event-sourced (their state is itself an aggregate). This means a crashed process manager wakes up, replays its history, and resumes mid-workflow.

### 3.5 Module layout

```
openresearch/
├── eventstore/                          # The bedrock
│   ├── interface.py                     # EventStore Protocol
│   ├── sqlite_store.py                  # SQLite + WAL implementation
│   ├── jsonl_store.py                   # JSONL implementation (debug, ops)
│   ├── snapshot.py                      # Snapshot Protocol + storage
│   ├── upcaster.py                      # Schema-version upcasters
│   ├── subscription.py                  # Long-lived subscription primitive
│   └── replay.py                        # Bulk replay engine
├── messaging/
│   ├── command.py                       # Command base + dispatcher
│   ├── event.py                         # Event base + envelope (correlation/causation IDs)
│   └── bus.py                           # In-proc bus + NATS adapter (slot-in)
├── ingestion/                           # Umbrella #3
│   ├── intake/                          # Issue #12
│   │   ├── commands.py                  # CreateProject command + handler
│   │   ├── events.py                    # ProjectCreated, PaperFetched, MetadataExtracted, ...
│   │   ├── aggregate.py                 # ProjectAggregate
│   │   ├── adapters/                    # PDF, arXiv, DOI fetch adapters
│   │   └── projections.py               # ProjectsProjection
│   ├── parser/                          # Issue #13
│   │   ├── commands.py                  # StartParsing
│   │   ├── events.py                    # ParsingStarted, SectionExtracted, ReferenceExtracted, ParsingCompleted, ParsingFailed
│   │   ├── aggregate.py                 # ParsedPaperAggregate
│   │   ├── pymupdf_parser.py
│   │   ├── nougat_parser.py
│   │   └── projections.py               # ParsedPapersProjection
│   └── discovery/                       # Issue #14
│       ├── commands.py                  # StartDiscovery
│       ├── events.py                    # DiscoveryStarted, ArtifactFound, AdapterFailed, DiscoveryCompleted
│       ├── aggregate.py                 # DiscoveryAggregate
│       ├── adapters/                    # GitHub, PWC, HF, Semantic Scholar
│       └── projections.py               # ArtifactsProjection
├── context/                             # Umbrella #4
│   ├── indexer/                         # Issue #15
│   │   ├── commands.py                  # StartIndexing
│   │   ├── events.py                    # SourceRegistered, ChunkCreated, IndexingCompleted, IndexingFailed
│   │   ├── aggregate.py                 # IndexAggregate
│   │   ├── chunkers/
│   │   └── projections.py               # SourcesProjection, SemanticIndexProjection
│   └── workspace/                       # Issue #16
│       ├── commands.py                  # BuildWorkspace, AttachCitation, CallTool, EnrichVariable
│       ├── events.py                    # WorkspaceCreated, VariableLoaded, VariableEnriched, CitationAttached, ToolInvoked, WorkspaceReady, WorkspaceClosed
│       ├── aggregate.py                 # WorkspaceAggregate
│       ├── model.py                     # Cited[T], Citation, Provenance
│       ├── tools/
│       └── projections.py               # WorkspaceProjection, CitationGraphProjection
├── flows/
│   ├── ingestion_flow.py
│   ├── indexing_flow.py
│   ├── workspace_ready_flow.py
│   ├── retry_flow.py
│   └── captured_response_flow.py
└── shared/
    ├── ids.py
    ├── envelope.py                      # event envelope: correlation_id, causation_id, occurred_at, schema_version
    ├── errors.py
    ├── http.py                          # httpx + rate limit + circuit breaker
    ├── observability.py                 # structlog + OpenTelemetry
    └── config.py                        # pydantic-settings
```

## 4. The Event Store

### 4.1 Protocol

```python
class EventStore(Protocol):
    async def append(
        self,
        aggregate_id: AggregateId,
        aggregate_type: str,
        events: Sequence[DomainEvent],
        expected_version: int | None = None,   # optimistic concurrency
        correlation_id: CorrelationId | None = None,
        causation_id: EventId | None = None,
    ) -> AppendResult: ...

    async def load(
        self,
        aggregate_id: AggregateId,
        from_version: int = 0,
    ) -> AsyncIterator[StoredEvent]: ...

    async def load_global(
        self,
        from_position: int = 0,
        types: Iterable[str] | None = None,
    ) -> AsyncIterator[StoredEvent]: ...

    async def subscribe(
        self,
        from_position: int = 0,
        types: Iterable[str] | None = None,
    ) -> AsyncIterator[StoredEvent]: ...

    async def get_aggregate_version(self, aggregate_id: AggregateId) -> int: ...
```

### 4.2 SQLite-backed implementation (production default)

Schema:

```sql
CREATE TABLE events (
    global_position INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,                -- ULID, idempotency
    aggregate_id TEXT NOT NULL,
    aggregate_type TEXT NOT NULL,
    aggregate_version INTEGER NOT NULL,            -- per-aggregate sequence
    event_type TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL,                    -- the serialized event
    metadata_json TEXT NOT NULL,                   -- envelope: correlation/causation/occurred_at/source
    occurred_at TEXT NOT NULL,                     -- ISO8601 UTC
    UNIQUE (aggregate_id, aggregate_version)       -- enforces optimistic concurrency
);

CREATE INDEX idx_events_aggregate ON events(aggregate_id, aggregate_version);
CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_occurred_at ON events(occurred_at);
```

**Why SQLite:**
- Embedded, zero ops overhead.
- WAL mode + `synchronous=NORMAL` → ~10K writes/sec on commodity hardware.
- Single-writer model fits CQRS naturally (commands serialized through one writer per aggregate).
- One file per environment; trivial to copy, replay, version.
- Migrates cleanly to Postgres or EventStoreDB later through the same Protocol.

### 4.3 Snapshots

For aggregates that grow large (e.g., a `Workspace` with hundreds of `VariableEnriched` events), periodic snapshots keep replay cheap:

```python
@dataclass(frozen=True)
class Snapshot:
    aggregate_id: AggregateId
    aggregate_type: str
    aggregate_version: int
    payload_json: str
    schema_version: int
    taken_at: datetime

class SnapshotStore(Protocol):
    async def put(self, snapshot: Snapshot) -> None: ...
    async def get_latest(self, aggregate_id: AggregateId) -> Snapshot | None: ...
```

Policy: snapshot every 50 events for `Workspace`; every 200 for `Index`. Snapshots are themselves event-sourced (they're a derived projection over the canonical event log) — they're a cache, never canonical.

### 4.4 Upcasters

```python
class Upcaster(Protocol):
    @property
    def event_type(self) -> str: ...
    @property
    def from_version(self) -> int: ...
    @property
    def to_version(self) -> int: ...
    def upcast(self, payload: dict[str, Any]) -> dict[str, Any]: ...
```

When loading events, the store applies the chain `from_version → from_version+1 → ... → current` for any matching event type. Ensures old events stay readable forever.

### 4.5 Captured external responses

Replay determinism requires capturing every external API response into the event log:

```python
class ExternalApiCalled(DomainEvent):
    request_url: str
    request_method: str
    request_headers: dict[str, str]
    response_status: int
    response_body_sha256: str
    response_body_path: Path           # blob stored in `runs/{project_id}/blobs/{sha}.bin`
    duration_ms: int
```

In replay mode, HTTP calls short-circuit through the captured-response cache instead of hitting upstream. This is what makes runs *truly* reproducible.

## 5. Commands and Events — Module by Module

### 5.1 Intake (#12)

#### Commands

```python
@dataclass(frozen=True)
class CreateProject(Command):
    source: PaperSource

@dataclass(frozen=True)
class FetchPaperContent(Command):
    project_id: ProjectId
    source: PaperSource

@dataclass(frozen=True)
class ExtractMetadata(Command):
    project_id: ProjectId
```

#### Events

```python
class ProjectCreated(DomainEvent):
    schema_version = 1
    project_id: ProjectId
    source: PaperSource

class PaperFetched(DomainEvent):
    schema_version = 1
    project_id: ProjectId
    raw_paper_path: str
    pdf_sha256: str
    fetched_via: str    # adapter name

class PaperFetchFailed(DomainEvent):
    schema_version = 1
    project_id: ProjectId
    cause: str
    cause_type: str
    retryable: bool

class MetadataExtracted(DomainEvent):
    schema_version = 1
    project_id: ProjectId
    metadata: PaperMetadata
```

#### Aggregate

```python
class ProjectAggregate(Aggregate):
    aggregate_type = "project"
    project_id: ProjectId
    state: ProjectState     # CREATED | FETCHED | METADATA_EXTRACTED | FAILED

    def handle(self, cmd: Command) -> Sequence[DomainEvent]:
        match cmd:
            case CreateProject(): return [ProjectCreated(...)]
            case FetchPaperContent(): return self._handle_fetch(cmd)
            case ExtractMetadata(): return self._handle_extract(cmd)
            case _: raise UnsupportedCommand(cmd)

    def apply(self, ev: DomainEvent) -> None:
        match ev:
            case ProjectCreated(): self.state = ProjectState.CREATED
            case PaperFetched(): self.state = ProjectState.FETCHED
            case ...
```

### 5.2 Parser (#13)

#### Events

```python
class ParsingStarted(DomainEvent):
    project_id: ProjectId
    parser_name: str
    parser_version: str

class SectionExtracted(DomainEvent):
    project_id: ProjectId
    section: Section

class ReferenceExtracted(DomainEvent):
    project_id: ProjectId
    reference: Reference

class FigureExtracted(DomainEvent):
    project_id: ProjectId
    figure: Figure

class TableExtracted(DomainEvent):
    project_id: ProjectId
    table: Table

class ParsingCompleted(DomainEvent):
    project_id: ProjectId
    section_count: int
    reference_count: int
    parse_duration_ms: int

class ParsingFailed(DomainEvent):
    project_id: ProjectId
    parser_name: str
    cause: str
    cause_type: str
    retryable: bool
```

The parser emits one event per discovered section/reference/figure rather than dumping the whole `ParsedPaper` in a single event. This gives the dashboard fine-grained streaming and lets failed mid-parse runs preserve partial progress in the log.

### 5.3 Discovery (#14)

```python
class DiscoveryStarted(DomainEvent):
    project_id: ProjectId
    adapters: tuple[str, ...]

class AdapterStarted(DomainEvent):
    project_id: ProjectId
    adapter: str

class ArtifactFound(DomainEvent):
    project_id: ProjectId
    adapter: str
    artifact: DiscoveredArtifact

class AdapterFailed(DomainEvent):
    project_id: ProjectId
    adapter: str
    cause: str
    cause_type: str
    retryable: bool

class AdapterCompleted(DomainEvent):
    project_id: ProjectId
    adapter: str
    artifact_count: int
    duration_ms: int

class DiscoveryCompleted(DomainEvent):
    project_id: ProjectId
    adapter_summaries: tuple[AdapterSummary, ...]
```

Each adapter is a sub-flow: `AdapterStarted → many ArtifactFound → AdapterCompleted | AdapterFailed`.

### 5.4 Indexer (#15)

```python
class IndexingStarted(DomainEvent):
    project_id: ProjectId

class SourceRegistered(DomainEvent):
    project_id: ProjectId
    source: SourceRef

class ChunkCreated(DomainEvent):
    project_id: ProjectId
    chunk: Chunk

class ChunkEmbedded(DomainEvent):
    project_id: ProjectId
    chunk_id: ChunkId
    embedding_model: str
    embedding_dim: int
    embedding_vector_path: str   # stored as blob; we don't put float32[1536] in the event payload

class IndexingCompleted(DomainEvent):
    project_id: ProjectId
    source_count: int
    chunk_count: int
    embedding_count: int
    duration_ms: int

class IndexingFailed(DomainEvent):
    project_id: ProjectId
    cause: str
    retryable: bool
```

Embeddings stored as binary blobs referenced by hash; the event records the path. Same pattern as `ExternalApiCalled` for response bodies.

### 5.5 Workspace (#16)

```python
class WorkspaceCreated(DomainEvent):
    project_id: ProjectId
    workspace_id: WorkspaceId
    agent_name: str

class VariableLoaded(DomainEvent):
    workspace_id: WorkspaceId
    variable_name: str
    value_payload: dict        # serialized value
    citations: tuple[Citation, ...]
    source_agent: str | None

class VariableEnriched(DomainEvent):
    workspace_id: WorkspaceId
    variable_name: str
    value_payload: dict
    citations: tuple[Citation, ...]
    enriched_by: str           # agent that produced this variable

class CitationAttached(DomainEvent):
    workspace_id: WorkspaceId
    decision_id: str
    decision_payload: dict
    citations: tuple[Citation, ...]

class ToolInvoked(DomainEvent):
    workspace_id: WorkspaceId
    tool_name: str
    arguments: dict
    result_payload: dict
    citations: tuple[Citation, ...]
    duration_ms: int

class WorkspaceReady(DomainEvent):
    workspace_id: WorkspaceId
    variable_count: int

class WorkspaceClosed(DomainEvent):
    workspace_id: WorkspaceId
    reason: str
```

**Citation invariant at the event boundary:** every event payload that carries an agent claim has `citations: tuple[Citation, ...]` typed as non-empty (validated by Pydantic). An event constructor with empty citations raises before the event ever reaches the store.

#### Cited[T] still exists — as a derived view

```python
T = TypeVar("T", covariant=True)

@dataclass(frozen=True)
class Cited(Generic[T]):
    value: T
    citations: tuple[Citation, ...]

    def __post_init__(self) -> None:
        if not self.citations:
            raise CitationMissingError(...)

    @classmethod
    def from_event(cls, event: VariableLoaded | VariableEnriched | ToolInvoked) -> "Cited[Any]":
        return cls(value=event.value_payload, citations=event.citations)
```

The `WorkspaceProjection` materializes a `Workspace` view by replaying events into a structure where each variable is a `Cited[T]`. The agent reads from the projection; the projection is rebuildable from the event log; the event log itself enforces the invariant.

## 6. Projections

### 6.1 General projection model

```python
class Projection(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def subscribed_event_types(self) -> Iterable[str]: ...
    async def apply(self, event: StoredEvent) -> None: ...
    async def reset(self) -> None: ...
```

Every projection persists a checkpoint (last applied `global_position`) so it can resume after restart. Projections rebuild from scratch by `reset()` + replay.

### 6.2 Concrete projections

**`ProjectsProjection`** — table indexed by `project_id` with current status, source, metadata. Used by dashboard project list + by intake idempotency checks.

**`ParsedPapersProjection`** — sections, references, figures, full text. Indexed by `(project_id, section_id)`. Backed by SQLite tables.

**`ArtifactsProjection`** — discovered artifacts indexed by `(project_id, kind)`. Powers the "official repo" lookup that downstream agents need.

**`SourcesProjection`** — `SourceRef` and `Chunk` tables. Indexed by `(source_id)` and `(chunk_id)`. The lookup tool reads here.

**`SemanticIndexProjection`** — Chroma collection per project. Embeddings written here on `ChunkEmbedded` events. The semantic search tool reads here. Rebuildable from events alone (re-embed everything on rebuild).

**`WorkspaceProjection`** — per-workspace variable bindings, materialized as `Cited[T]`. Indexed by `(workspace_id, variable_name)`. Agent reads happen here.

**`CitationGraphProjection`** — graph edges from claim → evidence chunks; from agent decision → cited evidence. Powers the dashboard's "show me where this claim came from" view.

**`EventTimelineProjection`** — flat event timeline per project, indexed by time. Powers the dashboard's reasoning trace and replay UI.

### 6.3 Projection rebuild

Operationally critical: any projection can be rebuilt by `projection.reset() && replay(EventStore.load_global())`. This is how schema changes happen, how new projections are introduced, how corrupted read state is recovered.

## 7. Process Managers (Flows)

```python
class ProcessManager(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def subscribed_event_types(self) -> Iterable[str]: ...
    async def react(self, event: StoredEvent, dispatcher: CommandDispatcher) -> None: ...
```

Process managers persist their own state as event-sourced aggregates (`flow_state` aggregate type). On crash + restart, they replay their state and resume.

### `IngestionFlow`
```
on ProjectCreated      → dispatch(StartParsing(project_id)),
                         dispatch(StartDiscovery(project_id))
```

### `IndexingFlow`
Joins parsing + discovery completion before triggering indexing:
```
on ParsingCompleted    → if discovery already complete: dispatch(StartIndexing)
                         else: mark parsing-complete in flow state
on DiscoveryCompleted  → if parsing already complete: dispatch(StartIndexing)
                         else: mark discovery-complete in flow state
```

### `WorkspaceReadyFlow`
```
on IndexingCompleted   → dispatch(BuildWorkspace) for every agent registered to the project
```

### `RetryFlow`
```
on *Failed where retryable=True
                       → schedule retry with exponential backoff (3 attempts max)
                         dispatch the original command from the failed event's causation chain
```

### `CapturedResponseFlow`
Listens to all external API calls made via `shared/http.py`, persists the captured request/response into the event log so replays are deterministic.

## 8. Cross-Cutting Concerns

### 8.1 Event envelope

Every stored event carries:

```python
@dataclass(frozen=True)
class EventEnvelope:
    event_id: EventId               # ULID
    correlation_id: CorrelationId   # request-scoped, propagated through the whole flow
    causation_id: EventId | None    # the event that caused this one
    occurred_at: datetime           # UTC
    source: str                     # service/module that produced the event
    schema_version: int
```

Causation chains: `CreateProject` command → `ProjectCreated` event (causation=command id) → `IngestionFlow` reacts → `StartParsing` command (causation=`ProjectCreated`) → `ParsingStarted` event (causation=`StartParsing`) → ... You can walk back along `causation_id` to reconstruct *why* any event happened.

### 8.2 Schema evolution

Every event has `schema_version`. When a payload changes shape, register an upcaster. Old events stay valid forever. We never rewrite history.

### 8.3 Optimistic concurrency

Each `append()` includes `expected_version`. If the aggregate's current version doesn't match, the append fails — the caller reloads and retries. Prevents lost updates without distributed locks.

### 8.4 Idempotency

- `event_id` is unique-indexed in the store. Re-emitting the same event id is a no-op.
- Commands carry their own command_id; aggregate handlers de-dupe by command_id stored in their state.
- Content-addressed IDs (Source, Chunk, Artifact) make re-ingest idempotent.

### 8.5 Replay determinism

Three sources of non-determinism are captured into events:
1. **External HTTP responses** — every call writes `ExternalApiCalled` with a body blob reference; replay short-circuits to the captured body.
2. **Wall-clock time** — captured on every event as `occurred_at`. Aggregates accept a `Clock` dependency; in replay mode the clock returns event timestamps.
3. **Random IDs** — generated from a seeded PRNG keyed by `correlation_id` so replays produce identical IDs.

### 8.6 Error handling

Errors are first-class events. Every command handler that fails emits a `*Failed` event with:
- `cause: str`
- `cause_type: str`  (e.g. `"httpx.ConnectError"`)
- `retryable: bool` (handler decides based on cause_type)

`RetryFlow` picks up retryable failures.

### 8.7 Observability

- `structlog` — every log line includes `event_id`, `correlation_id`, `causation_id`, `aggregate_id`.
- OpenTelemetry — spans named `Cmd:<CommandType>` and `Evt:<EventType>`; trace context propagated via envelope.
- Prometheus metrics:
  - `openresearch_events_appended_total{aggregate_type,event_type}`
  - `openresearch_command_duration_seconds{command_type}`
  - `openresearch_projection_lag{projection}` (events behind firehose)
  - `openresearch_replay_duration_seconds{aggregate_type}`
  - `openresearch_failed_events_total{event_type,retryable}`

### 8.8 Configuration

`pydantic-settings`-based `Settings` model loaded once at startup. Environment vars only for secrets. Frozen instance injected via DI container.

### 8.9 Concurrency

- Asyncio everywhere (`anyio`-portable).
- Event store: SQLite single-writer, multi-reader. Aggregate handlers are serialized per-aggregate via `asyncio.Lock` keyed on `aggregate_id`.
- Adapters run concurrently within their service (parser ∥ discovery; discovery adapters ∥ each other).
- Projections run on independent subscriptions; one slow projection cannot block another.

### 8.10 Security

- API tokens via env vars only.
- No `eval` / `exec` / `pickle` in the data path.
- All HTTP TLS; no `verify=False`.
- PDF parsing bounded: 100MB / 500 pages / 60s.

## 9. Data Model — Plain Dataclasses

Event payloads are Pydantic v2 (validation, JSON serialization). Domain types crossing aggregate boundaries are frozen dataclasses.

```python
@dataclass(frozen=True)
class PaperMetadata: ...
@dataclass(frozen=True)
class Section: ...
@dataclass(frozen=True)
class Reference: ...
@dataclass(frozen=True)
class Figure: ...
@dataclass(frozen=True)
class Table: ...
@dataclass(frozen=True)
class DiscoveredArtifact: ...
@dataclass(frozen=True)
class SourceRef: ...
@dataclass(frozen=True)
class Chunk: ...
@dataclass(frozen=True)
class Citation: ...
@dataclass(frozen=True)
class Cited(Generic[T]): ...
```

All IDs are `NewType` strings, content-hashed where possible, ULID-based otherwise.

## 10. Testing Strategy

### 10.1 Test pyramid for event-sourced systems

| Level | What | How |
|---|---|---|
| Aggregate behavior | Given events → command → expected new events | Pure unit, no infrastructure. The bread-and-butter test. |
| Event payload schemas | Pydantic round-trip; upcaster correctness | Property tests (Hypothesis) + golden fixtures |
| Projection replay | Stream of events → expected projection state | Run projection in-memory against synthetic event stream |
| Command handler integration | Command → aggregate → events appended → projection caught up | In-memory event store, real handlers, real projections |
| Adapter contract | Real adapter against recorded HTTP fixture (VCR.py) | One cassette per adapter happy path + ≥1 failure mode |
| End-to-end | Submit `CreateProject(arxiv://1707.06347)` → `WorkspaceReady` event observed | In-process app with real SQLite store, recorded HTTP fixtures |
| Replay determinism | Two runs of same recorded session produce identical event streams | Golden test: snapshot one run, replay, diff |
| Smoke (live API) | Hits real arXiv / GitHub / PWC | Marked `smoke`, nightly only |

### 10.2 Aggregate test pattern

```python
def test_create_project_emits_project_created():
    aggregate = ProjectAggregate.empty()
    events = aggregate.handle(CreateProject(source=ArxivId(id="1707.06347")))
    assert [type(e) for e in events] == [ProjectCreated]
    assert events[0].source == ArxivId(id="1707.06347")

def test_project_cannot_be_created_twice():
    aggregate = ProjectAggregate.empty()
    aggregate.apply(ProjectCreated(...))
    with pytest.raises(ProjectAlreadyExists):
        aggregate.handle(CreateProject(...))
```

### 10.3 Replay determinism test

```python
async def test_replay_is_byte_identical():
    # First run, captures all external HTTP into the event log
    run1 = await app.execute(CreateProject(source=ArxivId(id="1707.06347")))
    snapshot1 = await app.event_store.dump()

    # Tear down store; second run replays in deterministic mode
    await app.reset()
    run2 = await app.replay(snapshot1)
    snapshot2 = await app.event_store.dump()

    assert snapshot1 == snapshot2
```

### 10.4 Invariants verified by tests

1. `Cited[T](value=x, citations=())` raises `CitationMissingError`.
2. Event payloads with empty citations fail Pydantic validation.
3. Re-ingesting same source produces identical SourceIds and ChunkIds.
4. One failing discovery adapter does not fail the project (it emits `AdapterFailed`).
5. Optimistic concurrency: concurrent `append()` to same aggregate with same `expected_version` — exactly one succeeds.
6. Projection rebuild from event log produces identical state to the live projection.
7. Replay is deterministic when external responses are captured.

## 11. Sequencing & Migration

### 11.1 Implementation order

| Order | Module / Issue | Notes |
|---|---|---|
| 1 | `eventstore/` + `messaging/` + `shared/` | Foundation — nothing ships without it |
| 2 | `flows/captured_response_flow.py` + `shared/http.py` | Determinism in place from day 1 |
| 3 | `ingestion/intake/` (#12) | First aggregate; first projection; smoke test passes |
| 4 | `ingestion/parser/` (#13) | PyMuPdfParser; Nougat slot exists, ships later |
| 5 | `ingestion/discovery/` (#14) | GitHub + PWC adapters; HF + S2 land later |
| 6 | `flows/ingestion_flow.py` + `flows/indexing_flow.py` | Workflow comes alive |
| 7 | `context/indexer/` (#15) | SectionChunker first; embeddings shipped via separate event |
| 8 | `context/workspace/` (#16) | Cited, Workspace, lookup + semantic_search tools |
| 9 | `flows/workspace_ready_flow.py` | End-to-end runs from one command to a ready workspace |
| 10 | `flows/retry_flow.py` | Resilience hardening pass |
| 11 | All projections | Some land alongside their aggregates; CitationGraph + EventTimeline land last |

### 11.2 Migration when #8 lands (canonical schemas)

Our event payload classes (in `*/events.py`) become aliases for canonical models from rishi-golla's #8 package. Upcasters handle any divergence. Estimated cost: half a day.

### 11.3 Migration when #9 lands (SQLite repository)

#9 provides a *generic* SQLite repository layer. Two paths:

1. Adopt #9's repository for projections (read models). Our `EventStore` keeps using its own SQLite schema (event sourcing requires very specific schema; not a fit for a generic repo).
2. Continue using our own SQLite for both event store and projections; adopt #9 only where it adds value (e.g., shared transactions across read models).

I recommend (1). Estimated cost: 1 day.

### 11.4 When the system grows past one node

The Protocol-bounded `EventStore` swaps in:
- **EventStoreDB** (purpose-built, recommended for strong-consistency multi-node).
- **Postgres** with `notify` for subscriptions (operationally familiar).
- **NATS JetStream** (cloud-native, partitioned).

No service code changes; only the binding.

## 12. Open Questions / Decisions Pending

| # | Question | Default |
|---|----------|---------|
| OQ1 | Embedding model for `SemanticSearchTool` | `all-MiniLM-L6-v2`; pluggable |
| OQ2 | Snapshot frequency for `Workspace` | every 50 events |
| OQ3 | Should `ChunkEmbedded` events live in the same store as everything else? | Yes — uniform store; embeddings as blob references, not inline |
| OQ4 | How are issue/discussion threads chunked? | One chunk per top-level body, one per comment |
| OQ5 | Event retention policy | Forever for now; revisit at 100M events |
| OQ6 | Do projections live in the same DB file? | Yes (separate SQLite tables); ops simplicity wins. Move to dedicated read-DB later if needed. |
| OQ7 | Process manager state — own aggregates or simple key-value? | Own event-sourced aggregates (consistency with rest of system) |

## 13. Out-of-Scope for This Spec

- Agent runtime / Docker sandbox — separate umbrella.
- Agent orchestrator + spawn policy — separate umbrella.
- Verifier team and improvement agents — they are **event consumers** of this layer, but live elsewhere.
- Frontend rendering — subscribes to event store + projections.
- Six REPL tools beyond `lookup` and `semantic_search` — Protocol slots ready; impls land later.
- Knowledge graph (Graphify) — Phase 2 per PRD.

## 14. Acceptance Criteria

A complete pass of this spec produces:

- [x] `python -m openresearch.cli arxiv://1707.06347` triggers the full pipeline; `WorkspaceReady` is observed for the default agent set.
- [x] Re-ingesting the same source emits exactly the same `SourceRegistered` and `ChunkCreated` events (idempotent).
- [x] A discovery adapter raising `RuntimeError` produces `AdapterFailed` and `DiscoveryCompleted`; `IndexingStarted` still fires.
- [x] `Cited[T]` cannot be constructed empty; `VariableEnriched` cannot be appended with empty citations.
- [x] Replaying the captured event log on a fresh store produces a byte-identical store.
- [x] Killing the orchestrator mid-flow and restarting causes flows to resume from `correlation_id` reconstruction.
- [x] All projections rebuild cleanly from `EventStore.load_global()`.
- [x] `mypy --strict` passes; tests green; smoke skipped by default.
- [x] OpenTelemetry traces show full causation chains end-to-end.
- [x] Prometheus metrics exposed at `/metrics`.

## 15. Appendix — File-by-File Summary

(Module → file → purpose, for handoff to implementation.)

```
eventstore/interface.py                 EventStore Protocol
eventstore/sqlite_store.py              SQLiteEventStore (production default)
eventstore/jsonl_store.py               JsonlEventStore (debug, ops dump)
eventstore/snapshot.py                  Snapshot Protocol + SqliteSnapshotStore
eventstore/upcaster.py                  Upcaster Protocol + registry
eventstore/subscription.py              long-lived subscription primitive
eventstore/replay.py                    bulk replay engine

messaging/command.py                    Command base + CommandDispatcher
messaging/event.py                      DomainEvent base + StoredEvent + EventEnvelope
messaging/bus.py                        InProcMessageBus, NatsMessageBus (slot-in)

ingestion/intake/commands.py            CreateProject, FetchPaperContent, ExtractMetadata
ingestion/intake/events.py              ProjectCreated, PaperFetched, MetadataExtracted, *Failed
ingestion/intake/aggregate.py           ProjectAggregate
ingestion/intake/adapters/{pdf,arxiv,doi}.py
ingestion/intake/projections.py         ProjectsProjection

ingestion/parser/commands.py            StartParsing
ingestion/parser/events.py              ParsingStarted, SectionExtracted, ..., ParsingCompleted, ParsingFailed
ingestion/parser/aggregate.py           ParsedPaperAggregate
ingestion/parser/{pymupdf,nougat}_parser.py
ingestion/parser/projections.py         ParsedPapersProjection

ingestion/discovery/commands.py         StartDiscovery
ingestion/discovery/events.py           DiscoveryStarted, AdapterStarted, ArtifactFound, AdapterFailed, AdapterCompleted, DiscoveryCompleted
ingestion/discovery/aggregate.py        DiscoveryAggregate
ingestion/discovery/adapters/{github,papers_with_code,huggingface,semantic_scholar}.py
ingestion/discovery/projections.py      ArtifactsProjection

context/indexer/commands.py             StartIndexing
context/indexer/events.py               IndexingStarted, SourceRegistered, ChunkCreated, ChunkEmbedded, IndexingCompleted, IndexingFailed
context/indexer/aggregate.py            IndexAggregate
context/indexer/chunkers/{section,paragraph,code_block}.py
context/indexer/projections.py          SourcesProjection, SemanticIndexProjection

context/workspace/commands.py           BuildWorkspace, AttachCitation, CallTool, EnrichVariable, CloseWorkspace
context/workspace/events.py             WorkspaceCreated, VariableLoaded, VariableEnriched, CitationAttached, ToolInvoked, WorkspaceReady, WorkspaceClosed
context/workspace/aggregate.py          WorkspaceAggregate
context/workspace/model.py              Cited[T], Citation, Provenance
context/workspace/tools/{interface,lookup,semantic_search}.py
context/workspace/projections.py        WorkspaceProjection, CitationGraphProjection

flows/ingestion_flow.py                 IngestionFlow
flows/indexing_flow.py                  IndexingFlow (joins parsing + discovery)
flows/workspace_ready_flow.py           WorkspaceReadyFlow
flows/retry_flow.py                     RetryFlow
flows/captured_response_flow.py         CapturedResponseFlow

shared/ids.py                           NewType ID generators
shared/envelope.py                      EventEnvelope, CorrelationId helpers
shared/errors.py                        OpenResearchError hierarchy
shared/http.py                          httpx + rate limit + circuit breaker + capture
shared/observability.py                 structlog + OpenTelemetry helpers
shared/config.py                        pydantic-settings Settings
```

---

**End of design.**
