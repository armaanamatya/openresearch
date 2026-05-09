# Ingestion + Context Layer — Event-Sourced Design Spec (Revision 2)

| Field | Value |
|---|---|
| Date | 2026-05-09 |
| Owner | lolout1 |
| Status | Approved for implementation (post-Codex review) |
| Issues | #3 (umbrella), #4 (umbrella), #12, #13, #14, #15, #16 |
| Cross-team deps (slot-in) | #8 (canonical schemas, rishi-golla), #9 (SQLite repository, rishi-golla) |
| Architectural approach | **Event Sourcing + CQRS** with durable inbox/outbox coordinators |
| Revision history | r1 2026-05-09: initial event-sourced design. r2 2026-05-09: incorporates Codex principal-engineer review (aggregate purity, durable inbox/outbox, capture completeness, security hardening, ID composition, scope policy, acceptance semantics). |

## 1. Why event sourcing for this system

ReproLab's product thesis is **auditability**: every claim, every decision, every reproduction step must be traceable. Event sourcing is the natural representation of that domain.

| Domain need | Event sourcing answer |
|---|---|
| "What did the agent know at decision time?" | Replay events up to that point. |
| "Why did the verifier reject this?" | The rejection event is a first-class fact in the log. |
| "Reproduce the run on a different machine" | Replay the captured event stream — including LLM/embedding/HTTP responses. |
| "Show provenance for this citation" | Walk back along `causation_id` to the originating event. |
| "Add a new dashboard view" | Add a projection; rebuild from the existing event log. |
| "Detect a verifier disagreement" | A coordinator listens for divergent verifier events. |

Approach A (Protocol-bounded services) gives swappability but truth lives in mutable service state. Approach B couples too tightly. **Event sourcing inverts: the event log *is* the truth; service state is a derived projection.** That is the architectural keystone.

## 2. Goals, Non-Goals, Constraints

### Goals
1. **Single source of truth** — the event store is canonical; all state derives from it.
2. **Replay-as-debugging** — any past run can be replayed deterministically (HTTP, LLM, embeddings, web search, NotebookLM, Chroma queries — *all* captured).
3. **Time travel** — answer "what did agent X believe at time T" by replaying for project P up to T.
4. **Multiple consumers, zero coupling** — dashboard, verifiers, supervisors, future analytics each read their own projections of the same event log.
5. **Schema evolution without rewrites** — every event has `schema_version`; an upcaster registry migrates payloads forward at read time, including snapshots.
6. **Citation invariant defended in depth** — Pydantic validation, store-side re-validation, and `Cited[T]` materialization each enforce non-emptiness independently.
7. **No data loss on failure** — every command attempt produces an event (success or failure). Crashes mid-flow are recoverable through durable inbox/outbox tables, not just trace metadata.
8. **Aggregate purity** — aggregates validate state transitions only. IO is the responsibility of application services and adapters that append events afterward.
9. **Production observability** — structured logs, OpenTelemetry traces with `correlation_id` and `causation_id` chains, Prometheus metrics including outbox/dead-letter/capture metrics.
10. **Security at the data ingress** — PDF parsing isolated; embedding/parser model digests pinned; captured headers redacted; no implicit network-fetched code.
11. **`mypy --strict`** clean across the stack.

### Non-Goals
- Building the agent runtime (Docker sandbox / RuntimeBackend) — separate umbrella.
- Building the agent orchestrator + spawn policy — separate umbrella.
- Verifier team and improvement agents — they consume events from this layer; their internals live elsewhere.
- Frontend rendering — subscribes to projections and event firehose; not built here.
- Six REPL tools beyond `lookup` and `semantic_search` — Protocol slots ready (`graph_query`, `web_search`, `notebook_query`, `rlm_query`); implementations land later.
- Knowledge graph (Graphify) — Phase 2 per PRD.

### Constraints
- Python 3.11+, `mypy --strict`.
- All HTTP through `shared/http.py` with retries + rate limiting + circuit breakers + **synchronous capture into the event log**.
- All non-deterministic external surfaces (HTTP, LLM, embeddings, web search, NotebookLM, Chroma) flow through capture-aware client wrappers; all responses recorded for deterministic replay.
- Event store starts as embedded SQLite (WAL mode); explicit migration triggers (§4.5) move it to EventStoreDB or Postgres when those triggers fire.
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
                                  │ append       │ subscribe / replay
                                  │              ▼
   ┌──────────┐    cmd     ┌────────────────┐  ┌─────────────────┐    read     ┌──────────┐
   │  Caller  │──────────► │ Application    │  │   Projections   │ ◄─────────  │ Readers  │
   │ (CLI/API)│            │ Service +      │  └─────────────────┘             │(agents,  │
   └──────────┘            │ Aggregate      │           ▲                      │ dashboard│
                           │ (write side)   │           │                      │ verifiers│
                           └────────────────┘           │                      └──────────┘
                                  ▲                     │
                                  │                     │
                           ┌──────┴─────────────────────┴─────────┐
                           │   Coordinators (durable inbox +      │
                           │   transactional outbox)              │
                           └──────────────────────────────────────┘
```

- **Commands** carry intent (e.g., `CreateProject`, `RequestParse`, `RequestDiscoveryAdapter`).
- **Application services** orchestrate IO (HTTP fetches, parser invocations) and append result events.
- **Aggregates** validate state transitions and emit transition events. They do not call IO.
- **Events** are facts; immutable; appended.
- **Projections** subscribe to events and maintain read models.
- **Coordinators** are deterministic workflow drivers with a durable inbox (handled events) + transactional outbox (commands to dispatch). Most coordinators are not themselves event-sourced; that complexity is reserved for workflows with user-visible decisions.
- **Readers** consume projections, never aggregates.

### 3.2 Aggregates

| Aggregate | aggregate_type | Lifecycle states (validates only — no IO) | Owns |
|---|---|---|---|
| `Project` | `project` | NEW → REGISTERED → METADATA_KNOWN → ARCHIVED | Source identity, paper PDF reference, paper metadata |
| `ParsedPaper` | `parsed_paper` | PENDING → PARSING → PARSED \| FAILED | Parser identity (name+version), sections, references, figures |
| `Discovery` | `discovery` | PENDING → IN_PROGRESS → COMPLETED \| FAILED | Per-adapter outcomes |
| `Index` | `index` | PENDING → INDEXING → INDEXED \| FAILED | SourceRefs, Chunks, embedding job state |
| `Workspace` | `workspace` | CREATED → LOADING → READY → CLOSED | Per-agent variable bindings, tool call history, scope assignments |

Each aggregate has a content-hashed or ULID-based `aggregate_id` and is loaded by replaying its own event stream (with optional snapshot acceleration).

### 3.3 Application services (the IO boundary)

Application services are thin orchestrators that:
1. Receive a command.
2. Optionally load aggregates (for state validation).
3. Perform IO (HTTP, parser, embedder, etc.).
4. Append the resulting events to the store.

```python
class IntakeAppService:
    def __init__(self, store: EventStore, fetchers: Mapping[str, IntakeFetcher], clock: Clock) -> None: ...

    async def handle_register_project(self, cmd: RegisterProject) -> ProjectId:
        # Aggregate validates command shape + uniqueness.
        agg = await self._load_or_init(ProjectId.from_source(cmd.source))
        events = agg.handle_register(cmd)             # state-transition only
        await self.store.append(agg.id, "project", events, expected_version=agg.version)
        return agg.id

    async def handle_fetch_paper(self, cmd: FetchPaper) -> None:
        agg = await self._load(cmd.project_id)
        agg.guard_can_fetch()                          # validates state, no IO
        fetcher = self.fetchers[cmd.source.kind]
        try:
            mat = await fetcher.fetch(cmd.source)      # IO happens here
            await self.store.append(agg.id, "project", [PaperFetched(...)], expected_version=agg.version)
        except FetchError as e:
            await self.store.append(agg.id, "project", [PaperFetchFailed(...)], expected_version=agg.version)
```

This pattern is repeated for parser invocation, discovery adapter calls, indexing, and workspace builds.

### 3.4 Projections (read models)

| Projection | Subscribes to | Materializes |
|---|---|---|
| `ProjectsProjection` | `project` events | Project list, status, metadata |
| `ParsedPapersProjection` | `parsed_paper` events | Sections, references, figures, full text |
| `ArtifactsProjection` | `discovery` events | Discovered artifacts per project, by kind, with trust |
| `SourcesProjection` | `index` events | SourceRef and Chunk lookup tables |
| `SemanticIndexProjection` | `index` events | Chroma collection per project (built from stored embedding blobs, never by re-embedding on rebuild) |
| `WorkspaceProjection` | `workspace` events | Per-workspace, per-scope variable bindings (`Cited[T]`) |
| `CitationGraphProjection` | `workspace`, `verification` events | Edges from claim → evidence chunks |
| `EventTimelineProjection` | * (firehose) | Per-project timeline for dashboard replay |
| `OutboxBacklogProjection` | coordinator inbox/outbox tables (not event-sourced) | Operational health view |

Projections are eventually consistent. Each projection persists a checkpoint (last applied `global_position`) and supports `reset()` + replay-from-zero rebuild. **`SemanticIndexProjection` rebuild is special-cased**: it never re-embeds; it loads embedding vectors from stored blobs (referenced in `ChunkEmbedded` events). Re-embedding is opt-in via an explicit `ReindexEmbeddings` command and produces new `ChunkEmbedded` events with a new `embedding_model` field.

### 3.5 Coordinators (durable inbox + transactional outbox)

**Most coordinators are not event-sourced.** They use:

```sql
CREATE TABLE coordinator_inbox (
    coordinator_name TEXT NOT NULL,
    handled_event_id TEXT NOT NULL,            -- idempotent ack
    handled_at TEXT NOT NULL,
    PRIMARY KEY (coordinator_name, handled_event_id)
);

CREATE TABLE coordinator_state (
    coordinator_name TEXT NOT NULL,
    aggregate_key TEXT NOT NULL,               -- e.g. project_id when joining parsing+discovery
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (coordinator_name, aggregate_key)
);

CREATE TABLE coordinator_outbox (
    outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
    coordinator_name TEXT NOT NULL,
    command_type TEXT NOT NULL,
    command_payload TEXT NOT NULL,
    causation_event_id TEXT,
    correlation_id TEXT NOT NULL,
    enqueued_at TEXT NOT NULL,
    dispatched_at TEXT,                        -- NULL until dispatcher acks
    attempts INTEGER NOT NULL DEFAULT 0
);
```

A single transaction handles: insert into `coordinator_inbox` (idempotent ack), update `coordinator_state`, insert into `coordinator_outbox`. A separate dispatcher loop drains `coordinator_outbox` to the command bus with retry + dead-letter.

This gives **at-least-once** processing with **idempotent at-most-once** effects (downstream commands carry their own command_id; the idempotency table in §4.6 prevents duplicate handling).

Coordinators in scope:

| Coordinator | Reacts to | Issues commands | State |
|---|---|---|---|
| `IngestionCoordinator` | `ProjectCreated` | `FetchPaper`, `StartDiscovery` | None (stateless join) |
| `IndexingCoordinator` | `ParsingCompleted`, `DiscoveryCompleted` | `StartIndexing` once both arrive | `{project_id: {parsed: bool, discovered: bool}}` |
| `WorkspaceReadyCoordinator` | `IndexingCompleted`, `AgentRegistered` (from #10) | `BuildWorkspace` per registered agent | None (subscribes to AgentRegistry events owned by #10) |
| `RetryCoordinator` | `*Failed` with `retryable=True` | re-issues original command with backoff | `{causation_chain_id: {attempts: int, next_at: ts}}` |
| `EmbeddingCoordinator` | `ChunkCreated` | `EmbedChunk` (rate-limited via outbox) | `{project_id: {queued: int, in_flight: int}}` |

**Event-sourced coordinator state is reserved** for workflows with user-visible decisions, approvals, or long-running negotiations. None in this spec yet. The AgentRegistry that `WorkspaceReadyCoordinator` consults is **owned by #10/#2** (the orchestrator team) — we subscribe to their `AgentRegistered`/`AgentDeregistered` events; we do not implement the registry.

### 3.6 Module layout

```
openresearch/
├── eventstore/                          # The bedrock
│   ├── interface.py                     # EventStore + Subscription + StoreCapabilities Protocols
│   ├── sqlite_store.py                  # SQLite + WAL implementation
│   ├── jsonl_store.py                   # JSONL implementation (debug, ops dump)
│   ├── snapshot.py                      # Snapshot Protocol + integrity envelope
│   ├── upcaster.py                      # Upcaster Protocol + registry + golden fixture support
│   ├── subscription.py                  # Persistent subscription with ack/nack/lease
│   └── replay.py                        # Bulk replay + determinism harness
├── messaging/
│   ├── command.py                       # Command base + CommandDispatcher
│   ├── event.py                         # DomainEvent base + StoredEvent + EventEnvelope
│   ├── idempotency.py                   # Bounded (aggregate_id, command_id) -> result_event_ids table
│   └── bus.py                           # InProcMessageBus, NatsMessageBus (slot-in)
├── coordinators/                        # Durable inbox/outbox coordinators
│   ├── runtime.py                       # CoordinatorRuntime: inbox dedupe + state + outbox dispatch
│   ├── ingestion_coordinator.py
│   ├── indexing_coordinator.py
│   ├── workspace_ready_coordinator.py
│   ├── retry_coordinator.py
│   └── embedding_coordinator.py
├── ingestion/
│   ├── intake/                          # Issue #12
│   │   ├── commands.py                  # RegisterProject, FetchPaper, ExtractMetadata
│   │   ├── events.py                    # ProjectCreated, PaperFetched, MetadataExtracted, *Failed
│   │   ├── aggregate.py                 # ProjectAggregate (validates only)
│   │   ├── service.py                   # IntakeAppService (IO + append)
│   │   ├── fetchers/                    # PdfPathFetcher, ArxivFetcher, DoiFetcher
│   │   └── projections.py               # ProjectsProjection
│   ├── parser/                          # Issue #13
│   │   ├── commands.py                  # StartParsing
│   │   ├── events.py                    # ParsingStarted, SectionExtracted, EquationExtracted, ReferenceExtracted, FigureExtracted, TableExtracted, ParsingCompleted, ParsingFailed
│   │   ├── aggregate.py                 # ParsedPaperAggregate
│   │   ├── service.py                   # ParserAppService (runs parser worker → appends events)
│   │   ├── workers/                     # parser process isolation: pymupdf_worker.py, nougat_worker.py
│   │   ├── pymupdf_parser.py
│   │   ├── nougat_parser.py
│   │   └── projections.py               # ParsedPapersProjection
│   └── discovery/                       # Issue #14
│       ├── commands.py                  # StartDiscovery, RunDiscoveryAdapter
│       ├── events.py                    # DiscoveryStarted, AdapterStarted, ArtifactFound, AdapterFailed, AdapterCompleted, DiscoveryCompleted
│       ├── aggregate.py                 # DiscoveryAggregate
│       ├── service.py                   # DiscoveryAppService
│       ├── adapters/                    # github, papers_with_code, huggingface, semantic_scholar
│       ├── trust.py                     # ArtifactTrust scoring (official/recommended/community)
│       └── projections.py               # ArtifactsProjection
├── context/
│   ├── indexer/                         # Issue #15
│   │   ├── commands.py                  # StartIndexing, EmbedChunk, ReindexEmbeddings
│   │   ├── events.py                    # IndexingStarted, SourceRegistered, ChunkCreated, EmbeddingQueued, EmbeddingStarted, ChunkEmbedded, EmbeddingFailed, IndexingCompleted, IndexingFailed
│   │   ├── aggregate.py                 # IndexAggregate
│   │   ├── service.py                   # IndexerAppService
│   │   ├── chunkers/
│   │   ├── embedder.py                  # EmbedderClient with capture
│   │   └── projections.py               # SourcesProjection, SemanticIndexProjection
│   └── workspace/                       # Issue #16
│       ├── commands.py                  # BuildWorkspace, LoadVariable, EnrichVariable, AttachCitation, CallTool, PromoteVariable, CloseWorkspace
│       ├── events.py                    # WorkspaceCreated, VariableLoaded, VariableEnriched, VariablePromoted, CitationAttached, ToolInvoked, WorkspaceReady, WorkspaceClosed
│       ├── aggregate.py                 # WorkspaceAggregate
│       ├── service.py                   # WorkspaceAppService
│       ├── model.py                     # Cited[T], Citation, NonEmptyCitations, Provenance, Scope
│       ├── tools/                       # interface, lookup, semantic_search (others slot-in)
│       └── projections.py               # WorkspaceProjection, CitationGraphProjection
├── capture/                             # Synchronous external-call capture
│   ├── http_client.py                   # CapturingHttpClient (httpx wrapper)
│   ├── llm_client.py                    # CapturingLlmClient (Anthropic SDK wrapper)
│   ├── embedding_client.py              # CapturingEmbeddingClient
│   ├── chroma_client.py                 # CapturingChromaClient
│   ├── web_search_client.py             # CapturingWebSearchClient
│   ├── notebook_lm_client.py            # CapturingNotebookLmClient
│   ├── blob_store.py                    # Atomic blob write + GC + encryption
│   └── replay_mode.py                   # ReplayInterceptor: short-circuits to captured responses
└── shared/
    ├── ids.py                           # Composed content-addressed IDs
    ├── envelope.py                      # EventEnvelope with correlation/causation/source/schema_version
    ├── errors.py                        # OpenResearchError hierarchy
    ├── observability.py                 # structlog + OpenTelemetry helpers
    ├── security.py                      # HeaderRedactor, SecretEncryptor, ModelDigestVerifier
    └── config.py                        # pydantic-settings
```

## 4. The Event Store

### 4.1 Protocols

```python
class EventStore(Protocol):
    @property
    def capabilities(self) -> StoreCapabilities: ...

    async def append(
        self,
        aggregate_id: AggregateId,
        aggregate_type: str,
        events: Sequence[DomainEvent],
        expected_version: int,
        envelope: EventEnvelope,
    ) -> AppendResult: ...

    async def load(
        self,
        aggregate_id: AggregateId,
        from_version: int = 0,
    ) -> AsyncIterator[StoredEvent]: ...

    async def load_global(
        self,
        from_position: int = 0,
        to_position: int | None = None,
        types: Iterable[str] | None = None,
        batch_size: int = 1000,
    ) -> AsyncIterator[StoredEvent]: ...

    async def subscribe(
        self,
        subscription_name: str,
        types: Iterable[str] | None = None,
    ) -> Subscription: ...

    async def get_aggregate_version(self, aggregate_id: AggregateId) -> int: ...


class Subscription(Protocol):
    """Persistent subscription with checkpoint, ack/nack, and lease."""
    @property
    def name(self) -> str: ...
    @property
    def position(self) -> int: ...

    async def __aiter__(self) -> AsyncIterator[StoredEvent]: ...
    async def ack(self, event: StoredEvent) -> None: ...        # advances checkpoint
    async def nack(self, event: StoredEvent, *, retry_after_seconds: float) -> None: ...
    async def renew_lease(self) -> None: ...                   # for long-running handlers
    async def close(self) -> None: ...


@dataclass(frozen=True)
class StoreCapabilities:
    supports_persistent_subscriptions: bool
    supports_stream_categories: bool
    max_event_payload_bytes: int
    optimistic_concurrency: bool
    supports_transactional_outbox: bool


class AppendError(Exception): ...
class ConcurrencyError(AppendError):
    """Raised when expected_version != current_version. Carries actual_version."""
    actual_version: int
class DuplicateEventError(AppendError): ...
```

### 4.2 SQLite-backed implementation (default)

```sql
CREATE TABLE events (
    global_position INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    aggregate_id TEXT NOT NULL,
    aggregate_type TEXT NOT NULL,
    aggregate_version INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,           -- envelope: correlation, causation, source, occurred_at
    occurred_at TEXT NOT NULL,
    UNIQUE (aggregate_id, aggregate_version)
);
CREATE INDEX idx_events_aggregate     ON events(aggregate_id, aggregate_version);
CREATE INDEX idx_events_type          ON events(event_type);
CREATE INDEX idx_events_occurred_at   ON events(occurred_at);

CREATE TABLE subscription_checkpoints (
    subscription_name TEXT PRIMARY KEY,
    last_position INTEGER NOT NULL,
    leased_by TEXT,                        -- worker id holding the lease
    lease_expires_at TEXT,
    last_ack_at TEXT NOT NULL
);

CREATE TABLE projection_checkpoints (
    projection_name TEXT PRIMARY KEY,
    last_position INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
```

PRAGMAs at startup: `journal_mode=WAL`, `synchronous=NORMAL`, `foreign_keys=ON`, `busy_timeout=5000`.

**Realistic throughput.** Under mixed load (event append + projection update + Chroma write + blob fsync), expect **hundreds to low-thousands of durable appends/sec on developer hardware**, not 10K. 10K is an isolated-write SQLite microbenchmark; ReproLab's load is far from isolated.

### 4.3 Migration triggers (when to leave SQLite)

Move to EventStoreDB or Postgres when **any** of:
1. More than one project ingests concurrently and queue depth grows.
2. Phase 3 remote runners arrive (PRD §2482-2496).
3. Phase 4 multi-user projects arrive (PRD §2504-2507).
4. Projection lag is user-visible (>5s steady state).
5. Need durable fan-out / backpressure across machine boundaries.
6. Any append latency p99 exceeds 100ms.

EventStoreDB is preferred when aggregate stream consistency and persistent subscriptions matter most. Postgres is preferred when SQL operability and shared transactional outbox matter most. NATS JetStream is a fan-out bus, not a canonical store — pair it with a database.

### 4.4 Snapshot integrity

```python
@dataclass(frozen=True)
class SnapshotEnvelope:
    aggregate_id: AggregateId
    aggregate_type: str
    aggregate_version: int
    global_position: int                   # last event included
    payload_json: str
    payload_schema_version: int
    included_event_ids_hash: str           # sha256 of sorted included event_ids
    upcaster_chain_versions: tuple[int, ...]   # upcaster versions applied to construct payload
    taken_at: datetime
```

On replay, the loader verifies:
1. `payload_schema_version` is supported (or upcasted to current).
2. `included_event_ids_hash` matches recomputed hash from store events 0…aggregate_version.
3. `upcaster_chain_versions` matches the registered chain.

A snapshot that fails any check is discarded and replay starts from event 0.

**Snapshot policy is measurement-driven, not fixed.** No "every 50 events" rule. The runtime tracks aggregate replay duration per aggregate_type; when replay exceeds 100ms on average, the snapshot scheduler kicks in for that type. Initial threshold tunable via config; revisit when metrics arrive.

### 4.5 Captured external interactions

Every external surface that introduces non-determinism is captured. Not just HTTP.

```python
class ExternalInteractionAttempted(DomainEvent):
    """Pre-call event so we know the call started even if the process crashes mid-call."""
    schema_version = 1
    interaction_id: InteractionId          # ULID
    surface: str                           # "http" | "llm" | "embedding" | "chroma" | "web_search" | "notebook_lm"
    request_summary: str                   # human-readable (e.g. "GET arxiv.org/...")
    request_fingerprint: str               # sha256 of (method, url, normalized body, surface-specific key)
    cache_key: str                         # for replay deduplication
    attempt: int                           # 1, 2, 3, ... for retries

class ExternalApiCalled(DomainEvent):
    schema_version = 1
    interaction_id: InteractionId
    surface: str
    request_method: str | None             # null for non-HTTP surfaces
    request_url: str | None
    request_headers_redacted_json: str     # bearer tokens, cookies stripped or encrypted-blob refs
    request_body_sha256: str | None
    request_body_blob_path: str | None     # if body > 4KB, stored as blob
    response_status: int | None
    response_headers_json: str
    response_body_sha256: str
    response_body_blob_path: str
    redirect_chain: tuple[str, ...]
    duration_ms: int

class ExternalApiFailed(DomainEvent):
    schema_version = 1
    interaction_id: InteractionId
    surface: str
    error_kind: str                        # "timeout" | "connection_reset" | "ssl_error" | "rate_limited" | "server_error" | "client_error"
    error_message: str                     # truncated to 1KB
    duration_ms: int
    will_retry: bool
```

`*Attempted` is appended **before** the call leaves the process. `*Called` or `*Failed` is appended **after**. If the process dies between them, replay sees the attempt with no resolution and the retry coordinator can re-issue.

**Captured surfaces:**

| Surface | Wrapper | Captured fields | Notes |
|---|---|---|---|
| HTTP (httpx) | `CapturingHttpClient` | method, url, headers (redacted), body sha + blob, status, response headers, redirect chain, body sha + blob, duration | Generic |
| Anthropic LLM | `CapturingLlmClient` | model name, model digest hint, prompt sha, prompt blob, system blob, tools fingerprint, response sha + blob, usage tokens | `model_digest_hint` is the first 12 chars of a digest published by Anthropic per model release; allows replay-time mismatch detection |
| Embeddings | `CapturingEmbeddingClient` | model name, model package + weight digest, input batch sha, output vectors blob, duration | Pinned model digests (§8.10) |
| Chroma queries | `CapturingChromaClient` | collection, query embedding sha, top-k, returned chunk_ids, distances, duration | Captures the query result, not the index |
| Web search | `CapturingWebSearchClient` | query, provider, top results URLs, snippets blob | |
| NotebookLM | `CapturingNotebookLmClient` | notebook_id, query sha, response sha + blob, sources cited | |

`shared/http.py` and the other clients perform synchronous capture inline. **There is no `CapturedResponseFlow`.** A separate `BlobLifecycleJob` may compact, dedupe, or GC blobs but is not the primary recorder.

### 4.6 Idempotency table

```sql
CREATE TABLE command_idempotency (
    aggregate_id TEXT NOT NULL,
    command_id TEXT NOT NULL,
    result_event_ids_json TEXT NOT NULL,    -- ['evt_...', 'evt_...']
    handled_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,               -- bounded retention
    PRIMARY KEY (aggregate_id, command_id)
);
CREATE INDEX idx_idempotency_expires ON command_idempotency(expires_at);
```

Bounded retention: default 30 days, configurable. A background job purges expired rows.

When an application service receives a command:
1. Compute `command_id` (caller-supplied or content-hashed).
2. Check the idempotency table; if hit, re-emit the previously recorded result events (or report success to caller without re-doing IO).
3. Otherwise, perform IO + append, then write the idempotency row in the same transaction as the append.

This handles command de-dupe without growing aggregate state.

### 4.7 Upcaster registry

```python
class Upcaster(Protocol):
    @property
    def event_type(self) -> str: ...
    @property
    def from_version(self) -> int: ...
    @property
    def to_version(self) -> int: ...     # always from_version + 1
    def upcast(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class UpcasterRegistry:
    def register(self, upcaster: Upcaster) -> None: ...
    def chain(self, event_type: str, from_version: int, to_version: int) -> Sequence[Upcaster]: ...
    def upcast(self, event_type: str, payload: dict[str, Any], from_version: int) -> dict[str, Any]: ...

    # Auto-discovery: any class with @register_upcaster decorator in `*.upcasters` modules
    def autodiscover(self, packages: Iterable[str]) -> None: ...
```

Registry rules:
1. For every `event_type`, the chain `from_version → from_version+1 → ... → current_version` must be complete (no gaps). Validated at startup; missing upcasters raise `UpcasterChainBroken`.
2. Renamed event types use `EventTypeRename(old_name, new_name, at_version)` records; loader applies the rename before invoking type-specific upcasters.
3. Each upcaster has at least one **golden fixture** test: input payload at `from_version` → expected payload at `to_version`. Fixtures are checked into `tests/upcaster_fixtures/{event_type}/v{from}_to_v{to}.json`.
4. Every upcaster output is validated against the next-version Pydantic schema before being yielded. An upcaster that produces invalid output raises `UpcasterProducedInvalid`.

Snapshots also flow through upcasters via their `payload_schema_version` field.

## 5. Commands and Events — Module by Module

### 5.1 Intake (#12)

```python
@dataclass(frozen=True)
class RegisterProject(Command):
    command_id: CommandId
    source: PaperSource

@dataclass(frozen=True)
class FetchPaper(Command):
    command_id: CommandId
    project_id: ProjectId

@dataclass(frozen=True)
class ExtractMetadata(Command):
    command_id: CommandId
    project_id: ProjectId


class ProjectCreated(DomainEvent):
    schema_version = 1
    project_id: ProjectId
    source: PaperSource

class PaperFetched(DomainEvent):
    schema_version = 1
    project_id: ProjectId
    raw_paper_path: str
    pdf_sha256: str
    pdf_size_bytes: int
    mime_validated: bool
    fetched_via: str
    fetched_interaction_id: InteractionId   # links to ExternalApiCalled

class PaperFetchFailed(DomainEvent):
    schema_version = 1
    project_id: ProjectId
    cause_kind: str
    cause_message: str
    retryable: bool

class MetadataExtracted(DomainEvent):
    schema_version = 1
    project_id: ProjectId
    metadata: PaperMetadata
    extractor_name: str
    extractor_version: str
```

**`ProjectAggregate.handle()` is pure**:

```python
class ProjectAggregate(Aggregate):
    aggregate_type = "project"
    project_id: ProjectId
    state: ProjectState
    version: int

    def handle_register(self, cmd: RegisterProject) -> Sequence[DomainEvent]:
        if self.state is not ProjectState.NEW:
            raise InvalidStateTransition(self.state, "register")
        return [ProjectCreated(project_id=cmd.source_to_id(), source=cmd.source)]

    def guard_can_fetch(self) -> None:
        if self.state is not ProjectState.REGISTERED:
            raise InvalidStateTransition(self.state, "fetch")

    def apply(self, ev: DomainEvent) -> None:
        match ev:
            case ProjectCreated(): self.state = ProjectState.REGISTERED
            case PaperFetched(): self.state = ProjectState.FETCHED
            case PaperFetchFailed(): pass  # remains REGISTERED so retry is valid
            case MetadataExtracted(): self.state = ProjectState.METADATA_KNOWN
```

The aggregate validates and emits state-transition events. **All HTTP, parsing, IO is in `IntakeAppService`.**

### 5.2 Parser (#13)

```python
class ParsingStarted(DomainEvent):
    project_id: ProjectId
    parser_name: str
    parser_version: str
    parser_digest: str            # hash of parser package + weights (Nougat) or version string (PyMuPDF)
    isolation_mode: str           # "in_process" | "subprocess" | "container"

class SectionExtracted(DomainEvent):
    project_id: ProjectId
    section: Section
    extraction_confidence: float  # 0..1
    parser_warnings: tuple[str, ...]

class EquationExtracted(DomainEvent):
    project_id: ProjectId
    equation: Equation            # latex, inline/display, anchor section_id

class ReferenceExtracted(DomainEvent): ...
class FigureExtracted(DomainEvent): ...
class TableExtracted(DomainEvent): ...

class ParsingCompleted(DomainEvent):
    project_id: ProjectId
    section_count: int
    reference_count: int
    figure_count: int
    table_count: int
    equation_count: int
    parse_duration_ms: int
    full_text_blob_path: str      # raw concatenated text stored as blob
    full_text_sha256: str

class ParsingFailed(DomainEvent):
    project_id: ProjectId
    parser_name: str
    cause_kind: str
    cause_message: str
    partial_progress: PartialProgress  # e.g. "extracted 4 of 9 sections"
    retryable: bool
```

Equations are first-class because the PRD requires equations in the REPL workspace (PRD §1219-1224). `extraction_confidence` lets verifiers and indexers down-weight low-confidence extractions.

### 5.3 Discovery (#14)

```python
class ArtifactTrust(StrEnum):
    OFFICIAL = "official"           # paper authors' repo (e.g. arxiv author affiliation match)
    RECOMMENDED = "recommended"     # Papers with Code canonical
    COMMUNITY = "community"         # high-star fork
    UNVERIFIED = "unverified"

class DiscoveredArtifact(BaseModel):
    id: ArtifactId
    project_id: ProjectId
    kind: ArtifactKind
    canonical_url: str
    title: str
    metadata: Mapping[str, Any]
    trust: ArtifactTrust
    license: str | None             # SPDX id or null
    commit_sha: str | None          # for repo artifacts
    last_updated: datetime | None
    discovered_by: str
    confidence: float
    contradiction_evidence: tuple[str, ...]   # e.g. README claims python==3.6, requirements.txt says 3.8
    discovered_at: datetime
    discovery_interaction_id: InteractionId

class ArtifactFound(DomainEvent):
    project_id: ProjectId
    adapter: str
    artifact: DiscoveredArtifact
```

Trust scoring lives in `ingestion/discovery/trust.py`, computed deterministically from adapter outputs. Verifiers and the supervisor read trust on `ArtifactsProjection`.

### 5.4 Indexer (#15)

```python
class IndexingStarted(DomainEvent):
    project_id: ProjectId
    chunker_name: str
    chunker_version: str

class SourceRegistered(DomainEvent):
    project_id: ProjectId
    source: SourceRef

class ChunkCreated(DomainEvent):
    project_id: ProjectId
    chunk: Chunk

class EmbeddingQueued(DomainEvent):
    project_id: ProjectId
    chunk_id: ChunkId
    embedding_model: str
    embedding_model_digest: str

class EmbeddingStarted(DomainEvent):
    project_id: ProjectId
    chunk_id: ChunkId

class ChunkEmbedded(DomainEvent):
    project_id: ProjectId
    chunk_id: ChunkId
    embedding_model: str
    embedding_model_digest: str
    embedding_dim: int
    embedding_blob_path: str
    duration_ms: int
    interaction_id: InteractionId

class EmbeddingFailed(DomainEvent):
    project_id: ProjectId
    chunk_id: ChunkId
    cause_kind: str
    retryable: bool

class IndexingCompleted(DomainEvent):
    project_id: ProjectId
    source_count: int
    chunk_count: int
    embedding_pending_count: int    # may be > 0; embeddings continue async
    duration_ms: int
```

`semantic_search()` may return results before all embeddings land. Result payloads carry `partial_index: bool` plus the count of embedded vs total chunks so callers can decide whether to wait.

### 5.5 Workspace (#16)

# Scope, Citation, and NonEmptyCitations are imported from the canonical schema
# packages owned by other teammates — we do NOT redefine them here.
#
#   from openresearch.contracts.blackboard import Scope             # owned by #11
#   from openresearch.contracts.citations import (                  # owned by #8
#       Citation,
#       NonEmptyCitations,
#   )
#
# Scope values (mirrors PRD §1078-1082 and #11):
#   PRIVATE_TO_PARENT  — only the spawning agent sees it
#   BRANCH_SHARED      — shared across an improvement branch
#   GLOBAL_VERIFIED    — promoted after verifier confirmation
#
# NonEmptyCitations = Annotated[tuple[Citation, ...], Field(min_length=1)]

class WorkspaceCreated(DomainEvent):
    workspace_id: WorkspaceId
    project_id: ProjectId
    agent_name: str
    parent_workspace_id: WorkspaceId | None
    branch_id: BranchId | None
    task_id: TaskId

class VariableLoaded(DomainEvent):
    workspace_id: WorkspaceId
    variable_name: str
    value_payload: dict
    citations: NonEmptyCitations
    scope: Scope
    source_agent: str | None

class VariableEnriched(DomainEvent):
    workspace_id: WorkspaceId
    variable_name: str
    value_payload: dict
    citations: NonEmptyCitations
    scope: Scope
    enriched_by: str

class VariablePromoted(DomainEvent):
    workspace_id: WorkspaceId
    variable_name: str
    from_scope: Scope
    to_scope: Scope
    promotion_event_id: EventId        # the verification event that authorized promotion

class CitationAttached(DomainEvent):
    workspace_id: WorkspaceId
    decision_id: str
    decision_payload: dict
    citations: NonEmptyCitations

class ToolInvoked(DomainEvent):
    workspace_id: WorkspaceId
    tool_name: str
    arguments: dict
    result_payload: dict
    citations: NonEmptyCitations
    duration_ms: int
    interaction_id: InteractionId | None    # null if tool was pure (lookup); set if external (semantic_search → embedding+Chroma)

class WorkspaceReady(DomainEvent):
    workspace_id: WorkspaceId
    variable_count: int

class WorkspaceClosed(DomainEvent):
    workspace_id: WorkspaceId
    reason: str
```

### 5.6 Citation invariant — defense in depth

Three layers, each independently sufficient:

1. **Pydantic event-payload validators** — `NonEmptyCitations = Annotated[tuple[Citation, ...], Field(min_length=1)]`. Construction with empty tuple raises `pydantic.ValidationError`.
2. **EventStore append re-validation** — every event arriving at `append()` is validated against its registered Pydantic class via `model_validate(...)`. **`model_construct` is banned outside tests** by an enforced lint rule (`ruff` check + reviewer guideline). Hand-rolled dicts that bypass the constructor still fail at append.
3. **`Cited[T]` materialization in projections** — `WorkspaceProjection` constructs `Cited[T]` whose `__post_init__` raises `CitationMissingError` on empty citations.

Additionally — **semantic citation validation**:

4. Every `Citation.source_id` referenced in a stored event must exist in `SourcesProjection`. The `WorkspaceAppService` re-checks before append (catches a stale handle before it lands in the log). The `WorkspaceProjection` re-checks on apply (catches projection-rebuild edge cases). Failed lookups emit `CitationSourceMissing` events, which `RetryCoordinator` treats as non-retryable until the corresponding source lands.

5. Every `Citation.chunk_id` (if non-null) must resolve to an immutable `Chunk` whose composed ID still matches its content (§7.2). A drifting chunk is detected and emits `CitationChunkDrifted`.

The single canonical type `NonEmptyCitations` is defined in the canonical schema package (#8) and re-exported here; no per-event ad hoc annotations.

## 6. Projections

### 6.1 General projection contract

```python
class Projection(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def subscribed_event_types(self) -> Iterable[str]: ...
    async def apply(self, event: StoredEvent) -> None: ...
    async def reset(self) -> None: ...

    # Atomic checkpointing requirement:
    #   If the projection's state lives in the same DB as projection_checkpoints
    #   (e.g., SQLite), `apply` MUST update both in one transaction.
    #   For external state (Chroma), see §6.2.
```

### 6.2 SemanticIndexProjection — special handling

Chroma cannot participate in the SQLite transaction. Therefore:

```sql
CREATE TABLE semantic_index_log (
    chunk_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_model_digest TEXT NOT NULL,
    event_global_position INTEGER NOT NULL,
    applied_at TEXT NOT NULL,
    chroma_collection TEXT NOT NULL,
    PRIMARY KEY (chunk_id, embedding_model, embedding_model_digest)
);
```

- On `ChunkEmbedded`, the projection inserts the row in the same SQLite transaction as advancing its checkpoint, **before** calling Chroma. Then `chroma.add(...)` is called.
- A repair job periodically diffs `semantic_index_log` against the Chroma collection contents and replays missing chunks.
- On rebuild, the projection reads `semantic_index_log` rows + their referenced embedding blobs, and replays into a fresh Chroma collection. **No re-embedding.**

### 6.3 Other projections

All SQLite-resident projections (`Projects`, `ParsedPapers`, `Artifacts`, `Sources`, `Workspace`, `CitationGraph`, `EventTimeline`) update state and checkpoint in one transaction.

## 7. Coordinators (the inbox/outbox runtime)

### 7.1 The runtime

```python
class CoordinatorRuntime:
    def __init__(self, name: str, store: EventStore, dispatcher: CommandDispatcher, db: Database) -> None: ...

    async def run(self, handler: CoordinatorHandler) -> None:
        sub = await self.store.subscribe(self.name, types=handler.subscribed_types)
        async for event in sub:
            async with self.db.transaction():
                if await self._already_handled(event):
                    await sub.ack(event); continue
                state = await self._load_state(handler.aggregate_key(event))
                outcome = handler.handle(event, state)
                await self._mark_handled(event)
                if outcome.new_state is not None:
                    await self._save_state(handler.aggregate_key(event), outcome.new_state)
                for cmd in outcome.commands:
                    await self._enqueue_outbox(cmd, event.event_id)
            await sub.ack(event)

        # outbox dispatcher runs in its own loop, drains coordinator_outbox
```

`handler.handle(event, state)` is **pure** — no IO. Returns `(new_state, list[Command])`. This makes coordinators trivially testable.

### 7.2 Composed content-addressed IDs

```python
def chunk_id_for(
    source_id: SourceId,
    chunker_name: str,
    chunker_version: str,
    text: str,
    span: tuple[int, int],
    normalization: str = "nfkc-strip-trailing-ws-v1",
) -> ChunkId:
    digest = sha256()
    digest.update(source_id.encode())
    digest.update(b"\x00")
    digest.update(chunker_name.encode()); digest.update(b"\x00")
    digest.update(chunker_version.encode()); digest.update(b"\x00")
    digest.update(span[0].to_bytes(8, "big")); digest.update(span[1].to_bytes(8, "big"))
    digest.update(normalization.encode()); digest.update(b"\x00")
    digest.update(text.encode())
    return ChunkId(f"chk_{digest.hexdigest()[:24]}")
```

A different chunker version produces a different ID for the same text — **no silent reuse**. Same for `SourceId`, `ArtifactId`, `EmbeddingId`.

## 8. Cross-Cutting Concerns

### 8.1 Event envelope

```python
@dataclass(frozen=True)
class EventEnvelope:
    event_id: EventId               # ULID
    correlation_id: CorrelationId
    causation_id: EventId | None
    occurred_at: datetime
    source: str                     # producing module
    schema_version: int
```

### 8.2 Idempotency (commands and events)

- Events: unique `event_id` index in the store catches exact duplicates.
- Commands: `command_idempotency` table (§4.6) caches result event IDs per (aggregate_id, command_id).
- Coordinator inbox (§3.5) catches event-handling duplicates per coordinator.

### 8.3 Concurrency

- Asyncio everywhere (`anyio`-portable).
- SQLite single-writer; aggregate handlers serialized per-aggregate via `asyncio.Lock` keyed on `aggregate_id`.
- Adapters within a service run concurrently (parser ∥ discovery; discovery adapters ∥ each other).
- Projections run on independent persistent subscriptions; one slow projection cannot block another.
- Coordinator outbox dispatchers run independently per coordinator with bounded concurrency.

### 8.4 Schema evolution

`UpcasterRegistry` (§4.7). Snapshots flow through upcasters via their `payload_schema_version`. Renamed event types use `EventTypeRename` records. Every upcaster ships a golden fixture.

### 8.5 Replay determinism

Three sources of non-determinism handled:

1. **External interactions** — every HTTP / LLM / embedding / Chroma / web search / NotebookLM call is wrapped in a capturing client (§4.5) and recorded as `ExternalInteractionAttempted` + `ExternalApiCalled`/`ExternalApiFailed`. Replay mode (set via `OPENRESEARCH_REPLAY_MODE=true`) installs `ReplayInterceptor` which short-circuits all wrapped clients to captured responses keyed by `cache_key`.
2. **Wall-clock time** — `Clock` Protocol injected into every service. `RealClock` in production; `EventTimeClock` in replay mode (returns `event.occurred_at`).
3. **Random IDs** — `IdGenerator` Protocol. Production uses ULIDs; replay mode uses a seeded PRNG keyed by `correlation_id`. ID collisions in replay are impossible because the production-recorded IDs are themselves what gets replayed.

Acceptance test (§14): two runs of a recorded session produce byte-identical event stores.

### 8.6 Error handling

- Every public method declares its raisable error types in the docstring.
- Adapters wrap external errors in module-typed errors with `__cause__` preserved.
- All failures emit a typed `*Failed` event with `retryable: bool` and `cause_kind` (a stable enum, not a free-form string).
- `RetryCoordinator` retries `retryable=True` failures with exponential backoff; max 3 attempts; further failures move to a dead-letter table for ops review.

### 8.7 Observability

**Logs (structlog)** — every line carries `event_id`, `correlation_id`, `causation_id`, `aggregate_id`, `service`.

**Traces (OpenTelemetry)** — span per command, per event apply, per projection apply, per external interaction. Trace context propagates via `EventEnvelope.correlation_id`.

**Metrics (Prometheus)** — beyond the obvious counters/histograms:
- `openresearch_events_appended_total{aggregate_type,event_type}`
- `openresearch_command_duration_seconds{command_type}`
- `openresearch_projection_lag{projection}` (events behind firehose)
- `openresearch_subscription_retry_total{subscription_name}`
- `openresearch_outbox_backlog{coordinator}`
- `openresearch_dead_letter_total{coordinator}`
- `openresearch_external_call_duration_seconds{surface,outcome}`
- `openresearch_external_capture_failures_total{surface}`
- `openresearch_circuit_breaker_state{host}` (gauge: 0=closed, 1=open, 2=half_open)
- `openresearch_blob_missing_total`
- `openresearch_blob_corrupt_total`
- `openresearch_llm_tokens_total{model,direction}` (cost tracking)
- `openresearch_security_policy_denied_total{policy}`

### 8.8 Configuration

`pydantic-settings` Settings model loaded once at startup. `OPENRESEARCH_*` env namespace. Frozen instance injected via DI. Secrets via env vars only.

### 8.9 Blob storage

```
runs/{project_id}/blobs/{first_two_of_sha}/{remaining_sha}.bin
```

- **Atomic write** — write to `.tmp.{ulid}`, fsync, rename.
- **SHA verification on read** — every blob read recomputes sha256 and matches the path. Mismatch raises `BlobCorruptError` and emits `BlobCorrupt` event.
- **Refcounting** — `blob_refs` table records `(sha, event_id)` pairs. Unreferenced blobs older than retention go to a quarantine directory before delete.
- **Encryption** — sensitive payloads (captured HTTP request body when it contains secrets, captured LLM prompts containing user data) are encrypted with a per-environment key (`OPENRESEARCH_BLOB_ENCRYPTION_KEY`) using authenticated encryption (AES-GCM).
- **Max blob size** — configurable per surface; default 50MB. Larger raises `BlobTooLarge`.
- **Cross-project dedupe** — content-addressed; a blob with sha X exists once globally and is referenced by all events that need it.

### 8.10 Security (data-ingress hardening)

**PDF parsing isolation** — `PyMuPdfParser` and `NougatParser` run in a subprocess worker with:
- No network namespace (denies external access).
- Memory limit 1 GB; CPU limit 2 cores; wall-clock 60s.
- seccomp filter banning `socket`, `connect`, `clone3` (where supported).
- AppArmor/SELinux profile when available.
- Worker reads PDF blob path on stdin; emits parsed payload on stdout; killed on timeout.
- A failed worker emits `ParsingFailed` with `cause_kind="worker_killed"`.

**Embedding model integrity** —
- `embedding_model_digest` is the sha256 of the model package + weights file. Pinned per environment via config. Mismatch at load time aborts startup.
- Prefer `safetensors` over pickle. `torch.load(..., weights_only=True)` only.
- No `trust_remote_code=True`. No implicit downloads in production (`HF_HUB_OFFLINE=1` after first cache).

**Captured-header redaction** —
- `HeaderRedactor` strips `Authorization`, `Cookie`, `Set-Cookie`, `Proxy-Authorization`, `X-Api-Key`, `X-Auth-Token` by default.
- Custom redaction patterns from config.
- Sensitive header values may be encrypted (per-env key) and stored as `enc:<blob_path>` references rather than plaintext, opt-in per surface.
- Bearer tokens and cookies **never** reach projection tables or dashboard events.

**Other hardening:**
- No `eval` / `exec` / `pickle` / `marshal` / `__import__(user_string)` in the data path.
- All HTTP TLS; no `verify=False`.
- Dependency pinning via `uv.lock` or `requirements.lock`; weekly `pip-audit` in CI.
- `bandit` and `ruff --select=S` in CI.

### 8.11 Replay cache key

```python
def cache_key_for(
    surface: str,
    request_method: str | None,
    request_url: str | None,
    request_body_sha256: str | None,
    surface_specific: Mapping[str, str],     # e.g. {"model": "claude-sonnet-4-6"}
) -> str:
    """Stable, surface-aware key used in replay-mode lookup."""
```

Stored in `ExternalInteractionAttempted.cache_key` and used by `ReplayInterceptor` to match captured responses. Includes surface-specific fields so two semantically-different calls with similar URLs do not collide (e.g., LLM calls with same prompt but different system message).

## 9. Data Model Summary

| Type | Defined in | Hash | Mutability |
|------|-----------|------|------------|
| `Project` | intake/aggregate | from-source ULID | frozen |
| `ParsedPaper` | parser/projection (derived) | composed | frozen |
| `Section`, `Reference`, `Figure`, `Table`, `Equation` | parser/events | content-hashed | frozen |
| `DiscoveredArtifact` | discovery/events | content-hashed | frozen |
| `SourceRef` | indexer/events | composed (parser+chunker version) | frozen |
| `Chunk` | indexer/events | composed (chunker+span+normalization+text) | frozen |
| `Citation` | workspace/model | n/a | frozen |
| `NonEmptyCitations` | canonical (#8) re-export | n/a | typed alias, validated |
| `Cited[T]` | workspace/model | n/a | frozen, ≥1 citation invariant |
| Event payloads | per-module `events.py` | event_id ULID | frozen, Pydantic-validated |

## 10. Testing Strategy

### 10.1 Test pyramid

| Level | Coverage |
|---|---|
| Aggregate unit | Given prior events → command → expected new events (pure, no infra) |
| Schema round-trip | Pydantic validate → JSON → validate, property-based via Hypothesis |
| Upcaster golden fixtures | One per `(event_type, from_version → to_version)` pair |
| Projection replay | Synthetic event stream → expected projection state |
| Service integration | Command → service → store → projection caught up |
| Adapter contract | Real adapter against recorded HTTP fixtures |
| Coordinator handler | `handler.handle(event, state)` purity tests |
| Replay determinism | Run twice; diff event store byte-by-byte |
| Crash injection | Faults injected at append/outbox/checkpoint boundaries; recovery to consistent state |
| End-to-end | `RegisterProject(arxiv://1707.06347)` → `WorkspaceReady` |
| Smoke (live API) | Marked `smoke`; nightly only |

### 10.2 Crash-injection matrix

A dedicated test harness inserts faults at:
- After append, before commit (transaction rollback).
- After commit, before outbox enqueue.
- After outbox enqueue, before dispatcher ack.
- During projection apply, between state mutation and checkpoint update.
- During capturing client, between `*Attempted` and `*Called`.
- During snapshot save, after metadata, before payload.

Each injection point has at least one assertion: "after restart, the system is in a consistent state and the same logical work eventually completes".

### 10.3 Invariants verified

1. `Cited[T]` cannot be constructed empty.
2. `NonEmptyCitations` validation rejects empty tuples at Pydantic level.
3. `EventStore.append` rejects events whose payload fails revalidation.
4. Re-ingesting the same source yields identical SourceIds and ChunkIds.
5. Concurrent appends to the same aggregate with same `expected_version` — exactly one succeeds.
6. Projection rebuild from event log == live projection state.
7. Replay is byte-deterministic when external responses are captured.
8. `coordinator_inbox` ensures each event handled at most once per coordinator.
9. Aggregates never call IO (static check via `pylint` rule + reviewer guideline).
10. `model_construct` is banned outside `tests/` (`ruff` rule).

## 11. Sequencing & Migration

### 11.1 Implementation order

| Order | Module | Cross-team blocker |
|---|---|---|
| 1 | `eventstore/` + `messaging/` + `shared/` + `capture/blob_store.py` | none |
| 2 | `capture/http_client.py` + `capture/replay_mode.py` | none |
| 3 | `coordinators/runtime.py` + base inbox/outbox tables | none |
| 4 | `ingestion/intake/` (#12) + `IngestionCoordinator` + `ProjectsProjection` | none |
| 5 | `ingestion/parser/` (#13) — PyMuPDF in subprocess worker | none |
| 6 | `ingestion/discovery/` (#14) — GitHub + PWC adapters first | none |
| 7 | `coordinators/indexing_coordinator.py` + `coordinators/embedding_coordinator.py` | none |
| 8 | `context/indexer/` (#15) — chunkers, indexer service, source/embedding events | uses placeholder #8 schemas; clean swap when #8 lands |
| 9 | `context/workspace/` (#16) — `Cited[T]`, scope, lookup tool | uses placeholder #8 schemas + #9 SQLite shim; clean swap |
| 10 | `coordinators/workspace_ready_coordinator.py` (subscribes to AgentRegistry events from #10/#2) | depends on #10's AgentRegistered event existing; can be stubbed against contract |
| 11 | `capture/llm_client.py` + `capture/embedding_client.py` + `capture/chroma_client.py` + `capture/web_search_client.py` + `capture/notebook_lm_client.py` | none |
| 12 | Remaining projections (`CitationGraph`, `EventTimeline`, `OutboxBacklog`) | none |
| 13 | Crash-injection test suite | gates production declaration |
| 14 | Security hardening pass (parser worker isolation, model digest pinning, header redaction in CI) | gates production declaration |

### 11.2 Migration when #8 lands

`NonEmptyCitations` and other shared types relocate to the canonical schema package. Our event payload classes import from there. Estimated cost: 1 day (the registry contract makes this mechanical; migration is gated by upcaster golden-fixture tests passing).

### 11.3 Migration when #9 lands

If #9's repository abstracts over SQLite the same way ours does, replace projection persistence with #9 (event store keeps its own schema; event sourcing schemas don't fit a generic repo). Estimated cost: 1.5 days including tests.

### 11.4 Move beyond SQLite

Triggered by §4.3. Effort:
- EventStoreDB: ~1 week to port `EventStore` and `Subscription` Protocols + golden replay tests.
- Postgres: ~1.5 weeks (we lose persistent subscriptions but gain SQL ops); add a notification bridge (LISTEN/NOTIFY).
- NATS JetStream: only as a fan-out bus paired with one of the above.

## 12. Open Questions

| # | Question | Default |
|---|---|---|
| OQ1 | Default embedding model | `sentence-transformers/all-MiniLM-L6-v2`; pluggable; digest pinned |
| OQ2 | Default snapshot threshold | start at 100ms aggregate replay; tune with metrics |
| OQ3 | Issue/discussion thread chunking | one chunk per top-level body; one per comment |
| OQ4 | Event retention policy | forever; revisit at 100M events |
| OQ5 | Projection storage | same SQLite file as event store; separate read-DB later |
| OQ6 | Captured-header encryption-by-default vs redact-only | redact-only by default; encrypt for `embedding`/`llm`/`notebook_lm` surfaces (these may carry user data) |
| OQ7 | Idempotency retention | 30 days |
| OQ8 | Dead-letter retention | forever (operational evidence) |
| OQ9 | Trust scoring rules for `ArtifactTrust` | initial heuristic in `discovery/trust.py`; promote to config when we have real data |
| OQ10 | When does AgentRegistry promote a variable to `GLOBAL_VERIFIED`? | on receipt of a passing Method Fidelity + Data & Metrics + Artifact verifier triple, with Supervisor binding |

## 13. Out-of-Scope

- Agent runtime / Docker sandbox — separate umbrella.
- Agent orchestrator + spawn policy — separate umbrella.
- Verifier team and improvement agents — they consume our events; they live elsewhere.
- Frontend rendering — subscribes to projections + firehose; not built here.
- Six REPL tools beyond `lookup` and `semantic_search` — Protocol slots ready; impls land later.
- Knowledge graph (Graphify) — Phase 2 per PRD.

## 14. Acceptance Criteria

These are **requirements**, not a checklist. Each must have explicit tests.

- [ ] `python -m openresearch.cli register arxiv://1707.06347` triggers full pipeline; `WorkspaceReady` is observed for the default agent set, witnessed in `EventTimelineProjection`. Test: `tests/e2e/test_arxiv_to_workspace_ready.py`.
- [ ] Re-ingesting the same source emits identical `SourceRegistered`/`ChunkCreated`/`ChunkEmbedded` events. Test: `tests/e2e/test_idempotent_reingest.py`.
- [ ] One discovery adapter raising `RuntimeError` produces `AdapterFailed` and `DiscoveryCompleted`; `IndexingStarted` still fires. Test: `tests/integration/test_discovery_isolation.py`.
- [ ] `Cited[T](value=x, citations=())` raises `CitationMissingError`. Test: `tests/unit/test_cited_invariant.py`.
- [ ] `VariableEnriched(citations=())` fails Pydantic validation. Test: `tests/unit/test_event_validators.py`.
- [ ] `EventStore.append([VariableEnriched_with_empty_citations])` raises `AppendError`. Test: `tests/integration/test_append_revalidation.py`.
- [ ] Replaying captured event log on fresh store produces byte-identical store. Test: `tests/replay/test_byte_identical_replay.py`.
- [ ] Killing parser worker mid-flow → `ParsingFailed(retryable=True)` → `RetryCoordinator` re-issues → second attempt completes. Test: `tests/integration/test_parser_crash_retry.py`.
- [ ] Killing the orchestrator after `coordinator_outbox` write but before dispatch → outbox dispatcher recovers on restart and emits the queued command. Test: `tests/crash/test_outbox_recovery.py`.
- [ ] Projection rebuild from `EventStore.load_global()` produces state byte-identical to live projection. Test: `tests/integration/test_projection_rebuild.py`.
- [ ] Captured `Authorization` header never appears in any projection table. Test: `tests/security/test_no_credential_leak.py`.
- [ ] Loaded embedding model digest mismatch aborts startup. Test: `tests/security/test_model_digest_pin.py`.
- [ ] PDF parser running in subprocess; killed on timeout; emits `ParsingFailed(cause_kind="worker_killed")`. Test: `tests/security/test_parser_isolation.py`.
- [ ] `mypy --strict` passes. Test: CI gate.
- [ ] OpenTelemetry traces show full causation chain end-to-end. Test: `tests/observability/test_trace_propagation.py`.
- [ ] Prometheus metrics exposed at `/metrics` and contain all metric names listed in §8.7. Test: `tests/observability/test_metrics_exposed.py`.

## 15. Cross-Team Contracts and Boundaries

This spec depends on contracts owned by other teammates. Our work must compose without drift.

### 15.1 Imports from #8 (canonical schemas — armaanamatya)

| Imported | Used in | Notes |
|---|---|---|
| `EventEnvelope` (correlation_id, causation_id, occurred_at, schema_version, source) | every domain event | We declare event payloads; **#8 owns the envelope shape** |
| `Citation` | every citation-bearing event | Owned by #8 |
| `NonEmptyCitations = Annotated[tuple[Citation, ...], Field(min_length=1)]` | `VariableLoaded`, `VariableEnriched`, `CitationAttached`, `ToolInvoked`, etc. | Single canonical type; no per-event ad hoc annotations |
| `TaskStatus` enum (`created`, `context_prepared`, `running`, `artifact_submitted`, `verification_pending`, `verified`, `failed`, `blocked_requires_human`) | task references | We do not redefine; we reference |
| Event-stream payload contract for `task updates`, `context enrichment`, `citations`, `artifact submissions` | every event we publish to the dashboard firehose | We register our domain events against #8's payload schema; if the shapes diverge, we open a coordination PR |

### 15.2 Imports from #11 (blackboard scopes — armaanamatya)

| Imported | Used in | Notes |
|---|---|---|
| `Scope` enum (`private_to_parent`, `branch_shared`, `global_verified`) | `WorkspaceCreated`, `VariableLoaded`, `VariableEnriched`, `VariablePromoted` | **Same enum** across blackboard and workspace; we import, not redeclare |
| Blackboard publish API | a `BlackboardPublishCoordinator` (consumes `VariablePromoted` events) | Promotion to `BRANCH_SHARED` or `GLOBAL_VERIFIED` triggers a blackboard publish via #11's API; we do **not** duplicate blackboard storage |
| Delegation tree IDs (`task_id`, `parent_task_id`, `branch_id`) | `WorkspaceCreated.task_id`, `parent_task_id`, `branch_id` | Foreign keys into #11's delegation tree |

### 15.3 Imports from #10 (task lifecycle — armaanamatya)

| Imported | Used in | Notes |
|---|---|---|
| `TaskId`, `agent_task` identity | foreign key on `Workspace*` events | We carry foreign keys, never duplicate the lifecycle |
| `AgentRegistered` / `AgentDeregistered` events (or equivalent) | `WorkspaceReadyCoordinator` subscription | We subscribe to learn which agents need workspaces built |

We do **not** own any module called `registry/`. The agent registry concept belongs to #2/#10.

### 15.4 Boundary with #9 (SQLite repository — armaanamatya)

- **Our event store is its own SQLite file** (`runs/events.db`) with its own schema. It does **not** use #9's repository abstraction; event-sourcing schemas don't fit a generic CRUD repo.
- **Projections** that materialize entities #9 owns (`messages`, `artifacts`, `runs`, `verifications`) MAY be implemented by writing through #9's repository when those projections come online. Default is to keep our projection storage separate to avoid blocking on #9.
- We expose our event firehose (`EventStore.subscribe`) so #9-owned services or any other consumer can derive their own state.

### 15.5 Boundary with #7 (backend app skeleton — armaanamatya)

- The module layout in §3.6 slots **under #7's chosen app skeleton**. Final paths may shift (e.g., `src/openresearch/...` vs `openresearch/...`) once #7 lands; we will rebase rather than redefine.
- We adopt #7's config-loading pattern; `pydantic-settings` is the agreed default.
- Our `shared/observability.py` integrates with #7's structured logger if one exists; otherwise we ship structlog and #7 adopts.

### 15.6 Boundary with #2 / #11 (orchestrator + blackboard — armaanamatya)

- We do **not** implement orchestrator state. Our coordinators (§3.5) coordinate **our own pipeline**, not agent task spawn/lifecycle.
- When a workspace variable is promoted to `BRANCH_SHARED` or `GLOBAL_VERIFIED`, our `BlackboardPublishCoordinator` calls #11's blackboard publish API. We do not write blackboard records directly.
- Spawn-policy guards (max depth, max fan-out) are **#11's responsibility**. We respect their `agent_task` lineage; we do not enforce policy ourselves.

### 15.7 Boundary with #5 / #17 (LocalDocker sandbox — rishi-golla)

- Out of scope for our spec. We do not define `RuntimeBackend` and we do not own the Docker lifecycle.

### 15.8 Boundary with #18 (experiment runner — rishi-golla)

- Out of scope. Our pipeline ends at `WorkspaceReady`. The runner consumes downstream task records, not our events.

### 15.9 Boundary with #19 (provenance manifests — rishi-golla)

- **Citations and provenance are different concepts** that may cross-reference:
  - A `Citation.source_id` may point to a `SourceRef` (paper section, repo file, issue thread) — owned by us.
  - A `Citation.source_id` may also point to an `Artifact` recorded by #19's provenance manifest (e.g., "metric was produced by this captured run"). We add a `SourceKind.PROVENANCE_ARTIFACT` for this case, and the `SourceRef.locator` carries #19's artifact ID.
- Coordination needed with #19 owner: agree on the artifact ID format and lookup interface so our citation system can link to their provenance records without duplication.

### 15.10 Boundary with #20 / #21 (Next.js dashboard — rishi-golla)

- Dashboard subscribes to:
  1. Our event firehose (via `EventTimelineProjection` and the `/events/stream` SSE endpoint owned by #20).
  2. Our citation graph (`CitationGraphProjection`).
  3. Our workspace projection for variable views.
- All event payloads conform to #8's contract so the dashboard's mock event adapter (#20) can simulate them deterministically.
- Headers and other potentially-sensitive fields are **redacted before they leave our process** (§8.10), so the dashboard never sees credentials.

### 15.11 Where we ship contracts back

Some types in this spec are ours to define and they need to be visible to teammates:

| Contract | Owned by | Consumers |
|---|---|---|
| `SourceRef`, `Chunk`, `SourceKind`, `ChunkType` | us | #21 dashboard (citation panel), #19 may reference for cross-link |
| `Workspace*` event payloads | us (registered via #8 envelope) | #21 dashboard, verifiers |
| `ExternalApiCalled` and other capture events | us | replay tooling, audit views in #21 |
| `Cited[T]` materialized view | us | agent runtime (out of scope here) |

We surface these via a small `openresearch.contracts.ingestion_context` module checked in alongside our code; teammates import from there.

### 15.12 Update protocol when upstream contracts change

1. Open a PR against this spec referencing the upstream change.
2. Add an upcaster (§4.7) if event payloads shift.
3. Run `tests/integration/test_cross_team_contracts.py` which validates we still consume the upstream shape correctly.
4. If incompatible, escalate; do not silently drift.

### 15.13 Integration deltas (post-`d5413d3` reality, locked)

Armaanamatya landed `d5413d3` implementing #7-#11. The implementation is **CRUD-with-broadcasts**, not the event-sourced canonical contract this spec assumed. We adopt **Option 2**: keep event sourcing as our internal canonical, integrate at the edges. These pins lock the integration:

**Persistence**

- We piggyback on their `backend.persistence.database.Database` and the configured `REPROLAB_DATABASE_URL` (default `sqlite:///reprolab.db`). **One DB file**, our tables alongside theirs, all named `event_store_*` / `projection_*` / `coordinator_*` to avoid collisions.
- Same SQLite connection (PRAGMAs already set: `journal_mode=WAL`, `foreign_keys=OFF`). We bring our own `executescript` for our schema.
- We do **not** introduce async sqlite. Everything sync. Foreshadowed §8.3 ("Asyncio everywhere") is replaced by **sync everywhere**, threading for parallel adapter execution. The `EventStore` Protocol's async signatures collapse to sync (`def append`, `def load`, etc.).

**Event publishing bridge**

- Our domain events live in our event store. To make them visible to teammates' dashboard (#20/#21):
  - On every `append`, a thin `EventPayloadBridge` translates the domain event into their `EventPayload` and emits it through `TaskLifecycleService`-style listeners (or a sibling `DomainEventBus`).
  - We use existing `EventType` values where they fit:
    - `agent_reasoning_step` for `ToolInvoked`, `VariableEnriched`
    - `context_enrichment` for `VariableLoaded`, `WorkspaceReady`
    - `shared_state_updated` for `VariablePromoted`
    - `agent_started` / `agent_completed` for workspace lifecycle
  - The full domain event id is included in `EventPayload.data["domain_event_id"]` so the dashboard can re-fetch the canonical event from our event store.
- We do **not** add new values to `backend.schemas.events.EventType`. That enum belongs to teammate; we keep our domain-event types in our own enum (`DomainEventType`) and bridge.

**Citation type**

- `backend.schemas.citations` is **created by us** as a local module:
  ```python
  class Citation(BaseModel):
      source_id: str
      chunk_id: str | None
      quote: str
      locator: str
      confidence: float = 1.0

  NonEmptyCitations = Annotated[tuple[Citation, ...], Field(min_length=1)]
  ```
- We propose this upstream to #8 in a follow-up PR. Until then, both halves of the codebase import from `backend.schemas.citations`.

**Scope type**

- We define `backend.schemas.scope.Scope` as a typed enum with the three values `private_to_parent` / `branch_shared` / `global_verified`. It is structurally compatible with the `scope: str` field on their `BlackboardRecord` — we coerce on the bridge edge. We propose upgrading `BlackboardRecord.scope` to `Scope` upstream.

**Module layout (locked)**

```
backend/
├── eventstore/                          # NEW (us)
├── messaging/                           # NEW (us): envelope, command, event base, idempotency, bus
├── coordinators/                        # NEW (us)
├── capture/                             # NEW (us)
├── persistence/                         # THEIRS — we use Database from here
├── schemas/
│   ├── citations.py                     # NEW (us; propose upstream to #8)
│   ├── scope.py                         # NEW (us; propose upstream to #11)
│   └── (their existing files)
├── services/
│   ├── ingestion/                       # WAS empty stub; WE FILL: intake/, parser/, discovery/
│   ├── context/                         # WAS empty stub; WE FILL: indexer/, workspace/
│   ├── orchestration/                   # THEIRS
│   ├── runtime/                         # rishi-golla
│   └── verification/                    # not in scope
└── (other their files)
```

**Pyproject deps to add** (we'll PR these into the existing `pyproject.toml`):

- `pymupdf>=1.24` (parser)
- `chromadb>=0.5` (semantic search) — optional, behind a flag for the first slice
- `sentence-transformers>=2.7` (embeddings) — optional
- `tenacity>=8` (retry policy)
- `structlog>=24` (structured logging)
- `ulid-py>=1.1` (event ids)

Optional extras gate the heavy ones (`reprolab-backend[ingestion]`, `[context]`).

**Tests live where their tests live**

- `tests/test_eventstore_*.py`, `tests/test_intake_*.py`, etc., at repo root. Same conftest convention they used.

**Fully-sync EventStore Protocol (revised)**

```python
class EventStore(Protocol):
    @property
    def capabilities(self) -> StoreCapabilities: ...

    def append(
        self,
        aggregate_id: AggregateId,
        aggregate_type: str,
        events: Sequence[DomainEvent],
        expected_version: int,
        envelope: EventEnvelope,
    ) -> AppendResult: ...

    def load(self, aggregate_id: AggregateId, from_version: int = 0) -> Iterator[StoredEvent]: ...
    def load_global(self, from_position: int = 0, to_position: int | None = None,
                    types: Iterable[str] | None = None, batch_size: int = 1000) -> Iterator[StoredEvent]: ...
    def subscribe(self, subscription_name: str, types: Iterable[str] | None = None) -> Subscription: ...
    def get_aggregate_version(self, aggregate_id: AggregateId) -> int: ...
```

`Subscription` becomes a sync iterator with sync `ack`/`nack`. Threading handles concurrent subscriptions.

**Why these deltas don't break the architecture**

- Event sourcing's *correctness* properties (immutability, replay, audit, idempotency, causation chains) are paradigm-level — they don't depend on async or on owning the database connection.
- The thin bridge to their `EventPayload` means teammates and the dashboard see "events" they recognize; replay determinism (§8.5) and the citation invariant (§5.6) remain intact in our store.
- The `Citation` type lives in our codebase right now but its shape is the canonical one; if/when #8 absorbs it, we re-export and remove the local definition. No data migration.

## 16. Appendix — File-by-File Summary

```
eventstore/interface.py                 EventStore + Subscription + StoreCapabilities Protocols
eventstore/sqlite_store.py              SQLiteEventStore (production default)
eventstore/jsonl_store.py               JsonlEventStore (debug, ops dump)
eventstore/snapshot.py                  Snapshot + SnapshotEnvelope + integrity checks
eventstore/upcaster.py                  Upcaster + UpcasterRegistry + autodiscover
eventstore/subscription.py              Persistent subscription with checkpoint/ack/nack/lease
eventstore/replay.py                    Bulk replay + determinism harness

messaging/command.py                    Command + CommandDispatcher
messaging/event.py                      DomainEvent + StoredEvent + EventEnvelope
messaging/idempotency.py                CommandIdempotencyTable
messaging/bus.py                        InProcMessageBus, NatsMessageBus (slot-in)

coordinators/runtime.py                 CoordinatorRuntime (inbox + state + outbox)
coordinators/ingestion_coordinator.py
coordinators/indexing_coordinator.py
coordinators/workspace_ready_coordinator.py
coordinators/retry_coordinator.py
coordinators/embedding_coordinator.py

ingestion/intake/{commands,events,aggregate,service}.py
ingestion/intake/fetchers/{pdf,arxiv,doi}.py
ingestion/intake/projections.py

ingestion/parser/{commands,events,aggregate,service}.py
ingestion/parser/{pymupdf_parser,nougat_parser}.py
ingestion/parser/workers/{pymupdf_worker,nougat_worker}.py    # subprocess isolation
ingestion/parser/projections.py

ingestion/discovery/{commands,events,aggregate,service,trust}.py
ingestion/discovery/adapters/{github,papers_with_code,huggingface,semantic_scholar}.py
ingestion/discovery/projections.py

context/indexer/{commands,events,aggregate,service,embedder}.py
context/indexer/chunkers/{section,paragraph,code_block}.py
context/indexer/projections.py                                # SourcesProjection, SemanticIndexProjection (with semantic_index_log)

context/workspace/{commands,events,aggregate,service,model}.py
context/workspace/tools/{interface,lookup,semantic_search}.py
context/workspace/projections.py                              # WorkspaceProjection, CitationGraphProjection

capture/{http_client,llm_client,embedding_client,chroma_client,web_search_client,notebook_lm_client}.py
capture/blob_store.py                  Atomic blob write + GC + encryption
capture/replay_mode.py                 ReplayInterceptor

shared/ids.py                          Composed content-addressed IDs
shared/envelope.py                     EventEnvelope helpers
shared/errors.py                       OpenResearchError hierarchy
shared/observability.py                structlog + OpenTelemetry helpers
shared/security.py                     HeaderRedactor + SecretEncryptor + ModelDigestVerifier
shared/config.py                       pydantic-settings Settings
```

---

**End of design (revision 2).**
