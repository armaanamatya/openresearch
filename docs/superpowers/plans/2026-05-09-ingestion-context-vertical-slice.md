# Implementation Plan — Ingestion + Context Vertical Slice

| Field | Value |
|---|---|
| Date | 2026-05-09 |
| Owner | lolout1 |
| Closes | #12 (partial — PDF only, arXiv/DOI deferred), #13, #15 (partial — section chunker only), #16 (partial — lookup tool only) |
| Defers | #14 (discovery), embeddings + semantic_search, multi-paper indexing |
| Spec | `docs/superpowers/specs/2026-05-09-ingestion-context-design.md` |

## 1. Goal

Stand up a runnable end-to-end ingestion+context vertical slice that takes a local PDF and produces a queryable Workspace with citation-bearing variables. Every layer of the spec is exercised with the *minimum* implementation needed for a credible demo:

```
$ python -m backend.cli ingest tests/fixtures/papers/ppo.pdf
project_id=prj_..., parsed=42 sections, sources=42, chunks=152, workspace=ready
$ python -m backend.cli inspect prj_... --variable claim_map
{... materialized Cited[T] dump ...}
```

The slice is intentionally narrow: one source kind (PdfPath), one parser (PyMuPDF), one chunker (SectionChunker), one tool (lookup). Discovery, embeddings, and the other parsers/tools are deferred — they plug into the same Protocols later without changing what we ship here.

## 2. Approach

Five sequential commits, each independently mergeable, each with passing tests + `mypy --strict` clean. Coordinator-free for this slice — the CLI wires the pipeline directly. Coordinators come in later when concurrency matters.

```
Commit 1: Intake (#12)
  Project aggregate + events + IntakeAppService + PdfPathFetcher
Commit 2: Parser (#13)
  ParsedPaper aggregate + events + ParserAppService + PyMuPdfParser
Commit 3: Indexer (#15)
  Index aggregate + events + IndexerAppService + SectionChunker
Commit 4: Workspace (#16)
  Workspace aggregate + events + WorkspaceAppService + LookupTool + Cited[T] materialization
Commit 5: CLI + end-to-end smoke
  python -m backend.cli ingest <pdf>; full-pipeline integration test
```

Each commit is a complete vertical of: aggregate (pure state machine) → events (registered Pydantic models) → application service (IO + append) → tests (aggregate unit + service integration + schema round-trip).

## 3. Module Layout (under teammate's `backend/`)

```
backend/
├── services/
│   ├── ingestion/                     # already-empty stub from #7
│   │   ├── intake/
│   │   │   ├── __init__.py
│   │   │   ├── sources.py             # PaperSource discriminated union (PdfPath only this slice)
│   │   │   ├── events.py              # ProjectCreated, PaperFetched, PaperFetchFailed, MetadataExtracted
│   │   │   ├── aggregate.py           # ProjectAggregate (state machine; no IO)
│   │   │   ├── service.py             # IntakeAppService (IO + append)
│   │   │   └── fetchers/
│   │   │       ├── __init__.py
│   │   │       └── pdf_path.py        # PdfPathFetcher
│   │   └── parser/
│   │       ├── __init__.py
│   │       ├── events.py              # ParsingStarted, SectionExtracted, ReferenceExtracted, ParsingCompleted, ParsingFailed
│   │       ├── model.py               # Section, Reference (frozen dataclasses)
│   │       ├── aggregate.py           # ParsedPaperAggregate
│   │       ├── service.py             # ParserAppService
│   │       └── pymupdf_parser.py      # PyMuPdfParser
│   └── context/                       # already-empty stub from #7
│       ├── indexer/
│       │   ├── __init__.py
│       │   ├── events.py              # IndexingStarted, SourceRegistered, ChunkCreated, IndexingCompleted, IndexingFailed
│       │   ├── model.py               # SourceRef, Chunk, SourceKind, ChunkType
│       │   ├── aggregate.py           # IndexAggregate
│       │   ├── service.py             # IndexerAppService
│       │   └── chunkers/
│       │       ├── __init__.py
│       │       └── section.py         # SectionChunker
│       └── workspace/
│           ├── __init__.py
│           ├── events.py              # WorkspaceCreated, VariableLoaded, ToolInvoked, WorkspaceReady
│           ├── model.py               # Cited[T] re-export, Workspace projection type
│           ├── aggregate.py           # WorkspaceAggregate
│           ├── service.py             # WorkspaceAppService
│           ├── tools/
│           │   ├── __init__.py
│           │   ├── interface.py       # WorkspaceTool Protocol
│           │   └── lookup.py          # LookupTool
│           └── projections.py         # WorkspaceProjection (in-memory; rebuilds from events)
└── cli.py                             # python -m backend.cli ingest|inspect
```

## 4. Per-Issue Implementation

### Commit 1 — Intake (#12, partial)

**Aggregate state machine**

```python
class ProjectState(StrEnum):
    NEW = "new"
    REGISTERED = "registered"
    FETCHED = "fetched"
    METADATA_KNOWN = "metadata_known"
```

Transitions: `NEW -register-> REGISTERED -fetch-> FETCHED -extract_metadata-> METADATA_KNOWN`.
Aggregate validates transitions and emits events; **never calls IO**.

**Events** (all `@register_event` and Pydantic):
- `ProjectCreated(project_id, source)` — schema_version=1
- `PaperFetched(project_id, raw_paper_path, pdf_sha256, pdf_size_bytes)`
- `PaperFetchFailed(project_id, cause_kind, cause_message, retryable)`
- `MetadataExtracted(project_id, metadata)` — `PaperMetadata` frozen dataclass

**`IntakeAppService.handle_register_project(cmd)`** opens an aggregate by deterministic `project_id = "prj_" + sha256(source_kind + source_locator)[:16]` so re-registering the same PDF is idempotent. Calls `ProjectAggregate.handle_register` for the state-transition events, then appends to the event store with optimistic concurrency.

**`IntakeAppService.handle_fetch_paper(cmd)`** loads aggregate state, calls `PdfPathFetcher.fetch()` (the only IO), appends `PaperFetched` (success) or `PaperFetchFailed` (failure) accordingly. Idempotency via the `command_idempotency` table for `(project_id, command_id)`.

**`PdfPathFetcher`**: copies file into `runs/{project_id}/raw_paper.pdf`, computes sha256, validates it's a PDF (`%PDF-` magic), records size.

**Tests**:
- `tests/test_issue12_intake_aggregate.py` — pure aggregate state-machine tests (every transition valid/invalid)
- `tests/test_issue12_intake_service.py` — integration with SqliteEventStore: register → fetch → re-register is idempotent
- Test that re-issuing the same `RegisterProject` command returns the same project_id without writing duplicate events
- Test that `fetch_paper` failure emits `PaperFetchFailed` with retryable=True for transient errors

**Verification gate**: pytest green; mypy strict clean; aggregate handles every command without doing IO; service exclusively does IO and appends.

### Commit 2 — Parser (#13)

**`ParsedPaperAggregate`** lifecycle: `PENDING → PARSING → PARSED | FAILED`.

**Events**:
- `ParsingStarted(project_id, parser_name, parser_version)`
- `SectionExtracted(project_id, section_id, title, text, char_offset, parent_id, depth)` — one per section
- `ReferenceExtracted(project_id, reference_id, raw_text, arxiv_id, doi, title)`
- `ParsingCompleted(project_id, section_count, reference_count, parse_duration_ms, full_text_blob_path, full_text_sha256)`
- `ParsingFailed(project_id, parser_name, cause_kind, cause_message, retryable)`

**`PyMuPdfParser`** (subclass of `Parser` Protocol):
- Uses `fitz` (PyMuPDF) to extract text + layout.
- Section detection: regex on numbered headings (`^\d+(\.\d+)*\s+[A-Z]`) + known section names (Abstract, Introduction, Methods, Results, Discussion, References, Appendix). Falls back to "single section" if no headings match — robustness over precision for the slice.
- Reference parsing: simple regex for arXiv ID (`arXiv:\d{4}\.\d{4,5}`) and DOI (`10\.\d{4,9}/[^\s]+`) within the references section.
- Stores full concatenated text as a blob at `runs/{project_id}/parsed_full_text.txt`, sha256 referenced in `ParsingCompleted`.
- **Runs in-process for the slice.** Subprocess isolation (spec §8.10) is a follow-up commit.

**`ParserAppService.handle_start_parsing(cmd)`**:
1. Load `ProjectAggregate`; require state == FETCHED.
2. Load `ParsedPaperAggregate`; require state == PENDING.
3. Append `ParsingStarted`.
4. Invoke parser (the IO).
5. For each section / reference, append events (one per).
6. Append `ParsingCompleted` or `ParsingFailed`.

**Tests**:
- `tests/test_issue13_parser_aggregate.py` — pure state-machine tests
- `tests/test_issue13_parser_pymupdf.py` — feed a tiny synthetic 2-page PDF (generated by reportlab in conftest, or a tiny fixture), assert sections detected
- `tests/test_issue13_parser_service.py` — full service integration; idempotent re-parse
- Round-trip event test: every event in the parser stream re-validates via `StoredEvent.into()`

### Commit 3 — Indexer (#15, partial)

**`IndexAggregate`** lifecycle: `PENDING → INDEXING → INDEXED | FAILED`.

**Events**:
- `IndexingStarted(project_id, chunker_name, chunker_version)`
- `SourceRegistered(project_id, source: SourceRef)` — per source
- `ChunkCreated(project_id, chunk: Chunk)` — per chunk
- `IndexingCompleted(project_id, source_count, chunk_count, duration_ms)`
- `IndexingFailed(project_id, cause_kind, retryable)`

**Models** (frozen Pydantic):
```python
class SourceKind(StrEnum):
    paper_section = "paper_section"
    paper_reference = "paper_reference"
    # repo/issue/dataset later

class SourceRef(BaseModel):
    id: str  # composed: src_{sha256(project_id + source_kind + upstream_id)[:16]}
    project_id: str
    kind: SourceKind
    locator: str  # "PPO §3.2"
    upstream_id: str | None  # the section_id from parser

class ChunkType(StrEnum):
    section = "section"
    paragraph = "paragraph"

class Chunk(BaseModel):
    id: str  # composed: chk_{sha256(source_id + chunker_name + version + span + text)[:24]}
    source_id: str
    project_id: str
    text: str
    span: tuple[int, int]
    chunk_type: ChunkType
```

**`SectionChunker`**: one chunk per section (whole-section text). ParagraphChunker is deferred.

**`IndexerAppService.handle_start_indexing`**:
1. Load `ProjectAggregate`; require state == FETCHED or METADATA_KNOWN.
2. Load `ParsedPaperAggregate`; require state == PARSED.
3. Stream parsed events out of the event store, build `SourceRef`s for each section/reference, run `SectionChunker` over them, append `SourceRegistered` + `ChunkCreated` per result.
4. Append `IndexingCompleted` or `IndexingFailed`.

**Projection**: `SourcesProjection` (in-memory dict for slice; SQLite-backed later). Subscribes to `index` events.

**Tests**:
- Aggregate state-machine tests
- `tests/test_issue15_indexer_chunker.py` — SectionChunker on synthetic sections
- `tests/test_issue15_indexer_service.py` — service writes correct events; idempotent re-index produces identical SourceIds and ChunkIds (proves content-addressed ID composition with chunker version included)
- Sources projection rebuild from event log produces identical state to live projection

### Commit 4 — Workspace (#16, partial)

**`WorkspaceAggregate`** lifecycle: `CREATED → LOADING → READY → CLOSED`.

**Events**:
- `WorkspaceCreated(workspace_id, project_id, agent_name, parent_workspace_id, branch_id, task_id)`
- `VariableLoaded(workspace_id, variable_name, value_payload, citations: NonEmptyCitations, scope, source_agent)`
- `ToolInvoked(workspace_id, tool_name, arguments, result_payload, citations: NonEmptyCitations, duration_ms)`
- `WorkspaceReady(workspace_id, variable_count)`
- `WorkspaceClosed(workspace_id, reason)`

**Models**:
- `Cited[T]` re-imported from `backend.schemas.citations` (already exists).
- `Workspace` (the projection type — not a Pydantic model, a plain class with `.get(name) -> Cited[Any]`, `.search(query)`, `.call_tool(name, **kwargs) -> Cited[Any]`, `.variables` property).

**Tools**:
- `WorkspaceTool` Protocol with `name`, `call(workspace, **kwargs) -> Cited[Any]`.
- `LookupTool` — exact source lookup against the SourcesProjection. Returns `Cited[SourceRef]` whose citations are `(Citation(source_id=src.id, quote=src.locator, locator=src.locator),)`.

**`WorkspaceAppService.handle_build_workspace`**:
1. Load `ProjectAggregate`; require state == FETCHED or METADATA_KNOWN.
2. Load `IndexAggregate`; require state == INDEXED.
3. Append `WorkspaceCreated`.
4. Pre-load a basic `claim_map` variable from parsed sections (deterministic, no LLM in this slice).
5. Append `VariableLoaded`.
6. Append `WorkspaceReady`.

**Tests**:
- Aggregate state-machine tests
- `tests/test_issue16_workspace_service.py` — full path: build → query workspace.get("claim_map") → returns a `Cited` value with citations referencing real source_ids
- `LookupTool` tests
- WorkspaceProjection rebuild from events produces identical `Workspace` state

### Commit 5 — CLI + end-to-end smoke

**`backend/cli.py`** with two subcommands:
- `ingest <pdf-path>` — runs the full pipeline through the four services, prints the project_id and a summary.
- `inspect <project_id> [--variable VAR]` — opens the latest workspace for the project, prints all variables (or one) with their citations.

**Fixtures**:
- `tests/fixtures/papers/ppo.pdf` — a short real PPO paper PDF (≤ 1 MB; checked in). The same paper drives both unit tests and the e2e smoke.

**End-to-end smoke**: `tests/test_e2e_pdf_to_workspace.py`:
1. Runs `IntakeAppService → ParserAppService → IndexerAppService → WorkspaceAppService` in sequence on the fixture PDF.
2. Asserts: a `Workspace` is built; `claim_map` variable exists; `claim_map.citations` are non-empty and resolve to real `SourceRef`s in the projection; reproducing the run yields identical SourceIds and ChunkIds (replay determinism for the deterministic parts).

## 5. Cross-cutting (every commit)

- **mypy --strict** clean across new modules.
- **No new dependency** unless added to `pyproject.toml` in the same commit. PyMuPDF (`pymupdf>=1.24`) is added in commit 2 only.
- **Tests organized two ways**: per-issue files (`test_issue12_*.py`, `test_issue13_*.py`, …) for tracking against issue acceptance criteria; module-level files (`test_intake_*.py`, `test_parser_*.py`) for tight feedback loops. Mirroring teammate's `test_issueN_*.py` convention.
- **No coordinators in this slice**. The CLI wires the four services in sequence. When concurrency lands, swap the CLI for `IngestionCoordinator + IndexingCoordinator + WorkspaceReadyCoordinator` per spec §3.5.
- **Bridge to dashboard**: out of scope for this plan. The events flow into the event store; the bridge to `EventPayload` ships in a follow-up.

## 6. Acceptance Criteria

For the plan to be considered complete (every box must be a real test):

- [ ] `python -m backend.cli ingest tests/fixtures/papers/ppo.pdf` runs to completion; prints project_id, sources count, chunks count, workspace status.
- [ ] `python -m backend.cli inspect <project_id>` prints the materialized workspace including the `claim_map` variable's value and citations.
- [ ] All five commits' tests green (pytest); `mypy --strict` clean across all new modules.
- [ ] Re-running ingest on the same PDF yields the *same* `project_id`, *same* `SourceId`s, *same* `ChunkId`s. (Idempotent re-ingest test.)
- [ ] A `VariableLoaded` event with empty citations is rejected at the constructor (per-issue test in #16).
- [ ] An induced parser failure produces a `ParsingFailed(retryable=True)` event and leaves the project in `FETCHED` state for retry. (Per-issue test in #13.)
- [ ] Each per-issue test file references the issue number in its docstring.

## 7. Out-of-Scope (explicit, won't be built in this plan)

- arXiv / DOI fetchers (#12 — only PdfPath ships)
- Discovery adapters (#14 entirely)
- Embeddings + semantic_search (#15 — Chroma defers)
- Other parsers (Nougat) (#13 — PyMuPDF only)
- Other tools (graph_query, web_search, notebook_query, rlm_query) (#16 — lookup only)
- Coordinators / process managers — slice uses sequential CLI
- Dashboard event payload bridge — separate follow-up
- Subprocess isolation for the parser (spec §8.10) — follow-up

## 8. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| PyMuPDF section detection fails on unusual papers | Medium | Fall back to "single section = full document" so the pipeline never blocks on parse heuristics |
| ChunkId composition still allows collisions if normalization is wrong | Low | Property-based test (Hypothesis) on identity composition with random text |
| Re-running pipeline on same PDF produces different IDs due to non-deterministic source ordering | Medium | Sort sections by (depth, char_offset) before chunking |
| Tests slow because of real PyMuPDF parses | Low | Use a 2-page synthetic fixture for parser tests; only the e2e smoke uses the real PPO PDF |
| Disk usage from `runs/` blob storage | Low | `runs/` already in `.gitignore`; per-test fixtures use `tmp_path` |

## 9. Estimated Footprint

~2000 LOC of source + ~1500 LOC of tests across 5 commits. Per-commit budget:
- Commit 1 (intake): 350 LOC source + 250 tests
- Commit 2 (parser): 500 LOC source + 350 tests
- Commit 3 (indexer): 400 LOC source + 300 tests
- Commit 4 (workspace): 500 LOC source + 400 tests
- Commit 5 (CLI + e2e): 250 LOC source + 200 tests

## 10. Plan Self-Review (placeholder / contradiction / scope check)

- No "TBD" or "TODO" left.
- Aggregate purity is consistently enforced across all four aggregates (state-machine only, no IO).
- The slice is one cohesive unit (PDF → Workspace). Discovery is correctly excluded — it would expand scope without enabling the CLI demo.
- Every acceptance criterion has a corresponding test file referenced.

---

**End of plan.**
