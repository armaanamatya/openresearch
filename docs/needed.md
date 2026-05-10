# RLM Paper Usage & Memory Layer Analysis

## How the RLM Paper Is Used

The **RLM paper** (arXiv:2512.24601, Zhang/Kraska/Khattab, MIT CSAIL) is the **conceptual foundation for the context management layer** in ReproLab. It was integrated during chat session #11 of the PRD planning (see `docs/chathistory.md:750-812`).

### Core Idea Borrowed From RLM

Instead of stuffing context into prompts or embedding everything in a vector store, store context as **typed Python variables in per-agent workspaces**. Agents programmatically explore, filter, and drill into these variables rather than doing retrieval-by-embedding. This gives ~2-3k tokens per query vs 95k+ for naive prompt stuffing.

### Three-Layer Context Strategy (PRD Design)

| Layer | Role | Status |
|-------|------|--------|
| **Layer 1 — RLM workspace** (primary) | Variables + programmatic exploration. Precise, structured queries. | **Complete** — full event-sourced workspace with variable preloading, progressive enrichment, scope promotion, 5 workspace tools, `Cited[T]` invariant |
| **Layer 2 — Semantic search** (fallback) | Fuzzy similarity, discovery of things the agent didn't know to look for | **Complete** — `SemanticSearchTool` with Chroma vector embeddings (all-MiniLM-L6-v2) + BM25 fallback. `ChromaEmbeddingStore` in `backend/services/context/semantic/`. Optional dep: `pip install reprolab-backend[semantic]` |
| **Layer 3 — Knowledge Graph / Graphify** | Structural code/doc navigation via AST | **Phase 2**, not yet implemented |

### What's Actually Built

The RLM influence manifests in these concrete implementations:

- **`backend/services/context/workspace/`** — Per-agent workspaces where context lives as named variables (`paper_text`, `paper_sections`, `claim_map`), not as embedded chunks.
- **`Cited[T]`** (`workspace/model.py`) — The architectural keystone: every variable and every tool result is paired with mandatory citations. You literally cannot construct a `Cited[T]` without evidence. Four-layer defense chain.
- **7 domain events** — `WorkspaceCreated`, `VariableLoaded`, `VariableEnriched`, `VariablePromoted`, `CitationAttached`, `ToolInvoked`, `WorkspaceReady`, `WorkspaceClosed`. Full lifecycle including scope promotion.
- **Progressive enrichment** — `enrich_variable()` lets agents write structured outputs back as workspace variables for downstream agents.
- **Scope promotion** — `promote_variable()` transitions visibility (private_to_parent -> branch_shared -> global_verified).
- **5 workspace tools**:
  - `LookupTool` — Exact source lookup (Layer 1: structured, precise).
  - `SemanticSearchTool` — BM25 lexical fallback (Layer 2: fuzzy discovery).
  - `ListVariablesTool` — Discover available variables and metadata.
  - `InspectVariableTool` — Deep-dive into a specific variable's value and citations.
  - `RlmQueryTool` — Core RLM capability: recursive LLM sub-query over a context segment (~2-3k tokens per query vs 95k+ for naive prompt stuffing). Uses `LlmClient` protocol for provider-agnostic integration.
- **Workspace lifecycle** — `build_workspace()`, `enrich_variable()`, `promote_variable()`, `close_workspace()`, `materialize_view()`.

The RLM paper's influence is **architectural**, not a code dependency — no `import rlm` anywhere.

---

## Memory Layer

### Current State: No Memory Layer Exists

The current system has no persistent memory across projects or sessions. Each workspace is scoped to one `(project_id, agent_name)` pair and lives only for that reproduction run.

### What the System Does Have (Memory-Adjacent)

- **Progressive enrichment** within a single run (agents build on each other's outputs via `enrich_variable()`).
- **Event-sourced audit trail** (full replay of how context evolved).
- **Scope promotion** — variables can be promoted from private to shared to global via `promote_variable()`.
- **Shared Memory Objects** in the PRD design (Claim Map, Assumption Ledger, Experiment Ledger, etc.).

### What the System Does Not Have

- **Cross-project memory** — learning from reproducing Paper A to help with Paper B.
- **Episodic memory** — "last time we saw a PyTorch 1.x dependency, we had to do X."
- **Domain knowledge accumulation** — building a growing understanding of a field over time.
- **Strategy meta-learning** — improving reproduction heuristics from past successes/failures.

### Why a Memory Layer Would Be Valuable

It is a natural fit for the RLM workspace architecture. A memory layer could be:

1. **A fourth context layer** alongside RLM/Semantic/Graph — a persistent store of `MemoryVariable` events from past runs that get selectively loaded into new workspaces.
2. **Event-sourced** (same pattern as everything else) — `MemoryRecorded`, `MemoryRetrieved`, `MemoryDecayed` events.
3. **Citation-graded** — memories carry citations back to the original run that produced them, maintaining the `Cited[T]` invariant.

### Concrete Things a Memory Layer Could Remember

- Environment resolution patterns ("PyTorch 1.x papers usually need CUDA 11.x").
- Common paper ambiguities and how they were resolved.
- Dead ends and why they failed.
- Successful improvement strategies by paper type.
- Dataset/benchmark quirks.

### Where It Would Live

It would slot naturally between the workspace service and the indexer as a `backend/services/context/memory/` module.
