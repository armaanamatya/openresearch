# RLM & Hermes — Visualized

How the two "smart" subsystems plug into the ReproLab pipeline:

- **RLM** (`backend/services/context/workspace/tools/rlm_query.py`) — a *tool* agents use to ask focused questions over very large workspace variables (e.g. extracted paper text), via recursive LLM decomposition.
- **Hermes audit** (`backend/hermes_audit/`) — a *separate auditor* called after each agent step and verification checkpoint, with a self-learning **provider-routing** chain (Nous-Hermes → Claude → SDK → OpenAI → Codex CLI).

Neither is a "free-roaming agent that picks any LLM." Both are scoped: RLM fans out one configured LLM recursively; Hermes routes one audit prompt to one provider per call.

---

## 1. Where RLM and Hermes sit in the 14-stage pipeline

```
                         ┌──────────────────────────────────────────────┐
                         │   ReproLabOrchestrator.run() — Python loop   │
                         └──────────────────────────────────────────────┘
                                              │
                                              ▼
   stage 1   ingest                ┌──────────────────────┐
   stage 2   paper_understood ───► │  agent (one runtime: │ ─── invokes tools ──► RlmQueryTool
   stage 3   artifacts_discovered  │  Anthropic OR OpenAI)│                       (recursive LLM
   stage 4   environment_built     └──────────────────────┘                        over paper_text)
   stage 5   plan_created                       │                                       │
   stage 6   GATE 1  ◄──── Hermes checkpoint audit (downgrade?) ──┐                     │
   stage 7   baseline_implemented               │                 │                     │
   stage 8   baseline_run                       │                 │   uses workspace
   stage 9   GATE 2  ◄──── Hermes checkpoint audit ───────────────┤   variables + citations
   stage 10  improvements_selected              │                 │                     │
   stage 11  improvements_run                   │                 │                     │
   stage 12  GATE 3  ◄──── Hermes checkpoint audit ───────────────┤                     │
   stage 13  research_map_generated             │                 │                     ▼
   stage 14  complete                           ▼                 │            Cited[answer]
                                       Hermes step audit ─────────┘            (provenance preserved)
                                       (after every agent step,
                                        scope = HermesAuditScope.step)
```

- `PipelineStage` enum: `backend/agents/orchestrator.py` lines ~150–172
- Hermes hooks: `_append_hermes_report`, `_apply_checkpoint_report_to_gate` (same file, ~lines 862–910)
- Step/checkpoint payloads: `backend/hermes_audit/payloads.py`

---

## 2. RLM — Recursive Language Model tool

### 2.1 Why it exists
A paper PDF after PyMuPDF extraction can be hundreds of KB. Stuffing it whole into every agent prompt is wasteful and may exceed context. The Zhang/Kraska/Khattab RLM paradigm (arXiv 2512.24601) treats the variable as an external environment: the LLM recursively examines slices instead of seeing it all at once.

### 2.2 Algorithm (faithful to `rlm_query.py`)

```
recursive_query(content, question, depth):
    if len(content) ≤ leaf_budget:             ← base case 1: small enough → 1 LLM call
        return llm_answer(content, question)

    if depth ≥ max_depth:                      ← base case 2: stop recursing
        return llm_answer(truncate(content), question)

    chunks = chunk(content, chunk_size)
    if selection_enabled and len(chunks) > top_k:
        picked = llm_select(chunks, question, top_k)   ← routing LLM call
    else:
        picked = all chunks

    sub_answers = [ recursive_query(c, question, depth+1) for c in picked ]
    return llm_aggregate(question, sub_answers)         ← synthesis LLM call
```

### 2.3 Recursion shape (depth 2, top_k=3, picking 2 of 4 chunks)

```
                              recursive_query(paper_text, Q, depth=0)
                                          │
                                  len > leaf_budget → chunk
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                       llm_select(4 chunks, Q)   (picks idx [0, 2])
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
       recursive_query(chunk0, Q, 1)   recursive_query(chunk2, Q, 1)
                │                           │
        len > leaf_budget → chunk    len ≤ leaf_budget → leaf
                │                           │
        llm_select(...) → [a]               └─► llm_answer(chunk2, Q)   ★ LEAF
                │
                ▼
        recursive_query(sub-a, Q, 2)
                │
        depth = max_depth → truncate & answer  ★ LEAF
                │
                ▼
        llm_aggregate(question, [leaf-a])   ← synthesis at depth 1

                          ▼
        llm_aggregate(question, [agg-from-depth1, leaf-chunk2])   ← synthesis at depth 0
                          │
                          ▼
              Cited[ { answer, depth_reached,
                       llm_calls, chunks_examined,
                       selection_path, … } ]
                       (citations propagated from the
                        workspace variable’s provenance)
```

### 2.4 Defaults & safety rails

| Knob | Default | Why |
|---|---|---|
| `leaf_budget` | 12 000 chars | One LLM call comfortably fits a normal paper section |
| `chunk_size` | 12 000 chars | Matches leaf budget so chunks are themselves leaf-sized |
| `max_depth` | 3 | Hard cap on recursion tree |
| `selection_top_k` | 5 | Only the most relevant chunks recurse |
| `max_llm_calls` | 24 | Total LLM-call budget per `call()` |
| `selection_enabled` | True | Skip selection if you want exhaustive fanout |

### 2.5 LLM-client abstraction

```
                  ┌───────────────────────────────┐
                  │   LlmClient  (Protocol)       │
                  │     .complete(system, user)   │
                  └───────────────┬───────────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  ▼                               ▼
        ClaudeLlmClient                    _CountingLlm (test stub)
        (claude-agent-sdk)                 (asserts recursion shape)
```

RLM is **provider-agnostic**: it doesn't pick Anthropic vs OpenAI itself — it uses whichever `LlmClient` the workspace was wired with.

### 2.6 Who calls RLM

```
backend/services/context/workspace/tools/__init__.py    ─ registers RlmQueryTool
backend/services/context/workspace/__init__.py          ─ re-exports
backend/schemas/events.py                               ─ event type 'rlm_query_executed'
tests/test_rlm_query_recursive.py                       ─ pins recursion shape
tools/test-rlm-on-paper.py                              ─ end-to-end smoke
```

Any agent that runs inside the workspace context (paper-understanding, environment-detective, etc.) can request `rlm_query(workspace_id, question, variable_name)` as a tool call. The result carries citations back to the original paper span, so downstream stages (claim map, rubric, final report) trace every claim.

### 2.7 What RLM is NOT

- ❌ Not per-agent LLM selection. The agents have one configured runtime each.
- ❌ Not training/finetuning. It's pure inference recursion.
- ❌ Not unbounded. `max_depth` + `max_llm_calls` guarantee termination.
- ❌ Not spawning sub-agents. It spawns sub-*calls* to one LLM.

---

## 3. Hermes — Nous-Hermes oversight layer

### 3.1 Why it exists
The pipeline produces structured outputs (claim maps, environment specs, baselines, improvements). A **separate model** audits whether each output is *grounded in the evidence* the agent claims to have used, and can downgrade verification gates when claims are unsupported.

### 3.2 Audit lifecycle

```
                  agent step finishes
                          │
                          ▼
              build_step_audit_payload(
                project_id,
                target,                       e.g. "paper_understanding"
                state_snapshot,
                structured_output,             ← the agent's JSON
                trace_text,                    ← raw reasoning trace
                artifact_paths
              )
                          │
                          ▼
              HermesAuditService.audit(scope=step, target, payload)
                          │
                          ▼
              NousHermesClient.audit(...)            ◄── provider chain (§3.3)
                          │
                          ▼
              HermesAuditReport {
                target, scope, status,                    grounded | caveat | unsupported |
                summary, findings,                        unavailable | system_error
                unsupported_claims,
                evidence_refs,
                recommended_intervention,                 annotate | retry_step |
                corrective_note,                          request_evidence | downgrade_claim |
                confidence,                               suppress_publication | escalate_human
                provider, raw_response
              }
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
       persist via   appended to     if scope=checkpoint and
       HermesAudit   state.hermes_*  status ∈ {unsupported, caveat}:
       Storage       _reports          → downgrade Gate 1/2/3 status
                                         (verified → verified_with_caveats
                                          → partial_reproduction → …)
                                       → push unsupported_claims onto
                                         gate.blocking_issues
```

### 3.3 Provider chain & "self-learning"

This is the only sense in which Hermes "learns" — it learns **which auditor backend is healthy**, NOT what the agents are doing.

```
                     NousHermesClient.audit(...)
                                │
                                ▼
            AdapterMemory.preferred_order([...])
            (read from <runs_root>/.hermes_adapter_memory.json)
                                │
                                ▼
            ┌────────────────────────────────────────────┐
            │  Default chain, reordered by past success: │
            │                                            │
            │  1. NousHermesProvider                     │ pip install hermes-agent
            │       (run_agent.AIAgent in-venv,          │   OR
            │        OR `hermes` CLI subprocess)         │ npm install -g hermes-agent
            │                                            │
            │  2. ClaudeAuditProvider                    │ ANTHROPIC_API_KEY
            │                                            │
            │  3. ClaudeCodeSdkProvider                  │ Claude Code subscription
            │                                            │
            │  4. OpenAIAuditProvider                    │ OPENAI_API_KEY
            │                                            │
            │  5. CodexCliProvider                       │ Codex CLI subscription
            └────────────────────────────────────────────┘
                                │
              for each provider in (re-ordered) chain:
                                │
                ┌──── is_available()? ───┐
                │ no                     │ yes
                ▼                        ▼
            skip + record         call(prompt) → raw text
            failure                      │
                                         ▼
                                 extract_audit_json(text)
                                 strategies (in order):
                                   1. ```json ... ``` fence
                                   2. first balanced { ... }
                                   3. strip prose prefix
                                         │
                          parse ok? ─────┴───── parse fail?
                            │                         │
                            ▼                         ▼
                   record_success            record_failure(error)
                   memory.last_successful    consecutive_failures++
                   _provider = name          if ≥ 3 → quarantine
                            │                         │
                            ▼                         ▼
                  return report           try next provider in chain
                                          (whole chain exhausted →
                                           status=unavailable, never
                                           silently substitute "ok")
                                                      │
                                                      ▼
                                            save_memory(updated AdapterMemory)
```

`AdapterMemory.preferred_order`:
1. `last_successful_provider` first (if not quarantined),
2. then remaining candidates by success rate (descending),
3. quarantined providers last.

### 3.4 What Hermes is NOT

- ❌ Not learning from agent LLM outputs in any training sense.
- ❌ Not remembering prior audits semantically — each audit is a single short JSON-producing LLM call on the current payload.
- ❌ Not silently substituting "ok" — chain exhaustion produces `status=unavailable`.
- ❌ Not coupled to one provider — it's a Protocol-based chain, new auditors plug in via registration.

### 3.5 Files

```
backend/hermes_audit/
├── __init__.py           re-exports
├── models.py             HermesAuditScope, ...Status, ...Confidence,
│                         HermesInterventionType, HermesEvidenceRef,
│                         HermesAuditReport (Pydantic v2)
├── payloads.py           build_step_audit_payload, build_checkpoint_audit_payload
├── client.py             NousHermesClient (provider chain, JSON extraction,
│                         self-learning memory wiring)
├── providers.py          AuditProvider Protocol + 5 impls,
│                         extract_audit_json (fence / braces / prose-strip)
├── service.py            HermesAuditService.audit → persist
├── storage.py            HermesAuditStorage (filesystem persistence)
└── memory.py             AdapterMemory, ProviderStats, load/save,
                          preferred_order, quarantine logic
```

---

## 4. Putting it together — one full run

```
 user uploads paper.pdf ──► frontend /api/demo (multipart, ≤50 MB)
                                │
                                ▼
                       backend.cli reproduce  ──►  IntakeAppService
                                                   (RegisterProject, FetchPaper,
                                                    PdfPathFetcher, PyMuPDF parser)
                                                          │
                                                          ▼
                                              workspace variable `paper_text`
                                                  (with provenance citations)
                                                          │
        ┌─────────────────────────────────────────────────┘
        ▼
 ReproLabOrchestrator.run():
   for stage in [paper_understanding, artifact_discovery,
                 environment_detective, planner, …]:
       ┌───────────────────────────────────────────────┐
       │ agent (Anthropic/OpenAI runtime)              │
       │   ├── may invoke RlmQueryTool ──┐             │
       │   │       recursive LLM over    │             │
       │   │       paper_text → Cited[T] │             │
       │   │       (depth≤3, calls≤24)   │             │
       │   └── produces structured_output│             │
       └───────────────────────────────────────────────┘
                          │
                          ▼
            Hermes step audit (provider chain, self-routing)
                          │
                          ▼
       PipelineState.hermes_step_reports[target] ← report
                          │
   (at gate stages 6/9/12):
                          ▼
            Hermes checkpoint audit
                          │
                          ▼
       _apply_checkpoint_report_to_gate(gate, report)
            ├── may downgrade gate.status
            └── may append gate.blocking_issues
                          │
                          ▼
   ...continue stages... ─► report_generator.generate_final_report(...)
                          │
                          ▼
         runs/<project_id>/
           final_report.md / .json
           pipeline_state.json
           assumption_ledger.json
           hermes/                       ← per-audit chain checkpoints
           .hermes_adapter_memory.json   ← provider routing memory
           code/, Dockerfile, raw_paper.pdf
```

## 5. One-line mental model

> **RLM** = a tool agents use to ask focused questions over big workspace variables, by recursively sub-dividing the variable across LLM calls; results stay cited.
>
> **Hermes** = an external auditor invoked after every agent step and gate, with a self-routing provider chain that remembers which auditor backend last worked; its job is to flag unsupported claims and (at gates) downgrade the verification status — never to learn the agents themselves.
