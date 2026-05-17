# RLM Integration Design — Wiring the Recursive Language Model into ReproLab

**Status:** Proposed
**Date:** 2026-05-17
**Author:** investigation drafted by Claude
**Reference:** Zhang/Kraska/Khattab, *Recursive Language Models* (arXiv:2512.24601)

---

## 1. Context

`backend/services/context/workspace/tools/rlm_query.py` is a faithful, tested, depth-bounded implementation of the RLM paradigm. **It is wired into the workspace service but no production code path invokes it.** All 18 `RlmQueryTool(...)` instantiations live in `tests/` or `tools/test-rlm-on-paper.py`. The CHANGELOG-claimed "RLM workspace service wired into the orchestrator" refers only to the surrounding state store; the recursive tool itself is dormant.

Hermes audit, by contrast, is fully invoked at every step + checkpoint (`orchestrator.py:957, 989`) and can downgrade gate decisions. The integration target here is RLM only; Hermes will be a *consumer* of RLM, not the subject of new wiring.

---

## 2. Why now

Three pain points the current pipeline cannot solve without RLM:

1. **Context bloat at large-paper stages.** `paper-understanding`, `reproduction-planner`, and `rubric-verifier` currently dump the full paper text + JSON state into the agent prompt. For long ML papers (>150k chars after pymupdf) this approaches Claude/OpenAI window limits and degrades attention quality.
2. **Ungrounded rubric verification.** Gates 1–3 evaluate rubric pass/fail from agent-summarized claims. The original paper text is never re-queried at verification time. False positives slip through when the summarizer hallucinates.
3. **Hermes audits are blind to source.** Hermes currently judges agent output against the prior pipeline state, not the source paper. `unsupported_claims` is impressionistic.

RLM converts "stuff the haystack into context" into "agent asks targeted questions, gets cited answers." It is the missing primitive for *grounded* verification.

---

## 3. Two integration patterns (hybrid)

### Pattern A — Orchestrator-driven (deterministic, primary)

The orchestrator owns a fixed question pack per stage. Before/after each agent call it issues `rlm_query(question, variable_name=paper_text)` and injects the cited answer into the prompt or into the verification payload. Predictable cost, no agent autonomy needed, easy to budget.

### Pattern B — Tool-mode (autonomous, secondary)

Expose `rlm_query` as a Claude SDK / OpenAI tool that the agent can call mid-turn. Higher autonomy, harder cost control. Reserved for `improvement-path` agents where the question space is open-ended.

**Recommendation:** ship Pattern A first across the hot stages, layer Pattern B on top later for improvement paths.

---

## 4. Insertion points (ranked by leverage)

| # | Stage | What RLM does | Variable queried | Pattern | Cost class |
|---|---|---|---|---|---|
| 1 | **rubric-verifier** | One query per rubric item: "Does the paper claim X? Cite the passage." | `paper_text` | A | High value, ~N×depth calls |
| 2 | **paper-understanding** | Replace whole-paper prompt with per-claim queries: contributions, datasets, metrics, hyperparameters. | `paper_text` | A | Medium |
| 3 | **artifact-discovery** | "What GitHub/HF/dataset URLs does the paper reference?" | `paper_text` | A | Low (single call usually) |
| 4 | **environment-detective** | "What Python, CUDA, GPU, library versions are stated?" | `paper_text` | A | Low |
| 5 | **Hermes `_audit_step`** | Hermes payload gains `rlm_evidence`: for each unsupported_claim candidate, RLM checks the paper. | `paper_text` | A | Compounds — gate behind threshold |
| 6 | **improvement-path** | Open-ended grounding mid-experiment ("what baseline did the paper report on CIFAR-100?"). | `paper_text`, `baseline_result` | B (tool-mode) | Variable |

Stages 1 and 5 are the biggest accuracy wins; 2–4 are simple latency/quality plays.

---

## 5. System-flow changes

### 5.1 Today

```
ingest → workspace.enrich(paper_text) →
  for each stage:
    prompt = template(state.model_dump())   # whole paper sometimes inlined
    output = runtime.invoke(agent, prompt)
    state.* = parse(output)
    _audit_step(state, structured_output)   # Hermes
    workspace.enrich_variable(...)
    advance_stage(...)
```

### 5.2 With RLM

```
ingest → workspace.enrich(paper_text) →
  rlm = RlmQueryTool(view_provider=workspace, llm_client=ClaudeLlmClient(...))
  for each stage:
    evidence = {q: rlm.call(workspace_id, question=q, variable_name="paper_text")
                for q in RLM_QUESTIONS[stage]}            # Pattern A
    prompt = template(state.model_dump(), evidence)        # cited snippets only
    output = runtime.invoke(agent, prompt, tools=[rlm_tool] if AUTONOMOUS else [])  # B
    state.* = parse(output)
    _audit_step(state, structured_output, rlm_evidence=evidence)   # Hermes sees citations
    workspace.enrich_variable(...)
    advance_stage(...)
```

### 5.3 Rubric verification (the killer loop)

```
for item in rubric_spec.items:
    ans = rlm.call(workspace_id="…",
                   question=f"Does the paper claim: {item.claim}? Quote it.",
                   variable_name="paper_text")
    item.grounded = "yes" in ans.value["answer"].lower()
    item.citations = ans.citations
gate_decision = aggregate(items)
```

Gate verdicts are now backed by the paper's own text, not the agent's summary of it.

---

## 6. Concrete wiring changes

### Backend

1. **`backend/agents/orchestrator.py`**
   - Constructor: require `workspace_service` + `workspace_id` (today both `Any | None = None`); fall back to in-memory workspace when callers don't supply one so Pattern A always has a target.
   - Construct an `RlmQueryTool` instance once per orchestrator with the active provider's `LlmClient` (Claude → `ClaudeLlmClient`, OpenAI → new `OpenAILlmClient` that already exists in `tools/test-rlm-on-paper.py` — promote it to `backend/services/context/workspace/tools/openai_client.py`).
   - Add helper: `_rlm_query(question: str, variable: str = "paper_text", **budget) -> Cited[dict]` that routes through `workspace_service.invoke_tool(...)` so the canonical `ToolInvoked` / `rlm_query_executed` event is emitted (needed for dashboard telemetry).
   - Per-stage budget: extend `RunBudget` with `rlm_calls_remaining` and short-circuit to "skip RLM, use raw paper text" once exhausted.

2. **`backend/agents/rlm_questions.py`** *(new)*
   Static question packs per stage (`PAPER_UNDERSTANDING_QS`, `ARTIFACT_DISCOVERY_QS`, …). Edit-not-code knob.

3. **`backend/hermes_audit/payloads.py`**
   Extend `build_step_audit_payload(...)` with optional `rlm_evidence: list[dict]`. Schema: `{question, answer, citations, depth_reached, llm_calls}`. Hermes prompt template gains a "Grounded evidence" section. The audit report's `unsupported_claims` becomes a verifiable list (each claim Hermes flags carries the RLM citations that contradict it).

4. **`backend/services/context/workspace/service.py`**
   Already has `invoke_tool(...)` plumbing (per WorkspaceTool Protocol). Confirm `ToolInvoked` translates to a `rlm_query_executed` event on the bus (the enum value already exists at `backend/schemas/events.py:17`).

5. **`backend/agents/runtime/claude_runtime.py`** *(Pattern B, later)*
   Register `rlm_query` as an MCP-style tool the SDK exposes to the agent. Reuse `_tools_for_sub_agent(...)`'s merge path.

### Frontend

1. **`frontend/src/components/lab/`** — new `rlm-trace-panel.tsx` showing per-stage: `depth_reached`, `llm_calls`, `chunks_examined`, `selection_path`, top citations.
2. **`frontend/src/lib/events/contract.ts`** — surface `rlmQueryExecuted` events to the live-run state.
3. **`frontend/src/components/lab/hermes-audit-panel.tsx`** — render `rlm_evidence` citations underneath each finding.

### Tests

- Promote `tools/test-rlm-on-paper.py` shape into an integration test that exercises the orchestrator's `_rlm_query` helper end-to-end with `StubLlm` (deterministic recursion shape) + `demo_paper.pdf`.
- New: `tests/test_rlm_orchestrator_pattern_a.py` asserts that each enabled stage emits at least one `rlm_query_executed` event.

---

## 7. Cost / latency model

| Quantity | Today | After Pattern A (all stages) | After Pattern A+B |
|---|---|---|---|
| LLM calls per run (no RLM) | ~14 (one per stage) | 14 + Σ RLM packs | 14 + Σ + agent-initiated |
| Typical RLM calls per stage | 0 | 3–8 (leaf hits @ ~80k char paper) | 3–24 |
| Worst case (full recursion, 500k chars, depth 3) | — | 24 calls × stage | 24 × stage × N agent invocations |
| **Default cap** | — | `max_llm_calls=24`, `max_depth=3` | per-run `rlm_calls_remaining=120` |

Mitigations baked in:

- **Short-circuit on small content.** If `len(paper_text) ≤ leaf_budget (12k)`, RLM degenerates to a single call — same as inlining.
- **Budget-aware orchestrator** falls back to inlined text when `rlm_calls_remaining ≤ 0`.
- **Provider selection** — `ClaudeLlmClient` uses the subscription path (no per-token cost) when the user runs locally; `OpenAILlmClient` lands on `gpt-4o-mini` by default.

---

## 8. Phased rollout

| Phase | Scope | Risk | Acceptance |
|---|---|---|---|
| **P1** | Wire `RlmQueryTool` construction in orchestrator; helper `_rlm_query` exists; behavior unchanged (no callers). | None | unit tests prove instantiation + event emission |
| **P2** | Pattern A on `paper-understanding` only, behind `REPROLAB_RLM_ENABLED=1`. | Latency regression on small papers | golden-file diff stays within tolerance; lab-smoke passes |
| **P3** | Pattern A on `artifact-discovery` + `environment-detective`. | Same | lab-e2e-full passes; cost ledger shows < 1.5× baseline |
| **P4** | Pattern A on `rubric-verifier` (the win). | Gate flips on existing replays | re-run paperbench corpus; compare gate flip set; expect higher precision, slightly lower recall |
| **P5** | Hermes payload gains `rlm_evidence`. | Hermes prompt grows; provider chain may need bigger context | provider chain re-tested with longer payload |
| **P6** | Pattern B (tool-mode) for `improvement-path`. | Runaway tool calls | hard cap via `RunBudget.rlm_calls_remaining` |

P1 + P2 are the minimum viable; everything after compounds value.

---

## 9. Open questions

1. **Which variable feeds RLM at later stages?** `paper_text` is the obvious one; should `baseline_result` (experiment logs) also be queryable for `improvement-path`? Probably yes — logs are precisely the "large content, focused question" shape RLM was designed for.
2. **Determinism under provider rotation.** `ClaudeLlmClient` is `temperature` is implicit (Claude Code SDK default); `OpenAILlmClient` uses `temperature=0`. Pin both at 0 and record `provider:model` in `selection_path` so replays are bit-exact.
3. **Hermes provider conflict.** Hermes already runs an LLM per step; layering RLM on top of Hermes layers two recursive LLM systems. Cap RLM-inside-Hermes to depth=1 (single chunked call, no recursion) to keep audit cheap.
4. **Workspace lifecycle.** Today `_workspace_service` can be `None`. Either require it for SDK pipeline, or build a tiny in-process `MemoryWorkspaceService` so Pattern A is always available.

---

## 10. Non-goals

- Replacing the agent runtime or sandbox layer.
- Changing Hermes provider chain.
- Adding new gates.
- Productizing the RLM tool for end-users (it stays internal).

---

## 11. Decision log seed

- **Pattern A first** because deterministic cost beats agent autonomy at this maturity level.
- **`paper_text` as the canonical RLM target** because it is the only variable guaranteed to exceed the model window and is needed by ≥4 stages.
- **Hermes consumes RLM, not vice versa** — keep the audit layer thin; RLM is a primitive Hermes uses, not a replacement.
- **Workspace service is the gateway**, not direct `RlmQueryTool.call(...)` — preserves the `ToolInvoked` event invariant.
