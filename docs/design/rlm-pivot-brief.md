# OpenResearch / ReproLab — Full RLM Pivot Implementation Brief

> Committed to `main` 2026-05-19 as the canonical change brief for the RLM pivot.
> **Read the preamble corrections first.** The brief body below was drafted from
> the documented architecture and is stale in places; the corrections are verified
> against the code and the RLM paper (arXiv 2512.24601) and take precedence over
> the brief body where they conflict.

## Status & corrections (2026-05-19)

**Decisions (from review Q&A):**

- **Scope — do both.** Ship the RLM architecture *and* a real end-to-end
  reproduction. The brief's "7–10 days" is no longer a hard constraint; do not
  descope either deliverable to fit a timeline.
- **Architecture — radical.** The 14-stage `PipelineStage` state machine is an
  old primitive; move off it. The RLM root loop becomes the orchestrator and the
  stage agents become REPL-callable primitives — the brief's Section 3 / Section
  14 path, not the conservative Pattern A proposed in `docs/design/rlm-integration.md`.
- **First paper — vendor real `ftrl`.** `third_party/paperbench/ftrl/` is
  currently a placeholder bundle with a *synthetic* rubric. Replace it with the
  real upstream PaperBench artifacts, plus 1–2 genuinely easy papers, so scores
  are honestly comparable to PaperBench's published baselines.

**Corrections to the brief body (verified against the code):**

- **RLM is not greenfield.** `backend/services/context/workspace/tools/rlm_query.py`
  (514 lines, tested) already implements RLM-style recursion as a *dormant tool*
  with no production caller. `docs/design/rlm-integration.md` (2026-05-17) already
  designs an integration (the conservative Pattern A). Reuse `rlm_query.py`'s
  `_recursive_query` as the `sub_RLM` engine — it survives the refactor and should
  be added to Section 4. The REPL host, root loop, and `sub_LLM` are the genuinely
  new code. `rlm_query.py` is *not* Algorithm 1 — it has no REPL, no `state["Final"]`,
  no code execution.
- **No paper has been reproduced end-to-end — by any architecture.** There are
  zero `final_report.{json,md}` files in any run directory. The current "demo"
  (`runs/ui_sdk_anthropic_review_same_demo_*`, score 91.4) is a `workspace_fixture`:
  a self-authored PPO paper with a deterministically generated codebase, not a
  reproduction. `prj_8b78ac6368bad043` (the one run on a real paper PDF) halted at
  `gate_2`. Completing a real paper is the core deliverable, not a Phase 5 afterthought.
- **Stale file references.** The lab UI client is
  `frontend/src/components/lab/lab-shell.tsx` (not `repro-lab-client.tsx`, which
  was split away). Stage agents are flat modules (`backend/agents/<name>.py`), not
  packages. The live improvement selector is already an LLM agent
  (`improvement-orchestrator`) that is paper-aware and variable-length — the
  hardcoded five paths live only in `backend/agents/topology.py` and the UI. Gates
  are already binary-halt as of the "Option D" refactor.
- **Environment.** The venv runs Python **3.14.2**, not 3.11 as the brief and
  `CLAUDE.md` state. `rlm_query.py`'s `LlmClient.complete()` is synchronous while
  the orchestrator is async — bridging `sub_LLM`/`sub_RLM` from REPL code is a real
  design item. "Sandboxed namespace" means a controlled `globals` dict for `exec`,
  not a security sandbox; the root model is trusted.

## RLM paper accuracy check (arXiv 2512.24601v3 — verified 2026-05-19)

The brief's RLM claims were checked against the full paper. **Verified accurate:**
the arXiv ID `2512.24601` (v3, 11 May 2026; Zhang, Kraska, Khattab, MIT CSAIL), the
Algorithm 1 transcription, the three design properties (prompt-as-environment,
output-in-variable, programmatic recursion), and the Algorithm 2 contrast.
Corrections — points 1–3 and 7 are the implementation-critical ones:

1. **Recursion depth — `sub_RLM` is not a depth-1 primitive.** Section 7 lists
   `sub_RLM` as "depth-1 per paper default." Wrong: the paper's depth=1 (the
   default) exposes a sub-*LLM* call only (`llm_query`); the sub-*RLM* call
   (`rlm_query`) exists only at **depth>1**. The brief's own FM#10 caps at
   depth-1 — at which `sub_RLM` does not exist. To make `sub_RLM`, the
   `sub_rlm_spawned` event (§9), and the UI "R" marker (§11) real, **run
   depth=2** (paper Table 1 shows depth=2 is strong and still cheap). At the
   depth cap, `rlm_query` falls back to `llm_query`.

2. **`FINAL` / termination — FM#3 is stricter than the paper.** Algorithm 1
   abstracts termination as `state[Final]`. The reference implementation
   (Appendix C / B) terminates by parsing `FINAL(...)` / `FINAL_VAR(...)` tags
   from the model's response: `FINAL(text)` returns model-written text (an
   autoregressive finish — the thing FM#3 forbids), `FINAL_VAR(name)` returns a
   REPL variable. The paper itself calls this brittle (Appendix B). Correct
   design: **terminate when the model emits a `FINAL_VAR`-style tag, then read
   the answer from the named REPL variable.** FM#3 is right that the final
   *value* must come from a variable (so it can exceed context); it is wrong
   that termination must not be detected from parsed output — the tag *is*
   parsed output.

3. **System-prompt length — FM#6's ~2000-token cap contradicts the paper.**
   Appendix C's system prompt is long and deliberately carries several
   in-context decomposition examples. Figure 4(a) shows these "greatly improve
   both overall performance and the initial decomposition attempt... even if the
   example is unrelated to the actual task." Keep FM#6's ban on prescribing a
   rigid workflow, but **include in-context decomposition examples** and drop the
   token cap.

4. **Reference names.** The paper's implementation uses `context` (the prompt
   variable), `llm_query`, and `rlm_query`. The brief's `paper_text` / `sub_LLM`
   / `sub_RLM` are fine domain adaptations — just know that the reference repo
   (github.com/alexzhang13/rlm) and the repo's own `rlm_query.py` use the
   paper's names.

5. **Root model.** The paper validates **GPT-5 and Qwen3-Coder-480B as RLM
   roots**; Claude (Opus 4.1) appears only as a baseline coding agent, never as
   an RLM root — so Section 3's "Claude Opus" is unverified by the paper.
   Pattern: a strong root model with a cheaper sub-call model (the paper uses
   GPT-5 root / GPT-5-mini sub-calls). Roots need strong coding ability and a
   large output-token budget (Appendix B), and the same system prompt is not
   safe across models (Qwen needed an explicit anti-over-subcalling line).

6. **Async + sandbox are open problems — confirmed.** The paper (§7, Appendix B)
   flags asynchronous sub-calls and sandboxed REPLs as unsolved future work; its
   reference implementation is synchronous and blocking, which matches
   `rlm_query.py`'s sync `LlmClient`. FM#10's sub-call cost cap is well-motivated
   — the paper explicitly names "exploding sub-call costs" as a real failure mode.

7. **RLM-as-orchestrator is our extension — and the primitives are where it can
   quietly become Algorithm 2.** The paper validates RLM as a long-context
   *inference* scaffold: its REPL holds only the prompt plus
   `llm_query`/`rlm_query`/`print`, and it returns a string. It does *not*
   validate RLM driving side-effecting tools — our domain primitives
   (`build_environment`, `run_experiment`, `implement_baseline`, …) mutate
   Docker / filesystem / run state and are an extension beyond the paper. The
   closest paper support is Observation 5 / Appendix C.3 (LongCoT — the RLM
   decomposes a reasoning graph into nodes, delegates each via sub-calls,
   memoizes verified results, assembles); use that as the root-prompt template,
   not a fixed primitive order. FM#1 guards the *root's* payload (no `paper_text`
   in the root message) — it does not guard the *primitives*: a primitive that
   receives the whole paper as one argument and feeds it to an LLM re-creates the
   exact context-window problem RLM exists to eliminate (Algorithm 2's Flaw #1,
   hidden one level down). Invariant: `paper_text` / `supplementary` /
   `repo_files` stay offloaded variables; the root navigates them with
   root-written code and `llm_query`/`rlm_query` over **slices it constructs**;
   primitives take slices and specs, never the whole corpus. If the root never
   invokes `llm_query`/`rlm_query` over the paper at all, Property 3 is vestigial
   and "recursive" is false advertising. Section 1's "Capability" rationale —
   that RLM is "genuinely better-suited to paper reproduction" — is the team's
   hypothesis, not a paper result; label it as such.

8. **Add a root-iteration cap.** FM#10 caps sub-calls (50) and cost ($10); the
   paper also bounds the *root* loop — 20 iterations and ~4096 output tokens per
   turn in its runs (Appendix A). Add a root-iteration cap. The $10 is a
   sub-call guard only; the reproduction run budget (Docker builds, experiments)
   is separate and larger — size it independently.

---

## How to read this brief

This is the master change brief for a significant architectural pivot. The repository currently implements a 14-stage agent state machine. The target is a system architected around the Recursive Language Models (RLM) paradigm from arXiv 2512.24601. This document covers backend refactor, UI redesign, what to keep, what to throw away, and the order to do it in.

Read this entire brief before writing any code. The order of operations at the end is non-negotiable — getting the wrong things in the wrong order will break the only thing that currently demos.

If you find any instruction in this brief that contradicts the actual code in the repository, surface the contradiction in your first response. Do not silently route around it. The brief was written based on the documented architecture (README, system_overview.md, and inspection of orchestrator.py); if the reality differs, we adjust the plan.

## Section 1 — Strategic context

This is not a research project or a hobby refactor. This system is going to be reviewed by:

- Marcus Weller, serial founder (Deepinvent, formerly Blippar CEO), who has indicated intent to fund. He was sold on RLM-as-substrate. The current artifact does not honestly reflect that pitch.
- A senior Microsoft engineering leader (CVP/VP level, three layers under Satya Nadella), being introduced by Marcus.

Neither of these people will be moved by visual polish or architecture diagrams. They will open the repo, look at what the code does, and ask: "show me a paper this has reproduced end-to-end, with the score." The repo currently cannot answer that question — there is one project_id (prj_8b78ac6368bad043) appearing in both `runs_e2e_docker/` and `runs_e2e_local/`, which is effectively one canonical replayed demo run.

The pivot exists for two reasons, in this order of importance:

1. **Honesty.** The pitch to Marcus was RLM-based. The implementation is not. A serious technical reviewer who has read the RLM paper (or who pattern-matches on the paradigm) will catch this in minutes. The pivot is what makes the pitch true.

2. **Capability.** RLM as a paradigm is genuinely better-suited to paper reproduction than a fixed-stage state machine. The paper is a long context with dense, structured content; the reproduction process is naturally exploratory rather than linear. RLM was made for exactly this shape of problem.

The funding is verbal. There is no signed agreement, dollar amount, structure, or timeline known to be in place. The next 7-10 days determines whether verbal becomes real. Do not treat this as a leisurely refactor.

## Section 2 — What RLM actually is (read this twice)

The RLM paper introduces Algorithm 1, which is a specific inference paradigm with three non-negotiable properties. If any of these are violated, the system is not RLM, it is Algorithm 2 from the paper, which the paper explicitly calls out as the "deceptively similar" wrong version.

**Property 1: The prompt is offloaded to the environment, not the model's context.**

Given a prompt P (in our case: paper PDF text + supplementary + cited prior work + repo files if present), a persistent Python REPL environment E is initialized with P as a variable. The root LLM M never receives P in its context window. M sees only constant-size metadata about the variable: length, short prefix, type, how to access it.

**Property 2: The output is built in REPL variables, not via autoregressive Finish action.**

The root model does not return its final answer via a tool call like Finish(text). It sets `state["Final"] = ...` *inside the REPL via code execution*. The output can exceed M's context window because it is constructed programmatically across iterations.

**Property 3: Sub-calls are programmatic, not action types.**

The REPL exposes a `sub_RLM(prompt, ...)` function (and `sub_LLM(...)` for non-recursive sub-calls). The root LLM writes Python that can loop, branch, and call this function in arbitrary patterns, Ω(|P|) or Ω(|P|²) times inside a single code block. The recursion is symbolic, in the program. Sub-calls are not API-level tool_use blocks of the root model.

The Algorithm 1 loop:

```
state = InitREPL(prompt=P)
state = AddFunction(state, sub_RLM_M)
hist = [Metadata(state)]
while True:
    code = LLM_M(hist)
    state, stdout = REPL(state, code)
    hist = hist || code || Metadata(stdout)
    if state["Final"] is set:
        return state["Final"]
```

Note specifically: `Metadata(stdout)` not `stdout`. Only constant-size metadata about stdout returns to the root model's history. The full stdout stays in the REPL.

The reference implementation is at https://github.com/alexzhang13/rlm. Read it before implementing. Use it for the system prompt structure (paper Appendix C) and the loop mechanics. Do not copy it wholesale — the architecture of integrating it with our existing primitives is the work.

## Section 3 — Architecture decision: RLM as the orchestrator

We are doing the full refactor. Not "RLM in one stage" — RLM as the root orchestrator. The 14-stage pipeline is decomposed into a library of callable primitives that the root RLM invokes from inside its REPL. This is the architectural choice we are committed to.

Frontier models prompted as RLMs (GPT-5, Claude Opus). No fine-tuning. The paper's RLM-Qwen3-8B result is out of scope; we are not training a model. We are using prompted frontier models with the RLM scaffolding.

## Section 4 — What survives the refactor

These components stay, unchanged or near-unchanged. They are wrapped as primitives the root RLM can call. Do NOT rewrite or "improve" them during this refactor.

- `PaperExtractor` (REPROLAB_PAPER_EXTRACTION_MODE=hybrid) — produces the text that gets loaded into the REPL as `paper_text`
- Docker sandbox runtime + RunPod sandbox runtime
- Environment build-and-repair logic (Track 4) — wrapped as `build_environment()` primitive
- Rubric verifier agent (Track 3) — wrapped as `verify_against_rubric()` primitive
- Existing stage agent core logic (paper-understanding extraction, environment detection, planner, baseline implementer, experiment runner) — extracted as callable functions, no longer driven by stage transitions
- PaperBench vendored bundle in `third_party/paperbench` — still the source of rubrics
- SQLite event store schema — extends, does not get rewritten
- `assumption_ledger.json`, `cost_ledger.jsonl`, `agent_telemetry.jsonl` — still emitted, hooks rewired to fire from inside RLM primitives
- The Lab UI's general layout (header, project context, SSE connection pattern) — only the graph view changes

## Section 5 — What gets rewritten

These are the components that change. Order of work given in Section 14.

**Backend, heavy refactor:**

- `backend/agents/orchestrator.py` — the central change. Replace `PipelineStage` enum-driven loop with RLM root loop. Host the REPL. Manage iteration history and `state["Final"]`.
- `backend/services/events/live_runs.py` — SSE bridge. Emit new event schema for REPL iterations and primitive calls.

**Backend, moderate refactor:**

- `backend/agents/pipeline.py` — run modes. `sdk` mode now runs the RLM root. `offline` mode either drops or becomes a recorded-trajectory replay; decide based on testing needs.
- Each existing stage agent module (`backend/agents/paper_understanding/`, etc.) — extract the core logic into pure functions that take inputs and return outputs. They are no longer stages; they are primitives.

**Backend, new code:**

- `backend/agents/rlm/repl_host.py` — hosts the Python REPL. Handles code execution safely (sandboxed namespace). Manages variable state. Handles serialization for checkpointing.
- `backend/agents/rlm/root_loop.py` — implements Algorithm 1: builds the metadata prompt, calls the root LLM, parses code from the response, executes in the REPL, captures stdout metadata, loops until Final is set.
- `backend/agents/rlm/system_prompt.py` — the root LLM system prompt, adapted from the RLM paper's Appendix C. Plus a description of the available primitives and variables. Keep this prompt minimal; do not stuff workflow instructions into it.
- `backend/agents/rlm/primitives.py` — the function registry exposed to the REPL. Each primitive wraps an existing stage agent's core function. Each primitive emits a `primitive_call` event for the UI and updates cost_ledger.
- `backend/agents/rlm/sub_call.py` — implementation of `sub_LLM(prompt)` and `sub_RLM(prompt)`. Sub-calls are NOT tool_use blocks; they are Python functions the REPL exposes.

**Frontend, heavy refactor:**

- `frontend/components/lab/repro-lab-client.tsx` — graph view changes from fixed 14-stage strip to dynamic exploration tree (described in Section 11). The header, status panel, and project context stay the same.

**Frontend, new components:**

- A REPL state panel showing live variables in the REPL with names, types, summaries.
- A live iteration panel showing the code the root model just wrote and the stdout metadata that came back.
- A dynamic exploration tree showing the actual candidates the RLM proposed and what it did with each (the tree shape is preserved from the original UI but candidates appear dynamically, not as a hardcoded five-path fanout).
- A primitive call history list (collapsible).

## Section 6 — What gets thrown away

These are dead in the new architecture. Delete or quarantine them once the new system is working.

- The `PipelineStage` enum-driven advancement logic in `orchestrator.py`. The new orchestrator does not have ordered stages.
- The hardcoded five improvement paths (optimizer-path, backbone-path, augmentation-path, horizon-path, diffusion-path). These were hackathon fixtures. The new system uses `propose_improvements()` which returns a variable-length list of paper-specific candidates with proposer-assigned tags. There is no canonical five-path taxonomy.
- The Gate 1 / Gate 2 / Gate 3 control-flow gating. Verification is now a primitive (`verify_against_rubric()`) the root model calls when it judges appropriate. There are no fixed gate checkpoints. The rubric score progress bar in the UI is the only place verification status surfaces.
- The pipeline-strip UI (the horizontal stage indicator). It is replaced by the dynamic exploration tree and trajectory bar.

## Section 7 — The REPL environment in detail

At run initialization, the REPL is populated with the following variables and functions. The exact names matter — the root system prompt references them.

**Variables, loaded at init:**

- `paper_text: str` — full extracted text of the paper PDF (via PaperExtractor)
- `paper_metadata: dict` — title, authors, sections list, figure captions, table captions
- `supplementary_text: str | None` — appendix/supplementary material if present, else None
- `repo_files: dict[str, str] | None` — filename to content mapping, if the paper's open-source repo was found, else None
- `prior_work_refs: list[dict]` — cited prior work entries the model can fetch on demand (requires a fetch primitive)
- `rubric_spec: dict` — the PaperBench-style rubric the run will be scored against
- `Final: Any` — initially not set; setting this terminates the run

**Functions, exposed as Python callables:**

- `sub_LLM(prompt: str, model: str = "default") -> str` — single non-recursive LLM call
- `sub_RLM(prompt: str) -> str` — recursive RLM sub-call (depth-1 per paper default)
- `understand_section(text_slice: str) -> dict` — wraps existing paper-understanding extraction
- `extract_hyperparameters(text_slice: str) -> dict` — wraps existing hyperparameter extraction
- `detect_environment(method_spec: dict) -> dict` — wraps environment-detective
- `build_environment(env_spec: dict) -> dict` — wraps Docker build-and-repair loop, returns build result including success/failure and image_id if successful
- `plan_reproduction(method_spec: dict, env_spec: dict) -> dict` — wraps reproduction-planner
- `implement_baseline(plan: dict) -> str` — wraps baseline-implementation, returns path to generated code
- `run_experiment(code_path: str, env_id: str) -> dict` — wraps experiment-runner, returns metrics
- `verify_against_rubric(results: dict, rubric: dict) -> dict` — wraps rubric-verifier
- `propose_improvements(current_results: dict, rubric_scores: dict, k: int = None) -> list[dict]` — wraps improvement-selection. Returns a variable-length list of candidate improvements with proposer-assigned tags. The root decides which to try and in what order. THIS IS NOT THE HARDCODED FIVE PATHS.
- `set_final(report: dict) -> None` — convenience for setting state["Final"]

The root model gets metadata about all of these in the system prompt — variable names with type and length info, function signatures with one-line descriptions. The root model does NOT get the variable contents or function implementations.

## Section 8 — The exploration philosophy

This is the critical conceptual shift. In the old system, the orchestrator decides what runs next. In the new system, the root LLM decides what runs next, by writing code that calls primitives in any order it judges useful.

A reproduction run might look like:

- Iteration 1-3: root reads paper_metadata, peeks at paper_text via slicing, calls `understand_section` on the methods section
- Iteration 4: root calls `detect_environment` on the extracted method_spec
- Iteration 5: root calls `build_environment`, which fails. Root sees the failure metadata, calls it again with a modified env_spec.
- Iteration 6: root calls `plan_reproduction` and `implement_baseline`
- Iteration 7: root calls `run_experiment`, gets initial metrics
- Iteration 8: root calls `verify_against_rubric`, gets a baseline score of 0.31
- Iteration 9: root calls `propose_improvements`, gets back 7 candidate improvements with various tags
- Iteration 10-11: root chooses to try 4 of them, declines 3 with stated reasoning. Tries the most promising first (learning rate warmup), gets +0.18
- Iteration 12: root calls `sub_RLM` on a specific failure mode it noticed in the run_results
- Iteration 13-14: root iterates on improvements based on sub_RLM output, runs experiment again, verifies
- Iteration N: root calls `set_final({...})` with the final report

What is critical: this trajectory is NOT scripted. Two different papers will produce two different trajectories. A paper with no public repo will spend more iterations on `understand_section`. A paper whose environment is trivial to detect will skip ahead. A paper whose baseline already hits target rubric score might skip improvements entirely. The root decides.

The improvement-selection step in particular: `propose_improvements()` returns whatever the proposer thinks is worth trying for THIS paper based on THIS baseline's weak rubric nodes. Sometimes that's 3 candidates, sometimes 10. The tags are descriptive labels the proposer assigns (e.g. "optimizer", "regularization", "backbone", "data-augmentation", "inference-time", whatever the proposer chooses), NOT a fixed taxonomy. There is no canonical five-category split anywhere in the new architecture.

## Section 9 — Event schema for SSE

The SSE stream emits a new set of events to drive the UI. Event types and minimal fields:

```typescript
type ReplIterationEvent = {
  type: 'repl_iteration';
  iteration: number;
  code: string;            // the Python code the root wrote
  stdout_metadata: string; // length + short prefix of stdout, not full stdout
  duration_ms: number;
  timestamp: number;
};

type VariableUpdateEvent = {
  type: 'variable_update';
  iteration: number;
  variable: { name: string; type: string; summary: string; set: boolean };
  timestamp: number;
};

type PrimitiveCallEvent = {
  type: 'primitive_call';
  iteration: number;
  primitive: string;       // e.g. "verify_against_rubric"
  args_summary: string;    // brief description of args, not full args
  status: 'running' | 'completed' | 'failed';
  duration_ms?: number;
  result_summary?: string; // brief description of result
  rubric_delta?: number;   // if applicable
  timestamp: number;
};

type CandidateProposedEvent = {
  type: 'candidate_proposed';
  iteration: number;
  candidate: {
    id: string;
    title: string;
    tag: string;          // proposer-assigned, free-form
    description: string;
    reasoning: string;    // why proposer suggested this
  };
  timestamp: number;
};

type CandidateOutcomeEvent = {
  type: 'candidate_outcome';
  candidate_id: string;
  outcome: 'promoted' | 'marginal' | 'failed' | 'running' | 'skipped';
  iterations_spent: number;
  rubric_delta: number | null;
  reasoning?: string;     // present on 'skipped' (why root declined)
  timestamp: number;
};

type SubRLMSpawnedEvent = {
  type: 'sub_rlm_spawned';
  parent_iteration: number;
  parent_candidate_id?: string;
  sub_run_id: string;     // children of sub_RLM get their own event stream
  prompt_summary: string;
  timestamp: number;
};

type RubricScoreEvent = {
  type: 'rubric_score';
  iteration: number;
  score: number;
  target: number;
  weak_nodes: Array<{ id: string; criterion: string; score: number }>;
  timestamp: number;
};

type RootReasoningEvent = {
  type: 'root_reasoning';
  iteration: number;
  text: string;           // extracted natural-language reasoning fragment
  timestamp: number;
};

type RunCompleteEvent = {
  type: 'run_complete';
  final_report_path: string;
  final_score: number;
  total_iterations: number;
  total_cost_usd: number;
  total_duration_ms: number;
  timestamp: number;
};
```

The old event schema (`run_state`, `agent_log`, `dashboard_event` tied to stages) gets removed once the new schema is consumed by the UI. Keep both running in parallel during the transition so the old UI keeps working until the new UI is ready.

## Section 10 — Checkpointing and resume-safety

The REPL state must be checkpointed after each iteration so a run can be resumed if the process is killed.

Use the SQLite event store as the source of truth for the iteration history. Persist REPL variables to disk via pickle (or a custom serializer for non-picklable values). After each iteration:

1. Append the iteration event to the SQLite event store
2. Pickle the REPL globals dict (minus non-serializable handles) to `runs/<project_id>/repl_state.pickle`
3. Write the iteration's code and stdout metadata to `runs/<project_id>/iterations/i<N>.json`

On resume:

1. Load `runs/<project_id>/repl_state.pickle` to restore REPL globals
2. Replay the iteration history from the event store to rebuild the root model's iteration history
3. Continue the Algorithm 1 loop from where it stopped

Sub_RLM calls are themselves runs with their own project_id (e.g. `prj_xxx_sub_yyy`), so they checkpoint and resume independently. Use a parent_run_id field in the run metadata to link them.

Handle non-picklable values:

- File handles: don't put them in REPL globals; store paths instead
- Threads / async tasks from sub-calls: serialize as "pending" markers and re-issue on resume
- Large strings (paper_text especially): consider storing as separate file references rather than pickling the full string each iteration

## Section 11 — UI redesign in detail

The lab UI keeps its header, project context, and SSE connection pattern. The graph view changes substantially. Here is the spec.

**Region 1: Run header** (unchanged from current — project ID, source, status, counters)

**Region 2: Rubric score bar** — full width, prominent

- Large number (current rubric score, e.g. 0.53)
- Target (smaller, dimmed)
- Horizontal progress bar, color-coded by distance from target
- Below: "baseline 0.31 → current 0.53 (+0.22)"

**Region 3: Workspace (two columns)**

Left column (~280px): REPL state panel

- Header: "REPL state" with a variable icon
- List of variables with name (monospace), type, summary
- Variables that are set: normal color. `Final` unset: dimmed.
- Below the variable list: "Primitives available" with the callable function list in monospace

Right column (flex-1): Live iteration panel

- Header: "Iteration N · live" with "writing code" status indicator
- Code block showing the Python the root just wrote (syntax-highlighted)
- Below code: stdout metadata in a smaller, secondary section: "184 chars · weak nodes: ['attn-mask', 'pos-enc-init', 'lr-warmup']"

**Region 4: Dynamic exploration tree** — the centerpiece

This preserves the tree topology of the original UI (left-to-right flow, source feeds into baseline, baseline fans out into candidate branches). The lie in the original was that the fanout was a fixed five-path graph. The fix is making the fanout dynamic: branches appear as `propose_improvements()` returns candidates, with visual treatment encoding outcome.

Structure:
- Leftmost: source paper node (small, teal)
- Center: baseline node (purple, showing rubric score and code size after baseline-implementation completes)
- Right: candidate branches fanning out, vertically stacked, one per improvement candidate

Candidate branch node, roughly 180×36px:
- Title in 500 weight, dark text from the outcome's color ramp
- Subtitle showing iteration count and rubric delta ("3 iter · +0.18")
- Outcome badge pill on the right: "promoted" (green), "marginal" (amber), "failed" (coral), "running" (purple, pulsing), "skipped" (gray)
- Left-edge border color and connector edge to baseline match the outcome:
  - promoted: solid green
  - marginal: solid amber
  - failed: dashed coral (signals reverted)
  - running: solid thicker purple, with pulsing dot indicator
  - skipped: dotted thin gray
- Optional recursive marker: a small circular "R" node hanging off the right edge of any candidate that triggered sub_RLM calls. Clicking it drills into that sub-RLM's own tree.

Declined candidates: collapse into a single node titled "N candidates declined" with names hinted in subtitle. Click to expand inline with each decline reason from the root.

Below the bottom-most candidate: a faded dashed node labeled "+ propose more candidates" — when clicked, triggers another `propose_improvements()` call (only available if run is active and budget remains).

Ordering:
- During live runs: chronological top-to-bottom (in proposal order). New nodes appear at the bottom as they're proposed.
- After completion: re-sort by outcome (promoted, marginal, failed, running, skipped).

Soft cap: show top 8 branches, collapse rest into "+ N more candidates" expand toggle.

Legend below the tree with color swatch + label for each outcome.

Interactions:
- Hover a node: tooltip with iteration range, primitive calls within this branch, rubric trajectory
- Click a node: expands a side panel showing that branch's iterations, code blocks, and any sub_RLM calls it spawned
- Click the "R" sub-RLM marker: focused view of that sub-RLM's own tree, breadcrumb to navigate back

**Region 5: Rubric timeline + root reasoning** (two columns)

Left: small bar chart, one bar per iteration, height = rubric score at that iteration. Color: gray before any improvement, green when in promoted branch, coral when in failed branch. Current iteration outlined.

Right: most recent root-reasoning fragment as italic text. Single sentence or short paragraph, ~12-13px. Fades softly when content updates.

**Region 6: Primitive call history** (collapsible, default collapsed)

Reverse-chronological list of every primitive call:
- Function call in monospace, syntax-highlighted
- Status: "running 12s", "0.31 · 8s", "3 trials · 1m 47s"
- Latest calls expanded, older calls collapse into "N earlier calls" group

## Section 12 — Visual style

Keep the existing UI's visual language. Flat surfaces, 0.5px borders, generous whitespace, sentence case, no marketing-style aesthetic decisions.

Do NOT import the orbital-node-cloud / glow / dark-gradient marketing aesthetic from any reference. The reference visuals are hero images, not working interfaces — they look great in screenshots and would be illegible in live use with streaming updates.

Do NOT adopt positioning language like "self-improving superintelligence" or "automate knowledge discovery" in any visible UI copy. The honest description is "paper-reproduction agent on the RLM paradigm." Use that or simpler.

Color encoding for outcomes (use existing palette tokens):
- Promoted: green
- Marginal: amber
- Failed: coral
- Running: purple
- Skipped: gray

Color encoding for primitive categories (in trajectory bar tiles and call history):
- Understand: blue
- Environment: teal
- Plan/implement: purple
- Experiment: coral
- Verify: pink
- Retry: amber
- Recursive (sub_RLM): purple, slightly darker

Animation: subtle. Status dot 2s pulse. New tile fade-in 200ms. Running candidate badge low-contrast pulse. Respect `prefers-reduced-motion`.

## Section 13 — Failure modes to actively guard against

These are the specific ways this build can go wrong. Add tests or assertions where possible.

1. **Building Algorithm 2 instead of Algorithm 1.** The fastest way to fail. If the root LLM call's message payload includes paper_text anywhere, you have built CodeAct, not RLM. Add a test that asserts the root model's message content NEVER contains the substring of paper_text or supplementary_text. Print and inspect the payload during development.

2. **Pseudo-recursion.** sub_LLM and sub_RLM must be Python functions the REPL exposes, called by the root model's code. They must NOT appear as tool_use blocks in the root model's API call. Add a test that asserts the root model's tools list either does not include sub_LLM/sub_RLM as tools, or that they only appear as code-execution-style mentions in the system prompt, not as callable tools.

3. **Autoregressive Finish.** The root model must set `state["Final"]` via code in the REPL. The orchestrator reads Final from REPL state at loop end. The root must NOT return its final answer as text in its response. Add an assertion that the run only terminates when REPL state contains a Final variable, not based on parsed model output.

4. **Static improvement tree.** If `propose_improvements()` returns the same five candidates regardless of paper, you have recreated the hardcoded system. Test by running on three different PaperBench papers and asserting that the candidate lists differ in count or content.

5. **Root never declines candidates.** LLMs tend to try everything. The root system prompt must explicitly instruct triage: cost budget, time budget, decline candidates unlikely to address weak rubric nodes. If the run logs show every proposed candidate was attempted, the triage prompt isn't working — strengthen it.

6. **System prompt bloat.** The temptation is to stuff the root system prompt with workflow guidance ("first call understand_section, then call detect_environment..."). Resist. The system prompt provides RLM operating principles (from paper Appendix C), primitive signatures, and variable metadata. It does NOT prescribe a workflow. The root figures the workflow out from REPL exploration. If the system prompt grows past ~2000 tokens, you are doing this wrong.

7. **Breaking the demo while refactoring.** The current 14-stage system at least produces a happy-path demo run. Do not delete it until the new RLM system can do the same. Refactor on a branch. Keep main working. Cut over only after the new system has successfully completed at least one PaperBench paper end-to-end with a real rubric score.

8. **Scope creep into research.** The paper describes fine-tuned RLM-Qwen3-8B with 28.3% improvement. That's a research project. We are not doing that. We are using prompted frontier models with RLM scaffolding. If the work starts to drift into "let's also try fine-tuning" or "let's also evaluate on OOLONG," kill it. Out of scope.

9. **UI lying about the backend.** If the backend produces dynamic candidates but the UI shows them in fixed five-category slots, the UI is lying. If the backend has resume-safety but the UI claims "real-time" with no resume affordance, the UI is overclaiming. Keep the UI's claims aligned with what the backend actually does.

10. **Sub_RLM cost explosion.** sub_RLM calls can recurse and call more sub_RLMs (though we cap at depth-1 per paper default). Add a hard cap on total sub-call count and total cost per run. Default: 50 sub-calls max, $10 cost max per run, configurable. Without this, a runaway run will burn the API budget.

## Section 14 — Order of operations

Do these in order. Each phase has a clear "done" condition. Do not start a phase until the previous one is done.

### Phase 1: Preparation (Day 1)

- Create branch `rlm-pivot` off main
- Read the RLM paper (arXiv 2512.24601) end to end
- Read https://github.com/alexzhang13/rlm reference implementation
- Read `backend/agents/orchestrator.py` to understand the current state machine
- Read `system_overview.md` to verify the documented architecture matches the code
- Map each existing stage agent module to the primitive it will become

Done condition: write a one-page document in `docs/rlm-pivot-mapping.md` mapping each current stage agent to its target primitive, with the function signature.

### Phase 2: Backend foundation (Days 2-3)

In `backend/agents/rlm/`:

1. `repl_host.py` — Python REPL host with safe code execution, variable management, serialization. Test: load a string variable, execute code that slices it, retrieve the result.
2. `sub_call.py` — `sub_LLM()` and `sub_RLM()` implementations as Python functions. Test: call sub_LLM in isolation and confirm a single API call happens. Call sub_RLM and confirm the nested loop runs.
3. `system_prompt.py` — root system prompt, adapted from paper Appendix C. Document each variable and primitive available. Keep it short.
4. `primitives.py` — function registry. Each primitive wraps an existing stage agent's core function. Each emits a primitive_call event. Each updates cost_ledger.
5. `root_loop.py` — Algorithm 1 implementation. Test by feeding it a tiny mock paper, mock primitives, and asserting it produces a Final value via code-set state["Final"], not autoregressive output.

Done condition: a standalone RLM root can run on a mock paper with mocked primitives, terminate via state["Final"], and produce all the SSE events in the new schema.

### Phase 3: Primitive integration (Days 4-5)

For each existing stage agent:

1. Extract the core function from its current stage-driven wrapper
2. Wire it into `primitives.py` as a callable function
3. Add cost_ledger and event emission
4. Test the primitive in isolation: call it with realistic inputs, verify outputs and events

The hardest primitives to extract cleanly will be:
- `build_environment` (because the build-and-repair loop has its own retry logic)
- `propose_improvements` (because the old system hardcoded five categories; the new one must produce variable-length lists with proposer-assigned tags)
- `run_experiment` (because it depends on sandbox state which lives outside the REPL)

For propose_improvements specifically: rewrite the proposer's system prompt to produce paper-specific candidates with free-form tags, not five-category-slotted ones. Validate by running on three different papers and confirming candidate variation.

Done condition: each primitive can be called from inside the REPL, returns realistic outputs, emits correct events. End-to-end test: run the full RLM root on a real PaperBench paper, primitive by primitive, asserting at each step that the right thing happened.

### Phase 4: Frontend redesign (Days 5-7)

In parallel with late Phase 3 (Phase 4 can start once event schema is stable):

1. New event types in the frontend's SSE handler
2. REPL state panel component
3. Live iteration panel component (code block + stdout metadata)
4. Dynamic exploration tree component (the centerpiece, replaces the static graph)
5. Rubric timeline + root reasoning row
6. Primitive call history (collapsible)
7. Click-to-drill-down: candidate node → side panel with branch detail; "R" sub-RLM marker → focused sub-tree view

Wire to mocked data first to nail visual fidelity, then switch to live SSE.

Done condition: lab UI renders correctly against both mocked and live RLM run events. Updates feel smooth. Drill-down works. No fixed five-path graph anywhere in the UI.

### Phase 5: End-to-end runs (Days 7-9)

1. Pick the easiest PaperBench paper from the ICML 2024 Spotlight/Oral list in `third_party/paperbench`. Easiest means: small model, public code, public data, clear single metric.
2. Run the new RLM system on it end-to-end. Produce a `final_report.md` with a rubric score.
3. Read the report and the assumption_ledger and figure out what went wrong (most things will go wrong on the first run). Fix the obvious bugs.
4. Run a second paper. Compare candidate variation between paper 1 and paper 2 — they must differ meaningfully.
5. Run a third paper if time permits.

Done condition: at least 2 PaperBench papers have completed runs with `final_report.md` files in their run directories, with real rubric scores. Even if the scores are low, the artifacts exist.

### Phase 6: Cleanup and surface polish (Day 9-10)

1. Update README to reflect RLM architecture. Lead with what the system does, then how to run it, then one paragraph on the architecture.
2. Update system_overview.md
3. Move screenshots and stray PDFs at repo root into `docs/`
4. Confirm `docker compose up --build` works on a fresh clone with just env vars
5. Pin the demo to a specific run (prj_xxx) so anyone clicking through sees a successful reproduction
6. Delete the dead pipeline stage advance code, gates, and hardcoded five-path config
7. Run cleanup script on outputs from old runs that no longer match the new schema

Done condition: a senior engineer can clone the repo, follow the README, hit `docker compose up`, open the lab UI, upload a paper, and watch a successful RLM run produce a final report. They can open `backend/agents/rlm/root_loop.py` and see Algorithm 1 implemented. They can read the system prompt and recognize the RLM paradigm.

## Section 15 — What success looks like

A reviewer (Microsoft VP, Aljoša at Deepinvent, anyone technical) can:

1. Clone the repo and `docker compose up --build`
2. Hit the lab UI and upload a paper from the PaperBench bundle
3. Watch the dynamic tree populate with candidates as the RLM proposes them, with outcomes visible in real time
4. See the run complete and produce `final_report.md` with a real rubric score against PaperBench's rubric for that paper
5. Open the repo and find `backend/agents/rlm/root_loop.py` implementing Algorithm 1 from the paper
6. Read the system prompt in `backend/agents/rlm/system_prompt.py` and recognize the RLM paradigm
7. Find at least 2-3 different completed `runs/prj_xxx/` directories with `final_report.md` files showing different papers and different exploration trajectories

The pitch sentence becomes true: "OpenResearch is a paper-reproduction agent built on the Recursive Language Model paradigm. The root model treats the paper, supplementary material, and any prior work as variables in a persistent REPL environment, writing Python code to navigate and decompose them, recursively invoking sub-models on programmatically-constructed slices, and accumulating reproduction artifacts (environment specs, generated code, experiment results) as REPL state. The full reproduction is a single RLM inference call."

That sentence maps 1:1 to code a reviewer can find in the repo. If it doesn't, the refactor failed regardless of how nice the code looks.

## Section 16 — What I (Claude Code) should do first

When you start:

1. Confirm you have read this entire brief
2. Open `backend/agents/orchestrator.py`, `system_overview.md`, and `third_party/paperbench` and report back any discrepancies between what this brief describes and what's actually there
3. Surface any tooling or dependency gaps you notice (Python version, missing packages, sandbox configuration issues)
4. Confirm the easiest PaperBench paper you'd pick for Phase 5 testing, with a one-line justification
5. Wait for confirmation before starting Phase 1 work

Do not start coding before doing steps 1-4 and getting an OK. The first time we touch code, we touch it on a branch, deliberately, in the order described above.

End of brief.
