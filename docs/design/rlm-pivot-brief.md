# OpenResearch / ReproLab — RLM Pivot Plan

> Canonical plan for the RLM pivot. Supersedes the earlier implementation brief,
> the deleted `rlm-integration.md`, and the deleted pre-pivot architecture docs.
> One consistent document — no corrections preamble. If something here is wrong
> against the code, fix it here in the same change. Last revised 2026-05-20.

> **✅ Fork RESOLVED (2026-05-21) — §3/§5/§11 confirmed.** The architecture
> fork (drift D1) is closed: the `rlm` library wins. A spike installed and
> probed every candidate — `rlm.RLM`'s real signature matches this brief's §3
> table exactly, including `environment='docker'` (which ReproLab needs and
> `dspy.RLM`'s WASM sandbox cannot do). Hand-building (the old issue #59 / PR
> #65 skeleton) is retired; `dspy.RLM` (#66) is evaluated and not adopted.
> Evidence + verdict: `docs/design/rlms-spike-report.md`. §3, §5, and the §11
> phase plan are the canonical architecture — implement against them.

## 1. Context — why

ReproLab reproduces research papers end-to-end and scores the result against a
PaperBench-style rubric. It is being shown to potential investors/reviewers (a
serial founder; a senior Microsoft engineering leader) who were told the system
is built on the **Recursive Language Model (RLM)** paradigm. Two reasons drive
the pivot, in order of importance:

1. **Honesty.** The pitch is RLM-based; the implementation is a fixed 14-stage
   pipeline state machine. A reviewer who has read the RLM paper catches that in
   minutes. The pivot is what makes the pitch true.
2. **Capability (our hypothesis, not a paper result).** We believe RLM suits
   paper reproduction better than a fixed-stage machine — the paper is a long,
   dense context and reproduction is exploratory, not linear. The RLM paper
   validates RLM for long-context *reasoning*; it does **not** validate RLM as an
   agentic orchestrator. This is our bet — label it as such, do not cite it as
   proven.

**RLM is the substrate, not a target.** ReproLab is *built on* the RLM paradigm;
it *reproduces other papers*. The RLM paper itself is not a reproduction target.
(Reproducing it with an RLM-based system is possible, but explicitly out of
scope.)

**The real gap.** No paper has ever been reproduced end-to-end by ReproLab — by
any architecture. There are zero `final_report.{json,md}` files for a real
paper; the existing "demo" is a self-authored PPO fixture with a generated
codebase. The pivot is not finished when the RLM loop runs — it is finished when
a real PaperBench paper completes with a real rubric score. That is the
deliverable bar; the RLM architecture is how we reach it honestly.

## 2. What RLM is

The RLM paper (arXiv 2512.24601; Zhang, Kraska, Khattab, MIT CSAIL) defines
**Algorithm 1**: an inference paradigm with three properties. Violate any one and
you have **Algorithm 2** — the "deceptively similar" CodeAct-like scaffold the
paper explicitly calls out as the wrong version.

1. **The prompt is offloaded to the environment.** The prompt P is set as a
   variable in a persistent Python REPL. The root model never receives P in its
   context window — only constant-size metadata (length, type, a short prefix,
   how to access it).
2. **The output is built in a REPL variable.** Not returned via an autoregressive
   `Finish(text)`. The output can exceed the model's context window because it is
   constructed programmatically across iterations.
3. **Sub-calls are programmatic.** The REPL exposes `llm_query()` / `rlm_query()`
   as Python functions. The root writes code that loops and branches over them —
   the recursion is symbolic, in the program, not an API-level tool-use block.

The Algorithm 1 loop: initialise the REPL with P → loop { root model writes code
→ REPL executes it → only *metadata* about stdout returns to the model's history
} → stop when the final-answer variable is set. **We do not reimplement this —
see Section 3.**

## 3. Architecture decision — build on the `rlms` library

The RLM authors publish a maintained reference implementation: the **`rlms`**
PyPI package (`pip install rlms`; source `github.com/alexzhang13/rlm`). It *is*
Algorithm 1. We **depend on it** rather than reimplement it — less code, and
faithful by construction (a reviewer sees `from rlm import RLM`, not our
re-derivation of the paradigm).

**RLM is the root orchestrator (the radical pivot).** The 14-stage `PipelineStage`
state machine is retired. The stages become a library of **primitives**; the RLM
root model decides what to call, and in what order, by writing REPL code.

The `rlms` `RLM` class provides, as first-class constructor arguments, almost
everything the earlier brief planned to hand-build:

| Need | `rlms` provides |
|---|---|
| Algorithm 1 root loop + REPL host | `RLM(...).completion(prompt)` |
| Sub-calls | `llm_query` / `rlm_query` (+ `_batched`) — reserved, built in |
| Our domain primitives in the REPL | `custom_tools={...}` constructor arg |
| REPL backend | `environment="local"` (also `docker`/`e2b`/`modal`/…) |
| Recursion depth | `max_depth` (default `1` → `llm_query` only; set `2` to enable `rlm_query`) |
| Root-iteration cap | `max_iterations` (default 30) |
| Cost / time / token caps | `max_budget`, `max_timeout`, `max_tokens` |
| Cheaper sub-call model | `other_backends` |
| Live UI events | `on_iteration_start/complete`, `on_subcall_start/complete` callbacks |
| Trajectory logging | `RLMLogger` → JSONL |
| System-prompt override | `custom_system_prompt` |

Models: a strong frontier root with a cheaper sub-call model via `other_backends`
(the paper's GPT-5 / GPT-5-mini pattern). The paper validates **GPT-5 and
Qwen3-Coder** as RLM roots; **Claude as a root is not paper-validated** — if used,
verify empirically. No fine-tuning; RLM-Qwen3-8B is out of scope.

## 4. What survives

Wrapped as primitives, or used unchanged — do not rewrite these during the pivot:

- `PaperExtractor` — produces the paper text loaded into the REPL `context`.
- Docker + RunPod sandbox runtimes.
- Environment build-and-repair logic → the `build_environment()` primitive.
- Rubric verifier → the `verify_against_rubric()` primitive.
- Stage-agent **core logic** (paper-understanding, environment-detective,
  planner, baseline implementer, experiment runner, improvement proposer) —
  extracted as plain functions, no longer stage-driven.
- PaperBench vendored bundle (`third_party/paperbench/`) — the rubric source.
- SQLite event store; `assumption_ledger.json`, `cost_ledger.jsonl`,
  `agent_telemetry.jsonl` — still emitted, now from inside the primitives.
- The lab UI shell — header, project context, SSE pattern. Only the graph view
  changes.
- `backend/services/context/workspace/tools/rlm_query.py` — the existing dormant
  recursion tool; superseded by the `rlms` library. Keep as reference or retire;
  do not invest further in it.

## 5. What we build

The library is the engine; our code is the domain layer plus glue — and it is
small:

- **`backend/agents/rlm/primitives.py`** — the domain primitive functions, each
  wrapping surviving stage-agent core logic, assembled into the `custom_tools`
  dict. Each primitive emits a `primitive_call` event and updates `cost_ledger`.
- **`backend/agents/rlm/system_prompt.py`** — the reproduction-domain RLM system
  prompt, passed as `custom_system_prompt`. It carries RLM operating principles,
  the primitive signatures, `context` metadata, and **in-context decomposition
  examples** (the paper's Figure 4a shows these measurably improve performance —
  even unrelated examples help). It does **not** prescribe a fixed workflow.
- **`backend/agents/rlm/run.py`** — the new run entry, replacing the
  `PipelineStage` machine in `backend/agents/orchestrator.py`: builds the
  `RLM(...)`, wires SSE emission into the `on_*` callbacks, calls `.completion()`,
  and writes `final_report.{json,md}` from the result.
- The operational layer the library does not provide: the event-store bridge and
  checkpoint/resume (Section 10).

We do **not** write a REPL host, a root loop, or `sub_LLM`/`sub_RLM` — `rlms`
provides all three.

## 6. What gets removed

- The `PipelineStage` enum and stage-advancement logic in
  `backend/agents/orchestrator.py`.
- Gate 1/2/3 as control-flow. Verification becomes a primitive
  (`verify_against_rubric`) the root calls when it judges useful — there are no
  fixed gate checkpoints.
- The hardcoded five improvement paths in `backend/agents/topology.py`
  (optimizer / backbone / augmentation / horizon / diffusion). The new
  `propose_improvements()` returns a variable-length, paper-specific list with
  proposer-assigned tags.
- The fixed 14-stage pipeline strip in the lab UI.

## 7. The RLM environment

At run start the `rlms` REPL is initialised with:

**`context`** — the offloaded prompt, a dict the root slices into; the root sees
only metadata about it, never its contents: `paper_text`, `paper_metadata`,
`supplementary_text`, `repo_files`, `prior_work_refs`.

**`custom_tools`** — passed to `RLM(custom_tools=...)`; callables become REPL
functions, non-callables become REPL variables:
- data: `rubric_spec`, run config.
- primitives (callables): `understand_section(text_slice)`,
  `extract_hyperparameters(text_slice)`, `detect_environment(method_spec)`,
  `build_environment(env_spec)`, `plan_reproduction(method_spec, env_spec)`,
  `implement_baseline(plan)`, `run_experiment(code_path, env_id)`,
  `verify_against_rubric(results, rubric)`,
  `propose_improvements(results, rubric_scores)` — returns a variable-length list
  of paper-specific candidates with free-form tags (**not** the old five paths).

**Reserved (library built-ins, cannot be overridden):** `llm_query`, `rlm_query`,
`llm_query_batched`, `rlm_query_batched`, `SHOW_VARS`, `answer`, `context`,
`history`.

**Termination:** the root sets the reserved `answer` variable (the library's
final-answer dict). The final report is built up as REPL state and placed in
`answer`; it is read from the REPL, not from autoregressive model text.

Primitive signatures here are the contract; re-verify them against the
implementation when `primitives.py` is written.

## 8. RLM fidelity — invariants to hold and test

The system is genuinely RLM only if these hold. Add tests/assertions for each.

1. **The paper never enters the root's context.** It is offloaded as `context`.
   Test: the root model's message payload never contains a `paper_text`
   substring. If it does, you have built Algorithm 2.
2. **Primitives take slices, not the corpus.** Invariant #1 guards the *root*'s
   payload — it does not guard the *primitives*. A primitive that receives the
   whole paper and feeds it to an LLM re-creates the context-window problem RLM
   exists to eliminate (Algorithm 2's flaw, one level down). Primitives take
   slices the root constructed, and specs — never the whole `context`.
3. **Recursion is programmatic.** `llm_query`/`rlm_query` are REPL functions the
   root calls from code — never tool-use blocks in the root's API request.
4. **Output is a REPL variable.** Termination is the library's `answer`
   mechanism; the report is built as REPL state so it can exceed the context
   window.
5. **The root actually recurses over the paper.** If the root only calls domain
   primitives in sequence and never uses `llm_query`/`rlm_query` over `context`,
   the "recursive" claim is hollow — it is the old pipeline in a REPL.
6. **Depth.** Default `max_depth=1` (sub-`llm_query` only). Set `max_depth=2` to
   enable sub-`rlm_query`. At the cap, `rlm_query` falls back to `llm_query`.
7. **The root triages.** The system prompt must instruct the root to decline
   candidates unlikely to lift weak rubric nodes — cost and time budgets are
   real. If logs show every proposed candidate was attempted, strengthen the
   triage prompt.
8. **Caps.** Use the library's `max_iterations`, `max_budget`, `max_timeout` —
   they cover the runaway-sub-call cost the paper names as a real failure mode.
   The reproduction *run* budget (Docker builds, experiments) is separate and
   larger; size it independently of the LM-call budget.
9. **Different papers → different trajectories.** Test on ≥3 papers; candidate
   lists and iteration counts must differ meaningfully. Identical trajectories
   mean the dynamism is fake.

Note: the `rlms` REPL is `exec` in a controlled namespace, not a security
sandbox — the root model is trusted. The paper flags sandboxed REPLs and async
sub-calls as open problems; do not sink time into either.

## 9. Events and UI

**Events.** Wire SSE emission into the library's callbacks
(`on_iteration_start/complete`, `on_subcall_start/complete`) and into the
primitive wrappers. Event types (the full schema lives in code): `repl_iteration`,
`primitive_call`, `candidate_proposed`, `candidate_outcome`, `sub_rlm_spawned`,
`rubric_score`, `root_reasoning`, `run_complete`. Only *metadata* about stdout
goes to the model history (Algorithm 1) and to the UI — never full stdout.

**UI.** The lab shell (header, project context, SSE) stays. The fixed 14-stage
strip is replaced by:
- a prominent **rubric score bar** — current vs target, baseline → current;
- a **REPL state panel** — live variables and the available primitives;
- a **live iteration panel** — the Python the root just wrote + stdout metadata;
- a **dynamic exploration tree** (the centerpiece) — source → baseline →
  candidate branches that appear as `propose_improvements()` returns them, each
  colored by outcome (promoted / marginal / failed / running / skipped /
  declined). Not a hardcoded fan-out;
- a collapsible **primitive-call history**.

Visual language unchanged: flat surfaces, hairline borders, sentence case, no
marketing aesthetic. UI claims must match backend reality — no fake fixed slots,
no "real-time" overclaim.

## 10. Checkpointing and resume

`rlms` runs one `.completion()`; it does not checkpoint a long agentic run for
process-kill resume. We own this:

- After each root iteration: append the iteration to the SQLite event store;
  persist REPL variable state (store large values like `paper_text` as file
  references, not full pickles); record the iteration code + stdout metadata.
- On resume: restore REPL state, replay iteration history, continue the loop.
- `rlm_query` sub-calls are their own runs (`prj_xxx_sub_yyy`) with a
  `parent_run_id`; they checkpoint independently.

## 11. Build order

Each phase has a done condition; do not start one before the last is done.
Refactor on the `rlm-pivot` branch; keep `main` runnable until cutover.

**Phase 1 — Spike.** `pip install rlms`. Stand up a minimal `RLM(custom_tools=…)`
with two mock primitives on a tiny mock paper; confirm Algorithm 1 runs, the
`custom_tools` are callable in the REPL, and the `on_*` callbacks fire.
*Done:* a standalone RLM run terminates via `answer` with mock primitives.

**Phase 2 — Primitives.** Extract each surviving stage-agent's core logic into a
plain function in `primitives.py`; wire cost-ledger + event emission; test each
in isolation. Hardest: `build_environment` (own retry loop), `run_experiment`
(sandbox state), `propose_improvements` (must return variable-length,
paper-specific candidates — validate variation across papers).
*Done:* every primitive callable from the REPL, with correct outputs and events.

**Phase 3 — Orchestrator + system prompt.** Write `backend/agents/rlm/` —
`system_prompt.py`, `run.py`, the SSE bridge, and `final_report` writing. Wire
run modes (`pipeline.py`): `sdk` mode runs the RLM root.
*Done:* a real RLM run on a PaperBench paper, primitive by primitive.

**Phase 4 — Frontend.** New SSE event handling; REPL state panel; live iteration
panel; dynamic exploration tree; primitive-call history. Mock data first, then
live SSE.
*Done:* the lab UI renders a live RLM run; no fixed stage strip anywhere.

**Phase 5 — End-to-end.** Vendor the real `ftrl` bundle (Section 13). Run the RLM
system on it end-to-end; produce `final_report.md` with a real rubric score; fix
what breaks; run a second paper.
*Done:* ≥2 PaperBench papers with completed runs and real rubric scores on disk.

**Phase 6 — Cleanup.** Delete the dead `PipelineStage` / gate / five-path code.
Update `README.md` and `system_overview.md` to drop the "current (pre-pivot)"
framing. Pin a successful run as the demo.

## 12. Success criteria

A technical reviewer can: clone the repo, `docker compose up`, upload a paper,
watch the dynamic tree populate live, and see the run produce `final_report.md`
with a real PaperBench rubric score; open `backend/agents/rlm/` and see the
system is the `rlms` library driven by domain primitives; find ≥2 completed
`runs/prj_xxx/` directories with real reports for different papers.

The pitch sentence becomes true: *"ReproLab is a paper-reproduction agent built
on the Recursive Language Model paradigm. The root model treats the paper as a
variable in a persistent REPL, writes Python to navigate and decompose it,
recursively invokes sub-models on programmatic slices, and accumulates the
reproduction as REPL state."* It maps 1:1 to code in the repo — and the RLM
engine is the authors' own published library.

## 13. First paper

`third_party/paperbench/ftrl/` is currently a placeholder bundle with a
*synthetic* rubric. **Vendor the real upstream `ftrl` artifacts** (real
`paper.md`, `addendum.md`, `rubric.json` from the public PaperBench repo), plus
1–2 genuinely easy papers, so scores are honestly comparable to PaperBench's
published baselines.

The `agent-eval-integ` branch vendored the *RLM paper* as a bundle
(`third_party/paperbench/rlm/`). That is **not** paper #1 — RLM is the substrate,
not a reproduction target (Section 1). It can remain as a self-test curiosity.
