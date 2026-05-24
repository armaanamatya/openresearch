# OpenResearch / ReproLab — Project Rebuild Spec

**Audience:** a future "me" rebuilding this system from scratch in a new framework
(DSPy, LangGraph, custom REPL, …) without reading the existing Python.
**Scope:** the *what* and *why*, not the *how*. Code is referenced by role, not by
import path. Everything here is implementation-agnostic.

**Companion files (kept for verification, not required reading):**
- `system_overview.md` — current state of the 14-stage system
- `docs/design/rlm-integration.md` — conservative Pattern-A wiring (RLM as a tool)
- `docs/design/rlm-pivot-brief.md` — radical pivot (RLM as the orchestrator)
- `docs/design/rlm-and-hermes-visualized.md` — diagrams of the two layers in place
- `docs/agents/*.md` — per-agent contracts

---

## 1. Project goal in one paragraph

ReproLab is an **agent system that reproduces a research paper end-to-end**: given an
arXiv URL or PDF, it (a) ingests and understands the paper, (b) discovers code/data
artifacts, (c) builds a containerized environment, (d) plans and implements a
baseline that re-creates the paper's headline result, (e) runs that baseline in a
sandbox (local Docker or remote GPU pod), (f) verifies the result against a
PaperBench-style weighted rubric, (g) explores improvement paths in parallel, and
(h) emits a benchmark report comparing the reproduction's metrics against the
paper's claims. Negative results, ambiguous assumptions, and abandoned paths are
preserved as first-class output — a "research map", not just a pass/fail.

The two cross-cutting layers that make this trustworthy are **RLM** (grounded
context exploration) and **Hermes** (independent oversight). The rest of this doc
explains those, plus the surface they sit on.

---

## 2. The substrate: a paper-reproduction pipeline

### 2.1 The 14 stages (current state — the substrate RLM/Hermes plug into)

```
1  ingest                   PDF → text (pymupdf), arXiv resolver
2  paper_understood         claims, metrics, hyperparams, datasets extracted
3  artifacts_discovered     code repos, datasets, model weights located
4  environment_built        Dockerfile generated + built (build-and-repair loop)
5  plan_created             reproduction plan: scope, scripts, success criteria
6  GATE 1                   plan looks sound? (Hermes checkpoint audit here)
7  baseline_implemented     code generated under runs/<id>/code/
8  baseline_run             experiment executed in sandbox
9  GATE 2                   baseline matches paper? rubric-verifier scores here
10 improvements_selected    candidate improvements proposed
11 improvements_run         improvement paths executed (parallel, was 5-fanout)
12 GATE 3                   improved result better? rubric re-scored
13 research_map_generated   trajectory + negative results compiled
14 complete                 final_report.{json,md} written, SSE emits "complete"
```

Each stage has a corresponding agent. State is checkpointed to
`runs/<project_id>/pipeline_state.json` after every transition — resume-safe.

### 2.2 Two opt-in loops inside the linear strip
- **Rubric loop (Track 3).** If the Gate-3 score is below `rubric_target_score`,
  the orchestrator re-enters stages 10–12 until either the score clears or
  `rubric_max_improvement_iterations` is exhausted. Fail-closed: a verifier error
  degrades to a heuristic rubric rather than silently passing.
- **Environment build-and-repair (Track 4).** At stage 4 the Dockerfile is built;
  on failure the build error is fed back to `environment-detective` in repair
  mode, capped by `environment_build_max_attempts`. Fail-soft: when the cap is
  spent, Gate 2 is allowed to record an honest partial reproduction instead of
  halting the run.

### 2.3 Process model and storage
- **One Docker image, two processes** (FastAPI backend on `:8000`, Next.js
  frontend on `:$PORT`). The browser never talks to the backend; the frontend
  proxies server-side via `/api/demo/*`. No CORS layer by design.
- **Each run is a long-lived subprocess.** Run state lives entirely on disk
  under `runs/<project_id>/`: `demo_status.json` (UI snapshot),
  `pipeline_state.json` (checkpoint), `final_report.{json,md}`, `*.jsonl` agent
  event logs (the SSE source), `code/` (generated reproduction), and the Hermes
  audit chain artifacts.
- **SQLite** is the event/persistence store with CQRS-style projections. Schema
  is additive; nothing rewrites it.
- **SSE** is the UI bridge. Frame types: `run_state`, `agent_log`,
  `dashboard_event`. The client coalesces enriched frames so a timed-out
  payload-less frame never regresses the graph view.

### 2.4 Sandbox tiers
- `local` — host process, fastest, no isolation
- `docker` — network/memory/CPU-controlled
- `runpod` — remote GPU pods (RTX 4090-class); preflight check runs before the
  pipeline boots
The sandbox's `command_timeout` is the **single source of truth for compute
budget**; the planner and baseline-implementer are prompted with it explicitly so
they pick env/seed/timestep counts that fit. Smoke-test + one reduced-scope run
is still a faithful reproduction *check* (not a paper-replication).

---

## 3. RLM — Recursive Language Model layer

### 3.1 Why it exists
Three failure modes the linear pipeline cannot fix on its own:
1. **Context bloat.** Long ML papers (>150k chars post-pymupdf) plus accumulated
   JSON state overflow the model window. Attention degrades.
2. **Ungrounded rubric verification.** Gates score against the agent's *summary*
   of the paper, not the paper itself. Summary hallucinations slip through.
3. **Blind Hermes audits.** Hermes judges agent output against prior pipeline
   state, never the source paper. "Unsupported claims" becomes impressionistic.

RLM converts *"stuff the haystack into context"* into *"agent asks targeted
questions, gets cited answers."* It is the missing primitive for **grounded**
verification.

### 3.2 The recursion shape (faithful to Zhang/Kraska/Khattab, arXiv:2512.24601)
A query against a long source variable (`paper_text`) runs as:

```
RlmQuery(question, variable, depth=0)
  ├─ split variable into chunks (≈4 by default)
  ├─ score each chunk for relevance to question (cheap LLM pass)
  ├─ keep top-k (e.g. 2 of 4)
  ├─ for each kept chunk:
  │     if chunk fits the model window:
  │         answer directly → return {answer, citations}
  │     else:
  │         RlmQuery(question, chunk, depth+1)   # recurse
  └─ aggregate children's answers → return Cited[T]
```

**Defaults / rails:**
- `max_depth = 3` (paper uses 2; 3 is forgiving for very long papers)
- `max_llm_calls = 24` per top-level query
- `top_k = 2` of 4 chunks kept per level
- Token budget enforced per call; over-budget → return partial with `degraded:
  true`
- Output is `Cited[T]` — a typed answer plus list of `{chunk_id, char_span,
  quoted_excerpt}` citations

### 3.3 LLM-client abstraction
RLM is provider-agnostic. The interface is a tiny `LlmClient` protocol with
`complete(prompt, max_tokens)`. Concrete clients: `ClaudeLlmClient`,
`OpenAILlmClient`. Adding a provider = adding one adapter. RLM never depends on
the agent runtime; it can be unit-tested with a fake client.

### 3.4 Two integration patterns (both are real; ship both)
- **Pattern A — Orchestrator-driven (deterministic, primary).** Each stage agent
  is given a static *question pack* (e.g. `PAPER_UNDERSTANDING_QS`). The
  orchestrator runs those questions through RLM *before* invoking the agent,
  injects the cited answers into the agent's prompt, and the agent works from
  pre-computed grounded evidence. Cost is predictable.
- **Pattern B — Tool-mode (autonomous, secondary).** The agent runtime exposes
  `rlm_query(question, variable)` as a callable tool. The agent decides when to
  ask. Bounded by `RunBudget.rlm_calls_remaining`. Used where exploration is
  paper-specific (improvement-path proposers).

### 3.5 The killer loop: rubric verification
```
for item in rubric_spec.items:
    ans = rlm.call(workspace_id=..., question=f"Does the paper claim: {item.claim}? Quote it.",
                   variable_name="paper_text")
    item.grounded = "yes" in ans.value["answer"].lower()
    item.citations = ans.citations
gate_decision = aggregate(items)
```
Every gate verdict is now backed by the paper's own text, not the agent's
recollection of it.

### 3.6 The radical pivot: RLM as the orchestrator
The full target architecture (`rlm-pivot-brief.md`) decomposes the 14 stages into
a **library of REPL-callable primitives** invoked by a root RLM loop:

**REPL variables at init:** `paper_text`, `paper_metadata`, `supplementary_text`,
`prior_work_refs`, `current_results`, `rubric_scores`, `run_results`.

**REPL primitives (signatures only; the root LLM never sees their bodies):**
- `understand_section(section_name) -> dict`
- `text_slice(start, end) -> str`
- `extract_hyperparameters() -> dict`
- `detect_environment() -> EnvSpec`
- `build_environment(env_spec) -> BuildResult`
- `plan_reproduction(method_spec) -> Plan`
- `implement_baseline(plan) -> CodeArtifact`
- `run_experiment(code, config) -> RunResults`
- `verify_against_rubric(rubric_spec, results) -> RubricScores`
- `propose_improvements(current_results, rubric_scores, k=None) -> list[dict]`
- `sub_LLM(prompt) -> str` and `sub_RLM(prompt) -> Cited[T]`
- `set_final(report: dict) -> None`

The root model receives only *metadata* in its system prompt: variable
names+types+length summaries, function signatures, one-line docs. **Workflow is
not prescribed.** The root LLM writes Python that calls primitives in any order
it judges useful. If the system prompt grows past ~2000 tokens, you are doing
this wrong.

**What gets thrown away in the pivot:**
- The `PipelineStage` enum and stage-ordered advancement
- The hardcoded five improvement paths (optimizer / backbone / augmentation /
  horizon / diffusion — hackathon fixtures)
- The fixed Gate 1/2/3 control-flow gating (verification becomes a primitive the
  root calls when appropriate)
- The horizontal pipeline-strip UI

**What survives the pivot:**
- Paper ingest (PDF→text), Docker + RunPod sandboxes, build-and-repair logic,
  rubric verifier core, PaperBench rubric bundle, SQLite event store schema,
  cost/assumption/telemetry ledgers, Lab UI shell + SSE pattern.

---

## 4. Hermes — independent oversight layer

### 4.1 Role
Hermes is **not** the agent doing the work. It is a separate auditor that runs
*after* an agent produces output, scoring it for unsupported claims, missing
evidence, and contradictions with prior state. Outputs are persistent audit
reports written to disk and into `PipelineState.hermes_step_reports`.

### 4.2 Audit lifecycle
Every stage produces an audit payload:
- *Step audit* — runs once per stage on the agent's structured output. Result
  attached to the stage; can flag issues but does not block.
- *Checkpoint audit* — runs at Gates 1, 2, 3. Can **downgrade** a gate's status
  (e.g. `pass → blocked_requires_human`) and append blocking issues.

Payload schema (the Hermes prompt sees): the stage's structured output,
references to prior state, the rubric (if applicable), and — when RLM is
wired in — an `rlm_evidence: list[{question, answer, citations, depth_reached,
llm_calls}]` field. Hermes-flagged unsupported claims then carry the RLM
citations that contradict them.

### 4.3 Provider chain — Hermes' "self-learning"
Hermes does not learn what *agents* are doing. It learns **which auditor backend
is healthy**. The default chain, reordered per-run from a tiny on-disk memory at
`<runs_root>/.hermes_adapter_memory.json`:

```
1. NousHermesProvider     (hermes-agent — pip or npm install)
2. ClaudeAuditProvider    (ANTHROPIC_API_KEY)
3. ClaudeCodeSdkProvider  (Claude Code subscription)
4. OpenAIAuditProvider    (OPENAI_API_KEY)
5. CodexCliProvider       (Codex CLI subscription)
```

For each provider in chain order:
- `is_available()` — gate on env/install presence
- `call(prompt)` → raw text
- Three-strategy JSON extraction: triple-backtick fence → first balanced
  `{ ... }` → strip-prose prefix
- Success → update memory (`last_successful_provider = name`), return report
- Failure → increment `consecutive_failures`; after 3 → quarantine; try next
- Whole chain exhausted → status `unavailable`. **Never silently substitute
  "ok".** Failure of the audit layer is itself an observable signal.

### 4.4 What Hermes is not
- Not a verifier of *correctness* against the paper (that's the rubric +
  RLM).
- Not a re-runner of the experiment.
- Not a gatekeeper with its own success criteria — it can only downgrade gates
  the pipeline already defined.

### 4.5 Relationship to RLM
**Hermes consumes RLM, RLM never calls Hermes.** Keep the audit layer thin.
RLM is the grounded-evidence primitive; Hermes uses that evidence to make its
verdict verifiable. Reversing the dependency would tangle two layers that do
different jobs.

---

## 5. Cross-cutting contracts (the bits that hold everything together)

- **Mandatory citations.** Every agent decision carries a citation list. The UI
  surfaces them; Hermes audits against them.
- **Dynamic confidence thresholds.** Gates use a complexity-scaled threshold,
  not a fixed number — long, ambiguous papers raise the bar.
- **Assumption ledger.** When an agent fills a gap that the paper didn't
  specify, it writes to `assumption_ledger.json` with severity and rationale.
- **Cost ledger.** Every LLM call writes a line to `cost_ledger.jsonl`
  (provider, model, tokens, USD). Hard caps via `--max-usd`.
- **Telemetry.** `agent_telemetry.jsonl` captures per-call latency, retries,
  token counts. Cheap to grep, structured for replay.
- **Run budget.** A single `RunBudget` object tracks remaining wall clock,
  remaining USD, remaining RLM calls. Primitives check it before doing
  expensive work.

---

## 6. Rebuilding in another framework (e.g. DSPy)

A direct port maps cleanly:

| Concept here | DSPy analogue |
|---|---|
| Stage agent with structured output | `dspy.Module` with a `Signature` |
| Static prompt + JSON schema | `Signature` with typed input/output fields |
| RLM query (`Cited[T]`) | A `dspy.Module` that recursively calls sub-modules; output type `Cited[T]` becomes a Pydantic model |
| Question pack (Pattern A) | A list of `Signature`s composed before the main module |
| Tool-mode RLM (Pattern B) | `dspy.ReAct` with `rlm_query` as a tool |
| Rubric verifier | `dspy.Module` that maps `(rubric, evidence) → scores`; can be optimized with `BootstrapFewShot` against PaperBench gold scores |
| Hermes auditor | A separate, non-optimized `dspy.Module` whose signature is `(agent_output, prior_state, rlm_evidence) → audit_report` |
| Provider chain | DSPy's `LM` indirection plus a small "try-each-in-order" wrapper |
| 14-stage state machine | Either: orchestrator Python calling modules in order (Pattern A), or a root `dspy.ReAct` whose tools are the primitives (the pivot architecture) |
| SQLite event store + JSONL ledgers | Keep as-is; DSPy doesn't dictate persistence |
| Sandbox runtime | Keep as-is; DSPy doesn't dictate execution |

**What DSPy gives you for free that this codebase hand-rolls:**
- Typed signatures (replaces hand-written JSON schemas + extraction)
- Compile-time prompt optimization against a labeled dev set (PaperBench scores
  are the natural labels)
- A trace object per call (replaces ad-hoc telemetry decoration)

**What DSPy does not solve:**
- The RLM recursion shape — you still write the recursive module yourself.
- The Hermes provider chain — DSPy assumes a single `lm`; the failover wrapper
  is your code.
- Long-lived subprocess + SSE bridge + sandbox process model — orthogonal.

**Suggested build order in a DSPy rewrite:**
1. Define signatures for each primitive (paper-understanding, environment-
   detective, planner, baseline-implementer, rubric-verifier). Pin output
   schemas.
2. Build the RLM module as a stand-alone `dspy.Module[Cited[T]]`. Unit-test
   with a fake LM.
3. Wire Pattern A: orchestrator Python calls signatures in stage order, RLM
   feeds each.
4. Bring up Hermes as a separate module + provider-chain wrapper.
5. Add the rubric loop and the build-and-repair loop on top of the linear
   spine.
6. Once stable, swap the orchestrator for a `dspy.ReAct` root and lift the
   stage signatures into REPL-callable tools (the pivot architecture).
7. Use `BootstrapFewShot` (or `MIPRO`) to optimize the rubric-verifier and
   planner signatures against the PaperBench corpus.

---

## 7. Decision log seeds (preserve these even if the framework changes)

- Citations are mandatory, not optional.
- Negative results are product features; the research map ships with the
  reproduction.
- Verification (rubric) and oversight (Hermes) are **separate layers**, even
  though both are "checking the agents."
- RLM is the only acceptable way to ground claims in long source text. No
  vector-store retrieval substitute.
- Hermes audits **never silently substitute "ok"** when the provider chain is
  exhausted. The failure is itself a signal.
- The sandbox time budget is the planner's input, not a post-hoc clamp.
- Stage count is held at 14 in the current substrate; loops (rubric,
  build-repair) happen *inside* existing stages rather than minting new ones.
  In the pivoted substrate the stage concept goes away entirely.
- Pattern A first (deterministic cost), Pattern B second (agent autonomy where
  it earns its keep).

---

*Last updated: 2026-05-20.*
