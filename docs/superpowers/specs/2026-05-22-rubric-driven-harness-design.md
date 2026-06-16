# Rubric-Driven Reproduction Harness (`rdr`) — Design Spec

_Date: 2026-05-22 · Status: **implemented**._

This spec is self-contained: §1–§11 are the design; §12 captures the session
context and standing instructions preserved from the original implementation handoff.

---

## 1. Motivation

The current RLM harness (`backend/agents/rlm/`, `--mode rlm`) produces
low-quality reproductions. Observed this session:

| Run | Paper | Leaf score | Outcome |
|-----|-------|-----------:|---------|
| run 1 | sequential-neural-score-estimation | 0.366 | partial |
| run 2b | mechanistic-understanding | 0.079 | failed (honest) |
| run 3 | GoRL (arXiv 2512.02581) | 0.0 | looped — never reproduced |

Four root failure modes:

1. **Wandering orchestration** — the free-form RLM root loop wanders. Run 3
   burned all 21 root iterations calling `understand_section` and never reached
   `detect_environment` / `implement_baseline` / `run_experiment`.
2. **Degenerate baselines** — the code sub-agent writes shallow/synthetic
   implementations. Run 2b built a *synthetic* toxicity dataset that scored 0.0
   toxicity throughout — it "ran" but did not reproduce the paper's experiment.
3. **Under-scoping** — the baseline covers a slice; the rubric grades the whole
   paper. Run 1 implemented 2 of 9 benchmark tasks.
4. **Rubric is end-of-pipeline only** — the agent is graded by the PaperBench
   rubric but never *sees* it while working; it optimizes "reproduce the paper"
   (vague) instead of the concrete gradable leaves.

## 2. Decisions (from brainstorming)

| Question | Decision |
|----------|----------|
| Root orchestrator model | **Claude** (via the Claude Code OAuth the sub-agent already uses) — not Featherless Qwen3-Coder. |
| Rubric's role | **Rubric-driven** — the exact PaperBench rubric tree is the spine of the run. |
| Compute budget | **Local GPU when available**, wall-clock-bounded; compute-heavy papers → honest partials. |
| Orchestration approach | **C — Hybrid**: a deterministic Python controller orchestrating scoped Claude agents. |

## 3. Architecture

```
  PaperBench bundle:  paper.md  +  official rubric.json
                 │
                 ▼
  ┌── RUBRIC-DRIVEN CONTROLLER  (deterministic Python) ──────┐
  │                                                          │
  │  Decompose rubric tree ──► ordered work-clusters          │
  │                                                          │
  │  for each cluster (Code-Dev → Exec → Result-Analysis):    │
  │     Context Engineer ──► Reproduction Agent (Claude)      │
  │        (leaves, paper          │                         │
  │         sections, prior     artifacts (code, configs)    │
  │         artifacts, scores)                               │
  │                                                          │
  │  assemble ─► Sandbox Executor ─► Leaf Scorer (exact rubric)│
  │                                       │                  │
  │              repair weak clusters ◄───┘  (capped loop)   │
  │                                                          │
  └──────────────► final_report.{json,md} + DC#4 artifacts ──┘
```

A deterministic Python **Controller** owns all control flow. **Reproduction
Agents** (Claude) do the open-ended reproduction work — one per rubric
work-cluster — each given a precisely-engineered context window. The official
PaperBench rubric tree is the backbone: it is decomposed into work-clusters, and
every leaf is a controller obligation (attempted, scored, repaired-if-weak).

The PaperBench `rubric.json` node schema (verified on the
sequential-neural-score-estimation bundle): `{id, requirements, weight,
sub_tasks, task_category, finegrained_task_category}`. The tree is ~6 levels
deep, 92 leaves, 7 top-level areas; `task_category ∈ {Code Development, Code
Execution, Result Analysis}` (inner grouping nodes have `task_category: null`).

## 4. Components

5 new (`backend/agents/rdr/`), 4 reused.

### 4.1 Rubric Decomposer — `rdr/decomposer.py` (new)

Parses the official `rubric.json`; cuts the tree into agent-sized **work-clusters**
at coherent mid-level nodes (the ~7 top-level areas, split further when a subtree
is too large for one agent context); tags each by dominant `task_category`;
parses paper citations out of each leaf's `requirements` text ("Appendix E.1",
"Section 5", "Table 2"); orders clusters by category dependency (Code Development
→ Code Execution → Result Analysis) then descending weight.

```python
# rdr/models.py
@dataclass(frozen=True)
class RubricLeaf:
    id: str
    requirements: str          # verbatim rubric text — the gradable contract
    weight: float
    task_category: str         # Code Development | Code Execution | Result Analysis
    paper_citations: list[str] # parsed from `requirements`

@dataclass
class WorkCluster:
    id: str
    title: str                 # the cluster node's `requirements` text
    leaves: list[RubricLeaf]
    dominant_category: str
    weight: float              # Σ leaf weights
    depends_on: list[str]      # cluster ids
    paper_citations: list[str] # union of leaf citations

# rdr/decomposer.py
def decompose(rubric_tree: dict, *, max_leaves_per_cluster: int = 12) -> list[WorkCluster]:
    """Official rubric.json tree -> ordered, dependency-sorted work-clusters."""
```

### 4.2 Controller — `rdr/controller.py` (new)

Deterministic Python. Iterates clusters in dependency order; owns the stage
sequence, the sandbox lifecycle, and the score→repair loop. **No LLM in the
control flow.** Checkpoints after every cluster (resume-safe). Assembles the
final report from structured artifacts — there is no LLM "final answer" to parse.

```python
# rdr/run.py
async def run_pipeline_rdr(project_id, runs_root, bundle, *, ctx) -> RdrResult:
    rubric   = bundle.rubric()
    clusters = decompose(rubric)
    artifacts: dict[str, str] = {}
    for cluster in clusters:                       # dependency order
        agctx = build_context(cluster, paper=paper, artifacts=artifacts, prior_scores=None)
        artifacts |= await reproduce(agctx, ctx=ctx)
    write_project(run_dir, artifacts)
    run_experiment(run_dir, env_id, ctx=ctx)       # reused, hardened
    scores = score_reproduction(rubric, run_dir, llm_client=ctx.llm_client)
    for _ in range(ctx.max_repair_iterations):
        weak = [c for c in clusters if cluster_score(c, scores) < repair_target]
        if not weak: break
        for cluster in weak:
            agctx = build_context(cluster, paper=paper, artifacts=artifacts, prior_scores=scores)
            artifacts |= await reproduce(agctx, ctx=ctx)
        write_project(run_dir, artifacts)
        run_experiment(run_dir, env_id, ctx=ctx)
        scores = score_reproduction(rubric, run_dir, llm_client=ctx.llm_client)
    return build_report(run_dir, rubric, scores, artifacts)
```

### 4.3 Context Engineer — `rdr/context_engineer.py` (new)

Per cluster, deterministically assembles the minimal-correct context window.
See §6 for the methodology.

```python
@dataclass
class AgentContext:
    cluster: WorkCluster
    leaf_contract: str                 # verbatim leaves + weights, formatted
    paper_sections: list[CitedSection] # retrieved by citation, semantic fallback
    dependency_artifacts: dict[str, str]
    prior_feedback: str | None         # on repair: failed leaves + scorer justifications
    working_summary: str

def build_context(cluster, *, paper, artifacts, prior_scores, token_budget=...) -> AgentContext:
```

### 4.4 Reproduction Agent — `rdr/agent.py` (new wrapper)

One Claude agent invocation per cluster. Defaults to Claude **Opus** for code
quality (configurable). Given the engineered context + the existing 9 RLM
primitives as tools (`understand_section`, `extract_hyperparameters`,
`detect_environment`, `build_environment`, etc.) + a `paper_search` escape-hatch
tool. Returns `Artifacts` (relative-path → file content, run commands, notes).
Fail-soft: an agent error marks the cluster failed and returns empty artifacts.

### 4.5 Repair Loop — part of `rdr/controller.py` (new)

After scoring, clusters whose weighted leaf score is below `repair_target` are
re-dispatched to a Reproduction Agent with `prior_feedback` (the exact failed
leaves + the leaf scorer's justification verbatim). Capped at
`max_repair_iterations` (config, default 2–3).

### 4.6 Sandbox Executor — `backend/agents/rlm/primitives.py` (reuse)

`run_experiment` / `_execute_in_sandbox`, already hardened this session (Fix
A/B/C — see §12). The Controller chooses smoke-vs-full scale per the local-GPU
budget for `Code Execution` leaves.

### 4.7 Leaf Scorer — `backend/evals/paperbench/leaf_scorer.py` (reuse)

`score_reproduction` — flattens the rubric tree, batch-LLM-grades leaves,
weighted roll-up. Grades against the **exact** PaperBench bundle rubric. Already
the in-loop scorer (issue I2).

### 4.8 Report Builder — `backend/agents/rlm/report.py` (reuse / adapt)

Emits `final_report.{json,md}` and the issue-#62 DC#4 artifacts
(`repl_state.pickle` equivalent, per-cluster `iterations/`). Verdict reconciled
against the leaf score (issue I9).

## 5. Data flow

`ingest` (reuse `ResolvingParser`) → `decompose` rubric → **per cluster**
[`build_context` → `reproduce` → collect artifacts] → assemble project →
`run_experiment` → `score_reproduction` → repair weak clusters (capped loop) →
`build_report`.

## 6. Context-engineering methodology

Every agent invocation gets a precisely-assembled context window — never a dump,
never a free-form variable the model must slice itself. The Context Engineer is
deterministic Python; per cluster it builds:

1. **Rubric leaves, verbatim** — the exact `requirements` text + weights of every
   leaf in the cluster. The agent's *contract*: "graded on exactly these N
   requirements; weights say what matters most."
2. **Paper sections, by rubric citation** — the rubric `requirements` explicitly
   cite paper locations ("as described in Appendix E.1", "results in Section 5").
   Parse those citations; pull *exactly those sections* from `parsed_full_text.txt`.
   Semantic retrieval over the indexer is the fallback when a leaf cites nothing.
3. **Dependency artifacts** — code/configs from the clusters this one depends on
   (Result-Analysis clusters see the Code-Development clusters' code). Dependency
   closure only.
4. **Prior scores + failure feedback** (repair only) — the exact failed leaves +
   the leaf scorer's justification verbatim.
5. **Compact working-set summary** — project structure + key decisions so far.

Methodology principles:
- *Rubric-anchored* — leaf requirements are the spine of every window.
- *Deterministic & token-budgeted* — pure-Python assembly, reproducible and
  testable; a focused window beats an overloaded one.
- *Citation-following retrieval* — the rubric says which paper section each leaf
  concerns; follow it, don't make the model guess.
- *Escape hatch* — the agent keeps a `paper_search` tool to pull more context if
  the engineered window is insufficient.
- *Compounding feedback* — each repair iteration's window is strictly richer;
  the agent never restarts cold.

## 7. Reliability & error handling

- **Deterministic control** — the Controller is pure Python; run 3's
  wander/loop/iteration-cap failure mode is *structurally impossible*.
- **Deterministic termination** — the Controller ends the run and *assembles*
  the report from structured artifacts; no LLM `FINAL_VAR` emission, so run 3's
  "unparseable RLM response" cannot recur.
- **Per-cluster fail-soft** — one cluster's agent failing marks that cluster
  failed and continues; the report records honest partial coverage.
- **Bounded everything** — per-agent and per-sandbox deadlines (reuse
  `_timeout_for`); the repair loop is capped.
- **Compute-honest** — execution/result leaves beyond the local-GPU budget get a
  smoke-scale run and an honest partial; no fabricated numbers.
- **Reused hardening** — run_experiment Fix A/B/C, corpus-leak redaction at
  egress, atomic per-cluster checkpointing.

## 8. Testing strategy

- **Unit per component** — Decomposer (fixture `rubric.json` → expected
  clusters), Context Engineer (cluster → window: leaves verbatim, citations
  followed, token budget respected), Repair Loop selection, Controller
  sequencing. All deterministic.
- **Offline e2e** — a mode where Reproduction Agents are mocked (fixed
  artifacts): exercises Decomposer + Controller + Context Engineer + scoring +
  repair end-to-end, no LLM, no Docker — the regression backbone.
- **Smoke e2e** — one small bundle paper, real agents, asserts the run completes
  with all DC#4 artifacts and a non-degenerate score.
- Contract tests over `WorkCluster` / `AgentContext` / `Artifacts` shapes.

## 9. Migration, rollout, scope

- **New code** — `backend/agents/rdr/` (~5–7 focused modules: `models.py`,
  `decomposer.py`, `context_engineer.py`, `agent.py`, `controller.py`, `run.py`).
  Tests under `tests/rdr/`.
- **Reused** — ingestion (`ResolvingParser`), the 9 primitives (as agent tools),
  the sandbox, the leaf scorer, the report builder, the SQLite event store, the
  SSE bridge.
- **New opt-in run mode** `rdr` — wired into `backend/cli.py` (`reproduce
  --mode rdr`) and a `scripts/rdr_paperbench.py` launcher paralleling
  `scripts/rlm_paperbench.py`. `rlm` and `rlm-pure` stay untouched —
  backward-compatible; existing tests/runs unaffected.
- **Scope** — `rdr` targets **PaperBench bundle papers** (official rubric
  required). arXiv papers can use `rdr` with a generated rubric, secondarily.

## 10. Success criteria

- `rdr` produces **higher leaf scores than `rlm`** on the PaperBench bundle
  papers (beat the current best ≈ 0.37; target the PaperBench published-agent
  range).
- **Zero loop/crash failures** — every run completes with a final report + all
  DC#4 artifacts.
- **Every rubric leaf is provably attempted** (a controller obligation) and
  scored; honest partials where compute-capped, no fabricated metrics.
- The offline e2e is deterministic and green.

## 11. Non-goals

- Not replacing `rlm` / `rlm-pure` — `rdr` is additive.
- Not RunPod / multi-GPU compute — local-GPU budget.
- Not primary support for arXiv papers (no official rubric) — secondary mode.
- Not a re-write of ingestion, the sandbox, or the leaf scorer — those are reused.

## 12. Session context & standing instructions (compaction-survival)

This design was produced inside a long RLM debug-and-harden session. Context a
fresh session needs:

### 12.1 The project

OpenResearch / OpenResearch — an agent pipeline that reproduces research papers
end-to-end and benchmarks the reproduction against the paper's claims. Backend:
Python 3.14 / FastAPI. The RLM orchestrator (`backend/agents/rlm/`) is the
current production path (`--mode rlm`); see `CLAUDE.md`, `system_overview.md`,
`docs/design/rlm-pivot-brief.md`.

### 12.2 What this session did (commits on branch `merge`)

- `2630a77` — P0 correctness/honesty (I1 metrics, I2 tree-aware rubric scoring,
  I9 verdict reconciliation, #62 DC#4 checkpoint artifacts).
- `4e7b4a4` — catalogue hardening I5–I13 + test-registry isolation.
- `52625d6` — run_experiment Bug A/B/C + I3 (I3 later reverted).
- `d656c7d` — I4 (workspace `paper_text` from the parser blob) + idempotency-test
  deflake.
- `c22feb7` — **reverted I3** (`_PAPER_GROUNDING` root-prompt section caused the
  run-3 understand-loop; see `learn.md` 2026-05-22).

**run_experiment Bug A/B/C** (in `52625d6`, verified live by run 2b): A —
`_execute_in_sandbox` dropped stderr; fixed via `_combine_command_output`. B —
the experiment ran an image built before the code existed; `run_experiment` now
rebuilds from `ctx.project_dir/Dockerfile`. C — the sandbox ran
`network_disabled`; `_execute_in_sandbox` now enables network for the experiment
container (user-approved). Test suite: 1252 passing.

Run 3's re-run (after the I3 revert) also surfaced a hard limit: the Featherless
Qwen3-Coder plan caps context at 49 152 tokens, which the RLM root's accumulated
prompt exceeds within a few iterations (`context_length_exceeded`, HTTP 400).
This independently validates §2's decision to run the root on Claude (200k+
window) — the deterministic, token-budgeted Context Engineer (§6) is also the
structural fix for unbounded context growth.

### 12.3 GitHub issue #62 — "Phase 5 end-to-end runs" (closed)

#62 is **closed** — all four done-conditions met by run 1 + run 2b: ≥2
PaperBench papers with completed runs + `final_report.md`; real rubric scores
(0.366 and 0.079); candidate lists differ between papers (4 vs 3 improvements,
distinct content); run dirs contain `final_report.{json,md}`,
`repl_state.pickle`, `iterations/`. Run 3 (GoRL, arXiv 2512.02581) was a bonus
run — an honest documented failure (see §12.2). The session was squashed to a
single commit on `origin/main` whose `Closes #62` keyword closed the issue.

### 12.4 Standing instructions / constraints (must honor)

- **Git remote**: commit/push to `origin` (the `openresearch` repo) — **never**
  the `replix` remote.
- **Commit messages**: **no** `Co-Authored-By` / AI-attribution trailer.
- **Commit granularity**: infrequent — few substantial commits at milestones.
- **Branch**: the debug-and-harden session is squashed onto `origin/main`; the
  `rdr` implementation happens on the **`rlm_rubric_orchestration`** branch off
  that `main`.
- **RLM/`rdr` runs are serial** — the Featherless backend has a 4-unit
  concurrency cap; never run two paper-runs concurrently.
- **Corpus-leak redaction** — the paper corpus must never reach the SSE stream
  or the event store; redact at every egress.
- **Working discipline**: use the `/iterate` skill — root-level fixes (one
  canonical abstraction + a guard test, not scattered patches); test what you
  change; keep `learn.md` / `CHANGELOG.md` / `system_overview.md` current
  (the doc-update contract).
- **Trackers**: keep `progress.md` and `runlog.md` updated.
- **Delegation**: Opus does planning + code execution + design review; Sonnet
  sub-agents execute well-specified bounded tasks; cross-turn waiting on long
  runs is owned by the main session via background tasks (a subagent cannot
  survive a multi-hour wait).
