# Implementation Prompt — Rubric-Driven Reproduction Harness (`rdr`)

_Self-contained executable prompt. Written 2026-05-22. Paste this (or run
"read this file and execute it") in a fresh session to implement the `rdr`
harness._

---

## Mission

Implement a new **rubric-driven paper-reproduction harness** — a new `rdr` run
mode — for the OpenResearch / ReproLab repo (`/home/abheekp/openresearch`), per
the approved design spec:

**`docs/superpowers/specs/2026-05-22-rubric-driven-harness-design.md`** — read it
first and in full. It is the source of truth for the architecture, the 8
components, the context-engineering methodology, reliability, testing, success
criteria, and the session context (its §12).

The goal: a robust, reliable harness that scores **higher than the current
`rlm` harness** on PaperBench papers by making the **exact PaperBench rubric**
the spine of the reproduction.

## Why this exists (one paragraph)

The current RLM harness scores poorly — run 1: 0.366, run 2b: 0.079, run 3:
looped to 0.0 — because of four failure modes: a wandering free-form root loop,
degenerate/shallow baselines, under-scoping, and the rubric being used only to
score at the end. The `rdr` design kills all four: a **deterministic Python
controller** (no wandering), the **rubric tree decomposed into work-clusters**
(every leaf is a controller obligation → no under-scoping), **scoped Claude
agents** with **precisely-engineered context windows** (no degenerate baselines),
and the rubric **drives** the work (not just scores it).

## Orientation — read before writing code

1. The design spec (above).
2. `CLAUDE.md` — repo conventions, commands, architecture.
3. Code to **REUSE** (do not rewrite):
   - `backend/agents/rlm/primitives.py` — the 9 primitives (become agent tools);
     `run_experiment` / `_execute_in_sandbox` (the hardened sandbox).
   - `backend/evals/paperbench/leaf_scorer.py` — `score_reproduction`.
   - `backend/evals/paperbench/bundle.py` — `load_paperbench_bundle`.
   - `backend/agents/rlm/report.py` — report building; `backend/agents/rlm/run.py`
     — the run-entry pattern to parallel; `backend/agents/rlm/context.py`
     (`RunContext`).
   - `backend/agents/baseline_implementation.py` (`run_with_sdk`) — the existing
     Claude-SDK sub-agent; model the Reproduction Agent on it.
4. Fixture rubric: `third_party/paperbench/sequential-neural-score-estimation/rubric.json`
   (node schema `{id, requirements, weight, sub_tasks, task_category,
   finegrained_task_category}`, ~6 levels deep, 92 leaves, 7 top-level areas).

## Working discipline

- Use the `/iterate` skill. Root-level fixes — one canonical abstraction + a
  guard test, not scattered patches. Test what you change.
- Each phase below is `step → verify`. Do not begin a phase until the previous
  phase's tests are green.
- Opus plans, executes judgment-heavy code, and reviews every diff; delegate
  well-specified bounded implementation to Sonnet subagents where it helps.
- Doc-update contract: update `CHANGELOG.md` (the new mode), `system_overview.md`
  (the new architecture), `learn.md` (any lesson). Keep `progress.md` current.

## Constraints (must honor)

- **Git remote**: commit/push to `origin` (the `openresearch` repo) — **never**
  the `replix` remote.
- **Commit messages**: **no** `Co-Authored-By` / AI-attribution trailer.
- **Commit granularity**: infrequent — few substantial commits at milestones.
- **Branch**: `rlm_rubric_orchestration`.
- **Runs are serial** — never run two paper-runs (or a run + leaf-scoring)
  concurrently; the Featherless leaf-scorer has a 4-unit concurrency cap.
- **Corpus-leak redaction** — the paper corpus must never reach the SSE stream or
  the event store; redact at every egress (reuse the existing redaction).

## Phased implementation plan

### Phase 0 — Scaffolding
- Create `backend/agents/rdr/` and `tests/rdr/` (with `__init__.py` / conftest
  as the repo pattern requires).
- `rdr/models.py` — dataclasses: `RubricLeaf`, `WorkCluster`, `CitedSection`,
  `AgentContext`, `Artifacts`, `RdrResult` (shapes in design spec §4).
- **Verify**: the package imports clean; `pytest tests/rdr/` collects (even if
  empty).

### Phase 1 — Rubric Decomposer
- `rdr/decomposer.py` — `decompose(rubric_tree: dict, *, max_leaves_per_cluster=12)
  -> list[WorkCluster]`:
  - Walk the `rubric.json` tree; form clusters at coherent mid-level nodes; split
    any node whose leaf-count exceeds the cap.
  - Parse paper citations from each leaf's `requirements` (regex: "Section N",
    "Appendix X[.Y]", "Table N", "Figure N").
  - Tag clusters by dominant `task_category`; order Code-Development →
    Code-Execution → Result-Analysis, then descending weight; populate `depends_on`.
- **Test** `tests/rdr/test_decomposer.py` — fixture = the sequential-neural
  `rubric.json`: every leaf lands in exactly one cluster; clusters are
  category-ordered; citations parsed; no cluster exceeds the cap.
- **Verify**: `pytest tests/rdr/test_decomposer.py` green.

### Phase 2 — Context Engineer
- `rdr/context_engineer.py` — `build_context(cluster, *, paper, artifacts,
  prior_scores, token_budget) -> AgentContext` (design spec §4.3, §6):
  - `leaf_contract`: cluster leaves verbatim + weights.
  - `paper_sections`: resolve each citation to a section of
    `parsed_full_text.txt`; semantic-retrieval fallback (reuse the indexer) for
    un-cited leaves.
  - `dependency_artifacts`: the dependency closure from `artifacts`.
  - `prior_feedback`: on repair, the failed leaves + scorer justifications.
  - Token-budget: trim/summarize lowest-priority content first.
- **Test** `tests/rdr/test_context_engineer.py` — leaves appear verbatim; a cited
  section is pulled; budget respected; repair feedback present when `prior_scores`
  is given.

### Phase 3 — Reproduction Agent
- `rdr/agent.py` — `async reproduce(agent_context, *, ctx) -> Artifacts`. A
  Claude SDK agent (model: Opus, configurable), tools = the 9 primitives + a
  `paper_search` escape-hatch tool. Fail-soft (agent error → empty `Artifacts`,
  cluster marked failed). Model on `baseline_implementation.run_with_sdk`.
- **Test** `tests/rdr/test_agent.py` — contract test with a mocked SDK: an
  `AgentContext` in → a well-formed `Artifacts` out.

### Phase 4 — Controller + run entry
- `rdr/controller.py` + `rdr/run.py` — `run_pipeline_rdr(project_id, runs_root,
  bundle, *, ctx) -> RdrResult` (design spec §4.2):
  decompose → per-cluster (build_context → reproduce) → write project →
  `run_experiment` → `score_reproduction` → repair loop (capped at
  `max_repair_iterations`) → `build_report`.
  Per-cluster checkpointing (resume-safe); per-agent/per-sandbox deadlines
  (reuse `_timeout_for`); per-cluster fail-soft; deterministic termination
  (the controller assembles the report — no LLM `FINAL_VAR`).
- **Test** `tests/rdr/test_controller.py` — mocked agents (fixed artifacts):
  the full loop runs; repair triggers on a weak cluster; the report is assembled.

### Phase 5 — Wire the run mode + e2e
- `backend/cli.py` — add `reproduce --mode rdr`; `scripts/rdr_paperbench.py`
  launcher (parallel `scripts/rlm_paperbench.py`).
- `tests/rdr/test_rdr_offline_e2e.py` — deterministic end-to-end with mocked
  agents — the regression backbone.
- Smoke e2e: run one small bundle paper for real; assert completion + all DC#4
  artifacts + a non-degenerate score.

### Phase 6 — Docs + milestone commit
- Update `CHANGELOG.md`, `system_overview.md`, `learn.md`, `progress.md`.
- One milestone commit (no co-author trailer, to `origin`, branch `merge`).

## Verification / done criteria

- `pytest tests/rdr/` green; the full suite (`pytest tests/ -n auto`) green.
- A real `rdr` run on a PaperBench bundle paper completes, produces all four
  #62 DC#4 artifacts, and scores **higher than the `rlm` baseline (≈0.37)**.
- Zero loop/crash failures; honest partials recorded where compute-capped; no
  fabricated metrics.

## Note on the prior session (already closed out)

The debug-and-harden session that produced this spec is fully closed out:
GitHub **issue #62** (Phase 5 — end-to-end PaperBench runs) is **closed** — all
four done-conditions met by run 1 (`sequential-neural-score-estimation`, leaf
0.366) and run 2b (`mechanistic-understanding`, leaf 0.079). The session's
seven commits were squashed into one on `origin/main`. The `rdr` implementation
starts fresh from that `main` on the `rlm_rubric_orchestration` branch —
nothing here is blocked.
