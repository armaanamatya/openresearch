# Optimization Platform — Design Spec

> Status: ratified 2026-05-27 (cheap-run phase executing today)
> Author: working notes
> Related: `2026-05-25-gepa-prompt-optimization-design.md`,
> `2026-05-26-gepa-phase0-audit.md`,
> `2026-05-26-systemic-failure-resolution-design.md`

## 1. Context — why

ReproLab just shipped GEPA (Lane B — improvement-prompt mutation). The
question now is what *else* to build in the optimization stack. The
candidate landscape after a research pass:

| Method | Output | Search level | Status here |
|---|---|---|---|
| **GEPA** (arXiv 2507.19457) | mutated prompt strings | text component | shipped |
| **HALO** (context-labs) | diagnostic report → human | trace decomposition | candidate |
| **SkillOpt** (arXiv 2605.23904) | `best_skill.md` runtime-loaded | skill doc, train/val/test split | candidate |
| **AFlow** (arXiv 2410.10762) | MCTS workflow graph | code-represented graph | deferred |
| **ADAS** (arXiv 2408.08435) | Meta Agent Search over code | whole agent code | deferred |

Three structural facts decide the plan:

1. **We have 10 historical runs** (`runs/`), 7 with `final_report.json`.
   This is too few for HALO's "mine production traffic for systemic
   patterns" methodology at full strength, but enough for a programmatic
   first pass that quantifies the trace signal.
2. **Our recent systemic fixes** (Lane G Rubric Guard, Lane H Forced
   Iteration, dynamic GPU, macOS Keychain OAuth) **were code/infra, not
   prompt-content.** This is signal that our actual bottleneck class
   is HALO-shaped, not GEPA/SkillOpt-shaped.
3. **SkillOpt claims to beat GEPA in 52/52 cells** with a runtime-loaded
   skill doc, with the biggest gains *inside agentic loops* (+24.8 on
   Codex, +19.1 on Claude Code, > +23.5 direct chat). This is exactly
   our shape — RLM-driven agent loop. But it's a 2-day-old paper from
   the team that built it; no third-party reproduction yet.

We don't know which method actually moves Hermes-clamped score on paper
reproduction. **Nobody has benchmarked any of these on paper
reproduction.** The plan therefore prioritizes *measurement before
investment* and pipes evidence from the cheap phase into the
gating decisions for the expensive phases.

## 2. Budget envelope

- **Right now**: $0 LLM API, $7 RunPod (held in reserve)
- **After budget approvals**: ~$50–100 LLM API for a single bake-off pass
- All Phase 4+5 work is gated on Phase 3 evidence, which is gated on
  budget approvals

This means Phase 1 — the only deliverable that produces evidence on the
present budget — must be **programmatic**, not LLM-driven.

## 3. Phases

### Phase 0 — Shared scaffolding ($0)

Reusable across every lane:

- `backend/agents/optimization/cost_tracker.py` — `LMCostTracker` wraps
  `gepa.lm.LM` (or any LM client) and accumulates `total_cost`,
  `total_tokens_in`, `total_tokens_out` per logical optimization run.
  Persists to `<opt_run_dir>/cost.json` after each LM call.
- `backend/agents/optimization/base_adapter.py` — `BaseOptimizationAdapter`
  protocol defining the minimum a lane adapter must implement
  (`evaluate`, `surface_id`, `cost_tracker`). GEPA adapter and the
  future SkillOpt adapter both implement it.
- Directory tree: `findings/`, `skills/`, `proposed_mutations/`
  (the last already exists from GEPA).
- `scripts/optimize.py` — single CLI entry point with subcommands
  `mine`, `gepa`, `skillopt`, `bake-off`, `promote`. Each subcommand
  is a thin wrapper around the existing per-lane script. Goal:
  consistent surface for users + cron.

No tests at this layer — these are integration seams exercised by the
per-lane test suites.

### Phase 1 — Programmatic trace mining ($0) ★ THE CHEAP RUN

`scripts/mine_traces.py` — pure Python, no LLM calls. Reads every
`runs/<id>/` directory and emits a structured stats document:

**Inputs per run:**
- `final_report.json` — paper_id, status, scores, exit reason
- `final_report.md` — narrative summary (string slice, optional)
- `dashboard_events.jsonl` — event stream (primitive calls, repairs)
- `experiment_runs.jsonl` (where present) — `run_experiment` outcomes
- `cost_ledger.jsonl` (where present) — per-primitive spend
- `demo_status.json` — start/end timestamps for wall-clock
- `iterations/` — per-iteration checkpoints (count only, not contents)

**Aggregations produced:**

1. **Per-run summary table**: paper_id, status, Hermes-clamped score,
   total cost, wall-clock, exit reason, iteration count.
2. **Failure-mode counts**: across the corpus, count occurrences of
   each named failure (build failure, OOM, repair loop > N, FINAL_VAR
   refusal under forced-iteration, Hermes "caveat", Hermes "ungrounded",
   wall-clock-exceeded, budget-exceeded).
3. **Primitive call frequency**: across all runs, how often each of
   the 12 primitives was called, and the average wall-clock per call.
4. **Repair-loop hotspots**: which primitives accumulated > 3 repair
   attempts in any run; cluster by primitive + run.
5. **Top-K error clusters**: extract error strings from stderr logs
   and `run_experiment` failures; cluster via
   `difflib.SequenceMatcher` with threshold 0.85; report top K=10.
6. **Per-archetype rollup** (where ≥ 2 runs share archetype):
   mean Hermes, mean cost, dominant failure mode.

**Output**: `findings/2026-05-27-stats-001.md` — structured stats
followed by a narrative-findings section that the current Claude
session (no extra API cost) appends after reading the stats.

**Failure mode for the cheap run itself**: if stats reveal too little
signal to mine (e.g., all 10 runs are partial / wall-clock-exceeded with
no Hermes scores), that's *evidence to defer Phase 4* — kill HALO Lane D
and route the optimization budget to Lane E (SkillOpt) or GEPA polish.

### Phase 2 — GEPA hardening ($0)

Close the 3 gaps surfaced by the context7 audit in
`backend/agents/optimization/gepa_adapter.py`:

1. **Dict-shape reflective records** — `_to_gepa_reflective_record`
   currently returns `{"Inputs": str, "Generated Outputs": str,
   "Feedback": str}`. Canonical GEPA adapter shape uses dicts:
   `{"Inputs": {"archetype": ..., "rubric_overall_before": ...},
   "Generated Outputs": {"candidate_text": ..., "metrics": ...},
   "Feedback": str}`. Strings work; dicts are idiomatic and let the
   reflection LM parse fields cleanly.
2. **`gepa.lm.LM` instance for cost tracking** — `optimize_prompts_gepa.py`
   currently passes `reflection_lm` as a model-name string. Swap to
   `LM("openai/gpt-4.1")` and persist `total_cost` post-run into
   `validation_run.json` and `proposed_mutations/<id>/metadata.json`.
   Shared with Phase 0's `LMCostTracker`.
3. **`claude-oauth` reflection callable** — write
   `claude_oauth_reflection_lm(messages, **kwargs) -> str` that routes
   through `ClaudeOauthClient`, letting users with only the Claude
   subscription run Lane B at zero API cost.

Tests updated:
- `tests/test_gepa_reflective_shape.py` — pin the dict shape for all
  three keys; assert nested fields present.
- New `tests/test_gepa_cost_tracker.py` — verify cost persists across
  process boundaries.

No new LLM calls — Phase 2 is code + tests only.

### Phase 3 — Bake-off measurement (~$50 LLM + $5 RunPod, AWAITS BUDGET)

A `scripts/bake_off.py` runs three SDAR mini-reproductions:

| Cell | Description | Hermes |
|---|---|---|
| A | current `main`, SDAR mini scope | X |
| B | A + GEPA Phase-2 candidate applied | X' |
| C | A + Phase-1 hand-fixes applied | X'' |

Output: `findings/2026-05-27-bake-off-001.md` with Δ-Hermes per cell
and per-$ engineering. This is the gate for Phases 4 and 5.

### Phase 4 — Lane D production HALO miner (~$200/quarter, AWAITS Phase 1 + 3)

Only built if Phase 1 finds ≥ 5 distinct systemic patterns AND Phase 3
shows the hand-fixed cell C outperforms cells A and B.

`scripts/mine_traces.py` extended:
- Optional `--llm` flag — RLM pass on top of the programmatic stats,
  producing richer narrative findings
- GitHub issue emitter (one issue per finding, idempotent via finding
  hash)
- Findings → GEPA trainset hint pipeline (closed loop with Lane B)
- Weekly cron in `.github/workflows/halo-miner.yml`

### Phase 5 — Lane E SkillOpt (~$500–1000/training, AWAITS Phase 3)

Only built if Phase 3 shows GEPA hits a ceiling AND we believe
SkillOpt's "+24.8 agentic-loop bonus" transfers to reproduction.

> **Phase-1 evidence update:** `findings/2026-05-27-stats-001.md` §8
> Conclusion shows 7/8 runs in the current corpus never reach the
> improvement loop (async-cleanup bug + OOM dominate). Until those
> infra bugs are resolved, any prompt-class mutation (including
> SkillOpt-style runtime skills) is invisible to the rubric. Phase 5
> priority drops accordingly; revisit only after Phase 3 shows ≥ 3
> consecutive completed runs with Hermes ≥ 0.3.

Components:
- `backend/agents/optimization/skill_adapter.py` — `ReproLabSkillOptAdapter`
  mirroring our GEPA adapter pattern
- `backend/agents/optimization/skill_loader.py` — runtime loader that
  appends `skills/<archetype>.md` to the RLM root system prompt
- Per-archetype skill files (`skills/rl-agent.md`, `skills/diffusion.md`,
  `skills/optimizer.md`, ...) loaded based on detected archetype
- Cheap-rollout-unit design: this is a real sub-problem — likely
  one rubric-question evaluation on a cached intermediate state, not a
  full reproduction. Detailed in a Phase-5-specific spec written after
  Phase 3 evidence is in.

## 4. File layout (after Phase 0 + 1 + 2)

```
backend/agents/optimization/
  __init__.py                    # exists
  base_adapter.py                # NEW Phase 0
  cost_tracker.py                # NEW Phase 0
  eval_budget.py                 # exists
  eval_entrypoint.py             # exists
  gepa_adapter.py                # modified Phase 2
  mutable_regions.py             # exists
  prompt_overrides.py            # exists
  trace_minimizer.py             # exists
findings/
  2026-05-27-stats-001.md        # NEW Phase 1 output
proposed_mutations/              # exists from GEPA
skills/                          # NEW Phase 0 (empty until Phase 5)
scripts/
  mine_traces.py                 # NEW Phase 1
  optimize.py                    # NEW Phase 0 — unified CLI
  optimize_prompts_gepa.py       # exists (modified Phase 2)
  promote_gepa_mutation.py       # exists
tests/
  test_gepa_cost_tracker.py      # NEW Phase 2
  test_gepa_reflective_shape.py  # modified Phase 2 (dict-shape pin)
  test_trace_miner.py            # NEW Phase 1
docs/superpowers/specs/
  2026-05-27-optimization-platform-design.md   # this file
```

## 5. Decisions log

- **Why no SkillOpt in v1**: 2-day-old paper, no independent
  reproduction, our rollout cost is ~100× SkillOpt's benchmarks. Need
  Phase 3 evidence GEPA is insufficient before investing in a different
  paradigm.
- **Why no HALO miner with LLM in v1**: $0 LLM budget. Programmatic
  stats first; LLM pass is a v2 upgrade once budget arrives.
- **Why no AFlow / ADAS**: search-space-bigger methods; right tool only
  if both GEPA and SkillOpt prove insufficient. Phase 3 + 5 evidence
  required before considering.
- **Why hold $7 RunPod**: a test-of-test on RunPod (e.g., spinning up
  a local Qwen as a free root model) is engineering-heavy and produces
  results we can't trust. Save for a real SDAR mini-reproduction in
  Phase 3.
- **Why Phase 1 is the cheap run, not a GEPA validation**: GEPA was
  recently shipped and we already validated its plumbing works in
  `scripts/gepa_docker_smoke.sh` (56/56 tests inside container). The
  marginal value of another GEPA-plumbing validation is low; the
  marginal value of *any* evidence about cross-run failure patterns
  is high.

## 6. Success criteria

**Phase 0**: scaffolding files exist, no regressions in existing tests.

**Phase 1**: `findings/2026-05-27-stats-001.md` exists with:
- Complete per-run table for all 10 runs
- ≥ 3 named failure-mode buckets with counts
- ≥ 1 top-error cluster identified
- A narrative findings section with ≥ 3 actionable patterns

**Phase 2**: GEPA tests still pass (56/56 baseline) + new tests for
the dict shape and cost tracker. `claude_oauth_reflection_lm` is
importable and has a unit test (mocked, no live call).

**Phase 3+**: deferred; success criteria written when each is unblocked.
