# GEPA Prompt Optimization for ReproLab - Design Spec

> Status: draft proposal (2026-05-25)
> Author: working notes, not yet ratified
> Related: `docs/design/rlm-pivot-brief.md`, `backend/agents/rlm/system_prompt.py`,
> `backend/agents/prompts/improvement.py`, `backend/agents/prompts/baseline_implementation.py`

## 1. Context - why

ReproLab today has five large textual parameters that materially shape run
quality, and zero machinery to optimize them:

1. The RLM root system prompt - `backend/agents/rlm/system_prompt.py`
2. `IMPROVEMENT_ORCHESTRATOR_PROMPT` + `ADAPTIVE_POOL_GENERATION_PROMPT` +
   `ADAPTIVE_RERANK_PROMPT` - `backend/agents/prompts/improvement.py`
3. The `implement_baseline` Sonnet sub-agent prompt -
   `backend/agents/prompts/baseline_implementation.py`
4. The rubric leaf-scorer prompt - `backend/agents/prompts/rubric_verifier.py`
5. The Hermes audit prompt - internal to `backend/hermes_audit/client.py`

GEPA (arXiv 2507.19457; `pip install gepa`) is built to optimize textual
parameters using natural-language reflection on execution traces rather than
scalar reward signals. ReproLab fits the target regime: expensive rollouts,
scarce data, API-only models, rich textual feedback, and a rubric tree that is
hard to reduce to one scalar without losing information.

ReproLab already produces structured natural-language feedback on every
iteration: `weak_leaves` from `verify_against_rubric`, `repair_context` from
failed `run_experiment`, `forced_iteration` warnings, blanket-decline refusals,
and Hermes `unsupported_claims` + `evidence_refs`. Today these feed the next
iteration of a single run. Under GEPA they also become cross-run prompt
evolution signal.

## 2. What GEPA does

GEPA maintains a Pareto frontier of candidate prompts rather than a single best
candidate. Each iteration samples a candidate from the frontier, executes it on
a minibatch, captures execution traces, asks a reflection LM to diagnose failure
modes from the traces, proposes a mutation, and accepts the mutation if it
improves the frontier. It can also perform system-aware merge across two
Pareto-optimal candidates that excel on different examples.

Adapter contract: implement `evaluate(batch, candidate, capture_traces=False)`
and `make_reflective_dataset(candidate, eval_batch, components_to_update)`.
`evaluate` returns a GEPA `EvaluationBatch` with per-example `scores`,
`outputs`, and, when `capture_traces=True`, per-example `trajectories`. GEPA
handles selection, mutation, Pareto bookkeeping.

## 3. Integration surface - landing spots

### Lane GEPA-B - land first

Optimize `IMPROVEMENT_ORCHESTRATOR_PROMPT` in
`backend/agents/prompts/improvement.py`.

- Trainset row shape: `(current_results, rubric_scores, paper_archetype) ->
  (selected_hypothesis, post-run rubric score)`.
- Metric: Hermes-clamped absolute rubric score, with paired rubric delta used
  as trace metadata or a tie-breaker.
- ASI: `record_candidate_outcome` statuses, the `weak_leaves` the candidate was
  meant to address, and the before/after rubric area deltas.
- Cost: cheapest surface. It should not re-run full baselines unless validating
  a provisional mutation. Use cached prior run artifacts and one scoped
  candidate execution where needed.
- First real gate: a mutation accepted by stub/synthetic evaluation must pass at
  least one real `run_experiment` validation on a held-out paper before human
  review.

### Lane GEPA-A - land second

Optimize mutable components of the RLM root system prompt
(`build_system_prompt` in `backend/agents/rlm/system_prompt.py`).

Candidate mutable regions:

- `_DECOMPOSITION_EXAMPLE` - best initial mutation target.
- Optional hints/triage wording if telemetry shows repeated path-selection
  failure.
- Small local wording around decomposition strategy.

Immutable regions:

- RLM operating model.
- Algorithm-2 guard.
- `FINAL_VAR` JSON contract.
- forced-iteration repair policy.
- chat steering.
- heartbeat.
- GPU selection.
- model-specific addenda.
- security/sandbox instructions.

Metric: `final_report.rubric.overall_score * hermes_grounded_multiplier`.

ASI: primitive call sequence, `weak_leaves`, forced-iteration warnings,
blanket-decline refusals, Hermes `unsupported_claims`, and redacted
`evidence_refs.snippet`.

### Lane GEPA-C - land third

Optimize `BASELINE_IMPLEMENTATION_PROMPT` in
`backend/agents/prompts/baseline_implementation.py`.

- Trainset: papers with cached `plan` + `repair_context` traces.
- Metric: `run_experiment.success` and `metrics.json` contains all rubric
  required keys; failures score `0.0` with trace.
- ASI: CUDA OOM markers, `RubricGuardFailure` JSON, missing-artifact errors,
  preflight violations, schema-mismatch reports.
- Cache warning: this lane is invalid unless `implement_baseline` cache keys
  include candidate hash + surface, or the cache is disabled for candidate
  evaluations.

### Out of scope for v1

- Lane GEPA-D - rubric leaf-scorer prompt. Optimizing the grader risks turning
  it into a GEPA accomplice.
- Lane GEPA-E - Hermes audit prompt. Hermes is the anti-Goodhart oracle and
  must stay outside the optimization loop.

## 4. Adapter design

File: `backend/agents/optimization/gepa_adapter.py` (new).

```python
from gepa.core.adapter import GEPAAdapter
from backend.agents.rlm.run import run_pipeline_rlm


class ReproLabGEPAAdapter(GEPAAdapter):
    def __init__(self, *, surface: str, cache_dir: Path, max_usd_per_eval: float):
        self.surface = surface  # "improvement" | "root_system" | "baseline_agent"
        self.cache_dir = cache_dir
        self.max_usd_per_eval = max_usd_per_eval

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[ReproLabTrace, ReproLabOutput]:
        # 1. Instantiate the target prompt surface from candidate[<key>].
        #    Do not monkey-patch module globals: registry entries copy prompt
        #    strings at import time.
        # 2. For each paper in batch: run run_pipeline_rlm with budget cap
        #    and a candidate-scoped run directory/cache namespace.
        # 3. Return per-example scores, outputs, and traces when requested.
        ...

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[ReproLabTrace, ReproLabOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        # Assemble per-example natural-language trace:
        #   - rubric.weak_leaves
        #   - repair_context JSON from last failed run_experiment
        #   - forced_iteration / blanket_decline warnings
        #   - hermes.unsupported_claims + evidence_refs.snippet
        # Redact paper text, secrets, env vars, large stdout, and artifact bodies.
        # Return reflective examples keyed by component name.
        ...
```

Driver: `scripts/optimize_prompts_gepa.py`.

```python
import gepa

result = gepa.optimize(
    seed_candidate={"improvement_orchestrator": IMPROVEMENT_ORCHESTRATOR_PROMPT},
    trainset=load_paper_set(["2605.15155", "...", "..."]),
    valset=load_paper_set(["heldout_1", "heldout_2"]),
    adapter=ReproLabGEPAAdapter(surface="improvement", ...),
    task_lm="openai/gpt-5",
    reflection_lm="openai/gpt-5",
    max_metric_calls=100,
    use_merge=True,
    max_merge_invocations=5,
    run_dir="runs/_gepa/<timestamp>",
    cache_evaluation=True,
    seed=0,
)
```

### Adapter implementation constraints

- No global monkey-patching. Pass candidate text through explicit prompt
  overrides or run context. This is mandatory for Lane C because
  `AGENT_REGISTRY` copies `BASELINE_IMPLEMENTATION_PROMPT` into an `AgentSpec`
  at import time.
- Candidate-scoped isolation. Each `(optimization_run, candidate_id, paper)`
  evaluation gets its own directory under `runs/_gepa/<timestamp>/evals/`.
- Candidate-scoped caches. Any score-affecting cache includes candidate hash and
  surface, or is disabled.
- Stable scoring. Per-example GEPA scores are `[0, 1]` floats.
- Failure handling. Paper-level failures return score `0.0` plus trajectory;
  only systemic setup failures raise.
- Trace minimization. `make_reflective_dataset` emits concise redacted JSON, not
  raw logs.

## 5. Cost model

| Lane | Eval cost per row | Rows/run | Reflection cost | Total/optimization |
|---|---:|---:|---:|---:|
| B improvement | ~$0.05 | ~50 | ~$5 | ~$10 |
| A root prompt | ~$1.00 + GPU | ~80 | ~$10 | ~$100 + GPU |
| C baseline agent | ~$0.50 + GPU | ~60 | ~$10 | ~$45 + GPU |

Mitigations:

- For GEPA runs, add candidate hash + surface to the `primitive_cache`
  namespace, or disable the cache for any primitive whose output is
  prompt-dependent.
- Set `REPROLAB_MIN_RUBRIC_ITERATIONS=0` during the inner loop only.
- Use `--max-usd 0.50` per eval via `RunBudget`.
- Use stub primitives only for provisional Lane B search; validate accepted
  mutations with real `run_experiment`.
- Run weekly, not per PR.

## 6. Guardrails

### G1. Hermes is the anti-Goodhart oracle

Hermes audits `final_report.json` + artifacts after the run completes. Its
status multiplies the metric:

```text
gepa_score = rubric.overall_score * {
    grounded:     1.0,
    caveat:       0.5,
    unsupported:  0.0,
    unavailable:  0.0,
    system_error: 0.0,
}
```

Hermes itself never enters the optimization loop.

### G2. Held-out validation set

At least 30% of papers in any optimization run go in `valset`, not `trainset`.
The Pareto front is selected on validation performance.

### G3. Immutable prompt regions

The Algorithm-2 guard is not mutable. Mark it with a sentinel comment in
`build_system_prompt` so the adapter excludes it from mutation.

Do the same for chat steering, heartbeat, GPU selection, forced-iteration repair
policy, `FINAL_VAR` JSON contract, model-specific addenda, and any
security/sandbox instruction. The adapter exposes explicit mutable component ids
instead of arbitrary substrings.

### G4. Reflection LM not equal to task LM when possible

Use GPT-5 for both by default, but allow `--reflection-lm` override so the
reflection LM can differ from the task LM.

### G5. Runtime policies remain runtime policies

`forced_iteration.py` and the Lane O blanket-decline check remain outside the
prompt. GEPA-optimized prompts still hit the same refusals.

### G6. Optimization audit trail

Persist every GEPA generation to `runs/_gepa/<timestamp>/`:

- `pareto_front.jsonl`
- `reflective_dataset.jsonl`
- `mutations.jsonl`
- `scores.jsonl`
- `eval_manifest.jsonl`
- `source_snapshot.patch`

The new prompt does not become default until a human reviews the mutation log
and merges it into source. GEPA proposes; humans commit.

### G7. Readability is a constraint

If GEPA replaces a prompt region with something the team cannot read at a
glance, reject the mutation regardless of score gain.

## 7. Pareto axes

Pareto frontier dimension is per-paper score. SDAR, an NLP eval paper, and a CV
ablation paper exercise different rubric trees. Without Pareto, the optimizer
collapses to whichever paper dominates the aggregate. With Pareto, prompts can
specialize per archetype and system-aware merge can combine specializations.

## 8. Phased rollout

| Phase | Deliverable | Gating signal |
|---|---|---|
| P1 | `ReproLabGEPAAdapter` skeleton + Lane B driver | One optimization run completes and produces a non-trivial Pareto front |
| P2 | Lane B on SDAR + 2 other papers | One mutation beats baseline on held-out paper by >=5% |
| P3 | Lane A component split + optimization run | Hermes-clamped held-out score improves by >=5% |
| P4 | Lane C baseline-agent prompt | `run_experiment.success` rate improves on 5-paper held-out set |
| P5 | Weekly background GEPA run | Mutation log lands as a PR weekly |

## 9. Open questions

1. Should the reflection LM see paper text? Default no. It should see redacted
   trace only.
2. Do optimized prompts stay per-archetype or merge into one union prompt?
3. Does optimization happen per root model or against one canonical root model?

## 10. Out of scope

- Optimizing the rubric leaf scorer or Hermes.
- Optimizing the `_ROOT_PROMPT` opening sentence in `run.py`.
- Multi-objective Pareto beyond `(rubric * Hermes)` in v1.
- Optimizing system prompts for sub-RLMs spawned by `rlm_query`.

## 11. References

- Agrawal et al. - GEPA: Reflective Prompt Evolution Can Outperform
  Reinforcement Learning, arXiv 2507.19457.
- `github.com/gepa-ai/gepa`
- `dspy.GEPA`
- Internal: `docs/design/rlm-pivot-brief.md`
