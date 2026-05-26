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

Optimize the three improvement-flow prompts in
`backend/agents/prompts/improvement.py`:
`IMPROVEMENT_ORCHESTRATOR_PROMPT`, `ADAPTIVE_POOL_GENERATION_PROMPT`,
`ADAPTIVE_RERANK_PROMPT`. All three are load-bearing — orchestrator chooses
which hypothesis to run, pool-generation produces the candidate set, rerank
breaks ties. Optimizing only the orchestrator while the other two stay
fixed leaves two confounders. v1 seeds them as three independent component
ids (`improvement.orchestrator.body`, `improvement.pool_generation.body`,
`improvement.rerank.body`); freeze any two as immutable controls only if
GEPA's search space proves too wide in practice.

Cache warning (same hazard as Lane C):
`AGENT_REGISTRY["improvement-orchestrator"]` copies
`IMPROVEMENT_ORCHESTRATOR_PROMPT` into an `AgentSpec` at import time
(`backend/agents/registry.py:104-112`). Monkey-patching the module global
will NOT take effect. Use the prompt-override seam from §4.3.

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
- ASI: CUDA OOM markers, `RubricGuardFailure` JSON
  (`backend/agents/rlm/rubric_guard.py`), missing-artifact errors,
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
  raw logs. See §4.4 for schema.
- Dependency. `gepa` is not yet vendored; pin a version in
  `backend/requirements.txt` (e.g. `gepa==0.x.y`) before P1 lands.

## 4.1 Component architecture

New modules under `backend/agents/optimization/`:

- `mutable_regions.py` — `MutableRegionRegistry`. Owns the default text for
  every mutable prompt section and a stable `component_id` for each.
- `prompt_overrides.py` — `PromptOverrideContext` (thread-local stack +
  context-manager). Read by `build_system_prompt` and `invoke_agent`. Empty
  by default — production runs see exactly today's prompts.
- `gepa_adapter.py` — `ReproLabGEPAAdapter` (the GEPA contract). Holds the
  optimization-run config and dispatches per-eval `run_pipeline_rlm` calls.
- `trace_minimizer.py` — turns `final_report.json` + artifacts +
  `cost_ledger.jsonl` + Hermes JSON into the reflective-dataset record
  (§4.4). Extends the redaction rules already used by
  `sse_bridge.sanitize_iteration`.
- `eval_budget.py` — `EvalBudgetEnforcer` wrapping `RunBudget`. Kills
  runaway evals; scores them `0.0` with a structured timeout trace.

Touched (small additive changes, no behavior change when no override is
active):

- `backend/agents/rlm/system_prompt.py:build_system_prompt` — accepts
  optional `overrides: dict[component_id, str]`.
- `backend/agents/registry.py:AgentSpec.to_runtime_spec` — accepts optional
  `prompt_override: str | None`.
- `backend/agents/runtime/invoke.py:invoke_agent` — reads the active
  `PromptOverrideContext` and threads through.

Driver: `scripts/optimize_prompts_gepa.py` (new, per-lane CLI flags).

## 4.2 Mutable region registry

```python
# backend/agents/optimization/mutable_regions.py
@dataclass(frozen=True)
class MutableRegion:
    component_id: str
    default_text: str
    description: str       # what this region governs
    char_budget: int       # soft cap; mutations exceeding are rejected

REGIONS: dict[str, MutableRegion] = {
    # Lane A
    "root_system.decomposition_example": MutableRegion(...),
    # optional follow-ups, gated on telemetry showing repeated failures:
    # "root_system.triage_hints", "root_system.decomposition_strategy"

    # Lane B
    "improvement.orchestrator.body": MutableRegion(...),
    "improvement.pool_generation.body": MutableRegion(...),
    "improvement.rerank.body": MutableRegion(...),

    # Lane C
    "baseline_agent.body": MutableRegion(...),
}
```

`system_prompt.py` `_DECOMPOSITION_EXAMPLE`, `improvement.py`'s three
prompt constants, and `baseline_implementation.py`'s
`BASELINE_IMPLEMENTATION_PROMPT` are moved into the registry; the original
modules re-export them as `REGIONS["..."].default_text` for source
compatibility. Immutable sections (Algorithm-2 guard, `FINAL_VAR`
contract, heartbeat, GPU selection, chat steering, forced-iteration
policy, security text) are not added to the registry — they cannot be
selected as targets at all.

## 4.3 Prompt-override seam

Two threading paths, both stateless at module load:

RLM root prompt (Lane A):

```python
# backend/agents/rlm/system_prompt.py
def build_system_prompt(
    ctx: RunContext,
    ...,
    overrides: Mapping[str, str] | None = None,
) -> str:
    text_for = lambda cid: (
        (overrides or {}).get(cid) or REGIONS[cid].default_text
    )
    # Assemble using text_for(...) for each mutable region.
```

The adapter constructs `overrides` from the GEPA candidate dict and threads
through `run_pipeline_rlm(..., system_prompt_overrides=overrides)`. No
process-global state, no monkey-patching.

AgentSpec-mediated sub-agents (Lanes B + C):

```python
# backend/agents/optimization/prompt_overrides.py
class PromptOverrideContext:
    @contextmanager
    def use(self, overrides: dict[str, str]) -> Iterator[None]: ...
    def current(self) -> dict[str, str]: ...

PROMPT_OVERRIDES = PromptOverrideContext()

# backend/agents/runtime/invoke.py
def invoke_agent(agent_id: str, ...) -> ...:
    override = PROMPT_OVERRIDES.current().get(agent_id)
    spec = AGENT_REGISTRY[agent_id].to_runtime_spec(
        provider=provider, prompt_override=override
    )
    ...
```

The adapter wraps each evaluation in
`with PROMPT_OVERRIDES.use({"baseline-implementation": candidate_text}):`.
Resolution happens at invocation, so the import-time
`AgentSpec.prompt = BASELINE_IMPLEMENTATION_PROMPT` snapshot is no longer
the load-bearing copy.

## 4.4 Reflective dataset schema

One record per (component, paper) eval, max ~8KB after redaction:

```json
{
  "component_id": "improvement.orchestrator.body",
  "example_id": "arxiv:2605.15155",
  "paper_archetype": "rl-agent",
  "score": 0.62,
  "hermes_clamped_score": 0.62,
  "input": {
    "rubric_overall_before": 0.55,
    "rubric_areas_before": {"...": 0.4},
    "weak_leaves_before": [
      {"name": "...", "score": 0.0, "rationale": "<=200 chars"}
    ],
    "current_results_digest": "<=400 chars"
  },
  "candidate_output": {
    "selected_hypothesis": {"...": "..."},
    "candidate_id": "...",
    "category": "..."
  },
  "execution_trace": {
    "rubric_overall_after": 0.62,
    "rubric_delta_areas": {"...": 0.07},
    "weak_leaves_after": [],
    "run_experiment_success": true,
    "repair_summaries": ["<=200 chars each"],
    "forced_iteration_warnings": 0,
    "blanket_decline_count": 0,
    "hermes": {
      "status": "grounded|caveat|unsupported|unavailable|system_error",
      "unsupported_claims": ["..."],
      "evidence_refs_snippets": ["<=120 chars each (top 3)"]
    }
  }
}
```

Redaction (all enforced by `TraceMinimizer`):

- Strip paper corpus, REPL locals, secrets, env vars.
- Truncate stdout/stderr to first 200 + last 200 chars.
- Drop artifact bodies; keep filenames and schema keys only.
- Hash anything that looks like a file path under `runs/` to a short tag
  (the reflection LM never needs the path; mentioning it leaks scope).

## 4.5 Candidate isolation + concurrency

Artifact tree:

```
runs/_gepa/<ts>/
  manifest.json
  trainset.json, valset.json
  candidates/
    <candidate_hash>/
      overrides.json
      evals/
        <paper_id>/              # = RunContext.project_id for the eval
          demo_status.json
          rlm_state/
          final_report.json
          hermes_audit/
          primitive_cache.jsonl
          dashboard_events.jsonl # written but no live SSE consumer
  pareto_front.jsonl
  reflective_dataset.jsonl
  mutations.jsonl
  scores.jsonl
  eval_manifest.jsonl
  source_snapshot.patch
```

`primitive_cache` namespacing rule:

- Prompt-invariant primitives (`understand_section`,
  `extract_hyperparameters`, `detect_environment`) cache as today; their
  inputs already exclude prompt text.
- Prompt-dependent primitives (`implement_baseline`,
  `propose_improvements`, `verify_against_rubric` when the rubric is
  derived) include `candidate_hash` + `surface_id` in the cache key, or
  the cache is disabled for the lane.
- Validation runs (held-out, post-acceptance) always disable the cache.

Concurrency:

- Lane B is CPU/LLM-bound, no GPU. Parallelize evals via a
  `concurrent.futures.ProcessPoolExecutor` with `min(8, len(papers))`
  workers. `primitive_cache` writes are append-only and tolerate this.
- Lane A/C consume RunPod pods. Cap with
  `REPROLAB_GEPA_MAX_PARALLEL_PODS=2`; serialize when unset. Per-pod
  `RunBudget.max_usd` enforces the per-eval ceiling.
- GEPA's selection / mutation / merge phases are always serial — the
  library handles this.

## 4.6 Mutation acceptance protocol

When GEPA accepts a mutation, the adapter materializes:

```
runs/_gepa/<ts>/proposed_mutations/<mutation_id>/
  before.txt, after.txt
  diff.patch                # against current source default_text
  reflective_examples.jsonl # the records that drove this mutation
  eval_summary.json         # per-paper score deltas, train + val
  validation_run.json       # required: one real run_experiment on a
                            # held-out paper not in train+val. Missing
                            # or failed → mutation is parked, not
                            # promoted.
  pr_body.md                # human-readable PR description
```

Promotion is explicit: `scripts/promote_gepa_mutation.py <ts>/<id>` opens
a PR that updates `MutableRegionRegistry`'s `default_text` for that
`component_id`, attaches `pr_body.md` as the PR body, and tags
`gepa-mutation`. GEPA proposes; humans review and merge (G6 + G7). Parked
mutations stay on disk for archaeology — nothing auto-deletes.

## 5. Cost model

| Lane | Eval cost/row | Rows/run | Reflection | Hermes/eval | Validation run | Total/opt |
|---|---:|---:|---:|---:|---:|---:|
| B improvement | ~$0.05 | ~50 | ~$5 | ~$0.02 | ~$0.50 | ~$11 |
| A root prompt | ~$1.00 + GPU | ~80 | ~$10 | ~$0.02 | ~$1.00 + GPU | ~$102 + GPU |
| C baseline agent | ~$0.50 + GPU | ~60 | ~$10 | ~$0.02 | ~$0.50 + GPU | ~$46 + GPU |

Per-mutation validation (§4.6) costs one full real `run_experiment` per
accepted mutation on a held-out paper; budget 1-3 of these per
optimization run.

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
   GEPA's system-aware merge is the natural seam for this; revisit after
   Lane B produces ≥2 archetype-specialized Pareto candidates.
3. Does optimization happen per root model or against one canonical root model?
4. Reflection-LM context: does `gepa` call the reflection LM per-example or
   per-batch? Per-batch with minibatch=50 risks overflowing GPT-5 context
   even after redaction. Verify the library default and gate before P1.
5. Where do paper-archetype labels live? Proposal:
   `tests/fixtures/papers/<arxiv_id>/archetype.txt` (one of
   `rl-agent`, `nlp-eval`, `cv-ablation`, ...). The trainset loader reads
   this; archetype is a trace field, not a candidate input.
6. Noise floor: root-model + sandbox stochasticity make a single-run score
   noisy. Per-example n=1 (cheap, high variance) vs n=2 paired
   (2x cost, halved variance)? v1 picks n=1 plus a wider Pareto front; if
   acceptance noise dominates after P2, escalate to n=2 on a per-mutation
   basis at acceptance time.
7. Should the forced-iteration policy stay enabled during GEPA inner-loop
   evals? Disabling it speeds inner evals but biases the optimizer toward
   prompts that terminate too early. v1 leaves it ON; revisit if cost
   forces it off.

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
