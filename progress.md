# RLM Phase 5 — Progress

_Updated: 2026-05-21_

## Objective

Drive the RLM orchestrator end-to-end on PaperBench papers — deliver
**≥2 papers with real rubric scores** in `runs/<id>/final_report.json`.

## Status

`main` and `merge` synced. Full test suite: **1139 passing**. The RLM
pipeline now runs **end-to-end** — run 7 executed all 9 primitives and
produced an honest, leaf-scored report.

## Results

| Paper | Run | Verdict | Authoritative rubric (leaf scorer) |
|-------|-----|---------|------------------------------------|
| sequential-neural-score-estimation | 7 | failed* | **0.4042** (92/92 leaves graded) |
| mechanistic-understanding | 1 | — | in progress |

\* `failed` verdict = the generated `train.py` ran in Docker but produced
no metrics; the **rubric score is real** — the leaf scorer graded the
written code against the full PaperBench rubric.

## Done

**arXiv RLM end-to-end path.** An arXiv `--mode rlm` run now works end-to-end
and its results are retrievable through the REST API. Three gaps closed:
*(1)* the shared claim-map builder truncated every entry to 600 chars — right
for SDK prompts, wrong for RLM (the paper is offloaded whole into the REPL
`context`); RLM mode now carries the full `paper_text` un-truncated. *(2)* arXiv
runs self-generate a PaperBench-shaped rubric tree from the paper
(`rlm.rubric_gen.generate_rubric_tree`), persist it to `generated_rubric.json`,
and score against it — `score_run.py` finds it automatically and labels the
score `rubric_source="generated"` (honestly **not** PaperBench-official).
*(3)* `run_pipeline_rlm` now writes `demo_status.json`, so `GET /runs/{id}`
resolves for CLI- and script-launched RLM runs.

**Phase-2 merge.** Aayush's four `feat/rlm-phase2-foundation` Codex-review
commits merged into the line (primitives/binding hardening, Phase-1
skeleton removal).

**Schema robustness.** `PaperClaimMap` coerces the loosely-shaped dicts the
Qwen root emits — bare-string `claims`/`datasets`/`metrics` become dicts;
pre-built submodel instances pass through untouched.

**Sub-agent runtime.** `implement_baseline` resolves to Claude **Sonnet**
via an explicit `model_override` (beats the registry's Opus default).
Auth is SDK-resolved — `ANTHROPIC_API_KEY` for production, the Claude Code
OAuth login for dev.

**Honest reporting.** `build_final_report` sources the primitive trace from
the authoritative cost ledger and drops `baseline_metrics` the root
fabricated without a real `run_experiment` call.

**Featherless context cap.** `register_featherless_context_limits()` teaches
the rlm engine the plan's 49K input-context cap so compaction triggers before
the provider 400s the run.

**LLM client `max_tokens`.** Raised 600 → 4096 — 600 truncated the structured
JSON from `verify_against_rubric` and the leaf scorer.

**Experiment self-repair loop.** When `run_experiment` fails, the root re-calls
`implement_baseline` with `plan['repair_context']` (the failed result); the
code-writing agent diagnoses the error and fixes the existing code in place,
then `run_experiment` retries — up to 2 cycles. Each step is a distinct
`primitive_call` event, so Phase 4's UI renders the repair cycle naturally.

## Run history (SNSE)

| Run | Result        | Note                                                  |
|-----|---------------|-------------------------------------------------------|
| 3   | failed        | `implement_baseline` hit the OAuth quota (ran Opus)   |
| 5   | partial 0.65  | real code written; root skipped `run_experiment`, faked metrics |
| 6   | failed        | every primitive ran incl. `run_experiment`, real code — but `rlm.completion` 400'd: context exceeded the Featherless plan's 49K cap |
| 7   | failed / 0.40 | **completed end-to-end**; honest report; leaf scorer: 0.4042 |

## In flight

**Paper 2 — `mechanistic-understanding`** — full end-to-end run with every
fix applied. Leaf-score it on completion.

## Diagnosis — run 7's `failed` verdict

Ran `train.py` directly in run 7's Docker image: **training ran fully** (100
epochs, loss converged) and crashed only in posterior evaluation —
`score_network.py` `torch.cat` shape mismatch (500 posterior samples vs a
batch-1 observation never expanded). **Not** an API / Docker / OAuth fault —
infrastructure is sound; it is a bug in the AI-generated code. The
experiment self-repair loop (above) is the systemic fix.

## Next

- Leaf-score paper 2 → second authoritative rubric score (deliverable met)
- `run_experiment` real metric extraction — `_execute_in_sandbox` hardcodes
  `metrics: {}`; reading `metrics.json` from the run's outputs would give a
  measured-metrics path and enable a `reproduced` verdict

## Phase 4 readiness (#61 — frontend, not built here)

The repair loop is deliberately root-orchestrated so each attempt surfaces as
its own `primitive_call` SSE event — #61's UI renders the cycle with no extra
backend work. Event audit for #61 is a follow-up.

## Known issues (deferred, pre-existing)

- `has_provider_credentials` treats the `claude` CLI on `PATH` as proof of
  login — not proof the OAuth session is valid.
- `build_environment`'s `ThreadPoolExecutor` can block past its timeout.
- `--sandbox` flag is a no-op for RLM primitives — `run_experiment`
  hardcodes `LocalDockerBackend`.
