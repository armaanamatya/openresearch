# RLM Phase 5 — Progress

_Updated: 2026-05-21_

## Objective

Drive the RLM orchestrator end-to-end on PaperBench papers — deliver
**≥2 papers with real rubric scores** in `runs/<id>/final_report.json`.

## Status

`main` and `merge` synced. Full test suite: **1111 passing**.

## Done

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

## Run history

| Run | Result        | Note                                                  |
|-----|---------------|-------------------------------------------------------|
| 3   | failed        | `implement_baseline` hit the OAuth quota (ran Opus)   |
| 5   | partial 0.65  | real code written; root skipped `run_experiment`, faked metrics |
| 6   | in progress   | all fixes applied — honesty guard + run_experiment nudge |

## In flight

**SNSE run 6** — end-to-end reproduction with every fix. Verifying: GPU
engaged, code written, `run_experiment` runs, honest rubric score.

## Next

- Authoritative PaperBench score via the post-run leaf scorer
- Reproduce a 2nd paper (`ftrl` or `mechanistic-understanding`)

## Known issues (deferred, pre-existing)

- `has_provider_credentials` treats the `claude` CLI on `PATH` as proof of
  login — not proof the OAuth session is valid.
- `build_environment`'s `ThreadPoolExecutor` can block past its timeout.
- `--sandbox` flag is a no-op for RLM primitives — `run_experiment`
  hardcodes `LocalDockerBackend`.
