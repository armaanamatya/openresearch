# New improvement ideas — round 2 (2026-06-21)

## Scope and method

This is a read-only, zero-spend follow-up to the 7-theme opportunity menu. The
named menu is absent from this worktree; it was recovered verbatim from git object
`57e6d821:docs/superpowers/specs/2026-06-21-system-improvement-opportunities.md`.
That object, the dark-switch plan, the SDAR brief, the evidence-first ADR,
`CLAUDE.md`, and `system_overview.md` were treated as known work. Three independent
read-only passes covered run artifacts, adversarial scoring integrity, and
cross-paper generality; a fourth local pass covered root ergonomics and new
capabilities. No tests, GPU work, live API calls, or production actions occurred.

`NEW` means not materially present in the 7-theme menu or dark-switch plan. It
does **not** mean immediately safe to ship. Every proposed behavior change must be
default-OFF, fail-soft, hermetically tested both OFF and ON, and complete at least
three paired A/B runs before any default flip.

## The three highest-leverage NEW ideas

1. **A single, source-linked `ReproductionContract` (#1).** It resolves the
   present split between hints, YAML invariants, generated rubrics, scope, and
   grading. It is the semantic substrate needed for generic papers.
2. **Canonical evidence bundles plus exact leaf coverage (#2–#3).** They prevent
   a report, metric file, code snapshot, and broad claim from being assembled from
   different attempts or a single successful cell.
3. **Typed experiment dimensions and a metric contract (#4–#5).** They remove the
   SDAR-shaped `model × environment × baseline` assumption and make the score mean
   the same thing for sweeps, ablations, retrieval, and non-RL papers.

## Ranked catalog

| # | Status / idea | Grounding | Impact | Effort |
|---:|---|---|---|---|
| 1 | **NEW — Intake-persisted `ReproductionContract`.** Extract one source-linked contract containing required methods, variants, datasets, metrics, protocol, invariants, source spans/confidence, and explicit unresolved fields. Persist it in `rlm_state/`; scope, preflight, cells, certificate, grading, and report use it rather than parallel paper-specific sources. `OPENRESEARCH_REPRO_CONTRACT=1`; absent/ambiguous fields retain today's path and are stamped unresolved. | `paper_invariants.py:222-252` loads only hand-authored YAML; `schemas.py:899-929` carries only models/datasets/seeds; `run.py:2820-2829` separately persists a generated rubric. | Coverage +++ · honesty ++ · score ++ · cost - | L |
| 2 | **NEW — Canonical evidence-bundle receipt.** On a successful experiment, mint one immutable receipt `{attempt, ledger sequence, metrics SHA-256, code-tree digest, artifact dir, coordinates}`. Score and report must resolve through that receipt, never independently choose newest metrics, a latest ledger row, and current code. `OPENRESEARCH_CANONICAL_EVIDENCE_BUNDLE=1`; incoherence stamps `bundle_unverified` and preserves legacy behavior. | `leaf_scorer.py:302-354` selects result artifacts by mtime; `report.py:1248-1331` separately recovers ledger/metrics; `external_validator.py:418-443` fingerprints metrics only. | Honesty +++ · score calibration ++ · coverage ++ · cost 0 | M |
| 3 | **NEW — Exact result-leaf coverage matrix.** Compile each result leaf to required typed coordinates; an aggregate claim must prove every named model/environment/baseline/seed or state its numerator and denominator. Unknown leaves remain legacy-scored with `coverage_unknown`, rather than producing a false veto. `OPENRESEARCH_EXACT_LEAF_COVERAGE=1`. | `leaf_scorer.py:695-764` accepts token-overlap substantiation; the SDAR metric tree is described in `cell_matrix.py:1-47`; scope exclusions can shrink denominator in `leaf_scorer.py:779-817`. | Honesty +++ · coverage +++ · score calibration ++ · cost 0 | M/L |
| 4 | **NEW — Typed arbitrary experiment dimensions.** Replace lossy axis synonyming with `dimensions: {name: value}` in cells, results, claims, and evidence. Keep a legacy projection for SDAR, but distinguish optimizer, data fraction, depth, augmentation, retrieval setting, and hardware rather than coerce them into `baseline`. `OPENRESEARCH_TYPED_DIMENSIONS=1`. | `cell_matrix.py:134-188` maps `variant` and `optimizer` into `baseline`; the current nested result tree is fixed at `model_key → env → baseline` (`cell_matrix.py:10-23`). | Coverage +++ · honesty ++ · score ++ | L |
| 5 | **NEW — Paper-derived `MetricContract`.** Define metric unit, direction, valid range, aggregation, split, denominator, and required uncertainty from the reproduction contract. Use it for metric validation, cross-seed aggregation, claim comparison, and rendering. Start warning-only with `OPENRESEARCH_METRIC_CONTRACT=1`. | `metric_semantics.py:42-100` assumes named rates are fractions; cross-seed rich statistics are limited to a fixed key list in `cell_matrix.py:620-680`; `DatasetSlice` calls split/episodes advisory at `schemas.py:881-887`. | Honesty ++ · coverage ++ · score ++ | M |
| 6 | **NEW — Harness-owned provenance validity.** Verify provenance schema/source, coordinate completeness, and bundle hashes; distinguish mechanical harness facts from agent assertions. Do not credit a valid JSON file merely because it exists. `OPENRESEARCH_PROVENANCE_VERIFY=1`; missing information produces an unverified warning/cap, never a crash. | Validator health is existence-only at `external_validator.py:91-110,327-342`; agent-writable provenance is merged by `provenance.py:298-366`; the audit also has presence-only checks in `evidence_audit.py:26-35`. | Honesty +++ · coverage ++ · score + · cost 0 | M |
| 7 | **NEW — Finalize-time validator cap.** A fresh, machine-verified final validator veto must cap affected leaves/verdict on terminal paths, not merely be stamped into the report. Reuse the stored verdict; unavailable/stale transport remains a no-op. `OPENRESEARCH_FINALIZE_VALIDATOR_CAP=1`. | The clean-loop validator can refuse progress in `run.py:3173-3240`; `external_validator.py:182-191` leaves rerun agreement pending; terminal report writing serializes rather than necessarily acts on the panel result (`report.py:1828-1865`). | Honesty +++ · score calibration ++ · cost 0 | M |
| 8 | **NEW — Independent evaluator receipts for result leaves.** For supported tasks, independently score saved predictions/trajectories against gold data or the environment and attach a harness-owned receipt to the evidence bundle. Begin with SDAR's ALFWorld, WebShop, and Search-QA adapters; unsupported papers remain unverified, not failed. `OPENRESEARCH_INDEPENDENT_EVAL_RECEIPT=1`. | An arbitrary `train.py` writes metrics consumed by execution; validator checks only variation and plausibility (`external_validator.py:113-179`), while its rerun predicate is a stub (`:182-191`). | Grounded score ceiling +++ · honesty +++ · coverage ++ · cost + | L |
| 9 | **NEW — Deterministic remote stall guard.** Kill a remote command only after two independent stale signals (streamed output/`exec.log` growth plus checkpoint or GPU/CPU liveness). Persist its evidence packet and classify `exec_stalled`, so the root repairs a real hang rather than waits for the hard timeout. `OPENRESEARCH_REMOTE_STALL_GUARD=1`. | `runpod_backend.py:74-81,391-425` documents absent remote liveness detection; `failure_classifier.py:115-127` already has a precise repair narrative for `exec_stalled`; SDAR artifact `best_runs/sdar/attempt-1-oom-foreign-proc/experiment_runs.jsonl:1` shows the paid failure class pressure. | Cost +++ · reliability +++ · honesty + | M |
| 10 | **NEW — Bounded root orientation and rate-limit circuit breaker.** Persist structured orientation summaries and track raw REPL/sub-RLM bytes; stop redundant full-paper/rubric dumps once those summaries exist, then direct the root to a source slice. No report evidence may use this cache. `OPENRESEARCH_ORIENTATION_BUDGET=1`. | A historical first iteration emits a 51 KB rubric at `runs/pb_mechanistic-understanding_1780068083/iterations/iteration_0001.json:106-128`; another run ended at a provider rate limit (`runs/prj_e2d9aebb05d4340f/final_report.md:21-24`); context map is flat, 8 KB, and navigation-only (`context_map.py:1-21,97-145`). | Cost +++ · reliability ++ · coverage ++ | M |
| 11 | **NEW — Durable failure capsules.** Before classification, atomically persist a bounded/redacted traceback and log tail, command/environment fingerprints, artifact paths, and error signature in `experiment_runs.jsonl`; pass the capsule into repair context. `OPENRESEARCH_FAILURE_CAPSULES=1`. | Historical unknown failures retain no useful tail in `runs/prj_e2d9aebb05d4340f/experiment_runs.jsonl:1-2`; the classifier expects tails and falls back to a generic message at `failure_classifier.py:203-204,232-242`. | Reliability +++ · root efficiency ++ · honesty + | M |
| 12 | **NEW — Atomic finalization identity.** Write a `finalization_id` and evidence digest across status, rubric snapshot, report, and terminal event; UI/leaderboard flag mixed identities rather than display contradictory state. `OPENRESEARCH_FINALIZATION_ID=1`. | `runs/prj_09047604e591d969/demo_status.json:11-16` remained running while its `final_report.md:1-18` reported a terminal failure; system overview identifies these as separate file-backed surfaces. | Honesty ++ · operator reliability +++ · score indirect | M |
| 13 | **NEW — Capability profiles, not SDAR guidance behind a generic RL flag.** Select `supervised`, `on_policy_rl`, `offline_rl`, or `agentic_rl` from the contract. Keep GRPO/OPSD parameters in the SDAR profile; unknown profiles warn and use today's generic path. `OPENRESEARCH_CAPABILITY_PROFILES=1`. | `baseline_implementation.py:2436-2443` injects RL scaffold guidance; `rl_scaffold.py:30-108` encodes SDAR's OPSD, β=10, and λ=.1. | Coverage ++ · honesty ++ · score ++ | M |
| 14 | **NEW — Resource-provenance audit for arbitrary papers.** Record paper-linked URLs/repositories at ingest and audit tool fetches against exact normalized identities. Start evidence-only; enforcement blocks only exact paper-owned sources, never broad framework dependencies. `OPENRESEARCH_PAPER_RESOURCE_AUDIT=1`. | The blocklist is a hand-maintained hint/operator union (`cli.py:1756-1779`); SDAR is protected only by a specific hint. | Honesty +++ · coverage ++ | M |
| 15 | **NEW — Runtime-neutral cell entrypoints.** Add `CellEntrypoint {runtime, argv, metric_path, checkpoint protocol}` while preserving today's Python runner as default. This permits JAX, shell launchers, and native research code without forcing scientifically risky rewrites. `OPENRESEARCH_CELL_ENTRYPOINTS=1`. | Current runner always constructs `[sys.executable, cell_script]` at `gpu_cell_runner.py:509-512`; generated code requires Python/argparse in `baseline_implementation.py:1287-1335`. | Coverage +++ · cost ++ · SDAR neutral | L |
| 16 | **NEW — Reproducibility lock bundle.** Capture non-secret source, image/package, model/dataset revision, preprocessing/config, launch, CUDA/GPU, and artifact hashes in `repro_lock.json`. Missing fields downgrade reproducibility claims only. `OPENRESEARCH_REPRO_LOCK=1`. | Dependency derivation deliberately drops URLs/VCS specs (`requirements_derive.py:88-93`); dataset protocol remains advisory (`schemas.py:881-887`). | Honesty ++ · coverage ++ · score + | M |
| 17 | **NEW — Environment-adapter plugins.** Provide declarative detect/provision/health/action-observation/exclusion adapters. Keep SDAR adapters intact; unknown environments stay explicitly unsupported, never silently available. `OPENRESEARCH_ENV_ADAPTERS=1`. | CLI has special full-scope guidance only for ALFWorld/WebShop/Search-QA at `cli.py:1718-1738`; cache identities are the same narrow set in `env_cache.py:117-119`. | Coverage +++ · score ++ · cost + | L |
| 18 | **NEW — Deterministic next-action card.** Derive a compact state/action card from lifecycle stage, last outcome, failure capsule, coverage gaps, and remaining budget; inject it after each primitive result. This replaces paid, advisory next-tool reasoning when the next action is already deterministic. `OPENRESEARCH_ACTION_CARDS=1`. | `run.py:3090-3125` can infer a required stage; `recommend_next_tool` nevertheless spends an LLM call for a generic tool list (`primitives.py:8082-8121`); the prompt carries broad workflow rules at `system_prompt.py:295-332`. | Cost ++ · root efficiency +++ · reliability + | M |

## Explicitly not re-proposed

These are real needs, but are already in the known menu or its in-flight plan:

- **Dark execution switches:** cache persistence, cell resume, orphan cleanup,
  dead-training stop, OOM hard cap, import smoke, patch re-preflight, and spot
  capacity are T1/T6 and covered by `2026-06-21-dark-switches-plan.md`.
- **Table/equation extraction, generated invariants, rubric source grounding, and
  truncation repair** are T2/T3.
- **Value grounding, a labeled honest/fabricated corpus, and generic deterministic
  leaf routing** are T5/T7.
- **Score-history-fed proposals, plateau gating, champion restore, candidate outcome
  persistence, and cross-paper lessons** are T4.
- **Generic sealed/HMAC provenance** is already explicitly recorded as an existing
  remediation direction. The new bundle proposal (#2) is intentionally narrower:
  it first prevents cross-attempt mixing and creates a concrete object any later
  sealing mechanism can sign.

## Recommended sequencing

First specify #1, #2–#3, and #4–#5 as one compatibility-aware semantic/evidence
foundation: each should be default-OFF, generate comparison artifacts, and retain
the legacy projection. Then choose either #6–#8 for the honesty frontier or
#9–#11 for immediate iteration/cost reliability. Do not default-flip any proposal
without three stamped paired A/B runs and a grader-noise check.

The shared design and execution order are now captured in
`2026-06-21-semantic-evidence-foundation-design.md` and
`2026-06-21-semantic-evidence-foundation-plan.md`; implementation starts with
their Phase 0 compatibility fixtures.
