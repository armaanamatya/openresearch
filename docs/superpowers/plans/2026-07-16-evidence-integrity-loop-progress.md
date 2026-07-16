# Evidence-Integrity & Observability — Autonomous Loop Progress Tracker

> Durable state for the self-paced loop (15-min cadence until 09:00 PDT 2026-07-16).
> Each iteration: reread this file, do the next chunk, update this file, commit, report, reschedule.
> This survives context summarization + cache-cold wakeups — it is the source of truth for "where am I".

## Mandate (from user)
- Deliver a **mega-spec covering all 5 workstreams** (W1 deepest), then a full implementation plan.
- Constraints (hard): every new mechanism **default-OFF flag-gated**, byte-identical when off;
  **evidence-not-grade red line** preserved (no verdict routed through an LLM grade);
  **each ships behind hermetic tests** (pytest-socket).
- Fork decision: **C — detect-now (W1 in-process) / prevent-later (W5 out-of-process grader).**
- Loop: recheck & update every ~15 min, continue iterating until 09:00.

## Workstreams
- **W1** Grader-tampering + leakage detection — `evidence_bundle.py`, `evidence_gate.py` (flag `OPENRESEARCH_GRADER_INTEGRITY`)
- **W2** GroundEval state-contracts — new `state_contracts.py`, `leaf_scorer.py` (flag `OPENRESEARCH_STATE_CONTRACTS`)
- **W3** PaperBench head-to-head scorecard — `backend/evals/paperbench/` (CLI subcommand)
- **W4** Cost observability — `pricing.py`, cost ledger, `demo_status.json` (flag `OPENRESEARCH_COST_OBSERVABILITY`)
- **W5** Sandbox-ingress hardening + out-of-process grader — `backend/services/runtime/` (flag `OPENRESEARCH_HARDENED_SANDBOX`)

## Phase plan
- **A. Spec** (iter 1-3): write mega-spec, self-review, commit. — IN PROGRESS
- **B. Verify** (iter 4-6): dispatch Explore agents to confirm every file/symbol claim; fix spec.
- **C. Plan** (iter 7-12): writing-plans → full implementation plan, W1 TDD steps deepest.
- **D. Build W1** (iter 13-20, optional): TDD on isolated branch, default-OFF, subagent-review each diff.
- **E. Handoff** (near 09:00): synthesis + review-ready summary.

## Iteration log
- **iter 1 (02:51)**: created this tracker; wrote mega-spec; committed b9dd9acf.
- **iter 2 (03:1x)**: Phase B verification (3 Explore agents) — all core claims confirmed + corrections.
  KEY DISCOVERIES: (a) `grader_digest.py` = metrics-compaction (A6), NOT integrity → W1 must avoid
  that name; (b) `deterministic_leaf_checker.py` already implements typed on-disk predicates via
  `check_kind`+`assertion` → W2 becomes a new `deterministic:state_contract` kind, standalone
  state_contracts.py WITHDRAWN; (c) W1 sharpened to *grading-input* integrity (rubric/provenance/
  metrics artifacts), grader-code is out-of-sandbox. Spec §14 records all corrections. Now starting
  TDD on W1-M1 rubric pinning (pure module).

## Phase status
- A. Spec — DONE (+ §14 revisions)
- B. Verify — DONE
- C. Plan — folding into per-workstream TDD (mega-spec §4-8 + §14 serve as the plan)
- D. Build W1 — STARTING: W1-M1 rubric pinning, TDD, branch `feat/evidence-integrity-w1`

- **iter 3 (mid-cycle, user-driven) → LOOP END (10:17, past 09:00 stop)**: TDD'd W1-M1 rubric
  pinning to completion on branch `feat/evidence-integrity-w1`. Pure `rubric_fingerprint` +
  disk-backed `write_rubric_pin` + fail-closed `verify_rubric_integrity` + flag-gated
  `check_grading_input_integrity` (OPENRESEARCH_GRADER_INTEGRITY, default-OFF). 9 hermetic tests
  incl. OFF/ON pair, all green; ruff clean. Committed c97f9564. **NOT yet wired into leaf_scorer /
  evidence_gate** — that's the next (higher-risk) slice. Loop stopped: 09:00 passed.

## W1-M1 — DONE + WIRED (2026-07-16, commits c97f9564, d706dc5f, bc78dd02, 263d7f94)
- Pure module `backend/evals/paperbench/grading_input_integrity.py`: `rubric_fingerprint` (byte-identical
  to canonical `attempt_assessment.rubric_sha256`, consistency-tested), `write_rubric_pin` /
  `maybe_write_rubric_pin` (flag-gated), `verify_rubric_integrity` (fail-closed), `check_grading_input_integrity`.
- WIRED: pin at `run.py` after the spec-validator (rubric finalized); guard atop `verify_against_rubric`
  (primitives.py) rejects a rubric ≠ pin as `evidence_tampered` (repairable) BEFORE spending an LLM grade.
- Flag `OPENRESEARCH_GRADER_INTEGRITY` (default-OFF, registered in flags.md). OFF = byte-identical (tested).
- Tests: 14 module + 2 integration (make_context OFF/ON); 43+114 green; ruff clean.
- CAVEAT for default-flip: root may pass a benignly-different rubric object → real-run A/B needed before ON.
  Discovered existing campaign-level rubric pin (`attempt_assessment.rubric_sha256`, cross-attempt) — reused
  its hash semantics; W1-M1 covers the single-run (incl. non-campaign) gap it doesn't.

## W1-M4 / W1-M3 — ALREADY EXIST (do NOT rebuild)
`backend/agents/rlm/eval_provenance.py` (`OPENRESEARCH_EVAL_PROVENANCE_GUARD`) already does the
reported-vs-true metric cross-check (`abs(reported − mean(records)) > 1e-3` → veto) AND leakage
(eval_ids ∩ train_ids). W1-M4 and most of W1-M3 are COVERED. Spec §4.2 M4 withdrawn as duplicative.

## W2 — DONE (core, 2026-07-16, commit 73381ccb)
Added `deterministic:state_contract` check_kind to `deterministic_leaf_checker.py` (flag
`OPENRESEARCH_STATE_CONTRACTS`, default-OFF, registered). Predicates grounded in existing
`eval_provenance.json`: `min_eval_n` (eval-coverage) + `require_held_out`. 7 hermetic tests.
- ACTIVATION CAVEAT: `check_leaf` fires only when `OPENRESEARCH_DETERMINISTIC_LEAVES` is on (its
  caller in `score_reproduction`) AND a rubric leaf carries the `deterministic:state_contract`
  annotation. Checker is complete+tested; full live activation needs rubric-gen to emit the
  annotation (same shipped pattern as the existing hparam/artifact/numeric kinds).

## RESUME HERE (next slices)
1. Rubric-gen: emit `deterministic:state_contract` annotations so W2 fires live (`rubric_gen.py`).
2. W1-M2 file-access audit (the one genuinely-missing producer; needs sandbox entrypoint hook, spec §4.2).
3. W4 cost: the REAL fix is 2-part — make the Foundry client RECORD tokens (client code) THEN add
   pricing.py entries (a pricing entry alone does nothing while tokens log 0/0). Plus idle-GPU accounting.
4. W3 (PaperBench scorecard) + W5 (sandbox hardening) per spec §§6-8, §14.
5. Consider opening a PR for the W1-M1 + W2 work (branch `feat/evidence-integrity-w1`).

## Key facts for implementation (verified)
- New W1 module: `backend/evals/paperbench/grading_input_integrity.py` (sibling to deterministic_leaf_checker)
- W2: extend `deterministic_leaf_checker.py` with `deterministic:state_contract` check_kind
- Flag idiom: `os.environ.get("FLAG","").strip().lower() in ("1","true","yes")`
- Test pattern to mirror: `tests/evals/test_leaf_scorer_feasibility_scope.py`
- rubric file: `runs/<id>/rubric_tree.json`; provenance: `code/provenance.json` or `code/outputs/*/provenance.json`
- evidence bundle: `mint_bundle(project_dir)` / `resolve_bundle(project_dir)`, field `attempt_id`

## Pre-existing suite pollution (NOT caused by this branch — flagged for a separate fix)
Full `tests/rlm/` run (alone, WITHOUT any of my evals test files) reproduces 3 order-dependent failures:
`test_external_validator.py::test_external_validator_disabled_by_default`,
`::test_flag_off_panel_unavailable_with_none_client`,
`test_report_validation_stamp.py::test_mismatched_fingerprint_leaves_validation_empty`.
Proof it's not mine: each passes in isolation; all pass when grouped with my new/modified tests
(102 passed, fixed order); my code changes are function-local imports with no global side effects;
the failing files are pre-existing/untouched. Root cause = some earlier rlm test leaks validator
state (cached client / module singleton) in full-suite order. Separate cleanup task, not part of
the evidence-integrity work. My own suites (evals + the touched rlm tests) are fully green.

## Guardrails for autonomous edits
- No code edits until spec + plan done AND verified against real code.
- W1 touches fail-closed evidence gates → keep default-OFF SACRED; A/B flag-OFF test must prove byte-identical.
- Commit ONLY files I create/touch for this work (explicit `git add <path>`); do NOT sweep the pre-existing dirty tree.
- Do not push. Do not merge. Leave for user review.
