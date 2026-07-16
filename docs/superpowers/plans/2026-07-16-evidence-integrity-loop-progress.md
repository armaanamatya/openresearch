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

## Key facts for implementation (verified)
- New W1 module: `backend/evals/paperbench/grading_input_integrity.py` (sibling to deterministic_leaf_checker)
- W2: extend `deterministic_leaf_checker.py` with `deterministic:state_contract` check_kind
- Flag idiom: `os.environ.get("FLAG","").strip().lower() in ("1","true","yes")`
- Test pattern to mirror: `tests/evals/test_leaf_scorer_feasibility_scope.py`
- rubric file: `runs/<id>/rubric_tree.json`; provenance: `code/provenance.json` or `code/outputs/*/provenance.json`
- evidence bundle: `mint_bundle(project_dir)` / `resolve_bundle(project_dir)`, field `attempt_id`

## Guardrails for autonomous edits
- No code edits until spec + plan done AND verified against real code.
- W1 touches fail-closed evidence gates → keep default-OFF SACRED; A/B flag-OFF test must prove byte-identical.
- Commit ONLY files I create/touch for this work (explicit `git add <path>`); do NOT sweep the pre-existing dirty tree.
- Do not push. Do not merge. Leave for user review.
