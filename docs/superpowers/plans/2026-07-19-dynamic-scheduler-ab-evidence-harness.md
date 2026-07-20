# Scheduler authority A/B evidence harness

## Goal

Produce a hermetic CLI that consumes paired shadow/authority run artifacts,
requires at least three complete pairs, measures grader-score variation, and emits
a machine-readable evidence verdict.  It authorizes no default change itself.

## Test-first steps

1. Specify a stdlib-only JSON manifest: each pair has a stable `pair_id`, one
   shadow run directory, one authority run directory, and optional verified cost
   observations.  Reject duplicate IDs, missing paths, unpaired variants, and fewer
   than three pairs.
2. Extend the shadow-report reader with a pure JSON-producing API sufficient to
   count advisory coverage/actions without importing campaign code.
3. Add fixture-led tests that reject self-authored/no-op applied actions, <3 pairs,
   missing shadow coverage, mismatched terminal evidence verdict, grader sigma above
   the configured maximum, unbound calibration, and unverifiable/ledger-only cost
   claims. There is intentionally no passing action fixture until the campaign
   produces an immutable deterministic receipt.
4. Implement `asha_authority_gate.py` as stdlib-only. Read terminal verdict and
   grader sample scores from run artifacts; calculate population standard deviation
   per variant and paired deltas; mark costs `verified` only when a declared source
   record is present (tokens totals plus a run-bound node observation or provider
   bill-export record whose amount/identity can be validated).
5. Emit JSON and a concise terminal table.  The result has `eligible_for_operator_review`
   only when at least three pairs, complete scheduler coverage, matching deterministic
   evidence verdicts, and both sigmas <= the configured gate pass.  It never edits
   environment defaults or flags.
6. Document invocation, artifact provenance, and the required operator sign-off in
   `backend/agents/rlm/CLAUDE.md` and a dated runbook template.
7. Run its hermetic tests, doc registry/fidelity checks, and full campaign suite.

## Acceptance

The harness makes evidence gaps explicit: a $0 ledger, absent node observation, or
insufficient pairs is not success. Current runs lack the provenance-validated
metric/checkpoint and collector-attested cost receipts, so the harness must reject
every current authority action; it never turns on a default. Future receipt support
must additionally bind branch-to-attempt, full terminal evidence, paired run-spec/
seed/budget/model/sandbox controls, and collector attestation before an eligible
result is even possible.
