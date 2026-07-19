# Enrichment 2 — full-budget safety bracket

## Goal

Persist an explicit Hyperband `s=0` safety-slot marker from planning through the
attempt assessment into ASHA.  The marker exempts one faithful branch from halving;
it never changes the optimizer-step fidelity ladder.

## Test-first steps

1. Add policy tests showing the first faithful plan gets `is_safety_bracket=True`
   and later/ambiguity plans do not accidentally receive a second slot.
2. Add directive and assessment round-trip tests for the boolean, including legacy
   JSON rows defaulting to `False`.
3. Add adapter and shadow-advisory ON tests that show a marked branch receives the
   core's safety-bracket promotion behavior, while a non-marked peer can freeze.
4. Add an OFF equality test around the tree flag, proving this new metadata is not
   consulted by normal campaign decisions.
5. Thread the marker through `NextAttemptPlan`, `AttemptDirectives`,
   `AttemptAssessment`, the PLAN/LAUNCH builders, and the adapter.  Derive it only
   from harness-owned policy state, not scores or agent output.
6. Document that the safety bracket is a full-step-budget exception to halving, not
   a separate cost or fidelity meter.
7. Run focused tests and all campaign regressions.

## Acceptance

Exactly one initial faithful branch is protected for a campaign tree.  True-kill for
the existing `training_diverged` evidence remains possible; no other failure is
converted to a delete.
