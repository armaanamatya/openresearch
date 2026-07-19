# Enrichment 1 — typed campaign branches

## Goal

Carry a deterministic branch type from policy plan to directives, assessment and
ASHA observation.  The initial campaign branch is `faithful`; a future ambiguity or
discovery fork can only be selected by an explicit policy plan, never by a grade.

## Test-first steps

1. Extend policy and directives round-trip tests to assert `branch_type` defaults
   to `faithful`, serializes to durable directives, and survives reload.
2. Extend assessment construction/serialization tests to assert the planned type is
   recorded on every assessment and adapter observation.
3. Add adapter tests for all three accepted types (`faithful`, `ambiguity`,
   `discovery`) and reject/normalize no untrusted free-text values; the adapter must
   not infer type from score or failure class.
4. Add `branch_type` to `NextAttemptPlan`, `AttemptDirectives`, and
   `AttemptAssessment`, update all builders/readers, and pass the plan's value at
   the sole fork/application point.  Legacy ledger/directive rows load as faithful.
5. Make initial and existing lineage arms explicitly plan `faithful`; do not add
   autonomous discovery spawning.
6. Verify that evidence-gate, verdict, champion and seeding predicates remain
   unchanged and do not key on an LLM grade.
7. Run focused type-propagation tests plus all campaign regressions.

## Acceptance

The ASHA core sees the durable branch type, but the OFF scheduler path is exactly
the preexisting serial policy path.  Discovery remains quarantined by the core and
is never silently created by a retry.
