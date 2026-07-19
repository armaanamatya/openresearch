# Enrichment 5 — gated scheduler authority

## Goal

Adopt ASHA only under `OPENRESEARCH_SCHEDULER_AUTHORITATIVE`; preserve the normal
deterministic campaign terminal checks and replace only an otherwise-CONTINUE next
plan with a promoted/safety branch plan.  The existing tree flag remains the
shadow-observation switch.

## Test-first steps

1. Add a flag-off byte-equality test for `CampaignStages.decide`: with all new
   inputs present and authority absent, the output equals the preexisting policy
   decision byte-for-byte.
2. Add an authority-on test where the policy returns each terminal kind
   (`REPRODUCED`, `CONTRADICTED`, `INFEASIBLE`, `EXHAUSTED`); assert all five
   `Decision.to_dict()` keys and values are preserved exactly, including terminal
   champion pointers.
3. Add authority-on CONTINUE tests for promote, freeze-only, true-kill, malformed
   advisory, and no-advisory cohorts.  Assert the output always contains the five
   decision keys and a valid `next_plan.scope_rung`; freeze/kill-only outcomes must
   retain serial continuation rather than create a new stop condition.
4. Add adversarial test data where a high LLM grade conflicts with evidence and
   assert scheduler adoption is driven only by stored assessment evidence and the
   core's deterministic action.
5. Implement a pure `_maybe_apply_asha_authority` after `decision.to_dict()` and
   before attaching the advisory.  Gate it with the canonical explicit truthy
   expression; it must no-op unless both tree and authority flags are true.
6. For an eligible CONTINUE, select only an existing promoted or safety branch,
   copy the existing plan then retain its `scope_rung`, branch metadata, and all
   unrelated plan fields.  Never manufacture a terminal decision, lower a scope,
   or overwrite stop rules.
7. Register and document the new default-OFF flag; regenerate the flag registry;
   run the doc-fidelity guard, focused authority tests and campaign regressions.

## Acceptance

Authority is opt-in, cannot bypass any evidence terminal rule, and cannot true-delete
anything except the core's existing provable `training_diverged` mapping.  Default
behavior stays byte-identical.
