# Enrichment 5 — gated scheduler authority

## Goal

Provide the default-OFF authority seam and its durable audit contract without
misrepresenting an LLM-score shadow advisory as an authority decision. A future
mapper may adopt ASHA only under `OPENRESEARCH_SCHEDULER_AUTHORITATIVE`, preserving
the normal deterministic terminal checks and replacing only an otherwise-CONTINUE
next plan. The existing tree flag remains the shadow-observation switch.

## Test-first steps

1. Add a flag-off byte-equality test for `CampaignStages.decide`: with all new
   inputs present and authority absent, the output equals the preexisting policy
   decision byte-for-byte.
2. Add an authority-on test where the policy returns each terminal kind
   (`REPRODUCED`, `CONTRADICTED`, `INFEASIBLE`, `EXHAUSTED`); assert all five
   `Decision.to_dict()` keys and values are preserved exactly, including terminal
   champion pointers.
3. Add authority-on CONTINUE tests proving the current grade-derived advisory is
   refused, the five base keys and `next_plan.scope_rung` survive unchanged, and
   every terminal decision takes precedence.
4. Add adversarial test data where a high LLM grade conflicts with evidence and
   assert scheduler adoption is driven only by stored assessment evidence and the
   core's deterministic action.
5. Implement a pure `_maybe_apply_asha_authority` after `decision.to_dict()` and
   before attaching the advisory.  Gate it with the canonical explicit truthy
   expression; it must no-op unless both tree and authority flags are true.
6. Do not select a promoted/frozen branch until the campaign persists a validated
   metric/checkpoint receipt and has a real branch queue. The authority seam must
   explicitly record `applied:false` rather than manufacture a changed plan.
7. Register and document the new default-OFF flag; regenerate the flag registry;
   run the doc-fidelity guard, focused authority tests and campaign regressions.
8. When the authority flag is evaluated, persist an additive
   `asha_authority_audit` mapping only on the flag-on path.  It records
   `enabled`, `applied`, action, source branch, and deterministic evidence basis;
   an unapplied fail-closed decision is explicit rather than inferred from env.

## Acceptance

Authority is opt-in and fail-closed: until the required receipt exists it is an
explicit audit only, cannot bypass any evidence terminal rule, and cannot
true-delete anything. Default behavior stays byte-identical.

## Feasibility guard

The current shadow adapter ranks `final_report.score` and receives a campaign
scope rung, neither of which is an authoritative deterministic defining metric or
a paper-pinned optimizer-step checkpoint rung.  Therefore the authoritative mapper
must preserve the base `CONTINUE` unchanged unless future input supplies a
harness-proven deterministic metric and step lineage.  This is a fail-closed
authority guard, not a grade-derived fallback; the A/B gate cannot approve a
default change without at least one audited deterministic action.
