# Enrichment 4 — branch-lineage event emission

## Goal

Emit the existing seven branch-tree domain events when an ASHA-enabled campaign
applies a continuation, using the existing F10 novelty fingerprint for dedup.

## Test-first steps

1. Add an event-store-backed test for a flag-off continuation and assert zero
   branch-tree events and unchanged ledger/SSE payloads.
2. Add a flag-on test that drives an initial plan and continuation and asserts a
   `BranchSpawned` then `RungClimbed` event on aggregate `branch-tree:<campaign_id>`.
3. Add promotion, frozen-pool, true-kill, and dedup test fixtures.  Assert only
   `training_diverged` may emit `BranchTrueKilled`; every repairable failure emits
   at most `FrozenPoolEviction`.
4. Implement a small fail-soft emitter beside `_decide_and_continue`/
   `_apply_continue` that maps the persisted plan and ASHA action to the existing
   event classes.  Read the F10 fingerprint already present in campaign directives;
   do not hash a second dedup key.
5. Preserve optimistic concurrency: load the aggregate version, append events in
   order, and treat event-store errors as non-authoritative observability failures
   until authority is enabled.
6. Update the ASHA documentation and run event-registry plus campaign regressions.

## Acceptance

Events audit scheduler branch state without changing verdict/evidence decisions.
Flag-off runs do not open or touch the event store for scheduler lineage.

## Feasibility guard

Only emit facts the current serial campaign actually performed.  A shadow advisory
is not a promotion, freeze, kill, checkpoint, or revival.  Until durable
checkpoint/step-rung transitions and authoritative deterministic evidence exist,
do not emit `FrozenPoolEviction`, `BranchRevived`, `BranchPromoted`, or
`BranchTrueKilled`; their required facts are unavailable.  Emit a root
`BranchSpawned` only after its durable launch/directive fingerprint is available,
and retain the other event schemas for their real transition owners.
