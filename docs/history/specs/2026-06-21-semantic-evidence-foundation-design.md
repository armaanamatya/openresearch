# Semantic evidence foundation — design

> **Status:** proposed · companion to `2026-06-21-new-ideas-round-2.md`.
> All new controls default OFF, fail soft, and require at least three stamped paired
> A/B runs before a default changes. No feature may upgrade a score or verdict from
> incomplete evidence.

## Problem

The current pipeline can produce a rubric, scope, code, cells, metrics, and report
for an arbitrary paper, but those representations are independently derived. The
result is a SDAR-shaped execution tree (`model × env × baseline`) and evidence that
can be selected from separate files or attempts. This limits generality and leaves
the score vulnerable to accidental cross-attempt or cross-coordinate attribution.

## Target architecture

```text
paper spans ──> ReproductionContract ──> typed cells/results ──> EvidenceBundle
                     │                         │                    │
                     ├─ MetricContract          └─ exact leaf coverage └─ scorer/report
                     └─ capability/environment profiles
```

### ReproductionContract

`rlm_state/reproduction_contract.json` is a versioned, source-linked intake
artifact. It contains requirements, variants, typed dimensions, metric contracts,
algorithm invariants, resource identities, capability/environment profiles, and
an explicit `unresolved` list. Every asserted field carries source spans and a
confidence. The generator may be LLM-assisted, but consumers only use normalized
facts plus their evidence links.

When `OPENRESEARCH_REPRO_CONTRACT` is unset, current PaperHint/YAML/generated-rubric
paths remain byte-identical. When enabled and contract extraction is incomplete,
consumers retain current behavior for that field and stamp it unresolved; they never
invent a requirement.

### Typed dimensions and MetricContract

Cells and results gain `dimensions: dict[str, str | int | float | bool]` while
maintaining the legacy three-axis projection for existing SDAR scoring. A
MetricContract declares identifier, unit, direction, range, aggregation, split,
denominator, and uncertainty requirements. It drives validation and evidence
matching; it does not reject a run at first. The first rollout is warning-only.

### EvidenceBundle and exact coverage

At an experiment's successful terminal boundary, the harness writes an immutable
receipt containing attempt/ledger identity, code-tree digest, metrics digest,
artifact paths, typed coordinates, and an optional evaluator receipt. The scorer
and report resolve a single receipt through a canonical selector. A result leaf is
then covered only if its required coordinates are satisfied by the selected bundle;
an unparseable leaf is `coverage_unknown`, not a false failure.

The bundle is a coherence layer, not a cryptographic trust boundary. A later sealed
receipt may sign it, but this design first eliminates selection ambiguity.

## Rollout and safety

1. Persist and compare contracts/bundles without score or verdict mutation.
2. Enable warning-only typed-dimension, metric, and coverage diagnostics on paired
   arms; retain legacy projections.
3. Add deterministic caps only for internally contradictory or cross-bundle
   evidence. Unknown/unavailable evidence remains unverified, never a crash.
4. Require three paired A/B runs with pinned rubric, comparable scope, stamped
   `experiment_arm`, and grader-noise review before changing defaults.

## Follow-on capabilities

The foundation unlocks the remaining round-two work: provenance validity and final
validator caps; independent evaluator receipts; remote stall/failure capsules;
capability profiles; resource audits; runtime-neutral cell entrypoints; locks; and
environment adapters. Each remains a separately gated implementation slice.
