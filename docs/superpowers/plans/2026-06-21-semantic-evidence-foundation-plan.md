# Semantic evidence foundation — implementation plan

> **Status:** proposed. Execute in order; no GPU or live API is needed for any
> implementation/test task. Every new flag defaults OFF and fails soft.

## Phase 0 — fixtures and compatibility contracts (S)

- [ ] Add hermetic SDAR, All-CNN, and generic sweep fixtures with paper spans,
  typed dimensions, duplicate metric values, and multiple attempts.
- [ ] Establish off-state snapshot tests for scope, cell aggregation, leaf scoring,
  and report serialization.
- [ ] Add a paired-A/B manifest validator requiring rubric pinning, scope equality,
  evidence fingerprints, and `experiment_arm` stamps.

## Phase 1 — contract model and intake persistence (M)

- [x] Persist the existing validated planner contract under an explicit default-OFF
  flag and attach it to `RunContext`, including cache-hit paths (`f64077ef`).
- [x] Run hermetic paired A/B coverage across SDAR-like RL, vision, and JAX-sweep
  planner inputs; ON preserved the public planner result and added only the
  contract artifact/context attachment (`97efffee`).
- [x] Add versioned Pydantic schemas for `ReproductionContract`, source links,
  unresolved fields, capability profile, `MetricContract`, and dimensions
  (`backend/agents/rlm/semantic_contract.py` — additive + inert: `SourceSpan`,
  `Provenance`(source spans + 0..1 confidence), `Dimension`, `MetricContract`,
  `AlgorithmInvariant`, `ResourceIdentity`, `CapabilityProfile`,
  `SemanticReproductionContract` v1 with an `unresolved` list; tests
  `tests/rlm/test_semantic_contract.py`).
- [ ] Implement fail-soft contract persistence under `rlm_state/` and an explicit
  legacy fallback resolver.
- [ ] Adapt PaperHint, YAML invariants, generated rubric, and effective scope as
  contract inputs without deleting their current consumers.
- [ ] Add unit tests for source/confidence preservation, missing fields, and
  unchanged behavior when `OPENRESEARCH_REPRO_CONTRACT` is off.

## Phase 2 — typed cells/results and metric semantics (L)

- [ ] Add `dimensions` to cell manifests and results; retain the current three-axis
  projection for SDAR.
- [ ] Make aggregation preserve dimensions rather than coercing arbitrary concepts
  into `baseline`.
- [ ] Drive metric-range, direction, split, denominator, and seed-stat diagnostics
  from `MetricContract` in warning-only mode.
- [ ] Exercise SDAR, image classification, and a non-model/environment/baseline
  sweep in hermetic tests.

## Phase 3 — canonical evidence bundle and coverage matrix (L)

- [ ] Mint immutable bundle receipts from the experiment/ledger boundary using
  deterministic digests and typed coordinates.
- [ ] Route scorer and report artifact selection through a canonical receipt
  resolver, with legacy fallback and `bundle_unverified` stamp.
- [ ] Compile result-leaf coverage obligations; distinguish unmet, covered, and
  unknown coverage without an LLM decision.
- [ ] Add adversarial fixtures for cross-attempt mixing, same metric value in two
  coordinates, and one-cell support for an aggregate claim.

## Phase 4 — trust and capability extensions (separate PRs)

- [ ] Provenance validity + finalize validator cap.
- [ ] Independent evaluator receipts, beginning with SDAR environments.
- [ ] Resource provenance audit and reproducibility lock bundle.
- [ ] Capability profiles, environment adapters, and runtime-neutral cell entrypoints.
- [ ] Remote stall guard, failure capsules, finalization identity, orientation budget,
  and deterministic next-action cards.

## Promotion gate

- [ ] Run targeted hermetic tests and the full suite for each PR.
- [ ] Run at least three paired A/B arms per score-affecting feature.
- [ ] Compare legacy versus typed/contract score, coverage, cost, false-positive
  rate, and grader variance; preserve a rollback flag.
- [ ] Only then propose a default flip.
