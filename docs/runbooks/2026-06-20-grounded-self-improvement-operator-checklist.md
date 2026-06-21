# Grounded self-improvement + harness reliability — operator checklist

> **Status:** Current · runbook · authored 2026-06-20. Companion to the design
> spec `docs/superpowers/specs/2026-06-20-grounded-self-improvement-and-harness-reliability-redesign-design.md`
> and the implementation plan `docs/superpowers/plans/2026-06-20-grounded-self-improvement-implementation-plan.md`.
>
> Everything below ships **flag-gated, default-OFF**. With every flag unset the
> harness is byte-identical to its prior baseline. The code + hermetic tests are
> done; the **GPU/API validation runs in this checklist are operator-run** and were
> NOT executed during implementation (no money spent, no default flipped).

## 1. What shipped (the three-tier trust model)

A self-improvement loop's fitness signal is the **deterministic evidence layer**,
never the LLM grade.

- **Tier 1 — deterministic floor.** `OPENRESEARCH_ZERO_METRICS_GUARD` adds a
  zero/constant-metrics veto in `run_experiment`: an all-zero or constant
  result that *claims* GPU training but has **no `provenance.json`** is degraded
  to the repairable `fabrication_suspected`. This is the guard that would have
  caught the SDAR v6 hallucination (real 8-GPU training, all-0.0 metrics, real
  keys) that the stub guard and VRAM antifab both missed.
- **Tier 2 — external adversarial validator.** A *separate-model* panel points at
  suspicions; the harness machine-checks each **typed predicate** (provenance /
  not-all-constant / gpu-claim-plausible / rerun-agrees) against the artifact —
  citation existence is necessary but not sufficient. Min-aggregation veto.
- **Tier 3 — grounded self-improvement.** The veto (Tier 1 or Tier 2) feeds a
  **bounded, honest** fix-first repair loop; cross-run *positive recipes* are
  admitted only on Tier-1 + validator evidence, never the grade (the red line).

## 2. The new flags

| Flag | Default | What it does |
|---|---|---|
| `OPENRESEARCH_ZERO_METRICS_GUARD` | off | Tier-1 zero/constant-metrics fabrication veto in `run_experiment`. |
| `OPENRESEARCH_LIFECYCLE_LEDGER` | off | Append-only, redacted, record-only per-primitive evidence ledger (`rlm_state/lifecycle/ledger.jsonl`). |
| `OPENRESEARCH_EXTERNAL_VALIDATOR` | off | Engages the Tier-2 adversarial panel + the fix-first loop driven by its veto. |
| `OPENRESEARCH_VALIDATOR_BACKEND` | unset | Validator transport: `azure` / `oauth` / `anthropic` / `openai` / `azure-foundry`. Fail-CLOSED (a requested-but-unbuildable validator raises). |
| `OPENRESEARCH_VALIDATOR_MODEL` | unset | Validator model; for `azure` this is the **deployment**, overriding `AZURE_OPENAI_DEPLOYMENT`. |
| `OPENRESEARCH_VALIDATOR_PANEL_N` | 2 | Panel sample count. |
| `OPENRESEARCH_REPAIR_MAX_ITERATIONS` | 4 | Ceiling for the fix-first repair loop (atop `OPENRESEARCH_MIN_REPAIR_ITERATIONS`); exhaustion → honest `repair_exhausted`. |
| `OPENRESEARCH_POSITIVE_RECIPES` | off | Cross-run positive-recipe memory (admit at finalize, inject at the implementer prompt). |
| `--run-spec <path.json>` | — | CLI: load a JSON of `OPENRESEARCH_*` into the env sink before flag resolution (explicit flags win); the GCP launcher ships one spec instead of the env whitelist. |

## 3. The recommended validator panel for the two funded transports

The validator requires `validator_family ≠ executor_family` for full
independence, but **same-provider-different-model is also supported** (a "weak"
panel). With only Azure OpenAI + Claude OAuth funded:

- **`independent` (strongest):** `--models executor=sonnet,validator=gpt-4o-azure`
  + `OPENRESEARCH_VALIDATOR_BACKEND=azure`. Executor = `claude-oauth` (Sonnet),
  validator = Azure OpenAI gpt-4o. Cross-family, $0 OAuth executor, no dead keys.
- **`weak` (supported):** `--models executor=gpt-4o-azure,validator=azure` with
  `AZURE_OPENAI_DEPLOYMENT=<deployA>` and `OPENRESEARCH_VALIDATOR_MODEL=<deployB>`.
  Both Azure OpenAI, **different deployments** — emits a soft `validator_separation_weak`
  notice, never blocks.
- **`degraded` (loud warning):** same deployment for both — emits
  `validator_separation_degraded`; the machine-checked veto still stands, only the
  LLM's suspicion-pointing is treated as non-independent.

A fully-OFF `OPENRESEARCH_VALIDATOR_BACKEND` leaves the validator unavailable and
the Tier-1 floor is the sole backstop.

## 4. Operator validation runs (NOT run during implementation)

These cost money and need a working RLM root (see the handoff for the root-matrix
state). Run them before flipping any flag to default-ON.

- [ ] **§5.2 precondition experiment (~$2).** Run SDAR once with
  `OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD=16` (≈ pre-2026-06-17). Self-recovers
  ⇒ the detector regressed (lean nudge-first); still loops ⇒ the OAuth root is
  genuinely degenerate (lean on a keyed/Foundry root). Record the outcome.
- [ ] **Per-flag A/B (≥3 paired SDAR runs each).** For each score-changing flag,
  run ≥3 paired SDAR runs (`scripts/ab_compare.py`, `experiment_arm` stamp) before
  flipping its default. Keep adaptive gating OFF on A/B arms.
- [ ] **End-to-end SDAR with the guards on.** Recommended config:
  `OPENRESEARCH_ZERO_METRICS_GUARD=1 OPENRESEARCH_EXTERNAL_VALIDATOR=1`
  `OPENRESEARCH_VALIDATOR_BACKEND=azure` + the `independent` panel above, on the
  proven Sonnet-executor / cells-route / 8×A100 path. Confirm a v6-style all-zero
  cell is vetoed → repaired → either fixed or shipped as an honest `repair_exhausted`.

## 5. Remaining polish (documented, not blocking)

- The degenerate-loop nudge already names the missing lifecycle stage; citing the
  exact `rlm_state/lifecycle/ledger.jsonl` record in the nudge text is a small
  follow-on (only meaningful when the ledger flag is on).
- `OPENRESEARCH_VALIDATOR_RERUN_ON_SUSPICION` (the §7.5 perturbed re-run) is
  stubbed in P2; the GPU-backed re-materialization is a future increment.
