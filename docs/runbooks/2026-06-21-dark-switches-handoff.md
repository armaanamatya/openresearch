# Dark-Switches handoff — what ships now vs what needs your A/B (T1/T6)

> Companion to the plan (`docs/superpowers/plans/2026-06-21-dark-switches-plan.md`) and the
> research menu (`docs/superpowers/specs/2026-06-21-system-improvement-opportunities.md`).
> Branch: `feat/dark-switches` (off the trunk, **separate from `main`**). No GPU/API spend
> was incurred building this.

## The core idea
The system already contains expensive cost-savers and safety-nets, but they ship **OFF by
default**. The win is turning them on — but the repo rule is **≥3 paired A/B runs before
flipping any default**, and behavior-changing flips need GPU. So:

- **Phase 1 (shipped hermetically, default-ON / pure fixes):** no quality tradeoff to
  validate → these are live on the branch.
- **Phase 2 (mechanism ready, default OFF):** changes run behavior/quality → **you** run the
  A/B, then flip the default. Commands below.

## Phase 1 — DONE (verify in the PR, then merge to the trunk)
| Switch | What changed | Risk |
|---|---|---|
| Re-preflight after patch | A patch that still leaves a confident AST violation falls through to a full rewrite instead of returning success | none (pure correctness; fail-soft on scan error) |
| Orphan-guard default-ON | Abandoned GPU subprocesses are killed on a per-primitive timeout (opt-out `OPENRESEARCH_ORPHAN_GUARD=0`) | none (fail-soft per-pgid; reliability-only — can't change a score) |

Both covered by hermetic tests + full suite green (7037→7040 passing).

> **Preflight import-smoke was RECLASSIFIED to Phase 2** during implementation: a
> false-positive smoke failure short-circuits training and would block a run that would
> otherwise succeed — an outcome risk, so it must be A/B-validated, not flipped blind.

## Phase 2 — your A/B before flipping (each is a real $/quality lever)

For **each** switch: run ≥3 paired runs (same paper, same seed), control = flag OFF, arm =
flag ON, compare with the A/B harness, and flip the default only if the arm is ≥ control on
score AND wins on the lever (cost/reliability). Generic A/B shape:

```bash
# control (flag OFF) and arm (flag ON), paired
OPENRESEARCH_AB_ARM=control OPENRESEARCH_AB_PAIR_ID=ds-<switch>-1 \
  python -m backend.cli reproduce <paper> --mode rlm --sandbox runpod --model gpt-5 ...
OPENRESEARCH_AB_ARM=bes <THE_SWITCH_FLAG>=1 OPENRESEARCH_AB_PAIR_ID=ds-<switch>-1 \
  python -m backend.cli reproduce <paper> --mode rlm --sandbox runpod --model gpt-5 ...
# repeat for pair ids -2, -3, then:
python scripts/ab_compare.py --pair-id ds-<switch>-1 --require-stamped   # (and -2, -3)
```

| Switch | Flag to set ON in the arm | What to watch | Flip criterion |
|---|---|---|---|
| Cell-resume | `OPENRESEARCH_RESUME_CELLS=1` | reruns skip passed cells; **no stale-skip** of a cell that should rerun | score unchanged + GPU cost down on rerun-after-partial |
| Dead-training early-stop | `OPENRESEARCH_DEAD_LOSS_EARLYSTOP=1` | **no false early-stop** of a slow-but-healthy run | no score regression + dead cells stopped sooner |
| OOM hard-memcap | `OPENRESEARCH_OOM_ENFORCE=1` | OOM cells salvaged vs killed; no over-constraint | OOM cells recovered + score unchanged |
| HF/dataset cache persist | provision a RunPod network volume + set `OPENRESEARCH_RUNPOD_NETWORK_VOLUME_ID=<id>` | pod boot time ↓ (no 2–5 GB re-pull) | faster boots, identical results |
| Spot/interruptible GPUs | `OPENRESEARCH_RUNPOD_INTERRUPTIBLE=1` | preempt→resume works; cell finishes | ~50–70% cost down + runs complete |

**One-time infra (operator):** cache persistence (Task 8) and spot (Task 9) need a real
RunPod network volume / interruptible pods — provision once, then they apply to all runs.

## Recommended order to validate
1. **Cell-resume** + **cache persistence** — biggest, safest $ wins; validate first.
2. **Dead-training** + **OOM-enforce** — reliability; watch false-positive/over-constraint.
3. **Spot GPUs** — biggest absolute $ lever; validate last (most infra).

## What's explicitly NOT in scope here
Scoring default-flips (external_validator, evidence_audit) — those need the **labeled
honest/fab corpus** first (theme T7 in the research menu); separate effort.
