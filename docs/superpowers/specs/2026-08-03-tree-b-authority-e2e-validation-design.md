<!-- doc-meta: status=current; authored=2026-08-03 -->
# Tree-B authority — end-to-end validation + first real run (design)

**Date:** 2026-08-03 · **Status:** current · **Owner:** operator

## Problem

"Tree-B" = the reproduction scheduler **authority** (freeze / branch / revive / true-kill),
an ASHA successive-halving controller that promotes/freezes/kills same-paper branches in a
`campaign` loop. A 2026-08-03 recon (and this spec's own verification) established that the
**entire producer→receipt→authority chain is already built and wired**:

- `gpu_cell_runner.py:804` **unconditionally** sets `OPENRESEARCH_CELL_CHECKPOINT_DIR =
  <output>/checkpoints` for every cell.
- `baseline_implementation.py:76,1363-1376` copies `cell_checkpoint.py` into the cell sandbox and
  instructs the trainer to emit the 5-field checkpoint via `cell_checkpoint.write_checkpoint(...)`
  (model / optimizer / lr_scheduler / rng / data_order). The recovered `base_rn/train_cell.py`
  actually did (its lines 188-294).
- `reproduction_campaign.py:1087-1126` (`_authority_dispatch_impl`) reads the **real** latest
  checkpoint (`cell_checkpoint.latest_checkpoint_dir`), **fails closed** if absent, builds the
  receipt (`scheduler_receipt_producer.build_raw_receipt`), and records it to the controller with
  the fail-closed campaign ledger as attestor.
- The controller (`scheduler_authority_controller.py`), runtime (`scheduler_runtime.py`), evidence
  contract (`scheduler_evidence.py`), ASHA core (`asha_scheduler.py`), branch lineage
  (`branch_lineage.py`), and offline A/B gate (`asha_authority_gate.py`) are implemented + tested.
- An authority spec exists: `configs/adam_authority_spec.json`.

**The real gap:** every existing test drives the controller with **synthetic checkpoint bytes**
(`tests/rlm/test_scheduler_receipt_producer.py::_write_checkpoint_components` = fabricated data),
and **the authority campaign has never executed end-to-end.** The chain is proven in well-tested
pieces, but the **seams** between a *real trainer's* on-disk output and the receipt builder /
controller are unproven — exactly where integration bugs hide. This is a **validate + debug + run**
job, NOT a "build the producer" job.

## Goal

Prove Tree-B applies real freeze / promote / (true-)kill transitions from **real trainer-emitted
checkpoints**, hermetically first (no GPU, ~$0), fix whatever seams surface, then demonstrate it
once on a real GPU campaign. Preserve every invariant: **default-OFF + byte-identical off**,
**evidence-not-grade**, **fail-closed** (no fabricated receipt ever upgrades a decision).

## Non-goals

- No change to the ASHA decision math, the receipt contract, or the evidence gate.
- No default-flip of `OPENRESEARCH_SCHEDULER_AUTHORITATIVE` (stays OFF; a flip needs the standing
  ≥3-paired-A/B + grader-σ gate + operator sign-off — out of scope here).
- Not building the 7 Tree-A single-feature arms (operator deferred them).

## Design

### Phase 1 — Hermetic end-to-end (no GPU)

A new hermetic test (`tests/rlm/test_authority_e2e_real_checkpoint.py`) that exercises the chain
with a **real tiny trainer**, closing the synthetic-checkpoint gap:

1. **Tiny real trainer** — a minimal, CPU-only, train_cell.py-shaped module: a 2-layer `nn.Module`,
   a few SGD steps on synthetic tensors, then `torch.save(model.state_dict())` /
   `optimizer.state_dict()` / lr-scheduler state / `cell_checkpoint.capture_rng_state()` / a
   data-order blob → `cell_checkpoint.write_checkpoint(OPENRESEARCH_CELL_CHECKPOINT_DIR, step, …)`.
   Produces a genuine 5-component checkpoint dir on disk — the exact shape a real cell writes.
   (torch is a test-only import, guarded with `pytest.importorskip("torch")`.)
2. **≥2 competing branches** — run the tiny trainer twice into two branch cell-output dirs, one
   converging (low error metric) and one diverging (a `training_diverged`-classified metric), so
   the authority policy has a real promote candidate and a real kill candidate.
3. **Drive the real controller** — construct `SchedulerAuthorityController` with a tmp authority
   spec, `bootstrap()`, `claim_launches()`, then for each branch resolve the real checkpoint via
   `cell_checkpoint.latest_checkpoint_dir`, `build_raw_receipt(...)`, `record_cell_receipt(raw,
   attest=…)`, then `decide_rung(...)`.
4. **Assert real transitions** — the converging branch is **promoted**, the underperformer
   **frozen** (reversible), the diverged branch **true-killed**; the corresponding
   `branch_lineage` `DomainEvent`s (`BranchPromoted` / `FrozenPoolEviction` / `BranchTrueKilled`)
   are emitted to the `branch-tree:<campaign>` aggregate. Also assert the **fail-closed** path: a
   branch with no checkpoint dir raises rather than yielding a receipt.

This test is the deliverable of Phase 1: it proves the real producer output drives real authority
decisions, hermetically.

### Phase 2 — Fix the seams (TDD)

Phase 1 is expected to surface 1-3 real integration bugs (field name / dtype / path-shape
mismatches between the trainer's on-disk layout and `build_raw_receipt`'s expectations, or a
`data_order` / `capture_rng_state` contract mismatch). Each fix follows TDD: the failing Phase-1
assertion (or a narrower regression test) is RED first, then the minimal fix. No invariant is
weakened to make a test pass; if a real bug is in the receipt contract, fix the producer/consumer
seam, never the gate.

### Phase 3 — First real GPU campaign A/B

Run the real authority campaign once on GPU to demonstrate Tree-B end-to-end on a real paper:

```
OPENRESEARCH_SCHEDULER_TREE=1 OPENRESEARCH_SCHEDULER_AUTHORITATIVE=1 \
python -m backend.cli campaign <paper> --campaign-driver unified \
  --sandbox local --billing-sandbox gcp \
  --authority-spec-path configs/adam_authority_spec.json \
  --max-llm-usd <X> --max-gpu-usd <Y> --max-gpu-hours <Z>
```

- **VM:** allon-vm or a fresh L4 (NEVER base-vm — operator's friend's). 18h cap, STOP-not-DELETE,
  venv on PATH, seeded rubric, poll-scp collection, startup-script launch — every lesson from the
  2026-08-03 baseline/all_on runs applies (see the GCP runbook).
- **Deliverable:** a campaign whose `branch-tree:<campaign>` event log shows real
  freeze/promote/kill transitions backed by verified receipts; the offline `asha_authority_gate`
  can then evaluate it. One run is a demonstration, not a default-flip.

## Components & boundaries

| Unit | Responsibility | Touched? |
|---|---|---|
| tiny trainer (test fixture) | emit a real 5-field checkpoint from a CPU model | NEW (test-only) |
| `test_authority_e2e_real_checkpoint.py` | drive controller from real checkpoints; assert transitions | NEW |
| `scheduler_receipt_producer` / `_authority_dispatch_impl` | build/record receipt from real dir | fix seams only if Phase-1 reveals bugs |
| `cell_checkpoint` / controller / runtime / ASHA / lineage | unchanged | none (unless a seam bug) |

## Error handling / invariants

- **Fail-closed preserved:** no checkpoint ⇒ `SchedulerRuntimeError`, never a fabricated receipt
  (`reproduction_campaign.py:1090`). The Phase-1 test asserts this explicitly.
- **Default-OFF:** authority constructs nothing unless both flags + spec are present
  (`test_authority_controller_wiring.py` already guards byte-identical-off; unchanged).
- **Evidence-not-grade:** receipts key on the deterministic metric + checkpoint hashes, never an
  LLM grade — unchanged; the tiny trainer emits a real measured metric.

## Testing

- Phases 1-2: hermetic pytest only (`pytest.importorskip("torch")`, tmp_path, no sockets, no GPU).
  Run alongside the existing `tests/rlm/test_scheduler_*` + `test_asha_*` suites (must stay green).
- Phase 3: one operator GPU campaign; verify via the `branch-tree` event log + `asha_shadow_report`
  / `asha_authority_gate`, never the ledger alone.

## Success criteria

1. A hermetic test proves real-trainer checkpoints drive a real **promote + freeze + true-kill**
   through the authority controller, with the matching `DomainEvent`s — green.
2. The full `test_scheduler_*` / `test_asha_*` suite stays green (no regressions); any seam bug
   found is fixed with a named regression test.
3. One GPU authority campaign runs end-to-end and its branch-tree log shows verified
   receipt-backed transitions (demonstration; not a default-flip).
