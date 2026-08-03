# Tree-B Authority End-to-End Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Prove the already-built Tree-B authority chain applies real freeze/promote/true-kill transitions from a REAL trainer's 5-field checkpoints (hermetically first), fix any seams, then run one GPU campaign A/B.

**Architecture:** The producer→receipt→authority chain exists (cells emit `cell_checkpoint.write_checkpoint`; `reproduction_campaign._authority_dispatch_impl` reads it → `build_raw_receipt` → `controller.record_cell_receipt` → `decide_rung`). Every current test uses SYNTHETIC checkpoint bytes. We add a hermetic test driving the real `SchedulerAuthorityController` from a real tiny-CPU-trainer's checkpoint, fix whatever seams surface (TDD), then a GPU run.

**Tech Stack:** Python 3.12, pytest (`pytest.importorskip("torch")`), stdlib `cell_checkpoint`, `scheduler_authority_controller`, `scheduler_receipt_producer`, `scheduler_evidence`.

---

## Reference interfaces (verified 2026-08-03)

- `SchedulerAuthorityController(run_dir, *, campaign_id, spec: SchedulerAuthoritySpec, event_sink=None)` → `.bootstrap()`, `.claim_launches(max_parallel=)`, `.record_cell_receipt(raw, attest=callable)`, `.decide_rung(rung=, provider_gpu_usd_by_branch=)`.
- Spec loaded by `campaign_composition.load_authority_spec(path, paper_ref=...)`; format = `configs/adam_authority_spec.json` (`ladder{paper_ref,metric_id,direction,r_max_steps,rung_steps,schedule_source_sha256}`, `width{gpu_usd_budget,a100_cap,safety_gpu_usd_budget,discovery_gpu_usd_budget,eta,noise_floor}`, `branches[{branch_id,branch_type,hypothesis_fingerprint,seed,is_safety_bracket}]`).
- `cell_checkpoint.write_checkpoint(dir, step, *, model:bytes, optimizer:bytes, lr_scheduler:bytes, rng:bytes, data_order:bytes)`; `latest_checkpoint_dir(dir)`; `capture_rng_state()`.
- `scheduler_receipt_producer.build_raw_receipt(*, run_dir, cell_output_dir, checkpoint_components_dir, ladder, campaign_id, branch_id, parent_branch_id, attempt_n, cell_id, from_step, to_step, seed, termination_cause, dataset_manifest_path, run_spec_path)`.
- Existing patterns: `tests/rlm/test_scheduler_authority_controller.py`, `test_authority_controller_wiring.py`, `test_scheduler_receipt_producer.py`.

---

## Task 1: Tiny real-trainer checkpoint fixture

**Files:**
- Test: `tests/rlm/test_authority_e2e_real_checkpoint.py` (create)

- [ ] **Step 1: Write a helper that runs a real tiny CPU trainer and emits a real 5-field checkpoint.** The helper trains a 2-layer `nn.Linear` model N SGD steps on fixed synthetic tensors, returns the final metric (train loss), and writes a real checkpoint:

```python
import io, json, struct
import pytest
from pathlib import Path
from backend.agents.rlm import cell_checkpoint

def _train_and_checkpoint(ckpt_dir: Path, *, steps: int, lr: float, seed: int) -> float:
    torch = pytest.importorskip("torch")
    torch.manual_seed(seed)
    x = torch.randn(64, 8); y = torch.randn(64, 1)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, steps // 2), gamma=0.1)
    loss_val = float("nan")
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward(); opt.step(); sched.step()
        loss_val = float(loss.detach())
    def _b(sd): buf = io.BytesIO(); torch.save(sd, buf); return buf.getvalue()
    cell_checkpoint.write_checkpoint(
        ckpt_dir, steps,
        model=_b(model.state_dict()), optimizer=_b(opt.state_dict()),
        lr_scheduler=json.dumps(sched.state_dict()).encode(),
        rng=cell_checkpoint.capture_rng_state(),
        data_order=struct.pack("<q", seed),
    )
    return loss_val
```

- [ ] **Step 2: Write a helper that lays out a branch cell-output dir** (metrics.json with the measured metric the receipt reads, plus the checkpoint under `checkpoints/`):

```python
def _branch_cell_out(root: Path, branch_id: str, *, metric: float, metric_id: str) -> Path:
    cell = root / branch_id / "code"; (cell).mkdir(parents=True, exist_ok=True)
    (cell / "metrics.json").write_text(json.dumps({metric_id: metric}))
    return cell  # checkpoints/ live under cell/checkpoints
```

- [ ] **Step 3: Run** `pytest tests/rlm/test_authority_e2e_real_checkpoint.py -q` — expected: no tests yet / collection ok (helpers only). Commit.

```bash
git add tests/rlm/test_authority_e2e_real_checkpoint.py
git commit -m "Add tiny real-trainer checkpoint fixture for Tree-B E2E"
```

## Task 2: Hermetic E2E — real checkpoint → receipt → authority transition

**Files:**
- Test: `tests/rlm/test_authority_e2e_real_checkpoint.py` (modify)

- [ ] **Step 1: Write the failing E2E test.** Load a tiny authority spec (2-3 branches: one converging, one under-performing, one diverged), construct the real controller, bootstrap, claim, run the real trainer per branch into its cell-out + `checkpoints/`, resolve the real checkpoint via `cell_checkpoint.latest_checkpoint_dir`, `build_raw_receipt(...)`, `record_cell_receipt(raw, attest=<ledger append>)`, then `decide_rung(...)`. Assert: converging branch **promoted**, under-performer **frozen**, diverged branch **true-killed**; and the matching `branch_lineage` events exist in the event store. Use the real `build_raw_receipt`/controller — no synthetic bytes. (Exact spec construction + attest callable + metric plumbing resolved against `test_scheduler_authority_controller.py` and `scheduler_evidence.write_verified_receipt` while writing.)

- [ ] **Step 2: Run to verify it fails** for the RIGHT reason (a real seam mismatch, not a typo): `pytest tests/rlm/test_authority_e2e_real_checkpoint.py -xvs`. Record the first real failure.

- [ ] **Step 3: Assert the fail-closed path too** — a branch whose `checkpoints/` dir is empty makes `latest_checkpoint_dir` return None and the receipt build/record path raise (mirrors `reproduction_campaign.py:1090`).

- [ ] **Step 4: Commit the test** once it expresses the intended behavior (may still be RED pending Task 3 seam fixes).

```bash
git add tests/rlm/test_authority_e2e_real_checkpoint.py
git commit -m "Add hermetic Tree-B E2E test: real checkpoint drives real authority transitions"
```

## Task 3: Fix the seams (TDD loop, emergent)

**Files:** whichever the failure points to — likely `scheduler_receipt_producer.py`, `scheduler_evidence.py`, or `reproduction_campaign.py` (the real-trainer-output ⇄ receipt-builder seam).

- [ ] **Step 1:** For each failure from Task 2: read the exact error, identify the seam (field name / dtype / path shape / metric key), confirm it's a real seam bug (not a test error). If it's a test-harness mistake, fix the test.
- [ ] **Step 2:** Write/keep a narrow RED assertion for the seam, apply the MINIMAL production fix, do NOT weaken any fail-closed/evidence invariant.
- [ ] **Step 3:** Re-run `pytest tests/rlm/test_authority_e2e_real_checkpoint.py -xvs` → GREEN.
- [ ] **Step 4:** Commit each seam fix with its regression test.

```bash
git add -A && git commit -m "Fix Tree-B seam: <what> (real checkpoint -> receipt)"
```

## Task 4: Full authority-suite regression

**Files:** none (verification).

- [ ] **Step 1:** Run the whole scheduler/authority suite: `pytest tests/rlm/test_asha_*.py tests/rlm/test_scheduler_*.py tests/rlm/test_branch_lineage.py tests/rlm/test_authority_*.py tests/rlm/test_campaign_cohort_loop.py -q`. Expected: all green (no regressions from the seam fixes).
- [ ] **Step 2:** `uvx ruff@0.15.16 check backend/agents/rlm/ tests/rlm/test_authority_e2e_real_checkpoint.py`. Expected: clean.
- [ ] **Step 3:** Commit if any lint fixups.

## Task 5: First real GPU authority campaign (Phase 3)

**Files:** none (operator run); uses `configs/adam_authority_spec.json` or a ResNet-shaped spec.

- [ ] **Step 1:** Provision a fresh L4 VM (NEVER base-vm — operator's friend's) with the 2026-08-03 recipe: 18h cap, STOP-not-DELETE, venv on PATH, torch, Foundry creds, startup-script launch, serial-console verify, poll-scp collect. Seed the rubric.
- [ ] **Step 2:** Launch: `OPENRESEARCH_SCHEDULER_TREE=1 OPENRESEARCH_SCHEDULER_AUTHORITATIVE=1 python -m backend.cli campaign <paper> --campaign-driver unified --sandbox local --billing-sandbox gcp --authority-spec-path configs/adam_authority_spec.json --max-llm-usd X --max-gpu-usd Y --max-gpu-hours Z`.
- [ ] **Step 3:** Collect the campaign; verify the `branch-tree:<campaign>` event log shows real receipt-backed freeze/promote/kill transitions (via `python -m backend.agents.rlm.asha_shadow_report <run_dir>` + `asha_authority_gate`), never the ledger alone. Record the outcome in `docs/2026-08-01-feature-ablation-results.md` (Tree-B row) + stop the VM.

---

## Self-review

- **Spec coverage:** Phase 1 → Tasks 1-2; Phase 2 → Task 3; regression → Task 4; Phase 3 GPU → Task 5. All spec sections covered.
- **Placeholders:** the emergent seam fixes (Task 3) and the exact spec/attest plumbing (Task 2) are deliberately resolved against the real interfaces during TDD — the test *intent* + calls are concrete; the last-mile signatures are read at implementation time (executor has full context). Phase-3 `<paper>`/`X/Y/Z` are operator inputs.
- **Type consistency:** controller/receipt/checkpoint signatures above match the verified 2026-08-03 reads.
