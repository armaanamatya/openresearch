"""Hermetic Tree-B end-to-end: a REAL tiny CPU trainer's 5-field checkpoint drives
the REAL ``SchedulerAuthorityController`` through a real promote + freeze + true-kill,
with the matching ``branch_lineage`` DomainEvents.

Every existing scheduler test drives the controller with SYNTHETIC checkpoint bytes.
This test closes that gap: the checkpoint components come from
``cell_checkpoint.write_checkpoint`` fed by an actual ``torch`` model/optimizer/
lr-scheduler/rng/data-order, so the producer→receipt→authority seam is exercised
against a real trainer's on-disk layout. torch is a test-only import (guarded).

No GPU, no network, no LLM. The metric is the deterministic train loss read from
``metrics.json`` — never an LLM grade.
"""
from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import pytest

from backend.agents.rlm import cell_checkpoint
from backend.agents.rlm.branch_lineage import (
    BranchPromoted,
    BranchSpawned,
    BranchTrueKilled,
    FrozenPoolEviction,
    RungClimbed,
)
from backend.agents.rlm.scheduler_authority_controller import SchedulerAuthorityController
from backend.agents.rlm.scheduler_evidence import PaperStepLadder
from backend.agents.rlm.scheduler_receipt_producer import build_raw_receipt
from backend.agents.rlm.scheduler_runtime import (
    BranchTemplate,
    SchedulerAuthoritySpec,
)

# The event sink emits ``(kind, payload)`` facts; these are the lineage
# DomainEvent classes each kind reconstitutes into (proves the payload is a
# valid branch_lineage event, per the design spec's success criterion).
_EVENT_CLASSES = {
    "branch_spawned": BranchSpawned,
    "rung_climbed": RungClimbed,
    "branch_promoted": BranchPromoted,
    "frozen_pool_eviction": FrozenPoolEviction,
    "branch_true_killed": BranchTrueKilled,
}


# --------------------------------------------------------------------------- #
# Task 1 — a tiny REAL CPU trainer emitting a real 5-field checkpoint           #
# --------------------------------------------------------------------------- #


def _train_and_checkpoint(ckpt_dir: Path, *, steps: int, lr: float, seed: int) -> float:
    """Train a real 2-layer MLP ``steps`` SGD steps on fixed synthetic tensors and
    write a genuine 5-component checkpoint (the exact shape a real cell emits).

    Returns the final train loss (a deterministic measured metric)."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(seed)
    x = torch.randn(64, 8)
    y = torch.randn(64, 1)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)
    )
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, steps // 2), gamma=0.1)
    loss_val = float("nan")
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        loss_val = float(loss.detach())

    def _b(state) -> bytes:
        buf = io.BytesIO()
        torch.save(state, buf)
        return buf.getvalue()

    cell_checkpoint.write_checkpoint(
        ckpt_dir,
        steps,
        model=_b(model.state_dict()),
        optimizer=_b(opt.state_dict()),
        lr_scheduler=json.dumps(sched.state_dict()).encode("utf-8"),
        rng=cell_checkpoint.capture_rng_state(),
        data_order=struct.pack("<q", seed),
    )
    return loss_val


def _branch_cell_out(
    root: Path, branch_id: str, *, metric: float, metric_id: str
) -> Path:
    """Lay out a branch cell-output dir: ``metrics.json`` (the measured metric the
    receipt reads) with the checkpoint under ``code/checkpoints/``. Returns the
    cell dir (``checkpoints/`` lives beneath it)."""
    cell = root / branch_id / "code"
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "metrics.json").write_text(json.dumps({metric_id: metric}), encoding="utf-8")
    return cell


# --------------------------------------------------------------------------- #
# Task 2 — spec / ladder / attest plumbing (mirrors the synthetic-driven tests) #
# --------------------------------------------------------------------------- #

_PAPER_REF = "1412.6980"
_METRIC_ID = "train_loss"


def _f10(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _spec() -> SchedulerAuthoritySpec:
    """Three same-rung halvable branches (all faithful/ambiguity, none safety/
    discovery) so one ``decide_rung`` yields promote + freeze + kill."""
    ladder = PaperStepLadder(
        paper_ref=_PAPER_REF,
        metric_id=_METRIC_ID,
        direction="minimize",  # train loss: lower is better
        r_max_steps=50,
        rung_steps=(10, 50),
        schedule_source_sha256="a" * 64,
    )
    return SchedulerAuthoritySpec(
        ladder=ladder,
        branches=(
            BranchTemplate("converging", "faithful", _f10("converging"), seed=1),
            BranchTemplate("underperformer", "ambiguity", _f10("underperformer"), seed=2),
            BranchTemplate("diverged", "ambiguity", _f10("diverged"), seed=3),
        ),
        # k = int(gpu_usd_budget // max_cost) = int(1.0 // 1.0) = 1 → top-1 promoted.
        gpu_usd_budget=1.0,
        a100_cap=3,
    )


def _attestor(run_dir: Path):
    """The fail-closed campaign ledger attestor (mirrors the synthetic tests):
    ``write_verified_receipt`` appends the ``scheduler_receipt`` row the evidence
    reader requires."""
    ledger = run_dir / "campaign" / "attempts.jsonl"

    def attest(row):
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row)) + "\n")
            handle.flush()

    return attest


def _write_run_spec_fingerprints(run_dir: Path) -> tuple[Path, Path]:
    """The dataset-manifest + run-spec files ``build_raw_receipt`` fingerprints."""
    rlm_state = run_dir / "rlm_state"
    rlm_state.mkdir(parents=True, exist_ok=True)
    dataset = rlm_state / "dataset-manifest.json"
    run_spec = rlm_state / "run-spec.json"
    dataset.write_text('{"dataset":"synthetic-pinned"}', encoding="utf-8")
    run_spec.write_text('{"image":"cpu-pinned@sha256"}', encoding="utf-8")
    return dataset, run_spec


def _build_and_record(
    controller: SchedulerAuthorityController,
    run_dir: Path,
    *,
    branch_id: str,
    metric: float,
    steps: int,
    lr: float,
    seed: int,
    attempt_n: int,
    termination_cause: str | None,
    dataset: Path,
    run_spec: Path,
) -> None:
    """Run the real trainer for one branch, resolve its real checkpoint, build the
    receipt from the real on-disk output, and record it to the controller."""
    ladder = controller.spec.ladder
    cell = _branch_cell_out(run_dir, branch_id, metric=metric, metric_id=ladder.metric_id)
    ckpt_root = cell / "checkpoints"
    _train_and_checkpoint(ckpt_root, steps=steps, lr=lr, seed=seed)
    components = cell_checkpoint.latest_checkpoint_dir(ckpt_root)
    assert components is not None, "real trainer must have written a checkpoint"

    branch = controller.branches[branch_id]
    raw = build_raw_receipt(
        run_dir=run_dir,
        cell_output_dir=cell,
        checkpoint_components_dir=components,
        ladder=ladder,
        campaign_id=controller.runtime.campaign_id,
        branch_id=branch_id,
        parent_branch_id=branch.parent_branch_id,
        attempt_n=attempt_n,
        cell_id=f"cell-{branch_id}",
        from_step=0,
        to_step=ladder.rung_steps[0],
        seed=branch.seed,
        termination_cause=termination_cause,
        dataset_manifest_path=dataset,
        run_spec_path=run_spec,
    )
    # Deterministic metric provenance: the receipt value is the measured loss,
    # never an LLM grade.
    assert raw["metric"]["value"] == metric
    assert raw["metric"]["id"] == ladder.metric_id
    controller.record_cell_receipt(raw, attest=_attestor(run_dir))


def test_real_checkpoint_drives_promote_freeze_and_true_kill(tmp_path):
    """The full hermetic Tree-B chain from a real trainer's checkpoints."""
    run_dir = tmp_path
    events: list[tuple[str, dict]] = []
    controller = SchedulerAuthorityController(
        run_dir,
        campaign_id="campaign-e2e",
        spec=_spec(),
        event_sink=lambda kind, payload: events.append((kind, dict(payload))),
    )
    assert controller.bootstrap() == 3
    dataset, run_spec = _write_run_spec_fingerprints(run_dir)

    # Claim the full cohort (a100_cap=3), then run the real trainer per branch.
    claimed = controller.claim_launches()
    assert {item.branch_id for item in claimed} == {"converging", "underperformer", "diverged"}

    # converging → low loss (many steps, healthy lr) → PROMOTE (top-k).
    _build_and_record(
        controller, run_dir, branch_id="converging", metric=0.05,
        steps=40, lr=0.05, seed=1, attempt_n=1, termination_cause=None,
        dataset=dataset, run_spec=run_spec,
    )
    # underperformer → high loss, no divergence → FREEZE (halved below top-k).
    _build_and_record(
        controller, run_dir, branch_id="underperformer", metric=0.90,
        steps=2, lr=0.001, seed=2, attempt_n=1, termination_cause=None,
        dataset=dataset, run_spec=run_spec,
    )
    # diverged → finite measured loss but literal training_diverged cause → TRUE-KILL.
    _build_and_record(
        controller, run_dir, branch_id="diverged", metric=0.70,
        steps=2, lr=0.001, seed=3, attempt_n=1, termination_cause="training_diverged",
        dataset=dataset, run_spec=run_spec,
    )

    result = controller.decide_rung(
        rung=0,
        provider_gpu_usd_by_branch={
            "converging": 1.0, "underperformer": 1.0, "diverged": 1.0,
        },
    )
    actions = {item.branch_id: item.action for item in result.actions}
    assert actions == {
        "converging": "promote",
        "underperformer": "freeze",
        "diverged": "kill",
    }

    # The matching branch_lineage DomainEvents were emitted to the event sink,
    # and each payload reconstitutes into its lineage event (the persist contract).
    kinds = [kind for kind, _ in events]
    assert "branch_promoted" in kinds
    assert "frozen_pool_eviction" in kinds
    assert "branch_true_killed" in kinds
    reconstructed = [
        _EVENT_CLASSES[kind](**payload)
        for kind, payload in events
        if kind in _EVENT_CLASSES
    ]
    promoted = [e for e in reconstructed if isinstance(e, BranchPromoted)]
    frozen = [e for e in reconstructed if isinstance(e, FrozenPoolEviction)]
    killed = [e for e in reconstructed if isinstance(e, BranchTrueKilled)]
    assert [e.branch_id for e in promoted] == ["converging"]
    assert [e.branch_id for e in frozen] == ["underperformer"]
    assert [e.branch_id for e in killed] == ["diverged"]
    # Authority events are receipt-bound (never a grade).
    assert promoted[0].receipt_sha256 and promoted[0].decision_evidence_sha256
    assert killed[0].termination_cause == "training_diverged"
    # The promoted branch actually climbed to the next rung.
    assert promoted[0].from_rung == 0 and promoted[0].to_rung == 1


def test_missing_checkpoint_fails_closed(tmp_path):
    """A branch whose ``checkpoints/`` dir is empty yields no latest checkpoint and
    the receipt build path fails closed — never a fabricated receipt (mirrors
    ``reproduction_campaign.py``'s fail-closed dispatch guard)."""
    run_dir = tmp_path
    controller = SchedulerAuthorityController(
        run_dir, campaign_id="campaign-fc", spec=_spec(),
    )
    controller.bootstrap()
    dataset, run_spec = _write_run_spec_fingerprints(run_dir)
    ladder = controller.spec.ladder

    cell = _branch_cell_out(run_dir, "converging", metric=0.05, metric_id=ladder.metric_id)
    empty_ckpt = cell / "checkpoints"
    empty_ckpt.mkdir(parents=True, exist_ok=True)

    # No checkpoint written → latest_checkpoint_dir returns None (the fail-closed
    # dispatch guard's trigger).
    assert cell_checkpoint.latest_checkpoint_dir(empty_ckpt) is None

    # Building a receipt against a checkpoint dir with no components fails closed.
    with pytest.raises(ValueError, match="checkpoint component"):
        build_raw_receipt(
            run_dir=run_dir,
            cell_output_dir=cell,
            checkpoint_components_dir=empty_ckpt,
            ladder=ladder,
            campaign_id="campaign-fc",
            branch_id="converging",
            parent_branch_id=None,
            attempt_n=1,
            cell_id="cell-converging",
            from_step=0,
            to_step=ladder.rung_steps[0],
            seed=1,
            termination_cause=None,
            dataset_manifest_path=dataset,
            run_spec_path=run_spec,
        )


def test_true_kill_requires_literal_diverged_cause(tmp_path):
    """The receipt contract's evidence-not-grade + fail-closed kill rule: a branch
    that ASHA would only kill on a literal ``training_diverged`` cause is instead
    frozen when its cause is a generic (repairable) failure — underperformance is a
    finding, never a true kill."""
    run_dir = tmp_path
    events: list[tuple[str, dict]] = []
    controller = SchedulerAuthorityController(
        run_dir, campaign_id="campaign-kill", spec=_spec(),
        event_sink=lambda kind, payload: events.append((kind, dict(payload))),
    )
    controller.bootstrap()
    dataset, run_spec = _write_run_spec_fingerprints(run_dir)
    controller.claim_launches()

    _build_and_record(
        controller, run_dir, branch_id="converging", metric=0.05,
        steps=40, lr=0.05, seed=1, attempt_n=1, termination_cause=None,
        dataset=dataset, run_spec=run_spec,
    )
    _build_and_record(
        controller, run_dir, branch_id="underperformer", metric=0.90,
        steps=2, lr=0.001, seed=2, attempt_n=1, termination_cause=None,
        dataset=dataset, run_spec=run_spec,
    )
    # A non-diverged, generic-cause laggard: repairable, NOT a true kill.
    _build_and_record(
        controller, run_dir, branch_id="diverged", metric=0.70,
        steps=2, lr=0.001, seed=3, attempt_n=1, termination_cause="oom",
        dataset=dataset, run_spec=run_spec,
    )

    result = controller.decide_rung(
        rung=0,
        provider_gpu_usd_by_branch={
            "converging": 1.0, "underperformer": 1.0, "diverged": 1.0,
        },
    )
    actions = {item.branch_id: item.action for item in result.actions}
    # Only the true-diverged branch could be killed; a generic cause freezes.
    assert "kill" not in actions.values()
    assert actions["converging"] == "promote"
    kinds = [kind for kind, _ in events]
    assert "branch_true_killed" not in kinds


def test_diverged_branch_needs_finite_metric(tmp_path):
    """Guards against a lazy fixture: even a diverged (true-kill) branch must carry
    a finite measured metric in ``metrics.json`` — the receipt contract rejects a
    NaN/Inf metric, so divergence is signalled by ``termination_cause``, not by a
    non-finite value laundered through the metric artifact."""
    run_dir = tmp_path
    controller = SchedulerAuthorityController(
        run_dir, campaign_id="campaign-nan", spec=_spec(),
    )
    controller.bootstrap()
    dataset, run_spec = _write_run_spec_fingerprints(run_dir)
    ladder = controller.spec.ladder

    cell = _branch_cell_out(run_dir, "diverged", metric=0.0, metric_id=ladder.metric_id)
    # Overwrite with a non-finite metric to prove the contract rejects it.
    (cell / "metrics.json").write_text(
        json.dumps({ladder.metric_id: float("nan")}), encoding="utf-8"
    )
    ckpt_root = cell / "checkpoints"
    _train_and_checkpoint(ckpt_root, steps=2, lr=0.001, seed=3)
    components = cell_checkpoint.latest_checkpoint_dir(ckpt_root)

    with pytest.raises(ValueError, match="finite"):
        build_raw_receipt(
            run_dir=run_dir,
            cell_output_dir=cell,
            checkpoint_components_dir=components,
            ladder=ladder,
            campaign_id="campaign-nan",
            branch_id="diverged",
            parent_branch_id=None,
            attempt_n=1,
            cell_id="cell-diverged",
            from_step=0,
            to_step=ladder.rung_steps[0],
            seed=3,
            termination_cause="training_diverged",
            dataset_manifest_path=dataset,
            run_spec_path=run_spec,
        )
