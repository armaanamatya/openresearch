"""Hermetic Tree-B end-to-end: a REAL tiny CPU trainer's 5-field checkpoint drives
the REAL ``SchedulerAuthorityController`` through a real promote + freeze + true-kill.

Task 1 (this commit): the tiny real-trainer checkpoint fixture. Every existing
scheduler test drives the controller with SYNTHETIC checkpoint bytes; these helpers
produce a genuine 5-component checkpoint from an actual ``torch`` model so the later
E2E test can exercise the producer->receipt->authority seam against real on-disk
output. torch is a test-only import (guarded).
"""
from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import pytest

from backend.agents.rlm import cell_checkpoint


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


def test_tiny_trainer_writes_real_five_field_checkpoint(tmp_path):
    """The Task-1 fixture produces a genuine 5-component checkpoint dir the receipt
    producer can read (proves the helper, before the full E2E lands)."""
    cell = _branch_cell_out(tmp_path, "b0", metric=0.1, metric_id="train_loss")
    ckpt_root = cell / "checkpoints"
    loss = _train_and_checkpoint(ckpt_root, steps=10, lr=0.05, seed=1)
    assert loss == loss  # finite, not NaN

    latest = cell_checkpoint.latest_checkpoint_dir(ckpt_root)
    assert latest is not None and latest.name == "step_10"
    assert sorted(p.name for p in latest.iterdir()) == [
        "data_order", "lr_scheduler", "model", "optimizer", "rng",
    ]
    # The model component is real torch bytes (reloads to a state_dict).
    torch = pytest.importorskip("torch")
    reloaded = torch.load(io.BytesIO((latest / "model").read_bytes()), weights_only=False)
    assert "0.weight" in reloaded
