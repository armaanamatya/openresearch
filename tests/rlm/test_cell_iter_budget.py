"""Scheduler-authority rung-step -> trainer iteration-budget wire.

An authority-campaign branch launch carries a ``from_step``/``to_step`` off the
paper-step ladder, but the real trainer ignores it and runs its own
``cells.json`` ``iters`` -- so successive-halving-at-rungs never actually caps
the training the child does. This module tests the two deterministic,
env-gated, byte-identical-when-off pieces that close the gap:

* :func:`primitives._apply_cell_iter_budget` -- the harness-owned pre-dispatch
  cell-dict rewrite in ``_execute_cell_matrix``: with a positive budget every
  cell's ``iters`` is set to the budget and any ``lr_milestones`` /
  ``warmup_max_iters`` beyond it are clamped, so the LR schedule stays coherent;
  with ``None`` (env unset/invalid) the cells are returned byte-identical.
* :func:`primitives._cell_iter_budget` -- the env reader: a positive int or
  ``None`` (fail-SAFE on ``""``/``0``/``-5``/``abc``).

HARD CONSTRAINT under test: the OFF path (env unset / malformed) is
byte-for-byte identical to today -- the system flag invariant.
"""

from __future__ import annotations

import copy

from backend.agents.rlm.primitives import (
    _apply_cell_iter_budget,
    _cell_iter_budget,
)


def _cells() -> list[dict]:
    # A minimal but representative cells fixture: two cells, one carrying the
    # LR-schedule fields the ResNet trainer keys off ``iters`` (milestones +
    # warmup boundary), one bare.
    return [
        {
            "id": "c1",
            "model_key": "resnet-20",
            "env": "cifar10",
            "baseline": "sgd",
            "iters": 64000,
            "lr_milestones": [32000, 48000],
            "warmup_max_iters": 400,
        },
        {
            "id": "c2",
            "model_key": "resnet-56",
            "env": "cifar10",
            "baseline": "sgd",
            "iters": 64000,
        },
    ]


# --------------------------------------------------------------------------- #
# (a) OFF -- budget None => byte-identical                                     #
# --------------------------------------------------------------------------- #
def test_apply_budget_none_is_byte_identical():
    cells = _cells()
    before = copy.deepcopy(cells)
    out = _apply_cell_iter_budget(cells, None)
    assert out == before, "budget=None must not rewrite any cell"


def test_env_reader_unset_returns_none(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_CELL_ITER_BUDGET", raising=False)
    assert _cell_iter_budget() is None


# --------------------------------------------------------------------------- #
# (b) ON -- budget=250 => every cell iters==250, milestones clamped <= 250     #
# --------------------------------------------------------------------------- #
def test_apply_budget_caps_iters_and_clamps_milestones():
    cells = _cells()
    out = _apply_cell_iter_budget(cells, 250)
    for cell in out:
        assert cell["iters"] == 250
    # c1's milestones (32000, 48000) were both > 250 => clamped to <= 250.
    c1 = next(c for c in out if c["id"] == "c1")
    assert all(m <= 250 for m in c1["lr_milestones"])
    # The warmup boundary keyed off iters is clamped too.
    assert c1["warmup_max_iters"] <= 250


def test_apply_budget_preserves_in_range_milestones():
    # A milestone already <= budget is left untouched (only the > budget ones
    # are clamped) -- the schedule shape below the cap is preserved.
    cells = [{"id": "c", "iters": 1000, "lr_milestones": [100, 900, 1500]}]
    out = _apply_cell_iter_budget(cells, 1000)
    (cell,) = out
    assert cell["iters"] == 1000
    assert cell["lr_milestones"] == [100, 900, 1000]


def test_env_reader_positive_int(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_CELL_ITER_BUDGET", "250")
    assert _cell_iter_budget() == 250


# --------------------------------------------------------------------------- #
# (c) malformed budget => no rewrite (byte-identical)                          #
# --------------------------------------------------------------------------- #
def test_apply_malformed_budget_is_byte_identical():
    for bad in ("", "0", "-5", "abc", 0, -5, None):
        cells = _cells()
        before = copy.deepcopy(cells)
        out = _apply_cell_iter_budget(cells, bad)  # type: ignore[arg-type]
        assert out == before, f"malformed budget {bad!r} must not rewrite any cell"


def test_env_reader_malformed_returns_none(monkeypatch):
    for bad in ("", "0", "-5", "abc", "  ", "2.5"):
        monkeypatch.setenv("OPENRESEARCH_CELL_ITER_BUDGET", bad)
        assert _cell_iter_budget() is None, f"malformed env {bad!r} must read as None"
