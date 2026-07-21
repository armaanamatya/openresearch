"""Eval-coverage floor — OPENRESEARCH_MIN_EVAL_N (a sub-knob of the eval-provenance guard).

The eval-provenance guard verifies the reported metric IS the mean of real [0,1]
per-example outcomes, and that eval/train are disjoint. It does NOT check that
*enough* examples were evaluated — a correct mean over 3 held-out examples still
passes. This closes that gap: when the guard is enabled AND OPENRESEARCH_MIN_EVAL_N
is set, a success cell whose ``n_eval`` is below the floor is vetoed as
``fabrication_suspected``.

Sub-knob of OPENRESEARCH_EVAL_PROVENANCE_GUARD (coverage needs the sidecar infra
that guard enforces). Default MIN_EVAL_N=0 ⇒ byte-identical to the guard today.
Hermetic — no network, no GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.agents.rlm.eval_provenance import eval_provenance_should_veto


def _make_cell(base: Path, metrics: dict, provenance: dict) -> None:
    cell = base / "outputs" / "run_01" / "cell_0"
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (cell / "eval_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")


def _records(n_ones: int, n_zeros: int) -> list[dict]:
    return (
        [{"id": f"one_{i}", "outcome": 1.0} for i in range(n_ones)]
        + [{"id": f"zero_{i}", "outcome": 0.0} for i in range(n_zeros)]
    )


# A mean-correct cell: reported accuracy == mean(outcomes); only the coverage
# floor could veto it (all other guard checks pass).
def _mean_correct_cell(base: Path, *, n_ones: int, n_zeros: int) -> None:
    n = n_ones + n_zeros
    mean = n_ones / n if n else 0.0
    _make_cell(
        base,
        metrics={"status": "ok", "accuracy": mean},
        provenance={
            "records": _records(n_ones, n_zeros),
            "metric_value": mean,
            "n_eval": n,
            "held_out": True,
        },
    )


def test_floor_off_does_not_veto_tiny_eval(tmp_path: Path, monkeypatch):
    """Guard ON, MIN_EVAL_N unset (0): a mean-correct 2-example cell passes —
    byte-identical to the eval-provenance guard today."""
    monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
    monkeypatch.delenv("OPENRESEARCH_MIN_EVAL_N", raising=False)
    _mean_correct_cell(tmp_path, n_ones=1, n_zeros=1)
    assert eval_provenance_should_veto(tmp_path) == (False, None)


def test_floor_vetoes_insufficient_coverage(tmp_path: Path, monkeypatch):
    """Guard ON, MIN_EVAL_N=100: a mean-correct 2-example cell is vetoed — the
    result is not substantiated by enough held-out examples."""
    monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
    monkeypatch.setenv("OPENRESEARCH_MIN_EVAL_N", "100")
    _mean_correct_cell(tmp_path, n_ones=1, n_zeros=1)
    veto, detail = eval_provenance_should_veto(tmp_path)
    assert veto is True
    assert detail is not None and "MIN_EVAL_N" in detail


def test_floor_passes_sufficient_coverage(tmp_path: Path, monkeypatch):
    """Guard ON, MIN_EVAL_N=100: a mean-correct 100-example cell passes."""
    monkeypatch.setenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", "1")
    monkeypatch.setenv("OPENRESEARCH_MIN_EVAL_N", "100")
    _mean_correct_cell(tmp_path, n_ones=50, n_zeros=50)
    assert eval_provenance_should_veto(tmp_path) == (False, None)


def test_floor_inert_when_master_guard_off(tmp_path: Path, monkeypatch):
    """MIN_EVAL_N set but the master guard OFF: coverage never runs — the guard's
    default-OFF contract (False, None) is preserved (byte-identical)."""
    monkeypatch.delenv("OPENRESEARCH_EVAL_PROVENANCE_GUARD", raising=False)
    monkeypatch.setenv("OPENRESEARCH_MIN_EVAL_N", "100")
    _mean_correct_cell(tmp_path, n_ones=1, n_zeros=1)
    assert eval_provenance_should_veto(tmp_path) == (False, None)
