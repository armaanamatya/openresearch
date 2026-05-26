"""Tests for calibration.py zero-token-skip behavior (PR-ε.7).

Spec: docs/superpowers/specs/2026-05-25-three-source-budget-estimator-design.md §calibration
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services.pricing.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    _DEFAULT_PRIORS,
    _MIN_RUNS_FOR_CATEGORY,
    get_primitive_priors,
    recompute_calibration,
)


def _make_run_with_ledger(runs_root: Path, run_id: str, primitives: list[dict]) -> None:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / ".preserved").write_text(
        json.dumps({"verdict": "partial", "schema_version": 1}),
        encoding="utf-8",
    )
    lines = "\n".join(json.dumps(p) for p in primitives)
    (run_dir / "cost_ledger.jsonl").write_text(lines, encoding="utf-8")


# ---------------------------------------------------------------------------
# recompute_calibration: zero rows are skipped
# ---------------------------------------------------------------------------

def test_zero_token_rows_skipped(tmp_path, monkeypatch):
    """Rows with input_tokens=0 AND output_tokens=0 must be excluded."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    _make_run_with_ledger(runs_root, "run_oauth", [
        # OAuth runs produce zero-token rows
        {"primitive": "understand_section", "input_tokens": 0, "output_tokens": 0},
        {"primitive": "implement_baseline", "input_tokens": 0, "output_tokens": 0},
    ])
    _make_run_with_ledger(runs_root, "run_real", [
        {"primitive": "understand_section", "input_tokens": 12000, "output_tokens": 800},
    ])

    cal_path = tmp_path / "calibration.json"
    import backend.services.pricing.calibration as cal_mod
    monkeypatch.setattr(cal_mod, "_calibration_path", lambda: cal_path)

    result = recompute_calibration(runs_root)

    per_prim = result["per_primitive"]
    assert "understand_section" in per_prim
    # Average should be from the real run only (12000), not poisoned by zero
    assert per_prim["understand_section"]["avg_input_tokens"] == pytest.approx(12000.0)
    assert per_prim["understand_section"]["avg_output_tokens"] == pytest.approx(800.0)
    # implement_baseline should NOT be in the calibration (all its rows were zero)
    assert "implement_baseline" not in per_prim


def test_mixed_zero_and_real_rows_averages_only_real(tmp_path, monkeypatch):
    """Mix of zero and non-zero rows: average over non-zero only."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    _make_run_with_ledger(runs_root, "run_mixed", [
        {"primitive": "run_experiment", "input_tokens": 0, "output_tokens": 0},
        {"primitive": "run_experiment", "input_tokens": 5000, "output_tokens": 200},
        {"primitive": "run_experiment", "input_tokens": 7000, "output_tokens": 300},
    ])

    cal_path = tmp_path / "calibration.json"
    import backend.services.pricing.calibration as cal_mod
    monkeypatch.setattr(cal_mod, "_calibration_path", lambda: cal_path)

    result = recompute_calibration(runs_root)
    per_prim = result["per_primitive"]
    assert "run_experiment" in per_prim
    # Average of (5000, 7000) = 6000; NOT (0 + 5000 + 7000) / 3 = 4000
    assert per_prim["run_experiment"]["avg_input_tokens"] == pytest.approx(6000.0)
    assert per_prim["run_experiment"]["avg_output_tokens"] == pytest.approx(250.0)
    assert per_prim["run_experiment"]["n_samples"] == 2


def test_all_zero_rows_produces_no_calibration_entry(tmp_path, monkeypatch):
    """If all rows for a primitive are zero, it should not appear in calibration."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    _make_run_with_ledger(runs_root, "run_all_oauth", [
        {"primitive": "verify_against_rubric", "input_tokens": 0, "output_tokens": 0},
        {"primitive": "verify_against_rubric", "input_tokens": 0, "output_tokens": 0},
    ])

    cal_path = tmp_path / "calibration.json"
    import backend.services.pricing.calibration as cal_mod
    monkeypatch.setattr(cal_mod, "_calibration_path", lambda: cal_path)

    result = recompute_calibration(runs_root)
    assert "verify_against_rubric" not in result["per_primitive"]


def test_schema_version_is_2(tmp_path, monkeypatch):
    """After PR-ε.7, CALIBRATION_SCHEMA_VERSION must be 2."""
    assert CALIBRATION_SCHEMA_VERSION == 2


# ---------------------------------------------------------------------------
# get_primitive_priors: zero-valued calibration entries fall back to defaults
# ---------------------------------------------------------------------------

def test_get_primitive_priors_zero_entry_falls_back(tmp_path, monkeypatch):
    """A zero-valued calibration entry should be treated as missing."""
    cal_path = tmp_path / "calibration.json"
    # Simulate a v1 calibration.json that was poisoned by OAuth zero rows
    calibration_data = {
        "schema_version": 1,
        "based_on_n_preserved_runs": _MIN_RUNS_FOR_CATEGORY,
        "per_primitive": {
            "understand_section": {
                "avg_input_tokens": 0.0,   # poisoned by OAuth
                "avg_output_tokens": 0.0,
                "n_samples": 3,
            }
        },
    }
    cal_path.write_text(json.dumps(calibration_data), encoding="utf-8")

    import backend.services.pricing.calibration as cal_mod
    monkeypatch.setattr(cal_mod, "_calibration_path", lambda: cal_path)

    priors = get_primitive_priors("nlp_seq", "strict")
    # Should fall back to the static default, not return 0.0
    assert priors["understand_section"]["avg_input_tokens"] == pytest.approx(
        _DEFAULT_PRIORS["understand_section"]["avg_input_tokens"]
    )
    assert priors["understand_section"]["avg_output_tokens"] == pytest.approx(
        _DEFAULT_PRIORS["understand_section"]["avg_output_tokens"]
    )


def test_get_primitive_priors_real_entry_used(tmp_path, monkeypatch):
    """A non-zero calibration entry should be used, not fallen back."""
    cal_path = tmp_path / "calibration.json"
    calibration_data = {
        "schema_version": 2,
        "based_on_n_preserved_runs": _MIN_RUNS_FOR_CATEGORY,
        "per_primitive": {
            "understand_section": {
                "avg_input_tokens": 15000.0,
                "avg_output_tokens": 900.0,
                "n_samples": 8,
            }
        },
    }
    cal_path.write_text(json.dumps(calibration_data), encoding="utf-8")

    import backend.services.pricing.calibration as cal_mod
    monkeypatch.setattr(cal_mod, "_calibration_path", lambda: cal_path)

    priors = get_primitive_priors("nlp_seq", "strict")
    assert priors["understand_section"]["avg_input_tokens"] == pytest.approx(15000.0)
    assert priors["understand_section"]["avg_output_tokens"] == pytest.approx(900.0)
