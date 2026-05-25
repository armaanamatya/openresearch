"""Tests for calibration.py — reading preserved runs and computing priors.

Spec: docs/superpowers/specs/2026-05-25-budget-estimation-design.md §calibration.py
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


def _make_preserved_run(
    runs_root: Path,
    run_id: str,
    primitives: list[dict],
) -> None:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / ".preserved").write_text(
        json.dumps({"verdict": "partial", "schema_version": 1}),
        encoding="utf-8",
    )
    lines = "\n".join(json.dumps(p) for p in primitives)
    (run_dir / "cost_ledger.jsonl").write_text(lines, encoding="utf-8")


def test_recompute_calibration_averages(tmp_path: Path, monkeypatch):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    _make_preserved_run(runs_root, "run1", [
        {"primitive": "understand_section", "input_tokens": 10000, "output_tokens": 800},
        {"primitive": "implement_baseline", "input_tokens": 30000, "output_tokens": 8000},
    ])
    _make_preserved_run(runs_root, "run2", [
        {"primitive": "understand_section", "input_tokens": 14000, "output_tokens": 1200},
    ])

    # Redirect calibration.json to tmp_path
    cal_path = tmp_path / "calibration.json"
    import backend.services.pricing.calibration as cal_mod
    monkeypatch.setattr(cal_mod, "_calibration_path", lambda: cal_path)

    result = recompute_calibration(runs_root)
    assert result["based_on_n_preserved_runs"] == 2
    assert result["schema_version"] == CALIBRATION_SCHEMA_VERSION

    per_prim = result["per_primitive"]
    assert "understand_section" in per_prim
    avg_in = per_prim["understand_section"]["avg_input_tokens"]
    assert avg_in == pytest.approx(12000.0)
    avg_out = per_prim["understand_section"]["avg_output_tokens"]
    assert avg_out == pytest.approx(1000.0)

    assert "implement_baseline" in per_prim
    assert per_prim["implement_baseline"]["avg_input_tokens"] == pytest.approx(30000.0)


def test_recompute_calibration_skips_unpreserved(tmp_path: Path, monkeypatch):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    run_dir = runs_root / "run_no_marker"
    run_dir.mkdir()
    (run_dir / "cost_ledger.jsonl").write_text(
        json.dumps({"primitive": "understand_section", "input_tokens": 5000, "output_tokens": 300}),
        encoding="utf-8",
    )

    cal_path = tmp_path / "calibration.json"
    import backend.services.pricing.calibration as cal_mod
    monkeypatch.setattr(cal_mod, "_calibration_path", lambda: cal_path)

    result = recompute_calibration(runs_root)
    assert result["based_on_n_preserved_runs"] == 0
    assert result["per_primitive"] == {}


def test_recompute_calibration_atomic_write(tmp_path: Path, monkeypatch):
    """Ensures .tmp file is used (os.replace path)."""
    import os
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _make_preserved_run(runs_root, "run1", [
        {"primitive": "run_experiment", "input_tokens": 4000, "output_tokens": 150},
    ])

    cal_path = tmp_path / "calibration.json"
    import backend.services.pricing.calibration as cal_mod
    monkeypatch.setattr(cal_mod, "_calibration_path", lambda: cal_path)

    replaces = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: (replaces.append(str(src)), real_replace(src, dst))[1])

    recompute_calibration(runs_root)
    assert any(".tmp" in r for r in replaces), "expected .tmp in os.replace"


def test_get_primitive_priors_falls_back_to_defaults_when_no_calibration(tmp_path, monkeypatch):
    cal_path = tmp_path / "calibration.json"
    import backend.services.pricing.calibration as cal_mod
    monkeypatch.setattr(cal_mod, "_calibration_path", lambda: cal_path)

    priors = get_primitive_priors("nlp_seq", "strict")
    for primitive in _DEFAULT_PRIORS:
        assert primitive in priors
        assert priors[primitive]["avg_input_tokens"] == _DEFAULT_PRIORS[primitive]["avg_input_tokens"]


def test_get_primitive_priors_uses_calibration_when_enough_runs(tmp_path, monkeypatch):
    cal_path = tmp_path / "calibration.json"
    calibration_data = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "based_on_n_preserved_runs": _MIN_RUNS_FOR_CATEGORY,
        "per_primitive": {
            "understand_section": {
                "avg_input_tokens": 99999.0,
                "avg_output_tokens": 5555.0,
                "n_samples": 10,
            }
        },
    }
    cal_path.write_text(json.dumps(calibration_data), encoding="utf-8")

    import backend.services.pricing.calibration as cal_mod
    monkeypatch.setattr(cal_mod, "_calibration_path", lambda: cal_path)

    priors = get_primitive_priors("nlp_seq", "strict")
    assert priors["understand_section"]["avg_input_tokens"] == pytest.approx(99999.0)
    # Other primitives not in calibration should still fall back to defaults
    assert "implement_baseline" in priors
    assert priors["implement_baseline"]["avg_input_tokens"] == pytest.approx(
        _DEFAULT_PRIORS["implement_baseline"]["avg_input_tokens"]
    )


def test_get_primitive_priors_falls_back_when_too_few_runs(tmp_path, monkeypatch):
    cal_path = tmp_path / "calibration.json"
    calibration_data = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "based_on_n_preserved_runs": _MIN_RUNS_FOR_CATEGORY - 1,
        "per_primitive": {
            "understand_section": {"avg_input_tokens": 99999.0, "avg_output_tokens": 5555.0}
        },
    }
    cal_path.write_text(json.dumps(calibration_data), encoding="utf-8")

    import backend.services.pricing.calibration as cal_mod
    monkeypatch.setattr(cal_mod, "_calibration_path", lambda: cal_path)

    priors = get_primitive_priors("nlp_seq", "strict")
    # Should fall back because n_runs < _MIN_RUNS_FOR_CATEGORY
    assert priors["understand_section"]["avg_input_tokens"] == pytest.approx(
        _DEFAULT_PRIORS["understand_section"]["avg_input_tokens"]
    )
