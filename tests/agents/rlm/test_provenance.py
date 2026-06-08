"""Tests for backend.agents.rlm.provenance — Lane D2a (evidence legibility)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm.provenance import (
    RubricGuardFailure,
    assert_provenance,
    emit_figure_sidecar,
    emit_provenance,
)


# --------------------------------------------------------------------------- #
# emit_provenance
# --------------------------------------------------------------------------- #
def test_emit_provenance_writes_file(tmp_path: Path) -> None:
    """emit_provenance writes provenance.json with the expected envelope."""
    out = emit_provenance(
        tmp_path,
        experiments={
            "exp0": {
                "model_key": "qwen3_1.7b",
                "env": "alfworld",
                "baseline": "grpo",
                "seed": 0,
                "epochs": 3,
                "steps": 120,
                "batch_size": 128,
                "per_optimizer": {"adam": {"lr": 1e-3}},
                "hardware": "RTX 4090",
                "framework_versions": {"torch": "2.1.0"},
                "convergence": {"iteration": [0, 1, 2], "loss": [3.0, 2.0, 1.0]},
            }
        },
        run_id="run-abc",
    )
    assert out == tmp_path / "provenance.json"
    assert out.is_file()

    payload = json.loads(out.read_text())
    assert payload["schema_version"] == 1
    assert payload["run_id"] == "run-abc"
    # generated_at defaults to None (deterministic — no implicit clock read).
    assert payload["generated_at"] is None
    exp = payload["experiments"]["exp0"]
    assert exp["epochs"] == 3
    assert exp["batch_size"] == 128
    # Short convergence series is preserved verbatim (<= 32 entries).
    assert exp["convergence"]["loss"] == [3.0, 2.0, 1.0]


def test_emit_provenance_summarizes_long_series(tmp_path: Path) -> None:
    """A >32-entry convergence axis is stored as a SUMMARY dict, not the array."""
    loss = [float(i) for i in range(100)]  # 100 > 32 → must be summarized
    out = emit_provenance(
        tmp_path,
        experiments={
            "exp0": {
                "model_key": "qwen2.5_3b",
                "convergence": {"iteration": list(range(100)), "loss": loss},
            }
        },
    )
    payload = json.loads(out.read_text())
    stored = payload["experiments"]["exp0"]["convergence"]["loss"]

    # The stored form is the summary dict, NOT the raw 100-element array.
    assert isinstance(stored, dict)
    assert not isinstance(stored, list)
    assert stored["len"] == 100
    assert stored["first"] == 0.0
    assert stored["last"] == 99.0
    assert stored["min"] == 0.0
    assert stored["max"] == 99.0
    # Downsampled to <= 20 evenly-spaced points, endpoints retained.
    assert 0 < len(stored["sampled"]) <= 20
    assert stored["sampled"][0] == 0.0
    assert stored["sampled"][-1] == 99.0


def test_emit_provenance_fail_soft_on_bad_input(tmp_path: Path) -> None:
    """A non-dict experiments value never raises — fail-soft returns the path."""
    out = emit_provenance(tmp_path, experiments=None)  # type: ignore[arg-type]
    assert out == tmp_path / "provenance.json"
    # File still written with an empty experiments map.
    payload = json.loads(out.read_text())
    assert payload["experiments"] == {}


# --------------------------------------------------------------------------- #
# emit_figure_sidecar
# --------------------------------------------------------------------------- #
def test_emit_figure_sidecar_writes_stem_json(tmp_path: Path) -> None:
    """emit_figure_sidecar writes <png_stem>.json next to the PNG."""
    png = tmp_path / "fig_loss.png"
    out = emit_figure_sidecar(
        png,
        shows="training loss vs iteration",
        axis={"x": {"label": "iter", "scale": "linear"},
              "y": {"label": "loss", "scale": "log"}},
        series={"loss": [3.0, 2.0, 1.0]},
    )
    assert out == tmp_path / "fig_loss.json"
    assert out.is_file()

    payload = json.loads(out.read_text())
    assert payload["shows"] == "training loss vs iteration"
    # The log-scale axis is exactly the "log-scale axis not verifiable" answer.
    assert payload["axis"]["y"]["scale"] == "log"
    assert payload["series"]["loss"] == [3.0, 2.0, 1.0]


def test_emit_figure_sidecar_summarizes_long_series(tmp_path: Path) -> None:
    """A long plotted series in the sidecar is summarized too."""
    png = tmp_path / "fig_curve.png"
    out = emit_figure_sidecar(
        png,
        shows="curve",
        axis={"x": {"label": "x", "scale": "linear"},
              "y": {"label": "y", "scale": "linear"}},
        series={"y": list(range(50))},  # 50 > 32 → summary
    )
    stored = json.loads(out.read_text())["series"]["y"]
    assert isinstance(stored, dict)
    assert stored["len"] == 50


# --------------------------------------------------------------------------- #
# assert_provenance
# --------------------------------------------------------------------------- #
def test_assert_provenance_raises_on_missing_manifest(tmp_path: Path) -> None:
    """Absent provenance.json raises RubricGuardFailure with a JSON message."""
    with pytest.raises(RubricGuardFailure) as excinfo:
        assert_provenance(tmp_path)
    detail = json.loads(str(excinfo.value))
    assert detail["provenance_guard"] == "manifest_missing"


def test_assert_provenance_noop_when_manifest_present(tmp_path: Path) -> None:
    """require_series=False + manifest present (no figures) → no-op."""
    emit_provenance(
        tmp_path,
        experiments={"exp0": {"model_key": "m", "convergence": {}}},
    )
    # Should not raise.
    assert_provenance(tmp_path, require_series=False)


def test_assert_provenance_raises_when_series_required_but_absent(tmp_path: Path) -> None:
    """require_series=True with no non-empty convergence series raises."""
    emit_provenance(
        tmp_path,
        experiments={"exp0": {"model_key": "m", "convergence": {}}},
    )
    with pytest.raises(RubricGuardFailure) as excinfo:
        assert_provenance(tmp_path, require_series=True)
    detail = json.loads(str(excinfo.value))
    assert detail["provenance_guard"] == "series_missing"


def test_assert_provenance_passes_when_series_required_and_present(tmp_path: Path) -> None:
    """require_series=True is satisfied by a non-empty convergence series."""
    emit_provenance(
        tmp_path,
        experiments={
            "exp0": {
                "model_key": "m",
                "convergence": {"iteration": [0, 1, 2], "loss": [3.0, 2.0, 1.0]},
            }
        },
    )
    # Should not raise — exp0 carries a non-empty series.
    assert_provenance(tmp_path, require_series=True)


def test_assert_provenance_series_satisfied_by_summarized_long_series(tmp_path: Path) -> None:
    """A long (summarized) series still satisfies require_series=True."""
    emit_provenance(
        tmp_path,
        experiments={
            "exp0": {
                "model_key": "m",
                "convergence": {"loss": [float(i) for i in range(100)]},
            }
        },
    )
    # The stored series is a summary dict; require_series must still pass.
    assert_provenance(tmp_path, require_series=True)


def test_assert_provenance_raises_on_figure_without_sidecar(tmp_path: Path) -> None:
    """A fig_*.png with no <stem>.json sidecar raises RubricGuardFailure."""
    emit_provenance(tmp_path, experiments={"exp0": {"model_key": "m"}})
    # A PNG with no accompanying sidecar.
    (tmp_path / "fig_orphan.png").write_bytes(b"\x89PNG\r\n")
    with pytest.raises(RubricGuardFailure) as excinfo:
        assert_provenance(tmp_path)
    detail = json.loads(str(excinfo.value))
    assert detail["provenance_guard"] == "figure_sidecar_missing"
    assert "fig_orphan.png" in detail["missing_sidecars"]


def test_assert_provenance_passes_when_figure_has_sidecar(tmp_path: Path) -> None:
    """A fig_*.png WITH its sidecar does not raise."""
    emit_provenance(tmp_path, experiments={"exp0": {"model_key": "m"}})
    png = tmp_path / "fig_ok.png"
    png.write_bytes(b"\x89PNG\r\n")
    emit_figure_sidecar(
        png,
        shows="ok",
        axis={"x": {"label": "x", "scale": "linear"},
              "y": {"label": "y", "scale": "linear"}},
        series={"y": [1, 2, 3]},
    )
    # fig_ok.png now has fig_ok.json next to it.
    assert_provenance(tmp_path)
