"""Tests for figure_sidecars.py — harness-owned training-curve + figure-sidecar emitter.

Coverage:
  * emit_training_curves — nested dict, list-of-dicts, flat keys, mixed/multi,
    no-data (no file), multi-model, partial curves, only-step guard
  * emit_figure_sidecars_harness — comparison sidecar shape + grader glob,
    agent-sidecar guard, no-data skip, curve downsampling, log-axis hint
  * emit_sidecars — flag gate (OFF = no files, ON = emits), full integration,
    no per_model = no-op
  * Byte-identical guarantee: unset flag ⇒ nothing written
  * Never-raise: corrupted input never propagates
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backend.agents.rlm import figure_sidecars as fs


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_flag(monkeypatch):
    """Ensure flag is unset (OFF) by default for every test."""
    monkeypatch.delenv(fs.FLAG, raising=False)
    yield


def _make_project(tmp_path: Path, *, metrics: dict | None = None) -> Path:
    """Create a minimal project dir with optional metrics.json."""
    code = tmp_path / "code"
    code.mkdir(parents=True, exist_ok=True)
    if metrics is not None:
        (code / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return tmp_path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# is_enabled — flag contract
# ---------------------------------------------------------------------------

class TestIsEnabled:
    def test_default_off(self):
        assert fs.is_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "on", "yes", "TRUE", "ON"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv(fs.FLAG, val)
        assert fs.is_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "off", "no", "", "  "])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv(fs.FLAG, val)
        assert fs.is_enabled() is False


# ---------------------------------------------------------------------------
# _extract_curves_from_leaf — unit tests on the pure helper
# ---------------------------------------------------------------------------

class TestExtractCurvesFromLeaf:
    def test_nested_training_curves_dict(self):
        leaf = {"training_curves": {"reward": [0.1, 0.2, 0.3], "loss": [2.5, 2.0, 1.5]}}
        curves = fs._extract_curves_from_leaf(leaf)
        assert curves["reward"] == [0.1, 0.2, 0.3]
        assert curves["loss"] == [2.5, 2.0, 1.5]

    def test_nested_training_curves_dict_with_step(self):
        leaf = {"training_curves": {"step": [0, 10, 20], "reward": [0.1, 0.2, 0.3]}}
        curves = fs._extract_curves_from_leaf(leaf)
        assert curves["step"] == [0.0, 10.0, 20.0]
        assert curves["reward"] == [0.1, 0.2, 0.3]

    def test_training_curves_rewards_plural(self):
        leaf = {"training_curves": {"rewards": [0.4, 0.5]}}
        curves = fs._extract_curves_from_leaf(leaf)
        assert curves["reward"] == [0.4, 0.5]

    def test_training_curves_mean_reward(self):
        leaf = {"training_curves": {"mean_reward": [0.1, 0.15, 0.2]}}
        curves = fs._extract_curves_from_leaf(leaf)
        assert curves["reward"] == [0.1, 0.15, 0.2]

    def test_list_of_dicts(self):
        leaf = {
            "training_curves": [
                {"step": 0, "reward": 0.1, "loss": 2.5},
                {"step": 10, "reward": 0.2, "loss": 2.0},
            ]
        }
        curves = fs._extract_curves_from_leaf(leaf)
        assert curves["reward"] == [0.1, 0.2]
        assert curves["loss"] == [2.5, 2.0]
        assert curves["step"] == [0.0, 10.0]

    def test_flat_reward_history(self):
        leaf = {"status": "ok", "metric": 0.85, "reward_history": [0.1, 0.2, 0.5]}
        curves = fs._extract_curves_from_leaf(leaf)
        assert curves["reward"] == [0.1, 0.2, 0.5]

    def test_flat_loss_history(self):
        leaf = {"loss_history": [3.0, 2.5, 2.0]}
        curves = fs._extract_curves_from_leaf(leaf)
        assert curves["loss"] == [3.0, 2.5, 2.0]

    def test_no_curves_returns_empty(self):
        leaf = {"status": "ok", "metric": 0.75, "steps_run": 1000}
        assert fs._extract_curves_from_leaf(leaf) == {}

    def test_failed_leaf_no_curves(self):
        leaf = {"status": "failed", "metric": None, "error": "oom"}
        assert fs._extract_curves_from_leaf(leaf) == {}

    def test_single_step_value_not_extracted_as_step_curve(self):
        # A single step index is not a meaningful curve — must have ≥2 values
        leaf = {"step": [0]}
        curves = fs._extract_curves_from_leaf(leaf)
        assert "step" not in curves

    def test_non_numeric_values_in_list_skipped(self):
        leaf = {"training_curves": {"reward": [0.1, "bad", 0.3]}}
        curves = fs._extract_curves_from_leaf(leaf)
        # "bad" is dropped; result has the two valid floats
        assert curves["reward"] == [0.1, 0.3]

    def test_bool_in_list_skipped(self):
        # bool ⊂ int in Python, but we exclude booleans from numeric series
        leaf = {"training_curves": {"reward": [True, 0.1, False, 0.2]}}
        curves = fs._extract_curves_from_leaf(leaf)
        assert curves["reward"] == [0.1, 0.2]

    def test_non_dict_leaf_returns_empty(self):
        assert fs._extract_curves_from_leaf("not a dict") == {}  # type: ignore[arg-type]
        assert fs._extract_curves_from_leaf(None) == {}  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# emit_training_curves
# ---------------------------------------------------------------------------

class TestEmitTrainingCurves:
    def test_writes_file_when_curves_present(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = {
            "qwen3_1_7b": {
                "alfworld": {
                    "sdar": {"reward_history": [0.1, 0.2, 0.3], "metric": 0.3},
                    "grpo": {"reward_history": [0.05, 0.1, 0.15]},
                }
            }
        }
        result = fs.emit_training_curves(code_dir, per_model)
        assert result is True
        tc = _read_json(code_dir / "training_curves.json")
        assert "qwen3_1_7b" in tc
        assert tc["qwen3_1_7b"]["alfworld"]["sdar"]["reward"] == [0.1, 0.2, 0.3]
        assert tc["qwen3_1_7b"]["alfworld"]["grpo"]["reward"] == [0.05, 0.1, 0.15]

    def test_nested_training_curves_dict(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = {
            "m": {"e": {"b": {"training_curves": {"reward": [0.2, 0.4], "loss": [2.0, 1.5]}}}}
        }
        result = fs.emit_training_curves(code_dir, per_model)
        assert result is True
        tc = _read_json(code_dir / "training_curves.json")
        assert tc["m"]["e"]["b"]["reward"] == [0.2, 0.4]
        assert tc["m"]["e"]["b"]["loss"] == [2.0, 1.5]

    def test_no_file_when_no_curves(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = {
            "m": {"e": {"b": {"status": "ok", "metric": 0.85, "steps_run": 500}}}
        }
        result = fs.emit_training_curves(code_dir, per_model)
        assert result is False
        assert not (code_dir / "training_curves.json").exists()

    def test_multi_model_multi_env(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = {
            "m1": {
                "alfworld": {"sdar": {"reward_history": [0.1, 0.2]}},
                "searchqa": {"sdar": {"reward_history": [0.3, 0.4]}},
            },
            "m2": {
                "alfworld": {"grpo": {"loss_history": [2.0, 1.5]}},
            },
        }
        result = fs.emit_training_curves(code_dir, per_model)
        assert result is True
        tc = _read_json(code_dir / "training_curves.json")
        assert tc["m1"]["alfworld"]["sdar"]["reward"] == [0.1, 0.2]
        assert tc["m1"]["searchqa"]["sdar"]["reward"] == [0.3, 0.4]
        assert tc["m2"]["alfworld"]["grpo"]["loss"] == [2.0, 1.5]

    def test_mixed_cells_only_curves_included(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = {
            "m": {
                "e": {
                    "with_curves": {"reward_history": [0.1, 0.2]},
                    "no_curves": {"status": "ok", "metric": 0.5},
                }
            }
        }
        result = fs.emit_training_curves(code_dir, per_model)
        assert result is True
        tc = _read_json(code_dir / "training_curves.json")
        assert "with_curves" in tc["m"]["e"]
        assert "no_curves" not in tc["m"]["e"]

    def test_returns_false_on_empty_per_model(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        assert fs.emit_training_curves(code_dir, {}) is False
        assert not (code_dir / "training_curves.json").exists()

    def test_returns_false_on_non_dict(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        assert fs.emit_training_curves(code_dir, None) is False  # type: ignore[arg-type]

    def test_step_key_included_in_output(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = {
            "m": {"e": {"b": {"training_curves": {"step": [0, 100, 200], "reward": [0.1, 0.2, 0.3]}}}}
        }
        fs.emit_training_curves(code_dir, per_model)
        tc = _read_json(code_dir / "training_curves.json")
        assert tc["m"]["e"]["b"]["step"] == [0.0, 100.0, 200.0]
        assert tc["m"]["e"]["b"]["reward"] == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# emit_figure_sidecars_harness
# ---------------------------------------------------------------------------

_SDAR_METRICS = {
    "per_model": {
        "qwen3_1_7b": {
            "alfworld": {
                "sdar": {"status": "ok", "metric": 0.82},
                "grpo": {"status": "ok", "metric": 0.70},
            }
        }
    }
}


class TestEmitFigureSidecarsHarness:
    def test_writes_sidecar_with_correct_shape(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = _SDAR_METRICS["per_model"]
        written = fs.emit_figure_sidecars_harness(code_dir, per_model)
        assert len(written) == 1
        sc = _read_json(tmp_path / written[0])
        # Mandatory keys the grader reads
        assert "figure" in sc
        assert "shows" in sc
        assert "x_axis" in sc and "scale" in sc["x_axis"]
        assert "y_axis" in sc and "label" in sc["y_axis"]
        assert "series" in sc
        assert "source" in sc
        assert "note" in sc

    def test_series_contains_baseline_values(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = _SDAR_METRICS["per_model"]
        written = fs.emit_figure_sidecars_harness(code_dir, per_model)
        sc = _read_json(tmp_path / written[0])
        assert set(sc["series"]) == {"sdar", "grpo"}
        assert sc["series"]["sdar"] == 0.82
        assert sc["series"]["grpo"] == 0.70

    def test_grader_glob_matches(self, tmp_path):
        """_gather_figure_sidecars uses rglob('fig_*.json') — must find our sidecar."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = _SDAR_METRICS["per_model"]
        fs.emit_figure_sidecars_harness(code_dir, per_model)
        assert list(code_dir.rglob("fig_*.json"))

    def test_file_named_with_fig_auto_prefix(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = _SDAR_METRICS["per_model"]
        written = fs.emit_figure_sidecars_harness(code_dir, per_model)
        assert all("fig_auto_" in w for w in written)

    def test_skips_when_agent_emitted_own_sidecar(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        # An agent-owned sidecar (not prefixed fig_auto_) blocks harness emission
        (code_dir / "fig_alfworld_curves.json").write_text('{"figure": "real"}')
        per_model = _SDAR_METRICS["per_model"]
        written = fs.emit_figure_sidecars_harness(code_dir, per_model)
        assert written == []
        assert not list(code_dir.glob("fig_auto_*.json"))

    def test_empty_per_model_returns_empty(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        assert fs.emit_figure_sidecars_harness(code_dir, {}) == []
        assert fs.emit_figure_sidecars_harness(code_dir, None) == []  # type: ignore[arg-type]

    def test_group_with_no_numeric_values_skipped_grounded(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = {
            "empty_model": {"e": {"b": {"status": "failed", "error": "oom"}}},
            "real_model": {"e": {"b": {"metric": 0.9}}},
        }
        written = fs.emit_figure_sidecars_harness(code_dir, per_model)
        assert len(written) == 1
        assert "real_model" in written[0]

    def test_curve_mode_downsample(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = {
            "m": {"e": {"b": {"loss": list(range(200))}}}
        }
        written = fs.emit_figure_sidecars_harness(code_dir, per_model, max_points=40)
        sc = _read_json(tmp_path / written[0])
        assert isinstance(sc["series"]["b"], list)
        assert len(sc["series"]["b"]) == 40
        assert sc["x_axis"]["label"] == "training step"

    def test_loss_metric_gets_log_axis(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = {"m": {"e": {"b": {"loss": 2.5}}}}
        written = fs.emit_figure_sidecars_harness(code_dir, per_model)
        sc = _read_json(tmp_path / written[0])
        assert sc["y_axis"]["scale"] == "log"

    def test_accuracy_metric_gets_linear_axis(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = {"m": {"e": {"b": {"accuracy": 0.85}}}}
        written = fs.emit_figure_sidecars_harness(code_dir, per_model)
        sc = _read_json(tmp_path / written[0])
        assert sc["y_axis"]["scale"] == "linear"

    def test_max_sidecars_cap(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = {f"m{i}": {"e": {"b": {"metric": float(i)}}} for i in range(10)}
        written = fs.emit_figure_sidecars_harness(code_dir, per_model, max_sidecars=3)
        assert len(written) <= 3

    def test_note_says_grounded_not_fabricated(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = _SDAR_METRICS["per_model"]
        written = fs.emit_figure_sidecars_harness(code_dir, per_model)
        sc = _read_json(tmp_path / written[0])
        assert "grounded" in sc["note"].lower() or "measured" in sc["note"].lower()

    def test_never_raises_on_unwritable_dir(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir(mode=0o555)  # read-only
        per_model = _SDAR_METRICS["per_model"]
        try:
            result = fs.emit_figure_sidecars_harness(code_dir, per_model)
            # May write nothing or raise OSError internally — but must not propagate
            assert isinstance(result, list)
        finally:
            code_dir.chmod(0o755)


# ---------------------------------------------------------------------------
# emit_sidecars — the flag-gated entry point (wired into _persist_metrics)
# ---------------------------------------------------------------------------

class TestEmitSidecars:
    def test_flag_off_no_files_written(self, tmp_path):
        """Default-OFF: unset flag ⇒ byte-identical (nothing written)."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        metrics = {
            "per_model": {
                "m": {"e": {"b": {"reward_history": [0.1, 0.2], "metric": 0.2}}}
            }
        }
        fs.emit_sidecars(code_dir, metrics)
        assert not (code_dir / "training_curves.json").exists()
        assert not list(code_dir.glob("fig_auto_*.json"))

    def test_flag_on_emits_training_curves(self, tmp_path, monkeypatch):
        monkeypatch.setenv(fs.FLAG, "1")
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        metrics = {
            "per_model": {
                "m": {"e": {"b": {"reward_history": [0.1, 0.2, 0.3], "metric": 0.3}}}
            }
        }
        fs.emit_sidecars(code_dir, metrics)
        assert (code_dir / "training_curves.json").exists()
        tc = _read_json(code_dir / "training_curves.json")
        assert tc["m"]["e"]["b"]["reward"] == [0.1, 0.2, 0.3]

    def test_flag_on_emits_figure_sidecars(self, tmp_path, monkeypatch):
        monkeypatch.setenv(fs.FLAG, "1")
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        metrics = {
            "per_model": {
                "m": {"e": {"sdar": {"metric": 0.82}, "grpo": {"metric": 0.70}}}
            }
        }
        fs.emit_sidecars(code_dir, metrics)
        assert list(code_dir.rglob("fig_*.json"))

    def test_flag_on_no_per_model_noop(self, tmp_path, monkeypatch):
        monkeypatch.setenv(fs.FLAG, "1")
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        fs.emit_sidecars(code_dir, {"status": "partial"})
        assert not (code_dir / "training_curves.json").exists()
        assert not list(code_dir.glob("fig_auto_*.json"))

    def test_flag_on_no_curves_no_training_curves_file(self, tmp_path, monkeypatch):
        """Scalar-only per_model: figure sidecars emit; training_curves.json does NOT."""
        monkeypatch.setenv(fs.FLAG, "1")
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        metrics = {
            "per_model": {"m": {"e": {"b": {"metric": 0.85, "status": "ok"}}}}
        }
        fs.emit_sidecars(code_dir, metrics)
        assert not (code_dir / "training_curves.json").exists()
        assert list(code_dir.glob("fig_auto_*.json"))  # figure sidecar still emitted

    def test_accepts_path_str(self, tmp_path, monkeypatch):
        monkeypatch.setenv(fs.FLAG, "1")
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        metrics = {
            "per_model": {"m": {"e": {"b": {"reward_history": [0.1, 0.2]}}}}
        }
        fs.emit_sidecars(str(code_dir), metrics)   # str, not Path
        assert (code_dir / "training_curves.json").exists()

    def test_never_raises_on_garbage_input(self, tmp_path, monkeypatch):
        monkeypatch.setenv(fs.FLAG, "1")
        # Does not raise even with bizarre input
        fs.emit_sidecars(tmp_path / "code", None)  # type: ignore[arg-type]
        fs.emit_sidecars(tmp_path / "code", {"per_model": "not-a-dict"})


# ---------------------------------------------------------------------------
# Integration: training_curves.json read back by primitives._reward_curve
# ---------------------------------------------------------------------------

class TestTrainingCurvesCompatibility:
    """The harness-written training_curves.json shape must be readable by
    ``primitives._reward_curve`` which inspects individual per_model leaves."""

    def test_leaf_with_training_curves_dict_is_readable(self, tmp_path):
        """Verify our output leaf shape matches what _reward_curve reads."""
        from backend.agents.rlm.primitives import _reward_curve

        # Leaf shape with nested training_curves (as emit_training_curves writes it)
        leaf = {"training_curves": {"reward": [0.1, 0.2, 0.3]}}
        result = _reward_curve(leaf)
        assert result == [0.1, 0.2, 0.3]

    def test_leaf_with_reward_history_is_readable(self, tmp_path):
        from backend.agents.rlm.primitives import _reward_curve

        leaf = {"reward_history": [0.4, 0.5, 0.6]}
        result = _reward_curve(leaf)
        assert result == [0.4, 0.5, 0.6]

    def test_full_sdar_shape_round_trip(self, tmp_path, monkeypatch):
        """Flag ON: emit_sidecars writes training_curves.json AND fig sidecars.

        Realistic SDAR leaf shape: each cell carries both a scalar ``metric``
        (success rate — the primary grader signal) AND per-step curves (the
        grounding evidence for convergence-improvement leaves).
        """
        monkeypatch.setenv(fs.FLAG, "1")
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        per_model = {
            "qwen3_1_7b": {
                "alfworld": {
                    "sdar": {
                        "status": "ok",
                        "metric": 0.82,
                        "training_curves": {"reward": [0.1, 0.2, 0.3], "loss": [2.5, 2.0, 1.5]},
                    },
                    "grpo": {
                        "status": "ok",
                        "metric": 0.70,
                        "reward_history": [0.05, 0.1, 0.15],
                    },
                }
            }
        }
        metrics = {"per_model": per_model}
        fs.emit_sidecars(code_dir, metrics)

        # training_curves.json captures the per-step histories
        tc = _read_json(code_dir / "training_curves.json")
        assert tc["qwen3_1_7b"]["alfworld"]["sdar"]["reward"] == [0.1, 0.2, 0.3]
        assert tc["qwen3_1_7b"]["alfworld"]["sdar"]["loss"] == [2.5, 2.0, 1.5]
        assert tc["qwen3_1_7b"]["alfworld"]["grpo"]["reward"] == [0.05, 0.1, 0.15]

        # Figure sidecars are also written (comparison: sdar vs grpo final metric)
        sidecars = list(code_dir.rglob("fig_auto_*.json"))
        assert sidecars
        sc = _read_json(sidecars[0])
        assert "qwen3_1_7b" in sc["figure"] or "qwen3_1_7b" in sc["shows"]
