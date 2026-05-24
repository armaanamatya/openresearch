"""Tests for backend.agents.rlm.rubric_guard — Lane G."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backend.agents.rlm.rubric_guard import (
    RubricGuardFailure,
    assert_metrics_schema,
)


def test_all_keys_present_does_not_raise(tmp_path: Path) -> None:
    """Happy path: every required key is present; no raise."""
    metrics = {
        "mnist_baseline_final_acc": 0.81,
        "mnist_bn_final_acc": 0.83,
        "per_model": {"mlp": {"acc": 0.8}},
    }
    # No required artifacts — keys-only check.
    assert_metrics_schema(
        metrics,
        required_keys=["mnist_baseline_final_acc", "mnist_bn_final_acc", "per_model"],
    )


def test_missing_key_raises_with_key_name(tmp_path: Path) -> None:
    """Missing required key surfaces with the key name in the JSON detail."""
    metrics = {"mnist_baseline_final_acc": 0.81}
    with pytest.raises(RubricGuardFailure) as excinfo:
        assert_metrics_schema(
            metrics,
            required_keys=["mnist_baseline_final_acc", "mnist_bn_final_acc"],
        )
    detail = json.loads(str(excinfo.value))
    assert detail["rubric_guard"] == "schema_violation"
    assert "mnist_bn_final_acc" in detail["missing_keys"]
    # The present key should NOT be reported missing.
    assert "mnist_baseline_final_acc" not in detail["missing_keys"]


def test_nested_dotted_key_resolves(tmp_path: Path) -> None:
    """Dotted-path keys resolve against nested dicts."""
    metrics = {"per_model": {"qwen3_1.7b": {"acc": 0.74}}}
    # Dotted path resolves through the nested dict.
    assert_metrics_schema(
        metrics,
        required_keys=["per_model", "per_model.qwen3_1.7b"],
    )


def test_missing_artifact_raises_with_artifact_name(tmp_path: Path) -> None:
    """A missing artifact surfaces in the JSON detail's missing_artifacts list."""
    metrics = {"baseline_final_acc": 0.81}
    # tmp_path is empty — every required artifact is missing.
    with pytest.raises(RubricGuardFailure) as excinfo:
        assert_metrics_schema(
            metrics,
            required_keys=["baseline_final_acc"],
            required_artifacts=["README.md", "training_curves.json"],
            artifact_dir=tmp_path,
        )
    detail = json.loads(str(excinfo.value))
    assert "README.md" in detail["missing_artifacts"]
    assert "training_curves.json" in detail["missing_artifacts"]


def test_artifact_glob_matches_when_at_least_one_file(tmp_path: Path) -> None:
    """Glob ``fig_*.png`` matches when at least one figure exists."""
    (tmp_path / "fig_curve.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "README.md").write_text("# notes")
    # No raise — both literals + glob resolve.
    assert_metrics_schema(
        metrics={"acc": 0.8},
        required_keys=["acc"],
        required_artifacts=["README.md", "fig_*.png"],
        artifact_dir=tmp_path,
    )


def test_artifact_glob_mismatch_raises(tmp_path: Path) -> None:
    """Glob ``fig_*.png`` does NOT match when no png exists."""
    (tmp_path / "README.md").write_text("# notes")
    (tmp_path / "curves.csv").write_text("step,acc\n1,0.8\n")  # wrong extension
    with pytest.raises(RubricGuardFailure) as excinfo:
        assert_metrics_schema(
            metrics={"acc": 0.8},
            required_keys=["acc"],
            required_artifacts=["fig_*.png"],
            artifact_dir=tmp_path,
        )
    detail = json.loads(str(excinfo.value))
    assert "fig_*.png" in detail["missing_artifacts"]


def test_output_dir_env_var_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When artifact_dir is None, OUTPUT_DIR env var is used."""
    (tmp_path / "metrics.json").write_text("{}")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    # File `metrics.json` exists, so the literal `metrics.json` artifact resolves.
    assert_metrics_schema(
        metrics={"acc": 0.8},
        required_keys=["acc"],
        required_artifacts=["metrics.json"],
        # artifact_dir omitted — should resolve via OUTPUT_DIR.
    )


def test_non_dict_metrics_raises(tmp_path: Path) -> None:
    """Passing a non-dict object raises a structured failure."""
    with pytest.raises(RubricGuardFailure) as excinfo:
        assert_metrics_schema(
            metrics=[1, 2, 3],  # type: ignore[arg-type]
            required_keys=["any"],
        )
    detail = json.loads(str(excinfo.value))
    assert detail["rubric_guard"] == "metrics_not_dict"
    assert detail["got_type"] == "list"


def test_no_required_artifacts_means_only_keys_checked(tmp_path: Path) -> None:
    """When required_artifacts is None / empty, only key check runs."""
    metrics = {"acc": 0.8}
    # No raise; no artifacts to check.
    assert_metrics_schema(metrics, required_keys=["acc"], required_artifacts=None)
    assert_metrics_schema(metrics, required_keys=["acc"], required_artifacts=[])


def test_empty_required_keys_with_artifacts(tmp_path: Path) -> None:
    """An empty required_keys list still permits artifact-only validation."""
    (tmp_path / "README.md").write_text("ok")
    # Empty key list, all artifacts present — no raise.
    assert_metrics_schema(
        metrics={},
        required_keys=[],
        required_artifacts=["README.md"],
        artifact_dir=tmp_path,
    )
