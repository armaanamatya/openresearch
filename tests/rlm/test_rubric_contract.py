"""Tests for backend.agents.rlm.rubric_contract.

Pinned guarantees:

  * No contract → no violations (fail-soft default).
  * Missing required key → one Eval-protocol violation.
  * Missing required artifact (literal + glob) → one Artifact violation.
  * Variant in variants_required missing from per_model AND omitted
    → one Experiment-execution violation.
  * Variant honestly omitted → NO violation.
  * Result within tolerance → no violation.
  * Result off by >10% → one Result-match violation with the gap detail.
  * Non-numeric metric for a numeric paper_target → one violation.
  * paper_targets meta keys (required_metrics_keys, etc.) skipped during
    result-match scan.
  * load_paper_targets returns None when no YAML / no paper_targets section.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from backend.agents.rlm import rubric_contract as rc


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_no_contract_returns_empty(tmp_path: Path) -> None:
    report = rc.validate({}, tmp_path, None)
    assert report.compliant
    assert report.violations == []


def test_empty_contract_returns_empty(tmp_path: Path) -> None:
    report = rc.validate({}, tmp_path, {})
    assert report.compliant


def test_missing_required_metrics_key(tmp_path: Path) -> None:
    report = rc.validate(
        metrics={"other_key": 0.5},
        artifact_root=tmp_path,
        paper_targets={"required_metrics_keys": ["mnist_baseline_final_acc"]},
    )
    assert not report.compliant
    v = report.violations[0]
    assert v.area == "Evaluation protocol and metric correctness"
    assert "mnist_baseline_final_acc" in v.detail


def test_variant_missing_and_not_omitted(tmp_path: Path) -> None:
    report = rc.validate(
        metrics={"per_model": {"baseline": {"acc": 0.9}}},
        artifact_root=tmp_path,
        paper_targets={"variants_required": ["baseline", "bn"]},
    )
    assert len(report.violations) == 1
    v = report.violations[0]
    assert v.area == "Experiment execution and reproducibility"
    assert "'bn'" in v.detail


def test_variant_honestly_omitted_passes(tmp_path: Path) -> None:
    report = rc.validate(
        metrics={
            "per_model": {"baseline": {"acc": 0.9}},
            "omitted": {"bn": "ImageNet variant requires ~150 GB dataset"},
        },
        artifact_root=tmp_path,
        paper_targets={"variants_required": ["baseline", "bn"]},
    )
    assert report.compliant


def test_missing_required_artifact_literal(tmp_path: Path) -> None:
    report = rc.validate(
        metrics={},
        artifact_root=tmp_path,
        paper_targets={"required_artifacts": ["README.md"]},
    )
    v = report.violations[0]
    assert v.area == "Artifact completeness and provenance"
    assert "README.md" in v.detail


def test_required_artifact_present(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("ok")
    report = rc.validate(
        metrics={},
        artifact_root=tmp_path,
        paper_targets={"required_artifacts": ["README.md"]},
    )
    assert report.compliant


def test_required_artifact_glob_matches(tmp_path: Path) -> None:
    (tmp_path / "fig_mnist.png").write_text("png")
    report = rc.validate(
        metrics={},
        artifact_root=tmp_path,
        paper_targets={"required_artifacts": ["fig_*.png"]},
    )
    assert report.compliant


def test_required_artifact_glob_misses(tmp_path: Path) -> None:
    report = rc.validate(
        metrics={},
        artifact_root=tmp_path,
        paper_targets={"required_artifacts": ["fig_*.png"]},
    )
    assert len(report.violations) == 1
    assert "fig_*.png" in report.violations[0].detail


def test_result_match_within_tolerance(tmp_path: Path) -> None:
    # 0.962 vs target 0.965 → 0.31% relative error, well under 10%.
    report = rc.validate(
        metrics={"mnist_baseline_final_acc": 0.962},
        artifact_root=tmp_path,
        paper_targets={"mnist_baseline_final_acc": 0.965},
    )
    assert report.compliant


def test_result_match_off_by_more_than_tolerance(tmp_path: Path) -> None:
    # 0.78 vs 0.965 → ~19% off, well over 10%.
    report = rc.validate(
        metrics={"mnist_baseline_final_acc": 0.78},
        artifact_root=tmp_path,
        paper_targets={"mnist_baseline_final_acc": 0.965},
    )
    assert len(report.violations) == 1
    v = report.violations[0]
    assert v.area == "Result match versus the paper's reported targets"
    assert "0.78" in v.detail
    assert "0.965" in v.detail
    assert "%" in v.detail  # carries relative-error percentage


def test_paper_target_present_but_metric_missing(tmp_path: Path) -> None:
    # paper_targets has a number but metrics.json doesn't carry it.
    # Should produce one Result-match violation (not duplicated as Eval-protocol).
    report = rc.validate(
        metrics={},
        artifact_root=tmp_path,
        paper_targets={"mnist_baseline_final_acc": 0.965},
    )
    assert len(report.violations) == 1
    assert report.violations[0].area == "Result match versus the paper's reported targets"


def test_non_numeric_metric_for_numeric_target(tmp_path: Path) -> None:
    report = rc.validate(
        metrics={"mnist_baseline_final_acc": "n/a"},
        artifact_root=tmp_path,
        paper_targets={"mnist_baseline_final_acc": 0.965},
    )
    assert len(report.violations) == 1
    assert "not numeric" in report.violations[0].detail


def test_meta_keys_skipped_during_result_scan(tmp_path: Path) -> None:
    # required_metrics_keys / required_artifacts / variants_required are meta
    # — they should NOT be compared as numeric targets.
    report = rc.validate(
        metrics={},
        artifact_root=tmp_path,
        paper_targets={
            "required_metrics_keys": ["foo"],
            "required_artifacts": ["bar.png"],
            "variants_required": ["baz"],
        },
    )
    # Should produce: missing-key (Eval), missing-artifact (Artifact), missing-variant (Execution).
    # Should NOT produce a Result-match violation for any of the meta keys.
    areas = {v.area for v in report.violations}
    assert "Result match versus the paper's reported targets" not in areas


def test_summary_carries_counts_per_area(tmp_path: Path) -> None:
    report = rc.validate(
        metrics={},
        artifact_root=tmp_path,
        paper_targets={
            "required_metrics_keys": ["mnist_acc", "cifar_acc"],
            "required_artifacts": ["README.md"],
        },
    )
    assert "contract violation" in report.summary
    assert "Evaluation protocol" in report.summary
    assert "Artifact completeness" in report.summary


def test_compliant_summary(tmp_path: Path) -> None:
    report = rc.validate({}, tmp_path, {"required_metrics_keys": []})
    assert report.compliant
    assert "satisfied" in report.summary


# ---------------------------------------------------------------------------
# load_paper_targets()
# ---------------------------------------------------------------------------


def test_load_paper_targets_no_arxiv_id() -> None:
    assert rc.load_paper_targets(None) is None
    assert rc.load_paper_targets("") is None


def test_load_paper_targets_no_yaml(tmp_path: Path) -> None:
    assert rc.load_paper_targets("9999.99999", docs_root=tmp_path) is None


def test_load_paper_targets_yaml_without_section(tmp_path: Path) -> None:
    (tmp_path / "1234.5678.yaml").write_text(
        textwrap.dedent(
            """\
            algorithm_invariants:
              dropout_rate: 0.5
            """
        )
    )
    assert rc.load_paper_targets("1234.5678", docs_root=tmp_path) is None


def test_load_paper_targets_yaml_with_section(tmp_path: Path) -> None:
    (tmp_path / "1234.5678.yaml").write_text(
        textwrap.dedent(
            """\
            paper_targets:
              mnist_baseline_final_acc: 0.965
              required_metrics_keys:
                - mnist_baseline_final_acc
            """
        )
    )
    targets = rc.load_paper_targets("1234.5678", docs_root=tmp_path)
    assert targets is not None
    assert targets["mnist_baseline_final_acc"] == pytest.approx(0.965)
    assert targets["required_metrics_keys"] == ["mnist_baseline_final_acc"]


def test_load_paper_targets_malformed_yaml_is_fail_soft(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("not: valid: yaml: at all: [")
    assert rc.load_paper_targets("bad", docs_root=tmp_path) is None
