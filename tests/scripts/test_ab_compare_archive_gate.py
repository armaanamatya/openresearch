"""scripts/ab_compare.py — archive-completeness gate (Task 8).

When ``--require-stamped`` / ``OPENRESEARCH_REQUIRE_STAMPED_AB=1`` is active,
``validate_stamped_pair`` must refuse to emit a Δ when either arm's run-dir
archive is incomplete (the Adam lesson — incomplete archive → folklore numbers).

The gate is intentionally scoped to validator mode: the default unstamped
reporter path stays byte-for-byte unchanged.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "ab_compare", REPO_ROOT / "scripts" / "ab_compare.py"
)
ab_compare = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("ab_compare", ab_compare)
_SPEC.loader.exec_module(ab_compare)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_SCOPE = {
    "requested": "CIFAR-10: Models A/B/C",
    "ran": ["a_base", "a_strided"],
    "gaps": [],
}
_RUBRIC_TREE = {"leaves": [{"id": "leaf_x"}], "version": 1}


def _write_stamped_arm(
    runs_root: Path,
    project_id: str,
    *,
    arm: str,
    score: float = 0.60,
    pair_id: str = "ab-1",
    complete: bool = False,
) -> Path:
    """Write a stamped arm dir. ``complete=True`` writes ALL required archive
    artifacts so ``check_bes_archive`` returns True; ``complete=False`` writes
    only ``final_report.json`` (incomplete archive)."""
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "paper": {"id": "1412.6806", "title": "All-CNN"},
        "verdict": "partial",
        "scope": _SCOPE,
        "rubric": {"overall_score": score, "meets_target": score >= 0.6, "areas": [], "leaf_scores": []},
        "cost": {"llm_usd": 1.5},
        "iterations": 2,
        "started_at": "2026-06-11T00:00:00+00:00",
        "completed_at": "2026-06-11T02:00:00+00:00",
        "experiment_arm": {
            "arm": arm,
            "ab_pair_id": pair_id,
            "bes": {"enabled": arm == "bes", "candidates_per_cluster": 2 if arm == "bes" else 1},
        },
    }
    (run_dir / "final_report.json").write_text(json.dumps(report), encoding="utf-8")
    # rubric_tree.json always written (required for stamped-pair sha check)
    (run_dir / "rubric_tree.json").write_text(json.dumps(_RUBRIC_TREE), encoding="utf-8")

    if complete:
        # Write all artifacts required by check_bes_archive
        for fname in (
            "bes_candidates.json",
            "dashboard_events.jsonl",
            "experiment_runs.jsonl",
            "rubric_evaluation.json",
            "metrics.json",
            "generated_rubric.json",
        ):
            (run_dir / fname).write_text("{}", encoding="utf-8")
        (run_dir / "candidates").mkdir(exist_ok=True)

    return run_dir


# ---------------------------------------------------------------------------
# Archive-completeness helper is already functional (pre-change baseline)
# ---------------------------------------------------------------------------

def test_incomplete_arm_archive_blocks_delta(tmp_path: Path, monkeypatch):
    """Bare-minimum test: confirm check_bes_archive reports incomplete dirs."""
    ctrl_dir = tmp_path / "control"
    ctrl_dir.mkdir()
    bes_dir = tmp_path / "bes"
    bes_dir.mkdir()
    (ctrl_dir / "final_report.json").write_text("{}")
    (bes_dir / "final_report.json").write_text("{}")

    from backend.agents.rlm.archive_completeness import check_bes_archive
    assert check_bes_archive(ctrl_dir).complete is False
    assert check_bes_archive(bes_dir).complete is False


# ---------------------------------------------------------------------------
# Gate: validator mode refuses when control archive is incomplete
# ---------------------------------------------------------------------------

def test_validator_refuses_incomplete_control_archive(tmp_path: Path):
    # Control arm has only final_report.json + rubric_tree.json (incomplete).
    # BES arm is fully complete.
    _write_stamped_arm(tmp_path, "prj_ctl", arm="control", score=0.55, complete=False)
    _write_stamped_arm(tmp_path, "prj_bes", arm="bes", score=0.72, complete=True)

    cmp = ab_compare.build_comparison(tmp_path, pair_id="ab-1", require_stamped=True)

    assert cmp["validation_error"] is not None
    assert "archive" in cmp["validation_error"].lower() or "incomplete" in cmp["validation_error"].lower() or "missing" in cmp["validation_error"].lower()
    # No Δ emitted on refusal.
    assert cmp["deltas"] == {}
    assert cmp["top_leaf_moves"] == []


def test_validator_refuses_incomplete_bes_archive(tmp_path: Path):
    # Control arm is complete; BES arm is incomplete.
    _write_stamped_arm(tmp_path, "prj_ctl", arm="control", score=0.55, complete=True)
    _write_stamped_arm(tmp_path, "prj_bes", arm="bes", score=0.72, complete=False)

    cmp = ab_compare.build_comparison(tmp_path, pair_id="ab-1", require_stamped=True)

    assert cmp["validation_error"] is not None
    assert "archive" in cmp["validation_error"].lower() or "incomplete" in cmp["validation_error"].lower() or "missing" in cmp["validation_error"].lower()
    assert cmp["deltas"] == {}


def test_validator_refuses_both_arms_incomplete(tmp_path: Path):
    # Neither arm has a complete archive — both should be caught.
    _write_stamped_arm(tmp_path, "prj_ctl", arm="control", score=0.55, complete=False)
    _write_stamped_arm(tmp_path, "prj_bes", arm="bes", score=0.72, complete=False)

    cmp = ab_compare.build_comparison(tmp_path, pair_id="ab-1", require_stamped=True)

    assert cmp["validation_error"] is not None
    assert cmp["deltas"] == {}


# ---------------------------------------------------------------------------
# Gate: validator mode ACCEPTS a pair when both archives are complete
# ---------------------------------------------------------------------------

def test_validator_accepts_complete_archives(tmp_path: Path):
    _write_stamped_arm(tmp_path, "prj_ctl", arm="control", score=0.55, complete=True)
    _write_stamped_arm(tmp_path, "prj_bes", arm="bes", score=0.72, complete=True)

    cmp = ab_compare.build_comparison(tmp_path, pair_id="ab-1", require_stamped=True)

    assert cmp["validation_error"] is None
    # Δ is emitted (0.72 - 0.55 = 0.17)
    assert cmp["deltas"]["overall_score"] == pytest.approx(0.72 - 0.55)


# ---------------------------------------------------------------------------
# Gate: reporter mode (flag OFF) is unchanged — incomplete archives do NOT block
# ---------------------------------------------------------------------------

def test_reporter_mode_unchanged_with_incomplete_archives(tmp_path: Path):
    """Default (unstamped/reporter) mode must stay byte-for-byte unchanged."""
    _write_stamped_arm(tmp_path, "prj_ctl", arm="control", score=0.55, complete=False)
    _write_stamped_arm(tmp_path, "prj_bes", arm="bes", score=0.72, complete=False)

    # require_stamped defaults OFF — archive check must NOT fire
    cmp = ab_compare.build_comparison(tmp_path, pair_id="ab-1")

    # No validator keys at all in reporter mode
    assert "validation_error" not in cmp
    assert "require_stamped" not in cmp
    # Δ is still emitted
    assert cmp["deltas"]["overall_score"] == pytest.approx(0.72 - 0.55)


# ---------------------------------------------------------------------------
# validate_stamped_pair direct call — confirms the archive check is wired there
# ---------------------------------------------------------------------------

def test_validate_stamped_pair_refuses_incomplete_control_directly(tmp_path: Path):
    """Call validate_stamped_pair directly with arm dicts pointing to incomplete dirs."""
    ctrl_dir = tmp_path / "control"
    ctrl_dir.mkdir()
    bes_dir = tmp_path / "bes"
    bes_dir.mkdir()

    # Provide rubric_tree.json so the sha/scope checks pass — only archive is incomplete
    rubric_bytes = json.dumps(_RUBRIC_TREE).encode()
    (ctrl_dir / "rubric_tree.json").write_bytes(rubric_bytes)
    (bes_dir / "rubric_tree.json").write_bytes(rubric_bytes)
    # BES gets full archive; control only has rubric_tree.json
    for fname in (
        "bes_candidates.json",
        "dashboard_events.jsonl",
        "experiment_runs.jsonl",
        "rubric_evaluation.json",
        "final_report.json",
        "metrics.json",
        "generated_rubric.json",
    ):
        (bes_dir / fname).write_text("{}")
    (bes_dir / "candidates").mkdir()

    scope_str = json.dumps(_SCOPE, sort_keys=True)
    ctrl_arm = {
        "project_id": "ctrl",
        "_run_dir": ctrl_dir,
        "_stamped": True,
        "_scope": _SCOPE,
    }
    bes_arm = {
        "project_id": "bes",
        "_run_dir": bes_dir,
        "_stamped": True,
        "_scope": _SCOPE,
    }

    reason = ab_compare.validate_stamped_pair(ctrl_arm, bes_arm)
    assert reason is not None
    # Reason must mention the arm and include archive/missing context
    assert "ctrl" in reason or "control" in reason
    assert any(kw in reason.lower() for kw in ("archive", "incomplete", "missing"))
