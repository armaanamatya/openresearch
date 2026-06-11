"""scripts/ab_compare.py — deterministic pairing + rendering tests."""

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


def _write_report(
    runs_root: Path,
    project_id: str,
    *,
    arm: str | None,
    score: float,
    pair_id: str | None = None,
    completed_at: str = "2026-06-11T06:00:00+00:00",
    leaf_scores: list[dict] | None = None,
    bes_pool: list[dict] | None = None,
) -> None:
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "paper": {"id": "1412.6806", "title": "All-CNN"},
        "verdict": "partial",
        "rubric": {
            "overall_score": score,
            "meets_target": score >= 0.6,
            "areas": [],
            "leaf_scores": leaf_scores or [],
        },
        "cost": {"llm_usd": 2.0},
        "iterations": 3,
        "started_at": "2026-06-11T00:00:00+00:00",
        "completed_at": completed_at,
    }
    if arm is not None:
        report["experiment_arm"] = {
            "arm": arm,
            "ab_pair_id": pair_id,
            "bes": {
                "enabled": arm == "bes",
                "candidates_per_cluster": 2 if arm == "bes" else 1,
                "winner": "rlm_impl#1" if arm == "bes" else None,
                "pool": bes_pool or [],
            },
        }
    (run_dir / "final_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_pairs_control_and_bes_with_deltas(tmp_path: Path):
    _write_report(
        tmp_path, "prj_x_control", arm="control", score=0.60,
        leaf_scores=[{"id": "leaf_a", "score": 0.5}, {"id": "leaf_b", "score": 0.9}],
    )
    _write_report(
        tmp_path, "prj_x_bes", arm="bes", score=0.72,
        leaf_scores=[{"id": "leaf_a", "score": 0.9}, {"id": "leaf_b", "score": 0.9}],
        bes_pool=[
            {"candidate_id": "rlm_impl#0", "ok": True, "score": 0.3},
            {"candidate_id": "rlm_impl#1", "ok": True, "score": 0.7},
        ],
    )

    cmp = ab_compare.build_comparison(tmp_path, paper="1412.6806")

    assert cmp["control"]["project_id"] == "prj_x_control"
    assert cmp["bes"]["project_id"] == "prj_x_bes"
    assert cmp["deltas"]["overall_score"] == pytest.approx(0.12)
    moves = {m["leaf"]: m["delta"] for m in cmp["top_leaf_moves"]}
    assert moves == {"leaf_a": pytest.approx(0.4)}

    md = ab_compare.render_markdown(cmp)
    assert "Δ (bes − control)" in md
    assert "rlm_impl#1 ← selected" in md


def test_latest_run_wins_per_arm(tmp_path: Path):
    _write_report(tmp_path, "prj_old_bes", arm="bes", score=0.5,
                  completed_at="2026-06-10T00:00:00+00:00")
    _write_report(tmp_path, "prj_new_bes", arm="bes", score=0.4,
                  completed_at="2026-06-11T00:00:00+00:00")
    _write_report(tmp_path, "prj_ctl", arm="control", score=0.3)

    cmp = ab_compare.build_comparison(tmp_path, paper="1412.6806")
    assert cmp["bes"]["project_id"] == "prj_new_bes"  # latest, not best

    cmp_best = ab_compare.build_comparison(tmp_path, paper="1412.6806", select="best")
    assert cmp_best["bes"]["project_id"] == "prj_old_bes"


def test_unstamped_reports_can_serve_as_control(tmp_path: Path):
    _write_report(tmp_path, "prj_legacy", arm=None, score=0.55)
    _write_report(tmp_path, "prj_bes", arm="bes", score=0.6)

    cmp = ab_compare.build_comparison(tmp_path, paper="1412.6806")
    assert cmp["control"]["project_id"] == "prj_legacy"
    assert cmp["arms_found"] == {"bes": 1, "unstamped": 1}


def test_pair_id_filter_excludes_other_runs(tmp_path: Path):
    _write_report(tmp_path, "prj_pair_ctl", arm="control", score=0.5, pair_id="ab-1")
    _write_report(tmp_path, "prj_pair_bes", arm="bes", score=0.6, pair_id="ab-1")
    _write_report(tmp_path, "prj_other_bes", arm="bes", score=0.9, pair_id="ab-2")

    cmp = ab_compare.build_comparison(tmp_path, pair_id="ab-1")
    assert cmp["bes"]["project_id"] == "prj_pair_bes"
    assert cmp["control"]["project_id"] == "prj_pair_ctl"


def test_missing_arm_yields_incomplete_comparison(tmp_path: Path):
    _write_report(tmp_path, "prj_only_ctl", arm="control", score=0.5)
    cmp = ab_compare.build_comparison(tmp_path, paper="1412.6806")
    assert cmp["bes"] is None
    assert cmp["deltas"] == {}
    md = ab_compare.render_markdown(cmp)
    assert "_missing_" in md


def test_arm_own_report_preferred_over_seeded_ancestor(tmp_path: Path):
    """Arm dirs carry the seeded ancestor as an attempt; the ancestor may
    out-score the arm's own run AND lacks the experiment_arm stamp — pairing
    must read the arm's own top-level report (allcnn-ab-20260611 regression)."""
    _write_report(tmp_path, "prj_y_bes", arm="bes", score=0.7378, pair_id="ab-y")
    # Higher-scoring UNSTAMPED ancestor inside the arm's attempts/.
    anc = tmp_path / "prj_y_bes" / "attempts" / "20260610T235900-000000-best739"
    anc.mkdir(parents=True)
    (anc / "final_report.json").write_text(json.dumps({
        "paper": {"id": "1412.6806"},
        "verdict": "reproduced",
        "rubric": {"overall_score": 0.7395, "meets_target": True, "areas": []},
    }))
    _write_report(tmp_path, "prj_y_ctl", arm="control", score=0.65, pair_id="ab-y")

    cmp = ab_compare.build_comparison(tmp_path, pair_id="ab-y")
    assert cmp["bes"]["project_id"] == "prj_y_bes"
    assert cmp["bes"]["overall_score"] == pytest.approx(0.7378)  # the arm's own, not 0.7395
    assert cmp["control"]["project_id"] == "prj_y_ctl"
