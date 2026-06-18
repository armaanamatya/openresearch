"""A4 coherence: champion restore ships leaf_scores from the champion's rubric block.

When champion restore fires, the restored overall_score must be paired with the
champion's OWN leaf_scores (and other rubric fields), never the stale latest-verify
leaves that earned the *lower* score. Prior to the fix, only overall_score was
updated — leaf_scores stayed at the regressed/stale lower value.
"""

from __future__ import annotations

import json

from backend.agents.rlm import report as R
from backend.agents.rlm import champion_artifact as ca


def test_champion_restore_ships_coherent_leaves(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_CHAMPION_ARTIFACT", "1")

    proj = tmp_path

    # Create a minimal code/ dir so snapshot_code has something to copy
    code_dir = proj / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "train.py").write_text("# champion source", encoding="utf-8")

    # Snapshot dir layout: champions/<key>/code  +  champions/<key>/rubric_block.json
    snap = proj / "rlm_state" / "champions" / "k" / "code"
    ca.snapshot_code(code_dir, snap)

    ca.snapshot_rubric(
        {
            "overall_score": 0.73,
            "leaf_scores": [{"id": "a", "score": 0.73}],
            "meets_target": True,
            "target_score": 0.6,
        },
        snap.parent,  # entry dir, where rubric_block.json lives
    )

    reg = proj / "rlm_state" / "champions.json"
    ca.record_champion(
        reg,
        evidence_key="k",
        snapshot_dir=str(snap),
        median_score=0.73,
        sample_count=1,
    )

    # Stale latest-verify rubric (lower score, wrong leaves)
    stale = {
        "overall_score": 0.55,
        "leaf_scores": [{"id": "a", "score": 0.55}],
        "meets_target": False,
        "target_score": 0.6,
    }

    out = R._apply_champion_artifact(stale, proj)

    assert out["overall_score"] == 0.73
    # Champion leaves must replace the stale 0.55 leaves
    assert out["leaf_scores"] == [{"id": "a", "score": 0.73}]
    assert out["champion_restored"] is True
    assert out["champion_sample_count"] == 1
    # Rubric meta-fields are also restored from the champion block
    assert out["meets_target"] is True


# ---------------------------------------------------------------------------
# Integration test: champion leaves survive the full write-path merge
# ---------------------------------------------------------------------------

def test_champion_leaves_survive_write_path_merge(tmp_path, monkeypatch):
    """Champion overall_score=0.73 + its leaf_scores=[0.73] must survive
    `write_final_report_rlm` even when a STALE `rubric_evaluation.json`
    (overall_score=0.55, leaf_scores=[0.55]) sits on disk.

    The merge in write_final_report_rlm (~line 1730-1760) must NOT clobber the
    champion's higher score with the stale eval's lower value:
      - _eval_wins = (0.55 >= 0.73) is False  → champion score stands
      - current.get("leaf_scores") is not None → stale eval does not backfill

    This is the load-bearing coherence claim of the BES conversion work.
    """
    # Disable the evidence gate so a minimal fixture report isn't downgraded.
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_GATE", "0")
    # Disable best-of-run floor (reads dashboard_events.jsonl which won't exist)
    # — it is already a no-op when the file is absent, but be explicit.
    monkeypatch.setenv("OPENRESEARCH_EVIDENCE_FINGERPRINT", "0")

    proj = tmp_path

    # 1. Build a minimal RLMFinalReport whose rubric already carries champion
    #    leaves (overall_score=0.73).  This is the state AFTER
    #    _apply_champion_artifact has run inside build_final_report.
    champion_rubric = {
        "overall_score": 0.73,
        "leaf_scores": [{"id": "a", "score": 0.73}],
        "meets_target": True,
        "target_score": 0.6,
        "areas": [],
        "champion_restored": True,
    }
    report = R.RLMFinalReport(
        verdict="reproduced",
        rubric=champion_rubric,
    )

    # 2. Plant a STALE rubric_evaluation.json (lower score, stale leaves).
    stale_eval = {
        "overall_score": 0.55,
        "leaf_scores": [{"id": "a", "score": 0.55}],
        "meets_target": False,
        "target_score": 0.6,
        "weak_leaves": ["a"],
        "leaf_count": 1,
        "graded": True,
    }
    (proj / "rubric_evaluation.json").write_text(
        json.dumps(stale_eval), encoding="utf-8"
    )

    # 3. Run the full write path (the merge lives here).
    json_path, _md_path = R.write_final_report_rlm(report, proj)

    # 4. Read the written final_report.json and assert champion values survived.
    shipped = json.loads(json_path.read_text(encoding="utf-8"))
    shipped_rubric = shipped.get("rubric", {})

    # Champion score must survive — stale eval (0.55) must NOT overwrite it.
    assert shipped_rubric.get("overall_score") == 0.73, (
        f"Champion overall_score=0.73 was clobbered; shipped {shipped_rubric.get('overall_score')}"
    )

    # Champion leaves must survive — stale eval must NOT backfill 0.55 leaves.
    shipped_leaves = shipped_rubric.get("leaf_scores", [])
    assert shipped_leaves, "leaf_scores must be present in the shipped report"
    shipped_leaf_score = shipped_leaves[0].get("score")
    assert shipped_leaf_score == 0.73, (
        f"Champion leaf score=0.73 was clobbered; shipped leaf score={shipped_leaf_score}"
    )

    # meets_target must be consistent with the champion score (0.73 >= 0.6 → True).
    assert shipped_rubric.get("meets_target") is True, (
        f"meets_target should be True for score=0.73 >= target=0.6; "
        f"got {shipped_rubric.get('meets_target')}"
    )
