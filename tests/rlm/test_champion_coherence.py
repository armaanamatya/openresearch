"""A4 coherence: champion restore ships leaf_scores from the champion's rubric block.

When champion restore fires, the restored overall_score must be paired with the
champion's OWN leaf_scores (and other rubric fields), never the stale latest-verify
leaves that earned the *lower* score. Prior to the fix, only overall_score was
updated — leaf_scores stayed at the regressed/stale lower value.
"""

from __future__ import annotations

from pathlib import Path

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
