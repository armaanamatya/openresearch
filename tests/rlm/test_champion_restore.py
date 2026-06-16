"""A4: champion-artifact restore at finalize — score ≡ best artifact produced.

OPENRESEARCH_CHAMPION_ARTIFACT default OFF → no-op. On → finalize restores the
highest-median-graded code snapshot (recorded per verify by binding) and ships
that grade, but NEVER downgrades a better latest state.
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.champion_artifact import record_champion
from backend.agents.rlm.report import _apply_champion_artifact


def _mk_snapshot(tmp_path, name, content):
    d = tmp_path / name / "code"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.py").write_text(content, encoding="utf-8")
    return d


def test_off_is_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENRESEARCH_CHAMPION_ARTIFACT", raising=False)
    out = _apply_champion_artifact({"overall_score": 0.5}, tmp_path)
    assert out == {"overall_score": 0.5}


def test_restores_best_artifact_and_ships_its_grade(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_CHAMPION_ARTIFACT", "1")
    snap_a = _mk_snapshot(tmp_path, "snapA", "# A (earlier, weaker)")
    snap_b = _mk_snapshot(tmp_path, "snapB", "# B (best artifact)")
    reg = tmp_path / "rlm_state" / "champions.json"
    record_champion(reg, evidence_key="kA", snapshot_dir=str(snap_a), median_score=0.60)
    record_champion(reg, evidence_key="kB", snapshot_dir=str(snap_b), median_score=0.80)
    # current code regressed; current score below the best champion
    cur_code = tmp_path / "code"
    cur_code.mkdir(parents=True, exist_ok=True)
    (cur_code / "train.py").write_text("# current (regressed)", encoding="utf-8")

    out = _apply_champion_artifact({"overall_score": 0.55}, tmp_path)
    assert out["overall_score"] == pytest.approx(0.80)
    assert out["champion_restored"] is True
    # the best artifact's source was restored into code/
    assert (cur_code / "train.py").read_text(encoding="utf-8") == "# B (best artifact)"


def test_never_downgrades_a_better_latest_state(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_CHAMPION_ARTIFACT", "1")
    snap = _mk_snapshot(tmp_path, "snap", "# weaker champion")
    reg = tmp_path / "rlm_state" / "champions.json"
    record_champion(reg, evidence_key="k", snapshot_dir=str(snap), median_score=0.40)
    cur_code = tmp_path / "code"
    cur_code.mkdir(parents=True, exist_ok=True)
    (cur_code / "train.py").write_text("# current (best)", encoding="utf-8")

    # current 0.70 > champion 0.40 → keep current, no restore, no downgrade
    out = _apply_champion_artifact({"overall_score": 0.70}, tmp_path)
    assert out["overall_score"] == pytest.approx(0.70)
    assert "champion_restored" not in out
    assert (cur_code / "train.py").read_text(encoding="utf-8") == "# current (best)"


def test_no_champions_recorded_is_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENRESEARCH_CHAMPION_ARTIFACT", "1")
    out = _apply_champion_artifact({"overall_score": 0.5}, tmp_path)
    assert out == {"overall_score": 0.5}
