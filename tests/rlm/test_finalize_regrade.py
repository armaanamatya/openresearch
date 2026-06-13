"""Finalize-time freshness re-grade (finalize_regrade.py).

2026-06-13 All-CNN v5: a complete 13/14-converged grid shipped at 0.558
because it was graded ONCE on a partial grid and never re-graded. This rail
re-grades grown evidence at finalize and adopts a strictly-higher score.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.rlm import finalize_regrade as fr


def _grid_metrics(n_converged: int = 13, n_dead: int = 1) -> dict:
    pm: dict = {}
    for i in range(n_converged):
        pm[f"m{i}"] = {"cifar10": {"base": {"test_error_pct": 12.0 + i, "test_accuracy": 0.88}}}
    for j in range(n_dead):
        pm[f"d{j}"] = {"cifar10": {"base": {"test_error_pct": 90.0, "test_accuracy": 0.1}}}
    return {"status": "completed", "per_model": pm}


def _project(tmp_path: Path, *, graded_at: float | None, metrics_at: float | None,
             recorded: float = 0.5413, target: float = 0.7437) -> Path:
    code = tmp_path / "code"
    code.mkdir(parents=True, exist_ok=True)
    (tmp_path / "generated_rubric.json").write_text(json.dumps(
        {"source": "generated", "id": "r", "sub_tasks": []}))
    mp = code / "metrics.json"
    mp.write_text(json.dumps(_grid_metrics()))
    if metrics_at is not None:
        os.utime(mp, (metrics_at, metrics_at))
    if graded_at is not None:
        ev = tmp_path / "rubric_evaluation.json"
        ev.write_text(json.dumps({"overall_score": recorded, "target_score": target,
                                  "graded": 22, "leaf_count": 22}))
        os.utime(ev, (graded_at, graded_at))
    return tmp_path


# ---------------------------------------------------------------------------
# should_regrade gate
# ---------------------------------------------------------------------------


def test_fires_when_evidence_grew_after_grade(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    fire, reason = fr.should_regrade(p, recorded_score=0.5413, target=0.7437)
    assert fire is True
    assert "evidence_grew" in reason


def test_skips_when_grade_is_fresh(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 10, metrics_at=now)  # within margin
    fire, reason = fr.should_regrade(p, recorded_score=0.5413, target=0.7437)
    assert fire is False
    assert reason == "grade_is_fresh"


def test_skips_when_already_meets_target(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    fire, reason = fr.should_regrade(p, recorded_score=0.78, target=0.7437)
    assert fire is False
    assert reason == "already_meets_target"


def test_fires_when_no_recorded_grade(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=None, metrics_at=now)
    (p / "rubric_evaluation.json").unlink(missing_ok=True)
    fire, reason = fr.should_regrade(p, recorded_score=None, target=0.7437)
    assert fire is True


def test_skips_without_metrics(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    fire, reason = fr.should_regrade(tmp_path, recorded_score=0.5, target=0.74)
    assert fire is False
    assert reason == "no_metrics_on_disk"


def test_flag_disables(tmp_path, monkeypatch):
    monkeypatch.setenv(fr.ENV_FLAG, "0")
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    assert fr.should_regrade(p, recorded_score=0.54, target=0.74)[0] is False
    assert fr.is_enabled() is False


# ---------------------------------------------------------------------------
# converged-cell proxy
# ---------------------------------------------------------------------------


def test_converged_cell_count_excludes_chance(tmp_path):
    assert fr._converged_cell_count(_grid_metrics(13, 1)) == 13
    assert fr._converged_cell_count({"per_model": {}}) == 0
    assert fr._converged_cell_count({}) == 0


# ---------------------------------------------------------------------------
# maybe_regrade — adopt / keep semantics
# ---------------------------------------------------------------------------


def _report(score=0.5413, target=0.7437, verdict="partial"):
    return SimpleNamespace(
        rubric={"overall_score": score, "target_score": target, "meets_target": False},
        verdict=verdict,
    )


def _ctx(project_dir, fresh_score):
    # llm_client is opaque to maybe_regrade; score_reproduction is monkeypatched.
    return SimpleNamespace(project_dir=project_dir, llm_client=object(),
                           paper_hint_invariants=[])


def test_adopts_strictly_higher_regrade(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        lambda **kw: {"overall_score": 0.731, "target_score": 0.7437,
                      "graded": 22, "leaf_count": 22, "leaf_scores": [], "areas": []},
    )
    report = _report()
    fresh = fr.maybe_regrade(_ctx(p, 0.731), report)
    assert fresh is not None
    assert report.rubric["overall_score"] == pytest.approx(0.731)
    # Persisted for the report merge.
    saved = json.loads((p / "rubric_evaluation.json").read_text())
    assert saved["overall_score"] == pytest.approx(0.731)


def test_keeps_recorded_when_regrade_not_higher(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        lambda **kw: {"overall_score": 0.52, "target_score": 0.7437},
    )
    report = _report()
    assert fr.maybe_regrade(_ctx(p, 0.52), report) is None
    assert report.rubric["overall_score"] == pytest.approx(0.5413)  # untouched


def test_adopted_regrade_meeting_target_flips_meets(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now, target=0.60)
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        lambda **kw: {"overall_score": 0.731, "target_score": 0.60,
                      "leaf_scores": [], "areas": []},
    )
    report = _report(target=0.60)
    fresh = fr.maybe_regrade(_ctx(p, 0.731), report)
    assert fresh["meets_target"] is True
    assert report.rubric["meets_target"] is True


def test_skips_regrade_when_no_converged_cells(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    now = time.time()
    p = _project(tmp_path, graded_at=now - 9 * 3600, metrics_at=now)
    (p / "code" / "metrics.json").write_text(json.dumps({"per_model": {}}))
    os.utime(p / "code" / "metrics.json", (now, now))
    called = []
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        lambda **kw: called.append(1) or {"overall_score": 0.9},
    )
    assert fr.maybe_regrade(_ctx(p, 0.9), _report()) is None
    assert called == []  # no LLM call spent on empty evidence


def test_never_raises(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    bad_ctx = SimpleNamespace(project_dir="/nonexistent/xyz", llm_client=None)
    assert fr.maybe_regrade(bad_ctx, _report()) is None
