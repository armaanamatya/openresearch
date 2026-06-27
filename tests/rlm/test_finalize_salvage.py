"""Finalize evidence-salvage re-grade for a FAILED run with a complete grid.

A run marked 'failed' because verify_against_rubric timed out / never ran can
still carry a COMPLETE trained grid on disk (the SDAR 22-cell grid whose 600s
verify cap expired → failed/not-scored). ``_finalize`` now fires
``regrade_and_emit`` on the failure path too; these pin that the earned score is
recovered and the verdict is set HONESTLY from the recovered score band, instead
of shipping ``failed / not scored``.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.rlm import finalize_regrade as fr


def _project_no_eval(tmp_path: Path) -> Path:
    """A completed grid on disk with NO rubric_evaluation.json (verify timed out)."""
    code = tmp_path / "code"
    code.mkdir(parents=True, exist_ok=True)
    (tmp_path / "generated_rubric.json").write_text(
        json.dumps({"source": "generated", "id": "r", "sub_tasks": []})
    )
    pm = {
        f"m{i}": {"search_qa": {"sdar": {"reward": 0.01 * i, "accuracy": 0.5}}}
        for i in range(6)
    }
    (code / "metrics.json").write_text(json.dumps({"status": "partial", "per_model": pm}))
    return tmp_path


def _failed_report():
    return SimpleNamespace(
        rubric={"overall_score": None, "target_score": None, "meets_target": None},
        verdict="failed",
    )


def _ctx(project_dir: Path):
    return SimpleNamespace(
        project_dir=project_dir, llm_client=object(), paper_hint_invariants=[]
    )


def _emit_capture():
    events: list = []
    return events, (lambda ev, payload: events.append((ev, payload)))


@pytest.mark.parametrize("fresh,expected_verdict", [
    (0.72, "reproduced"),   # >= 0.60
    (0.40, "partial"),      # 0.15 - 0.60
    (0.10, "failed"),       # < 0.15 — honest: a graded-but-low run stays failed
])
def test_salvage_sets_verdict_from_score_band(tmp_path, monkeypatch, fresh, expected_verdict):
    """A failed run with a complete grid and NO recorded grade is re-graded; its
    verdict is set from the recovered score band, and the score is populated."""
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    p = _project_no_eval(tmp_path)
    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction",
        lambda **kw: {"overall_score": fresh, "target_score": 0.70,
                      "graded": 6, "leaf_count": 6, "leaf_scores": [], "areas": []},
    )
    report = _failed_report()
    events, emit = _emit_capture()
    out = fr.regrade_and_emit(_ctx(p), report, emit)
    assert out is not None
    assert report.rubric["overall_score"] == pytest.approx(fresh)
    assert report.verdict == expected_verdict
    assert any(e[1].get("code") == "finalize_regrade_adopted" for e in events)


def test_salvage_noop_when_no_grid(tmp_path, monkeypatch):
    """A genuinely-empty failure (no metrics on disk) stays a no-op — no grading
    LLM call, verdict untouched."""
    monkeypatch.delenv(fr.ENV_FLAG, raising=False)
    (tmp_path / "generated_rubric.json").write_text(json.dumps({"id": "r", "sub_tasks": []}))
    called = {"n": 0}

    def _should_not_run(**kw):
        called["n"] += 1
        return {"overall_score": 0.9}

    monkeypatch.setattr(
        "backend.evals.paperbench.leaf_scorer.score_reproduction", _should_not_run
    )
    report = _failed_report()
    _events, emit = _emit_capture()
    out = fr.regrade_and_emit(_ctx(tmp_path), report, emit)
    assert out is None
    assert called["n"] == 0
    assert report.verdict == "failed"
