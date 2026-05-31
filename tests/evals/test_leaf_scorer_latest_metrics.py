"""Scorer reads the LATEST experiment's metrics, and the grader SEES them
(2026-05-31 fix).

Two bugs hid a successful result behind a stale/absent one:
1. `_detect_data_unavailable_leaves` (and evidence) selected the
   lexicographically-FIRST `code/outputs/<run-id>/metrics.json` among the many
   per-experiment dirs a run accumulates — an arbitrary stale/superseded result
   (e.g. an early SDAR-loses-to-GRPO attempt) rather than the latest.
2. `_gather_evidence` never included metrics.json at all (it's not a priority
   code extension), so the LLM grader graded result-match / experiment leaves
   against the CODE only, never the measured outcome — scoring ~0 even when the
   run succeeded and SDAR beat GRPO.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from backend.evals.paperbench.leaf_scorer import _gather_evidence, _latest_metrics_path


def _write_metrics(run_dir: Path, run_id: str, payload: dict, mtime: float) -> Path:
    d = run_dir / "code" / "outputs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / "metrics.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(p, (mtime, mtime))
    return p


def test_latest_metrics_picks_newest_not_first_alphabetical(tmp_path: Path):
    # 'a_old' sorts first but is OLDER; 'z_new' sorts last but is NEWER.
    _write_metrics(tmp_path, "a_old_stale", {"comparison": {"m": {"delta": -0.02}}}, mtime=1000)
    _write_metrics(tmp_path, "z_new_good", {"comparison": {"m": {"delta": +0.05}}}, mtime=2000)
    latest = _latest_metrics_path(tmp_path)
    assert latest is not None
    data = json.loads(latest.read_text())
    assert data["comparison"]["m"]["delta"] == 0.05  # the newer, winning result


def test_latest_metrics_considers_top_level_code_metrics(tmp_path: Path):
    _write_metrics(tmp_path, "run1", {"k": "outputs"}, mtime=1000)
    top = tmp_path / "code" / "metrics.json"
    top.write_text(json.dumps({"k": "top"}), encoding="utf-8")
    os.utime(top, (3000, 3000))  # newest
    latest = _latest_metrics_path(tmp_path)
    assert json.loads(latest.read_text())["k"] == "top"


def test_latest_metrics_skips_empty_in_progress_newest(tmp_path: Path):
    # An in-progress/just-created experiment dir is NEWEST but carries no results;
    # it must not shadow the most recent results-bearing metrics (else both the
    # result AND the scope declaration are lost).
    _write_metrics(tmp_path, "good_complete", {
        "per_model": {"qwen3_1_7b": {}, "qwen2_5_3b": {}},
        "comparison": {"qwen3_1_7b": {"delta": 0.05}},
        "scope": {"gaps": [{"item": "alfworld"}]},
    }, mtime=2000)
    _write_metrics(tmp_path, "zzz_in_progress", {"status": "running"}, mtime=3000)  # newest, empty
    latest = _latest_metrics_path(tmp_path)
    data = json.loads(latest.read_text())
    assert data.get("comparison")  # picked the results-bearing one, not the empty newest


def test_latest_metrics_none_when_absent(tmp_path: Path):
    (tmp_path / "code").mkdir(parents=True)
    assert _latest_metrics_path(tmp_path) is None


def test_gather_evidence_includes_latest_metrics(tmp_path: Path):
    (tmp_path / "code").mkdir(parents=True)
    _write_metrics(tmp_path, "exp_final", {
        "status": "completed",
        "comparison": {"qwen3_1_7b": {"sdar_f1": 0.13, "grpo_f1": 0.08, "delta": 0.05}},
    }, mtime=2000)
    evidence = _gather_evidence(tmp_path)
    assert "latest experiment metrics.json" in evidence
    assert "sdar_f1" in evidence and "0.05" in evidence  # the grader now SEES the win


def test_gather_evidence_uses_newest_metrics_in_grader_text(tmp_path: Path):
    (tmp_path / "code").mkdir(parents=True)
    _write_metrics(tmp_path, "a_old", {"comparison": {"m": {"delta": -0.02}}}, mtime=1000)
    _write_metrics(tmp_path, "z_new", {"comparison": {"m": {"delta": 0.07}}}, mtime=2000)
    evidence = _gather_evidence(tmp_path)
    assert "0.07" in evidence       # newest result surfaced
    assert "-0.02" not in evidence  # stale result not surfaced
