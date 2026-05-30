from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm import lesson_distiller as ld


def _seed(tmp, *lessons, arxiv="2605.15155"):
    (tmp / "_lessons").mkdir(parents=True, exist_ok=True)
    (tmp / "_lessons" / f"{arxiv}.json").write_text(
        json.dumps({"version": "v1", "arxiv_id": arxiv, "lessons": list(lessons)})
    )


def test_block_empty_when_off(tmp_path, monkeypatch):
    monkeypatch.delenv("REPROLAB_NEGATIVE_LESSONS", raising=False)
    _seed(tmp_path, {"failure_class": "dockerfile_invalid", "suggested_fix": "x",
                     "status": "active", "occurrences": 2})
    assert ld.render_block(tmp_path, "2605.15155") == ""


def test_block_empty_without_arxiv(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    assert ld.render_block(tmp_path, None) == ""


def test_block_only_active(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    _seed(tmp_path,
        {"failure_class": "missing_module", "suggested_fix": "add to requirements",
         "status": "candidate", "occurrences": 1},
        {"failure_class": "dockerfile_invalid", "suggested_fix": "FROM must be first",
         "status": "active", "occurrences": 3})
    block = ld.render_block(tmp_path, "2605.15155")
    assert "dockerfile_invalid" in block and "missing_module" not in block
    assert "seen 3" in block


def test_block_capped_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    lessons = [{"failure_class": f"c{i}", "suggested_fix": "y" * 500,
                "status": "active", "occurrences": i} for i in range(10)]
    _seed(tmp_path, *lessons)
    block = ld.render_block(tmp_path, "2605.15155")
    assert block.count("\n- ") <= 5  # K=5 cap (list-item count)
    assert all(len(line) < 260 for line in block.splitlines())  # 200-char fix bound + tag


def test_injection_block_wired_into_constraint_guidance(tmp_path, monkeypatch):
    """The implementer guidance assembler surfaces active lessons when enabled."""
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    from backend.agents.baseline_implementation import _compute_constraint_guidance
    # project_dir is runs_root/<project_id>; the lessons file lives at runs_root/_lessons/.
    runs_root = tmp_path
    project_dir = runs_root / "prj_x"
    project_dir.mkdir(parents=True)
    _seed(runs_root,
        {"failure_class": "dockerfile_invalid", "suggested_fix": "FROM must be first",
         "status": "active", "occurrences": 3})
    guidance = _compute_constraint_guidance(
        sandbox_mode="local", gpu_mode=None, project_dir=project_dir, arxiv_id="2605.15155",
    )
    assert "NEGATIVE LESSONS FROM PRIOR RUNS" in guidance
    assert "dockerfile_invalid" in guidance
