"""Prior-attempt evidence block — measured per-cell results carry forward.

Pins the 2026-06-10 All-CNN lesson: attempt N's working config (a_allcnn at
13.11% error) must reach attempt N+1's implementer, keyed across attempts even
though cell ids drift (seed suffixes, separator styles).
"""

from __future__ import annotations

import json

from backend.agents.rlm.prior_attempt_evidence import (
    build_evidence_block,
    is_enabled,
)


def _write_cell(root, attempt, run_id, cell, payload):
    d = root / "attempts" / attempt / "code" / "outputs" / run_id / cell
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(payload))


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("REPROLAB_PRIOR_ATTEMPT_EVIDENCE", raising=False)
    assert is_enabled() is False
    monkeypatch.setenv("REPROLAB_PRIOR_ATTEMPT_EVIDENCE", "1")
    assert is_enabled() is True
    monkeypatch.setenv("REPROLAB_PRIOR_ATTEMPT_EVIDENCE", "off")
    assert is_enabled() is False


def test_histories_join_across_attempts_despite_id_drift(tmp_path):
    # attempt 1 (older): plain id, paper-grade result
    _write_cell(tmp_path, "20260608T000000-000000-aaaaaa", "r1",
                "a_allcnn_cifar10_noaug",
                {"status": "ok", "test_error_pct": 13.11, "lr": 0.05})
    # attempt 2 (newer): seed-suffixed id, dead result with probed lr
    _write_cell(tmp_path, "20260609T000000-000000-bbbbbb", "r2",
                "a_allcnn__cifar10_noaug__s42",
                {"status": "ok", "test_error_pct": 90.0, "best_lr": 0.25})
    block = build_evidence_block(tmp_path)
    assert "a_allcnn_cifar10_noaug" in block
    # newest first, prior second — both metrics visible on one line
    line = next(ln for ln in block.splitlines() if "a_allcnn_cifar10_noaug" in ln)
    assert "[latest]" in line and "test_error_pct=90" in line
    assert "[prior]" in line and "test_error_pct=13.11" in line
    assert "lr=0.25" in line and "lr=0.05" in line


def test_worst_latest_cells_listed_first(tmp_path):
    _write_cell(tmp_path, "20260609T000000-000000-cccccc", "r1",
                "good_cell", {"status": "ok", "test_error_pct": 9.9})
    _write_cell(tmp_path, "20260609T000000-000000-cccccc", "r1",
                "dead_cell", {"status": "ok", "test_error_pct": 90.0})
    block = build_evidence_block(tmp_path)
    assert block.index("dead_cell") < block.index("good_cell")


def test_aggregates_and_smoke_dirs_skipped(tmp_path):
    _write_cell(tmp_path, "20260609T000000-000000-dddddd", "r1",
                "_cell_smoke", {"status": "ok", "test_error_pct": 1.0})
    # aggregate shape (has per_model) must not be summarised as a cell
    d = tmp_path / "attempts" / "20260609T000000-000000-dddddd" / "code" / "outputs" / "r1" / "agg"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"per_model": {"m": {}}, "status": "partial"}))
    assert build_evidence_block(tmp_path) == ""


def test_caps_respected(tmp_path):
    for i in range(40):
        _write_cell(tmp_path, "20260609T000000-000000-eeeeee", "r1",
                    f"cell_{i:02d}", {"status": "ok", "test_error_pct": 50.0 + i})
    block = build_evidence_block(tmp_path, max_cells=10)
    assert "(+30 more cells truncated)" in block
    assert len(block) <= 1810


def test_no_attempts_empty(tmp_path):
    assert build_evidence_block(tmp_path) == ""


def test_guidance_hook_integration(tmp_path, monkeypatch):
    from backend.agents.baseline_implementation import _compute_constraint_guidance

    _write_cell(tmp_path, "20260609T000000-000000-ffffff", "r1",
                "a_base_cifar10", {"status": "ok", "test_error_pct": 12.9, "lr": 0.05})
    monkeypatch.setenv("REPROLAB_PRIOR_ATTEMPT_EVIDENCE", "1")
    guidance = _compute_constraint_guidance("local", None, project_dir=tmp_path)
    assert "PRIOR-ATTEMPT MEASURED EVIDENCE" in guidance
    assert "a_base_cifar10" in guidance

    monkeypatch.delenv("REPROLAB_PRIOR_ATTEMPT_EVIDENCE")
    guidance_off = _compute_constraint_guidance("local", None, project_dir=tmp_path)
    assert "PRIOR-ATTEMPT MEASURED EVIDENCE" not in guidance_off
