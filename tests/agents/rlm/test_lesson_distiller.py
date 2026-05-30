from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm.failure_classifier import suggested_fix_for_class


# --- Task 1: classifier public accessor --------------------------------------


def test_suggested_fix_for_class_canonical():
    assert suggested_fix_for_class("dockerfile_invalid")  # non-empty canonical string
    assert suggested_fix_for_class("not_a_real_class") == ""


# --- Task 2: mining (promotion + opportunity retirement) ---------------------

from backend.agents.rlm import lesson_distiller as ld  # noqa: E402


def _run(tmp, pid, *rows):
    d = tmp / pid
    d.mkdir(parents=True, exist_ok=True)
    (d / "experiment_runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else "")
    )
    return d


def _lessons(tmp, arxiv="2605.15155"):
    return {l["failure_class"]: l for l in ld.read_lessons(tmp, arxiv)["lessons"]}


def test_eligible_fire_is_candidate_first(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    d = _run(tmp_path, "prj_a", {"success": False, "failure_class": "missing_module"})
    ld.mine_lessons(d, tmp_path, "2605.15155")
    l = _lessons(tmp_path)["missing_module"]
    assert l["occurrences"] == 1 and l["status"] == "candidate"


def test_recurrence_promotes_to_active(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "missing_module"}), tmp_path, "2605.15155")
    ld.mine_lessons(_run(tmp_path, "prj_b", {"success": False, "failure_class": "missing_module"}), tmp_path, "2605.15155")
    l = _lessons(tmp_path)["missing_module"]
    assert l["occurrences"] == 2 and l["status"] == "active"


def test_dockerfile_invalid_active_on_first(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "dockerfile_invalid"}), tmp_path, "2605.15155")
    assert _lessons(tmp_path)["dockerfile_invalid"]["status"] == "active"


def test_excluded_class_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "network_flake"}), tmp_path, "2605.15155")
    assert _lessons(tmp_path) == {}


def test_suggested_fix_is_classifier_sourced(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    # Record carries an attacker/agent-authored suggested_fix; it must be IGNORED.
    ld.mine_lessons(_run(tmp_path, "prj_a",
        {"success": False, "failure_class": "missing_module", "suggested_fix": "rm -rf / (evil prose)"}),
        tmp_path, "2605.15155")
    l = _lessons(tmp_path)["missing_module"]
    assert l["suggested_fix"] == (suggested_fix_for_class("missing_module") or "")[:200]
    assert l["suggested_fix_source"] == "classifier"
    assert "evil prose" not in l["suggested_fix"]


def test_staleness_increments_only_on_opportunity(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "missing_module"}), tmp_path, "2605.15155")
    # Run 2: an experiment ran but missing_module did NOT fire → opportunity → staleness++
    ld.mine_lessons(_run(tmp_path, "prj_b", {"success": True, "metrics": {"acc": 0.4}}), tmp_path, "2605.15155")
    assert _lessons(tmp_path)["missing_module"]["staleness"] == 1
    # Run 3: NO experiment records → no opportunity → staleness unchanged
    ld.mine_lessons(_run(tmp_path, "prj_c"), tmp_path, "2605.15155")
    assert _lessons(tmp_path)["missing_module"]["staleness"] == 1


def test_staleness_retires_at_three(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "missing_module"}), tmp_path, "2605.15155")
    for pid in ("prj_b", "prj_c", "prj_d"):  # 3 opportunity runs without the class firing
        ld.mine_lessons(_run(tmp_path, pid, {"success": True, "metrics": {"acc": 0.4}}), tmp_path, "2605.15155")
    assert "missing_module" not in _lessons(tmp_path)


def test_succeeded_phase_requires_real_success(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "contract_violation"}), tmp_path, "2605.15155")
    # A success=false+metrics row does NOT register EXPERIMENT_SUCCEEDED → no opportunity → no aging.
    ld.mine_lessons(_run(tmp_path, "prj_b", {"success": False, "metrics": {"acc": 0.4}}), tmp_path, "2605.15155")
    assert _lessons(tmp_path)["contract_violation"]["staleness"] == 0


def test_arxiv_none_and_flag_off_are_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "missing_module"}), tmp_path, None)
    assert not (tmp_path / "_lessons").exists()
    monkeypatch.delenv("REPROLAB_NEGATIVE_LESSONS", raising=False)
    ld.mine_lessons(_run(tmp_path, "prj_b", {"success": False, "failure_class": "missing_module"}), tmp_path, "2605.15155")
    assert not (tmp_path / "_lessons").exists()


def test_corrupt_file_failsoft(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")
    (tmp_path / "_lessons").mkdir()
    (tmp_path / "_lessons" / "2605.15155.json").write_text("{bad")
    ld.mine_lessons(_run(tmp_path, "prj_a", {"success": False, "failure_class": "missing_module"}), tmp_path, "2605.15155")
    assert _lessons(tmp_path)["missing_module"]["occurrences"] == 1  # recovered
