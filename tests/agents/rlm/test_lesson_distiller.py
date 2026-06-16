"""MUSE-lite per-paper negative lessons (FLAG-2, REPROLAB_NEGATIVE_LESSONS)."""
import json

import pytest

from backend.agents.rlm import lesson_distiller as ld


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("REPROLAB_NEGATIVE_LESSONS", "1")


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv("REPROLAB_NEGATIVE_LESSONS", raising=False)


def _run_dir(tmp_path, name, rows):
    d = tmp_path / name
    d.mkdir(parents=True)
    with (d / "experiment_runs.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return d


FAIL = {"success": False, "failure_class": "missing_module", "suggested_fix": "pip install trl"}
DOCKER = {"success": False, "failure_class": "dockerfile_invalid", "suggested_fix": "start with FROM"}


# --- off-state --------------------------------------------------------------

def test_off_state_noop(off, tmp_path):
    d = _run_dir(tmp_path, "r1", [FAIL])
    ld.mine_lessons(d, tmp_path, "2605.15155")
    assert not (tmp_path / "_lessons").exists()
    assert ld.negative_lessons_block(tmp_path, "2605.15155") == ""


def test_missing_arxiv_id_noop(on, tmp_path):
    d = _run_dir(tmp_path, "r1", [FAIL])
    ld.mine_lessons(d, tmp_path, None)
    assert not (tmp_path / "_lessons").exists()


# --- recurrence-gated promotion --------------------------------------------

def test_missing_module_needs_two_occurrences(on, tmp_path):
    aid = "2605.15155"
    ld.mine_lessons(_run_dir(tmp_path, "r1", [FAIL]), tmp_path, aid)
    assert ld.active_lessons(tmp_path, aid) == []  # 1 occ → not promoted
    ld.mine_lessons(_run_dir(tmp_path, "r2", [FAIL]), tmp_path, aid)
    active = ld.active_lessons(tmp_path, aid)
    assert len(active) == 1 and active[0]["failure_class"] == "missing_module"


def test_dockerfile_invalid_promotes_at_one(on, tmp_path):
    aid = "2605.15155"
    ld.mine_lessons(_run_dir(tmp_path, "r1", [DOCKER]), tmp_path, aid)
    active = ld.active_lessons(tmp_path, aid)
    assert len(active) == 1 and active[0]["failure_class"] == "dockerfile_invalid"


def test_only_correctable_classes_mined(on, tmp_path):
    aid = "x"
    rows = [{"success": False, "failure_class": "disk_exhausted", "suggested_fix": "free disk"}]
    ld.mine_lessons(_run_dir(tmp_path, "r1", rows), tmp_path, aid)
    store = json.loads((tmp_path / "_lessons" / "x.json").read_text())
    assert store == {}  # disk_exhausted is not agent-correctable


def test_successful_rows_ignored(on, tmp_path):
    aid = "x"
    rows = [{"success": True, "failure_class": "missing_module", "suggested_fix": "x"}]
    ld.mine_lessons(_run_dir(tmp_path, "r1", rows), tmp_path, aid)
    assert json.loads((tmp_path / "_lessons" / "x.json").read_text()) == {}


# --- retirement -------------------------------------------------------------

def test_staleness_retires_after_three_clean_runs(on, tmp_path):
    aid = "x"
    ld.mine_lessons(_run_dir(tmp_path, "r1", [FAIL]), tmp_path, aid)
    ld.mine_lessons(_run_dir(tmp_path, "r2", [FAIL]), tmp_path, aid)  # promoted
    assert ld.active_lessons(tmp_path, aid)
    # three runs that ran experiments but without the failure → retire
    other = [{"success": False, "failure_class": "syntax_error", "suggested_fix": "fix it"}]
    for i in range(3):
        ld.mine_lessons(_run_dir(tmp_path, f"c{i}", other), tmp_path, aid)
    store = json.loads((tmp_path / "_lessons" / "x.json").read_text())
    assert "missing_module" not in store  # retired at staleness>=3


def test_empty_run_does_not_accrue_staleness(on, tmp_path):
    aid = "x"
    ld.mine_lessons(_run_dir(tmp_path, "r1", [FAIL]), tmp_path, aid)
    ld.mine_lessons(_run_dir(tmp_path, "r2", [FAIL]), tmp_path, aid)
    # a run with NO experiment rows must not age the lesson
    for i in range(5):
        ld.mine_lessons(_run_dir(tmp_path, f"e{i}", []), tmp_path, aid)
    assert ld.active_lessons(tmp_path, aid)  # still active


# --- caps + formatting ------------------------------------------------------

def test_block_caps_and_format(on, tmp_path):
    aid = "x"
    # promote 7 distinct correctable classes (>MAX_LESSONS) twice each
    classes = list(ld.CORRECTABLE)
    for rnd in range(2):
        rows = [{"success": False, "failure_class": c, "suggested_fix": f"fix {c}"} for c in classes]
        ld.mine_lessons(_run_dir(tmp_path, f"r{rnd}", rows), tmp_path, aid)
    active = ld.active_lessons(tmp_path, aid)
    assert len(active) <= ld.MAX_LESSONS
    block = ld.negative_lessons_block(tmp_path, aid)
    assert "NEGATIVE LESSONS" in block
    assert block.count("\n- ") <= ld.MAX_LESSONS


def test_fix_truncated_to_max_len(on, tmp_path):
    aid = "x"
    longfix = "y" * 500
    rows = [{"success": False, "failure_class": "dockerfile_invalid", "suggested_fix": longfix}]
    ld.mine_lessons(_run_dir(tmp_path, "r1", rows), tmp_path, aid)
    store = json.loads((tmp_path / "_lessons" / "x.json").read_text())
    assert len(store["dockerfile_invalid"]["suggested_fix"]) <= ld.MAX_FIX_LEN
