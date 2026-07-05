"""MUSE-lite per-paper negative lessons (FLAG-2, OPENRESEARCH_NEGATIVE_LESSONS)."""
import json

import pytest

from backend.agents.rlm import lesson_distiller as ld


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_NEGATIVE_LESSONS", "1")


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_NEGATIVE_LESSONS", raising=False)


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


# --- hollow-lessons regression (2026-07-02) ---------------------------------
#
# Live campaign validation found cell_execution_error/preflight_blocked hit
# occurrences=2 but never surfaced a suggested_fix, so campaign directives
# carried memory_hints: []. Root cause: both classes (plus cell_smoke_failed)
# were absent from CORRECTABLE, so _scan_failures never added them to `seen`
# at all — no lesson was ever minted, regardless of how good the classifier's
# fix text was. Pin the fix using the REAL classify_failure() output (not a
# hand-typed fix) so this exercises the exact pipeline
# _persist_experiment_result feeds into experiment_runs.jsonl.

def test_campaign_hit_classes_are_correctable(on, tmp_path):
    assert {"cell_execution_error", "preflight_blocked", "cell_smoke_failed"} <= ld.CORRECTABLE


def test_cell_execution_error_mined_with_nonempty_fix(on, tmp_path):
    from backend.agents.rlm.failure_classifier import classify_failure

    aid = "x"
    _klass, fix = classify_failure({"success": False, "failure_class": "cell_execution_error"})
    row = {"success": False, "failure_class": "cell_execution_error", "suggested_fix": fix}
    ld.mine_lessons(_run_dir(tmp_path, "r1", [row]), tmp_path, aid)
    ld.mine_lessons(_run_dir(tmp_path, "r2", [row]), tmp_path, aid)  # occurrences=2 -> promoted

    active = ld.active_lessons(tmp_path, aid)
    assert len(active) == 1
    assert active[0]["failure_class"] == "cell_execution_error"
    assert active[0]["suggested_fix"].strip() != ""


def test_preflight_blocked_and_cell_smoke_failed_mined_together(on, tmp_path):
    """Mine all three campaign-hit classes in the SAME rounds (mirrors
    test_block_caps_and_format) so none accrues retirement staleness while
    the others are mined separately."""
    from backend.agents.rlm.failure_classifier import classify_failure

    aid = "x"
    guard_results = {
        "cell_execution_error": {"success": False, "failure_class": "cell_execution_error"},
        "preflight_blocked": {"success": False, "pre_flight_blocked": True},
        "cell_smoke_failed": {"success": False, "failure_class": "cell_smoke_failed"},
    }
    fixes: dict[str, str] = {}
    rows = []
    for expected_class, guard_result in guard_results.items():
        klass, fix = classify_failure(guard_result)
        assert klass == expected_class
        assert fix.strip()
        fixes[expected_class] = fix
        rows.append({"success": False, "failure_class": expected_class, "suggested_fix": fix})

    for rnd in range(2):  # occurrences=2 for every class -> all promoted
        ld.mine_lessons(_run_dir(tmp_path, f"r{rnd}", rows), tmp_path, aid)

    active = {a["failure_class"]: a for a in ld.active_lessons(tmp_path, aid)}
    for expected_class, fix in fixes.items():
        assert expected_class in active, f"{expected_class} lesson not promoted"
        assert active[expected_class]["suggested_fix"] == fix[: ld.MAX_FIX_LEN]
