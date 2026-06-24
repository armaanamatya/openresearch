"""Safety tests for the zero-GPU A1 capture: pool_ready is the kill-trigger that
guarantees we stop BEFORE any run_experiment / GPU cell. It must be False until the
graded pool is fully on disk, and True only then.

Also covers the degenerate-loop early-exit (Task 5): when dashboard_events.jsonl
emits root_degenerate_refusal_loop before a candidate pool exists, run_safe_capture
must kill the run immediately and return {ok:False, verdict:root_degenerate}.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


from scripts.bes_a1_safe_capture import degenerate_event, pool_ready


def test_pool_not_ready_when_nothing_written(tmp_path: Path):
    assert pool_ready(tmp_path, n=3) is False


def test_pool_not_ready_with_candidates_but_no_graded_pool(tmp_path: Path):
    # snapshots exist but the pool hasn't been graded/persisted yet -> NOT safe to read,
    # and (more importantly) this is not yet the documented safe-kill point.
    for i in range(3):
        (tmp_path / "candidates" / f"rlm_impl_{i}").mkdir(parents=True)
    assert pool_ready(tmp_path, n=3) is False


def test_pool_not_ready_with_graded_pool_but_too_few_snapshots(tmp_path: Path):
    (tmp_path / "rlm_state").mkdir()
    (tmp_path / "rlm_state" / "bes_candidates.json").write_text("{}")
    (tmp_path / "candidates" / "rlm_impl_0").mkdir(parents=True)  # only 1 of 3
    assert pool_ready(tmp_path, n=3) is False


def test_pool_ready_when_graded_pool_and_all_snapshots_present(tmp_path: Path):
    (tmp_path / "rlm_state").mkdir()
    (tmp_path / "rlm_state" / "bes_candidates.json").write_text("{}")
    for i in range(3):
        (tmp_path / "candidates" / f"rlm_impl_{i}").mkdir(parents=True)
    assert pool_ready(tmp_path, n=3) is True


# ---------------------------------------------------------------------------
# degenerate_event unit tests (Task 5)
# ---------------------------------------------------------------------------

_DEGENERATE_LINE = json.dumps({
    "event": "run_warning",
    "timestamp": "2026-06-17T00:00:00Z",
    "level": "warn",
    "code": "root_degenerate_refusal_loop",
    "message": "degenerate loop detected",
    "signature": "no_experiment",
    "count": 3,
    "required_stage": "need_baseline",
    "stage": "need_baseline",
})

_ORDINARY_LINE = json.dumps({
    "event": "run_warning",
    "code": "forced_iteration",
    "message": "ordinary warning",
})


def _write_events(path: Path, *lines: str) -> None:
    """Write JSONL lines to dashboard_events.jsonl under path."""
    (path / "dashboard_events.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_degenerate_event_returns_dict_when_present(tmp_path: Path):
    _write_events(tmp_path, _ORDINARY_LINE, _DEGENERATE_LINE)
    ev = degenerate_event(tmp_path)
    assert ev is not None
    assert ev["code"] == "root_degenerate_refusal_loop"
    assert ev["required_stage"] == "need_baseline"
    assert ev["count"] == 3


def test_degenerate_event_returns_none_for_only_ordinary_warnings(tmp_path: Path):
    _write_events(tmp_path, _ORDINARY_LINE, _ORDINARY_LINE)
    assert degenerate_event(tmp_path) is None


def test_degenerate_event_returns_none_when_file_missing(tmp_path: Path):
    # No dashboard_events.jsonl written at all
    assert degenerate_event(tmp_path) is None


def test_degenerate_event_skips_malformed_line_and_finds_valid(tmp_path: Path):
    malformed = "{not valid json%%"
    _write_events(tmp_path, malformed, _DEGENERATE_LINE)
    ev = degenerate_event(tmp_path)
    assert ev is not None
    assert ev["code"] == "root_degenerate_refusal_loop"


def test_degenerate_event_skips_blank_lines(tmp_path: Path):
    content = "\n\n" + _DEGENERATE_LINE + "\n\n"
    (tmp_path / "dashboard_events.jsonl").write_text(content, encoding="utf-8")
    ev = degenerate_event(tmp_path)
    assert ev is not None
    assert ev["code"] == "root_degenerate_refusal_loop"


# ---------------------------------------------------------------------------
# run_safe_capture integration tests (Task 5)
# ---------------------------------------------------------------------------

def _make_fake_proc(pid: int = 99999, returncode: int = 1):
    """Return a MagicMock that mimics a Popen whose process already exited."""
    proc = MagicMock()
    proc.pid = pid
    proc.returncode = returncode
    # poll() returns non-None immediately so _kill_group skips os.killpg
    proc.poll.return_value = returncode
    proc.wait.return_value = returncode
    return proc


def _setup_fake_run_dir(tmp_path: Path, pid: int) -> Path:
    """Return the run_dir that run_safe_capture will use (runs/<project_id>)."""
    run_dir = tmp_path / "runs" / str(pid)
    run_dir.mkdir(parents=True)
    return run_dir


def _run_capture(tmp_path, fake_proc, *, extra_events="", timeout_s=3600, poll_s=0.0):
    """Drive run_safe_capture with the fake proc and a redirected REPO_ROOT."""
    import scripts.bes_a1_safe_capture as mod

    pid = fake_proc.pid
    run_dir = _setup_fake_run_dir(tmp_path, pid)

    if extra_events:
        (run_dir / "dashboard_events.jsonl").write_text(extra_events, encoding="utf-8")

    with (
        patch.object(mod, "REPO_ROOT", tmp_path),
        patch("subprocess.Popen", return_value=fake_proc),
        patch.object(mod, "regrade_candidates", return_value=[]),
        patch.object(mod, "summarize_regrades", return_value={"ok": True, "summary": "mock"}),
    ):
        result = mod.run_safe_capture(
            "1412.6980",
            n=3, k=1, sigma=0.02,
            timeout_s=timeout_s,
            project_id=str(pid),
            poll_s=poll_s,
        )
    return result, run_dir


def test_run_safe_capture_exits_early_on_degenerate_event(tmp_path: Path):
    """Wrapper must exit early and write a1_result.json when degenerate event seen."""
    fake_proc = _make_fake_proc(pid=55501)
    # Seed the degenerate event but NO pool (no bes_candidates.json, no candidates/)
    degenerate_events_content = _ORDINARY_LINE + "\n" + _DEGENERATE_LINE + "\n"

    result, run_dir = _run_capture(
        tmp_path, fake_proc,
        extra_events=degenerate_events_content,
        timeout_s=3600,
        poll_s=0.0,
    )

    # Must return root_degenerate verdict
    assert result["ok"] is False
    assert result["verdict"] == "root_degenerate"
    assert result["required_stage"] == "need_baseline"

    # a1_result.json must be written
    result_file = run_dir / "a1_result.json"
    assert result_file.is_file(), "a1_result.json was not written"
    written = json.loads(result_file.read_text())
    assert written["ok"] is False
    assert written["verdict"] == "root_degenerate"
    assert written["required_stage"] == "need_baseline"
    # ok is False here (not the regrade path's ok:True) proving we bailed BEFORE
    # regrade_candidates/summarize_regrades ran — the early-exit, not grading.


def test_run_safe_capture_keeps_polling_on_ordinary_warnings(tmp_path: Path):
    """Ordinary warnings in dashboard_events.jsonl must NOT trigger early exit."""
    fake_proc = _make_fake_proc(pid=55502)
    # Only ordinary warnings, no pool → process exits on its own → no-pool generic result
    ordinary_events = _ORDINARY_LINE + "\n" + _ORDINARY_LINE + "\n"

    result, _run_dir = _run_capture(
        tmp_path, fake_proc,
        extra_events=ordinary_events,
        timeout_s=3600,
        poll_s=0.0,
    )

    # Must NOT be root_degenerate — process exited, no pool
    assert result.get("verdict") != "root_degenerate"
    assert result["ok"] is False
    assert "killed_reason" in result
