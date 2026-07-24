"""Tests for LiveRunService.resume_run (backend/services/events/live_runs.py).

Hermetic: subprocess.Popen is monkeypatched to a recording fake, mirroring
tests/services/events/test_live_runs_durable_seam.py's convention -- nothing
here spawns a real orchestrator process. A pid of 99999 (the same
"assume not a real running process on the test machine" convention used in
tests/services/events/test_run_liveness.py) stands in for a dead prior run.

The local (non-durable) subprocess path spawns
``[python, "-u", "-c", <generated script text>]`` -- there is no
``--execution-mode`` CLI flag. The resolved run config rides a single
embedded ``config = json.loads("<escaped-json>")`` line inside that script
text (see ``live_runs.py::_python_script``); ``_config_from_script`` below
extracts it back into a dict so a test can assert on the resolved value.
"""

from __future__ import annotations

import asyncio
import ast
import contextlib
import io
import json
import re
from pathlib import Path

import pytest

import backend.services.events.live_runs as lr_module
from backend.services.events.live_runs import FileLiveRunService, _DEFAULT_EXECUTION_MODE

_DEAD_PID = 99999
# Same convention as test_live_runs_durable_seam.py's _FAKE_PID: never a real
# pid, so the fire-and-forget _stderr_watchdog task this spawns exits on its
# first liveness check instead of polling a real, unrelated process forever.
_FAKE_SPAWNED_PID = 99999999
_CONFIG_RE = re.compile(r"config = json\.loads\((\".*?\")\)\n", re.DOTALL)


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.stdin = io.BytesIO()


def _recording_popen(calls: list[dict]):
    def _fake(argv, **kwargs):
        calls.append({"argv": argv, "kwargs": kwargs})
        return _FakeProcess(_FAKE_SPAWNED_PID)

    return _fake


async def _drain_stderr_watchdogs(project_id: str) -> None:
    name = f"stderr-watchdog-{project_id}"
    pending = [t for t in asyncio.all_tasks() if t.get_name() == name]
    for task in pending:
        task.cancel()
    for task in pending:
        with contextlib.suppress(asyncio.CancelledError):
            await task


def _write_status(runs_root: Path, project_id: str, payload: dict) -> None:
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True, exist_ok=True)
    full_payload = {"outputDir": str(run_dir), **payload}
    (run_dir / "demo_status.json").write_text(json.dumps(full_payload), encoding="utf-8")


def _config_from_popen_call(call: dict) -> dict:
    """Recover the resolved run ``config`` dict from a recorded Popen argv."""
    script_text = call["argv"][3]
    match = _CONFIG_RE.search(script_text)
    assert match is not None, "generated script must embed a config = json.loads(...) line"
    inner_json_string = ast.literal_eval(match.group(1))
    return json.loads(inner_json_string)


@pytest.mark.asyncio
async def test_resume_run_returns_none_for_unknown_project(tmp_path: Path) -> None:
    service = FileLiveRunService(runs_root=tmp_path)
    assert await service.resume_run("prj_does_not_exist") is None


@pytest.mark.asyncio
async def test_resume_run_is_a_noop_when_original_process_is_alive(tmp_path: Path, monkeypatch) -> None:
    """A live pid must never be re-spawned out from under itself."""
    import os

    _write_status(tmp_path, "prj_alive", {
        "projectId": "prj_alive", "status": "running", "pid": os.getpid(),
        "runMode": "rlm",
    })
    service = FileLiveRunService(runs_root=tmp_path)
    popen_calls: list[dict] = []
    monkeypatch.setattr(lr_module.subprocess, "Popen", _recording_popen(popen_calls))

    result = await service.resume_run("prj_alive")

    assert result is not None
    assert result.pid == os.getpid()
    assert popen_calls == []


@pytest.mark.asyncio
async def test_resume_run_execution_mode_falls_back_to_the_current_default_not_a_stale_literal(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression: a demo_status.json missing executionMode (any run predating
    the efficient->max default flip) must inherit StartRunRequest's ACTUAL
    current default, not a hardcoded stale "efficient" literal that can
    silently re-drift out of sync with a future default change."""
    _write_status(tmp_path, "prj_no_exec_mode", {
        "projectId": "prj_no_exec_mode", "status": "interrupted", "pid": _DEAD_PID,
        "runMode": "rlm",
        # executionMode deliberately absent.
    })
    service = FileLiveRunService(runs_root=tmp_path)
    popen_calls: list[dict] = []
    monkeypatch.setattr(lr_module.subprocess, "Popen", _recording_popen(popen_calls))

    await service.resume_run("prj_no_exec_mode")
    await _drain_stderr_watchdogs("prj_no_exec_mode")

    assert len(popen_calls) == 1
    resolved = _config_from_popen_call(popen_calls[0])["execution_mode"]
    assert resolved == _DEFAULT_EXECUTION_MODE
    assert resolved != "efficient", "must not silently downgrade to the stale hardcoded default"


@pytest.mark.asyncio
async def test_resume_run_preserves_an_explicit_execution_mode_from_the_prior_status(
    tmp_path: Path, monkeypatch
) -> None:
    """A run that WAS explicitly recorded with executionMode="efficient" keeps
    that choice on resume -- only the missing-key case should fall back to
    the current default."""
    _write_status(tmp_path, "prj_explicit_efficient", {
        "projectId": "prj_explicit_efficient", "status": "interrupted", "pid": _DEAD_PID,
        "runMode": "rlm", "executionMode": "efficient",
    })
    service = FileLiveRunService(runs_root=tmp_path)
    popen_calls: list[dict] = []
    monkeypatch.setattr(lr_module.subprocess, "Popen", _recording_popen(popen_calls))

    await service.resume_run("prj_explicit_efficient")
    await _drain_stderr_watchdogs("prj_explicit_efficient")

    assert _config_from_popen_call(popen_calls[0])["execution_mode"] == "efficient"


@pytest.mark.asyncio
async def test_resume_run_override_wins_over_stored_execution_mode(tmp_path: Path, monkeypatch) -> None:
    _write_status(tmp_path, "prj_override", {
        "projectId": "prj_override", "status": "interrupted", "pid": _DEAD_PID,
        "runMode": "rlm", "executionMode": "efficient",
    })
    service = FileLiveRunService(runs_root=tmp_path)
    popen_calls: list[dict] = []
    monkeypatch.setattr(lr_module.subprocess, "Popen", _recording_popen(popen_calls))

    await service.resume_run("prj_override", request_overrides={"executionMode": "max"})
    await _drain_stderr_watchdogs("prj_override")

    assert _config_from_popen_call(popen_calls[0])["execution_mode"] == "max"
