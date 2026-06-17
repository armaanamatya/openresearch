"""Tests for the flag-gated OAuth auto-drive backstop (Task 6).

``OPENRESEARCH_OAUTH_AUTODRIVE=1`` (default OFF, experimental) turns the
degenerate-refusal-loop callback from an *early-abort* (Task 4) into a *drive*:
when an oauth root degenerates on an implementable lifecycle stage, the harness
marks the event (``root_autodrive`` run_warning + ``rlm_state/root_autodrive.json``)
and drives exactly ONE missing step (``implement_baseline`` /
``build_environment`` / ``run_experiment``) itself, then hands control back to
the root.

Default-OFF MUST be byte-for-byte the Task-4 early-abort behaviour; the flag-ON
path must be inert for NON-oauth roots and NON-drivable stages (emit-only).

These tests use FAKE tools (a dict of ``{"name": {"tool": <recorder>}}``) so NO
real primitive executes — the recorder only records its name/args.  The real
``implement_baseline(plan, *, ctx)`` signature is documented inline; the v1
no-plan dispatch issues a structured ``recommend_next_tool(situation=...)`` step
(no assembled plan is persisted to disk to reconstruct), which is the "one final
structured step" the plan permits.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from backend.agents.rlm.forced_iteration import ForcedIterationPolicy
from backend.agents.rlm.run import (
    _autodrive_one_step,
    _make_degenerate_loop_callback,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fake_ctx(tmp_path, *, remaining_s: float | None = 3600.0) -> Any:
    ctx = MagicMock()
    ctx.remaining_s.return_value = remaining_s
    ctx._terminal_stop_reason = None
    ctx.project_dir = tmp_path
    return ctx


def _recorder(calls: list, name: str):
    def _tool(*args, **kwargs):
        calls.append((name, args, kwargs))
        return {"ok": True}

    return _tool


def _fake_tools(calls: list) -> dict:
    names = [
        "implement_baseline",
        "build_environment",
        "run_experiment",
        "plan_reproduction",
        "recommend_next_tool",
    ]
    return {n: {"tool": _recorder(calls, n)} for n in names}


def _payload(stage: str = "need_baseline") -> dict:
    return {"signature": "no_experiment", "count": 3, "required_stage": stage}


# ---------------------------------------------------------------------------
# Flag OFF → Task-4 early-abort only; NO tool called, NO marker written.
# ---------------------------------------------------------------------------


def test_flag_off_early_abort_no_drive(tmp_path) -> None:
    emitted: list[dict] = []
    calls: list = []
    ctx = _fake_ctx(tmp_path)
    policy = ForcedIterationPolicy(min_iterations=2)

    cb = _make_degenerate_loop_callback(
        emit=emitted.append,
        ctx=ctx,
        policy=policy,
        autodrive_enabled=False,
        tools=_fake_tools(calls),
        oauth_root=True,
    )
    cb(_payload())

    # Task-4 early-abort path: terminal stop marked, NO primitive driven.
    assert ctx._terminal_stop_reason is not None
    assert ctx._terminal_stop_reason["failure_class"] == "root_degenerate_loop"
    assert calls == []
    assert not (tmp_path / "rlm_state" / "root_autodrive.json").exists()
    assert not any(e.get("code") == "root_autodrive" for e in emitted)


# ---------------------------------------------------------------------------
# Flag ON + need_baseline + oauth_root → ONE implementation step, control back.
# ---------------------------------------------------------------------------


def test_flag_on_need_baseline_drives_one_step(tmp_path) -> None:
    emitted: list[dict] = []
    calls: list = []
    ctx = _fake_ctx(tmp_path)
    policy = ForcedIterationPolicy(min_iterations=2)

    cb = _make_degenerate_loop_callback(
        emit=emitted.append,
        ctx=ctx,
        policy=policy,
        autodrive_enabled=True,
        tools=_fake_tools(calls),
        oauth_root=True,
    )
    cb(_payload("need_baseline"))

    # Exactly ONE structured directive (no loop): v1 issues recommend_next_tool
    # naming implement_baseline as the concrete next call.
    assert [c[0] for c in calls] == ["recommend_next_tool"]
    assert "implement_baseline" in calls[0][2]["situation"]
    assert len(calls) == 1

    # Control returns to the root: NO early-abort / terminal stop.
    assert ctx._terminal_stop_reason is None
    assert policy._terminal_failure_class is None

    # Marker + event written.
    assert (tmp_path / "rlm_state" / "root_autodrive.json").exists()
    assert any(e.get("code") == "root_autodrive" for e in emitted)


def test_flag_on_need_environment_directs_to_build_environment(tmp_path) -> None:
    # v1: the harness cannot reconstruct build_environment's env_spec arg, so it
    # issues ONE structured recommend_next_tool directive naming build_environment.
    emitted: list[dict] = []
    calls: list = []
    ctx = _fake_ctx(tmp_path)
    policy = ForcedIterationPolicy(min_iterations=2)

    cb = _make_degenerate_loop_callback(
        emit=emitted.append,
        ctx=ctx,
        policy=policy,
        autodrive_enabled=True,
        tools=_fake_tools(calls),
        oauth_root=True,
    )
    cb(_payload("need_environment"))

    assert [c[0] for c in calls] == ["recommend_next_tool"]
    assert "build_environment" in calls[0][2]["situation"]
    assert ctx._terminal_stop_reason is None


def test_flag_on_need_experiment_directs_to_run_experiment(tmp_path) -> None:
    # v1: the harness cannot reconstruct run_experiment's code_path/env_id args,
    # so it issues ONE structured recommend_next_tool directive naming run_experiment.
    emitted: list[dict] = []
    calls: list = []
    ctx = _fake_ctx(tmp_path)
    policy = ForcedIterationPolicy(min_iterations=2)

    cb = _make_degenerate_loop_callback(
        emit=emitted.append,
        ctx=ctx,
        policy=policy,
        autodrive_enabled=True,
        tools=_fake_tools(calls),
        oauth_root=True,
    )
    cb(_payload("need_experiment"))

    assert [c[0] for c in calls] == ["recommend_next_tool"]
    assert "run_experiment" in calls[0][2]["situation"]
    assert ctx._terminal_stop_reason is None


# ---------------------------------------------------------------------------
# Terminal / wall-clock floor DISABLE auto-drive (same precedence as Task 4).
# ---------------------------------------------------------------------------


def test_wall_clock_floor_disables_drive(tmp_path) -> None:
    emitted: list[dict] = []
    calls: list = []
    ctx = _fake_ctx(tmp_path, remaining_s=30.0)
    policy = ForcedIterationPolicy(min_iterations=2)

    cb = _make_degenerate_loop_callback(
        emit=emitted.append,
        ctx=ctx,
        policy=policy,
        autodrive_enabled=True,
        tools=_fake_tools(calls),
        oauth_root=True,
    )
    cb(_payload())

    assert calls == []
    assert not (tmp_path / "rlm_state" / "root_autodrive.json").exists()


def test_existing_terminal_stop_disables_drive(tmp_path) -> None:
    emitted: list[dict] = []
    calls: list = []
    ctx = _fake_ctx(tmp_path)
    ctx._terminal_stop_reason = {"kind": "something_else"}
    policy = ForcedIterationPolicy(min_iterations=2)

    cb = _make_degenerate_loop_callback(
        emit=emitted.append,
        ctx=ctx,
        policy=policy,
        autodrive_enabled=True,
        tools=_fake_tools(calls),
        oauth_root=True,
    )
    cb(_payload())

    assert calls == []
    assert not (tmp_path / "rlm_state" / "root_autodrive.json").exists()


# ---------------------------------------------------------------------------
# Gate: non-oauth root + flag ON → emit-only, no drive.
# ---------------------------------------------------------------------------


def test_non_oauth_root_flag_on_emit_only(tmp_path) -> None:
    emitted: list[dict] = []
    calls: list = []
    ctx = _fake_ctx(tmp_path)
    policy = ForcedIterationPolicy(min_iterations=2)

    cb = _make_degenerate_loop_callback(
        emit=emitted.append,
        ctx=ctx,
        policy=policy,
        autodrive_enabled=True,
        tools=_fake_tools(calls),
        oauth_root=False,
    )
    cb(_payload())

    # No drive, no marker — flag-ON is inert for a non-oauth root. (And no
    # early-abort either: the AUTODRIVE-ON branch never falls into Task-4's
    # terminal-stop path.)
    assert calls == []
    assert not (tmp_path / "rlm_state" / "root_autodrive.json").exists()
    assert ctx._terminal_stop_reason is None


# ---------------------------------------------------------------------------
# Gate: non-drivable stage + flag ON → emit-only, no drive.
# ---------------------------------------------------------------------------


def test_non_drivable_stage_flag_on_emit_only(tmp_path) -> None:
    for stage in ("need_verification", "can_finalize"):
        emitted: list[dict] = []
        calls: list = []
        ctx = _fake_ctx(tmp_path)
        policy = ForcedIterationPolicy(min_iterations=2)

        cb = _make_degenerate_loop_callback(
            emit=emitted.append,
            ctx=ctx,
            policy=policy,
            autodrive_enabled=True,
            tools=_fake_tools(calls),
            oauth_root=True,
        )
        cb(_payload(stage))

        assert calls == [], stage
        assert not (tmp_path / "rlm_state" / "root_autodrive.json").exists(), stage


# ---------------------------------------------------------------------------
# Gate: no tools available + flag ON → emit-only, no drive.
# ---------------------------------------------------------------------------


def test_no_tools_flag_on_emit_only(tmp_path) -> None:
    emitted: list[dict] = []
    ctx = _fake_ctx(tmp_path)
    policy = ForcedIterationPolicy(min_iterations=2)

    cb = _make_degenerate_loop_callback(
        emit=emitted.append,
        ctx=ctx,
        policy=policy,
        autodrive_enabled=True,
        tools=None,
        oauth_root=True,
    )
    cb(_payload())

    assert not (tmp_path / "rlm_state" / "root_autodrive.json").exists()
    assert ctx._terminal_stop_reason is None


# ---------------------------------------------------------------------------
# Marker contents + module-level helper directly.
# ---------------------------------------------------------------------------


def test_marker_records_stage_and_payload(tmp_path) -> None:
    emitted: list[dict] = []
    calls: list = []
    ctx = _fake_ctx(tmp_path)

    _autodrive_one_step(
        stage="need_baseline",
        tools=_fake_tools(calls),
        ctx=ctx,
        emit=emitted.append,
        payload=_payload("need_baseline"),
    )

    marker = tmp_path / "rlm_state" / "root_autodrive.json"
    assert marker.exists()
    data = json.loads(marker.read_text())
    assert data["stage"] == "need_baseline"
    assert data["required_stage"] == "need_baseline"
    assert data["signature"] == "no_experiment"
    assert data["count"] == 3


def test_drive_dispatch_failure_is_failsoft(tmp_path) -> None:
    """A raising drive tool must not propagate; marker + event still written."""
    emitted: list[dict] = []

    def _boom(*_a, **_k):
        raise RuntimeError("drive broke")

    tools = {"implement_baseline": {"tool": _boom}, "recommend_next_tool": {"tool": _boom}}
    ctx = _fake_ctx(tmp_path)

    # Must not raise.
    _autodrive_one_step(
        stage="need_baseline",
        tools=tools,
        ctx=ctx,
        emit=emitted.append,
        payload=_payload("need_baseline"),
    )

    assert (tmp_path / "rlm_state" / "root_autodrive.json").exists()
    assert any(e.get("code") == "root_autodrive" for e in emitted)
