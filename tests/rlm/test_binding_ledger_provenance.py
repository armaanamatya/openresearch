"""binding.wrap_primitive per-row ledger provenance (audit 2026-06-10).

The evidence gate's success-compatible cross-check is only as good as the
stamps — these tests prove the wrapper writes "ok"/"failed"/"raised" on the
exit paths, end-to-end through the real wrap_primitive (including the
run_experiment contract guard, whose rejection is itself a "failed" stamp:
a guard-blocked call never ran an experiment and must not back a success row).
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.binding import wrap_primitive


def _last_row(ctx):
    assert ctx.cost_ledger.entries, "wrap_primitive must always append a row"
    return ctx.cost_ledger.entries[-1]


def test_success_result_stamps_ok(make_context, tmp_path):
    ctx = make_context(tmp_path)
    wrapped = wrap_primitive(
        "run_experiment",
        lambda code_path, *, ctx: {"success": True, "metrics": {"a": 1}},
        ctx,
    )

    wrapped("code/")

    row = _last_row(ctx)
    assert row.agent_id == "run_experiment"
    assert row.outcome == "ok"
    assert ctx.cost_ledger.session_success_compatible_count("run_experiment") == 1


def test_failure_shaped_result_stamps_failed(make_context, tmp_path):
    ctx = make_context(tmp_path)
    wrapped = wrap_primitive(
        "run_experiment",
        lambda code_path, *, ctx: {"success": False, "metrics": {}, "error": "exec died"},
        ctx,
    )

    wrapped("code/")

    assert _last_row(ctx).outcome == "failed"
    assert ctx.cost_ledger.session_success_compatible_count("run_experiment") == 0
    # ...while the total in-process count still sees the call (partial-cap tier).
    assert ctx.cost_ledger.session_call_count("run_experiment") == 1


def test_raising_primitive_stamps_raised(make_context, tmp_path):
    ctx = make_context(tmp_path)

    def _boom(code_path, *, ctx):
        raise RuntimeError("kaboom")

    wrapped = wrap_primitive("run_experiment", _boom, ctx)

    with pytest.raises(RuntimeError):
        wrapped("code/")

    assert _last_row(ctx).outcome == "raised"
    assert ctx.cost_ledger.session_success_compatible_count("run_experiment") == 0


def test_contract_guard_rejection_stamps_failed(make_context, tmp_path):
    """A guard-blocked call (missing code_path) never ran an experiment — its
    row must be failed-stamped so it can never back a forged success row."""
    ctx = make_context(tmp_path)
    called = []
    wrapped = wrap_primitive(
        "run_experiment",
        lambda code_path, *, ctx: called.append(1),
        ctx,
    )

    result = wrapped()  # no args → contract guard fires, primitive never runs

    assert called == []
    assert result["failure_class"] == "contract_guard"
    assert _last_row(ctx).outcome == "failed"
    assert ctx.cost_ledger.session_success_compatible_count("run_experiment") == 0
