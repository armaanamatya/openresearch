"""binding.wrap_primitive lifecycle ledger wire (P1.2).

Verifies that the lifecycle ledger sidecar records the right outcome for every
exit path through wrap_primitive, that the flag gate is respected (default-OFF
leaves no files), and that project_inputs redaction holds end-to-end through
the wire (raw paper text / canary strings never appear on disk).

Harness mirrors test_binding_ledger_provenance.py: uses make_context / tmp_path
fixtures from conftest.py and calls wrap_primitive directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.rlm.binding import wrap_primitive
from backend.agents.rlm.lifecycle_ledger import read_records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ledger_path(project_dir: Path) -> Path:
    return project_dir / "rlm_state" / "lifecycle" / "ledger.jsonl"


def _records(ctx) -> list:
    """Read lifecycle records for this run's project dir."""
    return read_records(ctx.project_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_success_result_writes_ok_record(make_context, tmp_path, monkeypatch):
    """Flag ON + success result → exactly one record with outcome='ok'."""
    monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
    ctx = make_context(tmp_path)

    wrapped = wrap_primitive(
        "run_experiment",
        lambda code_path, *, ctx: {"success": True, "metrics": {"reward": 0.8}},
        ctx,
    )
    wrapped("code/")

    records = _records(ctx)
    assert len(records) == 1, f"expected 1 record, got {len(records)}"
    assert records[0].outcome == "ok"
    assert records[0].primitive == "run_experiment"


def test_failure_result_writes_failed_record(make_context, tmp_path, monkeypatch):
    """Flag ON + failure-shaped result → outcome='failed', never 'ok'."""
    monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
    ctx = make_context(tmp_path)

    wrapped = wrap_primitive(
        "run_experiment",
        lambda code_path, *, ctx: {
            "success": False,
            "error": "training crashed",
            "metrics": {},
        },
        ctx,
    )
    wrapped("code/")

    records = _records(ctx)
    assert len(records) == 1
    assert records[0].outcome == "failed"
    assert records[0].outcome != "ok"


def test_retryable_hang_envelope_writes_timeout(make_context, tmp_path, monkeypatch):
    """The retryable-hang envelope (outcome=retryable, error=primitive_hung) → outcome='timeout'."""
    monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
    ctx = make_context(tmp_path)

    # Return the exact retryable-hang envelope the timeout path produces.
    retryable_result = {
        "outcome": "retryable",
        "error": "primitive_hung",
        "primitive": "run_experiment",
        "wall_clock_s": 1800,
        "orphan_groups_killed": 0,
    }
    wrapped = wrap_primitive(
        "run_experiment",
        lambda code_path, *, ctx: retryable_result,
        ctx,
    )
    wrapped("code/")

    records = _records(ctx)
    assert len(records) == 1
    assert records[0].outcome == "timeout"
    assert records[0].outcome != "ok"


def test_partial_timeout_failure_writes_timeout(make_context, tmp_path, monkeypatch):
    """A partial_timeout failure-class result → outcome='timeout', not 'failed'."""
    monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
    ctx = make_context(tmp_path)

    wrapped = wrap_primitive(
        "run_experiment",
        lambda code_path, *, ctx: {
            "success": False,
            "failure_class": "partial_timeout",
            "partial_timeout": True,
            "error": "wall clock exceeded",
            "metrics": {"reward": 0.3},
        },
        ctx,
    )
    wrapped("code/")

    records = _records(ctx)
    assert len(records) == 1
    assert records[0].outcome == "timeout"
    assert records[0].outcome != "ok"


def test_flag_off_writes_no_ledger_dir(make_context, tmp_path, monkeypatch):
    """Flag OFF (default) → no rlm_state/lifecycle/ directory is created at all."""
    monkeypatch.delenv("OPENRESEARCH_LIFECYCLE_LEDGER", raising=False)
    ctx = make_context(tmp_path)

    wrapped = wrap_primitive(
        "run_experiment",
        lambda code_path, *, ctx: {"success": True, "metrics": {"reward": 1.0}},
        ctx,
    )
    wrapped("code/")

    assert not _ledger_path(ctx.project_dir).exists(), (
        "lifecycle/ledger.jsonl must not exist when OPENRESEARCH_LIFECYCLE_LEDGER is unset"
    )
    # Confirm no lifecycle directory was created at all
    lifecycle_dir = ctx.project_dir / "rlm_state" / "lifecycle"
    assert not lifecycle_dir.exists(), (
        "rlm_state/lifecycle/ dir must not exist when flag is OFF"
    )


def test_redaction_canary_not_in_inputs_projection(make_context, tmp_path, monkeypatch):
    """project_inputs redaction: a canary string in kwargs must NOT appear in the
    on-disk inputs_projection.  This guards the security property end-to-end
    through the wire (not just in the lifecycle_ledger unit test)."""
    monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
    ctx = make_context(tmp_path)

    canary = "CANARY_PAPER_TEXT_DO_NOT_LOG"

    # understand_section is not a projected primitive → project_inputs returns {}
    # but we use plan_reproduction so the projection path is exercised:
    # section_text is NOT in plan_reproduction's projection (only section_ids and
    # hparam_keys are emitted).
    wrapped = wrap_primitive(
        "plan_reproduction",
        lambda section_text, section_ids, *, ctx: {
            "success": True,
            "method_spec": "some spec",
        },
        ctx,
    )
    wrapped(canary, ["s1", "s2"])

    records = _records(ctx)
    assert len(records) == 1
    rec = records[0]
    # The canary must not appear anywhere in inputs_projection
    projection_str = str(rec.inputs_projection)
    assert canary not in projection_str, (
        f"Canary found in inputs_projection: {projection_str!r}"
    )
    # The on-disk JSONL must also not contain the canary
    ledger_text = _ledger_path(ctx.project_dir).read_text(encoding="utf-8")
    assert canary not in ledger_text, (
        f"Canary found in on-disk ledger: {ledger_text!r}"
    )


def test_raised_primitive_writes_raised_record(make_context, tmp_path, monkeypatch):
    """A primitive that RAISES records outcome='raised' (then the exception re-raises).

    Codex review fix: the post-validation append is unreachable from the except
    path, so a raised primitive would otherwise be absent from the ledger entirely.
    """
    monkeypatch.setenv("OPENRESEARCH_LIFECYCLE_LEDGER", "1")
    ctx = make_context(tmp_path)

    def _boom(code_path, *, ctx):
        raise ValueError("kaboom")

    wrapped = wrap_primitive("run_experiment", _boom, ctx)
    with pytest.raises(ValueError):
        wrapped("code/")

    records = _records(ctx)
    assert len(records) == 1
    assert records[0].outcome == "raised"
    assert records[0].primitive == "run_experiment"
    # Only the value-free exception TYPE name is stored, never a message/value.
    assert records[0].outputs_pointer.get("error_type") == "ValueError"
