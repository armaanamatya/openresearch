"""WS-B: external-validator finalize coverage.

When OPENRESEARCH_EXTERNAL_VALIDATOR is ON, every finalize path must stamp
final_report.json.validation with either a fresh verdict or an explicit
"missing"/"unavailable" marker — never a silent {}. The individual pieces
(the shared helper's own contract, the report.py chokepoint's stamping rules)
are covered in test_finalize_validation_panel.py and test_report_validation_stamp.py
respectively; this file covers the two seams those unit tests can't reach:

  1. `_finalize` no longer skips the panel on a failed run (the `if not
     run_failed:` guard removed) — pinned at the source level since driving
     `_finalize` itself hermetically needs a live RunContext + RLM result_obj.
  2. An end-to-end drive of `_hard_stop_with_report` (the real watchdog/SIGTERM
     finalizer) with the flag ON + a ctx carrying a validator_client, proving
     the report that lands on disk actually carries a stamped validation dict —
     this exact path (flag ON, ctx present, hard-stop) was previously untested.
"""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.agents.rlm import run


class _ExitCalled(Exception):
    def __init__(self, code: int) -> None:
        self.code = code


def test_finalize_calls_validation_panel_unconditionally():
    """WS-B: the validation panel call inside `_finalize` must not be gated behind
    `if not run_failed:` any more — a failed run's evidence deserves the critic too
    (mirrors `_finalize_fatal_primitive_abort`'s existing rationale). Source-pinned:
    `_finalize` needs a live RunContext + RLM result_obj to drive end-to-end, but the
    exact removed guard is a precise, easy-to-regress text pattern worth pinning
    directly.
    """
    src = inspect.getsource(run._finalize)
    assert "if not run_failed:\n        _run_finalize_validation_panel" not in src

    lines = src.splitlines()
    call_idx = next(
        i for i, ln in enumerate(lines)
        if "_run_finalize_validation_panel(ctx, report, project_dir)" in ln
    )
    prev_nonblank = next(
        ln for ln in reversed(lines[:call_idx]) if ln.strip() and not ln.strip().startswith("#")
    )
    assert not prev_nonblank.strip().startswith("if not run_failed"), (
        "the validation panel call must run whether or not the run failed"
    )


def test_hard_stop_stamps_validation_dict_end_to_end(monkeypatch, tmp_path):
    """WS-B highest-value test: drive the real `_hard_stop_with_report` finalizer
    (the watchdog/SIGTERM path) with the validator flag ON and a ctx carrying a
    validator_client. The shipped final_report.json must carry a non-empty,
    fresh-fingerprinted `validation` dict — not the pre-WS-B silent `{}`.

    The fake validator_client is a bare `object()`: `run_validation_panel` degrades
    any transport failure to `status="unavailable"` (fail-soft, no network call),
    so this stays hermetic while still exercising the real client-present branch.
    """
    monkeypatch.setenv("OPENRESEARCH_EXTERNAL_VALIDATOR", "1")

    def _raise_exit(code: int) -> None:
        raise _ExitCalled(code)

    monkeypatch.setattr(run.os, "_exit", _raise_exit)

    ctx = SimpleNamespace(
        validator_client=object(),
        role_selection=None,
        llm_client=None,
        cost_ledger=None,
        arxiv_id=None,
    )
    emit = MagicMock()

    with pytest.raises(_ExitCalled) as ei:
        run._hard_stop_with_report(
            project_dir=tmp_path,
            emit=emit,
            done=3,
            summary="partial summary",
            status_error="terminated",
            exit_code=143,
            ctx=ctx,
        )
    assert ei.value.code == 143

    report_path = tmp_path / "final_report.json"
    assert report_path.exists()
    result = json.loads(report_path.read_text())
    val = result.get("validation", {})
    assert val, "validation must be stamped (fresh verdict or explicit marker), never {}"
    assert val.get("status") in {"clean", "vetoed", "unavailable", "missing"}


def test_hard_stop_flag_off_leaves_validation_empty(monkeypatch, tmp_path):
    """Default-off contract sanity: with the flag unset, the hard-stop path's
    validation stays exactly `{}` (byte-identical to before WS-B)."""
    monkeypatch.delenv("OPENRESEARCH_EXTERNAL_VALIDATOR", raising=False)

    def _raise_exit(code: int) -> None:
        raise _ExitCalled(code)

    monkeypatch.setattr(run.os, "_exit", _raise_exit)

    ctx = SimpleNamespace(validator_client=object(), role_selection=None)
    emit = MagicMock()

    with pytest.raises(_ExitCalled):
        run._hard_stop_with_report(
            project_dir=tmp_path,
            emit=emit,
            done=3,
            summary="partial summary",
            status_error="terminated",
            exit_code=143,
            ctx=ctx,
        )

    result = json.loads((tmp_path / "final_report.json").read_text())
    assert result.get("validation", {}) == {}
