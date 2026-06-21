"""Tests for the empty-code-block degenerate-loop early-abort (run.py).

A chat-aligned root that emits only prose (no ```repl``` block) would churn
to the rlm iteration cap before this fix.  ``_FatalBackendGateLogger.log``
now calls ``_register_iteration_progress(has_code_block)`` after each
``super().log()``.  When N consecutive iterations carry no code block the
helper raises ``_FatalPrimitiveAbort`` with ``failure_class="root_degenerate_loop"``
so the run finalises fast via the existing abort path.

Tested via ``_register_iteration_progress`` directly — no real
``RLMIteration`` or live run needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.agents.rlm.forced_iteration import (
    ForcedIterationPolicy,
    forced_iteration_policy,
)
from backend.agents.rlm.run import (
    _FatalBackendGateLogger,
    _FatalPrimitiveAbort,
)


def _make_logger() -> _FatalBackendGateLogger:
    """Minimal ``_FatalBackendGateLogger`` with no-op emit/checkpointer."""
    return _FatalBackendGateLogger(
        emit=lambda _e: None,
        checkpointer=MagicMock(),
    )


# ---------------------------------------------------------------------------
# _register_iteration_progress — streak logic
# ---------------------------------------------------------------------------


def test_two_empty_iterations_do_not_raise(monkeypatch) -> None:
    """Streak below threshold must NOT raise."""
    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "3")
    logger = _make_logger()

    logger._register_iteration_progress(False)
    logger._register_iteration_progress(False)
    # No exception — streak is 2, threshold is 3.
    assert logger._empty_iter_streak == 2


def test_third_consecutive_empty_iteration_raises(monkeypatch) -> None:
    """Third consecutive empty iteration trips the threshold and raises."""
    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "3")
    logger = _make_logger()

    logger._register_iteration_progress(False)
    logger._register_iteration_progress(False)
    with pytest.raises(_FatalPrimitiveAbort) as exc_info:
        logger._register_iteration_progress(False)

    abort = exc_info.value
    assert abort.result["failure_class"] == "root_degenerate_loop"
    assert "3" in abort.result["error"]
    assert "suggested_fix" in abort.result


def test_code_block_resets_streak(monkeypatch) -> None:
    """A True (has_code_block) call resets the streak; the threshold is never reached."""
    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "3")
    logger = _make_logger()

    logger._register_iteration_progress(False)
    logger._register_iteration_progress(False)
    logger._register_iteration_progress(True)   # reset
    logger._register_iteration_progress(False)
    logger._register_iteration_progress(False)  # streak == 2 again; no raise

    assert logger._empty_iter_streak == 2


def test_reset_prevents_abort_at_threshold(monkeypatch) -> None:
    """F, F, True, F, F — the True resets; threshold 3 is never reached."""
    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "3")
    logger = _make_logger()

    for has_code in (False, False, True, False, False):
        logger._register_iteration_progress(has_code)  # must not raise

    assert logger._empty_iter_streak == 2


# ---------------------------------------------------------------------------
# policy callback wiring — None policy is safe
# ---------------------------------------------------------------------------


def test_none_policy_does_not_crash_on_abort(monkeypatch) -> None:
    """When no forced-iteration policy is active, the abort still raises cleanly."""
    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "1")
    logger = _make_logger()

    # No policy pushed on the thread-local stack — _current_policy() returns None.
    with pytest.raises(_FatalPrimitiveAbort) as exc_info:
        logger._register_iteration_progress(False)

    assert exc_info.value.result["failure_class"] == "root_degenerate_loop"


def test_policy_callback_invoked_before_abort(monkeypatch) -> None:
    """When a policy with on_degenerate_refusal_loop is active, the callback fires."""
    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "3")
    captured: list[dict] = []

    policy = ForcedIterationPolicy(
        min_iterations=0,
        on_degenerate_refusal_loop=lambda payload: captured.append(payload),
    )
    logger = _make_logger()

    with forced_iteration_policy(policy):
        logger._register_iteration_progress(False)
        logger._register_iteration_progress(False)
        with pytest.raises(_FatalPrimitiveAbort):
            logger._register_iteration_progress(False)

    assert len(captured) == 1
    assert captured[0]["signature"] == "empty_code_block"
    assert captured[0]["count"] == 3
    assert captured[0]["required_stage"] is None


def test_raising_policy_callback_does_not_suppress_abort(monkeypatch) -> None:
    """A raising on_degenerate_refusal_loop must not block the _FatalPrimitiveAbort."""
    monkeypatch.setenv("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "1")

    def _bad_cb(_payload: dict) -> None:
        raise RuntimeError("callback broke")

    policy = ForcedIterationPolicy(
        min_iterations=0,
        on_degenerate_refusal_loop=_bad_cb,
    )
    logger = _make_logger()

    with forced_iteration_policy(policy):
        with pytest.raises(_FatalPrimitiveAbort) as exc_info:
            logger._register_iteration_progress(False)

    assert exc_info.value.result["failure_class"] == "root_degenerate_loop"
