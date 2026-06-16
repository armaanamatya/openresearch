"""C6 (2026-06-16): runpod exec has NO liveness-based stall detection.

The 2026-06-08 execution-reliability redesign (streaming + stall kill +
finalize-on-timeout) is ``local``-scoped. The runpod exec path has only the
hard wall-clock ``timeout``; a genuine remote hang burns the full per-command
cap before it is reaped. A full remote heartbeat is deferred, so the minimal
fix documents the gap LOUDLY and exactly once per process via the module logger
(``RUNPOD_NO_STALL_DETECTION``). These tests pin that one-time-loud contract.
"""

from __future__ import annotations

import logging

import pytest

import backend.services.runtime.runpod_backend as rb


@pytest.fixture(autouse=True)
def _reset_warn_guard(monkeypatch):
    """Each test starts with the one-time guard un-fired."""
    monkeypatch.setattr(rb, "_STALL_WARN_EMITTED", False, raising=False)
    yield


def test_warn_emits_once_at_warning_level(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """First call emits a single WARNING naming RUNPOD_NO_STALL_DETECTION."""
    monkeypatch.delenv("REPROLAB_RUNPOD_STALL_WARN", raising=False)
    with caplog.at_level(logging.WARNING, logger=rb.__name__):
        rb._warn_no_remote_stall_detection_once()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"Expected exactly one WARNING, got {warnings}"
    assert "RUNPOD_NO_STALL_DETECTION" in warnings[0].getMessage()
    # The message must name what's missing AND the silence escape hatch.
    msg = warnings[0].getMessage()
    assert "stall" in msg.lower()
    assert "REPROLAB_RUNPOD_STALL_WARN=0" in msg


def test_warn_is_one_time_per_process(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Repeated calls emit the warning only once (guarded by the module flag)."""
    monkeypatch.delenv("REPROLAB_RUNPOD_STALL_WARN", raising=False)
    with caplog.at_level(logging.WARNING, logger=rb.__name__):
        for _ in range(5):
            rb._warn_no_remote_stall_detection_once()

    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "RUNPOD_NO_STALL_DETECTION" in r.getMessage()
    ]
    assert len(warnings) == 1, f"Warning must fire once per process, got {len(warnings)}"
    # The module guard must now be set.
    assert rb._STALL_WARN_EMITTED is True


@pytest.mark.parametrize("disable", ["0", "false", "no", "off", ""])
def test_warn_suppressed_when_flag_disabled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, disable: str
) -> None:
    """REPROLAB_RUNPOD_STALL_WARN falsy suppresses the warning entirely."""
    monkeypatch.setenv("REPROLAB_RUNPOD_STALL_WARN", disable)
    with caplog.at_level(logging.WARNING, logger=rb.__name__):
        rb._warn_no_remote_stall_detection_once()

    warnings = [
        r for r in caplog.records
        if "RUNPOD_NO_STALL_DETECTION" in r.getMessage()
    ]
    assert warnings == [], f"Disabled flag must emit nothing, got {warnings}"
    # Suppression must not consume the one-time guard (so re-enabling later still warns).
    assert rb._STALL_WARN_EMITTED is False


@pytest.mark.parametrize("enable", ["1", "true", "yes", "on", "TRUE"])
def test_warn_enabled_for_truthy_flag_values(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, enable: str
) -> None:
    monkeypatch.setenv("REPROLAB_RUNPOD_STALL_WARN", enable)
    with caplog.at_level(logging.WARNING, logger=rb.__name__):
        rb._warn_no_remote_stall_detection_once()
    assert any(
        "RUNPOD_NO_STALL_DETECTION" in r.getMessage() for r in caplog.records
    )


def test_exec_calls_the_warning_once() -> None:
    """Source guard: RunpodBackend.exec must invoke the one-time warning helper
    at exec start — catches a refactor that drops the wiring."""
    import inspect

    src = inspect.getsource(rb.RunpodBackend.exec)
    assert "_warn_no_remote_stall_detection_once()" in src, (
        "RunpodBackend.exec no longer calls the stall-coverage warning"
    )
