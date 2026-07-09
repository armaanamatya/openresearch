"""Explicitly choosing the legacy runpod backend emits a one-line info log."""

from __future__ import annotations

import logging

from backend.agents.execution import SandboxMode


def test_runpod_selection_logs_legacy_notice(caplog, monkeypatch):
    import backend.services.runtime as runtime
    import backend.agents.rlm.primitives as primitives

    monkeypatch.setattr(runtime, "ensure_runpod_available", lambda: None)
    monkeypatch.setattr(
        primitives, "RunpodBackend", lambda **kw: object(), raising=False
    )
    # Also patch the runpod_backend module symbol if the function imports it locally:
    import backend.services.runtime.runpod_backend as rb
    monkeypatch.setattr(rb, "RunpodBackend", lambda **kw: object(), raising=False)

    with caplog.at_level(logging.INFO, logger="backend.agents.rlm.primitives"):
        primitives._backend_for_sandbox_mode(
            SandboxMode.runpod, run_budget=None, gpu_plan=None
        )
    assert any("legacy" in r.message.lower() for r in caplog.records)
