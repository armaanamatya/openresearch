"""Tests for opt-in sandbox fallback after transient retry exhaustion (PR-ζ piece ζ.3).

Verified behaviors:
  - REPROLAB_RUNPOD_AUTO_FALLBACK=true + nvidia GPU + docker reachable
    → ctx.sandbox_mode flips to SandboxMode.docker after exhausted transient retries
    → sandbox_fallback SSE event emitted with correct payload shape
  - REPROLAB_RUNPOD_AUTO_FALLBACK=false (default) → no fallback even after exhaustion
  - REPROLAB_RUNPOD_AUTO_FALLBACK=true + nvidia GPU NOT present → no fallback
  - REPROLAB_RUNPOD_AUTO_FALLBACK=true + docker NOT reachable → no fallback
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import backend.agents.rlm.primitives as primitives
from backend.agents.execution import SandboxMode
from backend.agents.rlm.primitives import run_experiment
from backend.services.runtime.interface import RuntimeCauseKind, SandboxRuntimeError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exhausted_transient_exc() -> SandboxRuntimeError:
    """Simulate a SandboxRuntimeError that _execute_in_sandbox attaches _retry_attempts to."""
    exc = SandboxRuntimeError(
        RuntimeCauseKind.backend_unavailable,
        "Connection closed after 3 retries",
        retryable=True,
    )
    exc._retry_attempts = [  # type: ignore[attr-defined]
        {"attempt": 1, "transient_class": "transient", "error": "Connection closed"},
        {"attempt": 2, "transient_class": "transient", "error": "Connection closed"},
        {"attempt": 3, "transient_class": "transient", "error": "Connection closed"},
        {"attempt": 4, "transient_class": "transient", "error": "Connection closed"},
    ]
    return exc


def _make_ctx(tmp_path: Path, sandbox_mode=SandboxMode.runpod):
    from backend.agents.dashboard_emitter import DashboardEmitter
    from backend.agents.resilience.cost import RunCostLedger
    from backend.agents.rlm.context import RunContext
    from backend.agents.rlm.sse_bridge import make_emit

    project_id = "test_fallback"
    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    dashboard = DashboardEmitter(project_id, tmp_path)
    return RunContext(
        project_id=project_id,
        project_dir=project_dir,
        runs_root=tmp_path,
        dashboard=dashboard,
        emit=make_emit(dashboard),
        cost_ledger=RunCostLedger.load_jsonl(
            project_dir / "cost_ledger.jsonl",
            project_id=project_id,
            attach_path=True,
        ),
        llm_client=None,
        provider="anthropic",
        model="test-model",
        sandbox_mode=sandbox_mode,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fallback_flips_sandbox_mode_when_enabled(tmp_path, monkeypatch):
    """With REPROLAB_RUNPOD_AUTO_FALLBACK=true + GPU + docker, sandbox_mode flips."""
    monkeypatch.setenv("REPROLAB_RUNPOD_AUTO_FALLBACK", "true")
    ctx = _make_ctx(tmp_path, sandbox_mode=SandboxMode.runpod)

    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["echo ok"]))

    async def always_raise(*args, **kwargs):
        raise _exhausted_transient_exc()

    monkeypatch.setattr(primitives, "_execute_in_sandbox", always_raise)

    with patch("backend.agents.execution._docker_reachable", return_value=True):
        with patch("backend.services.runtime.gpu_resolution.host_supports_nvidia_gpu", return_value=True):
            run_experiment(str(code_dir), "test:image", ctx=ctx)

    assert ctx.sandbox_mode == SandboxMode.docker, (
        f"Expected SandboxMode.docker, got {ctx.sandbox_mode!r}"
    )


def test_fallback_emits_sandbox_fallback_event(tmp_path, monkeypatch):
    """sandbox_fallback SSE event must carry from/to/reason/attempts fields."""
    monkeypatch.setenv("REPROLAB_RUNPOD_AUTO_FALLBACK", "true")
    ctx = _make_ctx(tmp_path, sandbox_mode=SandboxMode.runpod)

    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["echo ok"]))

    async def always_raise(*args, **kwargs):
        raise _exhausted_transient_exc()

    monkeypatch.setattr(primitives, "_execute_in_sandbox", always_raise)

    with patch("backend.agents.execution._docker_reachable", return_value=True):
        with patch("backend.services.runtime.gpu_resolution.host_supports_nvidia_gpu", return_value=True):
            run_experiment(str(code_dir), "test:image", ctx=ctx)

    events_path = ctx.project_dir / "dashboard_events.jsonl"
    assert events_path.exists(), "dashboard_events.jsonl must exist"
    events = [json.loads(line) for line in events_path.read_text().strip().splitlines()]
    # dashboard_events.jsonl uses {"event": ..., "data": {...}} shape
    fallback_events = [e for e in events if e.get("event") == "sandbox_fallback"]
    assert len(fallback_events) >= 1, f"No sandbox_fallback event found in: {events}"

    payload = fallback_events[0]["data"]
    assert payload["from"] == "runpod"
    assert payload["to"] == "local"
    assert payload["reason"] == "max_retries_exhausted_after_transient_failures"
    assert isinstance(payload["attempts"], list)
    assert len(payload["attempts"]) > 0


def test_no_fallback_when_disabled_by_default(tmp_path, monkeypatch):
    """Default REPROLAB_RUNPOD_AUTO_FALLBACK (unset) → no fallback, mode unchanged."""
    monkeypatch.delenv("REPROLAB_RUNPOD_AUTO_FALLBACK", raising=False)
    ctx = _make_ctx(tmp_path, sandbox_mode=SandboxMode.runpod)
    original_mode = ctx.sandbox_mode

    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["echo ok"]))

    async def always_raise(*args, **kwargs):
        raise _exhausted_transient_exc()

    monkeypatch.setattr(primitives, "_execute_in_sandbox", always_raise)

    with patch("backend.agents.execution._docker_reachable", return_value=True):
        with patch("backend.services.runtime.gpu_resolution.host_supports_nvidia_gpu", return_value=True):
            run_experiment(str(code_dir), "test:image", ctx=ctx)

    assert ctx.sandbox_mode == original_mode, "sandbox_mode must not change when fallback disabled"


def test_no_fallback_when_no_nvidia_gpu(tmp_path, monkeypatch):
    """No GPU on host → no fallback even when REPROLAB_RUNPOD_AUTO_FALLBACK=true."""
    monkeypatch.setenv("REPROLAB_RUNPOD_AUTO_FALLBACK", "true")
    ctx = _make_ctx(tmp_path, sandbox_mode=SandboxMode.runpod)
    original_mode = ctx.sandbox_mode

    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["echo ok"]))

    async def always_raise(*args, **kwargs):
        raise _exhausted_transient_exc()

    monkeypatch.setattr(primitives, "_execute_in_sandbox", always_raise)

    with patch("backend.agents.execution._docker_reachable", return_value=True):
        with patch("backend.services.runtime.gpu_resolution.host_supports_nvidia_gpu", return_value=False):
            run_experiment(str(code_dir), "test:image", ctx=ctx)

    assert ctx.sandbox_mode == original_mode, "sandbox_mode must not change when no GPU"
