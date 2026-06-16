"""Tests for local-sandbox routing in backend.agents.rlm.primitives.

Two responsibilities pinned here:

A1. ``_backend_for_sandbox_mode(SandboxMode.local)`` returns a
    ``LocalProcessBackend`` and NOT a ``LocalDockerBackend``.

A2. ``build_environment(...)`` short-circuits to a no-docker no-op
    when ``ctx.sandbox_mode`` resolves to ``"local"``.  The test guards
    that no docker entry-point (``_make_docker_client``) is invoked;
    touching it would raise ``AssertionError`` if the local path ever
    regresses to the docker branch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.agents.rlm.primitives import (
    _backend_for_sandbox_mode,
    build_environment,
)
from backend.agents.execution import SandboxMode
from backend.services.runtime.local_process import LocalProcessBackend
from backend.services.runtime.local_docker import LocalDockerBackend


# ---------------------------------------------------------------------------
# A1: _backend_for_sandbox_mode routing
# ---------------------------------------------------------------------------


def test_local_mode_returns_local_process_backend() -> None:
    """SandboxMode.local must route to LocalProcessBackend, not LocalDockerBackend."""
    backend = _backend_for_sandbox_mode(SandboxMode.local)
    assert isinstance(backend, LocalProcessBackend), (
        f"Expected LocalProcessBackend but got {type(backend).__name__}"
    )


def test_local_mode_is_not_docker_backend() -> None:
    """SandboxMode.local must NOT return a LocalDockerBackend."""
    backend = _backend_for_sandbox_mode(SandboxMode.local)
    assert not isinstance(backend, LocalDockerBackend), (
        "SandboxMode.local must not be routed to LocalDockerBackend"
    )


def test_docker_mode_returns_docker_backend() -> None:
    """Baseline sanity: SandboxMode.docker still routes to LocalDockerBackend."""
    backend = _backend_for_sandbox_mode(SandboxMode.docker)
    assert isinstance(backend, LocalDockerBackend)


def test_none_mode_falls_back_to_docker_backend() -> None:
    """None sandbox_mode uses the default (LocalDockerBackend)."""
    backend = _backend_for_sandbox_mode(None)
    assert isinstance(backend, LocalDockerBackend)


# ---------------------------------------------------------------------------
# A2: build_environment local no-op
# ---------------------------------------------------------------------------


def _make_local_ctx(tmp_path) -> SimpleNamespace:
    """Build a minimal RunContext-like object with sandbox_mode='local'.

    The real code uses:
        _sb_mode = getattr(ctx, "sandbox_mode", None)
        _sb_key = getattr(_sb_mode, "value", str(_sb_mode) if _sb_mode is not None else None)
        if _sb_key == "local": ...

    We supply the real SandboxMode.local enum so .value == "local".
    """
    from backend.agents.dashboard_emitter import DashboardEmitter
    from backend.agents.resilience.cost import RunCostLedger
    from backend.agents.rlm.context import RunContext
    from backend.agents.rlm.sse_bridge import make_emit

    project_id = "test-local-noopbuild"
    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    dashboard = DashboardEmitter(project_id, tmp_path)
    ctx = RunContext(
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
    )
    # Override sandbox_mode to the local enum value
    object.__setattr__(ctx, "sandbox_mode", SandboxMode.local)
    return ctx


def test_build_environment_local_skips_docker(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """build_environment must short-circuit before touching Docker when sandbox_mode='local'.

    The monkeypatch makes _make_docker_client (and _image_exists) raise
    AssertionError if called, so any regression that reaches Docker code
    immediately fails the test.
    """
    import backend.services.runtime.local_docker as _ld
    import backend.agents.rlm.primitives as _prim

    def _should_not_call_docker(*_args, **_kwargs):  # type: ignore[override]
        raise AssertionError(
            "build_environment with sandbox_mode='local' must never touch Docker"
        )

    monkeypatch.setattr(_ld, "_make_docker_client", _should_not_call_docker)
    monkeypatch.setattr(_prim, "_image_exists", _should_not_call_docker)

    ctx = _make_local_ctx(tmp_path)
    result = build_environment({"dockerfile": "FROM python:3.12-slim"}, ctx=ctx)

    # Must be a success result
    assert result["ok"] is True, f"Expected ok=True, got: {result}"


def test_build_environment_local_sets_skipped_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """build_environment local result must include skipped=True."""
    import backend.services.runtime.local_docker as _ld
    import backend.agents.rlm.primitives as _prim

    monkeypatch.setattr(_ld, "_make_docker_client", lambda *a, **k: None)
    monkeypatch.setattr(_prim, "_image_exists", lambda *a, **k: False)

    ctx = _make_local_ctx(tmp_path)
    result = build_environment({"dockerfile": "FROM python:3.12-slim"}, ctx=ctx)

    assert result.get("skipped") is True, (
        f"Expected skipped=True in local no-op result; got keys: {list(result.keys())}"
    )


def test_build_environment_local_image_tag_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """build_environment local no-op result must have image_tag == '' (no image built)."""
    import backend.services.runtime.local_docker as _ld
    import backend.agents.rlm.primitives as _prim

    monkeypatch.setattr(_ld, "_make_docker_client", lambda *a, **k: None)
    monkeypatch.setattr(_prim, "_image_exists", lambda *a, **k: False)

    ctx = _make_local_ctx(tmp_path)
    result = build_environment({"dockerfile": "FROM python:3.12-slim"}, ctx=ctx)

    assert result.get("image_tag") == "", (
        f"Expected image_tag='' for local sandbox; got {result.get('image_tag')!r}"
    )


def test_build_environment_local_outcome_is_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """build_environment local result must carry outcome='ok' (PrimitiveOutcome.ok.value)."""
    import backend.services.runtime.local_docker as _ld
    import backend.agents.rlm.primitives as _prim

    monkeypatch.setattr(_ld, "_make_docker_client", lambda *a, **k: None)
    monkeypatch.setattr(_prim, "_image_exists", lambda *a, **k: False)

    ctx = _make_local_ctx(tmp_path)
    result = build_environment({"dockerfile": "FROM python:3.12-slim"}, ctx=ctx)

    assert result.get("outcome") == "ok", (
        f"Expected outcome='ok'; got {result.get('outcome')!r}"
    )


# ---------------------------------------------------------------------------
# A3: build_environment runpod no-op (2026-05-30)
# The pod boots OPENRESEARCH_RUNPOD_IMAGE over SSH; the local build is never used
# and HARD-FAILED when the base image wasn't pullable from docker.io.
# ---------------------------------------------------------------------------


def _make_runpod_ctx(tmp_path):
    ctx = _make_local_ctx(tmp_path)
    object.__setattr__(ctx, "sandbox_mode", SandboxMode.runpod)
    return ctx


def test_build_environment_runpod_skips_docker(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Under runpod, build_environment must NOT touch local Docker (skip the unused build)."""
    import backend.services.runtime.local_docker as _ld
    import backend.agents.rlm.primitives as _prim

    def _should_not_call_docker(*_a, **_k):
        raise AssertionError("build_environment under runpod must not touch local Docker")

    monkeypatch.setattr(_ld, "_make_docker_client", _should_not_call_docker)
    monkeypatch.setattr(_prim, "_image_exists", _should_not_call_docker)

    ctx = _make_runpod_ctx(tmp_path)
    # A base image that does NOT exist on docker.io — proves we never try to build it.
    result = build_environment(
        {"dockerfile": "FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04"},
        ctx=ctx,
    )
    assert result["ok"] is True, f"Expected ok=True (skipped), got: {result}"
    assert result.get("skipped") is True
    # The short-circuit reports the settings-configured pod image (so
    # run_experiment can hand it to the RunPod backend) — not an empty tag.
    from backend.config import get_settings
    assert result.get("image_tag") == get_settings().runpod_image
