"""Test that the RLM path actually instantiates RunpodBackend for sandbox_mode=runpod
and propagates run_budget."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from backend.agents.execution import SandboxMode
from backend.agents.resilience.budget import RunBudget


def test_backend_for_sandbox_mode_returns_runpod_backend_for_runpod_mode(monkeypatch):
    """Sandbox mode 'runpod' must construct a real RunpodBackend, not fall back to docker."""
    monkeypatch.setenv("REPROLAB_RUNPOD_API_KEY", "fake-key")
    # Stub ensure_runpod_available so the SSH-key check doesn't trip.
    with patch("backend.services.runtime.ensure_runpod_available", lambda: None):
        from backend.agents.rlm.primitives import _backend_for_sandbox_mode
        from backend.services.runtime.runpod_backend import RunpodBackend

        backend = _backend_for_sandbox_mode(SandboxMode.runpod, run_budget=None)
        assert isinstance(backend, RunpodBackend), (
            f"Expected RunpodBackend, got {type(backend).__name__}"
        )


def test_backend_for_sandbox_mode_propagates_run_budget(monkeypatch):
    """The run_budget passed in must reach RunpodBackend._run_budget."""
    monkeypatch.setenv("REPROLAB_RUNPOD_API_KEY", "fake-key")
    budget = RunBudget(max_pod_seconds=300.0)
    with patch("backend.services.runtime.ensure_runpod_available", lambda: None):
        from backend.agents.rlm.primitives import _backend_for_sandbox_mode

        backend = _backend_for_sandbox_mode(SandboxMode.runpod, run_budget=budget)
        assert backend._run_budget is budget


def test_backend_for_sandbox_mode_still_returns_docker_for_docker_mode():
    """Docker mode must continue to return LocalDockerBackend (regression guard)."""
    from backend.agents.rlm.primitives import _backend_for_sandbox_mode
    from backend.services.runtime.local_docker import LocalDockerBackend

    backend = _backend_for_sandbox_mode(SandboxMode.docker, run_budget=None)
    assert isinstance(backend, LocalDockerBackend)


def test_backend_for_sandbox_mode_falls_back_for_unsupported_modes():
    """Unknown / unsupported modes still fall back to LocalDockerBackend with a warning."""
    from backend.agents.rlm.primitives import _backend_for_sandbox_mode
    from backend.services.runtime.local_docker import LocalDockerBackend

    backend = _backend_for_sandbox_mode(SandboxMode.simulate, run_budget=None)
    assert isinstance(backend, LocalDockerBackend)
