"""auto must resolve to docker/local only - never a paid remote backend.
RunPod is legacy; GCP/Azure are the primary clouds but selected explicitly."""

from __future__ import annotations

import pytest

import backend.agents.execution as ex
from backend.agents.execution import SandboxMode, resolve_sandbox_mode


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_FORCE_SANDBOX", raising=False)
    # Clear lru_cache before the test; after monkeypatching the attribute may
    # become a plain lambda (no cache_clear), so guard teardown too.
    if hasattr(ex._docker_reachable, "cache_clear"):
        ex._docker_reachable.cache_clear()
    if hasattr(ex._is_wsl, "cache_clear"):
        ex._is_wsl.cache_clear()
    yield
    if hasattr(ex._docker_reachable, "cache_clear"):
        ex._docker_reachable.cache_clear()
    if hasattr(ex._is_wsl, "cache_clear"):
        ex._is_wsl.cache_clear()


def test_auto_with_docker_resolves_docker(monkeypatch):
    monkeypatch.setattr(ex, "_docker_reachable", lambda: True)
    monkeypatch.setattr(ex, "_is_wsl", lambda: False)
    assert resolve_sandbox_mode("auto", pipeline_mode="rlm") is SandboxMode.docker


def test_auto_without_docker_resolves_local(monkeypatch):
    monkeypatch.setattr(ex, "_docker_reachable", lambda: False)
    monkeypatch.setattr(ex, "_is_wsl", lambda: False)
    assert resolve_sandbox_mode("auto", pipeline_mode="rlm") is SandboxMode.local


def test_auto_never_resolves_runpod(monkeypatch):
    monkeypatch.setattr(ex, "_docker_reachable", lambda: True)
    monkeypatch.setattr(ex, "_is_wsl", lambda: False)
    assert resolve_sandbox_mode("auto", pipeline_mode="rlm") is not SandboxMode.runpod


def test_explicit_runpod_unchanged():
    assert resolve_sandbox_mode("runpod", pipeline_mode="rlm") is SandboxMode.runpod


def test_force_env_still_wins(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_FORCE_SANDBOX", "gcp")
    assert resolve_sandbox_mode("auto", pipeline_mode="rlm") is SandboxMode.gcp
