"""Tests for WSL-aware sandbox auto-detection in resolve_sandbox_mode.

Covers:
- _is_wsl() detection via /proc/version content
- _docker_reachable() via shutil.which + subprocess.run
- resolve_sandbox_mode("auto") WSL routing
- Explicit docker choice is never overridden
- OPENRESEARCH_FORCE_SANDBOX env override takes precedence
"""

from __future__ import annotations

from unittest.mock import MagicMock


from backend.agents.execution import (
    SandboxMode,
    _docker_reachable,
    _is_wsl,
    resolve_sandbox_mode,
)


# ---------------------------------------------------------------------------
# _is_wsl() tests
# ---------------------------------------------------------------------------


def test_is_wsl_detects_microsoft_proc_version(monkeypatch, tmp_path):
    """Returns True when /proc/version contains 'Microsoft'."""
    _is_wsl.cache_clear()
    fake_content = "Linux version 5.15.90.1-microsoft-standard-WSL2 (Microsoft)"

    def _fake_open(path, *args, **kwargs):
        if path == "/proc/version":
            from io import StringIO
            return StringIO(fake_content)
        return open(path, *args, **kwargs)  # noqa: WPS515

    monkeypatch.setattr("builtins.open", _fake_open)
    assert _is_wsl() is True
    _is_wsl.cache_clear()


def test_is_wsl_detects_wsl_keyword_in_proc_version(monkeypatch):
    """Returns True when /proc/version contains 'WSL' (case-insensitive)."""
    _is_wsl.cache_clear()
    fake_content = "Linux version 5.15.90.1 (wsl kernel build)"

    def _fake_open(path, *args, **kwargs):
        if path == "/proc/version":
            from io import StringIO
            return StringIO(fake_content)
        return open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _fake_open)
    assert _is_wsl() is True
    _is_wsl.cache_clear()


def test_is_wsl_returns_false_on_native_linux(monkeypatch):
    """Returns False when /proc/version is a plain Linux kernel string."""
    _is_wsl.cache_clear()
    fake_content = "Linux version 6.5.0-27-generic (buildd@Ubuntu) #28-Ubuntu SMP"

    def _fake_open(path, *args, **kwargs):
        if path == "/proc/version":
            from io import StringIO
            return StringIO(fake_content)
        return open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _fake_open)
    assert _is_wsl() is False
    _is_wsl.cache_clear()


def test_is_wsl_returns_false_when_proc_version_missing(monkeypatch):
    """Returns False when /proc/version does not exist (macOS, Windows native)."""
    _is_wsl.cache_clear()

    def _fake_open(path, *args, **kwargs):
        if path == "/proc/version":
            raise FileNotFoundError("/proc/version not found")
        return open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _fake_open)
    assert _is_wsl() is False
    _is_wsl.cache_clear()


# ---------------------------------------------------------------------------
# _docker_reachable() tests
# ---------------------------------------------------------------------------


def test_docker_reachable_when_binary_missing(monkeypatch):
    """Returns False when 'docker' is not on PATH."""
    _docker_reachable.cache_clear()
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert _docker_reachable() is False
    _docker_reachable.cache_clear()


def test_docker_reachable_when_info_succeeds(monkeypatch):
    """Returns True when docker binary exists AND `docker info` exits 0."""
    _docker_reachable.cache_clear()
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None)

    mock_result = MagicMock()
    mock_result.returncode = 0

    monkeypatch.setattr(
        "backend.agents.execution.subprocess.run",
        lambda *args, **kwargs: mock_result,
    )
    assert _docker_reachable() is True
    _docker_reachable.cache_clear()


def test_docker_reachable_when_info_fails(monkeypatch):
    """Returns False when docker binary exists but `docker info` exits non-zero."""
    _docker_reachable.cache_clear()
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None)

    mock_result = MagicMock()
    mock_result.returncode = 1

    monkeypatch.setattr(
        "backend.agents.execution.subprocess.run",
        lambda *args, **kwargs: mock_result,
    )
    assert _docker_reachable() is False
    _docker_reachable.cache_clear()


# ---------------------------------------------------------------------------
# resolve_sandbox_mode() routing tests
# ---------------------------------------------------------------------------


def test_resolve_auto_on_wsl_no_docker_returns_local(monkeypatch):
    """On WSL with no reachable docker, auto → local."""
    _is_wsl.cache_clear()
    _docker_reachable.cache_clear()
    monkeypatch.delenv("OPENRESEARCH_FORCE_SANDBOX", raising=False)
    monkeypatch.setattr("backend.agents.execution._is_wsl", lambda: True)
    monkeypatch.setattr("backend.agents.execution._docker_reachable", lambda: False)

    result = resolve_sandbox_mode("auto", pipeline_mode="rlm")
    assert result is SandboxMode.local

    _is_wsl.cache_clear()
    _docker_reachable.cache_clear()


def test_resolve_auto_on_wsl_with_docker_uses_default(monkeypatch):
    """On WSL with docker reachable, auto → DEFAULT_SANDBOX_MODE (not overridden)."""
    from backend.agents.execution import DEFAULT_SANDBOX_MODE

    _is_wsl.cache_clear()
    _docker_reachable.cache_clear()
    monkeypatch.delenv("OPENRESEARCH_FORCE_SANDBOX", raising=False)
    monkeypatch.setattr("backend.agents.execution._is_wsl", lambda: True)
    monkeypatch.setattr("backend.agents.execution._docker_reachable", lambda: True)

    result = resolve_sandbox_mode("auto", pipeline_mode="rlm")
    # Should NOT be forced to local — docker is reachable so we use the default.
    assert result is DEFAULT_SANDBOX_MODE
    assert result is not SandboxMode.local

    _is_wsl.cache_clear()
    _docker_reachable.cache_clear()


def test_explicit_docker_not_overridden_on_wsl(monkeypatch):
    """Explicit sandbox='docker' is never overridden, even on WSL without docker daemon."""
    _is_wsl.cache_clear()
    _docker_reachable.cache_clear()
    monkeypatch.delenv("OPENRESEARCH_FORCE_SANDBOX", raising=False)
    monkeypatch.setattr("backend.agents.execution._is_wsl", lambda: True)
    monkeypatch.setattr("backend.agents.execution._docker_reachable", lambda: False)

    result = resolve_sandbox_mode("docker", pipeline_mode="rlm")
    assert result is SandboxMode.docker

    _is_wsl.cache_clear()
    _docker_reachable.cache_clear()


def test_OPENRESEARCH_FORCE_SANDBOX_still_wins(monkeypatch):
    """OPENRESEARCH_FORCE_SANDBOX=runpod overrides even WSL+no-docker auto resolution."""
    _is_wsl.cache_clear()
    _docker_reachable.cache_clear()
    monkeypatch.setenv("OPENRESEARCH_FORCE_SANDBOX", "runpod")
    monkeypatch.setattr("backend.agents.execution._is_wsl", lambda: True)
    monkeypatch.setattr("backend.agents.execution._docker_reachable", lambda: False)

    result = resolve_sandbox_mode("auto", pipeline_mode="rlm")
    assert result is SandboxMode.runpod

    _is_wsl.cache_clear()
    _docker_reachable.cache_clear()
    monkeypatch.delenv("OPENRESEARCH_FORCE_SANDBOX", raising=False)
