"""Tests for the dedicated WebShop venv feature.

Covers:
1. _default_webshop_launcher honours OPENRESEARCH_WEBSHOP_PYTHON when set,
   and falls back to sys.executable when unset.
2. ensure_assets(..., webshop_python_version=None) calls the legacy
   install_webshop path and leaves OPENRESEARCH_WEBSHOP_PYTHON unset.
3. ensure_assets(..., webshop_python_version="3.10") calls
   install_webshop_dedicated, sets OPENRESEARCH_WEBSHOP_PYTHON, and
   appends "webshop:dedicated-venv" to report.ensured.

All tests avoid network calls, real venv creation, or ML library imports.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.agents.schemas import AssetSpec
from backend.services.runtime.asset_provisioning import (
    _resolve_webshop_python,
    ensure_assets,
    webshop_importable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(**kwargs) -> AssetSpec:
    defaults = dict(requirements_files=[], models=[], datasets=[], webshop=True)
    defaults.update(kwargs)
    return AssetSpec(**defaults)


# ---------------------------------------------------------------------------
# 1. _default_webshop_launcher interpreter selection
# ---------------------------------------------------------------------------

def test_webshop_launcher_uses_env_python_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When OPENRESEARCH_WEBSHOP_PYTHON is set, Popen argv[0] uses that path."""
    fake_python = "/opt/webshop-venv/bin/python"
    monkeypatch.setenv("OPENRESEARCH_WEBSHOP_PYTHON", fake_python)

    captured: list[list[str]] = []

    class _FakePopen:
        def __init__(self, args, **kwargs):
            captured.append(list(args))
            self.pid = 12345

    import backend.services.runtime.env_cache as ec

    with patch.object(ec, "_default_webshop_launcher", wraps=ec._default_webshop_launcher):
        # Patch subprocess.Popen inside the env_cache module
        with patch("subprocess.Popen", _FakePopen):
            pid = ec._default_webshop_launcher(tmp_path, 3001)

    assert pid == 12345
    assert len(captured) == 1
    assert captured[0][0] == fake_python


def test_webshop_launcher_falls_back_to_sys_executable_when_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When OPENRESEARCH_WEBSHOP_PYTHON is unset, Popen argv[0] is sys.executable."""
    monkeypatch.delenv("OPENRESEARCH_WEBSHOP_PYTHON", raising=False)

    captured: list[list[str]] = []

    class _FakePopen:
        def __init__(self, args, **kwargs):
            captured.append(list(args))
            self.pid = 99

    import backend.services.runtime.env_cache as ec

    with patch("subprocess.Popen", _FakePopen):
        pid = ec._default_webshop_launcher(tmp_path, 3002)

    assert pid == 99
    assert captured[0][0] == sys.executable


def test_webshop_launcher_falls_back_when_env_is_empty_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An empty OPENRESEARCH_WEBSHOP_PYTHON is treated the same as unset."""
    monkeypatch.setenv("OPENRESEARCH_WEBSHOP_PYTHON", "")

    captured: list[list[str]] = []

    class _FakePopen:
        def __init__(self, args, **kwargs):
            captured.append(list(args))
            self.pid = 7

    import backend.services.runtime.env_cache as ec

    with patch("subprocess.Popen", _FakePopen):
        ec._default_webshop_launcher(tmp_path, 3003)

    assert captured[0][0] == sys.executable


# ---------------------------------------------------------------------------
# 2. ensure_assets with webshop_python_version=None → legacy path
# ---------------------------------------------------------------------------

def test_ensure_assets_none_version_calls_legacy_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """webshop_python_version=None keeps the existing install_webshop path."""
    monkeypatch.delenv("OPENRESEARCH_WEBSHOP_PYTHON", raising=False)

    legacy_calls: list[str] = []
    dedicated_calls: list[str] = []

    def _fake_install_webshop(pip_cache, cache_root):
        legacy_calls.append("called")

    def _fake_install_dedicated(cache_root, *, python_version="3.10"):
        dedicated_calls.append(python_version)
        return Path("/fake/venv/bin/python")

    # Patch _module_exists to return False so install_webshop is actually called
    with (
        patch(
            "backend.services.runtime.asset_provisioning._module_exists",
            return_value=False,
        ),
        patch(
            "backend.services.runtime.asset_provisioning.install_webshop",
            side_effect=_fake_install_webshop,
        ),
        patch(
            "backend.services.runtime.asset_provisioning.install_webshop_dedicated",
            side_effect=_fake_install_dedicated,
        ),
    ):
        report = ensure_assets(_make_spec(), cache_root=tmp_path, webshop_python_version=None)

    assert legacy_calls == ["called"], "legacy install_webshop must be called"
    assert dedicated_calls == [], "install_webshop_dedicated must NOT be called"
    assert "webshop:web_agent_site" in report.ensured
    assert "webshop:dedicated-venv" not in report.ensured
    # OPENRESEARCH_WEBSHOP_PYTHON must NOT be set by the legacy path
    assert "OPENRESEARCH_WEBSHOP_PYTHON" not in os.environ


def test_ensure_assets_none_version_skips_when_already_importable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """webshop_python_version=None skips install when web_agent_site is importable."""
    monkeypatch.delenv("OPENRESEARCH_WEBSHOP_PYTHON", raising=False)

    legacy_calls: list[str] = []
    dedicated_calls: list[str] = []

    with (
        patch(
            "backend.services.runtime.asset_provisioning._module_exists",
            return_value=True,
        ),
        patch(
            "backend.services.runtime.asset_provisioning.install_webshop",
            side_effect=lambda *a: legacy_calls.append("called"),
        ),
        patch(
            "backend.services.runtime.asset_provisioning.install_webshop_dedicated",
            side_effect=lambda *a, **kw: dedicated_calls.append("called"),
        ),
    ):
        report = ensure_assets(_make_spec(), cache_root=tmp_path, webshop_python_version=None)

    assert legacy_calls == [], "install_webshop must not be called when already importable"
    assert dedicated_calls == [], "install_webshop_dedicated must not be called"
    assert "webshop:web_agent_site" in report.skipped


# ---------------------------------------------------------------------------
# 3. ensure_assets with webshop_python_version="3.10" → dedicated path
# ---------------------------------------------------------------------------

def test_ensure_assets_dedicated_version_calls_dedicated_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """webshop_python_version='3.10' routes to install_webshop_dedicated."""
    monkeypatch.delenv("OPENRESEARCH_WEBSHOP_PYTHON", raising=False)

    fake_venv_python = tmp_path / "webshop" / ".venv-webshop" / "bin" / "python"
    legacy_calls: list[str] = []
    dedicated_calls: list[tuple[Path, str]] = []

    def _fake_install_dedicated(cache_root, *, python_version="3.10"):
        dedicated_calls.append((cache_root, python_version))
        return fake_venv_python

    with (
        patch(
            "backend.services.runtime.asset_provisioning.install_webshop",
            side_effect=lambda *a: legacy_calls.append("called"),
        ),
        patch(
            "backend.services.runtime.asset_provisioning.install_webshop_dedicated",
            side_effect=_fake_install_dedicated,
        ),
    ):
        report = ensure_assets(
            _make_spec(), cache_root=tmp_path, webshop_python_version="3.10"
        )

    assert legacy_calls == [], "legacy install_webshop must NOT be called"
    assert len(dedicated_calls) == 1, "install_webshop_dedicated must be called exactly once"
    assert dedicated_calls[0][1] == "3.10"
    assert "webshop:dedicated-venv" in report.ensured
    assert "webshop:web_agent_site" not in report.ensured

    # OPENRESEARCH_WEBSHOP_PYTHON must be set to the returned venv python path
    assert os.environ.get("OPENRESEARCH_WEBSHOP_PYTHON") == str(fake_venv_python)


def test_ensure_assets_dedicated_sets_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """OPENRESEARCH_WEBSHOP_PYTHON is set to the venv python when dedicated path is used."""
    monkeypatch.delenv("OPENRESEARCH_WEBSHOP_PYTHON", raising=False)

    expected_python = Path("/dedicated/venv/bin/python")

    with patch(
        "backend.services.runtime.asset_provisioning.install_webshop_dedicated",
        return_value=expected_python,
    ):
        ensure_assets(_make_spec(), cache_root=tmp_path, webshop_python_version="3.10")

    assert os.environ["OPENRESEARCH_WEBSHOP_PYTHON"] == str(expected_python)


def test_ensure_assets_default_version_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Default webshop_python_version=None → legacy path (backward-compatible)."""
    monkeypatch.delenv("OPENRESEARCH_WEBSHOP_PYTHON", raising=False)

    dedicated_calls: list[str] = []

    with (
        patch(
            "backend.services.runtime.asset_provisioning._module_exists",
            return_value=True,
        ),
        patch(
            "backend.services.runtime.asset_provisioning.install_webshop_dedicated",
            side_effect=lambda *a, **kw: dedicated_calls.append("called"),
        ),
    ):
        # Call without the new kwarg — must behave exactly as before
        report = ensure_assets(_make_spec(), cache_root=tmp_path)

    assert dedicated_calls == []
    assert "webshop:dedicated-venv" not in report.ensured


# ---------------------------------------------------------------------------
# 4. Preflight check probes the WebShop interpreter, not the run venv.
#    (Regression guard: the dedicated-venv split puts web_agent_site in a
#    separate venv, so a current-interpreter probe would always RED the gate.)
# ---------------------------------------------------------------------------

def test_resolve_webshop_python_prefers_env(monkeypatch: pytest.MonkeyPatch):
    """OPENRESEARCH_WEBSHOP_PYTHON, when set, is the WebShop interpreter."""
    monkeypatch.setenv("OPENRESEARCH_WEBSHOP_PYTHON", "/opt/webshop/bin/python")
    assert _resolve_webshop_python() == "/opt/webshop/bin/python"


def test_resolve_webshop_python_falls_back_to_current(monkeypatch: pytest.MonkeyPatch):
    """Unset or empty resolves to the current interpreter (mirrors the launcher)."""
    monkeypatch.delenv("OPENRESEARCH_WEBSHOP_PYTHON", raising=False)
    assert _resolve_webshop_python() == sys.executable
    monkeypatch.setenv("OPENRESEARCH_WEBSHOP_PYTHON", "")
    assert _resolve_webshop_python() == sys.executable


def test_webshop_importable_subprocess_probes_dedicated_interpreter(
    monkeypatch: pytest.MonkeyPatch,
):
    """A dedicated interpreter is probed out-of-process; a bogus path → False."""
    monkeypatch.setenv("OPENRESEARCH_WEBSHOP_PYTHON", "/nonexistent/python")
    assert webshop_importable() is False


def test_webshop_importable_delegates_to_module_exists_for_run_venv(
    monkeypatch: pytest.MonkeyPatch,
):
    """When the interpreter is the run venv, use the fast in-process probe."""
    monkeypatch.delenv("OPENRESEARCH_WEBSHOP_PYTHON", raising=False)
    with patch(
        "backend.services.runtime.asset_provisioning._module_exists",
        return_value=True,
    ) as m:
        assert webshop_importable() is True
        m.assert_called_once_with("web_agent_site")


# ---------------------------------------------------------------------------
# 5. Dedicated WebShop install is best-effort — a failure must not block the run.
# ---------------------------------------------------------------------------

def test_ensure_assets_dedicated_webshop_is_best_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A dedicated WebShop install failure is recorded and skipped, not raised.

    WebShop's 2022 stack + JVM + data are fragile; env_cache already degrades a
    missing WebShop to an exclusion, so ensure_assets must not let an install
    failure abort a multi-hour run. The env var stays unset so the launcher won't
    point at a broken interpreter.
    """
    from backend.services.runtime.asset_provisioning import AssetProvisionError

    monkeypatch.delenv("OPENRESEARCH_WEBSHOP_PYTHON", raising=False)

    def _boom(cache_root, *, python_version="3.10"):
        raise AssetProvisionError("simulated dedicated WebShop venv failure")

    with patch(
        "backend.services.runtime.asset_provisioning.install_webshop_dedicated",
        side_effect=_boom,
    ):
        report = ensure_assets(_make_spec(), cache_root=tmp_path, webshop_python_version="3.10")

    assert any("webshop:dedicated-venv" in f for f in report.failed), report.failed
    assert "webshop:dedicated-venv" not in report.ensured
    assert "OPENRESEARCH_WEBSHOP_PYTHON" not in os.environ
