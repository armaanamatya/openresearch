"""Regression tests for backend._env_bootstrap.

The bootstrap is wired into ``backend/__init__.py`` so any submodule
import populates os.environ from .env. These tests exercise the loader
directly with synthetic .env files — they don't import ``backend``
itself, since that would consume the global one-shot guard for the
whole pytest session.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def fresh_bootstrap(monkeypatch: pytest.MonkeyPatch):
    """Reload backend._env_bootstrap so each test gets a fresh _LOADED flag."""
    if "backend._env_bootstrap" in sys.modules:
        del sys.modules["backend._env_bootstrap"]
    module = importlib.import_module("backend._env_bootstrap")
    yield module
    # Reset for the next test in case it imports it again.
    if "backend._env_bootstrap" in sys.modules:
        del sys.modules["backend._env_bootstrap"]


def _write_env(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_via_explicit_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fresh_bootstrap,
):
    env_file = _write_env(
        tmp_path / "custom.env",
        "FOO_TEST_KEY=hello\nFOO_TEST_QUOTED=\"with spaces\"\n",
    )

    monkeypatch.setenv("REPROLAB_DOTENV_PATH", str(env_file))
    monkeypatch.delenv("FOO_TEST_KEY", raising=False)
    monkeypatch.delenv("FOO_TEST_QUOTED", raising=False)

    loaded = fresh_bootstrap.load_dotenv_once()

    assert loaded == env_file.resolve()
    assert os.environ["FOO_TEST_KEY"] == "hello"
    assert os.environ["FOO_TEST_QUOTED"] == "with spaces"


def test_existing_env_vars_win(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fresh_bootstrap,
):
    env_file = _write_env(tmp_path / ".env", "FOO_TEST_OVERRIDE=from_file\n")

    monkeypatch.setenv("REPROLAB_DOTENV_PATH", str(env_file))
    monkeypatch.setenv("FOO_TEST_OVERRIDE", "from_shell")

    fresh_bootstrap.load_dotenv_once()

    assert os.environ["FOO_TEST_OVERRIDE"] == "from_shell"


def test_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fresh_bootstrap,
):
    env_file = _write_env(tmp_path / ".env", "FOO_TEST_IDEMPOTENT=v1\n")

    monkeypatch.setenv("REPROLAB_DOTENV_PATH", str(env_file))
    monkeypatch.delenv("FOO_TEST_IDEMPOTENT", raising=False)

    first = fresh_bootstrap.load_dotenv_once()
    assert first == env_file.resolve()

    # Mutate the file and re-call — the second call must be a no-op.
    env_file.write_text("FOO_TEST_IDEMPOTENT=v2\n", encoding="utf-8")
    second = fresh_bootstrap.load_dotenv_once()

    assert second is None
    assert os.environ["FOO_TEST_IDEMPOTENT"] == "v1"


def test_missing_file_is_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fresh_bootstrap,
):
    monkeypatch.setenv("REPROLAB_DOTENV_PATH", str(tmp_path / "does-not-exist"))
    # cwd lookup also has to miss for a clean miss; point cwd at an empty dir.
    monkeypatch.chdir(tmp_path)

    # Keep the package's own .env from rescuing the test by pointing the
    # walk-up to the empty tmp dir.
    monkeypatch.setattr(
        fresh_bootstrap,
        "_candidate_paths",
        lambda: [tmp_path / "does-not-exist"],
    )

    assert fresh_bootstrap.load_dotenv_once() is None


def test_export_prefix_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fresh_bootstrap,
):
    env_file = _write_env(
        tmp_path / ".env",
        "export FOO_TEST_EXPORTED=bar\n# comment\n\n",
    )

    monkeypatch.setenv("REPROLAB_DOTENV_PATH", str(env_file))
    monkeypatch.delenv("FOO_TEST_EXPORTED", raising=False)

    fresh_bootstrap.load_dotenv_once()

    assert os.environ["FOO_TEST_EXPORTED"] == "bar"
