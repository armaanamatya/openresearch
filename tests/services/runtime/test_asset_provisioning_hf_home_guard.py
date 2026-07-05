"""Tests for the HF_HOME operator-passthrough clobber guard (Task #2c).

An operator who has explicitly staged HF_HOME onto the
``OPENRESEARCH_CELL_ENV_PASSTHROUGH`` allowlist (e.g. a persistent-disk cache
mounted before the harness runs) keeps their own HF_HOME value — ensure_assets
must NOT clobber it with the harness's shared cache path. Absent the
allowlist (today's default), HF_HOME is always overwritten — byte-identical
to before this guard existed.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.agents.schemas import AssetSpec
from backend.services.runtime.asset_provisioning import ensure_assets


def _make_spec(**kwargs) -> AssetSpec:
    defaults = dict(requirements_files=[], models=[], datasets=[], webshop=False)
    defaults.update(kwargs)
    return AssetSpec(**defaults)


class TestHfHomeClobberGuard:
    def test_allowlist_absent_hf_home_overwritten_as_today(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", raising=False)
        monkeypatch.setenv("HF_HOME", "/operator/staged/hf")

        ensure_assets(_make_spec(), cache_root=tmp_path, prepare=False)

        assert os.environ["HF_HOME"] == str(tmp_path / "hf")

    def test_allowlist_present_but_hf_home_unset_is_assigned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Allowlist names HF_HOME, but no operator value is set — assign as today."""
        monkeypatch.setenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", "HF_HOME")
        monkeypatch.delenv("HF_HOME", raising=False)

        ensure_assets(_make_spec(), cache_root=tmp_path, prepare=False)

        assert os.environ["HF_HOME"] == str(tmp_path / "hf")

    def test_allowlist_present_and_hf_home_set_survives(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", "HF_HOME")
        monkeypatch.setenv("HF_HOME", "/operator/staged/hf")

        ensure_assets(_make_spec(), cache_root=tmp_path, prepare=False)

        assert os.environ["HF_HOME"] == "/operator/staged/hf"

    def test_allowlist_present_with_other_names_hf_home_still_overwritten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """HF_HOME must be explicitly named — a passthrough list of OTHER
        names does not exempt it."""
        monkeypatch.setenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", "FOO,BAR")
        monkeypatch.setenv("HF_HOME", "/operator/staged/hf")

        ensure_assets(_make_spec(), cache_root=tmp_path, prepare=False)

        assert os.environ["HF_HOME"] == str(tmp_path / "hf")

    def test_pip_cache_dir_and_env_cache_dir_unaffected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """PIP_CACHE_DIR / OPENRESEARCH_ENV_CACHE_DIR are always assigned,
        even when HF_HOME is guarded."""
        monkeypatch.setenv("OPENRESEARCH_CELL_ENV_PASSTHROUGH", "HF_HOME")
        monkeypatch.setenv("HF_HOME", "/operator/staged/hf")
        monkeypatch.setenv("PIP_CACHE_DIR", "/operator/staged/pip")
        monkeypatch.setenv("OPENRESEARCH_ENV_CACHE_DIR", "/operator/staged/envs")

        ensure_assets(_make_spec(), cache_root=tmp_path, prepare=False)

        assert os.environ["PIP_CACHE_DIR"] == str(tmp_path / "pip")
        assert os.environ["OPENRESEARCH_ENV_CACHE_DIR"] == str(tmp_path / "envs")
