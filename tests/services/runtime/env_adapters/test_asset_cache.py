from pathlib import Path
from backend.services.runtime.asset_cache import AssetCache, default_cache_dir, _pid_alive


def test_locked_state_persists_across_instances(tmp_path: Path):
    with AssetCache(tmp_path).locked_state() as st:
        st["alfworld"] = {"ready": True, "data_path": "/x"}
    # A fresh instance (later run/cell) sees the persisted record.
    with AssetCache(tmp_path).locked_state() as st2:
        assert st2["alfworld"] == {"ready": True, "data_path": "/x"}


def test_locked_state_rolls_back_nothing_on_read(tmp_path: Path):
    with AssetCache(tmp_path).locked_state() as st:
        assert st == {}                       # empty on cold cache


def test_default_cache_dir_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_ENV_CACHE_DIR", str(tmp_path / "e"))
    assert default_cache_dir() == (tmp_path / "e").resolve()


def test_pid_alive_self_true():
    import os
    assert _pid_alive(os.getpid()) is True
    assert _pid_alive(-1) is False
