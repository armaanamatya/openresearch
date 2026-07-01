from pathlib import Path
from backend.services.runtime.asset_cache import AssetCache
from backend.services.runtime.env_adapters.alfworld import AlfworldAdapter
from backend.services.runtime.env_adapters.base import ProvisionCtx
from backend.agents.rlm import exclusion as X


class _DL:
    def __init__(self, fail=False, games=True):
        self.calls = 0; self.fail = fail; self.games = games
    def __call__(self, cache_dir: Path) -> None:
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        if self.games:
            g = Path(cache_dir) / "json_2.1.1" / "train" / "g0"
            g.mkdir(parents=True, exist_ok=True); (g / "traj_data.json").write_text("{}")


def test_downloads_once_then_cache_hit(tmp_path):
    dl = _DL()
    a = AlfworldAdapter(AssetCache(tmp_path), downloader=dl)
    r1 = a.provision(ProvisionCtx(display_name="ALFWorld"))
    assert r1.ok and dl.calls == 1 and r1.detail == "downloaded"
    r2 = a.provision(ProvisionCtx(display_name="ALFWorld"))
    assert r2.ok and dl.calls == 1 and r2.detail == "cache hit"


def test_empty_download_is_verified_exclusion(tmp_path):
    a = AlfworldAdapter(AssetCache(tmp_path), downloader=_DL(games=False))
    r = a.provision(ProvisionCtx(display_name="ALFWorld"))
    assert not r.ok and r.exclusion.verified and r.exclusion.kind == X.KIND_ENV_SETUP_FAILED


def test_smoke_reflects_games_present(tmp_path):
    dl = _DL()
    a = AlfworldAdapter(AssetCache(tmp_path), downloader=dl)
    r = a.provision(ProvisionCtx(display_name="ALFWorld"))
    assert a.smoke(ProvisionCtx(code_dir=None, display_name=r.data_path)).ok is True
