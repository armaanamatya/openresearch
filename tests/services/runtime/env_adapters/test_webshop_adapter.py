import signal
from backend.services.runtime.asset_cache import AssetCache
from backend.services.runtime.env_adapters.webshop import WebShopAdapter
from backend.services.runtime.env_adapters.base import ProvisionCtx


def test_one_server_many_leases_then_torn_down(tmp_path, monkeypatch):
    launches, kills = [], []
    monkeypatch.setattr("backend.services.runtime.env_adapters.webshop.os.kill",
                        lambda pid, sig: kills.append((pid, sig)))
    a = WebShopAdapter(AssetCache(tmp_path), server_launcher=lambda c, p: (launches.append(p) or 42),
                       probe=lambda u: True, pid_alive=lambda pid: True)
    r1 = a.provision(ProvisionCtx(display_name="WebShop")); assert r1.ok and r1.base_url and len(launches) == 1
    a.provision(ProvisionCtx(display_name="WebShop")); assert len(launches) == 1   # reuse
    a.release(); assert kills == []                                                # 1 lease left
    a.release(); assert kills and kills[-1][1] == signal.SIGTERM                   # torn down


def test_not_ready_fails_and_kills(tmp_path, monkeypatch):
    kills = []
    monkeypatch.setattr("backend.services.runtime.env_adapters.webshop.os.kill",
                        lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr("backend.services.runtime.env_adapters.webshop.time.sleep", lambda *_: None)
    a = WebShopAdapter(AssetCache(tmp_path), server_launcher=lambda c, p: 9, probe=lambda u: False,
                       pid_alive=lambda pid: True, server_ready_timeout_s=0.01)
    r = a.provision(ProvisionCtx(display_name="WebShop"))
    assert not r.ok and r.exclusion.verified and kills and kills[-1][1] == signal.SIGTERM


def test_stale_pid_relaunches(tmp_path, monkeypatch):
    alive = {42: True}; launches = []
    monkeypatch.setattr("backend.services.runtime.env_adapters.webshop.os.kill", lambda p, s: None)
    a = WebShopAdapter(AssetCache(tmp_path), server_launcher=lambda c, p: (launches.append(1) or 42),
                       probe=lambda u: True, pid_alive=lambda pid: alive.get(pid, False))
    a.provision(ProvisionCtx(display_name="WebShop")); assert len(launches) == 1
    alive[42] = False; a.release(); alive[42] = True
    a.provision(ProvisionCtx(display_name="WebShop")); assert len(launches) == 2


def test_inprocess_path(tmp_path, monkeypatch):
    data = tmp_path / "d"; data.mkdir()
    monkeypatch.setenv("WEBSHOP_DATA_DIR", str(data))
    monkeypatch.delenv("WEBSHOP_PACKAGE_DIR", raising=False)
    a = WebShopAdapter(AssetCache(tmp_path), inprocess_smoke=lambda d: True)
    r = a.provision(ProvisionCtx(display_name="WebShop"))
    assert r.ok and r.as_env_vars() == {"WEBSHOP_DATA_DIR": str(data)} and r.base_url is None
