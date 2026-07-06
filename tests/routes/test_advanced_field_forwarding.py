from backend.services.events.live_runs import StartRunRequest, FileLiveRunService


def _svc(tmp_path):
    # Build a *real* instance (not __new__). The brief's helper used
    # FileLiveRunService.__new__(FileLiveRunService) to skip __init__, but
    # _subprocess_env reads self.repo_root (live_runs.py:507, the .env-precedence
    # block) before it reaches the budget-field logic — a __new__ instance has no
    # repo_root and raised AttributeError. __init__ is cheap + hermetic (only
    # Path.resolve() on repo_root/runs_root); tmp_path has no .env → loads nothing.
    return FileLiveRunService(repo_root=tmp_path)


def test_subprocess_env_threads_budget_fields(tmp_path):
    req = StartRunRequest(dynamic_gpu=True, force_single_gpu=True,
                          max_gpu_usd_per_hour=7.5, vram_gb=24)
    env = FileLiveRunService._subprocess_env(_svc(tmp_path), req)
    assert env["OPENRESEARCH_DYNAMIC_GPU"] == "true"
    assert env["OPENRESEARCH_FORCE_SINGLE_GPU"] == "true"
    assert env["OPENRESEARCH_MAX_GPU_USD_PER_HOUR"] == "7.5"
    assert env["OPENRESEARCH_VRAM_OVERRIDE_GB"] == "24"


def test_subprocess_env_omits_unset_budget_fields(tmp_path, monkeypatch):
    # _subprocess_env seeds env from os.environ (:498), so a shell export of any
    # budget key would leak in and cause a false failure — clear them first.
    for k in ("OPENRESEARCH_DYNAMIC_GPU", "OPENRESEARCH_FORCE_SINGLE_GPU",
              "OPENRESEARCH_MAX_GPU_USD_PER_HOUR", "OPENRESEARCH_VRAM_OVERRIDE_GB"):
        monkeypatch.delenv(k, raising=False)
    env = FileLiveRunService._subprocess_env(_svc(tmp_path), StartRunRequest())
    for k in ("OPENRESEARCH_DYNAMIC_GPU", "OPENRESEARCH_FORCE_SINGLE_GPU",
              "OPENRESEARCH_MAX_GPU_USD_PER_HOUR", "OPENRESEARCH_VRAM_OVERRIDE_GB"):
        assert k not in env


def test_arxiv_request_declares_advanced_fields():
    from backend.app import StartArxivRunRequest
    r = StartArxivRunRequest(url="https://arxiv.org/abs/2605.15155",
                             dynamic_gpu=True, vram_gb=48)
    assert r.dynamic_gpu is True and r.vram_gb == 48
