
from backend.config import Settings


def test_repo_flag_defaults():
    s = Settings()
    assert s.use_author_repo is False
    assert s.reproduction_mode == "adapt"
    assert s.repo_clone_timeout_s == 300
    assert s.repo_clone_max_mb == 2048
    assert s.repo_clone_lfs is False


def test_repo_flags_read_openresearch_env(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "true")
    monkeypatch.setenv("OPENRESEARCH_REPRODUCTION_MODE", "reference")
    monkeypatch.setenv("OPENRESEARCH_REPO_CLONE_TIMEOUT_S", "120")
    monkeypatch.setenv("OPENRESEARCH_REPO_CLONE_MAX_MB", "512")
    monkeypatch.setenv("OPENRESEARCH_REPO_CLONE_LFS", "1")
    s = Settings()
    assert s.use_author_repo is True
    assert s.reproduction_mode == "reference"
    assert s.repo_clone_timeout_s == 120
    assert s.repo_clone_max_mb == 512
    assert s.repo_clone_lfs is True


def test_repo_flags_legacy_reprolab_bridge(monkeypatch):
    # The REPROLAB_* -> OPENRESEARCH_* bridge runs once at import. Set the legacy
    # var, then re-invoke the aliaser so the counterpart is filled, mirroring how
    # an operator who still exports REPROLAB_* before startup is handled.
    monkeypatch.setenv("REPROLAB_USE_AUTHOR_REPO", "true")
    import backend.config as cfg
    cfg._apply_legacy_env_aliases()
    assert cfg.Settings().use_author_repo is True
