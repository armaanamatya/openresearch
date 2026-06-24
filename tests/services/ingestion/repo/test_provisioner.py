import subprocess
from pathlib import Path


from backend.services.ingestion.repo.provisioner import RepoProvisioner
from backend.services.ingestion.repo.resolver import RepoSpec


def _make_local_git_remote(tmp_path: Path) -> str:
    """Create a real local git repo with one commit; return its file:// URL."""
    src = tmp_path / "remote_src"
    src.mkdir()
    (src / "README.md").write_text("# fixture\n", encoding="utf-8")
    (src / "requirements.txt").write_text("torch\n", encoding="utf-8")
    (src / "train.py").write_text("print('hi')\n", encoding="utf-8")
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e",
    }
    subprocess.run(["git", "init", "-q"], cwd=src, check=True, env={**__import__("os").environ, **env})
    subprocess.run(["git", "add", "-A"], cwd=src, check=True, env={**__import__("os").environ, **env})
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=src, check=True, env={**__import__("os").environ, **env})
    return src.as_uri()  # file:///... — NEVER the network (suite is socket-hermetic)


def test_clone_success_returns_manifest_with_commit_sha(tmp_path):
    remote = _make_local_git_remote(tmp_path)
    spec = RepoSpec(url=remote, source="user", mode="adapt", reason="test")
    dest = tmp_path / "repo"
    manifest = RepoProvisioner.clone(spec, dest)
    assert manifest is not None
    assert (dest / "README.md").exists()
    assert (dest / "train.py").exists()
    assert manifest.commit_sha and len(manifest.commit_sha) >= 7
    assert "README.md" in manifest.key_files


def test_clone_nonexistent_path_returns_none(tmp_path):
    spec = RepoSpec(
        url=(tmp_path / "does_not_exist").as_uri(), source="user", mode="adapt", reason="test",
    )
    assert RepoProvisioner.clone(spec, tmp_path / "repo") is None


def test_clone_oversize_returns_none(tmp_path, monkeypatch):
    # Default cap is 2048 MB; simulate an oversize clone by forcing the measured
    # size above it (we no longer abuse max_mb=0, which now DISABLES the cap).
    from backend.services.ingestion.repo import provisioner as _prov
    monkeypatch.setattr(_prov, "_dir_size_mb", lambda _p: 99999.0)
    remote = _make_local_git_remote(tmp_path)
    spec = RepoSpec(url=remote, source="user", mode="adapt", reason="test")
    dest = tmp_path / "repo"
    assert RepoProvisioner.clone(spec, dest) is None
    # Oversize clone is discarded.
    assert not dest.exists()


def test_clone_max_mb_zero_disables_cap(tmp_path, monkeypatch):
    # 0 = disabled cap (codebase convention): a huge measured size is still allowed.
    from backend.services.ingestion.repo import provisioner as _prov
    monkeypatch.setattr(_prov, "_dir_size_mb", lambda _p: 99999.0)
    monkeypatch.setenv("OPENRESEARCH_REPO_CLONE_MAX_MB", "0")
    remote = _make_local_git_remote(tmp_path)
    spec = RepoSpec(url=remote, source="user", mode="adapt", reason="test")
    dest = tmp_path / "repo"
    manifest = RepoProvisioner.clone(spec, dest)
    assert manifest is not None  # cap disabled -> not rejected despite the huge size
    assert (dest / "README.md").exists()


def test_clone_none_url_returns_none(tmp_path):
    spec = RepoSpec(url=None, source="none", mode="scratch", reason="no repo")
    assert RepoProvisioner.clone(spec, tmp_path / "repo") is None
