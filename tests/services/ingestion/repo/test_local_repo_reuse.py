import subprocess
from pathlib import Path


from backend.services.ingestion.repo.provisioner import RepoProvisioner
from backend.services.ingestion.repo.resolver import RepoSpec


def _make_local_git_remote(tmp_path: Path, subdir: str = "remote_src") -> Path:
    """Create a real local git repo with one commit; return its filesystem path."""
    src = tmp_path / subdir
    src.mkdir()
    (src / "README.md").write_text("# fixture\n", encoding="utf-8")
    (src / "SENTINEL.txt").write_text("local-staged-copy\n", encoding="utf-8")
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e",
    }
    full_env = {**__import__("os").environ, **env}
    subprocess.run(["git", "init", "-q"], cwd=src, check=True, env=full_env)
    subprocess.run(["git", "add", "-A"], cwd=src, check=True, env=full_env)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=src, check=True, env=full_env)
    return src


def test_local_path_set_copies_local_repo_instead_of_cloning(tmp_path, monkeypatch):
    local_repo = _make_local_git_remote(tmp_path, "local_staged")
    monkeypatch.setenv("OPENRESEARCH_REPO_LOCAL_PATH", str(local_repo))
    # spec.url still names which repo (github url); it's not fetched over the
    # network because repo_local_path takes precedence.
    spec = RepoSpec(url="https://github.com/example/example", source="user", mode="adapt", reason="test")
    dest = tmp_path / "repo"

    manifest = RepoProvisioner.clone(spec, dest)

    assert manifest is not None
    assert (dest / "SENTINEL.txt").read_text() == "local-staged-copy\n"
    assert manifest.commit_sha and len(manifest.commit_sha) >= 7


def test_local_path_unset_uses_github_style_clone(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_REPO_LOCAL_PATH", raising=False)
    remote = _make_local_git_remote(tmp_path, "remote_src")
    spec = RepoSpec(url=remote.as_uri(), source="user", mode="adapt", reason="test")
    dest = tmp_path / "repo"

    manifest = RepoProvisioner.clone(spec, dest)

    assert manifest is not None
    assert (dest / "README.md").exists()
    assert manifest.commit_sha and len(manifest.commit_sha) >= 7


def test_local_path_with_commit_pin_checks_out_commit(tmp_path, monkeypatch):
    local_repo = _make_local_git_remote(tmp_path, "local_staged2")
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e",
    }
    full_env = {**__import__("os").environ, **env}
    first_rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=local_repo, check=True,
        capture_output=True, text=True, env=full_env,
    ).stdout.strip()
    # Add a second commit so HEAD moves past the pinned first commit.
    (local_repo / "SENTINEL.txt").write_text("second-commit\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=local_repo, check=True, env=full_env)
    subprocess.run(["git", "commit", "-q", "-m", "second"], cwd=local_repo, check=True, env=full_env)

    monkeypatch.setenv("OPENRESEARCH_REPO_LOCAL_PATH", str(local_repo))
    monkeypatch.setenv("OPENRESEARCH_REPO_COMMIT", first_rev)
    spec = RepoSpec(url="https://github.com/example/example", source="user", mode="adapt", reason="test")
    dest = tmp_path / "repo"

    manifest = RepoProvisioner.clone(spec, dest)

    assert manifest is not None
    assert manifest.commit_sha == first_rev
    assert (dest / "SENTINEL.txt").read_text() == "local-staged-copy\n"


def test_local_path_nonexistent_falls_back_to_github_clone(tmp_path, monkeypatch):
    # A stale/missing local_path must not break the run: fall back to the
    # normal clone path (fail-soft, never crash).
    monkeypatch.setenv("OPENRESEARCH_REPO_LOCAL_PATH", str(tmp_path / "does_not_exist"))
    remote = _make_local_git_remote(tmp_path, "remote_src3")
    spec = RepoSpec(url=remote.as_uri(), source="user", mode="adapt", reason="test")
    dest = tmp_path / "repo"

    manifest = RepoProvisioner.clone(spec, dest)

    assert manifest is not None
    assert (dest / "README.md").exists()
