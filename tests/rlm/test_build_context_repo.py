import json


import backend.agents.rlm.run as run_mod
from backend.services.ingestion.repo.manifest import RepoManifest


_WCM = {
    "entries": [{"title": "Intro", "source_id": "s1"}],
    "paper_id": "2605.15155",
    "paper_title": "SDAR",
    "rubric_spec": {},
}


def test_flag_off_repo_files_is_none_no_clone(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_USE_AUTHOR_REPO", raising=False)
    project_dir = tmp_path / "prj"
    project_dir.mkdir()
    called = {"clone": 0}
    monkeypatch.setattr(
        "backend.services.ingestion.repo.provisioner.RepoProvisioner.clone",
        lambda spec, dest: called.__setitem__("clone", called["clone"] + 1) or None,
    )
    ctx = run_mod._build_context(
        _WCM, project_dir=project_dir, repo_url="github:ZJU-REAL/SDAR",
        blacklist=set(), discovered=[],
    )
    assert ctx["repo_files"] is None
    assert called["clone"] == 0
    assert not (project_dir / "rlm_state" / "repo_spec.json").exists()


def test_flag_on_user_url_populates_repo_files_and_spec(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    project_dir = tmp_path / "prj"
    project_dir.mkdir()

    def _fake_clone(spec, dest):
        assert spec.url == "https://github.com/me/mine"
        from pathlib import Path
        Path(dest).mkdir(parents=True, exist_ok=True)
        return RepoManifest(
            path=str(dest), commit_sha="abc1234",
            file_tree=["train.py"], key_files={"README.md": "# x"},
            size_mb=0.1, lfs_skipped=True,
        )

    monkeypatch.setattr(
        "backend.services.ingestion.repo.provisioner.RepoProvisioner.clone", _fake_clone,
    )
    ctx = run_mod._build_context(
        _WCM, project_dir=project_dir, repo_url="github:me/mine",
        blacklist=set(), discovered=[],
    )
    assert ctx["repo_files"] is not None
    assert ctx["repo_files"]["commit_sha"] == "abc1234"
    spec_path = project_dir / "rlm_state" / "repo_spec.json"
    assert spec_path.exists()
    saved = json.loads(spec_path.read_text())
    assert saved["url"] == "https://github.com/me/mine"
    assert saved["source"] == "user"
    assert saved["commit_sha"] == "abc1234"


def test_flag_on_no_repo_resolved_repo_files_none(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    project_dir = tmp_path / "prj"
    project_dir.mkdir()
    ctx = run_mod._build_context(
        _WCM, project_dir=project_dir, repo_url=None, blacklist=set(), discovered=[],
    )
    assert ctx["repo_files"] is None
    # A scratch spec is still persisted (provenance), but with url=None.
    spec_path = project_dir / "rlm_state" / "repo_spec.json"
    assert spec_path.exists()
    assert json.loads(spec_path.read_text())["url"] is None


def test_build_context_default_args_byte_identical(tmp_path, monkeypatch):
    # Calling _build_context with ONLY the legacy positional arg behaves as before.
    monkeypatch.delenv("OPENRESEARCH_USE_AUTHOR_REPO", raising=False)
    ctx = run_mod._build_context(_WCM)
    assert ctx["repo_files"] is None
    assert set(ctx) == {
        "paper_text", "paper_metadata", "supplementary_text",
        "repo_files", "prior_work_refs", "rubric_spec",
    }
