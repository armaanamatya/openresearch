import backend.agents.rlm.run as run_mod


def _fail_clone(spec, dest):
    return None


def test_execute_mode_clone_failure_stays_execute_and_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    monkeypatch.setenv("OPENRESEARCH_REPRODUCTION_MODE", "execute")
    monkeypatch.delenv("OPENRESEARCH_REPO_LOCAL_PATH", raising=False)
    monkeypatch.setattr(
        "backend.services.ingestion.repo.provisioner.RepoProvisioner.clone", _fail_clone,
    )
    project_dir = tmp_path / "prj"
    project_dir.mkdir()
    events = []

    repo_files, spec = run_mod._resolve_and_clone_repo(
        project_dir, "github:me/mine", set(), [], emit=events.append,
    )

    assert repo_files is None
    assert spec is not None
    assert spec.mode == "execute"  # NOT downgraded to scratch
    warning_codes = [e.get("code") for e in events if e.get("event") == "run_warning"]
    assert "repo_execute_unavailable" in warning_codes

    import json
    saved = json.loads((project_dir / "rlm_state" / "repo_spec.json").read_text())
    assert saved["mode"] == "execute"
    assert saved["clone_succeeded"] is False


def test_adapt_mode_clone_failure_still_downgrades_to_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    monkeypatch.delenv("OPENRESEARCH_REPRODUCTION_MODE", raising=False)  # default "adapt"
    monkeypatch.delenv("OPENRESEARCH_REPO_LOCAL_PATH", raising=False)
    monkeypatch.setattr(
        "backend.services.ingestion.repo.provisioner.RepoProvisioner.clone", _fail_clone,
    )
    project_dir = tmp_path / "prj"
    project_dir.mkdir()
    events = []

    repo_files, spec = run_mod._resolve_and_clone_repo(
        project_dir, "github:me/mine", set(), [], emit=events.append,
    )

    assert repo_files is None
    assert spec is not None
    assert spec.mode == "scratch"  # existing byte-identical downgrade behavior
    warning_codes = [e.get("code") for e in events if e.get("event") == "run_warning"]
    assert "repo_clone_failed" in warning_codes
    assert "repo_execute_unavailable" not in warning_codes
