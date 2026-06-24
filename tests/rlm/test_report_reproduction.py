import json
from pathlib import Path


from backend.agents.rlm.report import _build_reproduction_block, _adaptation_delta


def _write_repo_spec(project_dir: Path, url="https://github.com/me/mine", mode="adapt"):
    rlm_state = project_dir / "rlm_state"
    rlm_state.mkdir(parents=True, exist_ok=True)
    (rlm_state / "repo_spec.json").write_text(json.dumps({
        "url": url, "source": "user", "mode": mode, "reason": "t",
        "commit_sha": "abc1234", "path": str(project_dir / "repo"),
        "clone_succeeded": True,
    }), encoding="utf-8")


def _write_success_experiment(project_dir: Path):
    (project_dir / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True, "metrics": {"accuracy": 0.9}, "experiment_run_id": "r1"}) + "\n",
        encoding="utf-8",
    )


def test_flag_off_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_USE_AUTHOR_REPO", raising=False)
    _write_repo_spec(tmp_path)
    assert _build_reproduction_block(tmp_path) is None


def test_no_repo_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    # repo_spec.json with url=None (scratch) -> no reproduction block
    rlm_state = tmp_path / "rlm_state"
    rlm_state.mkdir(parents=True)
    (rlm_state / "repo_spec.json").write_text(json.dumps({
        "url": None, "source": "none", "mode": "scratch", "reason": "x", "commit_sha": None,
    }), encoding="utf-8")
    assert _build_reproduction_block(tmp_path) is None


def test_execution_ran_true_with_success_row(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    _write_repo_spec(tmp_path)
    _write_success_experiment(tmp_path)
    block = _build_reproduction_block(tmp_path)
    assert block is not None
    assert block["mode"] == "adapt"
    assert block["repo_url"] == "https://github.com/me/mine"
    assert block["commit_sha"] == "abc1234"
    assert block["provider"] == "github"
    assert block["execution"]["ran"] is True
    assert block["execution"]["metrics_produced"] is True
    assert block["execution"]["status"] == "success"


def test_execution_ran_false_without_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    _write_repo_spec(tmp_path)
    block = _build_reproduction_block(tmp_path)
    assert block is not None
    assert block["execution"]["ran"] is False
    assert block["execution"]["status"] == "failed"


def test_clone_failed_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    rlm_state = tmp_path / "rlm_state"
    rlm_state.mkdir(parents=True)
    (rlm_state / "repo_spec.json").write_text(json.dumps({
        "url": "https://github.com/me/mine", "source": "user", "mode": "scratch",
        "reason": "clone failed", "commit_sha": None, "path": None,
        "clone_succeeded": False,
    }), encoding="utf-8")
    assert _build_reproduction_block(tmp_path) is None


def test_adaptation_delta_counts(tmp_path):
    repo = tmp_path / "repo"
    code = tmp_path / "code"
    (repo).mkdir(); (code).mkdir()
    (repo / "a.py").write_text("1\n", encoding="utf-8")
    (repo / "b.py").write_text("same\n", encoding="utf-8")
    (code / "a.py").write_text("CHANGED\n", encoding="utf-8")  # changed
    (code / "b.py").write_text("same\n", encoding="utf-8")     # unchanged
    (code / "c.py").write_text("new\n", encoding="utf-8")      # added
    # b.py present in both, a.py changed, c.py added, (no removed)
    delta = _adaptation_delta(repo, code)
    assert delta["files_changed"] == 1
    assert delta["files_added"] == 1
    assert delta["files_removed"] == 0
