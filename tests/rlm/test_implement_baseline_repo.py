import json
from pathlib import Path


from backend.agents.rlm.primitives import (
    _load_repo_spec,
    _seed_code_from_repo,
    _repo_artifact_index,
    _should_seed_code_from_repo,
)


def _write_repo_spec(project_dir: Path, **kw):
    rlm_state = project_dir / "rlm_state"
    rlm_state.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": "https://github.com/me/mine", "source": "user", "mode": "adapt",
        "reason": "test", "commit_sha": "abc1234",
        "path": str(project_dir / "repo"),
    }
    payload.update(kw)
    (rlm_state / "repo_spec.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_load_repo_spec_absent_returns_empty(tmp_path):
    assert _load_repo_spec(tmp_path) == {}


def test_load_repo_spec_reads_disk(tmp_path):
    _write_repo_spec(tmp_path)
    spec = _load_repo_spec(tmp_path)
    assert spec["url"] == "https://github.com/me/mine"
    assert spec["commit_sha"] == "abc1234"


def test_artifact_index_from_repo_spec_overrides_empty_plan(tmp_path):
    _write_repo_spec(tmp_path)
    ai = _repo_artifact_index(tmp_path, plan_artifact_index={})
    assert ai["repo_url"] == "https://github.com/me/mine"
    assert ai["commit_sha"] == "abc1234"
    assert ai["mode"] == "adapt"
    assert ai["path"].endswith("repo")


def test_seed_copies_tree_excluding_git(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: x", encoding="utf-8")
    (repo / "train.py").write_text("print(1)\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "m.py").write_text("x=1\n", encoding="utf-8")
    code = tmp_path / "code"
    n = _seed_code_from_repo(repo, code)
    assert n == 2  # train.py + src/m.py (NOT .git/HEAD)
    assert (code / "train.py").exists()
    assert (code / "src" / "m.py").exists()
    assert not (code / ".git").exists()


def test_should_seed_only_adapt_and_empty_code(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("x\n", encoding="utf-8")
    code = tmp_path / "code"
    code.mkdir()
    # adapt + empty code + repo present -> seed
    assert _should_seed_code_from_repo("adapt", repo, code) is True
    # reference mode -> never seed
    assert _should_seed_code_from_repo("reference", repo, code) is False
    # non-empty code (repair re-entry) -> never re-seed
    (code / "existing.py").write_text("y\n", encoding="utf-8")
    assert _should_seed_code_from_repo("adapt", repo, code) is False


def test_should_not_seed_without_repo(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    assert _should_seed_code_from_repo("adapt", tmp_path / "no_repo", code) is False
