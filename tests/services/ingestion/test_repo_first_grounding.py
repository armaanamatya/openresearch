"""Tests for the repo-first structure-only grounding module.

Flag-gated on the EXISTING ``OPENRESEARCH_USE_AUTHOR_REPO`` (feature_flags.py's
``use_author_repo``) — this module never re-resolves/clones, it only reads an
already-cloned repo dir from disk. Pure filesystem; no network sockets.
"""
import json

from backend.services.ingestion.repo_first_grounding import (
    ground_from_repo,
    repo_first_enabled,
    write_grounding,
)


def _make_fixture_repo(root):
    """A minimal author-repo fixture: train.py, requirements.txt, config.yaml, README.md."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (root / "requirements.txt").write_text(
        "torch==2.2.0\ntransformers>=4.30\n# a comment\n-e .\n\n", encoding="utf-8",
    )
    (root / "config.yaml").write_text("lr: 0.001\nbatch_size: 32\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# Example Paper Repo\n\nThis repository reproduces Example Paper.\n", encoding="utf-8",
    )
    return root


# --------------------------------------------------------------------------
# ground_from_repo
# --------------------------------------------------------------------------

def test_ground_from_repo_basic_structure(tmp_path):
    repo = _make_fixture_repo(tmp_path / "repo")
    result = ground_from_repo(repo)

    assert result["entry_points"] == ["train.py"]
    assert result["requirements"]["files"] == ["requirements.txt"]
    assert "torch" in result["requirements"]["packages"]
    assert "transformers" in result["requirements"]["packages"]
    assert result["key_configs"] == ["config.yaml"]
    assert "Example Paper Repo" in result["readme_summary"]
    assert result["module_tree"] == sorted(
        ["train.py", "requirements.txt", "config.yaml", "README.md"]
    )
    assert result["provenance"] == "author_repo"
    # Never asserts a reproduction pass — no verdict-shaped keys leak in.
    assert "verdict" not in result
    assert "meets_target" not in result


def test_ground_from_repo_is_sorted_and_deterministic(tmp_path):
    repo = _make_fixture_repo(tmp_path / "repo")
    first = ground_from_repo(repo)
    second = ground_from_repo(repo)
    assert first == second
    assert first["module_tree"] == sorted(first["module_tree"])
    assert first["entry_points"] == sorted(first["entry_points"])
    assert first["key_configs"] == sorted(first["key_configs"])
    assert first["requirements"]["files"] == sorted(first["requirements"]["files"])
    assert first["requirements"]["packages"] == sorted(first["requirements"]["packages"])


def test_ground_from_repo_bounded_by_max_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for i in range(50):
        (repo / f"data_{i:03d}.bin").write_text("x", encoding="utf-8")
    result = ground_from_repo(repo, max_files=10)
    assert len(result["module_tree"]) <= 10
    assert result["module_tree"] == sorted(
        f"data_{i:03d}.bin" for i in range(50)
    )[:10]


def test_ground_from_repo_entry_point_globs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ("train_gpt.py", "main.py", "run_experiment.py", "setup.py", "pyproject.toml"):
        (repo / name).write_text("# stub\n", encoding="utf-8")
    (repo / "utils.py").write_text("# not an entry point\n", encoding="utf-8")
    result = ground_from_repo(repo)
    assert result["entry_points"] == sorted(
        ["train_gpt.py", "main.py", "run_experiment.py", "setup.py", "pyproject.toml"]
    )
    assert "utils.py" not in result["entry_points"]


def test_ground_from_repo_pyproject_dependencies(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        "[project]\n"
        'name = "example"\n'
        'dependencies = ["torch>=2.0", "numpy==1.26.0"]\n',
        encoding="utf-8",
    )
    result = ground_from_repo(repo)
    assert "pyproject.toml" in result["requirements"]["files"]
    assert "torch" in result["requirements"]["packages"]
    assert "numpy" in result["requirements"]["packages"]
    # pyproject.toml is also a recognized entry-point config file.
    assert "pyproject.toml" in result["entry_points"]


def test_ground_from_repo_configs_top_two_levels_only(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "config.yaml").write_text("a: 1\n", encoding="utf-8")
    nested = repo / "configs"
    nested.mkdir()
    (nested / "model.yaml").write_text("b: 2\n", encoding="utf-8")
    deep = repo / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "too_deep.yaml").write_text("c: 3\n", encoding="utf-8")
    result = ground_from_repo(repo)
    assert "config.yaml" in result["key_configs"]
    assert "configs/model.yaml" in result["key_configs"]
    assert "a/b/c/too_deep.yaml" not in result["key_configs"]


def test_ground_from_repo_missing_dir_fails_soft(tmp_path):
    missing = tmp_path / "does-not-exist"
    result = ground_from_repo(missing)
    assert result["entry_points"] == []
    assert result["requirements"] == {"files": [], "packages": []}
    assert result["key_configs"] == []
    assert result["readme_summary"] == ""
    assert result["module_tree"] == []
    assert result["provenance"] == "author_repo"


# --------------------------------------------------------------------------
# repo_first_enabled
# --------------------------------------------------------------------------

def test_repo_first_enabled_default_off(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_USE_AUTHOR_REPO", raising=False)
    assert repo_first_enabled() is False


def test_repo_first_enabled_truthy(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    assert repo_first_enabled() is True


# --------------------------------------------------------------------------
# write_grounding
# --------------------------------------------------------------------------

def test_write_grounding_off_flag_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_USE_AUTHOR_REPO", raising=False)
    project_dir = tmp_path / "prj_off"
    repo_dir = _make_fixture_repo(project_dir / "repo")
    rlm_state = project_dir / "rlm_state"
    rlm_state.mkdir(parents=True)
    (rlm_state / "repo_spec.json").write_text(
        json.dumps({"path": str(repo_dir), "clone_succeeded": True}), encoding="utf-8",
    )
    before = sorted(p.relative_to(project_dir).as_posix() for p in project_dir.rglob("*") if p.is_file())

    result = write_grounding(project_dir)

    assert result is None
    assert not (rlm_state / "repo_grounding.json").exists()
    after = sorted(p.relative_to(project_dir).as_posix() for p in project_dir.rglob("*") if p.is_file())
    assert before == after  # byte-identical: nothing new written


def test_write_grounding_on_flag_uses_repo_spec_path(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    project_dir = tmp_path / "prj_on"
    repo_dir = _make_fixture_repo(project_dir / "repo")
    rlm_state = project_dir / "rlm_state"
    rlm_state.mkdir(parents=True)
    (rlm_state / "repo_spec.json").write_text(
        json.dumps({"path": str(repo_dir), "clone_succeeded": True}), encoding="utf-8",
    )

    result = write_grounding(project_dir)

    expected_path = rlm_state / "repo_grounding.json"
    assert result == expected_path
    assert expected_path.exists()
    written = json.loads(expected_path.read_text(encoding="utf-8"))
    assert written == ground_from_repo(repo_dir)
    assert written["provenance"] == "author_repo"


def test_write_grounding_on_flag_falls_back_to_code_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    project_dir = tmp_path / "prj_code_fallback"
    code_dir = _make_fixture_repo(project_dir / "code")
    # No repo_spec.json at all.

    result = write_grounding(project_dir)

    expected_path = project_dir / "rlm_state" / "repo_grounding.json"
    assert result == expected_path
    assert expected_path.exists()
    written = json.loads(expected_path.read_text(encoding="utf-8"))
    assert written == ground_from_repo(code_dir)


def test_write_grounding_on_flag_no_repo_present_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    project_dir = tmp_path / "prj_no_repo"
    project_dir.mkdir()

    result = write_grounding(project_dir)

    assert result is None
    assert not (project_dir / "rlm_state" / "repo_grounding.json").exists()


def test_write_grounding_idempotent_rewrite(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    project_dir = tmp_path / "prj_rewrite"
    repo_dir = _make_fixture_repo(project_dir / "repo")
    rlm_state = project_dir / "rlm_state"
    rlm_state.mkdir(parents=True)
    (rlm_state / "repo_spec.json").write_text(
        json.dumps({"path": str(repo_dir), "clone_succeeded": True}), encoding="utf-8",
    )

    first = write_grounding(project_dir)
    second = write_grounding(project_dir)

    assert first == second
    assert json.loads(first.read_text(encoding="utf-8")) == json.loads(second.read_text(encoding="utf-8"))
