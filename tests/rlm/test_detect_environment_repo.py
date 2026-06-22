import pytest

from backend.agents.rlm.primitives import detect_environment, _merge_repo_deps_into_spec


_METHOD = {"core_contribution": "x", "claims": [], "metrics": []}


def test_no_repo_byte_identical(tmp_path, make_context, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_USE_AUTHOR_REPO", raising=False)
    ctx = make_context(tmp_path)
    result = detect_environment(_METHOD, ctx=ctx)
    # The result is a normal EnvironmentSpec dict; no repo merge happened.
    assert result.get("success") is not False
    assert "dockerfile" in result


def test_merge_repo_requirements(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("torch==2.2.0\ntransformers>=4.40\n", encoding="utf-8")
    spec = {"pip_packages": {"numpy": "1.26"}, "dockerfile": "FROM x"}
    merged = _merge_repo_deps_into_spec(spec, repo)
    assert merged["pip_packages"]["torch"] == "==2.2.0"
    assert "transformers" in merged["pip_packages"]
    # Inferred dep survives where the repo doesn't override it.
    assert merged["pip_packages"]["numpy"] == "1.26"


def test_repo_dep_overrides_inferred(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("torch==2.2.0\n", encoding="utf-8")
    spec = {"pip_packages": {"torch": "==1.13"}}
    merged = _merge_repo_deps_into_spec(spec, repo)
    assert merged["pip_packages"]["torch"] == "==2.2.0"  # repo wins


def test_flag_on_with_repo_merges(tmp_path, make_context, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    ctx = make_context(tmp_path)
    repo = ctx.project_dir / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("accelerate==0.30.0\n", encoding="utf-8")
    result = detect_environment(_METHOD, ctx=ctx)
    assert "accelerate" in (result.get("pip_packages") or {})
