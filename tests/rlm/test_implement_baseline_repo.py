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
        "clone_succeeded": True,
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


def test_artifact_index_from_repo_spec_overrides_empty_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    _write_repo_spec(tmp_path)
    ai = _repo_artifact_index(tmp_path, plan_artifact_index={})
    assert ai["repo_url"] == "https://github.com/me/mine"
    assert ai["commit_sha"] == "abc1234"
    assert ai["mode"] == "adapt"
    assert ai["path"].endswith("repo")


def test_artifact_index_flag_off_returns_plan(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_USE_AUTHOR_REPO", raising=False)
    _write_repo_spec(tmp_path)
    ai = _repo_artifact_index(tmp_path, {"x": 1})
    assert ai == {"x": 1}


def test_seed_skips_escaping_symlink(tmp_path):
    # Real file inside repo + a symlink that points outside repo/
    repo = tmp_path / "repo"
    repo.mkdir()
    real_file = repo / "train.py"
    real_file.write_text("print(1)\n", encoding="utf-8")
    # Target outside the repo
    outside = tmp_path / "secret.txt"
    outside.write_text("secret\n", encoding="utf-8")
    evil_link = repo / "evil.py"
    evil_link.symlink_to(outside)

    code = tmp_path / "code"
    n = _seed_code_from_repo(repo, code)
    assert n == 1  # only train.py, not the symlink target
    assert (code / "train.py").exists()
    assert not (code / "evil.py").exists()


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


def test_should_seed_execute_mode_like_adapt(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("x\n", encoding="utf-8")
    code = tmp_path / "code"
    code.mkdir()
    # execute + empty code + repo present -> seed (same as adapt)
    assert _should_seed_code_from_repo("execute", repo, code) is True
    # non-empty code (repair re-entry) -> never re-seed
    (code / "existing.py").write_text("y\n", encoding="utf-8")
    assert _should_seed_code_from_repo("execute", repo, code) is False


def test_should_seed_case_and_padding_insensitive(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("x\n", encoding="utf-8")
    code = tmp_path / "code"
    code.mkdir()
    assert _should_seed_code_from_repo("  EXECUTE  ", repo, code) is True
    assert _should_seed_code_from_repo("  Adapt  ", repo, code) is True
    assert _should_seed_code_from_repo("  Reference  ", repo, code) is False


def test_should_not_seed_without_repo(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    assert _should_seed_code_from_repo("adapt", tmp_path / "no_repo", code) is False


# ---------------------------------------------------------------------------
# implement_baseline cache-key sensitivity to the persisted repo mode
# ---------------------------------------------------------------------------


class _FakeBaselineResult:
    commands_to_run = ["python train.py"]


def _write_minimal_code(project_id, runs_root):
    code_dir = runs_root / project_id / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")


def test_implement_baseline_cache_key_changes_with_repo_mode(make_context, tmp_path, monkeypatch):
    """Switching the persisted repo_spec.json mode (adapt -> execute) between
    attempts with an otherwise-identical plan must NOT serve a stale cached
    implementation from the prior mode."""
    import backend.agents.rlm.primitives as primitives
    from backend.agents.rlm.primitives import implement_baseline

    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    ctx = make_context(tmp_path)
    ctx.runtime = object()
    _write_repo_spec(ctx.project_dir, mode="adapt")

    call_count = {"n": 0}

    async def fake_run_with_sdk(project_id, runs_root, pcm, env, contract,
                                 artifact_index, **kw):
        call_count["n"] += 1
        _write_minimal_code(project_id, runs_root)
        return _FakeBaselineResult()

    monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)

    plan = {
        "paper_claim_map": {"core_contribution": "x"},
        "environment_spec": {},
        "reproduction_contract": None,
    }
    implement_baseline(plan, ctx=ctx)
    assert call_count["n"] == 1

    # Same plan, same repair_context (None) — normally a cache HIT — but the
    # persisted repo mode flipped underneath, so the cache key must differ.
    _write_repo_spec(ctx.project_dir, mode="execute")
    implement_baseline(plan, ctx=ctx)
    assert call_count["n"] == 2, (
        "A repo-mode change (adapt -> execute) must invalidate the "
        "implement_baseline cache entry, not serve the prior mode's result"
    )


def test_implement_baseline_cache_hit_when_repo_mode_unchanged(make_context, tmp_path, monkeypatch):
    """Sanity counterpart: an identical repo mode across two calls still hits
    cache (the mode key alone doesn't defeat caching when nothing changed)."""
    import backend.agents.rlm.primitives as primitives
    from backend.agents.rlm.primitives import implement_baseline

    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    ctx = make_context(tmp_path)
    ctx.runtime = object()
    _write_repo_spec(ctx.project_dir, mode="execute")

    call_count = {"n": 0}

    async def fake_run_with_sdk(project_id, runs_root, pcm, env, contract,
                                 artifact_index, **kw):
        call_count["n"] += 1
        _write_minimal_code(project_id, runs_root)
        return _FakeBaselineResult()

    monkeypatch.setattr(primitives, "_run_baseline_with_sdk", fake_run_with_sdk)

    plan = {
        "paper_claim_map": {"core_contribution": "x"},
        "environment_spec": {},
        "reproduction_contract": None,
    }
    implement_baseline(plan, ctx=ctx)
    implement_baseline(plan, ctx=ctx)
    assert call_count["n"] == 1
