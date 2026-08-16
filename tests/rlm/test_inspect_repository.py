
from backend.agents.rlm.primitives import (
    PRIMITIVE_REGISTRY,
    PRIMITIVE_DESCRIPTIONS,
    inspect_repository,
)


def test_disabled_when_flag_off(tmp_path, make_context, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_USE_AUTHOR_REPO", raising=False)
    ctx = make_context(tmp_path)
    assert inspect_repository(ctx=ctx) == {"status": "disabled"}


def test_lists_dir_when_flag_on(tmp_path, make_context, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    ctx = make_context(tmp_path)
    repo = ctx.project_dir / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "train.py").write_text("print(1)\n", encoding="utf-8")
    out = inspect_repository(ctx=ctx)
    assert out["status"] == "ok"
    assert out["kind"] == "dir"
    assert "train.py" in out["entries"]


def test_reads_file_bounded(tmp_path, make_context, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    ctx = make_context(tmp_path)
    repo = ctx.project_dir / "repo"
    repo.mkdir(parents=True)
    (repo / "big.txt").write_text("A" * 100, encoding="utf-8")
    out = inspect_repository(path="big.txt", max_bytes=10, ctx=ctx)
    assert out["status"] == "ok"
    assert out["kind"] == "file"
    assert len(out["content"]) <= 10
    assert out["truncated"] is True


def test_path_escape_rejected(tmp_path, make_context, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_USE_AUTHOR_REPO", "1")
    ctx = make_context(tmp_path)
    (ctx.project_dir / "repo").mkdir(parents=True)
    out = inspect_repository(path="../../etc/passwd", ctx=ctx)
    assert out["status"] == "error"


def test_registry_includes_inspect_repository():
    assert "inspect_repository" in PRIMITIVE_REGISTRY
    assert "inspect_repository" in PRIMITIVE_DESCRIPTIONS
    assert len(PRIMITIVE_REGISTRY) == 21
