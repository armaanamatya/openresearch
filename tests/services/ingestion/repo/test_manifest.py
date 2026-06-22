import json

from backend.services.ingestion.repo.manifest import (
    MAX_CONTEXT_BYTES,
    MAX_DEPTH,
    MAX_FILES,
    build_manifest,
)


def _make_repo(tmp_path):
    (tmp_path / "README.md").write_text("# SDAR\nrun: python train.py\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("torch==2.2.0\ntransformers\n", encoding="utf-8")
    (tmp_path / "train.py").write_text("def main():\n    pass\n", encoding="utf-8")
    sub = tmp_path / "src" / "sdar"
    sub.mkdir(parents=True)
    (sub / "model.py").write_text("class M: ...\n", encoding="utf-8")
    return tmp_path


def test_key_files_detected(tmp_path):
    repo = _make_repo(tmp_path)
    m = build_manifest(repo, commit_sha="abc1234")
    assert "README.md" in m.key_files
    assert "requirements.txt" in m.key_files
    assert "train.py" in m.key_files
    # Excerpt of README is captured (non-empty), not the whole file blindly.
    assert "SDAR" in m.key_files["README.md"]


def test_file_tree_capped_files_and_depth(tmp_path):
    repo = tmp_path
    # 250 shallow files -> capped to MAX_FILES.
    for i in range(250):
        (repo / f"f{i}.py").write_text("x = 1\n", encoding="utf-8")
    # A file deeper than MAX_DEPTH must be excluded.
    deep = repo / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    (deep / "too_deep.py").write_text("y = 2\n", encoding="utf-8")
    m = build_manifest(repo, commit_sha=None)
    assert len(m.file_tree) <= MAX_FILES
    assert all(p.count("/") < MAX_DEPTH for p in m.file_tree)
    assert "a/b/c/d/e/too_deep.py" not in m.file_tree


def test_as_context_under_byte_ceiling(tmp_path):
    repo = tmp_path
    for i in range(MAX_FILES + 50):
        (repo / f"file_{i:04d}.py").write_text("z = 3\n" * 50, encoding="utf-8")
    (repo / "README.md").write_text("R" * 100_000, encoding="utf-8")
    m = build_manifest(repo, commit_sha="deadbee")
    ctx = m.as_context()
    assert isinstance(ctx, dict)
    encoded = json.dumps(ctx)
    assert len(encoded.encode("utf-8")) <= MAX_CONTEXT_BYTES
    # Provenance survives the truncation.
    assert ctx["commit_sha"] == "deadbee"


def test_as_context_round_trips_simple(tmp_path):
    repo = _make_repo(tmp_path)
    m = build_manifest(repo, commit_sha="abc1234", size_mb=1.5, lfs_skipped=True)
    ctx = m.as_context()
    assert ctx["commit_sha"] == "abc1234"
    assert ctx["size_mb"] == 1.5
    assert ctx["lfs_skipped"] is True
    assert "train.py" in ctx["file_tree"]
