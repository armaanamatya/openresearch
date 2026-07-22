"""Phase 6 / Task 13 — keep runs/<id>/repo/ host-only.

Asserts that every cloud upload path excludes repo/ from the upload set.
Tests are socket-hermetic (no network I/O); clients are duck-typed fakes.
"""

from pathlib import Path



def _make_tree(root: Path):
    (root / "code").mkdir()
    (root / "code" / "train.py").write_text("x\n", encoding="utf-8")
    (root / "repo" / "src").mkdir(parents=True)
    (root / "repo" / "src" / "model.py").write_text("y\n", encoding="utf-8")
    (root / "repo" / "README.md").write_text("z\n", encoding="utf-8")


class _FakeContainer:
    def __init__(self):
        self.blobs: dict[str, bytes] = {}

    def upload_blob(self, name, data, overwrite=True):
        self.blobs[name] = data


def test_azure_excludes_repo(tmp_path):
    from backend.services.runtime import azure_blob
    assert "repo" in azure_blob._EXCLUDED_DIR_PARTS
    _make_tree(tmp_path)
    client = _FakeContainer()
    uploaded = azure_blob.upload_prefix(
        tmp_path, blob_prefix="runs/x", account_name="a", container_name="c", client=client,
    )
    assert any("code/train.py" in n for n in uploaded)
    assert not any("/repo/" in n or n.endswith("/repo") for n in uploaded)


def test_gcs_excludes_repo(tmp_path):
    from backend.services.runtime import gcs_blob
    assert "repo" in gcs_blob._EXCLUDED_DIR_PARTS
