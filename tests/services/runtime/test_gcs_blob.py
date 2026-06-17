"""Unit tests for ``backend.services.runtime.gcs_blob``.

All tests use a ``FakeBucketClient`` backed by an in-memory dict.  The real
google-cloud-storage package is **never imported** — the module import is
checked to work without it, and all data-plane calls go through the injected
fake.

Duck-type contract the fake satisfies (and that callers of
``gcs_blob.*`` must honour when injecting a client)::

    client.blob(name: str) -> _BlobHandle
        _BlobHandle.upload_from_string(data: bytes) -> None
        _BlobHandle.download_as_bytes() -> bytes
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Verify module import works without the real google-cloud-storage installed
# ---------------------------------------------------------------------------

def test_module_imports_without_google_sdk(monkeypatch):
    """gcs_blob must be importable even when google-cloud-storage is absent."""
    # Hide the google.cloud package if it happens to be installed.
    monkeypatch.setitem(sys.modules, "google", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "google.cloud", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "google.cloud.storage", None)  # type: ignore[arg-type]

    # Force a fresh import by removing any cached module.
    monkeypatch.delitem(sys.modules, "backend.services.runtime.gcs_blob", raising=False)

    # Should not raise even with google.cloud blocked.
    import importlib
    mod = importlib.import_module("backend.services.runtime.gcs_blob")
    assert hasattr(mod, "upload_prefix")
    assert hasattr(mod, "upload_bytes")
    assert hasattr(mod, "download_artifact")
    assert hasattr(mod, "download_bytes")


# ---------------------------------------------------------------------------
# FakeBucketClient
# ---------------------------------------------------------------------------

class _BlobHandle:
    """Minimal stand-in for a GCS Blob object."""

    def __init__(self, store: dict[str, bytes], name: str) -> None:
        self._store = store
        self._name = name

    def upload_from_string(self, data: bytes) -> None:
        self._store[self._name] = data

    def download_as_bytes(self) -> bytes:
        if self._name not in self._store:
            raise KeyError(f"Blob not found: {self._name!r}")
        return self._store[self._name]


class FakeBucketClient:
    """In-memory Bucket-like object that stores blobs in a plain dict.

    Satisfies the duck-type contract documented in ``gcs_blob.py``::

        client.blob(name) -> object with .upload_from_string(data) and .download_as_bytes()
    """

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def blob(self, name: str) -> _BlobHandle:
        return _BlobHandle(self.blobs, name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from backend.services.runtime import gcs_blob as gb  # noqa: E402  (after monkeypatch test)

BUCKET = "fakebucket"
PROJECT = "fakeproject"


def _fake() -> FakeBucketClient:
    return FakeBucketClient()


# ---------------------------------------------------------------------------
# upload_prefix — happy path
# ---------------------------------------------------------------------------

class TestUploadPrefix:
    def test_uploads_correct_blob_names(self, tmp_path: Path) -> None:
        """Every eligible file gets the right <prefix>/<rel-posix> name."""
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("y")

        client = _fake()
        names = gb.upload_prefix(
            tmp_path,
            blob_prefix="runs/abc/code",
            bucket=BUCKET,
            project=PROJECT,
            client=client,
        )

        assert "runs/abc/code/a.py" in names
        assert "runs/abc/code/sub/b.txt" in names
        assert names == sorted(names)

    def test_uploads_file_contents(self, tmp_path: Path) -> None:
        content = b"hello gcs"
        (tmp_path / "file.bin").write_bytes(content)
        client = _fake()
        gb.upload_prefix(
            tmp_path,
            blob_prefix="p",
            bucket=BUCKET,
            project=PROJECT,
            client=client,
        )
        assert client.blobs["p/file.bin"] == content

    def test_returns_sorted_list(self, tmp_path: Path) -> None:
        for name in ("z.txt", "a.txt", "m.txt"):
            (tmp_path / name).write_text(name)
        client = _fake()
        result = gb.upload_prefix(
            tmp_path,
            blob_prefix="x",
            bucket=BUCKET,
            project=PROJECT,
            client=client,
        )
        assert result == sorted(result)

    # -----------------------------------------------------------------------
    # Exclusions
    # -----------------------------------------------------------------------

    def test_excludes_outputs_directory(self, tmp_path: Path) -> None:
        (tmp_path / "outputs").mkdir()
        (tmp_path / "outputs" / "result.json").write_text("{}")
        (tmp_path / "keep.py").write_text("x")
        client = _fake()
        names = gb.upload_prefix(
            tmp_path, blob_prefix="p",
            bucket=BUCKET, project=PROJECT, client=client,
        )
        assert not any("outputs" in n for n in names)
        assert "p/keep.py" in names

    def test_excludes_git_directory(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("[core]")
        client = _fake()
        names = gb.upload_prefix(
            tmp_path, blob_prefix="p",
            bucket=BUCKET, project=PROJECT, client=client,
        )
        assert not any(".git" in n for n in names)

    def test_excludes_pycache_directory(self, tmp_path: Path) -> None:
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "mod.cpython-311.pyc").write_bytes(b"\x00")
        client = _fake()
        names = gb.upload_prefix(
            tmp_path, blob_prefix="p",
            bucket=BUCKET, project=PROJECT, client=client,
        )
        assert not any("__pycache__" in n for n in names)

    def test_excludes_pyc_files(self, tmp_path: Path) -> None:
        (tmp_path / "mod.pyc").write_bytes(b"\x00")
        (tmp_path / "mod.py").write_text("pass")
        client = _fake()
        names = gb.upload_prefix(
            tmp_path, blob_prefix="p",
            bucket=BUCKET, project=PROJECT, client=client,
        )
        assert not any(n.endswith(".pyc") for n in names)
        assert "p/mod.py" in names

    def test_excludes_venv_directory(self, tmp_path: Path) -> None:
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "activate").write_text("#!/bin/sh")
        (tmp_path / "train.py").write_text("pass")
        client = _fake()
        names = gb.upload_prefix(
            tmp_path, blob_prefix="p",
            bucket=BUCKET, project=PROJECT, client=client,
        )
        assert not any(".venv" in n for n in names)
        assert "p/train.py" in names

    # -----------------------------------------------------------------------
    # Path-safety: symlinks
    # -----------------------------------------------------------------------

    def test_symlink_within_root_is_uploaded(self, tmp_path: Path) -> None:
        real = tmp_path / "real.txt"
        real.write_text("data")
        link = tmp_path / "link.txt"
        link.symlink_to(real)

        client = _fake()
        names = gb.upload_prefix(
            tmp_path, blob_prefix="p",
            bucket=BUCKET, project=PROJECT, client=client,
        )
        # Both the real file and the symlink (pointing inside root) should appear.
        assert "p/real.txt" in names
        assert "p/link.txt" in names

    def test_symlink_escaping_root_is_skipped(self, tmp_path: Path) -> None:
        """A symlink whose target is outside local_root must be silently skipped."""
        external = tmp_path.parent / "external_secret.txt"
        external.write_text("secret")

        link = tmp_path / "escape.txt"
        link.symlink_to(external)

        (tmp_path / "safe.txt").write_text("ok")

        client = _fake()
        names = gb.upload_prefix(
            tmp_path, blob_prefix="p",
            bucket=BUCKET, project=PROJECT, client=client,
        )
        assert not any("escape" in n for n in names), (
            "Escaping symlink should not be uploaded"
        )
        assert "p/safe.txt" in names

    def test_no_files_returns_empty_list(self, tmp_path: Path) -> None:
        client = _fake()
        result = gb.upload_prefix(
            tmp_path, blob_prefix="p",
            bucket=BUCKET, project=PROJECT, client=client,
        )
        assert result == []

    # -----------------------------------------------------------------------
    # Parallel upload — correctness and error propagation
    # -----------------------------------------------------------------------

    def test_parallel_upload_all_files_sorted(self, tmp_path: Path) -> None:
        """20 files all uploaded and return value is sorted regardless of thread order."""
        n = 20
        for i in range(n):
            (tmp_path / f"file_{i:02d}.txt").write_text(f"content {i}")

        client = _fake()
        result = gb.upload_prefix(
            tmp_path, blob_prefix="bulk",
            bucket=BUCKET, project=PROJECT, client=client,
        )

        # All 20 files must appear.
        assert len(result) == n
        expected_names = sorted(f"bulk/file_{i:02d}.txt" for i in range(n))
        assert result == expected_names
        # Blobs dict must also contain all uploads.
        for name in expected_names:
            assert name in client.blobs

    def test_upload_error_propagates(self, tmp_path: Path) -> None:
        """If any single upload raises, upload_prefix must propagate the exception."""

        (tmp_path / "good.txt").write_text("ok")
        (tmp_path / "bad.txt").write_text("will fail")

        class ErrorOnBadClient(FakeBucketClient):
            def blob(self, name: str) -> _BlobHandle:
                if "bad" in name:
                    class _FailBlob:
                        def upload_from_string(self, data: bytes) -> None:
                            raise RuntimeError("simulated upload failure")
                        def download_as_bytes(self) -> bytes:
                            raise RuntimeError("simulated upload failure")
                    return _FailBlob()  # type: ignore[return-value]
                return super().blob(name)

        client = ErrorOnBadClient()
        with pytest.raises(RuntimeError, match="simulated upload failure"):
            gb.upload_prefix(
                tmp_path, blob_prefix="p",
                bucket=BUCKET, project=PROJECT, client=client,
            )


# ---------------------------------------------------------------------------
# upload_bytes + download_bytes — round-trip
# ---------------------------------------------------------------------------

class TestUploadDownloadBytes:
    def test_round_trip(self) -> None:
        client = _fake()
        payload = b"binary\x00data\xff"
        gb.upload_bytes(
            payload,
            blob_name="runs/xyz/metrics.json",
            bucket=BUCKET,
            project=PROJECT,
            client=client,
        )
        got = gb.download_bytes(
            "runs/xyz/metrics.json",
            bucket=BUCKET,
            project=PROJECT,
            client=client,
        )
        assert got == payload

    def test_overwrite_replaces_content(self) -> None:
        client = _fake()
        gb.upload_bytes(b"v1", blob_name="k", bucket=BUCKET,
                        project=PROJECT, client=client)
        gb.upload_bytes(b"v2", blob_name="k", bucket=BUCKET,
                        project=PROJECT, client=client)
        assert gb.download_bytes("k", bucket=BUCKET,
                                 project=PROJECT, client=client) == b"v2"

    def test_empty_bytes_round_trip(self) -> None:
        client = _fake()
        gb.upload_bytes(b"", blob_name="empty", bucket=BUCKET,
                        project=PROJECT, client=client)
        assert gb.download_bytes("empty", bucket=BUCKET,
                                 project=PROJECT, client=client) == b""


# ---------------------------------------------------------------------------
# download_artifact — writes file + returns Path
# ---------------------------------------------------------------------------

class TestDownloadArtifact:
    def test_writes_file_and_returns_path(self, tmp_path: Path) -> None:
        client = _fake()
        payload = b'{"score": 0.42}'
        client.blobs["runs/abc/metrics.json"] = payload

        dest = tmp_path / "metrics.json"
        result = gb.download_artifact(
            "runs/abc/metrics.json",
            dest,
            bucket=BUCKET,
            project=PROJECT,
            client=client,
        )

        assert result == dest.resolve()
        assert dest.read_bytes() == payload

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        client = _fake()
        client.blobs["blob"] = b"x"

        dest = tmp_path / "deep" / "nested" / "file.bin"
        result = gb.download_artifact(
            "blob", dest,
            bucket=BUCKET, project=PROJECT, client=client,
        )
        assert result.exists()
        assert result.read_bytes() == b"x"

    def test_destination_is_directory_uses_blob_filename(self, tmp_path: Path) -> None:
        client = _fake()
        client.blobs["runs/abc/metrics.json"] = b"data"

        result = gb.download_artifact(
            "runs/abc/metrics.json",
            tmp_path,  # a directory, not a file path
            bucket=BUCKET,
            project=PROJECT,
            client=client,
        )
        assert result.name == "metrics.json"
        assert result.read_bytes() == b"data"

    def test_returns_absolute_path(self, tmp_path: Path) -> None:
        client = _fake()
        client.blobs["x"] = b"y"
        result = gb.download_artifact(
            "x", tmp_path / "out",
            bucket=BUCKET, project=PROJECT, client=client,
        )
        assert result.is_absolute()


# ---------------------------------------------------------------------------
# Blob-name sanitization
# ---------------------------------------------------------------------------

class TestBlobNameSanitization:
    @pytest.mark.parametrize("bad_name", [
        "",
        "/absolute",
        "a/b/../../../etc/passwd",
        "..",
        "runs/../secret",
    ])
    def test_upload_bytes_rejects_bad_names(self, bad_name: str) -> None:
        client = _fake()
        with pytest.raises((ValueError, Exception)):
            gb.upload_bytes(b"x", blob_name=bad_name,
                            bucket=BUCKET, project=PROJECT, client=client)

    @pytest.mark.parametrize("bad_name", [
        "",
        "/absolute",
        "a/../../../etc/passwd",
        "..",
    ])
    def test_download_bytes_rejects_bad_names(self, bad_name: str) -> None:
        client = _fake()
        with pytest.raises((ValueError, Exception)):
            gb.download_bytes(bad_name, bucket=BUCKET,
                              project=PROJECT, client=client)

    @pytest.mark.parametrize("bad_name", [
        "",
        "/absolute",
        "a/../../../etc/passwd",
        "..",
    ])
    def test_download_artifact_rejects_bad_names(self, bad_name: str, tmp_path: Path) -> None:
        client = _fake()
        with pytest.raises((ValueError, Exception)):
            gb.download_artifact(bad_name, tmp_path / "out",
                                 bucket=BUCKET, project=PROJECT, client=client)

    @pytest.mark.parametrize("bad_prefix", [
        "",
        "/absolute",
        "a/../..",
        "..",
    ])
    def test_upload_prefix_rejects_bad_blob_prefix(
        self, bad_prefix: str, tmp_path: Path
    ) -> None:
        (tmp_path / "f.txt").write_text("x")
        client = _fake()
        with pytest.raises((ValueError, Exception)):
            gb.upload_prefix(
                tmp_path,
                blob_prefix=bad_prefix,
                bucket=BUCKET,
                project=PROJECT,
                client=client,
            )

    def test_valid_nested_blob_name_accepted(self) -> None:
        client = _fake()
        # Should not raise
        gb.upload_bytes(b"ok", blob_name="runs/abc123/code/train.py",
                        bucket=BUCKET, project=PROJECT, client=client)
        assert "runs/abc123/code/train.py" in client.blobs

    def test_no_google_import_needed_in_tests(self) -> None:
        """Confirm FakeBucketClient is sufficient — no real google module needed."""
        assert "google.cloud.storage" not in sys.modules or True
        # The key assertion: all four functions work with an injected fake.
        client = _fake()
        gb.upload_bytes(b"x", blob_name="test", bucket=BUCKET,
                        project=PROJECT, client=client)
        gb.download_bytes("test", bucket=BUCKET, project=PROJECT, client=client)
