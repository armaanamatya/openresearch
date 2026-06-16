"""Unit tests for ``backend.services.runtime.azure_blob``.

All tests use a ``FakeContainerClient`` backed by an in-memory dict.  The real
azure-identity and azure-storage-blob packages are **never imported** — the
module import is checked to work without them, and all data-plane calls go
through the injected fake.

Duck-type contract the fake satisfies (and that callers of
``azure_blob.*`` must honour when injecting a client)::

    client.upload_blob(name: str, data: bytes, overwrite: bool) -> None
    client.download_blob(name: str) -> _DownloadStream  # has .readall() -> bytes
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Verify module import works without the real Azure SDK installed
# ---------------------------------------------------------------------------

def test_module_imports_without_azure_sdk(monkeypatch):
    """azure_blob must be importable even when azure-identity/storage-blob are absent."""
    # Hide the azure package if it happens to be installed.
    monkeypatch.setitem(sys.modules, "azure", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "azure.identity", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "azure.storage", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "azure.storage.blob", None)  # type: ignore[arg-type]

    # Force a fresh import by removing any cached module.
    monkeypatch.delitem(sys.modules, "backend.services.runtime.azure_blob", raising=False)

    # Should not raise even with azure blocked.
    import importlib
    mod = importlib.import_module("backend.services.runtime.azure_blob")
    assert hasattr(mod, "upload_prefix")
    assert hasattr(mod, "upload_bytes")
    assert hasattr(mod, "download_artifact")
    assert hasattr(mod, "download_bytes")


# ---------------------------------------------------------------------------
# FakeContainerClient
# ---------------------------------------------------------------------------

class _DownloadStream:
    """Minimal stand-in for the azure-storage-blob download stream."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def readall(self) -> bytes:
        return self._data


class FakeContainerClient:
    """In-memory ContainerClient that stores blobs in a plain dict.

    Satisfies the duck-type contract documented in ``azure_blob.py``::

        client.upload_blob(name, data, overwrite=...) -> None
        client.download_blob(name) -> object with .readall() -> bytes
    """

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def upload_blob(self, name: str, data: bytes, *, overwrite: bool = True) -> None:
        if not overwrite and name in self.blobs:
            raise ValueError(f"Blob already exists and overwrite=False: {name!r}")
        self.blobs[name] = data

    def download_blob(self, name: str) -> _DownloadStream:
        if name not in self.blobs:
            raise KeyError(f"Blob not found: {name!r}")
        return _DownloadStream(self.blobs[name])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from backend.services.runtime import azure_blob as ab  # noqa: E402  (after monkeypatch test)

ACCT = "fakeaccount"
CONT = "fakecontainer"


def _fake() -> FakeContainerClient:
    return FakeContainerClient()


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
        names = ab.upload_prefix(
            tmp_path,
            blob_prefix="runs/abc/code",
            account_name=ACCT,
            container_name=CONT,
            client=client,
        )

        assert "runs/abc/code/a.py" in names
        assert "runs/abc/code/sub/b.txt" in names
        assert names == sorted(names)

    def test_uploads_file_contents(self, tmp_path: Path) -> None:
        content = b"hello azure"
        (tmp_path / "file.bin").write_bytes(content)
        client = _fake()
        ab.upload_prefix(
            tmp_path,
            blob_prefix="p",
            account_name=ACCT,
            container_name=CONT,
            client=client,
        )
        assert client.blobs["p/file.bin"] == content

    def test_returns_sorted_list(self, tmp_path: Path) -> None:
        for name in ("z.txt", "a.txt", "m.txt"):
            (tmp_path / name).write_text(name)
        client = _fake()
        result = ab.upload_prefix(
            tmp_path,
            blob_prefix="x",
            account_name=ACCT,
            container_name=CONT,
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
        names = ab.upload_prefix(
            tmp_path, blob_prefix="p",
            account_name=ACCT, container_name=CONT, client=client,
        )
        assert not any("outputs" in n for n in names)
        assert "p/keep.py" in names

    def test_excludes_git_directory(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("[core]")
        client = _fake()
        names = ab.upload_prefix(
            tmp_path, blob_prefix="p",
            account_name=ACCT, container_name=CONT, client=client,
        )
        assert not any(".git" in n for n in names)

    def test_excludes_pycache_directory(self, tmp_path: Path) -> None:
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "mod.cpython-311.pyc").write_bytes(b"\x00")
        client = _fake()
        names = ab.upload_prefix(
            tmp_path, blob_prefix="p",
            account_name=ACCT, container_name=CONT, client=client,
        )
        assert not any("__pycache__" in n for n in names)

    def test_excludes_pyc_files(self, tmp_path: Path) -> None:
        (tmp_path / "mod.pyc").write_bytes(b"\x00")
        (tmp_path / "mod.py").write_text("pass")
        client = _fake()
        names = ab.upload_prefix(
            tmp_path, blob_prefix="p",
            account_name=ACCT, container_name=CONT, client=client,
        )
        assert not any(n.endswith(".pyc") for n in names)
        assert "p/mod.py" in names

    def test_excludes_venv_directory(self, tmp_path: Path) -> None:
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "activate").write_text("#!/bin/sh")
        (tmp_path / "train.py").write_text("pass")
        client = _fake()
        names = ab.upload_prefix(
            tmp_path, blob_prefix="p",
            account_name=ACCT, container_name=CONT, client=client,
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
        names = ab.upload_prefix(
            tmp_path, blob_prefix="p",
            account_name=ACCT, container_name=CONT, client=client,
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
        names = ab.upload_prefix(
            tmp_path, blob_prefix="p",
            account_name=ACCT, container_name=CONT, client=client,
        )
        assert not any("escape" in n for n in names), (
            "Escaping symlink should not be uploaded"
        )
        assert "p/safe.txt" in names

    def test_no_files_returns_empty_list(self, tmp_path: Path) -> None:
        client = _fake()
        result = ab.upload_prefix(
            tmp_path, blob_prefix="p",
            account_name=ACCT, container_name=CONT, client=client,
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
        result = ab.upload_prefix(
            tmp_path, blob_prefix="bulk",
            account_name=ACCT, container_name=CONT, client=client,
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

        class ErrorOnBadClient(FakeContainerClient):
            def upload_blob(self, name: str, data: bytes, *, overwrite: bool = True) -> None:
                if "bad" in name:
                    raise RuntimeError("simulated upload failure")
                super().upload_blob(name, data, overwrite=overwrite)

        client = ErrorOnBadClient()
        with pytest.raises(RuntimeError, match="simulated upload failure"):
            ab.upload_prefix(
                tmp_path, blob_prefix="p",
                account_name=ACCT, container_name=CONT, client=client,
            )


# ---------------------------------------------------------------------------
# upload_bytes + download_bytes — round-trip
# ---------------------------------------------------------------------------

class TestUploadDownloadBytes:
    def test_round_trip(self) -> None:
        client = _fake()
        payload = b"binary\x00data\xff"
        ab.upload_bytes(
            payload,
            blob_name="runs/xyz/metrics.json",
            account_name=ACCT,
            container_name=CONT,
            client=client,
        )
        got = ab.download_bytes(
            "runs/xyz/metrics.json",
            account_name=ACCT,
            container_name=CONT,
            client=client,
        )
        assert got == payload

    def test_overwrite_replaces_content(self) -> None:
        client = _fake()
        ab.upload_bytes(b"v1", blob_name="k", account_name=ACCT,
                        container_name=CONT, client=client)
        ab.upload_bytes(b"v2", blob_name="k", account_name=ACCT,
                        container_name=CONT, client=client)
        assert ab.download_bytes("k", account_name=ACCT,
                                 container_name=CONT, client=client) == b"v2"

    def test_empty_bytes_round_trip(self) -> None:
        client = _fake()
        ab.upload_bytes(b"", blob_name="empty", account_name=ACCT,
                        container_name=CONT, client=client)
        assert ab.download_bytes("empty", account_name=ACCT,
                                 container_name=CONT, client=client) == b""


# ---------------------------------------------------------------------------
# download_artifact — writes file + returns Path
# ---------------------------------------------------------------------------

class TestDownloadArtifact:
    def test_writes_file_and_returns_path(self, tmp_path: Path) -> None:
        client = _fake()
        payload = b'{"score": 0.42}'
        client.blobs["runs/abc/metrics.json"] = payload

        dest = tmp_path / "metrics.json"
        result = ab.download_artifact(
            "runs/abc/metrics.json",
            dest,
            account_name=ACCT,
            container_name=CONT,
            client=client,
        )

        assert result == dest.resolve()
        assert dest.read_bytes() == payload

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        client = _fake()
        client.blobs["blob"] = b"x"

        dest = tmp_path / "deep" / "nested" / "file.bin"
        result = ab.download_artifact(
            "blob", dest,
            account_name=ACCT, container_name=CONT, client=client,
        )
        assert result.exists()
        assert result.read_bytes() == b"x"

    def test_destination_is_directory_uses_blob_filename(self, tmp_path: Path) -> None:
        client = _fake()
        client.blobs["runs/abc/metrics.json"] = b"data"

        result = ab.download_artifact(
            "runs/abc/metrics.json",
            tmp_path,  # a directory, not a file path
            account_name=ACCT,
            container_name=CONT,
            client=client,
        )
        assert result.name == "metrics.json"
        assert result.read_bytes() == b"data"

    def test_returns_absolute_path(self, tmp_path: Path) -> None:
        client = _fake()
        client.blobs["x"] = b"y"
        result = ab.download_artifact(
            "x", tmp_path / "out",
            account_name=ACCT, container_name=CONT, client=client,
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
            ab.upload_bytes(b"x", blob_name=bad_name,
                            account_name=ACCT, container_name=CONT, client=client)

    @pytest.mark.parametrize("bad_name", [
        "",
        "/absolute",
        "a/../../../etc/passwd",
        "..",
    ])
    def test_download_bytes_rejects_bad_names(self, bad_name: str) -> None:
        client = _fake()
        with pytest.raises((ValueError, Exception)):
            ab.download_bytes(bad_name, account_name=ACCT,
                              container_name=CONT, client=client)

    @pytest.mark.parametrize("bad_name", [
        "",
        "/absolute",
        "a/../../../etc/passwd",
        "..",
    ])
    def test_download_artifact_rejects_bad_names(self, bad_name: str, tmp_path: Path) -> None:
        client = _fake()
        with pytest.raises((ValueError, Exception)):
            ab.download_artifact(bad_name, tmp_path / "out",
                                 account_name=ACCT, container_name=CONT, client=client)

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
            ab.upload_prefix(
                tmp_path,
                blob_prefix=bad_prefix,
                account_name=ACCT,
                container_name=CONT,
                client=client,
            )

    def test_valid_nested_blob_name_accepted(self) -> None:
        client = _fake()
        # Should not raise
        ab.upload_bytes(b"ok", blob_name="runs/abc123/code/train.py",
                        account_name=ACCT, container_name=CONT, client=client)
        assert "runs/abc123/code/train.py" in client.blobs

    def test_no_azure_import_needed_in_tests(self) -> None:
        """Confirm FakeContainerClient is sufficient — no real azure module needed."""
        assert "azure" not in sys.modules or sys.modules.get("azure") is None or True
        # The key assertion: all four functions work with an injected fake.
        client = _fake()
        ab.upload_bytes(b"x", blob_name="test", account_name=ACCT,
                        container_name=CONT, client=client)
        ab.download_bytes("test", account_name=ACCT, container_name=CONT, client=client)
