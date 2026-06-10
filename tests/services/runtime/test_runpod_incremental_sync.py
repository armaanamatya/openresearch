"""Unit tests for RunpodBackend._sync_artifacts_to_host incremental SFTP sync.

These are purely unit tests — no real SSH or network.  The asyncssh SFTP
client is fully mocked via a small FakeSFTP helper class.
"""

from __future__ import annotations

import stat
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.services.runtime.interface import (
    RuntimeCauseKind,
    Sandbox,
    SandboxConfig,
    SandboxRuntimeError,
)
from backend.services.runtime.runpod_backend import RunpodBackend, _RunpodConnection


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_sandbox(*, tmp_path: Path) -> Sandbox:
    config = SandboxConfig(
        project_id="proj",
        run_id="run",
        project_root=tmp_path,
    )
    return Sandbox(
        sandbox_id="test-pod",
        name="test-pod",
        image="test-image",
        config=config,
        created_at=datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc),
    )


def _make_connection(remote_artifacts_dir: str = "/artifacts") -> _RunpodConnection:
    return _RunpodConnection(
        pod_id="test-pod",
        public_ip="1.2.3.4",
        ssh_port=22,
        remote_base="/workspace",
        remote_workdir="/workspace/work",
        remote_artifacts_dir=remote_artifacts_dir,
    )


class _Attrs:
    """Minimal stand-in for asyncssh SFTPAttrs."""

    def __init__(
        self,
        *,
        permissions: int | None,
        size: int | None = None,
        mtime: float | None = None,
        atime: float | None = None,
    ) -> None:
        self.permissions = permissions
        self.size = size
        self.mtime = mtime
        self.atime = atime


def _reg_attrs(*, size: int = 100, mtime: float = 1_000_000.0) -> _Attrs:
    return _Attrs(
        permissions=stat.S_IFREG | 0o644,
        size=size,
        mtime=mtime,
        atime=mtime,
    )


def _dir_attrs() -> _Attrs:
    return _Attrs(permissions=stat.S_IFDIR | 0o755)


def _lnk_attrs() -> _Attrs:
    return _Attrs(permissions=stat.S_IFLNK | 0o777, size=10, mtime=1_000_000.0)


class _FakeDirEntry:
    def __init__(self, filename: str) -> None:
        self.filename = filename


class _FakeSFTPFile:
    """Async-context-manager SFTP file handle."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    async def read(self, n: int) -> bytes:
        chunk = self._data[self._pos : self._pos + n]
        self._pos += n
        return chunk

    async def __aenter__(self) -> "_FakeSFTPFile":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class FakeSFTP:
    """Configurable fake asyncssh SFTP client.

    Build a ``tree`` dict mapping remote_path -> _Attrs (or None = missing).
    Build a ``file_data`` dict mapping remote_path -> bytes for file content.
    """

    def __init__(
        self,
        *,
        tree: dict[str, _Attrs],
        file_data: dict[str, bytes] | None = None,
        stat_raises: type[Exception] | None = None,
    ) -> None:
        self._tree = tree
        self._file_data = file_data or {}
        self._stat_raises = stat_raises
        self.opened_paths: list[str] = []

    async def stat(self, path: str) -> _Attrs:
        if self._stat_raises is not None:
            raise self._stat_raises(path)
        if path not in self._tree:
            raise FileNotFoundError(path)
        return self._tree[path]

    async def lstat(self, path: str) -> _Attrs:
        # lstat does not follow symlinks — same lookup in our fake tree.
        if path not in self._tree:
            raise FileNotFoundError(path)
        return self._tree[path]

    async def readdir(self, path: str) -> list[_FakeDirEntry]:
        prefix = path.rstrip("/") + "/"
        seen: set[str] = set()
        entries: list[_FakeDirEntry] = []
        for remote_path in self._tree:
            if remote_path == path:
                continue
            if not remote_path.startswith(prefix):
                continue
            rest = remote_path[len(prefix):]
            top = rest.split("/")[0]
            if top and top not in seen:
                seen.add(top)
                entries.append(_FakeDirEntry(top))
        return entries

    def open(self, path: str, mode: str) -> _FakeSFTPFile:
        self.opened_paths.append(path)
        data = self._file_data.get(path, b"")
        return _FakeSFTPFile(data)


def _build_backend(sandbox: Sandbox, fake_sftp: FakeSFTP) -> RunpodBackend:
    """Wire up a RunpodBackend with a mocked SSH connection returning *fake_sftp*."""
    backend = RunpodBackend(api_key="dummy")
    backend._connections[sandbox.sandbox_id] = _make_connection()
    backend._owned_pod_ids = {sandbox.sandbox_id}

    @asynccontextmanager
    async def _sftp_ctx():
        yield fake_sftp

    mock_conn = MagicMock()
    mock_conn.start_sftp_client = _sftp_ctx

    async def _fake_ssh(pod_id: str) -> Any:
        return mock_conn

    backend._ssh = _fake_ssh  # type: ignore[assignment]
    return backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_creates_new_files_locally(tmp_path: Path) -> None:
    """Remote has a file the local dir doesn't; it should be copied."""
    file_data = b"hello world"
    fake_sftp = FakeSFTP(
        tree={
            "/artifacts": _dir_attrs(),
            "/artifacts/output.txt": _reg_attrs(size=len(file_data), mtime=1_000_000.0),
        },
        file_data={"/artifacts/output.txt": file_data},
    )
    sandbox = _make_sandbox(tmp_path=tmp_path)
    backend = _build_backend(sandbox, fake_sftp)

    await backend._sync_artifacts_to_host(sandbox)

    local_file = sandbox.config.resolved_artifact_root() / "output.txt"
    assert local_file.exists(), "File should have been copied locally"
    assert local_file.read_bytes() == file_data


@pytest.mark.asyncio
async def test_sync_skips_unchanged_files(tmp_path: Path) -> None:
    """Remote and local have the same file with matching (size, mtime); SFTP open must NOT be called."""
    file_data = b"unchanged content"
    remote_mtime = 1_000_000.0

    # Pre-create the local file with matching size + mtime.
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True)
    local_file = artifact_root / "data.bin"
    local_file.write_bytes(file_data)
    import os
    os.utime(local_file, (remote_mtime, remote_mtime))

    fake_sftp = FakeSFTP(
        tree={
            "/artifacts": _dir_attrs(),
            "/artifacts/data.bin": _reg_attrs(size=len(file_data), mtime=remote_mtime),
        },
        file_data={"/artifacts/data.bin": file_data},
    )
    sandbox = _make_sandbox(tmp_path=tmp_path)
    backend = _build_backend(sandbox, fake_sftp)

    await backend._sync_artifacts_to_host(sandbox)

    # The file should NOT have been opened via SFTP.
    assert "/artifacts/data.bin" not in fake_sftp.opened_paths, (
        "SFTP open() was called for an unchanged file — skipping logic broken"
    )


@pytest.mark.asyncio
async def test_sync_transfers_changed_files(tmp_path: Path) -> None:
    """Remote file has a different size from local; it should be re-transferred."""
    old_data = b"old data"
    new_data = b"new data that is bigger"

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True)
    local_file = artifact_root / "result.txt"
    local_file.write_bytes(old_data)

    remote_mtime = time.time() + 100  # remote is newer
    fake_sftp = FakeSFTP(
        tree={
            "/artifacts": _dir_attrs(),
            "/artifacts/result.txt": _reg_attrs(size=len(new_data), mtime=remote_mtime),
        },
        file_data={"/artifacts/result.txt": new_data},
    )
    sandbox = _make_sandbox(tmp_path=tmp_path)
    backend = _build_backend(sandbox, fake_sftp)

    await backend._sync_artifacts_to_host(sandbox)

    assert local_file.read_bytes() == new_data, "Changed file should have been re-transferred"


@pytest.mark.asyncio
async def test_sync_refuses_symlinks(tmp_path: Path) -> None:
    """Remote symlinks are silently skipped — not transferred, no error raised."""
    fake_sftp = FakeSFTP(
        tree={
            "/artifacts": _dir_attrs(),
            "/artifacts/link.txt": _lnk_attrs(),
        },
        file_data={"/artifacts/link.txt": b"target content"},
    )
    sandbox = _make_sandbox(tmp_path=tmp_path)
    backend = _build_backend(sandbox, fake_sftp)

    # Should not raise.
    await backend._sync_artifacts_to_host(sandbox)

    local_file = sandbox.config.resolved_artifact_root() / "link.txt"
    assert not local_file.exists(), "Symlink should have been skipped — not created locally"
    assert "/artifacts/link.txt" not in fake_sftp.opened_paths


@pytest.mark.asyncio
async def test_sync_skips_entries_with_no_permissions_attr(tmp_path: Path) -> None:
    """Regression guard: asyncssh.SFTPAttrs.permissions is Optional[int].

    A misbehaving SFTP server can return None; stat.S_ISLNK(None) would raise
    TypeError and crash the sync. The implementation must skip such entries
    defensively rather than propagating the error.
    """
    nullperm_attrs = _Attrs(
        size=42,
        mtime=1_000_000.0,
        atime=1_000_000.0,
        permissions=None,  # The case under test.
    )
    fake_sftp = FakeSFTP(
        tree={
            "/artifacts": _dir_attrs(),
            "/artifacts/mystery.bin": nullperm_attrs,
        },
        file_data={"/artifacts/mystery.bin": b"data"},
    )
    sandbox = _make_sandbox(tmp_path=tmp_path)
    backend = _build_backend(sandbox, fake_sftp)

    # Must not raise TypeError or anything else.
    await backend._sync_artifacts_to_host(sandbox)

    # The entry must be skipped — neither created locally nor read remotely.
    assert not (sandbox.config.resolved_artifact_root() / "mystery.bin").exists()
    assert "/artifacts/mystery.bin" not in fake_sftp.opened_paths


@pytest.mark.asyncio
async def test_sync_refuses_path_escape(tmp_path: Path) -> None:
    """Remote walk yields a path that resolves outside local_root; must raise SandboxRuntimeError."""
    # Inject a path that after join + resolve would escape: use a known absolute
    # path on the host.  We can't actually make `..` work through our fake tree
    # easily, so we use the _relative_posix helper's output directly.
    # The simplest approach: subclass FakeSFTP to return a crafted entry whose
    # resolved local path escapes tmp_path.

    # Craft a fake entry that, when joined with local_root, resolves outside.
    # We do this by putting a long relative path with ".." components in the
    # tree.  Our readdir walks /artifacts and returns "evil", then the lstat
    # for /artifacts/evil is a regular file. The join makes local_root / "evil"
    # which is safe — we need the *resolved* path to escape.
    # The cleanest approach: override _relative_posix to return a crafted value.
    escape_target = str(tmp_path.parent / "escaped_file.txt")

    class EscapingFakeSFTP(FakeSFTP):
        async def readdir(self, path: str) -> list[_FakeDirEntry]:
            if path == "/artifacts":
                return [_FakeDirEntry("evil")]
            return []

        async def lstat(self, path: str) -> _Attrs:
            if path == "/artifacts/evil":
                return _reg_attrs(size=5, mtime=1_000_000.0)
            return await super().lstat(path)

    fake_sftp = EscapingFakeSFTP(
        tree={"/artifacts": _dir_attrs()},
    )

    sandbox = _make_sandbox(tmp_path=tmp_path)
    backend = _build_backend(sandbox, fake_sftp)

    # Patch _relative_posix to return a path that escapes the local root.
    # `../../<parent>/escaped_file.txt` from the artifact root resolves outside.
    artifact_root = sandbox.config.resolved_artifact_root()
    # Compute how many ".." we need to get from artifact_root to its grandparent.
    escape_relative = "../../../" + escape_target.lstrip("/")

    with patch(
        "backend.services.runtime.runpod_backend._relative_posix",
        return_value=escape_relative,
    ):
        with pytest.raises(SandboxRuntimeError) as exc_info:
            await backend._sync_artifacts_to_host(sandbox)

    assert exc_info.value.cause_kind == RuntimeCauseKind.copy_failed
    assert "unsafe" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_sync_handles_missing_remote_artifacts_dir(tmp_path: Path) -> None:
    """sftp.stat(remote_root) raises FileNotFoundError; sync is a silent no-op."""
    fake_sftp = FakeSFTP(
        tree={},
        stat_raises=FileNotFoundError,
    )
    sandbox = _make_sandbox(tmp_path=tmp_path)
    backend = _build_backend(sandbox, fake_sftp)

    # Must not raise, must not create any local files.
    await backend._sync_artifacts_to_host(sandbox)

    artifact_root = sandbox.config.resolved_artifact_root()
    if artifact_root.exists():
        children = list(artifact_root.iterdir())
        assert children == [], "No files should be created when remote dir is missing"


@pytest.mark.asyncio
async def test_sync_preserves_remote_mtime_on_local(tmp_path: Path) -> None:
    """After copying a file, local mtime is set to match remote mtime."""

    remote_mtime = 1_234_567.0
    file_data = b"fresh data"
    fake_sftp = FakeSFTP(
        tree={
            "/artifacts": _dir_attrs(),
            "/artifacts/checkpoint.pt": _reg_attrs(size=len(file_data), mtime=remote_mtime),
        },
        file_data={"/artifacts/checkpoint.pt": file_data},
    )
    sandbox = _make_sandbox(tmp_path=tmp_path)
    backend = _build_backend(sandbox, fake_sftp)

    await backend._sync_artifacts_to_host(sandbox)

    local_file = sandbox.config.resolved_artifact_root() / "checkpoint.pt"
    assert local_file.exists()
    local_mtime = local_file.stat().st_mtime
    assert abs(local_mtime - remote_mtime) < 2.0, (
        f"Local mtime {local_mtime} should match remote mtime {remote_mtime}"
    )


@pytest.mark.asyncio
async def test_sync_does_not_delete_local_only_files(tmp_path: Path) -> None:
    """Files that exist locally but not on the remote are left untouched."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True)
    local_only = artifact_root / "local_only.txt"
    local_only.write_bytes(b"keep me")

    # Remote has no files at all (empty dir).
    fake_sftp = FakeSFTP(
        tree={
            "/artifacts": _dir_attrs(),
        },
    )
    sandbox = _make_sandbox(tmp_path=tmp_path)
    backend = _build_backend(sandbox, fake_sftp)

    await backend._sync_artifacts_to_host(sandbox)

    assert local_only.exists(), "Local-only file should not be deleted"
    assert local_only.read_bytes() == b"keep me"
