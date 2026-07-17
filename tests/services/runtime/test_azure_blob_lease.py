"""Hermetic Azure ETag-CAS and durable lease tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.services.runtime import azure_blob
from backend.services.runtime.azure_blob_lease import AzureBlobLease
from backend.services.runtime.blob_lease import LEASE_TTL_S


class _PreconditionFailed(Exception):
    pass


class _NotFound(Exception):
    pass


@dataclass
class _Stream:
    data: bytes
    etag: str

    @property
    def properties(self) -> dict[str, str]:
        return {"etag": self.etag}

    def readall(self) -> bytes:
        return self.data


class _CasContainer:
    def __init__(self) -> None:
        self.blobs: dict[str, tuple[bytes, str]] = {}
        self.version = 0

    def upload_blob(
        self,
        name: str,
        data: bytes,
        *,
        overwrite: bool,
        etag: str | None = None,
        match_condition: object | None = None,
    ) -> dict[str, str]:
        current = self.blobs.get(name)
        if not overwrite and current is not None:
            raise _PreconditionFailed(name)
        if overwrite and (current is None or current[1] != etag):
            raise _PreconditionFailed(name)
        self.version += 1
        new_etag = f'"etag-{self.version}"'
        self.blobs[name] = (data, new_etag)
        return {"etag": new_etag}

    def download_blob(self, name: str) -> _Stream:
        try:
            data, etag = self.blobs[name]
        except KeyError as exc:
            raise _NotFound(name) from exc
        return _Stream(data, etag)


class _BlobClientContainer(_CasContainer):
    """Models the real SDK split: container returns a per-blob client."""

    class _BlobClient:
        def __init__(self, parent: "_BlobClientContainer", name: str) -> None:
            self.parent = parent
            self.name = name

        def upload_blob(self, data: bytes, **kwargs) -> dict[str, str]:
            return _CasContainer.upload_blob(
                self.parent, self.name, data, **kwargs
            )

    def get_blob_client(self, name: str) -> "_BlobClientContainer._BlobClient":
        return self._BlobClient(self, name)

    def upload_blob(self, *args, **kwargs):
        raise AssertionError("CAS must use BlobClient.upload_blob with the real SDK")


@pytest.fixture(autouse=True)
def _fake_azure_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        azure_blob, "_precondition_failed_exc_types", lambda: (_PreconditionFailed,)
    )
    monkeypatch.setattr(azure_blob, "_not_found_exc_types", lambda: (_NotFound,))


def _lease(client: _CasContainer) -> AzureBlobLease:
    return AzureBlobLease(
        account_name="account",
        container_name="artifacts",
        client=client,
    )


def test_azure_blob_cas_create_read_replace_and_stale_reject() -> None:
    client = _CasContainer()
    etag1 = azure_blob.upload_bytes_cas(
        b"one",
        blob_name="leases/x",
        account_name="account",
        container_name="artifacts",
        if_etag_match=None,
        client=client,
    )
    assert azure_blob.read_bytes_with_etag(
        "leases/x",
        account_name="account",
        container_name="artifacts",
        client=client,
    ) == (b"one", etag1)

    etag2 = azure_blob.upload_bytes_cas(
        b"two",
        blob_name="leases/x",
        account_name="account",
        container_name="artifacts",
        if_etag_match=etag1,
        client=client,
    )
    assert etag2 != etag1
    with pytest.raises(azure_blob.PreconditionFailedError):
        azure_blob.upload_bytes_cas(
            b"stale",
            blob_name="leases/x",
            account_name="account",
            container_name="artifacts",
            if_etag_match=etag1,
            client=client,
        )


def test_azure_blob_cas_uses_exact_blob_client_write_etag() -> None:
    client = _BlobClientContainer()
    etag = azure_blob.upload_bytes_cas(
        b"one",
        blob_name="leases/sdk-shape",
        account_name="account",
        container_name="artifacts",
        if_etag_match=None,
        client=client,
    )
    assert etag == '"etag-1"'
    assert client.blobs["leases/sdk-shape"] == (b"one", etag)


def test_azure_lease_renew_preserves_fence_and_advances_etag() -> None:
    lease = _lease(_CasContainer())
    token = lease.acquire("prj_x", "owner-a", 100.0)
    assert token is not None
    renewed = lease.renew(token, 120.0)
    assert renewed is not None
    assert renewed.generation != token.generation
    assert renewed.fence_epoch == token.fence_epoch == 1
    assert lease.is_current(token) is False
    assert lease.is_current(renewed) is True


def test_azure_lease_refuses_live_rival_and_bumps_fence_on_expired_takeover() -> None:
    lease = _lease(_CasContainer())
    first = lease.acquire("prj_x", "owner-a", 100.0)
    assert first is not None
    assert lease.acquire("prj_x", "owner-b", 101.0) is None

    takeover = lease.acquire("prj_x", "owner-b", 100.0 + LEASE_TTL_S)
    assert takeover is not None
    assert takeover.fence_epoch == first.fence_epoch + 1
    assert lease.is_current(first) is False


def test_azure_lease_same_owner_restart_keeps_fence() -> None:
    lease = _lease(_CasContainer())
    first = lease.acquire("prj_x", "prj_x", 100.0)
    assert first is not None
    restarted = lease.acquire("prj_x", "prj_x", 101.0)
    assert restarted is not None
    assert restarted.fence_epoch == first.fence_epoch
