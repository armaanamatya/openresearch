"""ETag-CAS single-writer lease for durable controllers on Azure Blob.

This is the Azure implementation of the lease shape exposed by
``blob_lease.BlobLease``. Azure ETags replace GCS generations as the mutable
CAS version; the integer ``fence_epoch`` remains stable across heartbeats and
only advances on a different-owner takeover.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from typing import Any

from backend.services.runtime import azure_blob
from backend.services.runtime.blob_lease import LEASE_TTL_S, LeaseToken

logger = logging.getLogger(__name__)

__all__ = ["AzureBlobLease"]


def _lease_blob_name(run_id: str) -> str:
    return f"runs/{run_id}/rlm_state/owner.lease"


def _encode_lease(
    *,
    run_id: str,
    owner_id: str,
    acquired_epoch: float,
    renewed_epoch: float,
    fence_epoch: int,
) -> bytes:
    return json.dumps(
        {
            "run_id": run_id,
            "owner_id": owner_id,
            "acquired_epoch": acquired_epoch,
            "renewed_epoch": renewed_epoch,
            "fence_epoch": fence_epoch,
        },
        sort_keys=True,
    ).encode("utf-8")


class AzureBlobLease:
    """Single-writer lease over one Azure Blob per run."""

    def __init__(
        self,
        *,
        account_name: str,
        container_name: str,
        client: Any | None = None,
    ) -> None:
        self._account_name = account_name
        self._container_name = container_name
        self._client = client

    def _read(self, run_id: str) -> tuple[bytes, str] | None:
        return azure_blob.read_bytes_with_etag(
            _lease_blob_name(run_id),
            account_name=self._account_name,
            container_name=self._container_name,
            client=self._client,
        )

    def _write(
        self,
        *,
        run_id: str,
        owner_id: str,
        acquired_epoch: float,
        renewed_epoch: float,
        fence_epoch: int,
        if_etag_match: str | None,
    ) -> str | None:
        try:
            return azure_blob.upload_bytes_cas(
                _encode_lease(
                    run_id=run_id,
                    owner_id=owner_id,
                    acquired_epoch=acquired_epoch,
                    renewed_epoch=renewed_epoch,
                    fence_epoch=fence_epoch,
                ),
                blob_name=_lease_blob_name(run_id),
                account_name=self._account_name,
                container_name=self._container_name,
                if_etag_match=if_etag_match,
                client=self._client,
            )
        except azure_blob.PreconditionFailedError:
            return None

    def acquire(
        self, run_id: str, owner_id: str, now_epoch: float
    ) -> LeaseToken | None:
        existing = self._read(run_id)
        if existing is None:
            new_etag = self._write(
                run_id=run_id,
                owner_id=owner_id,
                acquired_epoch=now_epoch,
                renewed_epoch=now_epoch,
                fence_epoch=1,
                if_etag_match=None,
            )
            if new_etag is None:
                return None
            return LeaseToken(
                run_id=run_id,
                generation=new_etag,
                owner_id=owner_id,
                acquired_epoch=now_epoch,
                fence_epoch=1,
            )

        data, current_etag = existing
        record = json.loads(data.decode("utf-8"))
        current_owner = record.get("owner_id")
        renewed_epoch = float(
            record.get("renewed_epoch", record.get("acquired_epoch", 0.0))
        )
        expired = (now_epoch - renewed_epoch) >= LEASE_TTL_S
        if current_owner != owner_id and not expired:
            return None

        previous_fence = int(record.get("fence_epoch", 1))
        fence_epoch = (
            previous_fence + 1 if current_owner != owner_id else previous_fence
        )
        new_etag = self._write(
            run_id=run_id,
            owner_id=owner_id,
            acquired_epoch=now_epoch,
            renewed_epoch=now_epoch,
            fence_epoch=fence_epoch,
            if_etag_match=current_etag,
        )
        if new_etag is None:
            return None
        return LeaseToken(
            run_id=run_id,
            generation=new_etag,
            owner_id=owner_id,
            acquired_epoch=now_epoch,
            fence_epoch=fence_epoch,
        )

    def renew(self, token: LeaseToken, now_epoch: float) -> LeaseToken | None:
        new_etag = self._write(
            run_id=token.run_id,
            owner_id=token.owner_id,
            acquired_epoch=token.acquired_epoch,
            renewed_epoch=now_epoch,
            fence_epoch=token.fence_epoch,
            if_etag_match=str(token.generation),
        )
        if new_etag is None:
            return None
        return dataclasses.replace(token, generation=new_etag)

    def is_current(self, token: LeaseToken) -> bool:
        existing = self._read(token.run_id)
        return existing is not None and existing[1] == token.generation

    def reap_stale_fence_epochs(
        self,
        run_id: str,
        token: LeaseToken,
        *,
        list_jobs: Callable[[str], list[tuple[str, int]]],
        delete_job: Callable[[str], None],
    ) -> int:
        """Best-effort deletion of Jobs from superseded fence epochs."""
        try:
            jobs = list_jobs(run_id)
        except Exception as exc:
            logger.warning(
                "azure lease reaper: list_jobs(%s) failed, skipping: %s",
                run_id,
                exc,
            )
            return 0

        deleted = 0
        for job_name, fence_epoch in jobs:
            if fence_epoch >= token.fence_epoch:
                continue
            try:
                delete_job(job_name)
            except Exception as exc:
                logger.warning(
                    "azure lease reaper: delete %s failed, skipping: %s",
                    job_name,
                    exc,
                )
                continue
            deleted += 1
        return deleted
