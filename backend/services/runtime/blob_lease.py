"""Cloud-agnostic single-writer generation-CAS lease for durable run ownership.

WS3 (durable cloud-native orchestration) design §4.1:
``docs/history/specs/2026-07-10-durable-cloud-native-orchestration-ws3-design.md``.

Exactly one driver may own a given ``run_id`` at a time. Ownership is a
compare-and-swap lease on a single small object,
``runs/<run_id>/rlm_state/owner.lease`` — this module contains no cloud-SDK
calls of its own; all reads/writes go through
:mod:`backend.services.runtime.gcs_blob`'s generation-CAS primitives
(``upload_bytes(if_generation_match=...)`` / ``read_bytes_with_generation``).

The GCS object's generation is the heartbeat CAS token: every successful
acquire/renew advances it, and an older token is superseded. The separate
integer ``fence_epoch`` is stable across same-owner heartbeats and advances
only on a different-owner takeover; Kubernetes Job names/labels use that
stable fence so a heartbeat never makes a controller reap its own work.

This module is the GCS implementation. Azure implements the same public shape
with Blob ETags in ``azure_blob_lease.py``.

Clock discipline
-----------------
Every method that reasons about time takes ``now_epoch: float`` explicitly
from the caller. Nothing in this module calls ``time.time()`` — that keeps
acquire/expiry/renew races fully deterministic under test.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any, Callable

from backend.services.runtime import gcs_blob

__all__ = [
    "LeaseToken",
    "BlobLease",
    "LEASE_TTL_S",
]

logger = logging.getLogger(__name__)

# The controller is expected to renew on a fixed heartbeat cadence once WS3's
# controller wiring lands (design §4.3); this module doesn't run that loop
# itself, it just defines the TTL a heartbeat-driven caller should honor.
# TTL = heartbeat x3 so a single missed/delayed renewal (GC pause, transient
# GCS latency) doesn't trigger an unnecessary takeover, while two consecutive
# missed heartbeats reliably do free the lease for a successor.
_HEARTBEAT_INTERVAL_S = 60
LEASE_TTL_S = _HEARTBEAT_INTERVAL_S * 3


@dataclasses.dataclass(frozen=True)
class LeaseToken:
    """An acquired/renewed lease's identity plus its fence token.

    ``generation`` is the GCS object generation (or Azure ETag) the lease
    currently holds. A write performed "under" this token is only
    safe while this generation is still the live one in the bucket; once a
    successor's acquire/renew advances it, this token is stale
    (``BlobLease.is_current`` returns ``False`` for it).
    """

    run_id: str
    generation: int | str
    owner_id: str
    acquired_epoch: float
    # A renew-invariant fence token, distinct from ``generation``: ``renew``
    # advances the CAS ``generation`` on every heartbeat, but ``fence_epoch``
    # stays constant for as long as the same owner holds the lease. Fenced Job
    # names embed ``fence_epoch`` (never ``generation``), so a controller does
    # NOT reap its own still-running Jobs after a heartbeat. It bumps by one
    # only on a real takeover (a DIFFERENT owner acquiring).
    fence_epoch: int = 1


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
    payload = {
        "run_id": run_id,
        "owner_id": owner_id,
        "acquired_epoch": acquired_epoch,
        "renewed_epoch": renewed_epoch,
        "fence_epoch": fence_epoch,
    }
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _decode_lease(data: bytes) -> dict[str, Any]:
    return json.loads(data.decode("utf-8"))


class BlobLease:
    """Single-writer CAS lease over a GCS object, one lease per ``run_id``.

    A single ``BlobLease`` instance is stateless across calls (it only
    carries the bucket connection parameters) and can service leases for
    any number of distinct ``run_id`` values.
    """

    def __init__(
        self,
        *,
        bucket: str,
        project: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._bucket = bucket
        self._project = project
        self._client = client

    # -- internal helpers ----------------------------------------------------

    def _read(self, run_id: str) -> tuple[bytes, int] | None:
        return gcs_blob.read_bytes_with_generation(
            blob_name=_lease_blob_name(run_id),
            bucket=self._bucket,
            project=self._project,
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
        if_generation_match: int,
    ) -> int | None:
        payload = _encode_lease(
            run_id=run_id,
            owner_id=owner_id,
            acquired_epoch=acquired_epoch,
            renewed_epoch=renewed_epoch,
            fence_epoch=fence_epoch,
        )
        try:
            return gcs_blob.upload_bytes(
                payload,
                blob_name=_lease_blob_name(run_id),
                bucket=self._bucket,
                project=self._project,
                client=self._client,
                if_generation_match=if_generation_match,
            )
        except gcs_blob.PreconditionFailedError:
            return None

    # -- public API ------------------------------------------------------

    def acquire(
        self, run_id: str, owner_id: str, now_epoch: float
    ) -> LeaseToken | None:
        """Try to become (or remain) the owner of ``run_id``.

        - No lease exists yet: create it (``if_generation_match=0``).
        - A lease exists and is still live under a *different* owner:
          refused (``None``) — someone else legitimately holds it.
        - A lease exists and is live under *this same* owner: treated as a
          reacquire (e.g. process restart with a stable ``owner_id``) and
          proceeds like a renewal.
        - A lease exists but its TTL has elapsed since the last
          acquire/renew: eligible for takeover regardless of the recorded
          owner.
        - Either takeover path races on the read-then-write via
          ``if_generation_match=<generation just read>``; if someone else
          wins that race, ``PreconditionFailedError`` degrades to ``None``.

        Returns the new :class:`LeaseToken` (whose ``.generation`` is the
        fence token to use for subsequent writes/renewals), or ``None`` if
        this call did not win ownership.
        """
        existing = self._read(run_id)

        if existing is None:
            new_gen = self._write(
                run_id=run_id,
                owner_id=owner_id,
                acquired_epoch=now_epoch,
                renewed_epoch=now_epoch,
                fence_epoch=1,
                if_generation_match=0,
            )
            if new_gen is None:
                return None
            return LeaseToken(
                run_id=run_id,
                generation=new_gen,
                owner_id=owner_id,
                acquired_epoch=now_epoch,
                fence_epoch=1,
            )

        data, current_gen = existing
        record = _decode_lease(data)
        current_owner = record.get("owner_id")
        renewed_epoch = record.get(
            "renewed_epoch", record.get("acquired_epoch", 0.0)
        )
        expired = (now_epoch - renewed_epoch) >= LEASE_TTL_S

        if current_owner != owner_id and not expired:
            return None  # live lease, held by someone else — refused.

        # A takeover by a DIFFERENT owner advances the fence so the successor
        # can reap the predecessor's now-orphaned Jobs. A same-owner reacquire
        # (process/Pod restart with a stable owner_id, expired or not) is not a
        # takeover — its Jobs are still its own, so the fence is preserved.
        prev_fence = int(record.get("fence_epoch", 1))
        fence_epoch = prev_fence + 1 if current_owner != owner_id else prev_fence

        new_gen = self._write(
            run_id=run_id,
            owner_id=owner_id,
            acquired_epoch=now_epoch,
            renewed_epoch=now_epoch,
            fence_epoch=fence_epoch,
            if_generation_match=current_gen,
        )
        if new_gen is None:
            return None  # raced with a rival acquirer/renewer.
        return LeaseToken(
            run_id=run_id,
            generation=new_gen,
            owner_id=owner_id,
            acquired_epoch=now_epoch,
            fence_epoch=fence_epoch,
        )

    def renew(self, token: LeaseToken, now_epoch: float) -> LeaseToken | None:
        """Heartbeat an already-held lease, advancing its fence generation.

        Blindly re-writes with ``if_generation_match=token.generation`` — no
        read-before-write, since the CAS itself is the correctness
        mechanism: if the live generation no longer equals ``token
        .generation``, someone else has since acquired/renewed, and this
        call must fail closed.

        Returns a new :class:`LeaseToken` carrying the advanced generation
        (``acquired_epoch`` is preserved from the original token — it is
        this owner's original acquisition time, not the renewal time), or
        ``None`` if superseded — the caller MUST stop writing on ``None``.
        """
        new_gen = self._write(
            run_id=token.run_id,
            owner_id=token.owner_id,
            acquired_epoch=token.acquired_epoch,
            renewed_epoch=now_epoch,
            fence_epoch=token.fence_epoch,
            if_generation_match=token.generation,
        )
        if new_gen is None:
            return None
        return dataclasses.replace(token, generation=new_gen)

    def is_current(self, token: LeaseToken) -> bool:
        """Cheap liveness check: is ``token`` still the current fence chain?

        Call this before every state/evidence write and before every GPU Job
        submit. A superseded owner (or one whose lease blob has vanished
        entirely, e.g. deleted by a bug) sees ``False`` and must not write.
        """
        existing = self._read(token.run_id)
        if existing is None:
            return False
        _, current_gen = existing
        return current_gen == token.generation

    def reap_older_generations(
        self,
        run_id: str,
        token: LeaseToken,
        *,
        list_jobs: Callable[[str], list[tuple[str, int]]],
        delete_job: Callable[[str], None],
    ) -> int:
        """Delete stale-generation K8s Jobs for ``run_id`` via injected I/O.

        Per design §4.3 ("Reaper"): on winning the lease, the new owner
        deletes every Job whose fenced name carries an older generation than
        ``token.generation``, closing the orphaned-GPU-Job cost leak. This
        module stays cloud-SDK-free — the real ``list_namespaced_job``/
        ``delete_namespaced_job`` calls belong to the (drill-time) caller,
        which supplies them here as plain callables:

        - ``list_jobs(run_id) -> [(job_name, generation), ...]`` — every
          fenced Job currently associated with ``run_id``, regardless of
          generation.
        - ``delete_job(job_name) -> None`` — best-effort delete of one Job
          by name (e.g. wrapping ``batch_api.delete_namespaced_job``).

        Every ``(job_name, gen)`` with ``gen < token.generation`` is
        deleted; Jobs at the current generation or newer are NEVER touched.

        Fail-soft, on purpose, at two levels:

        - A single ``delete_job`` failure is logged and skipped — the reap
          continues with the remaining stale Jobs (mirrors
          ``k8s_job_backend._delete_job_quietly``'s never-raise discipline:
          one bad delete must not abort the reap or block lease
          acquisition).
        - A ``list_jobs`` failure is logged and treated as "nothing (further)
          to reap" — it returns the count of Jobs already deleted (``0`` if
          the listing itself is what failed) rather than propagating.
          Reaping is best-effort cleanup, not a correctness precondition of
          holding the lease, so a transient listing error must never crash
          lease acquisition.

        Returns the number of Jobs actually deleted (successful
        ``delete_job`` calls only).
        """
        try:
            jobs = list_jobs(run_id)
        except Exception as exc:
            logger.warning(
                "reap_older_generations(%s): list_jobs failed, treating as "
                "nothing to reap: %s",
                run_id,
                exc,
            )
            return 0

        deleted = 0
        for job_name, gen in jobs:
            if gen >= token.generation:
                continue
            try:
                delete_job(job_name)
            except Exception as exc:
                logger.warning(
                    "reap_older_generations(%s): failed to delete stale Job "
                    "%s (generation %d < %d, ignored): %s",
                    run_id,
                    job_name,
                    gen,
                    token.generation,
                    exc,
                )
                continue
            deleted += 1
        return deleted

    def reap_stale_fence_epochs(
        self,
        run_id: str,
        token: LeaseToken,
        *,
        list_jobs: Callable[[str], list[tuple[str, int]]],
        delete_job: Callable[[str], None],
    ) -> int:
        """Delete Jobs whose fence epoch predates ``token.fence_epoch``.

        The fence-epoch analogue of :meth:`reap_older_generations`, and the
        method a durable controller MUST use: because ``fence_epoch`` is
        renew-invariant (unlike the CAS generation), a controller reaping by
        ``token.fence_epoch`` never deletes its own current-fence Jobs after a
        heartbeat renewal — only a predecessor's older-fence Jobs, on takeover.

        ``list_jobs(run_id) -> [(job_name, fence_epoch), ...]`` and
        ``delete_job(job_name) -> None`` are caller-injected (kept SDK-free per
        this module's purity contract). Every ``(job_name, fe)`` with
        ``fe < token.fence_epoch`` is deleted; current/newer fences are never
        touched. Fail-soft at both levels (a single ``delete_job`` raise is
        logged and skipped; a ``list_jobs`` raise is treated as nothing to
        reap) — reaping is best-effort cleanup, never a correctness
        precondition of holding the lease. Returns the count actually deleted.
        """
        try:
            jobs = list_jobs(run_id)
        except Exception as exc:
            logger.warning(
                "reap_stale_fence_epochs(%s): list_jobs failed, treating as "
                "nothing to reap: %s",
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
                    "reap_stale_fence_epochs(%s): failed to delete stale Job "
                    "%s (fence_epoch %d < %d, ignored): %s",
                    run_id,
                    job_name,
                    fence_epoch,
                    token.fence_epoch,
                    exc,
                )
                continue
            deleted += 1
        return deleted
