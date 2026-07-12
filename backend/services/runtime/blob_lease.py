"""Cloud-agnostic single-writer generation-CAS lease for durable run ownership.

WS3 (durable cloud-native orchestration) design §4.1:
``docs/superpowers/specs/2026-07-10-durable-cloud-native-orchestration-ws3-design.md``.

Exactly one driver may own a given ``run_id`` at a time. Ownership is a
compare-and-swap lease on a single small object,
``runs/<run_id>/rlm_state/owner.lease`` — this module contains no cloud-SDK
calls of its own; all reads/writes go through
:mod:`backend.services.runtime.gcs_blob`'s generation-CAS primitives
(``upload_bytes(if_generation_match=...)`` / ``read_bytes_with_generation``).

The GCS object's *generation* IS the fence token: every successful
acquire/renew stamps a new generation, and any driver holding an older
generation is, by construction, superseded — it must stop writing evidence
and stop keeping GPU Jobs alive (see design §4.2, fencing the work).

This is a **GCS-only implementation**. The public surface
(``acquire``/``renew``/``is_current``/``reap_older_generations``) is written
to be cloud-agnostic on purpose — an Azure Blob adapter would implement the
same shape using ETag/``If-Match`` instead of GCS generations — but per the
WS3 design's explicit non-goal, only the GCS backing is built here. Do not
add an Azure implementation to this module.

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

    ``generation`` is the GCS object generation the lease currently holds —
    this is the fence token. A write performed "under" this token is only
    safe while this generation is still the live one in the bucket; once a
    successor's acquire/renew advances it, this token is stale
    (``BlobLease.is_current`` returns ``False`` for it).
    """

    run_id: str
    generation: int
    owner_id: str
    acquired_epoch: float


def _lease_blob_name(run_id: str) -> str:
    return f"runs/{run_id}/rlm_state/owner.lease"


def _encode_lease(
    *, run_id: str, owner_id: str, acquired_epoch: float, renewed_epoch: float
) -> bytes:
    payload = {
        "run_id": run_id,
        "owner_id": owner_id,
        "acquired_epoch": acquired_epoch,
        "renewed_epoch": renewed_epoch,
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
        if_generation_match: int,
    ) -> int | None:
        payload = _encode_lease(
            run_id=run_id,
            owner_id=owner_id,
            acquired_epoch=acquired_epoch,
            renewed_epoch=renewed_epoch,
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
                if_generation_match=0,
            )
            if new_gen is None:
                return None
            return LeaseToken(
                run_id=run_id,
                generation=new_gen,
                owner_id=owner_id,
                acquired_epoch=now_epoch,
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

        new_gen = self._write(
            run_id=run_id,
            owner_id=owner_id,
            acquired_epoch=now_epoch,
            renewed_epoch=now_epoch,
            if_generation_match=current_gen,
        )
        if new_gen is None:
            return None  # raced with a rival acquirer/renewer.
        return LeaseToken(
            run_id=run_id,
            generation=new_gen,
            owner_id=owner_id,
            acquired_epoch=now_epoch,
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
