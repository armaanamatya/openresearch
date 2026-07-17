"""Pure GKE Job-fencing helpers for durable, generation-fenced orchestration.

WS3 (durable cloud-native orchestration) design's fencing scheme
(``docs/superpowers/specs/2026-07-10-durable-cloud-native-orchestration-ws3-design.md``
§4.2): every unit of GPU work is named/addressed by the tuple ``(run_id,
cell_id, generation)`` where ``generation`` is the fence token minted by
:class:`backend.services.runtime.blob_lease.BlobLease` (its
``LeaseToken.generation``). Two different generations therefore NEVER
collide on a Job name or a metrics blob path, so a superseded (but not yet
reaped) driver's writes can never be mistaken for the current owner's.

This module is entirely PURE -- no cloud SDK calls, no I/O, no clock reads.
The actual GKE Deployment/Job submission and the reaper that deletes
stale-generation Jobs are wired in a LATER phase; this module only supplies
the deterministic naming/decision primitives they will call.
"""

from __future__ import annotations

import hashlib
import re

__all__ = ["fenced_job_name", "fenced_blob_prefix", "adopt_or_submit"]

_JOB_NAME_MAX = 63
_JOB_NAME_PREFIX = "reprolab-cell-"
_INVALID_LABEL_CHARS_RE = re.compile(r"[^a-z0-9]+")
_LIVE_PHASES = frozenset({"Running", "Pending", "Active"})


def _sanitize_label_token(token: str) -> str:
    """Lowercase ``token`` and collapse any run of non-``[a-z0-9]`` chars to
    a single ``-``, trimming leading/trailing ``-`` (DNS-1123 labels must
    start and end alphanumeric)."""
    collapsed = _INVALID_LABEL_CHARS_RE.sub("-", str(token).lower())
    return collapsed.strip("-")


def fenced_job_name(run_id: str, cell_id: str, gen: int) -> str:
    """Deterministic, DNS-1123-label-safe, generation-fenced K8s Job name.

    Shape: ``reprolab-cell-<run>-<cell>-g<gen>``, lowercased, restricted to
    ``[a-z0-9-]``, starting/ending alphanumeric, capped at 63 chars. Same
    ``(run_id, cell_id, gen)`` always yields the identical name; a different
    ``gen`` always yields a different name (this is the fencing property --
    a superseded generation's Job can never collide with the current one's).

    When the natural (uncompressed) name would exceed 63 chars, the
    over-length middle section is deterministically replaced by an 8-hex
    ``sha256`` digest of the exact ``(run_id, cell_id, gen)`` tuple -- this
    keeps the result unique and deterministic while always fitting the cap;
    the ``-g<gen>`` tail is never truncated away, so the fencing property
    (different gen -> different name) survives compression too.
    """
    safe_run = _sanitize_label_token(run_id)
    safe_cell = _sanitize_label_token(cell_id)
    tail = f"-g{gen}"
    natural = f"{_JOB_NAME_PREFIX}{safe_run}-{safe_cell}{tail}"
    if len(natural) <= _JOB_NAME_MAX:
        return natural

    digest = hashlib.sha256(
        f"{run_id}\x00{cell_id}\x00{gen}".encode("utf-8")
    ).hexdigest()[:8]
    # Reserve room for the prefix, the digest, the tail, and the "-" that
    # joins the (possibly empty) middle to the digest.
    reserved = len(_JOB_NAME_PREFIX) + len(digest) + len(tail) + 1
    middle_budget = max(0, _JOB_NAME_MAX - reserved)
    middle = f"{safe_run}-{safe_cell}"[:middle_budget].strip("-")
    body = f"{middle}-{digest}" if middle else digest
    return f"{_JOB_NAME_PREFIX}{body}{tail}"


def fenced_blob_prefix(run_id: str, gen: int, *, cell_id: str | None = None) -> str:
    """Generation-scoped GCS prefix so two generations never share evidence.

    ``runs/<run_id>/gen-<gen>/`` (run-level), or
    ``runs/<run_id>/gen-<gen>/cells/<cell_id>/`` when ``cell_id`` is given --
    either way, a superseded generation's ``metrics.json`` lives under a
    disjoint prefix from the current generation's, so a stale writer can
    never clobber the live one's evidence.
    """
    if cell_id is None:
        return f"runs/{run_id}/gen-{gen}/"
    return f"runs/{run_id}/gen-{gen}/cells/{cell_id}/"


def adopt_or_submit(existing_phase: str | None, *, already_succeeded: bool) -> str:
    """Pure core of "adopt-by-name-on-409": what should the caller do next?

    - ``already_succeeded=True`` -> ``"skip"`` (the work is already done,
      regardless of what the existing Job's phase says).
    - ``existing_phase`` in ``{"Running", "Pending", "Active"}`` ->
      ``"adopt"`` (a live Job already owns this fenced name -- attach to it
      instead of creating a duplicate).
    - ``existing_phase is None`` (nothing exists under this fenced name), or
      any other phase not covered above -> ``"submit"`` (create the Job).
    """
    if already_succeeded:
        return "skip"
    if existing_phase in _LIVE_PHASES:
        return "adopt"
    return "submit"
