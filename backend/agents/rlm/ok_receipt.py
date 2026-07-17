"""Out-of-process re-grade ok-receipt -- forge-resistant success proof (Track E §6.6).

WHY: ``report.py::run_experiment_success_count`` returns ``None`` (out-of-process
/unknown) whenever ``ctx.cost_ledger`` is absent -- e.g. a re-grade of an
already-completed run, run in a fresh process with no in-memory ledger. Today
that caps the evidence gate at ``partial`` even when the run genuinely
succeeded, because nothing on disk can distinguish "genuinely ran and
succeeded" from "a replay that never actually called ``run_experiment`` at
all". The ``evidence_bundle`` receipt alone does NOT clear that ceiling: it is
derived from ``experiment_runs.jsonl`` after the fact, so a hand-edited/replayed
row could produce a matching bundle without ever running anything.

DESIGN (locked): a receipt is minted ONLY from inside the in-process success
path beside ``primitives.py::_persist_experiment_result`` (a later, separate
wiring phase -- this module is the receipt primitive itself), keyed to the same
``metrics_sha256`` the evidence bundle uses. A replay/re-grade process can read
this file but cannot forge a NEW forge-resistant row, because minting one
requires having actually executed ``run_experiment`` in-process. ``report.py``'s
fallback (also a separate, later wiring phase) may then trust
``count_ok_receipts`` in place of the in-memory ledger count when the ledger is
absent -- lifting the ``partial`` ceiling for a genuine out-of-process re-grade
without ever writing a verdict key itself.

FORGE-RESISTANCE INVARIANT: a receipt counts toward :func:`count_ok_receipts`
ONLY when it carries both ``ok is True`` AND a non-empty ``metrics_sha256``. A
failed run, or a receipt written without a real metrics digest, proves nothing
and must never count -- see the module's own docstring precedent in
``human_intervention.py``/``gpu_ledger.py`` for the shared "never a fitness
term, always additive" posture; this module additionally never gates or writes
any verdict key itself, it only supplies a bool input a LATER gate may consult.

GATING: ``OPENRESEARCH_OK_RECEIPT`` (default OFF). Off ⇒ :func:`write_ok_receipt`
is a pure no-op (returns ``False``, writes nothing) -- byte-identical to today.

FAIL-SOFT: every public function is wrapped so a missing directory, a bad
project_dir, or any unexpected input degrades to a safe value (``False``/``0``)
rather than raising into the run.

Durability recipe: :func:`_atomic_append_jsonl` is a self-contained port of
``human_intervention.py``'s recipe (itself ported from
``reproduction_campaign.py``'s ``CampaignLedger.append_row``) -- a torn-tail
repair (truncate back to the last complete, newline-terminated row before
appending) plus flush+fsync on every write, so a crash mid-append can never
corrupt ``experiment_ok_receipts.jsonl`` into an unreadable file; at worst it
loses only the one row that was mid-write.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from backend.agents.rlm.feature_flags import env_truthy

_STATE_DIRNAME = "rlm_state"
_RECEIPT_FILENAME = "experiment_ok_receipts.jsonl"


def ok_receipt_enabled() -> bool:
    """``OPENRESEARCH_OK_RECEIPT`` -- default OFF, read at call time."""
    return env_truthy("OPENRESEARCH_OK_RECEIPT")


def _atomic_append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON line to ``path``, repairing a torn tail first.

    Port of ``human_intervention.py``'s durability recipe: if a prior crash
    left a partial (non-newline-terminated) final line, truncate back to the
    last complete line before appending -- never entomb a fragment mid-file,
    never prefix a newline that would corrupt the good row before it. Every
    write is flushed + fsynced. The parent directory (``rlm_state/``) is
    created if absent so a caller need not pre-provision it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with path.open("r+b") as fh:
            data = fh.read()
            if not data.endswith(b"\n"):
                cut = data.rfind(b"\n") + 1  # 0 if no "\n" anywhere in the file
                fh.truncate(cut)
                fh.flush()
                os.fsync(fh.fileno())
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str))
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_ok_receipt(
    project_dir: Path | str,
    *,
    experiment_run_id: str,
    ok: bool,
    metrics_sha256: str,
    ts: str,
) -> bool:
    """Append one receipt row to ``rlm_state/experiment_ok_receipts.jsonl``.

    Returns ``True`` on a genuine write, ``False`` otherwise -- flag off, bad
    directory, or any on-disk error. Never raises: this is a durability
    primitive, never allowed to break the caller's real work (the
    ``run_experiment`` persist path). The row is written verbatim as given --
    forge-resistance is enforced at READ time by :func:`count_ok_receipts`, not
    by rejecting a failed/sha-less write here (a failed-run receipt is still a
    legitimate, honest record; it simply never counts as proof of success).
    """
    if not ok_receipt_enabled():
        return False
    try:
        row: dict[str, Any] = {
            "experiment_run_id": experiment_run_id,
            "ok": bool(ok),
            "metrics_sha256": metrics_sha256,
            "ts": ts,
        }
        path = Path(project_dir) / _STATE_DIRNAME / _RECEIPT_FILENAME
        _atomic_append_jsonl(path, row)
        return True
    except Exception:  # noqa: BLE001 -- must never break the caller
        return False


def _read_rows(project_dir: Path | str) -> list[dict[str, Any]]:
    path = Path(project_dir) / _STATE_DIRNAME / _RECEIPT_FILENAME
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def count_ok_receipts(project_dir: Path | str) -> int:
    """Count distinct ``experiment_run_id``\\s with a forge-resistant success receipt.

    A receipt counts ONLY when ``ok is True`` (strict identity, not truthy)
    AND ``metrics_sha256`` is a non-empty string -- a failed run or a receipt
    missing its digest proves nothing and is excluded. Multiple receipts for
    the same ``experiment_run_id`` (e.g. a repair-loop re-run of the same
    experiment) count once. Never raises: an absent/corrupt/unreadable receipt
    log degrades to ``0``, never breaking a caller that consults this as a
    fallback evidence count.
    """
    try:
        run_ids: set[str] = set()
        for row in _read_rows(project_dir):
            if not isinstance(row, dict):
                continue
            if row.get("ok") is not True:
                continue
            sha = row.get("metrics_sha256")
            if not isinstance(sha, str) or not sha:
                continue
            run_id = row.get("experiment_run_id")
            if run_id is None or run_id == "":
                continue
            run_ids.add(str(run_id))
        return len(run_ids)
    except Exception:  # noqa: BLE001 -- fail-soft, never raise into the caller
        return 0


__all__ = ["ok_receipt_enabled", "write_ok_receipt", "count_ok_receipts"]
