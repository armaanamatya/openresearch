"""Observed-DAG node recorder (Track G, step S1 — ``OPENRESEARCH_DAG_BACKBONE``).

WHY: today the reproduction lifecycle is driven imperatively — the root model (or
the ``lifecycle_driver``) decides what runs next, and the only durable trace is
``experiment_runs.jsonl``, a flat append log. The scorecard's ``dag_planning``
dimension can therefore only offer the **S0** view: a *post-hoc, linear* reading
of that log ("these experiments happened in this order"). This module records the
**S1** view instead — each executed unit of work as an explicit graph **node**
(with any known dependency **edges**) in ``rlm_state/dag_nodes.jsonl`` — so the
observed dependency structure is captured *as it happens*, the prerequisite for a
later opt-in scheduler (S2) that can parallelize independent branches and resume
from partial state.

INVARIANT: the DAG is an ORCHESTRATION + OBSERVABILITY artifact — it makes a run
faster and more legible, and it NEVER moves the verdict. The deterministic
evidence layer stays the sole gate (the scorecard's ``dag_planning`` row is
``status="display"``/``gates=False`` precisely because a richer plan graph can
never lift a reproduction verdict).

GATING: ``OPENRESEARCH_DAG_BACKBONE`` (default OFF). Off ⇒ :func:`append_dag_node`
is a pure no-op (returns ``False``, writes nothing) — byte-identical to today.

Durability: :func:`_atomic_append_jsonl` is the torn-tail-repair + fsync recipe
ported from ``human_intervention.py`` / ``gpu_ledger.py`` (itself from
``CampaignLedger.append_row``). Fail-soft: never raises into the caller.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from backend.agents.rlm.feature_flags import env_truthy

_STATE_DIRNAME = "rlm_state"
_DAG_FILENAME = "dag_nodes.jsonl"


def dag_backbone_enabled() -> bool:
    """``OPENRESEARCH_DAG_BACKBONE`` — default OFF, read at call time."""
    return env_truthy("OPENRESEARCH_DAG_BACKBONE")


def _atomic_append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON line, repairing a torn tail first, flushed + fsynced.

    Identical durability posture to the two committed precedents: if a prior
    crash left a partial (non-newline-terminated) final line, truncate back to
    the last complete line before appending so a fragment is never entombed
    mid-file. Creates ``rlm_state/`` if absent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with path.open("r+b") as fh:
            data = fh.read()
            if not data.endswith(b"\n"):
                fh.truncate(data.rfind(b"\n") + 1)  # 0 when no newline anywhere
                fh.flush()
                os.fsync(fh.fileno())
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str))
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


def append_dag_node(
    project_dir: Path | str,
    *,
    node_id: str,
    kind: str,
    ts: str,
    deps: Iterable[str] | None = None,
    status: str = "done",
) -> bool:
    """Record one observed DAG node to ``rlm_state/dag_nodes.jsonl``.

    ``node_id`` is the stable identity of the unit of work (e.g. an
    ``experiment_run_id``); ``deps`` are the ``node_id``\\s this node depended
    on (empty today — S1 records nodes as they execute; edge inference is a
    later step). Returns ``True`` on a genuine write, ``False`` when the flag is
    off or on any on-disk error — never raises (observability must never break a
    run).
    """
    if not dag_backbone_enabled():
        return False
    try:
        row = {
            "node_id": node_id,
            "kind": kind,
            "ts": ts,
            "deps": sorted({str(d) for d in (deps or []) if d}),
            "status": status,
        }
        _atomic_append_jsonl(Path(project_dir) / _STATE_DIRNAME / _DAG_FILENAME, row)
        return True
    except Exception:  # noqa: BLE001 — telemetry must never break the caller
        return False


def read_dag_nodes(project_dir: Path | str) -> list[dict[str, Any]]:
    """Read the recorded DAG nodes; ``[]`` when absent/unreadable (never raises)."""
    path = Path(project_dir) / _STATE_DIRNAME / _DAG_FILENAME
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    except OSError:
        return []
    return rows


__all__ = ["dag_backbone_enabled", "append_dag_node", "read_dag_nodes"]
