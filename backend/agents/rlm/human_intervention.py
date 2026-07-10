"""Deterministic capture of human/operator interventions during a run.

Every operator touch-point during an otherwise-autonomous reproduction run --
supplying credentials, approving a dataset download, resolving an environment
repair, steering via chat, resuming a paused campaign -- is a signal about how
autonomous the run actually was. This module gives every ingress point in the
harness ONE call to make: :func:`record_intervention`. It is purely additive
display/telemetry plumbing:

* It NEVER gates a verdict, score, or campaign decision -- see
  :func:`autonomy_metric`, which is a display-only stat (Track E §6.5).
* It is default-OFF (``OPENRESEARCH_HUMAN_INTERVENTION_LOG``) and byte-identical
  when off: :func:`record_intervention` no-ops and returns ``False`` without
  touching disk.
* It never raises -- a missing ``project_dir``, a permission error, a bad
  filesystem all degrade to ``False`` so a caller can fire-and-forget it
  alongside its real work.

Durability recipe: the JSONL append below is a self-contained port of
``reproduction_campaign.py``'s ``CampaignLedger.append_row`` -- a torn-tail
repair (truncate back to the last complete, newline-terminated row before
appending) plus flush+fsync on every write, so a crash mid-append can never
corrupt ``human_interventions.jsonl`` into an unreadable file; at worst it loses
only the one row that was mid-write.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from backend.agents.rlm.feature_flags import env_truthy

_LOG_FILENAME = "human_interventions.jsonl"

InterventionKind = Literal[
    "credentials",
    "clarification",
    "dataset-approval",
    "env-repair",
    "code-fix",
    "config-choice",
    "scientific-judgment",
    "compute-approval",
    "manual-artifact",
    "resume-approval",
]

# Per-kind autonomy weight: how much one intervention of this kind counts
# against autonomy_score. Legally-required/administrative gates (supplying a
# credential, approving spend, resuming a paused loop) say little about whether
# the AGENT could have proceeded alone, so they are weighted low.
# Technical/scientific assistance (a human had to fix code or make a judgment
# call the agent should have been able to make itself) is weighted at full
# strength.
_KIND_WEIGHTS: dict[str, float] = {
    "credentials": 0.2,
    "resume-approval": 0.3,
    "compute-approval": 0.4,
    "dataset-approval": 0.5,
    "config-choice": 0.6,
    "manual-artifact": 0.7,
    "clarification": 0.8,
    "env-repair": 0.8,
    "code-fix": 1.0,
    "scientific-judgment": 1.0,
}
_DEFAULT_KIND_WEIGHT = 0.7

# Saturation constant for the diminishing-returns density curve in
# autonomy_metric: chosen so a handful of full-weight interventions meaningfully
# move the score while it always stays in [0, 1] no matter how many
# interventions accumulate over a long run.
_DENSITY_SCALE = 3.0


def human_intervention_enabled() -> bool:
    """``OPENRESEARCH_HUMAN_INTERVENTION_LOG`` -- default OFF, read at call time."""
    return env_truthy("OPENRESEARCH_HUMAN_INTERVENTION_LOG")


def _atomic_append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON line to ``path``, repairing a torn tail first.

    Port of ``CampaignLedger.append_row``'s durability recipe: if a prior crash
    left a partial (non-newline-terminated) final line, truncate back to the
    last complete line before appending -- never entomb a fragment mid-file,
    never prefix a newline that would corrupt the good row before it. Every
    write is flushed + fsynced.
    """
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


def record_intervention(
    project_dir: Path,
    *,
    kind: str,
    what: str,
    why: str = "",
    artifact_diff: str = "",
    blocking: bool = False,
) -> bool:
    """Append one intervention row to ``<project_dir>/human_interventions.jsonl``.

    Returns ``True`` on a genuine write, ``False`` otherwise -- flag off, bad
    directory, or any on-disk error. Never raises: this is telemetry, never
    allowed to break the caller's real work. ``kind`` is expected to be one of
    :data:`InterventionKind` but is not hard-validated (a future kind should
    never be rejected by an older harness).
    """
    if not human_intervention_enabled():
        return False
    try:
        row: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "what": what,
            "why": why,
            "artifact_diff": artifact_diff,
            "blocking": blocking,
        }
        _atomic_append_jsonl(Path(project_dir) / _LOG_FILENAME, row)
        return True
    except Exception:  # noqa: BLE001 -- telemetry must never break the caller
        return False


def _read_rows(project_dir: Path) -> list[dict[str, Any]]:
    path = Path(project_dir) / _LOG_FILENAME
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


def autonomy_metric(project_dir: Path) -> dict[str, Any]:
    """Display-only autonomy stat derived from recorded interventions.

    Never gates anything (verdict, score, campaign decision) -- purely
    descriptive telemetry for the eval scorecard / UI. ``autonomy_score`` is
    ``1.0`` with zero interventions and decreases monotonically (never below
    ``0.0``) as weighted interventions accumulate; credentials/administrative
    kinds barely move it, technical/scientific-judgment kinds move it the most
    (see ``_KIND_WEIGHTS``).
    """
    rows = _read_rows(project_dir)
    n_interventions = len(rows)
    n_blocking = sum(1 for r in rows if r.get("blocking"))
    by_kind: dict[str, int] = {}
    weighted_sum = 0.0
    for r in rows:
        k = r.get("kind", "unknown")
        by_kind[k] = by_kind.get(k, 0) + 1
        weighted_sum += _KIND_WEIGHTS.get(k, _DEFAULT_KIND_WEIGHT)
    density = weighted_sum / (weighted_sum + _DENSITY_SCALE) if weighted_sum > 0 else 0.0
    autonomy_score = max(0.0, min(1.0, 1.0 - density))
    return {
        "n_interventions": n_interventions,
        "n_blocking": n_blocking,
        "by_kind": by_kind,
        "autonomy_score": autonomy_score,
    }
