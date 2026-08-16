"""Evidence-decision audit log — a structured, append-only record of every
evidence-gate decision so a verdict is auditable from logs alone.

Each row: ``{ts, gate, outcome, detail?, ...extra}`` appended to
``runs/<id>/rlm_state/evidence_decisions.jsonl``. This complements the free
`scripts/evidence_replay.py` audit: replay reconstructs what the gates WOULD
find; this log records what they DID decide during the live run.

Flag-gated ``OPENRESEARCH_EVIDENCE_DECISION_LOG`` (default OFF ⇒ no file written,
byte-identical). Pure I/O, fail-soft — a logging failure never breaks a run.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["record_evidence_decision", "read_evidence_decisions", "evidence_log_enabled"]

_FLAG = "OPENRESEARCH_EVIDENCE_DECISION_LOG"
_LOG_NAME = "evidence_decisions.jsonl"


def evidence_log_enabled() -> bool:
    """Canonical default-OFF flag idiom."""
    return os.environ.get(_FLAG, "").strip().lower() in ("1", "true", "yes")


def _log_path(run_dir: Path) -> Path:
    return Path(run_dir) / "rlm_state" / _LOG_NAME


def record_evidence_decision(
    run_dir: Path | str,
    *,
    gate: str,
    outcome: str,
    detail: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one structured evidence-gate decision row (no-op when the flag is off).

    Args:
        gate: the mechanism, e.g. ``"grader_integrity"`` / ``"eval_coverage"`` / ``"state_contract"``.
        outcome: the decision, e.g. ``"ok"`` / ``"evidence_tampered"`` / ``"veto"``.
        detail: optional human-readable reason.
        extra: optional structured fields (fingerprints, thresholds, cell ids).

    Fail-soft: any error is swallowed (logging must never break the run).
    """
    if not evidence_log_enabled():
        return
    try:
        row: dict[str, Any] = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "gate": gate,
            "outcome": outcome,
        }
        if detail is not None:
            row["detail"] = detail
        if extra:
            for k, v in extra.items():
                row.setdefault(k, v)
        path = _log_path(Path(run_dir))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    except Exception:  # noqa: BLE001 — the audit log must never break a run.
        logger.debug("record_evidence_decision failed", exc_info=True)


def read_evidence_decisions(run_dir: Path | str) -> list[dict[str, Any]]:
    """Every parseable decision row, in file order (empty if absent). Fail-soft."""
    path = _log_path(Path(run_dir))
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001 — missing / unreadable → empty.
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:  # noqa: BLE001 — skip a torn line.
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows
