"""Append-only, redacted, record-only evidence/lifecycle ledger sidecar.

Stores a ``LedgerRecord`` per primitive call under
``runs/<id>/rlm_state/lifecycle/ledger.jsonl``.  The ledger is **record-only** —
there is NO memoization short-circuit here (that lives in ``primitive_cache.py``).

Gated on ``OPENRESEARCH_LIFECYCLE_LEDGER`` (default OFF).  Every public function
is fail-soft: it never raises, and a disabled flag leaves no files on disk.

Per-primitive projections in ``project_inputs`` emit ONLY bounded, non-corpus
fields.  A sentinel canary test in the test suite verifies that raw paper text
never reaches the on-disk ledger.

Write pattern mirrors ``primitive_cache.put``: a single ``path.open("a")`` +
``write(json.dumps(entry) + "\\n")``; the single-line append is the atomic unit.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants                                                                     #
# --------------------------------------------------------------------------- #

_LEDGER_DIR = "lifecycle"
_LEDGER_FILE = "ledger.jsonl"
_VALID_OUTCOMES = frozenset({"ok", "failed", "raised", "timeout"})

# --------------------------------------------------------------------------- #
# Data model                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LedgerRecord:
    """Immutable record of one primitive invocation."""

    primitive: str
    seq: int
    inputs_projection: dict  # type: ignore[type-arg]
    outputs_pointer: dict  # type: ignore[type-arg]
    evidence_keys: list  # type: ignore[type-arg]
    outcome: str  # ok | failed | raised | timeout
    iteration: int

    def __post_init__(self) -> None:
        if self.outcome not in _VALID_OUTCOMES:
            raise ValueError(
                f"LedgerRecord.outcome must be one of {sorted(_VALID_OUTCOMES)!r},"
                f" got {self.outcome!r}"
            )


# --------------------------------------------------------------------------- #
# Feature flag                                                                  #
# --------------------------------------------------------------------------- #


def lifecycle_ledger_enabled() -> bool:
    """Return True when ``OPENRESEARCH_LIFECYCLE_LEDGER`` is set to a truthy value.

    Accepted truthy values: ``"1"``, ``"true"``, ``"yes"``, ``"on"`` (case-insensitive).
    Default is **OFF** (unset or any other value → False).
    """
    val = os.environ.get("OPENRESEARCH_LIFECYCLE_LEDGER", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Per-primitive input projections (redaction)                                  #
# --------------------------------------------------------------------------- #


def project_inputs(primitive: str, kwargs: dict) -> dict:  # type: ignore[type-arg]
    """Return a bounded, non-corpus projection of ``kwargs`` for ``primitive``.

    Only the bounded fields listed below are emitted; raw paper text, plan prose,
    and any other free-form content are NEVER included.

    Supported primitives and their projections:

    ``plan_reproduction``
        ``{"section_ids": [...], "hparam_keys": [...]}``
        section_ids = list of ids from ``kwargs.get("section_ids")``;
        hparam_keys = KEYS only (never values) from ``kwargs.get("hyperparameters", {})``.

    ``implement_baseline``
        ``{"plan_present": bool, "repair_context_present": bool,
           "sandbox_mode": str, "gpu_mode": str}``

    ``run_experiment``
        ``{"env_id": str, "code_present": bool}``

    All other primitives → ``{}`` (unknown, project nothing).
    """
    if primitive == "plan_reproduction":
        return _project_plan_reproduction(kwargs)
    if primitive == "implement_baseline":
        return _project_implement_baseline(kwargs)
    if primitive == "run_experiment":
        return _project_run_experiment(kwargs)
    return {}


def _project_plan_reproduction(kwargs: dict) -> dict:  # type: ignore[type-arg]
    # Emit only section_ids (a list of opaque ids) and the KEYS of hyperparameters.
    # Never emit section text, paper prose, or hyperparameter values.
    raw_section_ids = kwargs.get("section_ids")
    if isinstance(raw_section_ids, list):
        section_ids: list[Any] = [
            sid for sid in raw_section_ids if isinstance(sid, (str, int))
        ]
    else:
        section_ids = []

    raw_hparams = kwargs.get("hyperparameters")
    if isinstance(raw_hparams, dict):
        hparam_keys: list[Any] = list(raw_hparams.keys())
    else:
        hparam_keys = []

    return {"section_ids": section_ids, "hparam_keys": hparam_keys}


def _project_implement_baseline(kwargs: dict) -> dict:  # type: ignore[type-arg]
    plan_present = bool(kwargs.get("plan"))
    repair_context_present = bool(kwargs.get("repair_context"))
    sandbox_mode = str(kwargs.get("sandbox_mode", ""))
    gpu_mode = str(kwargs.get("gpu_mode", ""))
    return {
        "plan_present": plan_present,
        "repair_context_present": repair_context_present,
        "sandbox_mode": sandbox_mode,
        "gpu_mode": gpu_mode,
    }


def _project_run_experiment(kwargs: dict) -> dict:  # type: ignore[type-arg]
    env_id = str(kwargs.get("env_id", ""))
    code_present = bool(kwargs.get("code"))
    return {"env_id": env_id, "code_present": code_present}


# --------------------------------------------------------------------------- #
# Write                                                                         #
# --------------------------------------------------------------------------- #


def append_record(project_dir: Path, record: LedgerRecord) -> None:
    """Append one JSON line to ``rlm_state/lifecycle/ledger.jsonl``.

    Creates the directory if absent.  Fail-soft: catches and logs all exceptions,
    never raises.  The single-line append is the atomic unit (mirrors
    ``primitive_cache.put``).
    """
    if not lifecycle_ledger_enabled():
        return
    try:
        ledger_dir = project_dir / "rlm_state" / _LEDGER_DIR
        ledger_dir.mkdir(parents=True, exist_ok=True)
        path = ledger_dir / _LEDGER_FILE
        entry = {
            **asdict(record),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except Exception:  # noqa: BLE001 — observability MUST NOT block the run
        logger.exception("lifecycle_ledger: append_record failed for %s", record.primitive)


# --------------------------------------------------------------------------- #
# Read                                                                          #
# --------------------------------------------------------------------------- #


def read_records(project_dir: Path) -> list[LedgerRecord]:
    """Read all records from the ledger.  Fail-soft: returns ``[]`` on any error.

    Malformed JSON lines and records with unknown outcome values are silently
    skipped so a corrupted ledger never breaks the caller.
    """
    path = project_dir / "rlm_state" / _LEDGER_DIR / _LEDGER_FILE
    if not path.exists():
        return []
    records: list[LedgerRecord] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    rec = LedgerRecord(
                        primitive=str(raw.get("primitive", "")),
                        seq=int(raw.get("seq", 0)),
                        inputs_projection=dict(raw.get("inputs_projection") or {}),
                        outputs_pointer=dict(raw.get("outputs_pointer") or {}),
                        evidence_keys=list(raw.get("evidence_keys") or []),
                        outcome=str(raw.get("outcome", "")),
                        iteration=int(raw.get("iteration", 0)),
                    )
                    records.append(rec)
                except (ValueError, TypeError):
                    continue
    except Exception:  # noqa: BLE001
        logger.exception("lifecycle_ledger: read_records failed")
        return []
    return records
