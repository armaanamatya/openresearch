"""Optional durable store for the existing planner ``ReproductionContract``.

The planner already creates an in-memory contract, but it is not a durable
run-level artifact and is not threaded back through every consumer.  This module
starts the round-two migration safely: when explicitly enabled it validates,
persists, and attaches that existing contract to ``RunContext``.  It makes no
scoring, execution, or report decision.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from backend.agents.schemas import ReproductionContract

_FILENAME = "reproduction_contract.json"
_SCHEMA_VERSION = 1
_TRUE = frozenset({"1", "true", "yes", "on"})


def enabled() -> bool:
    """True only when the round-two contract migration is explicitly enabled."""
    return os.environ.get("OPENRESEARCH_REPRO_CONTRACT", "").strip().lower() in _TRUE


def _path(project_dir: Path | str) -> Path:
    return Path(project_dir) / "rlm_state" / _FILENAME


def persist(project_dir: Path | str, contract: ReproductionContract) -> Path | None:
    """Atomically write a validated contract, or return ``None`` on any error."""
    try:
        target = _path(project_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "contract": contract.model_dump(mode="json"),
        }
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, delete=False,
            prefix=".tmp_reproduction_contract_", suffix=".json",
        ) as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            temp_name = fh.name
        os.replace(temp_name, target)
        return target
    except Exception:  # noqa: BLE001 -- persistence is an optional observability layer
        return None


def load(project_dir: Path | str) -> ReproductionContract | None:
    """Load a stored contract when valid; malformed state is deliberately ignored."""
    try:
        raw = json.loads(_path(project_dir).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != _SCHEMA_VERSION:
            return None
        contract = raw.get("contract")
        return ReproductionContract.model_validate(contract) if isinstance(contract, dict) else None
    except Exception:  # noqa: BLE001 -- old/torn state must not affect a run
        return None


def activate(ctx: Any, payload: dict[str, Any]) -> ReproductionContract | None:
    """Validate, attach, and persist a planner result when the migration is on.

    The off state intentionally does not validate, mutate ``ctx``, or create a
    directory.  Enabled failures also leave ``ctx`` unchanged.
    """
    if not enabled():
        return None
    try:
        contract = ReproductionContract.model_validate(payload)
        if persist(ctx.project_dir, contract) is None:
            return None
        ctx.reproduction_contract = contract
        return contract
    except Exception:  # noqa: BLE001 -- never fail a reproduction for this migration
        return None


__all__ = ["activate", "enabled", "load", "persist"]
