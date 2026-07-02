"""
F2 env-liveness gate — dead-env detection from per-cell rollout health records.

Pure stdlib module — no third-party imports.  Detects environments where the
harness-owned rollout code wrote episode records but every episode was either
unavailable or zero-turn (no interaction was ever served by the env server).

Key design:
  - Producer: ``agentic_rollout.rollout_episode`` appends one JSON line per
    episode to ``<OPENRESEARCH_CELL_OUTPUT_DIR>/env_health.jsonl``.  The agent
    cannot suppress or forge this file because rollout_episode is harness-owned.
  - Consumer (this module): rglob ``env_health.jsonl`` under ``outputs/`` subtrees
    of ``code_dir``, aggregate by env name, and flag any env with POSITIVE evidence
    of no real interaction (episodes_total > 0 and episodes_served == 0).
  - Conservative: an env with NO health data is never flagged (can't tell).  An
    env with ANY served episode is never flagged.
  - Flag default-OFF: ``OPENRESEARCH_ENV_LIVENESS_GATE``.
  - Fail-soft everywhere: any exception -> safe fallback ([] or {}).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def env_liveness_gate_enabled() -> bool:
    """True iff OPENRESEARCH_ENV_LIVENESS_GATE is in {'1','true','yes','on'}."""
    return os.environ.get("OPENRESEARCH_ENV_LIVENESS_GATE", "").strip().lower() in (
        "1", "true", "yes", "on"
    )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_env_health(code_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Scan ``outputs/`` subtrees under ``code_dir`` for ``env_health.jsonl`` files.

    Each line is a JSON episode record:
        {"env": str, "n_turns": int, "reward": float, "unavailable": bool, "served": bool}

    Returns a dict keyed by env name::

        {env: {
            "episodes_total": int,
            "episodes_served": int,
            "episodes_unavailable": int,
            "mean_turns": float,
        }}

    Missing/unreadable files -> {}.  Fail-soft: any exception returns {}.
    """
    try:
        code_path = Path(code_dir)
        if not code_path.is_dir():
            return {}
    except Exception:  # noqa: BLE001
        return {}

    agg: dict[str, dict[str, Any]] = {}

    try:
        for health_path in code_path.rglob("env_health.jsonl"):
            try:
                # Only health files inside an outputs/ subtree.
                if "outputs" not in health_path.parts:
                    continue
                try:
                    lines = health_path.read_text(encoding="utf-8").splitlines()
                except Exception:  # noqa: BLE001
                    continue
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    if not isinstance(rec, dict):
                        continue
                    env_name = str(rec.get("env") or "")
                    n_turns = int(rec.get("n_turns") or 0)
                    unavailable = bool(rec.get("unavailable", False))
                    served = bool(rec.get("served", False))
                    if env_name not in agg:
                        agg[env_name] = {
                            "episodes_total": 0,
                            "episodes_served": 0,
                            "episodes_unavailable": 0,
                            "_turns_sum": 0,
                        }
                    bucket = agg[env_name]
                    bucket["episodes_total"] += 1
                    if served:
                        bucket["episodes_served"] += 1
                    if unavailable:
                        bucket["episodes_unavailable"] += 1
                    bucket["_turns_sum"] += n_turns
            except Exception:  # noqa: BLE001 — per-file fail-soft
                continue
    except Exception:  # noqa: BLE001 — outer scan fail-soft
        return {}

    # Compute mean_turns and drop the internal accumulator.
    result: dict[str, dict[str, Any]] = {}
    for env_name, bucket in agg.items():
        total = bucket["episodes_total"]
        turns_sum = bucket.pop("_turns_sum", 0)
        mean_turns: float = float(turns_sum) / total if total > 0 else 0.0
        result[env_name] = {
            "episodes_total": total,
            "episodes_served": bucket["episodes_served"],
            "episodes_unavailable": bucket["episodes_unavailable"],
            "mean_turns": mean_turns,
        }
    return result


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def dead_envs(code_dir: str | Path) -> list[tuple[str, str]]:
    """Return ``[(env, reason)]`` for each env with POSITIVE evidence of no interaction.

    An env is dead when:
      - ``episodes_total > 0`` (we have health records — the rollout ran), AND
      - ``episodes_served == 0`` (every episode was either unavailable or zero-turn).

    Conservative by design:
      - An env with NO health data is NOT flagged (can't distinguish absent server
        from non-SDAR cell).
      - An env with ANY served episode (``episodes_served >= 1``) is NOT flagged.

    Returns ``[]`` when the gate is disabled (default-OFF byte-identical invariant).
    Fail-soft: any exception -> [].
    """
    if not env_liveness_gate_enabled():
        return []
    try:
        health = read_env_health(code_dir)
        dead: list[tuple[str, str]] = []
        for env_name, stats in health.items():
            total = stats.get("episodes_total", 0)
            served = stats.get("episodes_served", 0)
            if total > 0 and served == 0:
                unavail = stats.get("episodes_unavailable", 0)
                if unavail > 0:
                    reason = (
                        f"env '{env_name}': {total} episode(s) recorded, "
                        f"0 served ({unavail} unavailable) — server was unreachable"
                    )
                else:
                    reason = (
                        f"env '{env_name}': {total} episode(s) recorded, "
                        f"0 served (all zero-turn) — env returned no observations"
                    )
                dead.append((env_name, reason))
        return dead
    except Exception:  # noqa: BLE001
        return []


def env_liveness_scope_gaps(code_dir: str | Path) -> list[dict[str, Any]]:
    """Return scope-gap dicts for each dead env.

    Each gap: ``{"item": env_name, "reason": reason, "kind": "env_setup_failed"}``
    (the canonical exclusion kind, so the gap round-trips through
    ``exclusion.Exclusion.from_gap`` and is excluded from scoring).
    Returns ``[]`` when the gate is disabled or no dead envs found.
    Fail-soft: any exception -> [].
    """
    try:
        gaps: list[dict[str, Any]] = []
        for env_name, reason in dead_envs(code_dir):
            gaps.append({"item": env_name, "reason": reason, "kind": "env_setup_failed"})
        return gaps
    except Exception:  # noqa: BLE001
        return []
