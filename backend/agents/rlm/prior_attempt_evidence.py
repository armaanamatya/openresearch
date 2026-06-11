"""Prior-attempt measured evidence — feed past per-cell results to the next attempt.

2026-06-10 All-CNN: attempt N trained `a_allcnn` to a paper-grade 13.11% error;
attempt N+1's lr "repair" moved that cell's lr and killed it (90% = chance),
while the working config sat unread in ``attempts/*/code/outputs/``. Each
attempt re-derives hyperparameters from scratch because nothing carries the
MEASURED outcomes forward.

This module builds a compact, capped text block summarising every prior
attempt's per-cell results (status, headline metric, chosen lr) keyed by a
seed-stripped cell id, newest attempt first — so the implementer can keep
configs that measurably worked and change only what measurably failed. It is
evidence, not advice: values come straight from the per-cell ``metrics.json``
files the cells themselves wrote.

Flag-gated (``OPENRESEARCH_PRIOR_ATTEMPT_EVIDENCE``, default off) and fail-soft:
any error yields an empty block. Pure stdlib; no LLM calls.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_FLAG = "OPENRESEARCH_PRIOR_ATTEMPT_EVIDENCE"

# Strip seed suffixes + collapse separators so cell ids join across attempts
# ("a_base_cifar10_noaug" == "a_base__cifar10_noaug__s42").
_SEED_SUFFIX = re.compile(r"[_]+s(eed)?\d+$", re.IGNORECASE)
_SEP_RUN = re.compile(r"_{2,}")

# Per-cell metric keys worth carrying, in display priority order.
_METRIC_KEYS: tuple[str, ...] = (
    "test_error_pct", "test_accuracy", "metric", "elbo_final", "final_nll",
)
_LR_KEYS: tuple[str, ...] = ("best_lr", "lr", "learning_rate")

_MAX_HISTORY_PER_CELL = 2


def is_enabled() -> bool:
    return bool(os.environ.get(ENV_FLAG, "").strip()) and os.environ.get(
        ENV_FLAG, ""
    ).strip().lower() not in ("0", "false", "off")


def _canon_cell_id(raw: str) -> str:
    cid = _SEED_SUFFIX.sub("", raw.strip().lower())
    return _SEP_RUN.sub("_", cid)


def _summarise_cell(m: dict[str, Any]) -> str | None:
    """One compact fragment for a per-cell metrics dict, or None if unusable."""
    if not isinstance(m, dict):
        return None
    # Aggregate-vs-cell discrimination: agents may nest per_model/per_dataset
    # fragments INSIDE per-cell files (2026-06-10 All-CNN grid #2 did), so the
    # presence of per_model alone is NOT an aggregate signal — only a per_model
    # WITHOUT any cell-level scalar metric is.
    if "per_model" in m and not any(m.get(k) is not None for k in _METRIC_KEYS):
        return None
    parts: list[str] = []
    status = m.get("status")
    if isinstance(status, str) and status:
        parts.append(status)
    for key in _METRIC_KEYS:
        if m.get(key) is not None:
            try:
                parts.append(f"{key}={float(m[key]):g}")
            except (TypeError, ValueError):
                pass
            break
    for key in _LR_KEYS:
        if m.get(key) is not None:
            parts.append(f"lr={m[key]}")
            break
    if m.get("epochs_run") is not None:
        parts.append(f"epochs={m.get('epochs_run')}")
    return " ".join(parts) if parts else None


def _scan_attempt(code_dir: Path) -> dict[str, str]:
    """Map canonical cell id → summary fragment for one attempt's code dir."""
    out: dict[str, str] = {}
    outputs = code_dir / "outputs"
    if not outputs.is_dir():
        return out
    for metrics_path in outputs.glob("*/*/metrics.json"):
        cell_dir = metrics_path.parent.name
        if cell_dir.startswith("_"):  # _cell_smoke etc.
            continue
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a torn file never blocks the block
            continue
        frag = _summarise_cell(m)
        if frag:
            out[_canon_cell_id(cell_dir)] = frag
    return out


def build_evidence_block(
    project_dir: Path | str,
    *,
    max_cells: int = 24,
    max_chars: int = 1800,
) -> str:
    """Return the guidance block (empty string when nothing to report)."""
    try:
        project_dir = Path(project_dir)
        attempts_root = project_dir / "attempts"
        if not attempts_root.is_dir():
            return ""
        # Newest attempt first (dir names are sortable timestamps).
        attempt_dirs = sorted(
            (p for p in attempts_root.iterdir() if p.is_dir()), reverse=True
        )
        history: dict[str, list[str]] = {}
        champions: dict[str, tuple[float, str]] = {}  # cid -> (err%, frag) lowest-err wins
        for idx, attempt in enumerate(attempt_dirs):
            scanned = _scan_attempt(attempt / "code")
            for cid, frag in scanned.items():
                rows = history.setdefault(cid, [])
                if len(rows) < _MAX_HISTORY_PER_CELL:
                    rows.append(frag)
                # Cell-level forward-search crossover: track the all-time champion
                # (champions are scattered across attempts — no single prior run
                # holds every cell's best config).
                m = re.search(r"test_error_pct=([\d.]+)", frag)
                if m:
                    err = float(m.group(1))
                    cur = champions.get(cid)
                    if cur is None or err < cur[0]:
                        champions[cid] = (err, frag)
        if not history:
            return ""

        def _latest_badness(rows: list[str]) -> float:
            # Sort worst-latest first so problems lead. err% high = bad;
            # cells without an error metric sort last.
            m = re.search(r"test_error_pct=([\d.]+)", rows[0])
            return -float(m.group(1)) if m else 0.0

        lines: list[str] = []
        for cid in sorted(history, key=lambda c: _latest_badness(history[c])):
            rows = history[cid]
            rendered = " | ".join(
                f"[{'latest' if i == 0 else 'prior'}] {frag}" for i, frag in enumerate(rows)
            )
            champ = champions.get(cid)
            if champ is not None and champ[1] not in rows:
                rendered += f" | [CHAMPION] {champ[1]}"
            lines.append(f"  {cid}: {rendered}")
        truncated = max(0, len(lines) - max_cells)
        lines = lines[:max_cells]

        block = (
            "\n\nPRIOR-ATTEMPT MEASURED EVIDENCE (same paper; harness-collected "
            "from earlier attempts' per-cell metrics.json — these are MEASURED "
            "results, trust them over fresh guesses):\n"
            "  Keep configurations that measurably worked (re-use their exact "
            "hyperparameters); change ONLY what measurably failed. A cell that "
            "reached a paper-grade metric in ANY attempt has a known-good "
            "config — restore it rather than re-deriving.\n"
            + "\n".join(lines)
        )
        if truncated:
            block += f"\n  (+{truncated} more cells truncated)"
        if len(block) > max_chars:
            block = block[: max_chars - 15].rstrip() + "\n  (truncated)"
        return block + "\n"
    except Exception:  # noqa: BLE001 — evidence is advisory, never fatal
        logger.debug("prior_attempt_evidence: block build failed", exc_info=True)
        return ""


__all__ = ["ENV_FLAG", "build_evidence_block", "is_enabled"]
