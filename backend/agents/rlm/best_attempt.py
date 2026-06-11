"""Best-attempt anti-regression rails — attempts improve on the best, never restart.

The Adam regression pattern (0.831 best on 2026-06-07, then 0.69 / 0.0 / 0.736
/ 0.762 / 0.151 across seven attempts): every attempt re-derived the entire
implementation from scratch, so the run that hit the sweet spot was treated as
disposable history. Its working code, its earned rubric leaves, and its target
were all forgotten — each new attempt explored freely and routinely landed
below the proven baseline.

Three rails, each flag-gated and fail-soft:

* ``seed_reference_code`` (``REPROLAB_SEED_BEST_ATTEMPT``) — copy the best
  prior attempt's ``code/`` into ``code/_best_attempt/`` at run start, so the
  implementer can start FROM the proven solution instead of from zero.
* ``best_attempt_guidance_block`` (same flag) — implementer-prompt block:
  the best score, the pointer to the seeded code, and a leaf-level regression
  list (which rubric leaves the best attempt earned that the latest attempt
  lost, with the best run's evidence summaries) — protect earned leaves before
  chasing new ones.
* ``floored_target`` (``REPROLAB_TARGET_BEST_FLOOR``) — raise the in-run
  ``target_score`` to the best prior attempt's score, so the forced-iteration
  policy refuses to finish below the proven baseline while budget remains
  (wall-clock bypass unchanged).

Leaf ids are stable across attempts of one paper (``generated_rubric.json`` is
a paper-level artifact, never archived), so leaf-level joins are exact.
Pure stdlib; never raises.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_SEED_FLAG = "REPROLAB_SEED_BEST_ATTEMPT"
ENV_TARGET_FLOOR_FLAG = "REPROLAB_TARGET_BEST_FLOOR"

REFERENCE_DIR_NAME = "_best_attempt"

# Heavy / recomputable artifacts never copied into the reference seed.
_SEED_SKIP_DIRS = frozenset({"outputs", "__pycache__", "datasets", REFERENCE_DIR_NAME})
_SEED_SKIP_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".npz")


def _flag_on(name: str) -> bool:
    val = os.environ.get(name, "").strip().lower()
    return bool(val) and val not in ("0", "false", "off")


def _read_report(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a torn report never breaks the rail
        return None


def _score_of(report: dict[str, Any] | None) -> float | None:
    if not isinstance(report, dict):
        return None
    rub = report.get("rubric")
    raw = (rub or {}).get("overall_score") if isinstance(rub, dict) else None
    if raw is None:
        raw = report.get("overall_score")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def find_best_attempt(project_dir: Path | str) -> dict[str, Any] | None:
    """Best-scoring PRIOR attempt of this paper, or None when nothing scored.

    Scans ``attempts/*/final_report.json`` (the current in-flight attempt has
    no final report yet, so it is never self-referenced). Returns
    ``{"dir", "score", "report"}`` with ``dir`` as a Path.
    """
    try:
        attempts_root = Path(project_dir) / "attempts"
        if not attempts_root.is_dir():
            return None
        best: dict[str, Any] | None = None
        for attempt in sorted(p for p in attempts_root.iterdir() if p.is_dir()):
            report = _read_report(attempt / "final_report.json")
            score = _score_of(report)
            if score is None:
                continue
            if best is None or score > best["score"]:
                best = {"dir": attempt, "score": score, "report": report}
        return best
    except Exception:  # noqa: BLE001
        logger.debug("best_attempt: scan failed", exc_info=True)
        return None


def _latest_scored_attempt(project_dir: Path) -> dict[str, Any] | None:
    try:
        attempts_root = Path(project_dir) / "attempts"
        if not attempts_root.is_dir():
            return None
        for attempt in sorted(
            (p for p in attempts_root.iterdir() if p.is_dir()), reverse=True
        ):
            report = _read_report(attempt / "final_report.json")
            if _score_of(report) is not None:
                return {"dir": attempt, "score": _score_of(report), "report": report}
        return None
    except Exception:  # noqa: BLE001
        return None


def _leaves(report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    rub = (report or {}).get("rubric") or {}
    out: dict[str, dict[str, Any]] = {}
    for leaf in rub.get("leaf_scores") or []:
        if isinstance(leaf, dict) and leaf.get("id") is not None and leaf.get("score") is not None:
            out[str(leaf["id"])] = leaf
    return out


def leaf_regressions(
    best_report: dict[str, Any] | None,
    latest_report: dict[str, Any] | None,
    *,
    top_n: int = 8,
    min_delta: float = 0.15,
) -> list[dict[str, Any]]:
    """Leaves the best attempt earned that the latest attempt lost.

    Joined on the stable leaf id; sorted by lost points descending; only
    deltas ≥ ``min_delta`` (grader noise stays out of the list).
    """
    best_leaves = _leaves(best_report)
    latest_leaves = _leaves(latest_report)
    rows: list[dict[str, Any]] = []
    for lid, bleaf in best_leaves.items():
        lleaf = latest_leaves.get(lid)
        if lleaf is None:
            continue
        try:
            delta = float(bleaf["score"]) - float(lleaf["score"])
        except (TypeError, ValueError):
            continue
        if delta < min_delta:
            continue
        rows.append({
            "id": lid,
            "best": float(bleaf["score"]),
            "latest": float(lleaf["score"]),
            "evidence": str(bleaf.get("justification") or "")[:200],
        })
    rows.sort(key=lambda r: r["best"] - r["latest"], reverse=True)
    return rows[:top_n]


def seed_reference_code(project_dir: Path | str) -> str | None:
    """Copy the best prior attempt's ``code/`` into ``code/_best_attempt/``.

    Reference material for the implementer (it copies what it wants) — heavy
    artifacts skipped, idempotent (re-seeds fresh each call), flag-gated.
    Returns the relative path seeded, or None.
    """
    if not _flag_on(ENV_SEED_FLAG):
        return None
    try:
        project_dir = Path(project_dir)
        best = find_best_attempt(project_dir)
        if best is None:
            return None
        src = best["dir"] / "code"
        if not src.is_dir():
            return None
        dst = project_dir / "code" / REFERENCE_DIR_NAME
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True, exist_ok=True)
        copied = 0
        for child in src.iterdir():
            if child.name in _SEED_SKIP_DIRS:
                continue
            if child.is_file():
                if child.suffix in _SEED_SKIP_SUFFIXES:
                    continue
                shutil.copy2(child, dst / child.name)
                copied += 1
            elif child.is_dir():
                shutil.copytree(
                    child, dst / child.name,
                    ignore=shutil.ignore_patterns(
                        "outputs", "__pycache__", "datasets",
                        *(f"*{s}" for s in _SEED_SKIP_SUFFIXES),
                    ),
                    dirs_exist_ok=True,
                )
                copied += 1
        (dst / "_BEST_ATTEMPT_README.txt").write_text(
            f"Best prior attempt: {best['dir'].name} — rubric {best['score']:.4f}.\n"
            "This is the COMPLETE working code of the best-scoring prior attempt.\n"
            "Start from it; copy files verbatim and change only what the\n"
            "regression list in your guidance names.\n",
            encoding="utf-8",
        )
        logger.info(
            "best_attempt: seeded %d item(s) from %s (score %.4f) into code/%s",
            copied, best["dir"].name, best["score"], REFERENCE_DIR_NAME,
        )
        return f"code/{REFERENCE_DIR_NAME}"
    except Exception:  # noqa: BLE001 — seeding must never block a run
        logger.debug("best_attempt: seeding failed", exc_info=True)
        return None


def best_attempt_guidance_block(project_dir: Path | str, *, max_chars: int = 2400) -> str:
    """Implementer-prompt block: best score + seeded-code pointer + regressions."""
    if not _flag_on(ENV_SEED_FLAG):
        return ""
    try:
        project_dir = Path(project_dir)
        best = find_best_attempt(project_dir)
        if best is None:
            return ""
        latest = _latest_scored_attempt(project_dir)
        lines = [
            "",
            "",
            f"BEST PRIOR ATTEMPT — rubric {best['score']:.3f} "
            f"({best['dir'].name}). Its COMPLETE working code is seeded at "
            f"code/{REFERENCE_DIR_NAME}/ — START FROM IT: copy files verbatim and "
            "modify only what is necessary. Do NOT rewrite the solution from "
            "scratch; unforced rewrites are how previous attempts regressed "
            "below this baseline. Protect already-earned rubric leaves before "
            "chasing new ones.",
        ]
        if latest is not None and latest["dir"] != best["dir"]:
            regs = leaf_regressions(best["report"], latest["report"])
            if regs:
                lines.append(
                    f"LEAVES THE BEST ATTEMPT EARNED THAT THE LATEST "
                    f"({latest['score']:.3f}) LOST — restore these specifically:"
                )
                for r in regs:
                    lines.append(
                        f"  - leaf {r['id'][:8]}: best {r['best']:.2f} vs latest "
                        f"{r['latest']:.2f} — best-run evidence: {r['evidence']}"
                    )
        block = "\n".join(lines)
        if len(block) > max_chars:
            block = block[: max_chars - 15].rstrip() + "\n  (truncated)"
        return block + "\n"
    except Exception:  # noqa: BLE001
        logger.debug("best_attempt: guidance block failed", exc_info=True)
        return ""


def floored_target(project_dir: Path | str, target: float | None) -> float | None:
    """Raise ``target_score`` to the best prior attempt's score (flag-gated).

    The forced-iteration policy then refuses FINAL_VAR below the proven
    baseline while iterations/budget remain — the wall-clock bypass is
    untouched, so an honest partial still ships at the deadline.
    """
    if not _flag_on(ENV_TARGET_FLOOR_FLAG):
        return target
    try:
        best = find_best_attempt(project_dir)
        if best is None:
            return target
        floor = float(best["score"])
        if target is None or floor > float(target):
            return floor
        return target
    except Exception:  # noqa: BLE001
        return target


__all__ = [
    "ENV_SEED_FLAG",
    "ENV_TARGET_FLOOR_FLAG",
    "REFERENCE_DIR_NAME",
    "best_attempt_guidance_block",
    "find_best_attempt",
    "floored_target",
    "leaf_regressions",
    "seed_reference_code",
]
