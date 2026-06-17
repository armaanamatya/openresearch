"""Content-addressed code snapshots for honest within-run anti-regression (A4).

The retired best-of-run ``max()`` floor papered over a real case: a repair pass
that REGRESSES the code while the grader (noisily) banks a better earlier score
— shipping a *score* detached from the *artifact*. The honest fix is to keep the
ACTUAL CODE that earned each grade and, at finalize, restore the snapshot whose
median-of-N grade is highest, so ``score ≡ best artifact actually produced``.

This module is the snapshot-grade-restore primitive that unifies:

* within-run anti-regression (this module's registry, keyed by ``evidence_key``),
* BES parallel-candidate champion (``bes_rlm._snapshot_code`` — same ignore set),
* ``best_attempt`` cross-attempt champion.

Snapshots exclude heavy artifacts (``*.pt``, ``datasets/``, ``outputs/``,
``.venv``, ...) — mirroring ``bes_rlm._SNAPSHOT_IGNORE`` — so a snapshot is just
SOURCE. Restore therefore only ever overwrites source files; it deliberately
does NOT delete the heavy artifacts (datasets, outputs, checkpoints) that live
in ``code/`` next to the source.

Pure of network/global state. Disk I/O (snapshot copy + JSON registry) is the
module's job; the registry is fail-soft (a corrupt/missing file reads as empty).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "SNAPSHOT_IGNORE",
    "snapshot_code",
    "restore_snapshot",
    "snapshot_rubric",
    "restore_rubric",
    "record_champion",
    "best_champion",
]

# Heavy artifacts never belong in a source snapshot — identical set to
# ``bes_rlm._SNAPSHOT_IGNORE`` / ``best_attempt.seed_reference_code``. Replicated
# (not imported) so this module has no dependency on the BES feature being wired.
SNAPSHOT_IGNORE = shutil.ignore_patterns(
    "__pycache__", "datasets", "outputs", ".venv", "wandb",
    "*.pt", "*.pth", "*.ckpt", "*.safetensors",
)


def snapshot_code(code_dir: Path, dest_dir: Path) -> Path:
    """Copy ``code_dir`` → ``dest_dir`` excluding heavy artifacts; return ``dest_dir``.

    ``dest_dir`` is removed first if it already exists, so a snapshot is always a
    clean content-addressed copy (no stale files from a prior write at the same
    path). Only source survives the ignore set, so the snapshot is cheap.
    """
    code_dir = Path(code_dir)
    dest_dir = Path(dest_dir)
    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(code_dir, dest_dir, ignore=SNAPSHOT_IGNORE, dirs_exist_ok=True)
    return dest_dir


def restore_snapshot(snapshot_dir: Path, code_dir: Path) -> None:
    """Restore a source snapshot back into ``code_dir`` (overwrite source only).

    Copies every file under ``snapshot_dir`` into the matching location in
    ``code_dir``, overwriting. Because a snapshot contains ONLY source (the
    ignore set stripped the heavy artifacts at snapshot time), this never
    touches the ``datasets/``, ``outputs/``, or ``*.pt`` artifacts that live in
    ``code_dir`` — they are left in place for the restored source to consume.

    It does not delete ``code_dir`` source files that are absent from the
    snapshot; the snapshot is authoritative for what it carries, and leaving
    extra files is the conservative choice for anti-regression (we are restoring
    a known-good source state, not pruning).
    """
    snapshot_dir = Path(snapshot_dir)
    code_dir = Path(code_dir)
    code_dir.mkdir(parents=True, exist_ok=True)
    for src in snapshot_dir.rglob("*"):
        rel = src.relative_to(snapshot_dir)
        dst = code_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        # copy2 preserves mtime so a downstream mtime-sensitive reader sees the
        # restored file as the (older) snapshot time, not "just written".
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Rubric-block snapshot / restore
# ---------------------------------------------------------------------------

def snapshot_rubric(rubric: dict, snapshot_dir: "Path") -> None:
    """Persist the graded rubric block next to the code snapshot so a restore
    re-materializes score+leaves+meets_target as ONE coherent evidence state."""
    p = Path(snapshot_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "rubric_block.json").write_text(json.dumps(rubric, indent=2), encoding="utf-8")


def restore_rubric(snapshot_dir: "Path") -> dict | None:
    """Return the snapshotted rubric block, or None if absent/unreadable (fail-soft)."""
    try:
        return json.loads((Path(snapshot_dir) / "rubric_block.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Champion registry — JSON-backed, fail-soft
# ---------------------------------------------------------------------------

def _load_registry(registry_path: Path) -> list[dict]:
    """Read the registry as a list of entries; fail-soft to ``[]``."""
    try:
        raw = Path(registry_path).read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict)]


def record_champion(
    registry_path: Path,
    *,
    evidence_key: str,
    snapshot_dir: Path | str,
    median_score: float,
    sample_count: int = 1,
) -> dict:
    """Append a champion entry to the JSON registry and return it.

    Each entry records the ``evidence_key`` it was graded at, the snapshot dir
    holding the code that earned the grade, and the ``median_score`` (the
    median-of-N grade — never a MAX-over-noise). ``seq`` is a monotonically
    increasing recency counter used to break ties in :func:`best_champion`.

    Fail-soft: a write failure logs and still returns the entry the caller can
    keep in memory; the registry is an optimization, never a correctness gate.
    """
    registry_path = Path(registry_path)
    entries = _load_registry(registry_path)
    try:
        score = float(median_score)
    except (TypeError, ValueError):
        score = 0.0
    seq = (max((int(e.get("seq", 0)) for e in entries), default=0) + 1) if entries else 1
    entry = {
        "evidence_key": str(evidence_key),
        "snapshot_dir": str(snapshot_dir),
        "median_score": score,
        "sample_count": int(sample_count),
        "seq": seq,
    }
    entries.append(entry)
    try:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("champion_artifact: could not persist registry %s", registry_path, exc_info=True)
    return entry


def best_champion(registry_path: Path) -> dict | None:
    """Return the registry entry with the HIGHEST ``median_score``.

    Ties are broken toward the most recently recorded entry (highest ``seq``).
    Returns ``None`` when the registry is empty / missing / unreadable.
    """
    entries = _load_registry(registry_path)
    if not entries:
        return None

    def _key(e: dict) -> tuple[float, int]:
        try:
            score = float(e.get("median_score"))
        except (TypeError, ValueError):
            score = float("-inf")
        try:
            seq = int(e.get("seq", 0))
        except (TypeError, ValueError):
            seq = 0
        return (score, seq)

    return max(entries, key=_key)
