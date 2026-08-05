"""Cross-machine discovery of preserved run artifacts for guard tests.

Several guard tests exercise fabrication/grounding defenses against REAL
artifacts from preserved reproduction runs (a parsed VAE paper text, observed
``train.py`` files). Those artifacts historically lived under ONE developer's
absolute home path, which made the tests machine-pinned: silently passing on
a fallback everywhere else, and failing (not skipping) on multi-user VMs
where the path exists but belongs to another user.

This helper generalizes discovery so ANY machine — a GCP VM with several
developer homes, a laptop, CI — exercises the real artifacts when a readable
copy exists, and degrades deterministically (fallback text / skip) otherwise.

Search order (first READABLE match wins; deterministic within each root):
  1. ``$OPENRESEARCH_PRESERVED_RUNS_ROOT`` — explicit override, always first.
  2. ``<repo>/runs`` — the checkout's own runs directory.
  3. ``~/openresearch/runs`` and ``~/or/runs`` — the current user's checkouts.
  4. ``/home/*/openresearch/runs`` and ``/home/*/or/runs`` — other users on a
     shared POSIX VM (sorted for determinism).

Existence is deliberately not enough: another user's home may exist but be
unreadable, and a ``PermissionError`` mid-test reads as a failure rather than
"artifact not available here". Every candidate is probed with a real
``open()`` before being returned.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _readable(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            fh.read(1)
        return True
    except OSError:
        return False


def candidate_runs_roots() -> list[Path]:
    """All runs/ roots to probe, best-first. Only existing directories."""
    roots: list[Path] = []
    override = os.environ.get("OPENRESEARCH_PRESERVED_RUNS_ROOT", "").strip()
    if override:
        roots.append(Path(override))
    roots.append(_REPO_ROOT / "runs")
    home = Path.home()
    roots.extend([home / "openresearch" / "runs", home / "or" / "runs"])
    home_root = Path("/home")
    if home_root.is_dir():  # POSIX multi-user VM
        try:
            for user_dir in sorted(home_root.iterdir()):
                roots.extend(
                    [user_dir / "openresearch" / "runs", user_dir / "or" / "runs"]
                )
        except OSError:
            pass
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        try:
            if root not in seen and root.is_dir():
                seen.add(root)
                out.append(root)
        except OSError:
            continue
    return out


def find_preserved_file(relative_glob: str) -> Path | None:
    """First readable file matching ``relative_glob`` under any runs root.

    The glob is applied per root (e.g. ``*prj_03271ba130d423fe/parsed_full_text.txt``
    matches both a plain project dir and a ``_preserved_<note>_prj_...`` rename).
    Returns ``None`` when no readable match exists anywhere — callers fall back
    or skip, they never fail on absence.
    """
    for root in candidate_runs_roots():
        try:
            matches = sorted(p for p in root.glob(relative_glob) if p.is_file())
        except OSError:
            continue
        for match in matches:
            if _readable(match):
                return match
    return None


__all__ = ["candidate_runs_roots", "find_preserved_file"]
