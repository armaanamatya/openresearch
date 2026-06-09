"""Tests for scripts/prune_runs.py — the runs/ GC honoring .preserved.

The contract under test (report.py:~1000): any cleanup that prunes runs/
MUST skip dirs carrying a .preserved marker; additionally only terminal,
old-enough runs are eligible, and dry-run must never delete.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "prune_runs", _REPO / "scripts" / "prune_runs.py"
)
prune_runs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_spec and prune_runs)


def _mk_run(root: Path, name: str, *, status: str | None, age_days: float = 30.0,
            preserved: bool = False) -> Path:
    d = root / name
    d.mkdir(parents=True)
    if status is not None:
        (d / "demo_status.json").write_text(
            json.dumps({"projectId": name, "status": status}), encoding="utf-8"
        )
    if preserved:
        (d / ".preserved").write_text("{}", encoding="utf-8")
    old = time.time() - age_days * 86400
    for p in [d, *d.iterdir()]:
        os.utime(p, (old, old))
    return d


def test_dry_run_selects_but_never_deletes(tmp_path: Path) -> None:
    d = _mk_run(tmp_path, "prj_old_done", status="completed")
    selected = prune_runs.prune(tmp_path, older_than_days=14, delete=False)
    assert selected == [d]
    assert d.exists(), "dry run must not delete anything"


def test_delete_removes_only_eligible(tmp_path: Path) -> None:
    doomed = _mk_run(tmp_path, "prj_doomed", status="failed")
    kept_preserved = _mk_run(tmp_path, "prj_kept", status="completed", preserved=True)
    kept_running = _mk_run(tmp_path, "prj_live", status="running")
    kept_young = _mk_run(tmp_path, "prj_young", status="completed", age_days=1)
    kept_unreadable = _mk_run(tmp_path, "prj_nostatus", status=None)

    selected = prune_runs.prune(tmp_path, older_than_days=14, delete=True)

    assert selected == [doomed]
    assert not doomed.exists()
    for survivor in (kept_preserved, kept_running, kept_young, kept_unreadable):
        assert survivor.exists(), f"{survivor.name} must survive"


def test_keep_list_overrides_eligibility(tmp_path: Path) -> None:
    d = _mk_run(tmp_path, "prj_vip", status="completed")
    selected = prune_runs.prune(
        tmp_path, older_than_days=14, delete=True, keep=frozenset({"prj_vip"})
    )
    assert selected == []
    assert d.exists()


def test_all_terminal_statuses_are_eligible(tmp_path: Path) -> None:
    names = []
    for status in ("completed", "failed", "stopped", "killed", "interrupted"):
        names.append(_mk_run(tmp_path, f"prj_{status}", status=status))
    selected = prune_runs.prune(tmp_path, older_than_days=14, delete=False)
    assert sorted(p.name for p in selected) == sorted(p.name for p in names)


def test_missing_runs_root_is_a_noop(tmp_path: Path) -> None:
    assert prune_runs.prune(tmp_path / "nope", older_than_days=14, delete=True) == []
