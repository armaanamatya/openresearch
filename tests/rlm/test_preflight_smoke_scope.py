"""Scope-narrowing tests for preflight_smoke — import-closure BFS from entry points.

With OPENRESEARCH_USE_AUTHOR_REPO the entire author repo is seeded into code/, including
orphan files whose module-level imports reference out-of-scope deps not needed by the
actual cell entry point. These tests verify:

  1. Orphan files NOT reachable from any entry point are ignored.
  2. A missing dep directly imported by the entry point IS still caught.
  3. A script named only via cells.json is treated as an entry point.
  4. When NO entry point exists at all, the legacy whole-dir scan is preserved.

The emitted smoke script is run as a real subprocess (just like in production) so we
test the real behaviour, not a mock. Only stdlib modules are used as "good" third-party
roots; ``definitely_missing_pkg_xyz`` is the canonical guaranteed-missing sentinel.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from backend.agents.rlm import preflight_smoke


_MISSING = "definitely_missing_pkg_xyz"


def _emit_and_run(code_dir: Path) -> subprocess.CompletedProcess:
    script = preflight_smoke.emit(code_dir)
    assert script.exists()
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=str(code_dir),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _result(code_dir: Path) -> dict:
    return json.loads(
        (code_dir / "preflight_smoke_result.json").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# Case 1 — Orphan ignored when entry point present
# ---------------------------------------------------------------------------

def test_orphan_not_in_closure_is_ignored(tmp_path: Path):
    """train_cell.py → helper.py (stdlib only); orphan.py has the missing dep.

    The smoke must pass because the orphan is unreachable from the entry point.
    The missing dep must NOT appear in ``probed``.
    """
    code = tmp_path / "code"
    code.mkdir()

    # Entry point: imports json (stdlib) and a local helper.
    (code / "train_cell.py").write_text(
        "import json\nfrom helper import run\n",
        encoding="utf-8",
    )
    # Local helper: imports only os (stdlib).
    (code / "helper.py").write_text(
        "import os\n\ndef run(): pass\n",
        encoding="utf-8",
    )
    # Orphan: module-level import of a guaranteed-missing package.
    # It lives in code/ but is NEVER imported by train_cell.py or helper.py.
    (code / "orphan.py").write_text(
        f"import {_MISSING}\n",
        encoding="utf-8",
    )

    proc = _emit_and_run(code)
    assert proc.returncode == 0, proc.stderr
    res = _result(code)
    assert res["ok"] is True
    assert _MISSING not in res["probed"], (
        f"{_MISSING!r} was probed despite being only in an orphan file: {res['probed']}"
    )


# ---------------------------------------------------------------------------
# Case 2 — Real missing dep in entry point IS still caught
# ---------------------------------------------------------------------------

def test_missing_dep_in_entry_point_is_caught(tmp_path: Path):
    """train_cell.py imports the missing dep at module level → smoke must fail (exit 3)."""
    code = tmp_path / "code"
    code.mkdir()

    (code / "train_cell.py").write_text(
        f"import {_MISSING}\nimport json\n",
        encoding="utf-8",
    )

    proc = _emit_and_run(code)
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    res = _result(code)
    assert res["ok"] is False
    assert any(f["module"] == _MISSING for f in res["failures"]), res["failures"]


# ---------------------------------------------------------------------------
# Case 3 — cells.json-named entry point is honoured
# ---------------------------------------------------------------------------

def test_cells_json_entry_honoured(tmp_path: Path):
    """No train_cell/train/main present; cells.json names mytrainer.py as the entry.

    mytrainer.py has the missing dep → smoke must fail (the cells.json entry is in scope).
    """
    code = tmp_path / "code"
    code.mkdir()

    # cells.json points at mytrainer.py
    (code / "cells.json").write_text(
        json.dumps([{"script": "mytrainer.py", "model": "qwen-1.7b"}]),
        encoding="utf-8",
    )
    (code / "mytrainer.py").write_text(
        f"import {_MISSING}\nimport os\n",
        encoding="utf-8",
    )

    proc = _emit_and_run(code)
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    res = _result(code)
    assert res["ok"] is False
    assert any(f["module"] == _MISSING for f in res["failures"]), res["failures"]


def test_cells_json_dict_form_entry_honoured(tmp_path: Path):
    """cells.json in dict form ``{"cells": [...]}`` with ``entry`` key is also honoured."""
    code = tmp_path / "code"
    code.mkdir()

    (code / "cells.json").write_text(
        json.dumps({"cells": [{"entry": "run_cell.py"}]}),
        encoding="utf-8",
    )
    (code / "run_cell.py").write_text(
        f"import {_MISSING}\n",
        encoding="utf-8",
    )

    proc = _emit_and_run(code)
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    res = _result(code)
    assert res["ok"] is False
    assert any(f["module"] == _MISSING for f in res["failures"]), res["failures"]


def test_cells_json_malformed_does_not_crash_smoke(tmp_path: Path):
    """Malformed cells.json must not crash the smoke (fail-soft, no entry → fallback scan)."""
    code = tmp_path / "code"
    code.mkdir()

    # Malformed JSON: the smoke must not exit with an unhandled exception (exit != 3
    # for non-failure reasons, and definitely not a crash that writes no result file).
    (code / "cells.json").write_text("this is not json {{", encoding="utf-8")
    # No standard entry point either → fallback whole-dir scan.
    (code / "orphan.py").write_text("import json\n", encoding="utf-8")

    proc = _emit_and_run(code)
    # With only stdlib deps the smoke should pass (fallback to whole-dir scan, all good).
    assert proc.returncode == 0, (proc.returncode, proc.stderr)
    assert (code / "preflight_smoke_result.json").exists()


# ---------------------------------------------------------------------------
# Case 4 — No entry point: legacy whole-dir fallback preserves false-negative protection
# ---------------------------------------------------------------------------

def test_fallback_whole_dir_scan_when_no_entry_point(tmp_path: Path):
    """When no train_cell/train/main/cells.json exist, the legacy scan catches the missing dep.

    The whole-dir scan is the backstop so non-cell papers keep full protection.
    """
    code = tmp_path / "code"
    code.mkdir()

    # Only an orphan-style file — no standard entry point.
    (code / "orphan.py").write_text(
        f"import {_MISSING}\n",
        encoding="utf-8",
    )

    proc = _emit_and_run(code)
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    res = _result(code)
    assert res["ok"] is False
    assert any(f["module"] == _MISSING for f in res["failures"]), res["failures"]


# ---------------------------------------------------------------------------
# Additional: BFS follows local imports transitively
# ---------------------------------------------------------------------------

def test_bfs_follows_local_imports_transitively(tmp_path: Path):
    """train_cell.py → local_a.py → local_b.py → missing_dep: must be caught."""
    code = tmp_path / "code"
    code.mkdir()

    (code / "train_cell.py").write_text(
        "import json\nfrom local_a import step\n",
        encoding="utf-8",
    )
    (code / "local_a.py").write_text(
        "from local_b import helper\n\ndef step(): pass\n",
        encoding="utf-8",
    )
    (code / "local_b.py").write_text(
        f"import {_MISSING}\n\ndef helper(): pass\n",
        encoding="utf-8",
    )

    proc = _emit_and_run(code)
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    res = _result(code)
    assert res["ok"] is False
    assert any(f["module"] == _MISSING for f in res["failures"]), res["failures"]


def test_bfs_does_not_follow_orphan_chain(tmp_path: Path):
    """train_cell.py has NO import of orphan_a.py; orphan_a.py → orphan_b.py → missing dep.

    The entire orphan chain must be excluded from the closure.
    """
    code = tmp_path / "code"
    code.mkdir()

    (code / "train_cell.py").write_text(
        "import json\n",
        encoding="utf-8",
    )
    (code / "orphan_a.py").write_text(
        "from orphan_b import fn\n",
        encoding="utf-8",
    )
    (code / "orphan_b.py").write_text(
        f"import {_MISSING}\n\ndef fn(): pass\n",
        encoding="utf-8",
    )

    proc = _emit_and_run(code)
    assert proc.returncode == 0, proc.stderr
    res = _result(code)
    assert res["ok"] is True
    assert _MISSING not in res["probed"]
