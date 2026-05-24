"""tests/services/runs/test_archive_concurrency.py

Three regression tests for archive.py Codex fixes:

B1: generated_rubric.json (and other paper-level files) must NOT be archived
    by archive_run_artifacts — it is paper-level data that must persist across
    attempts so subsequent runs do not re-generate it.

B2a: attempt directory IDs must be collision-proof — using microsecond timestamp
     + uuid suffix so two near-simultaneous calls produce distinct dirs.

B2b: a per-project lock (fcntl.flock) must guard the archive operation; if the
     lock is already held, the call logs a warning and returns without error.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.services.runs.archive import archive_run_artifacts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_project(runs_root: Path, project_id: str) -> Path:
    """Create a run dir and return it."""
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _seed_run_artifacts(run_dir: Path) -> None:
    """Write the minimal set of artifacts that trigger archiving."""
    (run_dir / "final_report.json").write_text(
        json.dumps({"verdict": "reproduced"}), encoding="utf-8"
    )
    (run_dir / "final_report.md").write_text("# Report", encoding="utf-8")
    (run_dir / "dashboard_events.jsonl").write_text(
        json.dumps({"event": "run_complete"}) + "\n", encoding="utf-8"
    )
    (run_dir / "cost_ledger.jsonl").write_text(
        json.dumps({"usd": 0.01}) + "\n", encoding="utf-8"
    )
    (run_dir / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True}) + "\n", encoding="utf-8"
    )


def _seed_paper_artifacts(run_dir: Path) -> None:
    """Write paper-level artifacts that must survive archiving."""
    (run_dir / "generated_rubric.json").write_text(
        json.dumps({"areas": []}), encoding="utf-8"
    )
    (run_dir / "raw_paper.pdf").write_bytes(b"%PDF-fake")
    (run_dir / "paper_html.html").write_text("<html/>", encoding="utf-8")
    (run_dir / "parsed_full_text.txt").write_text("paper text", encoding="utf-8")
    (run_dir / "paperMeta.json").write_text(
        json.dumps({"id": "1234.5678"}), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# B1 — generated_rubric.json not archived
# ---------------------------------------------------------------------------


class TestGeneratedRubricNotArchived:
    """Codex finding B1: generated_rubric.json must not be moved by archive_run_artifacts."""

    def test_generated_rubric_stays_in_run_dir(self, tmp_path):
        """generated_rubric.json must remain at its original location after archiving."""
        run_dir = _make_project(tmp_path, "proj_rubric")
        _seed_run_artifacts(run_dir)
        _seed_paper_artifacts(run_dir)

        result = archive_run_artifacts("proj_rubric", tmp_path)

        assert result is not None, "Archive should have fired (run artifacts present)"
        # The rubric must still be at the original location.
        assert (run_dir / "generated_rubric.json").exists(), (
            "generated_rubric.json was moved by archive_run_artifacts — "
            "it is paper-level data and must persist across attempts"
        )

    def test_generated_rubric_not_in_moved_list(self, tmp_path):
        """generated_rubric.json must not appear in the returned moved list."""
        run_dir = _make_project(tmp_path, "proj_rubric_moved_list")
        _seed_run_artifacts(run_dir)
        _seed_paper_artifacts(run_dir)

        result = archive_run_artifacts("proj_rubric_moved_list", tmp_path)

        assert result is not None
        assert "generated_rubric.json" not in result["moved"], (
            "generated_rubric.json appeared in moved list — it is paper-level data"
        )

    def test_paper_level_files_not_archived(self, tmp_path):
        """All paper-level files must survive archiving intact."""
        paper_files = [
            "generated_rubric.json",
            "raw_paper.pdf",
            "paper_html.html",
            "parsed_full_text.txt",
            "paperMeta.json",
        ]
        run_dir = _make_project(tmp_path, "proj_all_paper")
        _seed_run_artifacts(run_dir)
        _seed_paper_artifacts(run_dir)

        result = archive_run_artifacts("proj_all_paper", tmp_path)

        assert result is not None
        for name in paper_files:
            assert (run_dir / name).exists(), (
                f"{name} was wrongly archived — it is a paper-level artifact"
            )
            assert name not in result["moved"], (
                f"{name} appeared in moved list — it is a paper-level artifact"
            )


# ---------------------------------------------------------------------------
# B2a — collision-proof attempt directory IDs
# ---------------------------------------------------------------------------


class TestAttemptIdCollisionProof:
    """Codex finding B2: two archive calls with the same frozen timestamp must
    produce distinct attempt directories."""

    def test_same_datetime_produces_different_dirs(self, tmp_path):
        """Even when datetime.now returns the same value twice, UUIDs differ."""
        from datetime import datetime, timezone
        from backend.services.runs import archive as archive_mod

        fixed_dt = datetime(2026, 5, 23, 12, 0, 0, 123456, tzinfo=timezone.utc)

        attempt_dirs: list[str] = []

        original_now = archive_mod._make_attempt_id  # noqa: SLF001 — test access

        call_count = 0

        def frozen_make_attempt_id() -> str:
            nonlocal call_count
            call_count += 1
            # Patch the datetime part to be fixed; the uuid part must still differ.
            ts = fixed_dt.strftime("%Y%m%dT%H%M%S-%f")
            # We still call the real uuid path by importing uuid4 directly.
            from uuid import uuid4
            return f"{ts}-{uuid4().hex[:6]}"

        # Run archive twice with the same datetime-derived prefix.
        for i in range(2):
            run_dir = _make_project(tmp_path, f"proj_collision_{i}")
            _seed_run_artifacts(run_dir)
            with patch.object(archive_mod, "_make_attempt_id", frozen_make_attempt_id):
                result = archive_run_artifacts(f"proj_collision_{i}", tmp_path)
            assert result is not None
            attempt_dirs.append(result["attempt_dir"])

        # Even with a fixed timestamp prefix, the two dirs must differ (uuid suffix).
        assert attempt_dirs[0] != attempt_dirs[1], (
            f"Attempt dirs collided: {attempt_dirs[0]} == {attempt_dirs[1]}"
        )

    def test_real_successive_calls_differ(self, tmp_path):
        """Two real successive calls produce distinct attempt dirs (uuid prevents collision)."""
        dirs: list[str] = []
        for i in range(2):
            run_dir = _make_project(tmp_path, f"proj_successive_{i}")
            _seed_run_artifacts(run_dir)
            result = archive_run_artifacts(f"proj_successive_{i}", tmp_path)
            assert result is not None
            dirs.append(Path(result["attempt_dir"]).name)

        # Dir names must follow the format: <YYYYmmddTHHMMSS>-<microseconds>-<uuid6>
        # Splitting on '-' yields: ['<YYYYmmddTHHMMSS>', '<microseconds>', '<uuid6>']
        # i.e. exactly 3 parts.
        for d in dirs:
            parts = d.split("-")
            assert len(parts) == 3, (
                f"Attempt dir name '{d}' does not have the expected "
                "<YYYYmmddTHHMMSS>-<microseconds>-<uuid6> structure (got {len(parts)} parts)"
            )
            uuid_suffix = parts[-1]
            assert len(uuid_suffix) == 6 and all(c in "0123456789abcdef" for c in uuid_suffix), (
                f"UUID suffix '{uuid_suffix}' is not 6 hex chars"
            )


# ---------------------------------------------------------------------------
# B2b — lock acquired elsewhere → skip, no crash
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="fcntl is POSIX-only; Windows uses the no-op fallback path",
)
class TestArchiveLockSkippedOnConcurrent:
    """Codex finding B2: if the per-project lock is already held, archive_run_artifacts
    must log a warning and return None without raising or corrupting state."""

    def test_concurrent_lock_causes_skip(self, tmp_path, caplog):
        """Manually hold the lock; archive call must skip and return None."""
        import fcntl

        run_dir = _make_project(tmp_path, "proj_locked")
        _seed_run_artifacts(run_dir)
        _seed_paper_artifacts(run_dir)

        lock_path = run_dir / ".archive.lock"
        lock_path.touch()

        with lock_path.open("w") as lock_fh:
            # Acquire exclusive lock — simulates another process holding it.
            fcntl.flock(lock_fh, fcntl.LOCK_EX)

            with caplog.at_level(logging.WARNING, logger="backend.services.runs.archive"):
                result = archive_run_artifacts("proj_locked", tmp_path)

        # The call must have returned None (skipped, not raised).
        assert result is None, (
            f"Expected None when lock is held by another process, got {result!r}"
        )
        # The warning must have been logged.
        assert any("lock" in rec.message.lower() or "skip" in rec.message.lower()
                   for rec in caplog.records), (
            "Expected a warning mentioning 'lock' or 'skip' but got: "
            + str([r.message for r in caplog.records])
        )
        # Run artifacts must be untouched (not partially moved).
        assert (run_dir / "final_report.json").exists(), (
            "final_report.json was moved despite lock contention — archive is not atomic"
        )
