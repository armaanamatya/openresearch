"""Unit tests for cell_scheduler — shared pure helpers.

Tests cover:
  * STATUS_* constants are the correct string literals.
  * CellResult construction, to_dict shape.
  * headline_metric: prefers "metric", falls back to "reward_mean"/"accuracy",
    skips booleans, returns None for non-dict / missing / bool values.
  * load_cell_manifest: valid JSON, missing file, corrupt JSON, non-dict JSON.
  * should_skip_cell: all branches (force, no fingerprint, no manifest, non-ok
    manifest, fingerprint mismatch, match → True).
  * write_cell_manifest: full payload written, completed_at absent when now_iso
    is None, OSError swallowed, caller prefix appears in logged warning.
  * is_resume_armed: reads REPROLAB_RESUME_CELLS correctly.
  * deadline_from_timeout: None / zero / positive.
  * clamp_cell_timeout: all combinations of None/set per_cell / overall deadline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from backend.agents.rlm.cell_scheduler import (
    CELL_MANIFEST_NAME,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_OOM_FAILED,
    STATUS_SKIPPED,
    STATUS_TIMEOUT,
    CellResult,
    clamp_cell_timeout,
    deadline_from_timeout,
    headline_metric,
    is_resume_armed,
    load_cell_manifest,
    should_skip_cell,
    write_cell_manifest,
)


# ---------------------------------------------------------------------------
# STATUS_* constants
# ---------------------------------------------------------------------------

class TestStatusConstants:
    def test_ok(self):
        assert STATUS_OK == "ok"

    def test_oom_failed(self):
        assert STATUS_OOM_FAILED == "oom_failed"

    def test_skipped(self):
        assert STATUS_SKIPPED == "skipped"

    def test_error(self):
        assert STATUS_ERROR == "error"

    def test_timeout(self):
        assert STATUS_TIMEOUT == "timeout"

    def test_manifest_name(self):
        assert CELL_MANIFEST_NAME == "cell_manifest.json"


# ---------------------------------------------------------------------------
# CellResult
# ---------------------------------------------------------------------------

class TestCellResult:
    def test_construction_and_to_dict(self):
        r = CellResult(
            cell_id="c0",
            status=STATUS_OK,
            metrics={"acc": 0.9},
            gpu="0",
            retries=1,
            error=None,
        )
        d = r.to_dict()
        assert d == {
            "status": "ok",
            "metrics": {"acc": 0.9},
            "gpu": "0",
            "retries": 1,
            "error": None,
        }

    def test_to_dict_does_not_include_cell_id(self):
        r = CellResult(
            cell_id="c1", status=STATUS_ERROR, metrics=None, gpu="1", retries=0, error="oops"
        )
        assert "cell_id" not in r.to_dict()

    def test_slots_restrict_arbitrary_attributes(self):
        r = CellResult(
            cell_id="c2", status=STATUS_SKIPPED, metrics=None, gpu="0", retries=0, error=None
        )
        with pytest.raises(AttributeError):
            r.extra = "bad"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# headline_metric
# ---------------------------------------------------------------------------

class TestHeadlineMetric:
    def test_prefers_metric_key(self):
        assert headline_metric({"metric": 0.5, "reward_mean": 0.9}) == 0.5

    def test_falls_back_to_reward_mean(self):
        assert headline_metric({"reward_mean": 0.7}) == 0.7

    def test_falls_back_to_accuracy(self):
        assert headline_metric({"accuracy": 0.8}) == 0.8

    def test_none_when_no_known_key(self):
        assert headline_metric({"loss": 1.2}) is None

    def test_none_for_bool_metric(self):
        # Booleans are instances of int in Python; they must be rejected.
        assert headline_metric({"metric": True}) is None
        assert headline_metric({"metric": False}) is None

    def test_none_for_non_dict(self):
        assert headline_metric(None) is None
        assert headline_metric([1, 2]) is None  # type: ignore[arg-type]

    def test_none_for_empty_dict(self):
        assert headline_metric({}) is None

    def test_int_value_accepted(self):
        assert headline_metric({"metric": 3}) == 3

    def test_negative_float_accepted(self):
        assert headline_metric({"reward_mean": -0.5}) == -0.5


# ---------------------------------------------------------------------------
# load_cell_manifest
# ---------------------------------------------------------------------------

class TestLoadCellManifest:
    def test_loads_valid_json_dict(self, tmp_path: Path):
        d = {"cell_id": "c0", "status": "ok", "fingerprint": "FP", "metric": 0.5, "retries": 0}
        (tmp_path / CELL_MANIFEST_NAME).write_text(json.dumps(d), encoding="utf-8")
        result = load_cell_manifest(tmp_path)
        assert result == d

    def test_missing_file_returns_none(self, tmp_path: Path):
        assert load_cell_manifest(tmp_path) is None

    def test_corrupt_json_returns_none(self, tmp_path: Path):
        (tmp_path / CELL_MANIFEST_NAME).write_text("{bad", encoding="utf-8")
        assert load_cell_manifest(tmp_path) is None

    def test_non_dict_json_returns_none(self, tmp_path: Path):
        (tmp_path / CELL_MANIFEST_NAME).write_text("[1, 2, 3]", encoding="utf-8")
        assert load_cell_manifest(tmp_path) is None


# ---------------------------------------------------------------------------
# should_skip_cell
# ---------------------------------------------------------------------------

def _seed_manifest(output_dir: Path, status: str, fingerprint: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {"cell_id": "c0", "status": status, "fingerprint": fingerprint, "metric": 0.5, "retries": 0}
    (output_dir / CELL_MANIFEST_NAME).write_text(json.dumps(data), encoding="utf-8")


class TestShouldSkipCell:
    def test_force_cells_always_reruns(self, tmp_path: Path):
        d = tmp_path / "c0"
        _seed_manifest(d, "ok", "FP")
        assert should_skip_cell("c0", d, {"c0": "FP"}, force_cells={"c0"}) is False

    def test_no_fingerprint_returns_false(self, tmp_path: Path):
        d = tmp_path / "c0"
        _seed_manifest(d, "ok", "FP")
        # fingerprints dict does not have an entry for "c0"
        assert should_skip_cell("c0", d, {}, force_cells=set()) is False

    def test_missing_manifest_returns_false(self, tmp_path: Path):
        d = tmp_path / "c0"
        d.mkdir(parents=True, exist_ok=True)
        assert should_skip_cell("c0", d, {"c0": "FP"}, force_cells=set()) is False

    def test_non_ok_manifest_returns_false(self, tmp_path: Path):
        d = tmp_path / "c0"
        _seed_manifest(d, "oom_failed", "FP")
        assert should_skip_cell("c0", d, {"c0": "FP"}, force_cells=set()) is False

    def test_fingerprint_mismatch_returns_false(self, tmp_path: Path):
        d = tmp_path / "c0"
        _seed_manifest(d, "ok", "OLD_FP")
        assert should_skip_cell("c0", d, {"c0": "NEW_FP"}, force_cells=set()) is False

    def test_matching_ok_manifest_returns_true(self, tmp_path: Path):
        d = tmp_path / "c0"
        _seed_manifest(d, "ok", "FP_MATCH")
        assert should_skip_cell("c0", d, {"c0": "FP_MATCH"}, force_cells=set()) is True

    def test_error_manifest_not_skipped(self, tmp_path: Path):
        d = tmp_path / "c0"
        _seed_manifest(d, "error", "FP")
        assert should_skip_cell("c0", d, {"c0": "FP"}, force_cells=set()) is False


# ---------------------------------------------------------------------------
# write_cell_manifest
# ---------------------------------------------------------------------------

class TestWriteCellManifest:
    def test_writes_correct_fields(self, tmp_path: Path):
        output_dir = tmp_path / "c0"
        metrics = {"metric": 0.42, "reward_mean": 1.2}
        write_cell_manifest(
            output_dir,
            caller="gpu_cell_runner",
            cell_id="c0",
            status=STATUS_OK,
            fingerprint="FP_TEST",
            metrics=metrics,
            retries=1,
            now_iso="2026-06-07T00:00:00Z",
        )
        data = json.loads((output_dir / CELL_MANIFEST_NAME).read_text(encoding="utf-8"))
        assert data["cell_id"] == "c0"
        assert data["status"] == "ok"
        assert data["fingerprint"] == "FP_TEST"
        assert data["metric"] == 0.42  # headline from metrics dict
        assert data["retries"] == 1
        assert data["completed_at"] == "2026-06-07T00:00:00Z"

    def test_completed_at_omitted_when_now_iso_none(self, tmp_path: Path):
        output_dir = tmp_path / "c0"
        write_cell_manifest(
            output_dir,
            cell_id="c0",
            status=STATUS_OK,
            fingerprint=None,
            metrics=None,
            retries=0,
            now_iso=None,
        )
        data = json.loads((output_dir / CELL_MANIFEST_NAME).read_text(encoding="utf-8"))
        assert "completed_at" not in data

    def test_fingerprint_none_written_as_null(self, tmp_path: Path):
        output_dir = tmp_path / "c0"
        write_cell_manifest(
            output_dir,
            cell_id="c0",
            status=STATUS_ERROR,
            fingerprint=None,
            metrics=None,
            retries=0,
            now_iso=None,
        )
        data = json.loads((output_dir / CELL_MANIFEST_NAME).read_text(encoding="utf-8"))
        assert data["fingerprint"] is None

    def test_creates_output_dir_if_missing(self, tmp_path: Path):
        output_dir = tmp_path / "nested" / "c0"
        assert not output_dir.exists()
        write_cell_manifest(
            output_dir,
            cell_id="c0",
            status=STATUS_OK,
            fingerprint=None,
            metrics=None,
            retries=0,
            now_iso=None,
        )
        assert (output_dir / CELL_MANIFEST_NAME).is_file()

    def test_ioerror_swallowed_does_not_raise(self, tmp_path: Path, monkeypatch):
        """OSError during write is fail-soft — must not propagate."""
        output_dir = tmp_path / "c0"
        output_dir.mkdir(parents=True)

        def _bad_write(*args: Any, **kwargs: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", _bad_write)
        # Must not raise.
        write_cell_manifest(
            output_dir,
            cell_id="c0",
            status=STATUS_OK,
            fingerprint=None,
            metrics=None,
            retries=0,
            now_iso=None,
        )

    def test_caller_string_in_log_warning(self, tmp_path: Path, monkeypatch, caplog):
        """The caller= string must appear in the logged warning on write failure."""
        import logging
        output_dir = tmp_path / "c0"
        output_dir.mkdir(parents=True)

        def _bad_write(*args: Any, **kwargs: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", _bad_write)

        with caplog.at_level(logging.WARNING, logger="backend.agents.rlm.cell_scheduler"):
            write_cell_manifest(
                output_dir,
                caller="test_caller_prefix",
                cell_id="c0",
                status=STATUS_OK,
                fingerprint=None,
                metrics=None,
                retries=0,
                now_iso=None,
            )

        assert any("test_caller_prefix" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# is_resume_armed
# ---------------------------------------------------------------------------

class TestIsResumeArmed:
    def test_truthy_values(self, monkeypatch):
        for val in ("1", "true", "yes", "on"):
            monkeypatch.setenv("REPROLAB_RESUME_CELLS", val)
            assert is_resume_armed() is True

    def test_unset_returns_false(self, monkeypatch):
        monkeypatch.delenv("REPROLAB_RESUME_CELLS", raising=False)
        assert is_resume_armed() is False

    def test_empty_string_returns_false(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_RESUME_CELLS", "")
        assert is_resume_armed() is False

    def test_whitespace_only_returns_false(self, monkeypatch):
        monkeypatch.setenv("REPROLAB_RESUME_CELLS", "   ")
        assert is_resume_armed() is False


# ---------------------------------------------------------------------------
# deadline_from_timeout
# ---------------------------------------------------------------------------

class TestDeadlineFromTimeout:
    def test_none_returns_none(self):
        assert deadline_from_timeout(None) is None

    def test_zero_returns_none(self):
        assert deadline_from_timeout(0) is None

    def test_negative_returns_none(self):
        assert deadline_from_timeout(-1.0) is None

    def test_positive_returns_future_monotonic(self):
        before = time.monotonic()
        d = deadline_from_timeout(60.0)
        after = time.monotonic()
        assert d is not None
        assert before + 60.0 <= d <= after + 60.0


# ---------------------------------------------------------------------------
# clamp_cell_timeout
# ---------------------------------------------------------------------------

class TestClampCellTimeout:
    def test_both_none_returns_none(self):
        assert clamp_cell_timeout(None, None) is None

    def test_no_deadline_returns_per_cell(self):
        assert clamp_cell_timeout(30.0, None) == 30.0

    def test_no_deadline_per_cell_none_returns_none(self):
        assert clamp_cell_timeout(None, None) is None

    def test_deadline_only_returns_remaining(self):
        # Deadline 2 seconds from now; per_cell=None → returns ≈2s
        deadline = time.monotonic() + 2.0
        result = clamp_cell_timeout(None, deadline)
        assert result is not None
        assert 1.0 <= result <= 2.5  # generous window for test latency

    def test_per_cell_less_than_remaining_kept(self):
        # per_cell=5s, remaining=100s → clamped = 5s
        deadline = time.monotonic() + 100.0
        result = clamp_cell_timeout(5.0, deadline)
        assert result is not None
        assert 4.9 <= result <= 5.1

    def test_per_cell_greater_than_remaining_clamped(self):
        # per_cell=3600s, remaining=5s → clamped ≈ 5s
        deadline = time.monotonic() + 5.0
        result = clamp_cell_timeout(3600.0, deadline)
        assert result is not None
        assert 1.0 <= result <= 6.0

    def test_expired_deadline_returns_one(self):
        # Deadline already in the past → at least 1.0 (floor enforced)
        deadline = time.monotonic() - 10.0
        result = clamp_cell_timeout(None, deadline)
        assert result is not None
        assert result == pytest.approx(1.0)
