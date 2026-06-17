"""Safety tests for the zero-GPU A1 capture: pool_ready is the kill-trigger that
guarantees we stop BEFORE any run_experiment / GPU cell. It must be False until the
graded pool is fully on disk, and True only then."""
from pathlib import Path

from scripts.bes_a1_safe_capture import pool_ready


def test_pool_not_ready_when_nothing_written(tmp_path: Path):
    assert pool_ready(tmp_path, n=3) is False


def test_pool_not_ready_with_candidates_but_no_graded_pool(tmp_path: Path):
    # snapshots exist but the pool hasn't been graded/persisted yet -> NOT safe to read,
    # and (more importantly) this is not yet the documented safe-kill point.
    for i in range(3):
        (tmp_path / "candidates" / f"rlm_impl_{i}").mkdir(parents=True)
    assert pool_ready(tmp_path, n=3) is False


def test_pool_not_ready_with_graded_pool_but_too_few_snapshots(tmp_path: Path):
    (tmp_path / "rlm_state").mkdir()
    (tmp_path / "rlm_state" / "bes_candidates.json").write_text("{}")
    (tmp_path / "candidates" / "rlm_impl_0").mkdir(parents=True)  # only 1 of 3
    assert pool_ready(tmp_path, n=3) is False


def test_pool_ready_when_graded_pool_and_all_snapshots_present(tmp_path: Path):
    (tmp_path / "rlm_state").mkdir()
    (tmp_path / "rlm_state" / "bes_candidates.json").write_text("{}")
    for i in range(3):
        (tmp_path / "candidates" / f"rlm_impl_{i}").mkdir(parents=True)
    assert pool_ready(tmp_path, n=3) is True
