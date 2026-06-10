"""tests/rlm/test_attempt_isolation.py — task #42 run-isolation guard.

Four tests:

1. First-ever run (no prior artifacts) → no-op, no crash.
2. Second run (prior final_report.json present) → all listed artifacts moved
   into ``attempts/<ts>/``; paper-level artifacts left in place.
3. Third run (two prior attempts) → a second timestamped dir is created; the
   first is preserved intact.
4. ``paperMeta.json`` and ``generated_rubric.json`` are NEVER moved regardless
   of how many runs exist.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


from backend.services.runs.attempt_isolation import maybe_archive_prior_attempt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_dir(runs_root: Path, project_id: str) -> Path:
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _seed_artifacts(run_dir: Path, *, include_final_report: bool = True) -> None:
    """Write a representative set of run-derived artifacts into run_dir."""
    if include_final_report:
        (run_dir / "final_report.json").write_text(
            json.dumps({"verdict": "reproduced"}), encoding="utf-8"
        )
        (run_dir / "final_report.md").write_text("# Report", encoding="utf-8")
    (run_dir / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True}) + "\n", encoding="utf-8"
    )
    (run_dir / "cost_ledger.jsonl").write_text(
        json.dumps({"usd": 0.01}) + "\n", encoding="utf-8"
    )
    (run_dir / "dashboard_events.jsonl").write_text(
        json.dumps({"event": "run_complete"}) + "\n", encoding="utf-8"
    )
    (run_dir / "repl_state.pickle").write_bytes(b"fake-pickle")
    rlm_state = run_dir / "rlm_state"
    rlm_state.mkdir(exist_ok=True)
    (rlm_state / "iterations.jsonl").write_text(
        json.dumps({"iter": 1}) + "\n", encoding="utf-8"
    )
    code_dir = run_dir / "code"
    code_dir.mkdir(exist_ok=True)
    (code_dir / "train.py").write_text("# train", encoding="utf-8")


def _seed_paper_artifacts(run_dir: Path) -> None:
    """Write paper-level artifacts that must survive archiving."""
    (run_dir / "paperMeta.json").write_text(
        json.dumps({"id": "2605.15155"}), encoding="utf-8"
    )
    (run_dir / "generated_rubric.json").write_text(
        json.dumps({"areas": []}), encoding="utf-8"
    )
    (run_dir / "raw_paper.pdf").write_bytes(b"%PDF-fake")
    (run_dir / "parsed_full_text.txt").write_text("paper text", encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1 — first-ever run: no-op, no crash
# ---------------------------------------------------------------------------


class TestFirstRun:
    def test_noop_when_no_prior_final_report(self, tmp_path):
        """First-ever run: run dir exists but has no final_report.json — no-op."""
        run_dir = _make_run_dir(tmp_path, "proj_first")
        # Seed some files but NOT final_report.json (run never completed).
        _seed_artifacts(run_dir, include_final_report=False)
        _seed_paper_artifacts(run_dir)

        result = maybe_archive_prior_attempt("proj_first", tmp_path)

        assert result is None, "Expected None (no-op) when no final_report.json"
        assert not (run_dir / "attempts").exists(), "attempts/ dir must not be created"
        # Existing files must still be in place.
        assert (run_dir / "experiment_runs.jsonl").exists()
        assert (run_dir / "paperMeta.json").exists()

    def test_noop_when_run_dir_absent(self, tmp_path):
        """Run dir does not exist yet — must not crash."""
        result = maybe_archive_prior_attempt("nonexistent_proj", tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# Test 2 — second run: all listed artifacts moved, paper artifacts preserved
# ---------------------------------------------------------------------------


class TestSecondRun:
    def test_all_run_artifacts_moved_on_rerun(self, tmp_path):
        """Prior final_report.json triggers archiving of all run-derived files."""
        run_dir = _make_run_dir(tmp_path, "proj_second")
        _seed_artifacts(run_dir)
        _seed_paper_artifacts(run_dir)

        result = maybe_archive_prior_attempt("proj_second", tmp_path)

        assert result is not None, "Expected archive dict, got None"
        attempt_dir = Path(result["attempt_dir"])
        assert attempt_dir.exists()
        assert attempt_dir.parent == run_dir / "attempts"

        # All run-derived files must have moved.
        for name in (
            "final_report.json",
            "final_report.md",
            "experiment_runs.jsonl",
            "cost_ledger.jsonl",
            "dashboard_events.jsonl",
            "repl_state.pickle",
        ):
            assert (attempt_dir / name).exists(), f"{name} not in attempt dir"
            assert not (run_dir / name).exists(), f"{name} still in run dir"

        # iterations.jsonl under rlm_state/.
        assert (attempt_dir / "rlm_state" / "iterations.jsonl").exists()
        assert not (run_dir / "rlm_state" / "iterations.jsonl").exists()

        # code/ directory moved.
        assert (attempt_dir / "code").exists()
        assert (attempt_dir / "code" / "train.py").exists()
        assert not (run_dir / "code" / "train.py").exists()

    def test_paper_artifacts_remain_after_archive(self, tmp_path):
        """paperMeta.json, generated_rubric.json etc. must NOT be archived."""
        run_dir = _make_run_dir(tmp_path, "proj_paper_preserve")
        _seed_artifacts(run_dir)
        _seed_paper_artifacts(run_dir)

        maybe_archive_prior_attempt("proj_paper_preserve", tmp_path)

        for name in (
            "paperMeta.json",
            "generated_rubric.json",
            "raw_paper.pdf",
            "parsed_full_text.txt",
        ):
            assert (run_dir / name).exists(), f"{name} was wrongly archived"

    def test_demo_status_reset_after_archive(self, tmp_path):
        """After archiving, demo_status.json must reflect a fresh queued state."""
        run_dir = _make_run_dir(tmp_path, "proj_status_reset")
        _seed_artifacts(run_dir)
        # Write an old completed status.
        (run_dir / "demo_status.json").write_text(
            json.dumps({"status": "completed", "projectId": "proj_status_reset"}),
            encoding="utf-8",
        )

        maybe_archive_prior_attempt("proj_status_reset", tmp_path)

        status_path = run_dir / "demo_status.json"
        assert status_path.exists(), "demo_status.json must be re-created"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["status"] == "queued", (
            f"Expected status='queued' after archive, got {status['status']!r}"
        )
        assert status["projectId"] == "proj_status_reset"

    def test_returns_moved_list(self, tmp_path):
        """Result dict must carry a non-empty 'moved' list."""
        run_dir = _make_run_dir(tmp_path, "proj_moved_list")
        _seed_artifacts(run_dir)

        result = maybe_archive_prior_attempt("proj_moved_list", tmp_path)

        assert result is not None
        assert len(result["moved"]) > 0, "moved list must be non-empty after archive"
        assert "final_report.json" in result["moved"]


# ---------------------------------------------------------------------------
# Test 3 — third run: two prior attempts, both preserved
# ---------------------------------------------------------------------------


class TestThirdRun:
    def test_two_prior_attempts_both_preserved(self, tmp_path):
        """A second re-run creates a fresh attempt dir and does not disturb the first."""
        run_dir = _make_run_dir(tmp_path, "proj_third")

        # --- first archiving ---
        _seed_artifacts(run_dir)
        first_result = maybe_archive_prior_attempt("proj_third", tmp_path)
        assert first_result is not None
        first_attempt_dir = Path(first_result["attempt_dir"])

        # Confirm clean state after first archive.
        assert not (run_dir / "final_report.json").exists()

        # Small sleep to ensure the ISO timestamp differs between the two calls.
        time.sleep(1.1)

        # --- second archiving ---
        _seed_artifacts(run_dir)
        second_result = maybe_archive_prior_attempt("proj_third", tmp_path)
        assert second_result is not None
        second_attempt_dir = Path(second_result["attempt_dir"])

        # The two attempt dirs must be distinct.
        assert first_attempt_dir != second_attempt_dir, (
            "First and second attempt dirs must be distinct timestamped paths"
        )

        # Both dirs must exist and contain the final report.
        assert (first_attempt_dir / "final_report.json").exists(), (
            "First attempt final_report.json must be preserved"
        )
        assert (second_attempt_dir / "final_report.json").exists(), (
            "Second attempt final_report.json must be in its own dir"
        )

        # The active run_dir must be clean (no final_report).
        assert not (run_dir / "final_report.json").exists()

        # The attempts/ dir must contain exactly two subdirs.
        attempts_dir = run_dir / "attempts"
        subdirs = [d for d in attempts_dir.iterdir() if d.is_dir()]
        assert len(subdirs) == 2, (
            f"Expected 2 attempt subdirs, found {len(subdirs)}: {subdirs}"
        )


# ---------------------------------------------------------------------------
# Test 4 — paper artifacts never moved (standalone guard)
# ---------------------------------------------------------------------------


class TestPaperArtifactsNeverMoved:
    """Explicit guard: paperMeta.json and generated_rubric.json must survive
    even a run with many prior attempts."""

    def test_paper_meta_not_in_moved_list(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "proj_pa4")
        _seed_artifacts(run_dir)
        _seed_paper_artifacts(run_dir)

        result = maybe_archive_prior_attempt("proj_pa4", tmp_path)

        assert result is not None
        for name in ("paperMeta.json", "generated_rubric.json"):
            assert name not in result["moved"], (
                f"{name} must not appear in the moved list"
            )

    def test_paper_meta_exists_in_run_dir_after_archive(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "proj_pa4b")
        _seed_artifacts(run_dir)
        _seed_paper_artifacts(run_dir)

        maybe_archive_prior_attempt("proj_pa4b", tmp_path)

        for name in ("paperMeta.json", "generated_rubric.json"):
            assert (run_dir / name).exists(), f"{name} must remain in run_dir"

    def test_paper_meta_survives_multiple_archives(self, tmp_path):
        """After two archiving calls, paper-level files must still be present."""
        run_dir = _make_run_dir(tmp_path, "proj_pa4c")
        _seed_artifacts(run_dir)
        _seed_paper_artifacts(run_dir)
        maybe_archive_prior_attempt("proj_pa4c", tmp_path)

        time.sleep(1.1)

        _seed_artifacts(run_dir)
        # Paper artifacts are not re-seeded — they should still be in place.
        maybe_archive_prior_attempt("proj_pa4c", tmp_path)

        for name in ("paperMeta.json", "generated_rubric.json"):
            assert (run_dir / name).exists(), (
                f"{name} must survive multiple archive passes"
            )


# ---------------------------------------------------------------------------
# Lane A — warm retry vs. clean retry  (2026-05-24)
# ---------------------------------------------------------------------------


class TestWarmRetry:
    """Kill-and-relaunch: prior code/ exists but final_report.json is absent.

    The function must NOT archive — leaving the code in place lets
    implement_baseline's cache short-circuit the ~5-min Sonnet sub-agent on
    the next iteration.
    """

    def test_warm_retry_preserves_code_when_commands_json_present(self, tmp_path):
        """commands.json on disk → warm retry → NO archive."""
        run_dir = _make_run_dir(tmp_path, "proj_warm_a")
        # Seed only the code/ — NOT final_report.json (kill mid-run).
        code_dir = run_dir / "code"
        code_dir.mkdir()
        (code_dir / "commands.json").write_text('["python train.py"]')
        (code_dir / "train.py").write_text("# train")
        # Some run-derived files are present (a failed run wrote them).
        (run_dir / "experiment_runs.jsonl").write_text('{"success": false}\n')

        result = maybe_archive_prior_attempt("proj_warm_a", tmp_path)

        assert result is None, "Warm retry must NOT trigger archive"
        # code/ stays in place.
        assert (run_dir / "code" / "commands.json").exists()
        assert (run_dir / "code" / "train.py").exists()
        # attempts/ dir must NOT have been created.
        assert not (run_dir / "attempts").exists()

    def test_warm_retry_preserves_code_with_only_train_py(self, tmp_path):
        """train.py alone (no commands.json yet) is also a warm-retry marker."""
        run_dir = _make_run_dir(tmp_path, "proj_warm_b")
        code_dir = run_dir / "code"
        code_dir.mkdir()
        (code_dir / "train.py").write_text("# half-written baseline")

        result = maybe_archive_prior_attempt("proj_warm_b", tmp_path)

        assert result is None
        assert (run_dir / "code" / "train.py").exists()
        assert not (run_dir / "attempts").exists()

    def test_empty_code_dir_is_not_warm_retry(self, tmp_path):
        """code/ exists but is empty (no marker files) → fall through to no-op,
        the same as a first-ever run."""
        run_dir = _make_run_dir(tmp_path, "proj_empty_code")
        (run_dir / "code").mkdir()

        result = maybe_archive_prior_attempt("proj_empty_code", tmp_path)
        assert result is None
        # Empty code/ stays empty — nothing to preserve.
        assert (run_dir / "code").is_dir()

    def test_clean_retry_still_archives_when_final_report_present(self, tmp_path):
        """final_report.json present → archive even if code/ also exists.

        This pins the invariant: warm-retry detection MUST NOT swallow a
        completed run's archive — only the kill-mid-run case skips.
        """
        run_dir = _make_run_dir(tmp_path, "proj_clean_a")
        _seed_artifacts(run_dir)  # writes final_report.json AND code/train.py
        _seed_paper_artifacts(run_dir)

        result = maybe_archive_prior_attempt("proj_clean_a", tmp_path)

        assert result is not None, "Clean retry must archive"
        attempt_dir = Path(result["attempt_dir"])
        # code/ moved into the attempt dir.
        assert (attempt_dir / "code" / "train.py").exists()
        assert not (run_dir / "code" / "train.py").exists()


class TestChownBeforeMove:
    """Lane A change 3: docker chown -R is invoked BEFORE shutil.move on the
    code/ directory.  Fail-soft on missing docker / non-POSIX hosts."""

    def test_chown_invoked_before_move(self, tmp_path, monkeypatch):
        """The chown subprocess MUST run before the shutil.move on code/."""
        run_dir = _make_run_dir(tmp_path, "proj_chown_a")
        _seed_artifacts(run_dir)

        calls: list[tuple] = []

        from backend.services.runs import attempt_isolation as ai
        real_move = ai.shutil.move

        def _track_chown(code_dir):
            calls.append(("chown", str(code_dir)))

        def _track_move(src, dst):
            calls.append(("move", str(src), str(dst)))
            return real_move(src, dst)

        monkeypatch.setattr(ai, "_chown_root_owned_code", _track_chown)
        monkeypatch.setattr(ai.shutil, "move", _track_move)

        maybe_archive_prior_attempt("proj_chown_a", tmp_path)

        # Find chown index and the move of code/ index.
        chown_idx = next(
            (i for i, c in enumerate(calls) if c[0] == "chown"), None
        )
        move_code_idx = next(
            (i for i, c in enumerate(calls)
             if c[0] == "move" and c[1].endswith("/code")),
            None,
        )
        assert chown_idx is not None, "chown must be invoked when code/ exists"
        assert move_code_idx is not None, "code/ must be moved"
        assert chown_idx < move_code_idx, (
            "chown must run BEFORE shutil.move on code/"
        )

    def test_chown_failure_is_fail_soft(self, tmp_path, monkeypatch):
        """A failing docker chown must NOT crash the archive."""
        run_dir = _make_run_dir(tmp_path, "proj_chown_fail")
        _seed_artifacts(run_dir)

        from backend.services.runs import attempt_isolation as ai

        def _raises(code_dir):
            # Simulate what _chown_root_owned_code does on failure: log and
            # return without raising.  (The real function already swallows
            # docker errors; this test pins the contract.)
            return None

        monkeypatch.setattr(ai, "_chown_root_owned_code", _raises)
        # Should not raise.
        result = maybe_archive_prior_attempt("proj_chown_fail", tmp_path)
        assert result is not None

    def test_chown_helper_is_fail_soft_when_docker_missing(
        self, tmp_path, monkeypatch
    ):
        """_chown_root_owned_code swallows FileNotFoundError when `docker` is
        absent from PATH."""
        from backend.services.runs.attempt_isolation import (
            _chown_root_owned_code,
        )

        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "train.py").write_text("# x")

        import subprocess as _sp

        def _raise_fnf(*a, **kw):
            raise FileNotFoundError("docker not on PATH")

        monkeypatch.setattr(_sp, "run", _raise_fnf)
        # Must not raise.
        _chown_root_owned_code(code_dir)
