"""tests/rlm/test_attempt_driver.py — AttemptDriver seam (spec §7; Codex F6/F15/F5).

Covers: force-quarantine (Codex F6, the warm-retry hole), LiveCliDriver's
launch/await_result/abort lifecycle (spawn conventions, argv/env builders,
timeout + abort escalation), and the campaign seed-marker seam (Codex F5
delivery half — best_attempt.py's marker-first read side, driven by the
driver's marker-write side).

All hermetic: tmp_path-only, no real subprocess, no real sleep (clock/sleep
are injected fakes throughout).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import types
from pathlib import Path

import pytest

from backend.agents.rlm import attempt_driver as ad
from backend.agents.rlm import best_attempt
from backend.agents.rlm.attempt_driver import (
    AttemptHandle,
    DriverError,
    LiveCliDriver,
    build_attempt_env,
    build_reproduce_argv,
)
from backend.services.runs.attempt_isolation import (
    _is_warm_retry,
    force_archive_incomplete,
    maybe_archive_prior_attempt,
)


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


def _directives(**overrides):
    base = dict(
        attempt_n=1,
        project_id="prj_test",
        paper_ref="2605.15155",
        run_spec_path=None,
        enforcement={},
        seed_pointer=None,
        seed_lineage=None,
        target_floor=None,
        scope_spec=None,
        extra_guidance="",
        envelope=None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


class FakeClock:
    """Injectable monotonic-ish clock; ``advance`` moves it forward."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeSleep:
    """Records every sleep call; optionally drives a FakeClock forward so a
    poll loop makes progress without ever really sleeping."""

    def __init__(self, clock: FakeClock | None = None, step: float = 0.0) -> None:
        self.calls: list[float] = []
        self._clock = clock
        self._step = step

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self._clock is not None:
            self._clock.advance(self._step or seconds)


class FakePopen:
    def __init__(self, pid: int = 987654) -> None:
        self.pid = pid


def _fake_popen_factory(calls: list, pid: int = 987654):
    def _popen(argv, **kwargs):
        calls.append({"argv": argv, "kwargs": kwargs})
        return FakePopen(pid=pid)
    return _popen


# ---------------------------------------------------------------------------
# force_archive_incomplete (Codex F6)
# ---------------------------------------------------------------------------


def test_force_quarantine_archives_residue_without_warm_retry(tmp_path):
    """code/train.py + no final_report.json is exactly the warm-retry shape
    — force_archive_incomplete must archive it anyway, unconditionally."""
    run_dir = tmp_path / "proj_a"
    code_dir = run_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "train.py").write_text("# half-written baseline")

    result = force_archive_incomplete("proj_a", tmp_path, reason="campaign_pre_launch")

    assert result is not None
    assert not (run_dir / "code").exists()

    attempts = list((run_dir / "attempts").iterdir())
    assert len(attempts) == 1
    incomplete_dir = attempts[0]
    assert incomplete_dir.name.endswith("_incomplete")
    assert (incomplete_dir / "code" / "train.py").exists()
    assert (incomplete_dir / "archive_reason.json").exists()
    assert json.loads((incomplete_dir / "archive_reason.json").read_text()) == {
        "reason": "campaign_pre_launch"
    }
    assert result["reason"] == "campaign_pre_launch"
    assert result["attempt_dir"] == str(incomplete_dir)

    # Warm-retry heuristic is now structurally unreachable: no code/ left.
    assert _is_warm_retry(run_dir) is False


def test_force_archive_incomplete_completed_run_uses_non_incomplete_suffix(tmp_path):
    """final_report.json present -> the usual (non-suffixed) attempts/<ts>/
    naming, same as maybe_archive_prior_attempt uses for a completed run."""
    run_dir = tmp_path / "proj_complete"
    code_dir = run_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "train.py").write_text("# done")
    (run_dir / "final_report.json").write_text(json.dumps({"verdict": "reproduced"}))

    result = force_archive_incomplete("proj_complete", tmp_path, reason="x")

    assert result is not None
    attempt_dir_name = list((run_dir / "attempts").iterdir())[0].name
    assert not attempt_dir_name.endswith("_incomplete")


def test_maybe_archive_prior_attempt_unchanged_warm_retry_still_skips(tmp_path):
    """Existing heuristic path is byte-identical: the SAME shaped run dir
    given to maybe_archive_prior_attempt (not force_archive_incomplete)
    still skips archiving and keeps code/ in place."""
    run_dir = tmp_path / "proj_b"
    code_dir = run_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "commands.json").write_text('["python train.py"]')
    (code_dir / "train.py").write_text("# train")
    (run_dir / "experiment_runs.jsonl").write_text('{"success": false}\n')

    result = maybe_archive_prior_attempt("proj_b", tmp_path)

    assert result is None
    assert (run_dir / "code" / "commands.json").exists()
    assert (run_dir / "code" / "train.py").exists()
    assert not (run_dir / "attempts").exists()


def test_force_archive_none_when_no_residue(tmp_path):
    run_dir = tmp_path / "proj_c"
    run_dir.mkdir()
    (run_dir / "paperMeta.json").write_text('{"id": "x"}')

    result = force_archive_incomplete("proj_c", tmp_path, reason="campaign_pre_launch")

    assert result is None
    assert not (run_dir / "attempts").exists()
    assert (run_dir / "paperMeta.json").exists()


def test_force_archive_none_when_run_dir_absent(tmp_path):
    assert force_archive_incomplete("proj_missing", tmp_path, reason="x") is None


def test_force_archive_never_touches_campaign_or_paper_artifacts(tmp_path):
    run_dir = tmp_path / "proj_d"
    code_dir = run_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "train.py").write_text("# x")
    (run_dir / "paperMeta.json").write_text('{"id": "x"}')
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / "campaign.json").write_text("{}")

    force_archive_incomplete("proj_d", tmp_path, reason="x")

    assert (run_dir / "paperMeta.json").exists()
    assert (campaign_dir / "campaign.json").exists()


# ---------------------------------------------------------------------------
# LiveCliDriver.launch — force-quarantine ordering, argv, env
# ---------------------------------------------------------------------------


def test_launch_quarantines_then_spawns(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    repo_root = tmp_path / "repo"
    runs_root.mkdir()
    repo_root.mkdir()

    project_id = "prj_launch"
    run_dir = runs_root / project_id
    code_dir = run_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "train.py").write_text("# stale residue")

    order: list[str] = []
    real_force_archive = ad.force_archive_incomplete

    def _tracking_force_archive(project_id_arg, root, *, reason):
        order.append("archive")
        return real_force_archive(project_id_arg, root, reason=reason)

    monkeypatch.setattr(ad, "force_archive_incomplete", _tracking_force_archive)

    def _fake_popen(argv, **kwargs):
        order.append("popen")
        return FakePopen()

    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root, popen=_fake_popen,
        clock=FakeClock(), sleep=FakeSleep(),
    )

    directives = _directives(project_id=project_id, enforcement={"vm_ceiling_s": 100.0})
    handle = driver.launch(directives)

    assert order == ["archive", "popen"]
    assert not (run_dir / "code").exists()  # quarantined away before spawn
    assert handle.pid == 987654
    assert handle.run_dir == str(run_dir)
    assert handle.driver == "live"
    assert handle.attempt_n == 1
    # await_deadline travels ON the handle (not driver instance state) —
    # launched_at + vm_ceiling_s + the outer margin.
    assert handle.await_deadline == handle.launched_at + 100.0 + ad._AWAIT_TIMEOUT_MARGIN_S


def test_launch_calls_quarantine_unconditionally_but_noop_on_clean_dir(tmp_path, monkeypatch):
    """Item-0 fix: force_archive_incomplete is now called on EVERY launch
    (no narrow code/-or-three-markers pre-check) -- it already no-ops
    safely via its own _has_attempt_residue on a genuinely clean dir."""
    runs_root = tmp_path / "runs"
    repo_root = tmp_path / "repo"
    runs_root.mkdir()
    repo_root.mkdir()

    archive_calls: list[str] = []
    monkeypatch.setattr(
        ad, "force_archive_incomplete",
        lambda *a, **kw: archive_calls.append("called"),
    )

    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root,
        popen=_fake_popen_factory([]), clock=FakeClock(), sleep=FakeSleep(),
    )
    driver.launch(_directives(project_id="prj_clean"))

    assert archive_calls == ["called"]


def test_launch_quarantine_now_fires_for_residue_the_old_narrow_check_missed(tmp_path):
    """Before the item-0 fix, LiveCliDriver.launch only force-quarantined
    when code/ or one of the three top-level markers (final_report.json /
    experiment_runs.jsonl / dashboard_events.jsonl) existed -- missing a
    lone rlm_state/gpu_escalation_state.json. The unconditional call now
    catches it, since force_archive_incomplete's own _has_attempt_residue
    check covers the FULL manifest, not just those four paths."""
    runs_root = tmp_path / "runs"
    repo_root = tmp_path / "repo"
    runs_root.mkdir()
    repo_root.mkdir()

    project_id = "prj_narrow_miss"
    run_dir = runs_root / project_id
    (run_dir / "rlm_state").mkdir(parents=True)
    (run_dir / "rlm_state" / "gpu_escalation_state.json").write_text("{}")

    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root,
        popen=_fake_popen_factory([]), clock=FakeClock(), sleep=FakeSleep(),
    )
    driver.launch(_directives(project_id=project_id))

    assert not (run_dir / "rlm_state" / "gpu_escalation_state.json").exists()
    attempts = list((run_dir / "attempts").iterdir())
    assert len(attempts) == 1
    assert (attempts[0] / "rlm_state" / "gpu_escalation_state.json").exists()


def test_argv_carries_enforcement_flags_and_scope():
    directives = _directives(
        attempt_n=3,
        project_id="prj_argv",
        paper_ref="2605.15155",
        run_spec_path="/tmp/campaign_run_spec.json",
        enforcement={
            "cli_args": [
                ["--max-usd", "5.0"],
                ["--max-wall-clock", "3600"],
                ["--max-run-gpu-usd", "8.0"],
            ],
        },
        scope_spec='{"models": ["Qwen3-1.7B"]}',
    )

    argv = build_reproduce_argv(directives, python_exe="/usr/bin/python3")

    assert argv == [
        "/usr/bin/python3", "-m", "backend.cli", "reproduce", "2605.15155",
        "--project-id", "prj_argv",
        "--run-spec", "/tmp/campaign_run_spec.json",
        "--max-usd", "5.0",
        "--max-wall-clock", "3600",
        "--max-run-gpu-usd", "8.0",
        "--scope-spec", '{"models": ["Qwen3-1.7B"]}',
    ]


def test_argv_omits_run_spec_and_scope_when_unset():
    directives = _directives(project_id="prj_bare", paper_ref="pdf/path.pdf")
    argv = build_reproduce_argv(directives, python_exe=sys.executable)
    assert argv == [
        sys.executable, "-m", "backend.cli", "reproduce", "pdf/path.pdf",
        "--project-id", "prj_bare",
    ]


def test_env_carries_gpu_cap_guidance_and_seed_flags(tmp_path, monkeypatch):
    """seed_pointer set -> marker written + SEED flag "1"; a subsequent
    launch with no seed_pointer -> marker deleted + SEED flag "0"."""
    monkeypatch.delenv("OPENRESEARCH_BASELINE_EXTRA_GUIDANCE", raising=False)
    monkeypatch.setenv("SOME_UNRELATED_VAR", "keep-me")

    runs_root = tmp_path / "runs"
    repo_root = tmp_path / "repo"
    runs_root.mkdir()
    repo_root.mkdir()

    calls: list[dict] = []
    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root,
        popen=_fake_popen_factory(calls), clock=FakeClock(), sleep=FakeSleep(),
    )

    project_id = "prj_env_seeded"
    run_dir = runs_root / project_id
    source_dir = tmp_path / "champion_code"
    source_dir.mkdir()
    (source_dir / "train.py").write_text("# champion")

    directives = _directives(
        project_id=project_id,
        seed_pointer=str(source_dir),
        seed_lineage="champion",
        target_floor=0.7,
        extra_guidance="Use Qwen3-1.7B only.",
        enforcement={"env": {"OPENRESEARCH_MAX_GPU_USD_PER_HOUR": "10.0"}},
    )
    driver.launch(directives)

    marker_path = run_dir / "campaign" / "seed_staging.json"
    assert marker_path.exists()
    marker = json.loads(marker_path.read_text())
    assert marker == {
        "attempt_n": 1,
        "source_code_dir": str(source_dir),
        "target_floor": 0.7,
        "lineage": "champion",
    }

    kwargs = calls[-1]["kwargs"]
    env = kwargs["env"]
    assert env["OPENRESEARCH_SEED_BEST_ATTEMPT"] == "1"
    assert env["OPENRESEARCH_TARGET_BEST_FLOOR"] == "1"
    assert env["OPENRESEARCH_BASELINE_EXTRA_GUIDANCE"] == "Use Qwen3-1.7B only."
    assert env["OPENRESEARCH_MAX_GPU_USD_PER_HOUR"] == "10.0"
    assert env["SOME_UNRELATED_VAR"] == "keep-me"  # base_env (os.environ) preserved
    assert kwargs["cwd"] == str(repo_root)
    assert kwargs["stderr"] == subprocess.STDOUT
    assert kwargs["start_new_session"] is True

    # --- second attempt, same project: no seed_pointer -> marker deleted ---
    driver.launch(_directives(project_id=project_id, attempt_n=2))

    assert not marker_path.exists()
    env2 = calls[-1]["kwargs"]["env"]
    assert env2["OPENRESEARCH_SEED_BEST_ATTEMPT"] == "0"
    assert env2["OPENRESEARCH_TARGET_BEST_FLOOR"] == "0"
    assert "OPENRESEARCH_BASELINE_EXTRA_GUIDANCE" not in env2


def test_env_keys_all_pass_run_spec_contract():
    from backend.agents.rlm.run_spec_contract import run_spec_key_applies

    for key in (
        "OPENRESEARCH_BASELINE_EXTRA_GUIDANCE",
        "OPENRESEARCH_SEED_BEST_ATTEMPT",
        "OPENRESEARCH_TARGET_BEST_FLOOR",
    ):
        assert run_spec_key_applies(key)

    directives = _directives(extra_guidance="x", seed_pointer="/tmp/y", target_floor=0.5)
    env = build_attempt_env(directives, {})
    assert env["OPENRESEARCH_BASELINE_EXTRA_GUIDANCE"] == "x"
    assert env["OPENRESEARCH_SEED_BEST_ATTEMPT"] == "1"
    assert env["OPENRESEARCH_TARGET_BEST_FLOOR"] == "1"


def test_build_attempt_env_defends_against_a_renamed_key(monkeypatch):
    """If run_spec_key_applies ever stopped accepting one of the three
    per-attempt keys (a rename typo), build_attempt_env must refuse to
    launch silently — it raises DriverError instead."""
    monkeypatch.setattr(ad, "run_spec_key_applies", lambda key: False)
    with pytest.raises(DriverError):
        build_attempt_env(_directives(), {})


# ---------------------------------------------------------------------------
# Campaign seed-marker seam (Codex F5 delivery half)
# ---------------------------------------------------------------------------


def test_stale_marker_with_flags_unset_is_inert(tmp_path, monkeypatch):
    """FIX 1 (off-state gating): a stale campaign/seed_staging.json left
    over from an earlier campaign must NOT influence a MANUAL run with both
    flags unset — "no campaign env set => byte-identical behavior". Both
    seed_reference_code and floored_target must return their no-op values
    without ever staging code/_best_attempt, even though a perfectly valid,
    seedable marker sits on disk."""
    project_dir = tmp_path / "proj_stale_marker"
    project_dir.mkdir()

    src_code = project_dir / "attempts" / "20260701T000000-000000-aaaaaa" / "code"
    src_code.mkdir(parents=True)
    (src_code / "train.py").write_text("# stale campaign champion")
    marker_path = project_dir / best_attempt.CAMPAIGN_SEED_MARKER
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps({
        "attempt_n": 5, "source_code_dir": str(src_code),
        "target_floor": 0.8, "lineage": "stale",
    }))

    monkeypatch.delenv(best_attempt.ENV_SEED_FLAG, raising=False)
    monkeypatch.delenv(best_attempt.ENV_TARGET_FLOOR_FLAG, raising=False)

    assert best_attempt.seed_reference_code(project_dir) is None
    assert not (project_dir / "code" / best_attempt.REFERENCE_DIR_NAME).exists()
    assert best_attempt.floored_target(project_dir, 0.3) == 0.3
    assert best_attempt.floored_target(project_dir, None) is None


def test_seed_marker_steers_seed_reference_code_ignoring_scores(tmp_path, monkeypatch):
    """attempts/A scores 0.9 (guard-tripped junk); attempts/B scores 0.2.
    With a campaign marker pointing at B, B is staged despite the lower
    score. With no marker (flag on), the legacy best-by-score scan wins
    and stages A."""
    project_dir = tmp_path / "proj_seed"
    project_dir.mkdir()

    a_dir = project_dir / "attempts" / "20260701T000000-000000-aaaaaa"
    a_code = a_dir / "code"
    a_code.mkdir(parents=True)
    (a_code / "train.py").write_text("# A: guard-tripped junk, but scored 0.9")
    (a_dir / "final_report.json").write_text(
        json.dumps({"rubric": {"overall_score": 0.9, "leaf_scores": []}})
    )

    b_dir = project_dir / "attempts" / "20260701T010000-000000-bbbbbb"
    b_code = b_dir / "code"
    b_code.mkdir(parents=True)
    (b_code / "train.py").write_text("# B: guard-clean, campaign-selected")
    (b_dir / "final_report.json").write_text(
        json.dumps({"rubric": {"overall_score": 0.2, "leaf_scores": []}})
    )

    # --- no marker, flag on: existing best-by-score behavior (A wins) ---
    monkeypatch.setenv("OPENRESEARCH_SEED_BEST_ATTEMPT", "1")
    rel = best_attempt.seed_reference_code(project_dir)
    assert rel == f"code/{best_attempt.REFERENCE_DIR_NAME}"
    staged = (project_dir / "code" / best_attempt.REFERENCE_DIR_NAME / "train.py").read_text()
    assert "A: guard-tripped" in staged

    # --- marker present: campaign selects B despite its lower score ---
    marker_path = project_dir / best_attempt.CAMPAIGN_SEED_MARKER
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps({
        "attempt_n": 3,
        "source_code_dir": str(b_code),
        "target_floor": None,
        "lineage": "runner_up",
    }))

    rel2 = best_attempt.seed_reference_code(project_dir)
    assert rel2 == f"code/{best_attempt.REFERENCE_DIR_NAME}"
    staged2 = (project_dir / "code" / best_attempt.REFERENCE_DIR_NAME / "train.py").read_text()
    assert "B: guard-clean" in staged2


def test_seed_marker_source_missing_fails_closed_no_fallback(tmp_path, monkeypatch):
    """A marker whose source_code_dir does not exist returns None — it must
    NOT fall back to the score-ranked scan even when a scoreable attempt
    exists on disk and the flag is on."""
    project_dir = tmp_path / "proj_seed_missing_src"
    project_dir.mkdir()
    a_dir = project_dir / "attempts" / "20260701T000000-000000-aaaaaa"
    (a_dir / "code").mkdir(parents=True)
    (a_dir / "code" / "train.py").write_text("# would-be fallback")
    (a_dir / "final_report.json").write_text(
        json.dumps({"rubric": {"overall_score": 0.9, "leaf_scores": []}})
    )

    monkeypatch.setenv("OPENRESEARCH_SEED_BEST_ATTEMPT", "1")
    marker_path = project_dir / best_attempt.CAMPAIGN_SEED_MARKER
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps({
        "attempt_n": 2,
        "source_code_dir": str(project_dir / "nonexistent_source"),
        "target_floor": None,
        "lineage": "champion",
    }))

    assert best_attempt.seed_reference_code(project_dir) is None
    assert not (project_dir / "code" / best_attempt.REFERENCE_DIR_NAME).exists()


def test_floored_target_reads_marker_floor_without_scanning(tmp_path, monkeypatch):
    project_dir = tmp_path / "proj_floor"
    project_dir.mkdir()

    # A high-scoring prior attempt exists on disk (a scan would find 0.95).
    a_dir = project_dir / "attempts" / "20260701T000000-000000-aaaaaa"
    a_dir.mkdir(parents=True)
    (a_dir / "final_report.json").write_text(
        json.dumps({"rubric": {"overall_score": 0.95, "leaf_scores": []}})
    )

    marker_path = project_dir / best_attempt.CAMPAIGN_SEED_MARKER
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps({
        "attempt_n": 2, "source_code_dir": str(a_dir / "code"),
        "target_floor": 0.4, "lineage": "champion",
    }))

    # Flag ON — this is what build_attempt_env always sets whenever a
    # campaign's target_floor is not None (the real-world precondition for
    # the marker to matter at all; see the off-state gating test below for
    # the flag-OFF case).
    monkeypatch.setenv("OPENRESEARCH_TARGET_BEST_FLOOR", "1")

    # Marker floor (0.4) applies without ever scanning attempts/ for the
    # 0.95 score — the marker, not a raw score-ranked scan, owns the floor
    # once a campaign is driving.
    assert best_attempt.floored_target(project_dir, 0.3) == 0.4
    assert best_attempt.floored_target(project_dir, 0.5) == 0.5  # higher target kept
    assert best_attempt.floored_target(project_dir, None) == 0.4


def test_floored_target_marker_without_numeric_floor_leaves_target_unchanged(tmp_path, monkeypatch):
    project_dir = tmp_path / "proj_floor_none"
    project_dir.mkdir()
    marker_path = project_dir / best_attempt.CAMPAIGN_SEED_MARKER
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(json.dumps({
        "attempt_n": 1, "source_code_dir": str(project_dir), "target_floor": None,
        "lineage": "fresh",
    }))
    monkeypatch.setenv("OPENRESEARCH_TARGET_BEST_FLOOR", "1")  # flag ON
    assert best_attempt.floored_target(project_dir, 0.6) == 0.6


def test_fresh_lineage_with_floor_stages_marker_and_floored_target_skips_scan(
    tmp_path, monkeypatch
):
    """FIX 2 (F5 hole): a fresh-lineage attempt (no seed_pointer) that still
    carries a campaign target_floor must stage a marker with
    source_code_dir=None instead of deleting it outright — otherwise
    floored_target (flag on, per build_attempt_env whenever target_floor is
    set) has no marker to read and falls through to the score-ranked
    attempts/ scan, exactly what Codex F5 forbids once a campaign is
    driving. A decoy attempt with a much higher score sits on disk and must
    NOT win."""
    runs_root = tmp_path / "runs"
    repo_root = tmp_path / "repo"
    runs_root.mkdir()
    repo_root.mkdir()

    project_id = "prj_fresh_with_floor"
    run_dir = runs_root / project_id

    # Decoy: the legacy score-ranked scan would pick this up (0.99) if
    # floored_target fell through to it — it must not win.
    decoy_dir = run_dir / "attempts" / "20260701T000000-000000-decoy"
    decoy_dir.mkdir(parents=True)
    (decoy_dir / "final_report.json").write_text(
        json.dumps({"rubric": {"overall_score": 0.99, "leaf_scores": []}})
    )

    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root,
        popen=_fake_popen_factory([]), clock=FakeClock(), sleep=FakeSleep(),
    )
    directives = _directives(
        project_id=project_id, seed_pointer=None, target_floor=0.45,
    )
    driver.launch(directives)

    marker_path = run_dir / "campaign" / "seed_staging.json"
    assert marker_path.exists()  # NOT deleted, despite seed_pointer=None
    marker = json.loads(marker_path.read_text())
    assert marker["source_code_dir"] is None
    assert marker["target_floor"] == 0.45

    # build_attempt_env sets TARGET_BEST_FLOOR="1" whenever target_floor is
    # not None (mirrored here, this in-process test never spawns the child).
    monkeypatch.setenv("OPENRESEARCH_TARGET_BEST_FLOOR", "1")

    assert best_attempt.floored_target(run_dir, 0.2) == 0.45  # floor applied
    assert best_attempt.floored_target(run_dir, 0.6) == 0.6  # higher target kept
    # Never the decoy's 0.99 — the marker short-circuits the scan entirely.

    # seed_pointer=None -> build_attempt_env sets SEED_BEST_ATTEMPT="0", so
    # seed_reference_code (flag off) stages nothing from either the marker
    # (source_code_dir=None) or the decoy.
    monkeypatch.delenv("OPENRESEARCH_SEED_BEST_ATTEMPT", raising=False)
    assert best_attempt.seed_reference_code(run_dir) is None
    assert not (run_dir / "code" / best_attempt.REFERENCE_DIR_NAME).exists()


def test_driver_written_marker_is_consumed_by_best_attempt(tmp_path, monkeypatch):
    """End-to-end: the marker LiveCliDriver.launch() writes is exactly what
    best_attempt.seed_reference_code / floored_target read (Codex F5, both
    halves of the seam wired together). seed_reference_code/floored_target
    always run INSIDE the spawned child, whose actual os.environ carries the
    two flags build_attempt_env computed at launch — mirrored here via
    monkeypatch (flag-first gating, FIX 1) rather than through the child's
    real env dict (which this in-process test never spawns)."""
    runs_root = tmp_path / "runs"
    repo_root = tmp_path / "repo"
    runs_root.mkdir()
    repo_root.mkdir()

    champion_src = tmp_path / "champion_source"
    champion_src.mkdir()
    (champion_src / "train.py").write_text("# the campaign's chosen champion")

    project_id = "prj_end_to_end_seed"
    directives = _directives(
        project_id=project_id, seed_pointer=str(champion_src),
        seed_lineage="champion", target_floor=0.55,
    )
    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root,
        popen=_fake_popen_factory([]), clock=FakeClock(), sleep=FakeSleep(),
    )
    driver.launch(directives)

    monkeypatch.setenv("OPENRESEARCH_SEED_BEST_ATTEMPT", "1")
    monkeypatch.setenv("OPENRESEARCH_TARGET_BEST_FLOOR", "1")

    run_dir = runs_root / project_id
    rel = best_attempt.seed_reference_code(run_dir)
    assert rel == f"code/{best_attempt.REFERENCE_DIR_NAME}"
    staged = (run_dir / "code" / best_attempt.REFERENCE_DIR_NAME / "train.py").read_text()
    assert "the campaign's chosen champion" in staged
    assert best_attempt.floored_target(run_dir, 0.1) == 0.55


# ---------------------------------------------------------------------------
# Seed-marker ordering vs. the pre-launch archive (the cold-start regression)
# ---------------------------------------------------------------------------


def _attempt_with_live_code(runs_root, project_id: str, content: str):
    """A run dir shaped exactly like a just-assessed attempt: a working code/
    tree plus the final_report.json that makes it a COMPLETED prior attempt."""
    run_dir = runs_root / project_id
    live_code = run_dir / "code"
    live_code.mkdir(parents=True)
    (live_code / "train.py").write_text(content)
    (run_dir / "final_report.json").write_text(
        json.dumps({"rubric": {"overall_score": 0.831, "leaf_scores": []}})
    )
    return run_dir, live_code


def test_launch_stages_seed_marker_at_archived_code_not_the_emptied_live_dir(tmp_path):
    """THE cross-attempt-learning regression.

    force_archive_incomplete MOVES run_dir/code into attempts/<ts>/, and the
    plan-time seed_pointer names that very same live run_dir/code (it is the
    latest assessed attempt's tree). Staging the marker WITHOUT following the
    archive's returned path left source_code_dir naming a directory the
    archive had just emptied -- seed_reference_code then failed closed and
    every campaign attempt cold-started, which is exactly the Adam regression
    the rail exists to prevent.
    """
    runs_root, repo_root = tmp_path / "runs", tmp_path / "repo"
    runs_root.mkdir()
    repo_root.mkdir()
    run_dir, live_code = _attempt_with_live_code(runs_root, "prj_seed_order", "# proven tree")

    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root,
        popen=_fake_popen_factory([]), clock=FakeClock(), sleep=FakeSleep(),
    )
    # Exactly what PLAN hands the driver for attempt 2: a pointer at the
    # latest assessed attempt's still-live code/.
    driver.launch(_directives(
        attempt_n=2, project_id="prj_seed_order",
        seed_pointer=str(live_code), seed_lineage="champion",
    ))

    marker = json.loads((run_dir / "campaign" / "seed_staging.json").read_text())
    src = Path(marker["source_code_dir"])

    assert not live_code.exists()          # the archive emptied it, as always
    assert src != live_code                # ...so the marker must NOT name it
    assert src.is_dir()                    # the marker's source really exists
    assert (src / "train.py").read_text() == "# proven tree"
    assert src.parent.parent.name == "attempts"  # it followed the archive


def test_launched_marker_actually_seeds_the_prior_attempts_code(tmp_path, monkeypatch):
    """The other half of the seam: the child's seed_reference_code succeeds
    from the marker the driver just staged. This is cross-attempt learning
    working end to end -- pre-fix it returned None on every attempt."""
    runs_root, repo_root = tmp_path / "runs", tmp_path / "repo"
    runs_root.mkdir()
    repo_root.mkdir()
    run_dir, live_code = _attempt_with_live_code(runs_root, "prj_seed_e2e", "# proven tree")

    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root,
        popen=_fake_popen_factory([]), clock=FakeClock(), sleep=FakeSleep(),
    )
    driver.launch(_directives(
        attempt_n=2, project_id="prj_seed_e2e",
        seed_pointer=str(live_code), seed_lineage="champion",
    ))

    # The flags build_attempt_env sets for the child (mirrored: this
    # in-process test never spawns it).
    monkeypatch.setenv("OPENRESEARCH_SEED_BEST_ATTEMPT", "1")

    rel = best_attempt.seed_reference_code(run_dir)
    assert rel == f"code/{best_attempt.REFERENCE_DIR_NAME}"
    seeded = run_dir / "code" / best_attempt.REFERENCE_DIR_NAME / "train.py"
    assert seeded.read_text() == "# proven tree"


def test_launch_keeps_a_genuinely_missing_seed_source_failing_closed(tmp_path, monkeypatch):
    """The remap FOLLOWS the archive; it never INVENTS a source. A pointer at
    a path that never existed (and that this archive did not move) is written
    through unchanged, so seed_reference_code still fails CLOSED -- never a
    silent fallback to the score-ranked scan (Codex F5)."""
    runs_root, repo_root = tmp_path / "runs", tmp_path / "repo"
    runs_root.mkdir()
    repo_root.mkdir()
    run_dir, _live = _attempt_with_live_code(runs_root, "prj_missing_src", "# decoy tree")

    # A high-scoring decoy the legacy scan WOULD stage if the marker path
    # silently fell back to it.
    decoy = run_dir / "attempts" / "20260701T000000-000000-decoyy"
    (decoy / "code").mkdir(parents=True)
    (decoy / "code" / "train.py").write_text("# decoy 0.99")
    (decoy / "final_report.json").write_text(
        json.dumps({"rubric": {"overall_score": 0.99, "leaf_scores": []}})
    )

    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root,
        popen=_fake_popen_factory([]), clock=FakeClock(), sleep=FakeSleep(),
    )
    vanished = tmp_path / "never_existed"
    driver.launch(_directives(
        attempt_n=2, project_id="prj_missing_src",
        seed_pointer=str(vanished), seed_lineage="champion",
    ))

    marker = json.loads((run_dir / "campaign" / "seed_staging.json").read_text())
    assert marker["source_code_dir"] == str(vanished)  # unchanged, not invented

    monkeypatch.setenv("OPENRESEARCH_SEED_BEST_ATTEMPT", "1")
    assert best_attempt.seed_reference_code(run_dir) is None
    assert not (run_dir / "code" / best_attempt.REFERENCE_DIR_NAME).exists()


def test_seed_source_holding_nothing_seedable_fails_closed(tmp_path, monkeypatch):
    """An empty seed is not a seed. A marker whose source exists but yields
    zero copied items must NOT leave behind a README-only code/_best_attempt/
    claiming to be "the COMPLETE working code" of a seed that never
    materialized."""
    project_dir = tmp_path / "prj_empty_seed"
    empty_src = project_dir / "attempts" / "20260701T000000-000000-empty1" / "code"
    empty_src.mkdir(parents=True)

    marker_path = project_dir / best_attempt.CAMPAIGN_SEED_MARKER
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps({
        "attempt_n": 2, "source_code_dir": str(empty_src),
        "target_floor": None, "lineage": "champion",
    }))

    monkeypatch.setenv("OPENRESEARCH_SEED_BEST_ATTEMPT", "1")
    assert best_attempt.seed_reference_code(project_dir) is None
    assert not (project_dir / "code" / best_attempt.REFERENCE_DIR_NAME).exists()


def test_attempt_code_index_keeps_every_prior_attempt_addressable(tmp_path):
    """Each launch's archive rotation records the outgoing attempt's code
    under its ATTEMPT NUMBER, so attempt 1 stays resolvable long after
    attempt 3 is live. Without it only the newest attempt is locatable and a
    non-latest champion can never be seeded."""
    runs_root, repo_root = tmp_path / "runs", tmp_path / "repo"
    runs_root.mkdir()
    repo_root.mkdir()
    project_id = "prj_index"
    run_dir = runs_root / project_id

    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root,
        popen=_fake_popen_factory([]), clock=FakeClock(), sleep=FakeSleep(),
    )

    for n in (1, 2, 3):
        driver.launch(_directives(attempt_n=n, project_id=project_id))
        # The child would now write its code/ tree; mirrored here.
        (run_dir / "code").mkdir(parents=True, exist_ok=True)
        (run_dir / "code" / "train.py").write_text(f"# attempt {n}")
        (run_dir / "final_report.json").write_text(
            json.dumps({"rubric": {"overall_score": 0.5, "leaf_scores": []}})
        )

    for n in (1, 2):
        resolved = ad.resolve_attempt_code_dir(run_dir, n)
        assert resolved is not None, f"attempt {n} unresolvable"
        assert (Path(resolved) / "train.py").read_text() == f"# attempt {n}"
        assert Path(resolved).parent.parent.name == "attempts"  # archived

    # Attempt 3 is the live one.
    assert ad.resolve_attempt_code_dir(run_dir, 3) == str(run_dir / "code")
    # An attempt that never ran has no pointer (degrades to a fresh lineage).
    assert ad.resolve_attempt_code_dir(run_dir, 9) is None


def test_resolve_seed_source_leaves_an_unrelated_live_pointer_alone(tmp_path):
    """A width child (<id>_w<k>) launches in its OWN run dir while the seed
    points at the campaign dir's code/ -- a tree this launch's archive never
    touched. It must be used verbatim, not remapped."""
    other_code = tmp_path / "runs" / "prj_top" / "code"
    other_code.mkdir(parents=True)
    child_live = tmp_path / "runs" / "prj_top_w1" / "code"

    resolved = ad.resolve_seed_source(
        str(other_code),
        live_code_dir=child_live,
        archived={"attempt_dir": str(tmp_path / "runs" / "prj_top_w1" / "attempts" / "x")},
    )
    assert resolved == str(other_code)


# ---------------------------------------------------------------------------
# await_result
# ---------------------------------------------------------------------------


def test_await_returns_on_terminal_demo_status(tmp_path):
    runs_root = tmp_path / "runs"
    repo_root = tmp_path / "repo"
    project_id = "prj_await_terminal"
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True)
    (run_dir / "demo_status.json").write_text(json.dumps({"status": "completed"}))
    (run_dir / "final_report.json").write_text(json.dumps({"verdict": "reproduced"}))

    clock = FakeClock()
    sleep = FakeSleep()
    driver = LiveCliDriver(runs_root=runs_root, repo_root=repo_root, clock=clock, sleep=sleep)
    handle = AttemptHandle(
        attempt_n=1, project_id=project_id, run_dir=str(run_dir), driver="live",
        pid=None, launched_at=clock(), lease_ref=None,
    )

    result = driver.await_result(handle)

    assert result.exit_condition == "completed"
    assert result.report_path == str(run_dir / "final_report.json")
    assert sleep.calls == []  # returns on the FIRST check, no sleep needed


def test_await_returns_report_none_when_report_missing(tmp_path):
    runs_root = tmp_path / "runs"
    repo_root = tmp_path / "repo"
    project_id = "prj_await_no_report"
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True)
    (run_dir / "demo_status.json").write_text(json.dumps({"status": "failed"}))

    clock = FakeClock()
    driver = LiveCliDriver(runs_root=runs_root, repo_root=repo_root, clock=clock, sleep=FakeSleep())
    handle = AttemptHandle(
        attempt_n=1, project_id=project_id, run_dir=str(run_dir), driver="live",
        pid=None, launched_at=clock(), lease_ref=None,
    )

    result = driver.await_result(handle)
    assert result.exit_condition == "completed"
    assert result.report_path is None


def test_await_already_dead_when_pid_gone_without_terminal_status(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    repo_root = tmp_path / "repo"
    project_id = "prj_await_dead"
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True)
    (run_dir / "demo_status.json").write_text(json.dumps({"status": "running"}))

    def _fake_kill(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(os, "kill", _fake_kill)

    clock = FakeClock()
    sleep = FakeSleep()
    driver = LiveCliDriver(runs_root=runs_root, repo_root=repo_root, clock=clock, sleep=sleep)
    handle = AttemptHandle(
        attempt_n=1, project_id=project_id, run_dir=str(run_dir), driver="live",
        pid=424242, launched_at=clock(), lease_ref=None,
    )

    result = driver.await_result(handle)

    assert result.exit_condition == "already_dead"
    assert result.report_path is None
    assert sleep.calls == []


def test_await_timeout_aborts(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    repo_root = tmp_path / "repo"
    project_id = "prj_await_timeout"
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True)
    (run_dir / "demo_status.json").write_text(json.dumps({"status": "running"}))

    monkeypatch.setattr(ad, "_pid_alive", lambda pid: True)  # never "already_dead"

    clock = FakeClock()
    sleep = FakeSleep(clock=clock, step=10_000.0)  # each sleep jumps the clock far
    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root, clock=clock, sleep=sleep,
        poll_interval_s=1.0,
    )

    abort_calls: list[str] = []
    monkeypatch.setattr(driver, "abort", lambda handle, *, reason: abort_calls.append(reason))

    handle = AttemptHandle(
        attempt_n=1, project_id=project_id, run_dir=str(run_dir), driver="live",
        pid=999999, launched_at=clock(), lease_ref=None,
        await_deadline=clock() + 5.0,  # small ceiling -> fast test
    )

    result = driver.await_result(handle)

    assert result.exit_condition == "await_timeout"
    assert abort_calls == ["await_timeout"]


def test_await_resumed_driver_trusts_handle_deadline_not_launched_at(tmp_path, monkeypatch):
    """FIX 3 (resume money bug): a resumed campaign process constructs a
    FRESH LiveCliDriver with no memory of the original launch. For a
    healthy multi-hour run, launched_at is by then hours in the past — if
    await_result re-derived the deadline from launched_at (the old
    self._deadlines-miss fallback), it would compute an already-past
    deadline and abort a perfectly healthy paid GPU run on re-attach. With
    the deadline persisted ON the handle (still comfortably in the future),
    no abort happens."""
    runs_root = tmp_path / "runs"
    repo_root = tmp_path / "repo"
    project_id = "prj_resume_reattach"
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True)
    (run_dir / "demo_status.json").write_text(json.dumps({"status": "running"}))

    monkeypatch.setattr(ad, "_pid_alive", lambda pid: True)

    clock = FakeClock(start=2_000_000.0)

    def _sleep(seconds):
        # The run finishes organically during the poll wait — proves the
        # loop never even reaches an abort branch.
        (run_dir / "demo_status.json").write_text(json.dumps({"status": "completed"}))
        (run_dir / "final_report.json").write_text(json.dumps({"verdict": "reproduced"}))
        clock.advance(seconds)

    # A brand-new driver instance, exactly as a resumed campaign process
    # would construct — no self._deadlines (deleted by FIX 3) to miss.
    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root, clock=clock, sleep=_sleep,
        poll_interval_s=1.0,
    )
    abort_calls: list[str] = []
    monkeypatch.setattr(driver, "abort", lambda handle, *, reason: abort_calls.append(reason))

    # launched_at is 4 hours before "now" (a healthy long-running attempt
    # re-attached well into its lifetime); await_deadline — persisted on the
    # handle at the ORIGINAL launch — is still comfortably in the future.
    handle = AttemptHandle(
        attempt_n=1, project_id=project_id, run_dir=str(run_dir), driver="live",
        pid=999999, launched_at=clock() - 4 * 3600.0, lease_ref=None,
        await_deadline=clock() + 3600.0,
    )

    result = driver.await_result(handle)

    assert abort_calls == []
    assert result.exit_condition == "completed"


def test_await_result_with_no_deadline_falls_back_to_clock_plus_margin(tmp_path, monkeypatch):
    """FIX 3: a handle reconstructed with no persisted await_deadline (the
    dataclass default, None) must get a full margin measured from NOW — the
    same stale-launched_at trap as above, but exercising the explicit
    fallback branch in await_result rather than a value set at launch()."""
    runs_root = tmp_path / "runs"
    repo_root = tmp_path / "repo"
    project_id = "prj_resume_no_deadline"
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True)
    (run_dir / "demo_status.json").write_text(json.dumps({"status": "running"}))

    monkeypatch.setattr(ad, "_pid_alive", lambda pid: True)

    clock = FakeClock(start=2_000_000.0)

    def _sleep(seconds):
        (run_dir / "demo_status.json").write_text(json.dumps({"status": "completed"}))
        clock.advance(seconds)

    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root, clock=clock, sleep=_sleep,
        poll_interval_s=1.0,
    )
    abort_calls: list[str] = []
    monkeypatch.setattr(driver, "abort", lambda handle, *, reason: abort_calls.append(reason))

    handle = AttemptHandle(
        attempt_n=1, project_id=project_id, run_dir=str(run_dir), driver="live",
        pid=999999, launched_at=clock() - 4 * 3600.0, lease_ref=None,
        # await_deadline intentionally omitted -> defaults to None.
    )
    assert handle.await_deadline is None

    result = driver.await_result(handle)

    assert abort_calls == []
    assert result.exit_condition == "completed"


# ---------------------------------------------------------------------------
# abort
# ---------------------------------------------------------------------------


def test_abort_escalates_and_patches_demo_status(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    repo_root = tmp_path / "repo"
    project_id = "prj_abort"
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True)
    (run_dir / "demo_status.json").write_text(
        json.dumps({"status": "running", "projectId": project_id})
    )

    kill_calls: list[int] = []
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: kill_calls.append(sig))
    # The process never dies on its own -> the grace period must expire and
    # escalate to SIGKILL.
    monkeypatch.setattr(ad, "_pid_alive", lambda pid: True)

    clock = FakeClock()
    sleep = FakeSleep(clock=clock, step=40.0)  # exceeds abort_grace_s=30 after one sleep
    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root, clock=clock, sleep=sleep,
        abort_grace_s=30.0,
    )
    handle = AttemptHandle(
        attempt_n=1, project_id=project_id, run_dir=str(run_dir), driver="live",
        pid=555555, launched_at=clock(), lease_ref=None,
    )

    driver.abort(handle, reason="manual_abort")

    assert kill_calls == [signal.SIGTERM, signal.SIGKILL]
    status = json.loads((run_dir / "demo_status.json").read_text())
    assert status["status"] == "killed"
    assert status["killReason"] == "manual_abort"
    assert status["projectId"] == project_id  # existing fields preserved


def test_abort_already_dead_process_group_skips_escalation(tmp_path, monkeypatch):
    """os.getpgid raising ProcessLookupError (already dead) must not raise
    and must still patch demo_status.json (best-effort cleanup)."""
    runs_root = tmp_path / "runs"
    repo_root = tmp_path / "repo"
    project_id = "prj_abort_dead"
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True)
    (run_dir / "demo_status.json").write_text(json.dumps({"status": "running"}))

    def _raise(pid):
        raise ProcessLookupError()

    monkeypatch.setattr(os, "getpgid", _raise)
    killpg_calls: list[int] = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append(sig))

    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root, clock=FakeClock(), sleep=FakeSleep(),
    )
    handle = AttemptHandle(
        attempt_n=1, project_id=project_id, run_dir=str(run_dir), driver="live",
        pid=1, launched_at=FakeClock()(), lease_ref=None,
    )

    driver.abort(handle, reason="already_dead_reason")

    assert killpg_calls == []  # never reached SIGKILL — getpgid raised first
    status = json.loads((run_dir / "demo_status.json").read_text())
    assert status["status"] == "killed"


def test_abort_never_overwrites_an_existing_terminal_status(tmp_path):
    runs_root = tmp_path / "runs"
    repo_root = tmp_path / "repo"
    project_id = "prj_abort_terminal_already"
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True)
    (run_dir / "demo_status.json").write_text(
        json.dumps({"status": "completed", "completedAt": "then"})
    )

    driver = LiveCliDriver(
        runs_root=runs_root, repo_root=repo_root, clock=FakeClock(), sleep=FakeSleep(),
    )
    handle = AttemptHandle(
        attempt_n=1, project_id=project_id, run_dir=str(run_dir), driver="live",
        pid=None, launched_at=FakeClock()(), lease_ref=None,
    )

    driver.abort(handle, reason="late_abort")

    status = json.loads((run_dir / "demo_status.json").read_text())
    assert status["status"] == "completed"
    assert status["completedAt"] == "then"
