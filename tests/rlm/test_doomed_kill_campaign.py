"""tests/rlm/test_doomed_kill_campaign.py — the doomed-run early kill WIRED
into the campaign's AWAIT stage (spec §10.3, ``OPENRESEARCH_DOOMED_KILL``,
default OFF).

The comparator/watch/evidence-reader logic is unit-tested in
``test_doomed_run_comparator.py``. What is tested HERE is the part that
actually spends (or stops spending) money:

* a diverging attempt is killed mid-AWAIT and recorded as ``doomed_killed``
  -- honestly distinguishable from a science failure,
* a killed attempt can never become the champion, never seeds a lineage, and
  never has a lesson distilled from it,
* a healthy-but-slow attempt, an attempt with no baseline (attempt 1) and an
  in-process attempt (no pid to signal) are all NEVER killed,
* with the flag OFF the whole thing is byte-identical: no thread, no kill, no
  new assessment fields, no new events, no new report section.

Hermetic: no real child process is ever spawned and no real signal is ever
sent -- the process-group terminator is injected. Env is injected explicitly
per test (the suite is env-hermetic).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from backend.agents.rlm import reproduction_campaign as rc
from backend.agents.rlm.campaign_policy import select_champion, seeding_pool
from backend.agents.rlm.campaign_report import write_campaign_report
from backend.agents.rlm.doomed_run_comparator import DOOMED_KILL_ENV, DOOMED_POLLS_ENV
from backend.agents.rlm.attempt_assessment import AttemptAssessment
from backend.agents.rlm.reproduction_campaign import (
    CampaignLedger,
    CampaignStages,
    CampaignState,
    ReproductionCampaign,
)

_LIVE_PID = 424242
_CELL = "qwen-1.7b/alfworld/grpo"


# --------------------------------------------------------------------------- #
# curve fixtures                                                               #
# --------------------------------------------------------------------------- #


def _rising(n=11, start=0.1, stop=1.0, step_size=10):
    steps = [i * step_size for i in range(n)]
    values = [start + (stop - start) * i / (n - 1) for i in range(n)]
    return steps, values


def _write_curve(code_dir: Path, steps, values) -> None:
    """One cell's measured curve, in the aggregated ``per_model`` shape.
    Written atomically so the watcher thread can never read a torn file."""
    code_dir.mkdir(parents=True, exist_ok=True)
    model, env, baseline = _CELL.split("/")
    payload = {
        "per_model": {
            model: {
                env: {
                    baseline: {
                        "status": "ok",
                        "training_curves": {"step": list(steps), "reward": list(values)},
                    }
                }
            }
        }
    }
    tmp = code_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(code_dir / "metrics.json")


# --------------------------------------------------------------------------- #
# campaign fixtures                                                            #
# --------------------------------------------------------------------------- #


def _budget() -> dict:
    return {"max_llm_usd": 100.0, "max_gpu_usd": 100.0, "max_gpu_hours": 10.0, "max_attempts": 6}


def _cost() -> dict:
    return {"llm_usd": 3.5, "gpu_usd": 8.0, "gpu_hours": 1.0, "wall_s": 3600.0}


def _predicates(n: int) -> dict:
    keys = ["backed_by_ledger", "provenance_present", "metrics_non_degenerate", "metric_keys_real"]
    return {**{k: i < n for i, k in enumerate(keys)}, "run_level_clean": n >= 4}


def _assessment_dict(attempt_n: int, *, predicates: int = 3) -> dict:
    """What a real ``assess_attempt`` yields for an attempt whose child was
    killed: no final_report, so ``report_missing``, and (validator absent)
    soft-quarantined."""
    return {
        "attempt_n": attempt_n,
        "driver": "live",
        "project_id": "prj_test",
        "directives_sha256": f"dsha-{attempt_n}",
        "final_report": None,
        "evidence_predicates": _predicates(predicates),
        "guard_flags": {},
        "validator": {"status": "missing", "fingerprint": None, "fresh": False},
        "leaf_pass_count": None,
        "leaf_vector_ref": None,
        "failure_class": "report_missing",
        "failure_signature": "report_missing",
        "failure_scope": "infra",
        "cost": _cost(),
        "rubric_sha256_ok": None,
        "hard_quarantined": False,
        "soft_quarantined": True,
        "quarantine_reasons": ["validator:missing"],
    }


def _seed_completed_attempt_1(run_dir: Path, ledger: CampaignLedger, *, with_curve: bool = True) -> None:
    """A completed, trusted attempt 1: its ledger rows, its ARCHIVED ``code/``
    tree carrying the reference curve, and the driver-maintained attempt-code
    index that makes that archive addressable by attempt number."""
    ledger.append_row(
        {
            "attempt_n": 1,
            "status": "launched",
            "directives_sha256": "dsha-1",
            "envelope": {},
            "driver": "live",
            "project_id": "prj_test",
            "run_dir": str(run_dir),
            "launched_at": time.time() - 10_000,
        }
    )
    assessment = _assessment_dict(1, predicates=4)
    assessment["final_report"] = {"score": 0.42, "target": 0.8, "meets_target": False, "path": "x"}
    assessment["failure_class"] = None
    assessment["failure_signature"] = None
    assessment["failure_scope"] = None
    ledger.append_row(
        {"attempt_n": 1, "status": "assessed", "assessment": assessment, "assessed_at": time.time()}
    )

    archived_code = run_dir / "attempts" / "ts1" / "code"
    if with_curve:
        steps, values = _rising()
        _write_curve(archived_code, steps, values)
    else:
        archived_code.mkdir(parents=True, exist_ok=True)

    index = {"live": None, "archived": {"1": str(archived_code)}}
    index_path = run_dir / "campaign" / "attempt_code_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index), encoding="utf-8")


def _seed_state(run_dir: Path, ledger: CampaignLedger, *, next_attempt_n: int) -> CampaignState:
    now = time.time()
    state = CampaignState(
        project_id="prj_test",
        paper_ref="2605.15155",
        state="attempt_loop",
        next_attempt_n=next_attempt_n,
        mode="unattended",
        driver="live",
        budget=_budget(),
        spent={"llm_usd": 0.0, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 0.0},
        scope_rung=0,
        in_flight=None,
        understanding_sha256="u-sha",
        rubric_sha256=None,
        steering_cursor=0,
        pending_approval=None,
        warnings=[],
        terminal=None,
        created_at=now,
        updated_at=now,
    )
    ledger.write_state(state)
    return state


class _Harness:
    """A campaign whose child is simulated: ``await_result`` writes the live
    attempt's curve to disk exactly as a real child would, then blocks until
    it is killed (or finishes)."""

    def __init__(self, run_dir: Path, *, live_values, pid=_LIVE_PID):
        self.run_dir = run_dir
        self.live_values = live_values
        self.pid = pid
        self.killed = threading.Event()
        self.terminated_pids: list[int] = []
        self.distilled: list[dict] = []
        self.assessments: list[dict] = []
        self.events: list[tuple[str, dict]] = []

    def terminate(self, pid: int, **_kw) -> bool:
        self.terminated_pids.append(pid)
        self.killed.set()
        return True

    def _await_result(self, handle):
        code = Path(handle["run_dir"]) / "code"
        steps, _ = _rising()
        # The child writes its curve progressively, exactly like a real
        # trainer: 3 points, then 4, then 5... The watcher only counts an
        # observation when the curve ADVANCES, so this is what drives it.
        for i in range(2, len(steps)):
            _write_curve(code, steps[: i + 1], self.live_values[: i + 1])
            if self.killed.wait(0.05):
                break
        # Grace for an in-flight kill decision. The watcher polls 5x per write
        # above, so a kill (3 advancing observations past the progress floor)
        # always lands inside the loop -- this is belt-and-braces, not timing.
        self.killed.wait(0.5)
        return {
            "run_dir": handle["run_dir"],
            "report_path": None,
            "exit_condition": "already_dead" if self.killed.is_set() else "exited:0",
        }

    def _emit(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, dict(payload)))
        path = self.run_dir / "dashboard_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": event_type, **payload}, default=str) + "\n")

    def _assess(self, raw, planned):
        assessment = _assessment_dict(int(planned["attempt_n"]))
        self.assessments.append(assessment)
        return assessment

    def stages(self, *, attempt_n: int) -> CampaignStages:
        def _plan(state, rows):
            return {
                "attempt_n": attempt_n,
                "directives_sha256": f"dsha-{attempt_n}",
                "envelope": {"llm_usd": 1.0, "gpu_usd": 1.0, "gpu_hours": 0.1,
                             "wall_s": 600.0, "vm_ceiling_s": 900.0},
                "project_id": "prj_test",
                "run_dir": str(self.run_dir),
                "refusal": None,
                "downgrade_to_checkpoint": False,
                "launch_payload": {"attempt_n": attempt_n},
            }

        return CampaignStages(
            validate_init=lambda: [],
            understand=lambda: {"sha256": "u-sha", "blocking": []},
            plan_attempt=_plan,
            launch=lambda payload: {"pid": self.pid, "run_dir": str(self.run_dir), "lease_ref": None},
            await_result=self._await_result,
            assess=self._assess,
            assess_from_disk=lambda in_flight: _assessment_dict(in_flight.attempt_n),
            quarantine=lambda in_flight: None,
            distill=lambda assessment: self.distilled.append(dict(assessment)),
            decide=lambda state, rows: {
                "kind": "EXHAUSTED", "rule": "max_attempts", "stop_reason": "max_attempts",
                "next_plan": None, "champion_attempt_n": None,
            },
            write_reports=lambda state, rows, decision: write_campaign_report(
                self.run_dir, state=state.to_dict(), rows=rows
            ),
            liveness_probe=lambda in_flight: False,
            emit_event=self._emit,
        )


def _campaign(run_dir: Path, harness: _Harness, *, attempt_n: int = 2) -> ReproductionCampaign:
    return ReproductionCampaign(
        run_dir=run_dir,
        project_id="prj_test",
        paper_ref="2605.15155",
        budget=_budget(),
        mode="unattended",
        driver="live",
        stages=harness.stages(attempt_n=attempt_n),
        resume=True,
    )


@pytest.fixture
def fast_polls(monkeypatch):
    monkeypatch.setattr(rc, "_DOOMED_POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(rc, "_DOOMED_JOIN_TIMEOUT_S", 10.0)


def _enable(monkeypatch, *, polls: int = 3) -> None:
    monkeypatch.setenv(DOOMED_KILL_ENV, "1")
    monkeypatch.setenv(DOOMED_POLLS_ENV, str(polls))


def _assessed(run_dir: Path, attempt_n: int) -> dict:
    rows = CampaignLedger(run_dir / "campaign").read_rows()
    return CampaignLedger.latest_by_status(rows, attempt_n)["assessed"]["assessment"]


# --------------------------------------------------------------------------- #
# the kill                                                                      #
# --------------------------------------------------------------------------- #


def test_diverging_attempt_is_killed_and_recorded_as_doomed_killed(tmp_path, monkeypatch, fast_polls):
    """The headline: an attempt whose measured curve trails the best completed
    attempt's curve at the same step, for the configured number of consecutive
    advancing polls, is killed -- and recorded as ``doomed_killed``, NOT as a
    science failure."""
    _enable(monkeypatch, polls=3)
    run_dir = tmp_path / "prj_test"
    ledger = CampaignLedger(run_dir / "campaign")
    (run_dir / "campaign").mkdir(parents=True)
    _seed_completed_attempt_1(run_dir, ledger)
    _seed_state(run_dir, ledger, next_attempt_n=2)

    harness = _Harness(run_dir, live_values=[0.02] * 11)  # pinned flat while the reference rose
    monkeypatch.setattr(rc, "_terminate_process_group", harness.terminate)

    _campaign(run_dir, harness).run()

    # 1. We actually stopped paying: the child's process GROUP was signalled.
    assert harness.terminated_pids == [_LIVE_PID]

    # 2. The attempt is recorded honestly and distinguishably.
    assessment = _assessed(run_dir, 2)
    assert assessment["failure_class"] == "doomed_killed"
    assert assessment["failure_signature"] == "doomed_killed"
    assert assessment["failure_scope"] is None  # not infra, not method -- we interrupted it
    assert assessment["hard_quarantined"] is True
    assert any(r.startswith("doomed_killed:") for r in assessment["quarantine_reasons"])

    # 3. ...naming the MEASURED evidence that justified it.
    kill = assessment["doomed_kill"]
    assert kill["baseline_attempt_n"] == 1
    assert kill["reason"] == "curve_below_best_completed_attempt"
    assert kill["cells"] == [_CELL]
    assert kill["polls_required"] == 3
    assert kill["observations"] >= 3
    assert any(_CELL in d and "relative gap" in d for d in kill["detail"])

    # 4. It is NOT mislabelled as the harness failing to produce a report.
    assert assessment["failure_class"] != "report_missing"


def test_kill_emits_an_sse_event_and_a_run_warning_naming_the_evidence(tmp_path, monkeypatch, fast_polls):
    _enable(monkeypatch, polls=3)
    run_dir = tmp_path / "prj_test"
    ledger = CampaignLedger(run_dir / "campaign")
    (run_dir / "campaign").mkdir(parents=True)
    _seed_completed_attempt_1(run_dir, ledger)
    _seed_state(run_dir, ledger, next_attempt_n=2)

    harness = _Harness(run_dir, live_values=[0.02] * 11)
    monkeypatch.setattr(rc, "_terminate_process_group", harness.terminate)
    _campaign(run_dir, harness).run()

    by_type = {name: payload for name, payload in harness.events}
    assert "attempt_doomed_killed" in by_type
    assert by_type["attempt_doomed_killed"]["cells"] == [_CELL]

    warning = by_type["run_warning"]
    assert warning["code"] == "doomed_run_killed"
    assert "stopped paying" in warning["message"]
    assert warning["baseline_attempt_n"] == 1
    assert any(_CELL in d for d in warning["evidence"])

    # ...and both are durable on the SSE log, not just in-memory.
    logged = [json.loads(x) for x in (run_dir / "dashboard_events.jsonl").read_text().splitlines()]
    assert {e["event"] for e in logged} >= {"attempt_doomed_killed", "run_warning"}


def test_killed_attempt_cannot_become_champion_and_cannot_seed(tmp_path, monkeypatch, fast_polls):
    """Round-trip through the REAL ledger row -> AttemptAssessment -> policy."""
    _enable(monkeypatch, polls=3)
    run_dir = tmp_path / "prj_test"
    ledger = CampaignLedger(run_dir / "campaign")
    (run_dir / "campaign").mkdir(parents=True)
    _seed_completed_attempt_1(run_dir, ledger)
    _seed_state(run_dir, ledger, next_attempt_n=2)

    harness = _Harness(run_dir, live_values=[0.02] * 11)
    monkeypatch.setattr(rc, "_terminate_process_group", harness.terminate)
    _campaign(run_dir, harness).run()

    killed = AttemptAssessment.from_dict(_assessed(run_dir, 2))

    assert killed.grade_usable_for_terminal is False
    assert killed.usable_for_seeding is False
    assert select_champion([killed]) is None
    assert seeding_pool([killed]) == []


def test_killed_attempt_is_not_distilled_into_cross_run_memory(tmp_path, monkeypatch, fast_polls):
    """DISTILL writes memory that OUTLIVES the campaign. An attempt we
    interrupted observed nothing, so mining a 'lesson' from it would teach the
    harness about our own kill."""
    _enable(monkeypatch, polls=3)
    run_dir = tmp_path / "prj_test"
    ledger = CampaignLedger(run_dir / "campaign")
    (run_dir / "campaign").mkdir(parents=True)
    _seed_completed_attempt_1(run_dir, ledger)
    _seed_state(run_dir, ledger, next_attempt_n=2)

    harness = _Harness(run_dir, live_values=[0.02] * 11)
    monkeypatch.setattr(rc, "_terminate_process_group", harness.terminate)
    _campaign(run_dir, harness).run()

    assert harness.distilled == []


def test_killed_attempt_still_charges_its_real_spend(tmp_path, monkeypatch, fast_polls):
    """We stopped early, but the money up to that point was really spent -- and
    the spend meters are what keep the number of kills bounded."""
    _enable(monkeypatch, polls=3)
    run_dir = tmp_path / "prj_test"
    ledger = CampaignLedger(run_dir / "campaign")
    (run_dir / "campaign").mkdir(parents=True)
    _seed_completed_attempt_1(run_dir, ledger)
    _seed_state(run_dir, ledger, next_attempt_n=2)

    harness = _Harness(run_dir, live_values=[0.02] * 11)
    monkeypatch.setattr(rc, "_terminate_process_group", harness.terminate)
    _campaign(run_dir, harness).run()

    spent = ledger.load_state().spent
    assert spent["llm_usd"] == pytest.approx(_cost()["llm_usd"])
    assert spent["gpu_usd"] == pytest.approx(_cost()["gpu_usd"])


def test_campaign_report_discloses_the_kill_as_stopped_early_not_a_failure(tmp_path, monkeypatch, fast_polls):
    _enable(monkeypatch, polls=3)
    run_dir = tmp_path / "prj_test"
    ledger = CampaignLedger(run_dir / "campaign")
    (run_dir / "campaign").mkdir(parents=True)
    _seed_completed_attempt_1(run_dir, ledger)
    _seed_state(run_dir, ledger, next_attempt_n=2)

    harness = _Harness(run_dir, live_values=[0.02] * 11)
    monkeypatch.setattr(rc, "_terminate_process_group", harness.terminate)
    _campaign(run_dir, harness).run()

    report = (run_dir / "campaign_report.md").read_text(encoding="utf-8")

    assert "## Stopped early (doomed)" in report
    assert "the campaign stopped paying" in report
    assert "NOT science failures" in report
    # The attempt row reads "stopped-early", never "hard-quarantined" (which
    # in this report means fabrication, i.e. an untrustworthy attempt).
    assert "stopped-early (doomed)" in report
    assert "hard-quarantined" not in report
    assert _CELL in report  # the evidence is named in the deliverable


# --------------------------------------------------------------------------- #
# false-positive safety                                                         #
# --------------------------------------------------------------------------- #


def test_healthy_but_slow_attempt_is_never_killed(tmp_path, monkeypatch, fast_polls):
    """25% behind the reference at every step -- genuinely slower, but inside
    the 30% margin. Must run to completion."""
    _enable(monkeypatch, polls=3)
    run_dir = tmp_path / "prj_test"
    ledger = CampaignLedger(run_dir / "campaign")
    (run_dir / "campaign").mkdir(parents=True)
    _seed_completed_attempt_1(run_dir, ledger)
    _seed_state(run_dir, ledger, next_attempt_n=2)

    _steps, values = _rising()
    harness = _Harness(run_dir, live_values=[v * 0.75 for v in values])
    monkeypatch.setattr(rc, "_terminate_process_group", harness.terminate)
    _campaign(run_dir, harness).run()

    assert harness.terminated_pids == []
    assessment = _assessed(run_dir, 2)
    assert "doomed_kill" not in assessment
    assert assessment["failure_class"] == "report_missing"  # untouched by the guard
    assert harness.distilled  # a real attempt IS distilled


def test_attempt_1_is_structurally_unkillable(tmp_path, monkeypatch, fast_polls):
    """No completed attempt to compare against => no baseline => no watch is
    even constructed, no matter how bad the curve looks."""
    _enable(monkeypatch, polls=1)
    run_dir = tmp_path / "prj_test"
    ledger = CampaignLedger(run_dir / "campaign")
    (run_dir / "campaign").mkdir(parents=True)
    _seed_state(run_dir, ledger, next_attempt_n=1)

    harness = _Harness(run_dir, live_values=[0.0] * 11)
    monkeypatch.setattr(rc, "_terminate_process_group", harness.terminate)
    _campaign(run_dir, harness, attempt_n=1).run()

    assert harness.terminated_pids == []
    assert "doomed_kill" not in _assessed(run_dir, 1)


def test_no_baseline_curve_means_no_kill(tmp_path, monkeypatch, fast_polls):
    """Attempt 1 completed but persisted no curve at all: nothing to compare
    against, so the attempt cannot be judged and is never killed."""
    _enable(monkeypatch, polls=1)
    run_dir = tmp_path / "prj_test"
    ledger = CampaignLedger(run_dir / "campaign")
    (run_dir / "campaign").mkdir(parents=True)
    _seed_completed_attempt_1(run_dir, ledger, with_curve=False)
    _seed_state(run_dir, ledger, next_attempt_n=2)

    harness = _Harness(run_dir, live_values=[0.0] * 11)
    monkeypatch.setattr(rc, "_terminate_process_group", harness.terminate)
    _campaign(run_dir, harness).run()

    assert harness.terminated_pids == []


def test_in_process_attempt_with_no_pid_is_never_killed(tmp_path, monkeypatch, fast_polls):
    """The unified driver runs the attempt IN-PROCESS: there is no process
    group to signal. A guard that cannot actually stop the spend must not
    report a kill it did not perform."""
    _enable(monkeypatch, polls=1)
    run_dir = tmp_path / "prj_test"
    ledger = CampaignLedger(run_dir / "campaign")
    (run_dir / "campaign").mkdir(parents=True)
    _seed_completed_attempt_1(run_dir, ledger)
    _seed_state(run_dir, ledger, next_attempt_n=2)

    harness = _Harness(run_dir, live_values=[0.0] * 11, pid=None)
    monkeypatch.setattr(rc, "_terminate_process_group", harness.terminate)
    _campaign(run_dir, harness).run()

    assert harness.terminated_pids == []
    assert "doomed_kill" not in _assessed(run_dir, 2)


def test_a_run_that_finishes_inside_the_poll_window_is_not_mislabelled_a_kill(
    tmp_path, monkeypatch, fast_polls
):
    """The comparator fired, but by the time we signalled, the child had
    already exited on its own. We stopped nothing, so we claim nothing: its
    real (bad) result stands and is assessed exactly as it would have been."""
    _enable(monkeypatch, polls=3)
    run_dir = tmp_path / "prj_test"
    ledger = CampaignLedger(run_dir / "campaign")
    (run_dir / "campaign").mkdir(parents=True)
    _seed_completed_attempt_1(run_dir, ledger)
    _seed_state(run_dir, ledger, next_attempt_n=2)

    harness = _Harness(run_dir, live_values=[0.02] * 11)

    def _already_dead(pid, **_kw):
        harness.terminated_pids.append(pid)
        harness.killed.set()  # unblock the fake child
        return False  # ...but we signalled nothing: it was already gone

    monkeypatch.setattr(rc, "_terminate_process_group", _already_dead)
    _campaign(run_dir, harness).run()

    assert harness.terminated_pids == [_LIVE_PID]  # we tried
    assessment = _assessed(run_dir, 2)
    assert "doomed_kill" not in assessment  # ...and correctly claimed nothing
    assert assessment["failure_class"] == "report_missing"
    assert assessment["hard_quarantined"] is False


def test_a_slow_graceful_kill_is_still_recorded_when_await_returns_first(
    tmp_path, monkeypatch, fast_polls
):
    """REGRESSION: the kill must be RECORDED before it is SIGNALLED.

    A SIGTERM the child handles gracefully flips ``demo_status`` to "killed"
    almost immediately, so AWAIT can return -- and the caller can reach
    ``_apply_doomed_kill`` -- while the terminator is still inside its
    SIGTERM->SIGKILL grace loop waiting for the process to actually exit.

    Recording only AFTER the terminator returns loses that race: the attempt
    gets silently stopped and then reported as a plain science failure, which
    is the single dishonest outcome this feature must never produce. Here the
    join timeout is deliberately shorter than the kill, so a
    record-after-signal implementation cannot pass."""
    _enable(monkeypatch, polls=3)
    monkeypatch.setattr(rc, "_DOOMED_JOIN_TIMEOUT_S", 0.05)  # << the kill's duration
    run_dir = tmp_path / "prj_test"
    ledger = CampaignLedger(run_dir / "campaign")
    (run_dir / "campaign").mkdir(parents=True)
    _seed_completed_attempt_1(run_dir, ledger)
    _seed_state(run_dir, ledger, next_attempt_n=2)

    harness = _Harness(run_dir, live_values=[0.02] * 11)

    def _slow_graceful_kill(pid, **_kw):
        harness.terminated_pids.append(pid)
        harness.killed.set()  # the child's status goes terminal at once...
        time.sleep(0.5)  # ...but the process itself lingers, finalizing
        return True

    monkeypatch.setattr(rc, "_terminate_process_group", _slow_graceful_kill)
    _campaign(run_dir, harness).run()

    assert harness.terminated_pids == [_LIVE_PID]
    assessment = _assessed(run_dir, 2)
    assert assessment["failure_class"] == "doomed_killed"
    assert assessment["hard_quarantined"] is True
    assert assessment["doomed_kill"]["baseline_attempt_n"] == 1


# --------------------------------------------------------------------------- #
# flag OFF => byte-identical                                                     #
# --------------------------------------------------------------------------- #


def test_flag_off_never_kills_adds_no_fields_no_events_and_no_report_section(
    tmp_path, monkeypatch, fast_polls
):
    monkeypatch.delenv(DOOMED_KILL_ENV, raising=False)
    run_dir = tmp_path / "prj_test"
    ledger = CampaignLedger(run_dir / "campaign")
    (run_dir / "campaign").mkdir(parents=True)
    _seed_completed_attempt_1(run_dir, ledger)
    _seed_state(run_dir, ledger, next_attempt_n=2)

    # The most kill-worthy curve imaginable -- flat zero against a rising
    # reference. With the flag off, nothing looks at it.
    harness = _Harness(run_dir, live_values=[0.0] * 11)
    monkeypatch.setattr(rc, "_terminate_process_group", harness.terminate)

    def _await_no_kill(handle):
        code = Path(handle["run_dir"]) / "code"
        steps, _ = _rising()
        _write_curve(code, steps, [0.0] * len(steps))
        return {"run_dir": handle["run_dir"], "report_path": None, "exit_condition": "exited:0"}

    harness._await_result = _await_no_kill  # noqa: SLF001 -- deliberate: no kill to wait for
    _campaign(run_dir, harness).run()

    assert harness.terminated_pids == []

    assessment = _assessed(run_dir, 2)
    assert "doomed_kill" not in assessment
    assert assessment["failure_class"] == "report_missing"
    assert assessment["hard_quarantined"] is False
    assert assessment["quarantine_reasons"] == ["validator:missing"]

    assert not any(name in ("attempt_doomed_killed", "run_warning") for name, _ in harness.events)

    report = (run_dir / "campaign_report.md").read_text(encoding="utf-8")
    assert "Stopped early" not in report
    assert "doomed" not in report

    assert harness.distilled  # DISTILL runs normally


def test_flag_off_starts_no_watch_thread(tmp_path, monkeypatch, fast_polls):
    """Structural: with the flag off the AWAIT stage is a bare
    ``stages.await_result`` -- no thread, and not even a disk read."""
    monkeypatch.delenv(DOOMED_KILL_ENV, raising=False)
    run_dir = tmp_path / "prj_test"
    ledger = CampaignLedger(run_dir / "campaign")
    (run_dir / "campaign").mkdir(parents=True)
    _seed_completed_attempt_1(run_dir, ledger)
    _seed_state(run_dir, ledger, next_attempt_n=2)

    def _boom(*_a, **_kw):
        raise AssertionError("the evidence reader must never run with the flag off")

    monkeypatch.setattr(rc.doomed, "read_cell_curves", _boom)
    monkeypatch.setattr(rc.doomed, "select_baseline", _boom)

    harness = _Harness(run_dir, live_values=[0.0] * 11)
    harness.killed.set()  # await returns immediately
    monkeypatch.setattr(rc, "_terminate_process_group", harness.terminate)

    before = threading.active_count()
    _campaign(run_dir, harness).run()

    assert threading.active_count() <= before
    assert harness.terminated_pids == []
