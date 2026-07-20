"""State-machine tests for ``ReproductionCampaign`` (Unit 1).

No real stage logic is exercised here -- every ``CampaignStages`` callable is
a scripted fake (see ``_make_stages``). These tests drive: the write-ahead
intent invariant, halt-on-unwritable-ledger, every resume/crash-window
sub-case (spec §5 F7), the never-raises campaign-error fallback (§13),
fail-soft DISTILL, checkpoint-mode pausing, and per-meter spend accumulation.
Spec: docs/history/specs/2026-07-01-reproduction-campaign-and-self-
improving-harness-design.md §5, §14 ("Spend-ledger properties" + resume).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from backend.agents.rlm.reproduction_campaign import (
    CampaignInitError,
    CampaignLedger,
    CampaignLedgerError,
    CampaignState,
    CampaignStages,
    InFlight,
    ReproductionCampaign,
    default_liveness_probe,
)
from backend.eventstore.sqlite_store import SqliteEventStore
from backend.eventstore.interface import ConcurrencyError
from backend.messaging.envelope import AggregateId


# --------------------------------------------------------------------------- #
# Shared helpers                                                                #
# --------------------------------------------------------------------------- #


def _default_budget() -> dict:
    return {"max_llm_usd": 100.0, "max_gpu_usd": 100.0, "max_gpu_hours": 10.0, "max_attempts": 6}


def _zero_cost() -> dict:
    return {"llm_usd": 0.0, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 0.0}


class _Recorder:
    def __init__(self) -> None:
        self.calls: dict[str, list] = {}

    def record(self, name: str, *args, **kwargs) -> None:
        self.calls.setdefault(name, []).append((args, kwargs))

    def count(self, name: str) -> int:
        return len(self.calls.get(name, []))

    def first_args(self, name: str) -> tuple:
        return self.calls[name][0][0]


def _make_stages(run_dir: Path, rec: _Recorder, **overrides) -> CampaignStages:
    def _validate_init():
        rec.record("validate_init")
        return []

    def _understand():
        rec.record("understand")
        return {"sha256": "understanding-sha", "blocking": []}

    def _plan_attempt(state, rows):
        rec.record("plan_attempt", state.next_attempt_n)
        n = state.next_attempt_n
        return {
            "attempt_n": n,
            "directives_sha256": f"dsha-{n}",
            "envelope": {"llm_usd": 1.0, "gpu_usd": 1.0, "gpu_hours": 0.1, "wall_s": 600.0, "vm_ceiling_s": 900.0},
            "project_id": state.project_id,
            "run_dir": str(run_dir / f"attempt_{n}"),
            "refusal": None,
            "downgrade_to_checkpoint": False,
            "launch_payload": {"attempt_n": n},
        }

    def _launch(payload):
        rec.record("launch", payload)
        n = payload["attempt_n"]
        return {"pid": 10_000 + n, "run_dir": str(run_dir / f"attempt_{n}"), "lease_ref": None}

    def _await_result(handle):
        rec.record("await_result", handle)
        return {"run_dir": handle.get("run_dir"), "report_path": None, "exit_condition": "exited:0"}

    def _assess(raw_result, planned):
        rec.record("assess", raw_result, planned)
        return {"score": 1.0, "cost": _zero_cost()}

    def _assess_from_disk(in_flight):
        rec.record("assess_from_disk", in_flight)
        return {"score": 0.0, "cost": _zero_cost()}

    def _quarantine(in_flight):
        rec.record("quarantine", in_flight)

    def _distill(assessment):
        rec.record("distill", assessment)

    def _decide(state, rows):
        rec.record("decide", state.next_attempt_n)
        return {"kind": "REPRODUCED", "rule": "r1", "stop_reason": None, "next_plan": None}

    def _write_reports(state, rows, decision):
        rec.record("write_reports", state, rows, decision)

    def _liveness_probe(in_flight):
        rec.record("liveness_probe", in_flight)
        return True

    def _emit_event(name, payload):
        rec.record("emit_event", name, payload)

    kwargs: dict[str, Any] = dict(
        validate_init=_validate_init,
        understand=_understand,
        plan_attempt=_plan_attempt,
        launch=_launch,
        await_result=_await_result,
        assess=_assess,
        assess_from_disk=_assess_from_disk,
        quarantine=_quarantine,
        distill=_distill,
        decide=_decide,
        write_reports=_write_reports,
        liveness_probe=_liveness_probe,
        emit_event=_emit_event,
    )
    kwargs.update(overrides)
    return CampaignStages(**kwargs)


def _campaign(tmp_path: Path, stages: CampaignStages, **overrides) -> ReproductionCampaign:
    kwargs: dict[str, Any] = dict(
        run_dir=tmp_path / "run",
        project_id="proj_1",
        paper_ref="2605.15155",
        budget=_default_budget(),
        mode="unattended",
        driver="live_cli",
        stages=stages,
        resume=False,
    )
    kwargs.update(overrides)
    return ReproductionCampaign(**kwargs)


def _read_campaign_json(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "run" / "campaign" / "campaign.json").read_text(encoding="utf-8"))


def _seed_state(campaign_dir: Path, **overrides) -> CampaignState:
    base = dict(
        project_id="proj_1", paper_ref="2605.15155", state="attempt_loop",
        next_attempt_n=1, mode="unattended", driver="live_cli",
        budget=_default_budget(), spent=_zero_cost(), scope_rung=0,
        in_flight=None, understanding_sha256="understanding-sha", rubric_sha256=None,
        steering_cursor=0, pending_approval=None, warnings=[], terminal=None,
        created_at=1.0, updated_at=1.0,
    )
    base.update(overrides)
    state = CampaignState(**base)
    CampaignLedger(campaign_dir).write_state(state)
    return state


def _seed_row(campaign_dir: Path, **row) -> None:
    CampaignLedger(campaign_dir).append_row(row)


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #


def test_happy_path_single_attempt_reproduced_terminal(tmp_path):
    rec = _Recorder()
    stages = _make_stages(tmp_path / "run", rec)
    campaign = _campaign(tmp_path, stages)

    result = campaign.run()

    assert result["kind"] == "REPRODUCED"
    assert rec.count("launch") == 1
    assert rec.count("write_reports") == 1


def test_write_ahead_intent_row_precedes_launch(tmp_path):
    rec = _Recorder()
    run_dir = tmp_path / "run"

    def _launch(payload):
        rec.record("launch", payload)
        campaign_dir = run_dir / "campaign"
        rows_path = campaign_dir / "attempts.jsonl"
        assert rows_path.exists(), "intent row must be durable before launch() fires"
        rows = [json.loads(ln) for ln in rows_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert any(
            r["attempt_n"] == payload["attempt_n"] and r["status"] == "launched" for r in rows
        )
        # I4: step (b) -- the durable in_flight write -- must ALSO precede
        # launch(), not just step (a) the intent row.
        state_on_disk = json.loads((campaign_dir / "campaign.json").read_text(encoding="utf-8"))
        assert state_on_disk["in_flight"] is not None, "in_flight must be durable before launch() fires"
        assert state_on_disk["in_flight"]["attempt_n"] == payload["attempt_n"]
        n = payload["attempt_n"]
        return {"pid": 10_000 + n, "run_dir": str(run_dir / f"attempt_{n}"), "lease_ref": None}

    stages = _make_stages(run_dir, rec, launch=_launch)
    campaign = _campaign(tmp_path, stages)

    result = campaign.run()

    assert result["kind"] == "REPRODUCED"
    assert rec.count("launch") == 1


def test_default_launch_row_omits_scheduler_metadata_and_tree_marker_is_durable(tmp_path, monkeypatch):
    rec = _Recorder()
    run_dir = tmp_path / "run"

    defaults = _make_stages(run_dir, rec)
    default_campaign = _campaign(tmp_path, defaults)
    default_campaign.run()
    default_rows = CampaignLedger(run_dir / "campaign").read_rows()
    default_launch = next(row for row in default_rows if row["status"] == "launched")
    assert "branch_type" not in default_launch
    assert "is_safety_bracket" not in default_launch

    def _tree_plan(state, _rows):
        return {
            "attempt_n": state.next_attempt_n,
            "directives_sha256": "tree-dsha",
            "envelope": {"llm_usd": 1.0, "gpu_usd": 1.0, "gpu_hours": 0.1, "wall_s": 600.0, "vm_ceiling_s": 900.0},
            "project_id": state.project_id,
            "run_dir": str(run_dir / "tree_attempt"),
            "refusal": None,
            "downgrade_to_checkpoint": False,
            "launch_payload": {"attempt_n": state.next_attempt_n},
            "is_safety_bracket": True,
        }

    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "1")
    tree_stages = _make_stages(run_dir, _Recorder(), plan_attempt=_tree_plan)
    tree_campaign = _campaign(tmp_path, tree_stages, run_dir=tmp_path / "tree_run")
    tree_campaign.run()
    tree_rows = CampaignLedger(tmp_path / "tree_run" / "campaign").read_rows()
    tree_launch = next(row for row in tree_rows if row["status"] == "launched")
    assert tree_launch["is_safety_bracket"] is True


def test_scheduler_lineage_flag_off_never_touches_the_branch_event_store(tmp_path, monkeypatch):
    """Default-off is inert beyond serialisation: no branch-tree side effect."""
    monkeypatch.delenv("OPENRESEARCH_SCHEDULER_TREE", raising=False)

    class _MustStayUntouched:
        def __getattr__(self, _name):
            raise AssertionError("branch event store must not be touched while tree flag is off")

    rec = _Recorder()
    campaign = _campaign(
        tmp_path,
        _make_stages(tmp_path / "run", rec),
        branch_tree_event_store=_MustStayUntouched(),
    )

    assert campaign.run()["kind"] == "REPRODUCED"
    assert "branch_tree" not in [args[0] for args, _kwargs in rec.calls.get("emit_event", [])]
    launch = next(row for row in CampaignLedger(tmp_path / "run" / "campaign").read_rows()
                  if row["status"] == "launched")
    assert "branch_type" not in launch
    assert "is_safety_bracket" not in launch


def test_scheduler_lineage_tree_records_only_durable_root_f10_launch(tmp_path, monkeypatch):
    """Tree mode is factual shadow observability, not speculative ASHA control."""
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "yes")
    store = SqliteEventStore(f"sqlite:///{tmp_path / 'controller-events.db'}")
    rec = _Recorder()
    campaign = _campaign(
        tmp_path,
        _make_stages(tmp_path / "run", rec),
        branch_tree_event_store=store,
    )

    assert campaign.run()["kind"] == "REPRODUCED"

    events = list(store.load(AggregateId("branch-tree:proj_1")))
    assert len(events) == 1
    event = events[0]
    assert event.event_type == "branch_spawned"
    assert event.payload == {
        "branch_id": "1",
        "branch_type": "faithful",
        "parent_branch_id": None,
        "rung": 0,
        "hypothesis_fingerprint": "dsha-1",
    }
    assert event.envelope.source == "agents.rlm.reproduction_campaign"
    # It is not represented as an SSE decision/action and it does not imply a
    # checkpoint/rung transition beyond the root's durable launch intent.
    assert "branch_tree" not in [args[0] for args, _kwargs in rec.calls.get("emit_event", [])]
    assert {entry.event_type for entry in events} == {"branch_spawned"}
    store.close()


def test_scheduler_lineage_root_reemit_is_idempotent_on_existing_f10_fact(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "1")
    store = SqliteEventStore(f"sqlite:///{tmp_path / 'controller-events.db'}")
    campaign = _campaign(
        tmp_path,
        _make_stages(tmp_path / "run", _Recorder()),
        branch_tree_event_store=store,
    )

    assert campaign.run()["kind"] == "REPRODUCED"
    state = campaign._state
    assert state is not None
    launch = next(row for row in CampaignLedger(tmp_path / "run" / "campaign").read_rows()
                  if row["status"] == "launched")
    # Recovery retries the already-durable fact.  The F10 fingerprint is read
    # from the row and suppresses a second append without inventing a new key.
    campaign._maybe_emit_root_branch_spawned(state, launch)

    assert len(list(store.load(AggregateId("branch-tree:proj_1")))) == 1
    store.close()


@pytest.mark.parametrize("store_kind", ["outage", "concurrency"])
def test_scheduler_lineage_store_failures_are_fail_soft_after_durable_launch(
    tmp_path, monkeypatch, store_kind,
):
    # Establish the exact serial ledger/SSE baseline first.  Tree mode must
    # preserve it even if its optional EventStore projection is down/racing.
    monkeypatch.delenv("OPENRESEARCH_SCHEDULER_TREE", raising=False)
    baseline_rec = _Recorder()
    baseline_campaign = _campaign(
        tmp_path / "baseline",
        _make_stages(tmp_path / "baseline" / "run", baseline_rec),
    )
    baseline_result = baseline_campaign.run()
    baseline_rows = CampaignLedger(tmp_path / "baseline" / "run" / "campaign").read_rows()
    baseline_events = list(baseline_rec.calls["emit_event"])

    def _semantic_ledger(rows):
        # These values are controller timestamps or run-root paths, not
        # policy/ledger behavior.  Everything that can affect a launch,
        # assessment, decision, or terminal outcome must match exactly.
        normalized = []
        for row in rows:
            clean = dict(row)
            clean.pop("launched_at", None)
            clean.pop("assessed_at", None)
            clean.pop("run_dir", None)
            normalized.append(clean)
        return normalized

    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "true")

    class _BrokenStore:
        def load(self, _aggregate_id):
            if store_kind == "outage":
                raise RuntimeError("controller unavailable")
            return ()

        def get_aggregate_version(self, _aggregate_id):
            return 0

        def append(self, *_args, **_kwargs):
            if store_kind == "concurrency":
                raise ConcurrencyError("branch-tree:proj_1", expected=0, actual=1)
            raise AssertionError("outage must fail before append")

    rec = _Recorder()
    campaign = _campaign(
        tmp_path,
        _make_stages(tmp_path / "run", rec),
        branch_tree_event_store=_BrokenStore(),
    )

    assert campaign.run() == baseline_result
    rows = CampaignLedger(tmp_path / "run" / "campaign").read_rows()
    assert _semantic_ledger(rows) == _semantic_ledger(baseline_rows)
    # The serial decision and SSE stream are unchanged; only a durable
    # fail-soft warning records the unavailable observability projection.
    assert rec.calls["emit_event"] == baseline_events
    state = _read_campaign_json(tmp_path)
    assert any(warning.startswith("branch_lineage_emit_failed:") for warning in state["warnings"])


def test_ledger_error_on_intent_halts_and_never_launches(tmp_path, monkeypatch):
    rec = _Recorder()
    stages = _make_stages(tmp_path / "run", rec)
    campaign = _campaign(tmp_path, stages)

    def _raise(self, row):
        raise CampaignLedgerError("simulated append failure")

    monkeypatch.setattr(CampaignLedger, "append_row", _raise)

    with pytest.raises(CampaignLedgerError):
        campaign.run()

    assert rec.count("launch") == 0


def test_kill_between_intent_and_inflight_resume_relaunches_same_attempt_exactly_once(tmp_path):
    rec = _Recorder()
    run_dir = tmp_path / "run"
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)

    _seed_state(campaign_dir, next_attempt_n=1, in_flight=None)
    _seed_row(
        campaign_dir, attempt_n=1, status="launched", directives_sha256="dsha-1",
        envelope={}, driver="live_cli", project_id="proj_1",
        run_dir=str(run_dir / "attempt_1"), launched_at=999.0,
    )

    stages = _make_stages(run_dir, rec)
    campaign = _campaign(tmp_path, stages, resume=True)

    result = campaign.run()

    assert result["kind"] == "REPRODUCED"
    assert rec.count("launch") == 1
    assert rec.count("quarantine") == 1


def test_kill_after_launch_resume_reattaches_when_alive(tmp_path):
    rec = _Recorder()
    run_dir = tmp_path / "run"
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)

    in_flight = InFlight(
        attempt_n=1, driver="live_cli", run_dir=str(run_dir / "attempt_1"),
        pid=12345, lease_ref="gpu-lease-7", launched_at=999.0,
    )
    _seed_state(campaign_dir, next_attempt_n=1, in_flight=in_flight)
    _seed_row(
        campaign_dir, attempt_n=1, status="launched", directives_sha256="dsha-1",
        envelope={}, driver="live_cli", project_id="proj_1",
        run_dir=str(run_dir / "attempt_1"), launched_at=999.0,
    )

    stages = _make_stages(run_dir, rec)
    campaign = _campaign(tmp_path, stages, resume=True)

    result = campaign.run()

    assert result["kind"] == "REPRODUCED"
    assert rec.count("launch") == 0
    assert rec.count("await_result") == 1
    assert rec.count("assess") == 1
    assert rec.count("assess_from_disk") == 0
    # I3: the reattach must forward lease_ref -- a dropped lease_ref would
    # strand the driver unable to release/renew the compute lease on resume.
    await_handle = rec.first_args("await_result")[0]
    assert await_handle["lease_ref"] == "gpu-lease-7"


def test_kill_after_launch_resume_assesses_from_disk_when_dead_without_quarantine(tmp_path):
    # R1 (was test_kill_after_launch_resume_quarantines_and_assesses_from_disk_when_dead):
    # the dead branch must NOT quarantine before assess_from_disk when no
    # "assessed" row exists yet -- a completed-while-down attempt's result
    # lives in an INTACT run dir, and launch-time force-quarantine
    # (driver-side, Codex F6) already guarantees cleanliness before the next
    # launch. The old pinned behavior (quarantine THEN assess_from_disk)
    # would archive a real result out from under the assessor.
    rec = _Recorder()
    run_dir = tmp_path / "run"
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)

    in_flight = InFlight(
        attempt_n=1, driver="live_cli", run_dir=str(run_dir / "attempt_1"),
        pid=12345, lease_ref=None, launched_at=999.0,
    )
    _seed_state(campaign_dir, next_attempt_n=1, in_flight=in_flight)
    _seed_row(
        campaign_dir, attempt_n=1, status="launched", directives_sha256="dsha-1",
        envelope={}, driver="live_cli", project_id="proj_1",
        run_dir=str(run_dir / "attempt_1"), launched_at=999.0,
    )

    def _liveness_probe(inf):
        rec.record("liveness_probe", inf)
        return False

    stages = _make_stages(run_dir, rec, liveness_probe=_liveness_probe)
    campaign = _campaign(tmp_path, stages, resume=True)

    result = campaign.run()

    assert result["kind"] == "REPRODUCED"
    assert rec.count("launch") == 0
    assert rec.count("quarantine") == 0
    assert rec.count("assess_from_disk") == 1
    assert rec.count("assess") == 0
    rows = CampaignLedger(campaign_dir).read_rows()
    assert any(r["status"] == "assessed" for r in rows if r["attempt_n"] == 1)


def test_completed_while_down_attempt_is_assessed_from_intact_dir(tmp_path):
    # R1 (controller finding): an attempt that COMPLETED while the campaign
    # process was down must be read from the INTACT run dir. Quarantining it
    # first would archive a real, possibly-reproduced result out from under
    # assess_from_disk, superseding it with report_missing and losing its
    # cost. The quarantine fake below raises if it is ever called in this
    # scenario, and assess_from_disk itself asserts the evidence is intact.
    rec = _Recorder()
    run_dir = tmp_path / "run"
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)
    attempt_dir = run_dir / "attempt_1"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "metrics.json").write_text('{"score": 0.83}', encoding="utf-8")

    in_flight = InFlight(
        attempt_n=1, driver="live_cli", run_dir=str(attempt_dir),
        pid=54321, lease_ref=None, launched_at=999.0,
    )
    _seed_state(campaign_dir, next_attempt_n=1, in_flight=in_flight)
    _seed_row(
        campaign_dir, attempt_n=1, status="launched", directives_sha256="dsha-1",
        envelope={}, driver="live_cli", project_id="proj_1",
        run_dir=str(attempt_dir), launched_at=999.0,
    )

    def _liveness_probe(inf):
        return False

    def _quarantine(inf):
        raise AssertionError("quarantine must not run before assess_from_disk on the dead-completed path")

    def _assess_from_disk(inf):
        rec.record("assess_from_disk", inf)
        assert (attempt_dir / "metrics.json").exists(), "run dir must still be intact, not archived"
        return {"score": 0.83, "cost": _zero_cost()}

    stages = _make_stages(
        run_dir, rec,
        liveness_probe=_liveness_probe, quarantine=_quarantine, assess_from_disk=_assess_from_disk,
    )
    campaign = _campaign(tmp_path, stages, resume=True)

    result = campaign.run()

    assert result["kind"] == "REPRODUCED"
    assert rec.count("assess_from_disk") == 1


def test_crash_after_assessed_row_reuses_row_no_double_append_no_double_spend(tmp_path):
    # R1 item 1: crash landed between the ASSESS tail's
    # ledger.append_row("assessed") and its ledger.write_state (which would
    # have zeroed in_flight and rolled the cost into spent). Resume must
    # REUSE the durable assessed row -- never re-run assess_from_disk, never
    # append a second assessed row, and never double-count the cost.
    rec = _Recorder()
    run_dir = tmp_path / "run"
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)

    in_flight = InFlight(
        attempt_n=1, driver="live_cli", run_dir=str(run_dir / "attempt_1"),
        pid=24680, lease_ref=None, launched_at=999.0,
    )
    _seed_state(campaign_dir, next_attempt_n=1, in_flight=in_flight, spent=_zero_cost())
    _seed_row(
        campaign_dir, attempt_n=1, status="launched", directives_sha256="dsha-1",
        envelope={}, driver="live_cli", project_id="proj_1",
        run_dir=str(run_dir / "attempt_1"), launched_at=999.0,
    )
    _seed_row(
        campaign_dir, attempt_n=1, status="assessed",
        assessment={"score": 1.0, "cost": {"llm_usd": 2.0, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 0.0}},
        assessed_at=1000.0,
    )

    def _liveness_probe(inf):
        return False

    stages = _make_stages(run_dir, rec, liveness_probe=_liveness_probe)
    campaign = _campaign(tmp_path, stages, resume=True)

    result = campaign.run()

    assert result["kind"] == "REPRODUCED"
    assert rec.count("assess_from_disk") == 0
    assert rec.count("quarantine") == 0
    assert result["spent"]["llm_usd"] == pytest.approx(2.0)

    state_json = _read_campaign_json(tmp_path)
    assert state_json["spent"]["llm_usd"] == pytest.approx(2.0)
    assert state_json["in_flight"] is None

    rows = CampaignLedger(campaign_dir).read_rows()
    assessed_rows = [r for r in rows if r["attempt_n"] == 1 and r["status"] == "assessed"]
    assert len(assessed_rows) == 1


def test_resume_mid_loop_reenters_at_decide(tmp_path):
    rec = _Recorder()
    run_dir = tmp_path / "run"
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)

    _seed_state(campaign_dir, next_attempt_n=1, in_flight=None)
    _seed_row(
        campaign_dir, attempt_n=1, status="launched", directives_sha256="dsha-1",
        envelope={}, driver="live_cli", project_id="proj_1",
        run_dir=str(run_dir / "attempt_1"), launched_at=1.0,
    )
    _seed_row(
        campaign_dir, attempt_n=1, status="assessed",
        assessment={"score": 1.0, "cost": _zero_cost()}, assessed_at=2.0,
    )

    stages = _make_stages(run_dir, rec)
    campaign = _campaign(tmp_path, stages, resume=True)

    result = campaign.run()

    assert result["kind"] == "REPRODUCED"
    assert rec.count("launch") == 0
    assert rec.count("assess") == 0
    assert rec.count("assess_from_disk") == 0
    assert rec.count("decide") == 1


def test_resume_after_terminal_is_noop(tmp_path):
    rec = _Recorder()
    run_dir = tmp_path / "run"
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)

    terminal = {"kind": "REPRODUCED", "rule": "r1", "stop_reason": None,
                "champion_attempt_n": 1, "spent": _zero_cost()}
    _seed_state(campaign_dir, state="terminal", terminal=terminal)

    stages = _make_stages(run_dir, rec)
    campaign = _campaign(tmp_path, stages, resume=True)

    result = campaign.run()

    assert result == terminal
    assert rec.count("plan_attempt") == 0
    assert rec.count("launch") == 0
    assert rec.count("write_reports") == 0


def test_superseding_assessed_row_last_writer_wins(tmp_path):
    rec = _Recorder()
    run_dir = tmp_path / "run"
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)

    _seed_state(campaign_dir, next_attempt_n=1, in_flight=None)
    _seed_row(
        campaign_dir, attempt_n=1, status="launched", directives_sha256="dsha-1",
        envelope={}, driver="live_cli", project_id="proj_1",
        run_dir=str(run_dir / "attempt_1"), launched_at=1.0,
    )
    _seed_row(campaign_dir, attempt_n=1, status="assessed", assessment={"score": 0.1}, assessed_at=2.0)
    _seed_row(campaign_dir, attempt_n=1, status="assessed", assessment={"score": 0.9}, assessed_at=3.0)

    seen_scores = []

    def _decide(state, rows):
        latest = CampaignLedger.latest_by_status(rows, 1)
        seen_scores.append(latest["assessed"]["assessment"]["score"])
        return {"kind": "REPRODUCED", "rule": "r1", "stop_reason": None, "next_plan": None}

    stages = _make_stages(run_dir, rec, decide=_decide)
    campaign = _campaign(tmp_path, stages, resume=True)

    result = campaign.run()

    assert seen_scores == [0.9]
    assert result["kind"] == "REPRODUCED"


def test_stage_exception_degrades_to_exhausted_campaign_error(tmp_path):
    rec = _Recorder()

    def _plan_attempt(state, rows):
        rec.record("plan_attempt", state.next_attempt_n)
        raise RuntimeError("boom")

    stages = _make_stages(tmp_path / "run", rec, plan_attempt=_plan_attempt)
    campaign = _campaign(tmp_path, stages)

    result = campaign.run()

    assert result["kind"] == "EXHAUSTED"
    assert result["rule"] == "campaign_error"
    assert "RuntimeError" in result["stop_reason"]
    assert rec.count("write_reports") == 1


def test_degrade_path_ledger_error_still_halts(tmp_path, monkeypatch):
    # C1: _degrade_to_campaign_error's own terminal write_state (inside the
    # _finish_from_decision it calls) must NOT be swallowed by its bare
    # `except Exception` -- a CampaignLedgerError there means campaign.json
    # never recorded the terminal, so run() must RAISE rather than return a
    # terminal-shaped dict (which would read as a false "done" while
    # campaign.json still shows the live in_flight from before the crash).
    rec = _Recorder()

    def _plan_attempt(state, rows):
        rec.record("plan_attempt", state.next_attempt_n)
        raise RuntimeError("boom")

    stages = _make_stages(tmp_path / "run", rec, plan_attempt=_plan_attempt)
    campaign = _campaign(tmp_path, stages)

    original = CampaignLedger.write_state

    def _flaky(self, state):
        # Only the degrade path's OWN terminal write ever stamps
        # rule="campaign_error" -- target it specifically so the earlier,
        # legitimate pre-crash writes (validate_init/understand) still land.
        if state.terminal is not None and state.terminal.get("rule") == "campaign_error":
            raise CampaignLedgerError("simulated terminal write_state failure on degrade path")
        return original(self, state)

    monkeypatch.setattr(CampaignLedger, "write_state", _flaky)

    with pytest.raises(CampaignLedgerError):
        campaign.run()

    assert rec.count("write_reports") == 0
    # The failed write must never have landed: campaign.json still shows the
    # last successfully-committed (non-terminal) snapshot.
    state_json = _read_campaign_json(tmp_path)
    assert state_json["state"] == "attempt_loop"
    assert state_json["terminal"] is None


def test_degrade_path_non_ledger_write_reports_failure_returns_terminal_with_warning(tmp_path):
    # Companion to test_degrade_path_ledger_error_still_halts: a NON-ledger
    # failure inside the degrade path (e.g. write_reports itself blowing up)
    # must keep the old graceful behavior -- return the terminal-shaped dict
    # with a warning recorded, never raise. Only CampaignLedgerError re-raises.
    rec = _Recorder()

    def _plan_attempt(state, rows):
        rec.record("plan_attempt", state.next_attempt_n)
        raise RuntimeError("boom")

    def _write_reports(state, rows, decision):
        rec.record("write_reports", state, rows, decision)
        raise RuntimeError("write_reports boom")

    stages = _make_stages(
        tmp_path / "run", rec, plan_attempt=_plan_attempt, write_reports=_write_reports,
    )
    campaign = _campaign(tmp_path, stages)

    result = campaign.run()

    assert result["kind"] == "EXHAUSTED"
    assert result["rule"] == "campaign_error"
    assert "RuntimeError" in result["stop_reason"]
    # The warning is recorded on the in-memory state (the returned dict
    # itself, and campaign.json from the earlier successful terminal write,
    # intentionally carry no `warnings` key / stale warnings here -- the
    # write_reports failure happens AFTER that write, so there is no
    # subsequent write_state to persist it; "warn+return" is unchanged).
    assert any(
        w.startswith("campaign_error_report_failed:RuntimeError") for w in campaign._state.warnings
    )


def test_ledger_error_never_swallowed_by_never_raises_wrapper(tmp_path, monkeypatch):
    rec = _Recorder()
    stages = _make_stages(tmp_path / "run", rec)
    campaign = _campaign(tmp_path, stages)

    original = CampaignLedger.write_state

    def _flaky(self, state):
        if state.in_flight is not None and state.state == "attempt_loop":
            raise CampaignLedgerError("simulated write_state failure")
        return original(self, state)

    monkeypatch.setattr(CampaignLedger, "write_state", _flaky)

    with pytest.raises(CampaignLedgerError):
        campaign.run()


def test_distill_failure_is_fail_soft_warning(tmp_path):
    rec = _Recorder()

    def _distill(assessment):
        rec.record("distill", assessment)
        raise RuntimeError("distill boom")

    stages = _make_stages(tmp_path / "run", rec, distill=_distill)
    campaign = _campaign(tmp_path, stages)

    result = campaign.run()

    assert result["kind"] == "REPRODUCED"
    state_json = _read_campaign_json(tmp_path)
    assert any(w.startswith("distill_failed:RuntimeError") for w in state_json["warnings"])


def test_checkpoint_mode_pauses_after_decide_with_decision_recorded(tmp_path):
    rec = _Recorder()

    def _decide(state, rows):
        rec.record("decide", state.next_attempt_n)
        return {
            "kind": "CONTINUE", "rule": "r_continue", "stop_reason": None,
            "next_plan": {"lineage": "champion", "seed_attempt_n": 1, "seed_pointer": None,
                          "scope_rung": 0, "width": 1},
        }

    stages = _make_stages(tmp_path / "run", rec, decide=_decide)
    campaign = _campaign(tmp_path, stages, mode="checkpoint")

    result = campaign.run()

    assert result["kind"] == "PAUSED"
    assert result["pending_approval"]["decision"]["kind"] == "CONTINUE"
    state_json = _read_campaign_json(tmp_path)
    assert state_json["state"] == "paused"
    assert state_json["pending_approval"]["decision"]["kind"] == "CONTINUE"
    assert rec.count("write_reports") == 0
    # Unit 9a wires campaign_started/attempt_started/attempt_assessed/
    # campaign_decision emits earlier in the same run (spec §12) -- this
    # test's intent is "checkpoint mode notifies the operator", not "is the
    # very first emit_event call of the whole run", so check emission,
    # not position.
    assert any(call[0][0] == "campaign_awaiting_operator" for call in rec.calls["emit_event"])


def test_checkpoint_pause_emit_failure_is_fail_soft_with_warning(tmp_path):
    # The checkpoint-pause campaign_awaiting_operator emit must go through
    # _safe_emit -- a raising emitter would otherwise kill the loop at
    # exactly the pause transition instead of pausing gracefully.
    rec = _Recorder()

    def _decide(state, rows):
        rec.record("decide", state.next_attempt_n)
        return {
            "kind": "CONTINUE", "rule": "r_continue", "stop_reason": None,
            "next_plan": {"lineage": "champion", "seed_attempt_n": 1, "seed_pointer": None,
                          "scope_rung": 0, "width": 1},
        }

    def _emit_event(name, payload):
        rec.record("emit_event", name, payload)
        raise OSError("emitter down")

    stages = _make_stages(tmp_path / "run", rec, decide=_decide, emit_event=_emit_event)
    campaign = _campaign(tmp_path, stages, mode="checkpoint")

    result = campaign.run()

    assert result["kind"] == "PAUSED"
    assert result["pending_approval"]["decision"]["kind"] == "CONTINUE"
    assert any(call[0][0] == "campaign_awaiting_operator" for call in rec.calls["emit_event"])
    # The write_state that persisted state="paused" happens BEFORE this
    # specific emit (same "warning lands in-memory, not on the last-written
    # disk snapshot" shape as test_degrade_path_non_ledger_write_reports_
    # failure_returns_terminal_with_warning) -- assert on the in-memory state.
    assert any(
        w.startswith("emit_failed:campaign_awaiting_operator:OSError") for w in campaign._state.warnings
    )


def test_terminal_recorded_in_state_before_write_reports(tmp_path):
    rec = _Recorder()
    campaign_dir = tmp_path / "run" / "campaign"

    def _write_reports(state, rows, decision):
        rec.record("write_reports", state, rows, decision)
        on_disk = json.loads((campaign_dir / "campaign.json").read_text(encoding="utf-8"))
        assert on_disk["terminal"] is not None
        assert on_disk["terminal"]["kind"] == "REPRODUCED"

    stages = _make_stages(tmp_path / "run", rec, write_reports=_write_reports)
    campaign = _campaign(tmp_path, stages)

    result = campaign.run()

    assert result["kind"] == "REPRODUCED"
    assert rec.count("write_reports") == 1


def test_spent_accumulates_per_meter_from_assessment_cost(tmp_path):
    rec = _Recorder()
    decide_calls = {"n": 0}

    def _decide(state, rows):
        decide_calls["n"] += 1
        rec.record("decide", state.next_attempt_n)
        if decide_calls["n"] == 1:
            return {
                "kind": "CONTINUE", "rule": "r1", "stop_reason": None,
                "next_plan": {"lineage": "fresh", "seed_attempt_n": None, "seed_pointer": None,
                              "scope_rung": 0, "width": 1},
            }
        return {"kind": "REPRODUCED", "rule": "r2", "stop_reason": None, "next_plan": None}

    def _assess(raw_result, planned):
        rec.record("assess", raw_result, planned)
        return {"score": 1.0, "cost": {"llm_usd": 1.5, "gpu_usd": 0.5, "gpu_hours": 0.05, "wall_s": 100.0}}

    stages = _make_stages(tmp_path / "run", rec, decide=_decide, assess=_assess)
    campaign = _campaign(tmp_path, stages)

    result = campaign.run()

    assert result["kind"] == "REPRODUCED"
    assert rec.count("launch") == 2
    assert result["spent"]["llm_usd"] == pytest.approx(3.0)
    assert result["spent"]["gpu_usd"] == pytest.approx(1.0)
    assert result["spent"]["gpu_hours"] == pytest.approx(0.1)
    assert result["spent"]["wall_s"] == pytest.approx(200.0)


def test_understand_blocking_routes_to_infeasible_with_reports(tmp_path):
    rec = _Recorder()

    def _understand():
        rec.record("understand")
        return {"sha256": "u-sha", "blocking": ["dataset X unresolved: no source span"]}

    stages = _make_stages(tmp_path / "run", rec, understand=_understand)
    campaign = _campaign(tmp_path, stages)

    result = campaign.run()

    assert result["kind"] == "INFEASIBLE"
    assert "dataset X unresolved" in result["stop_reason"]
    assert rec.count("write_reports") == 1
    assert rec.count("launch") == 0
    assert rec.count("plan_attempt") == 0


def test_init_error_fails_at_zero_dollars(tmp_path):
    rec = _Recorder()

    def _validate_init():
        rec.record("validate_init")
        raise CampaignInitError("missing paper_ref")

    stages = _make_stages(tmp_path / "run", rec, validate_init=_validate_init)
    campaign = _campaign(tmp_path, stages)

    result = campaign.run()

    assert result["kind"] == "EXHAUSTED"
    assert result["rule"] == "init"
    assert "missing paper_ref" in result["stop_reason"]
    assert rec.count("launch") == 0
    assert rec.count("write_reports") == 1


def test_default_liveness_probe_pid_and_status(tmp_path):
    run_dir = tmp_path / "attempt_1"
    run_dir.mkdir()

    no_pid = InFlight(attempt_n=1, driver="live_cli", run_dir=str(run_dir),
                       pid=None, lease_ref=None, launched_at=1.0)
    assert default_liveness_probe(no_pid) is False

    (run_dir / "demo_status.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    assert default_liveness_probe(no_pid) is True

    (run_dir / "demo_status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    alive_pid = InFlight(attempt_n=1, driver="live_cli", run_dir=str(run_dir),
                          pid=os.getpid(), lease_ref=None, launched_at=1.0)
    assert default_liveness_probe(alive_pid) is True


def test_default_liveness_probe_pid_none_within_grace_window_is_alive(tmp_path):
    # I1: a crash between the write-ahead pid=None write and the post-launch
    # pid-stamped write must not read as instantly dead -- the freshly
    # spawned child may not have written demo_status.json yet.
    run_dir = tmp_path / "attempt_1"
    run_dir.mkdir()

    in_flight = InFlight(attempt_n=1, driver="live_cli", run_dir=str(run_dir),
                          pid=None, lease_ref=None, launched_at=time.time())
    assert default_liveness_probe(in_flight) is True


def test_default_liveness_probe_pid_none_past_grace_window_is_dead(tmp_path):
    # After the grace window elapses, a still-unknown pid reads dead exactly
    # like before I1.
    run_dir = tmp_path / "attempt_1"
    run_dir.mkdir()

    in_flight = InFlight(
        attempt_n=1, driver="live_cli", run_dir=str(run_dir),
        pid=None, lease_ref=None, launched_at=time.time() - 1000.0,
    )
    assert default_liveness_probe(in_flight) is False


def test_default_liveness_probe_pid_none_terminal_status_within_grace_is_dead(tmp_path):
    # A parseable, terminal demo_status resolves immediately -- a completed
    # run must not linger "alive" for the rest of the grace window.
    run_dir = tmp_path / "attempt_1"
    run_dir.mkdir()
    (run_dir / "demo_status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    in_flight = InFlight(attempt_n=1, driver="live_cli", run_dir=str(run_dir),
                          pid=None, lease_ref=None, launched_at=time.time())
    assert default_liveness_probe(in_flight) is False
