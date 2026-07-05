"""Tests for backend.agents.rlm.campaign_composition (Unit 9a).

Hermetic: tmp_path-only writes, no sockets, no LLM calls, no subprocess
spawns (the LiveCliDriver is constructed but never launched — every test
either calls a stage impl directly or drives the real ``ReproductionCampaign``
with scripted fake stages). The one repo READ is the checked-in canonical
profile ``configs/campaign_run_spec.json`` (the disjointness test's subject).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from backend.agents.rlm import campaign_composition as cc
from backend.agents.rlm.attempt_assessment import rubric_sha256
from backend.agents.rlm.attempt_driver import LiveCliDriver, PairedDriver, UnifiedRunDriver
from backend.agents.rlm.campaign_composition import CampaignOptions, build_campaign
from backend.agents.rlm.reproduction_campaign import (
    CampaignInitError,
    CampaignLedger,
    CampaignStages,
    CampaignState,
    InFlight,
    ReproductionCampaign,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean_campaign_env(monkeypatch):
    for var in (
        "OPENRESEARCH_NEGATIVE_LESSONS",
        "OPENRESEARCH_POSITIVE_RECIPES",
        "OPENRESEARCH_EXPERIENCE_MEMORY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def neutral_profile(tmp_path_factory) -> Path:
    """A profile whose PATH avoids U6's forbidden clean-context markers.

    ``tmp_path`` embeds the test name, so a test named ``...plan_attempt...``
    would put ``attempt_`` (a forbidden marker) into ``run_spec_path`` and
    fail directive synthesis on the fixture path rather than the behavior
    under test. ``tmp_path_factory.mktemp`` gives a neutral basename.
    """
    return _write_profile(tmp_path_factory.mktemp("campaign_cfg") / "profile.json")


# --------------------------------------------------------------------------- #
# Builders                                                                     #
# --------------------------------------------------------------------------- #


def _write_profile(path: Path, extra: dict | None = None, *, validator: str | None = "1") -> Path:
    profile: dict = {"OPENRESEARCH_REUSE_RUBRIC": "1"}
    if validator is not None:
        profile["OPENRESEARCH_EXTERNAL_VALIDATOR"] = validator
    profile.update(extra or {})
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


def _opts(tmp_path: Path, **overrides) -> CampaignOptions:
    base = dict(
        paper_ref="2605.15155",
        runs_root=tmp_path / "runs",
        repo_root=tmp_path,
        max_llm_usd=100.0,
        max_gpu_usd=100.0,
        max_gpu_hours=10.0,
        max_attempts=6,
        wall_clock_s=None,
        mode="unattended",
        driver="live",
        width=1,
        plateau_k=2,
        sandbox="local",
        billing_sandbox=None,
        gpu_usd_per_hr=None,
        est_gpu_hours=2.0,
        run_spec_path=str(_write_profile(tmp_path / "profile.json")),
        scope_spec=None,
        scope_ladder=("full",),
        paper_hint=None,
        require_cpu_tier=False,
        arxiv_id="2605.15155",
        paper_class="generic",
        resume=False,
    )
    base.update(overrides)
    return CampaignOptions(**base)


def _budget() -> dict:
    return {
        "max_llm_usd": 100.0, "max_gpu_usd": 100.0, "max_gpu_hours": 10.0,
        "max_attempts": 6, "max_wall_clock_s": None,
    }


def _tiny_budget() -> dict:
    # Small enough to be "tiny" while staying above every ENVELOPE_FLOORS /
    # MIN_USEFUL_WALL_S boundary (billing_sandbox="local" so the gpu_usd
    # rate-based bound never applies -- only gpu_hours/max_attempts needs to
    # clear 0.25h -> 900s effective wall clock).
    return {
        "max_llm_usd": 5.0, "max_gpu_usd": 5.0, "max_gpu_hours": 2.0,
        "max_attempts": 4, "max_wall_clock_s": None,
    }


def _zero_cost() -> dict:
    return {"llm_usd": 0.0, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 0.0}


def _state(**overrides) -> CampaignState:
    base = dict(
        project_id="prj_t", paper_ref="2605.15155", state="attempt_loop",
        next_attempt_n=1, mode="unattended", driver="live",
        budget=_budget(), spent=_zero_cost(), scope_rung=0,
        in_flight=None, understanding_sha256=None, rubric_sha256=None,
        steering_cursor=0, pending_approval=None, warnings=[], terminal=None,
        created_at=1.0, updated_at=1.0,
    )
    base.update(overrides)
    return CampaignState(**base)


def _assessment_dict(n: int, **overrides) -> dict:
    base = {
        "attempt_n": n, "driver": "live", "project_id": "prj_t",
        "directives_sha256": f"dsha-{n}",
        "final_report": {
            "score": 0.4, "target": 0.7, "meets_target": False,
            "implementation_verdict": None, "replication_verdict": None,
            "verdict": "partial", "stop_reason": None, "exclusions": [],
            "path": "final_report.json",
        },
        "evidence_predicates": {"backed_by_ledger": True, "run_level_clean": False},
        "guard_flags": {},
        "validator": {"status": "clean", "fingerprint": "f", "fresh": True},
        "leaf_pass_count": 3, "leaf_vector_ref": None,
        "failure_class": None, "failure_signature": None, "failure_scope": None,
        "cost": {"llm_usd": 1.0, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 10.0},
        "rubric_sha256_ok": None,
        "hard_quarantined": False, "soft_quarantined": False, "quarantine_reasons": [],
    }
    base.update(overrides)
    return base


def _scripted_stages(run_dir: Path, emitted: list, **overrides) -> CampaignStages:
    def _plan(state, rows):
        n = state.next_attempt_n
        return {
            "attempt_n": n, "directives_sha256": f"d-{n}",
            "envelope": {"llm_usd": 1.0, "gpu_usd": 1.0, "gpu_hours": 0.1, "wall_s": 600.0, "vm_ceiling_s": 900.0},
            "project_id": state.project_id, "run_dir": str(run_dir / f"a{n}"),
            "refusal": None, "downgrade_to_checkpoint": False,
            "launch_payload": {"attempt_n": n},
        }

    kwargs = dict(
        validate_init=lambda: [],
        understand=lambda: {"sha256": "u", "blocking": []},
        plan_attempt=_plan,
        launch=lambda payload: {"pid": None, "run_dir": str(run_dir / f"a{payload['attempt_n']}"), "lease_ref": None},
        await_result=lambda handle: {"run_dir": handle.get("run_dir"), "report_path": None,
                                     "exit_condition": "completed"},
        assess=lambda raw, planned: {"cost": _zero_cost()},
        assess_from_disk=lambda in_flight: {"cost": _zero_cost()},
        quarantine=lambda in_flight: None,
        distill=lambda assessment: None,
        decide=lambda state, rows: {"kind": "REPRODUCED", "rule": "r1", "stop_reason": None, "next_plan": None},
        write_reports=lambda state, rows, decision: None,
        liveness_probe=lambda in_flight: True,
        emit_event=lambda name, payload: emitted.append((name, payload)),
    )
    kwargs.update(overrides)
    return CampaignStages(**kwargs)


def _campaign(tmp_path: Path, stages: CampaignStages, **overrides) -> ReproductionCampaign:
    kwargs = dict(
        run_dir=tmp_path / "run", project_id="prj_t", paper_ref="2605.15155",
        budget=_budget(), mode="unattended", driver="live", stages=stages, resume=False,
    )
    kwargs.update(overrides)
    return ReproductionCampaign(**kwargs)


# --------------------------------------------------------------------------- #
# validate_init (F15 + F4)                                                     #
# --------------------------------------------------------------------------- #


def test_validate_init_rejects_bad_profile_key_at_zero_dollars(tmp_path):
    profile_path = tmp_path / "bad_profile.json"
    profile_path.write_text(json.dumps({"TYPO_KEY_NO_PREFIX": "1"}), encoding="utf-8")
    opts = _opts(tmp_path, run_spec_path=str(profile_path))

    with pytest.raises(CampaignInitError, match="TYPO_KEY_NO_PREFIX"):
        cc._validate_init_impl(opts)

    # End-to-end through build_campaign: INIT fails at $0 -- terminal reached
    # with no intent row ever written (nothing launched, no money moved).
    campaign = build_campaign("prj_t", opts)
    outcome = campaign.run()
    assert outcome["kind"] == "EXHAUSTED"
    rows = CampaignLedger(opts.runs_root / "prj_t" / "campaign").read_rows()
    assert not [r for r in rows if r.get("status") == "launched"]


def test_validate_init_rejects_unreadable_or_invalid_profile(tmp_path):
    opts_missing = _opts(tmp_path, run_spec_path=str(tmp_path / "absent.json"))
    with pytest.raises(CampaignInitError, match="cannot read"):
        cc._validate_init_impl(opts_missing)

    bad_json = tmp_path / "invalid.json"
    bad_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(CampaignInitError, match="not valid JSON"):
        cc._validate_init_impl(_opts(tmp_path, run_spec_path=str(bad_json)))

    non_object = tmp_path / "list.json"
    non_object.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(CampaignInitError, match="JSON object"):
        cc._validate_init_impl(_opts(tmp_path, run_spec_path=str(non_object)))


def test_validate_init_notices_validator_absence(tmp_path):
    no_validator = _write_profile(tmp_path / "nv.json", validator=None)
    notices = cc._validate_init_impl(_opts(tmp_path, run_spec_path=str(no_validator)))
    assert "validator_not_configured: REPRODUCED is unreachable unattended (spec §8.2 rule 1)" in notices

    off_validator = _write_profile(tmp_path / "off.json", validator="0")
    notices_off = cc._validate_init_impl(_opts(tmp_path, run_spec_path=str(off_validator)))
    assert any(n.startswith("validator_not_configured") for n in notices_off)

    on_validator = _write_profile(tmp_path / "on.json", validator="1")
    notices_on = cc._validate_init_impl(_opts(tmp_path, run_spec_path=str(on_validator)))
    assert not any(n.startswith("validator_not_configured") for n in notices_on)


def test_validate_init_notices_missing_gpu_rate_on_cloud_billing(tmp_path):
    profile = _write_profile(tmp_path / "p.json")
    cloud = cc._validate_init_impl(
        _opts(tmp_path, run_spec_path=str(profile), billing_sandbox="gcp", gpu_usd_per_hr=None)
    )
    assert any(n.startswith("no_gpu_rate") for n in cloud)

    local = cc._validate_init_impl(_opts(tmp_path, run_spec_path=str(profile), gpu_usd_per_hr=None))
    assert not any(n.startswith("no_gpu_rate") for n in local)

    rated = cc._validate_init_impl(
        _opts(tmp_path, run_spec_path=str(profile), billing_sandbox="gcp", gpu_usd_per_hr=2.5)
    )
    assert not any(n.startswith("no_gpu_rate") for n in rated)


def test_profile_and_driver_env_keys_disjoint(tmp_path):
    real_profile_path = _REPO_ROOT / "configs" / "campaign_run_spec.json"
    profile = json.loads(real_profile_path.read_text(encoding="utf-8"))

    from backend.agents.rlm.run_spec_contract import run_spec_key_applies

    assert set(profile).isdisjoint(cc._DRIVER_OWNED_ENV_KEYS)
    assert all(run_spec_key_applies(key) for key in profile)
    assert profile["OPENRESEARCH_EXTERNAL_VALIDATOR"] == "1"
    assert profile["OPENRESEARCH_CONTEXT_MAP"] == "on"
    assert profile["OPENRESEARCH_USE_AUTHOR_REPO"] == "1"

    # INIT enforces the disjointness fail-closed on ANY profile, not just the
    # shipped one (the child's --run-spec application would clobber the
    # driver's per-attempt env).
    clobbering = _write_profile(tmp_path / "clobber.json", {"OPENRESEARCH_SEED_BEST_ATTEMPT": "1"})
    with pytest.raises(CampaignInitError, match="driver-owned"):
        cc._validate_init_impl(_opts(tmp_path, run_spec_path=str(clobbering)))


# --------------------------------------------------------------------------- #
# plan_attempt                                                                 #
# --------------------------------------------------------------------------- #


def _seeded_rows(run_dir: Path) -> list[dict]:
    return [
        {"attempt_n": 1, "status": "launched", "directives_sha256": "dsha-1", "envelope": {},
         "driver": "live", "project_id": "prj_t", "run_dir": str(run_dir), "launched_at": 1.0},
        {"attempt_n": 1, "status": "assessed", "assessment": _assessment_dict(1), "assessed_at": 2.0},
        {"attempt_n": 1, "status": "decided", "decision": {
            "kind": "CONTINUE", "rule": "continue", "stop_reason": None,
            "next_plan": {"lineage": "champion", "seed_attempt_n": 1,
                          "seed_pointer": str(run_dir / "code"), "scope_rung": 0, "width": 1},
            "champion_attempt_n": None,
        }},
    ]


def test_plan_attempt_builds_from_decided_next_plan_and_novelty_iterates_arms(tmp_path, neutral_profile):
    opts = _opts(tmp_path, run_spec_path=str(neutral_profile))
    run_dir = opts.runs_root / "prj_t"
    (run_dir / "campaign").mkdir(parents=True)
    rows = _seeded_rows(run_dir)

    planned = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=2), rows)
    assert planned["refusal"] is None
    assert planned["attempt_n"] == 2
    directives = planned["launch_payload"]
    assert directives.seed_lineage == "champion"  # from the decided next_plan
    assert directives.seed_pointer == str(run_dir / "code")
    base_fingerprint = planned["directives_sha256"]
    assert directives.fingerprint == base_fingerprint

    # A launched row from a genuinely PRIOR/different attempt (attempt_n=1)
    # already carries the base plan's fingerprint --> the novelty gate (F10)
    # iterates lineage_arms to the next arm (fresh, since attempt 1 is both
    # champion and the only seedable attempt). Attributed to attempt_n=1
    # (not 2, the attempt being planned) so the fix's own-attempt exclusion
    # (novelty is scoped to PRIOR attempts, never this attempt's own
    # write-ahead residue) does not vacuously exempt this row.
    collide = rows + [
        {"attempt_n": 1, "status": "launched", "directives_sha256": base_fingerprint, "envelope": {},
         "driver": "live", "project_id": "prj_t", "run_dir": str(run_dir), "launched_at": 3.0},
    ]
    planned2 = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=2), collide)
    assert planned2["refusal"] is None
    assert planned2["launch_payload"].seed_lineage == "fresh"
    assert planned2["directives_sha256"] != base_fingerprint


def test_plan_attempt_refuses_no_novel_plan_when_arms_exhausted(tmp_path, neutral_profile):
    # Both burned fingerprints are attributed to attempt_n=1 (a genuinely
    # PRIOR/different attempt from the attempt_n=2 being planned) -- the
    # novelty domain is prior attempts' plans, never this attempt's own
    # write-ahead residue, so a same-attempt_n row would be (correctly)
    # exempted and this exhaustion could never be provoked.
    opts = _opts(tmp_path, run_spec_path=str(neutral_profile))
    run_dir = opts.runs_root / "prj_t"
    (run_dir / "campaign").mkdir(parents=True)
    rows = _seeded_rows(run_dir)

    champion_fp = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=2), rows)[
        "directives_sha256"
    ]
    with_champion = rows + [
        {"attempt_n": 1, "status": "launched", "directives_sha256": champion_fp, "envelope": {},
         "driver": "live", "project_id": "prj_t", "run_dir": str(run_dir), "launched_at": 3.0},
    ]
    fresh_fp = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=2), with_champion)[
        "directives_sha256"
    ]
    both_burned = with_champion + [
        {"attempt_n": 1, "status": "launched", "directives_sha256": fresh_fp, "envelope": {},
         "driver": "live", "project_id": "prj_t", "run_dir": str(run_dir), "launched_at": 4.0},
    ]

    refused = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=2), both_burned)
    assert refused == {"refusal": "no_novel_plan", "downgrade_to_checkpoint": False}


def _orphan_relaunch_opts(tmp_path: Path) -> CampaignOptions:
    return _opts(
        tmp_path,
        run_spec_path=str(_REPO_ROOT / "configs" / "campaign_run_spec.json"),
        driver="live",
        billing_sandbox="local",
        gpu_usd_per_hr=0.5,
        max_llm_usd=_tiny_budget()["max_llm_usd"],
        max_gpu_usd=_tiny_budget()["max_gpu_usd"],
        max_gpu_hours=_tiny_budget()["max_gpu_hours"],
        max_attempts=_tiny_budget()["max_attempts"],
    )


def test_plan_attempt_orphaned_intent_relaunch_reuses_same_fingerprint(tmp_path):
    """Controller-found bug fix: the spec-§5-sanctioned orphaned-intent resume
    (crash between the write-ahead intent append and the in_flight write ->
    ``_resume_orphaned_intent`` -> ``_loop(allow_supersede=True)`` ->
    plan_attempt for the SAME attempt_n) must not collide with its own
    just-written intent row. Durable, through the real ``CampaignLedger`` --
    not a monkeypatched plan stage."""
    opts = _orphan_relaunch_opts(tmp_path)
    run_dir = opts.runs_root / "prj_t"
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)
    ledger = CampaignLedger(campaign_dir)
    state = _state(next_attempt_n=1, budget=_tiny_budget())

    # The pre-crash write-ahead intent (spec §5 step 2a): plan attempt 1 from
    # a clean ledger and durably record it as "launched" -- then the process
    # dies before the in_flight write (step 2b), leaving this row orphaned
    # (no "assessed" row ever follows it).
    orphaned_fingerprint = cc._plan_attempt_impl(run_dir, opts, "prj_t", state, [])["directives_sha256"]
    ledger.append_row({
        "attempt_n": 1, "status": "launched", "directives_sha256": orphaned_fingerprint,
        "envelope": {}, "driver": "live", "project_id": "prj_t",
        "run_dir": str(run_dir), "launched_at": 1.0,
    })
    ledger.write_state(state)

    # Resume: reload durable state + rows exactly as
    # ``_resume_orphaned_intent`` -> ``_loop(allow_supersede=True)`` would,
    # and replan the SAME attempt_n=1.
    resumed_state = ledger.load_state()
    rows = ledger.read_rows()
    replanned = cc._plan_attempt_impl(run_dir, opts, "prj_t", resumed_state, rows)

    assert replanned["refusal"] is None
    assert replanned["attempt_n"] == 1
    # Same plan, legitimately re-armed -- not walked to a different arm.
    assert replanned["directives_sha256"] == orphaned_fingerprint


def test_plan_attempt_prior_attempt_fingerprint_still_collides(tmp_path):
    """Inverse pin: the fix narrowly exempts only THIS attempt's own
    write-ahead residue. A genuinely PRIOR attempt's (attempt_n=1)
    fingerprint sitting in the rows still collides when planning a
    DIFFERENT attempt_n (2) -- the novelty gate still walks arms / refuses
    when none remain, so the fix did not over-widen the exemption."""
    opts = _orphan_relaunch_opts(tmp_path)
    run_dir = opts.runs_root / "prj_t"
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)
    ledger = CampaignLedger(campaign_dir)
    budget = _tiny_budget()

    attempt1_fingerprint = cc._plan_attempt_impl(
        run_dir, opts, "prj_t", _state(next_attempt_n=1, budget=budget), []
    )["directives_sha256"]
    ledger.append_row({
        "attempt_n": 1, "status": "launched", "directives_sha256": attempt1_fingerprint,
        "envelope": {}, "driver": "live", "project_id": "prj_t",
        "run_dir": str(run_dir), "launched_at": 1.0,
    })
    ledger.write_state(_state(next_attempt_n=1, budget=budget))
    rows = ledger.read_rows()

    # No "assessed"/"decided" row exists for attempt 1, so attempt 2's base
    # plan is ALSO the bare default (lineage="fresh", scope_rung=0) over an
    # unchanged envelope (no assessments => attempts_completed stays 0) --
    # genuinely, deterministically the same fingerprint as attempt 1's. With
    # no assessments, lineage_arms has only the single fresh arm, so there is
    # no alternative left to walk to.
    planned2 = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=2, budget=budget), rows)

    assert planned2 == {"refusal": "no_novel_plan", "downgrade_to_checkpoint": False}


def test_plan_attempt_refuses_unenforceable_downgrade_checkpoint(tmp_path):
    # Cloud billing sandbox + no known $/hr rate + no gpu_plan.json -->
    # gpu_usd is unenforceable --> fail-closed refusal that downgrades to
    # checkpoint mode instead of launching unattended (spec §10.1).
    opts = _opts(tmp_path, billing_sandbox="gcp", gpu_usd_per_hr=None)
    run_dir = opts.runs_root / "prj_t"

    planned = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(), [])
    assert planned["downgrade_to_checkpoint"] is True
    assert planned["refusal"].startswith("unenforceable:")
    assert "gpu_usd_unenforceable:no_rate" in planned["refusal"]


def test_plan_attempt_budget_floor_refusal(tmp_path):
    opts = _opts(tmp_path)
    run_dir = opts.runs_root / "prj_t"
    state = _state(spent={"llm_usd": 99.5, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 0.0})

    planned = cc._plan_attempt_impl(run_dir, opts, "prj_t", state, [])
    assert planned == {"refusal": "budget_floor:llm_usd", "downgrade_to_checkpoint": False}


def test_seed_hint_only_latest_attempt(tmp_path):
    # Documented v1 limitation: only the LATEST assessed attempt is seedable
    # by pointer (its code/ is still live at run_dir/code; older attempts'
    # trees are archived by the next launch's force-quarantine).
    opts = _opts(tmp_path)
    run_dir = opts.runs_root / "prj_t"
    rows = [
        {"attempt_n": 1, "status": "assessed", "assessment": _assessment_dict(1), "assessed_at": 1.0},
        {"attempt_n": 2, "status": "assessed", "assessment": _assessment_dict(2), "assessed_at": 2.0},
    ]

    _assessments, _lineage, _rungs, hint = cc._gather_campaign_view(run_dir, rows)
    assert hint == {2: str(run_dir / "code")}


# --------------------------------------------------------------------------- #
# assess: pin-at-first-sight                                                   #
# --------------------------------------------------------------------------- #


def test_assess_closure_pins_rubric_at_first_sight(tmp_path):
    opts = _opts(tmp_path)
    run_dir = opts.runs_root / "prj_t"
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)
    tree = {"leaves": [{"id": "l1", "weight": 1.0}]}
    (run_dir / "generated_rubric.json").write_text(json.dumps(tree), encoding="utf-8")

    planned = {"attempt_n": 1, "directives_sha256": "d1", "project_id": "prj_t"}
    raw = {"run_dir": str(run_dir), "report_path": None, "exit_condition": "completed"}

    # No campaign.json yet (pin is None) --> the assess closure computes the
    # observed hash and stashes it for the machine to adopt.
    result = cc._assess_impl(opts, "prj_t", campaign_dir, raw, planned)
    assert result["rubric_sha256_observed"] == rubric_sha256(tree)
    assert result["rubric_sha256_ok"] is None  # no pin existed for THIS assessment

    # Once a pin exists, the observed key is never emitted again.
    CampaignLedger(campaign_dir).write_state(_state(rubric_sha256=rubric_sha256(tree)))
    pinned_result = cc._assess_impl(opts, "prj_t", campaign_dir, raw, planned)
    assert "rubric_sha256_observed" not in pinned_result
    assert pinned_result["rubric_sha256_ok"] is True

    # Machine adoption: an assessment carrying the observed key lands in
    # campaign.json via the state machine's very next write_state.
    emitted: list = []
    stages = _scripted_stages(
        tmp_path / "machine_run",
        emitted,
        assess=lambda _raw, _planned: {"cost": _zero_cost(), "rubric_sha256_observed": "abc123"},
    )
    campaign = _campaign(tmp_path, stages, run_dir=tmp_path / "machine_run")
    outcome = campaign.run()
    assert outcome["kind"] == "REPRODUCED"
    persisted = json.loads((tmp_path / "machine_run" / "campaign" / "campaign.json").read_text(encoding="utf-8"))
    assert persisted["rubric_sha256"] == "abc123"


# --------------------------------------------------------------------------- #
# distill                                                                      #
# --------------------------------------------------------------------------- #


def test_distill_invokes_miners_fail_soft(tmp_path, monkeypatch, caplog):
    opts = _opts(tmp_path)
    run_dir = opts.runs_root / "prj_t"
    run_dir.mkdir(parents=True)
    (run_dir / "experiment_runs.jsonl").write_text(
        json.dumps({"success": False, "error": "boom"}) + "\n", encoding="utf-8"
    )

    calls: list = []

    def _raising_mine(project_dir, runs_root, arxiv_id):
        raise RuntimeError("miner exploded")

    def _fake_admit(project_dir, runs_root, *, report, validator_verdict, paper_class):
        calls.append(("admit_recipe", paper_class, report))

    def _fake_attribute(row, *, arxiv_id=None, evidence_refs=()):
        calls.append(("attribute_failure", dict(row), arxiv_id))
        return "attribution-sentinel"

    class _FakeMemory:
        def __init__(self, runs_root):
            self.runs_root = runs_root

        def record(self, attribution, *, arxiv_id=None, hint=""):
            calls.append(("memory_record", attribution, arxiv_id))

    monkeypatch.setattr(cc.lesson_distiller, "mine_lessons", _raising_mine)
    monkeypatch.setattr(cc, "admit_recipe", _fake_admit)
    monkeypatch.setattr(cc.failure_attribution, "attribute_failure", _fake_attribute)
    monkeypatch.setattr(cc, "ExperienceMemory", _FakeMemory)

    with caplog.at_level(logging.WARNING, logger="backend.agents.rlm.campaign_composition"):
        cc._distill_impl(run_dir, opts, {"attempt_n": 1})

    assert any("mine_lessons failed" in record.message for record in caplog.records)
    names = [c[0] for c in calls]
    assert names == ["admit_recipe", "attribute_failure", "memory_record"]
    assert calls[1][1] == {"success": False, "error": "boom"}
    assert calls[2][1] == "attribution-sentinel"
    assert calls[2][2] == "2605.15155"


# --------------------------------------------------------------------------- #
# emit_event / poll_steering                                                   #
# --------------------------------------------------------------------------- #


def test_emit_event_appends_dashboard_row(tmp_path):
    run_dir = tmp_path / "run"
    cc._emit_event_impl(run_dir, "campaign_started", {"project_id": "prj_t"})
    cc._emit_event_impl(run_dir, "attempt_started", {"attempt_n": 1})

    lines = (run_dir / "dashboard_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "campaign_started"
    assert first["project_id"] == "prj_t"
    assert "timestamp" in first
    assert json.loads(lines[1])["event"] == "attempt_started"


def test_poll_steering_set_mode_advances_cursor(tmp_path):
    run_dir = tmp_path / "run"
    channel = run_dir / "campaign" / "user_messages.jsonl"
    channel.parent.mkdir(parents=True)
    channel.write_text(
        json.dumps({"id": "1", "ts": "t", "op": "set_mode", "mode": "checkpoint"}) + "\n"
        + "THIS IS NOT JSON\n"
        + json.dumps({"id": "2", "ts": "t", "op": "note", "content": "hello"}) + "\n",
        encoding="utf-8",
    )

    state = _state(mode="unattended", steering_cursor=0)
    cc._poll_steering_impl(run_dir, state)
    assert state.mode == "checkpoint"
    assert state.steering_cursor == 3  # malformed + note rows consumed too

    # Only rows after the cursor are consumed on the next poll.
    with channel.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "3", "ts": "t", "op": "set_mode", "mode": "unattended"}) + "\n")
    cc._poll_steering_impl(run_dir, state)
    assert state.mode == "unattended"
    assert state.steering_cursor == 4

    # No channel file at all is a no-op.
    fresh = _state()
    cc._poll_steering_impl(tmp_path / "elsewhere", fresh)
    assert fresh.steering_cursor == 0


# --------------------------------------------------------------------------- #
# write_reports                                                                #
# --------------------------------------------------------------------------- #


def test_write_reports_plan_only_on_infeasible(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    decision = {
        "kind": "INFEASIBLE", "rule": "blocking_gap",
        "stop_reason": "infeasible:asset:gated:weights", "champion_attempt_n": None,
    }

    cc._write_reports_impl(run_dir, _state(terminal=dict(decision)), [], decision)

    assert (run_dir / "campaign_report.md").exists()
    report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "plan_only"
    assert report["stop_reason"] == "infeasible:asset:gated:weights"
    assert report["what_would_unblock"] == ["infeasible:asset:gated:weights"]
    assert (run_dir / "final_report.md").exists()


def test_write_reports_plan_only_when_no_assessed_rows_regardless_of_kind(tmp_path):
    # F14's guard is an OR: kind == INFEASIBLE, OR no attempt was ever
    # assessed (e.g. an INIT/plan-refusal early exit). This isolates the
    # second arm -- kind is EXHAUSTED (not INFEASIBLE) but rows is empty --
    # which test_write_reports_plan_only_on_infeasible does not cover.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    decision = {
        "kind": "EXHAUSTED", "rule": "init",
        "stop_reason": "campaign_error:CampaignInitError: missing paper_ref",
        "champion_attempt_n": None,
    }

    cc._write_reports_impl(run_dir, _state(terminal=dict(decision)), [], decision)

    assert (run_dir / "campaign_report.md").exists()
    report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "plan_only"
    assert report["stop_reason"] == decision["stop_reason"]
    assert report["what_would_unblock"] == [decision["stop_reason"]]
    assert (run_dir / "final_report.md").exists()
    # No champion copy either -- nothing was ever assessed to be a champion.
    assert not (run_dir / "campaign" / "champion_final_report.json").exists()


def test_write_reports_champion_copy_on_terminal(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "final_report.json").write_text(json.dumps({"verdict": "partial", "n": 2}), encoding="utf-8")
    (run_dir / "final_report.md").write_text("# report for attempt 2\n", encoding="utf-8")
    rows = [
        {"attempt_n": 1, "status": "assessed", "assessment": _assessment_dict(1), "assessed_at": 1.0},
        {"attempt_n": 2, "status": "assessed", "assessment": _assessment_dict(2), "assessed_at": 2.0},
    ]
    decision = {"kind": "EXHAUSTED", "rule": "max_attempts", "stop_reason": "max_attempts",
                "champion_attempt_n": 2}

    cc._write_reports_impl(run_dir, _state(terminal=dict(decision)), rows, decision)

    champion_json = run_dir / "campaign" / "champion_final_report.json"
    champion_md = run_dir / "campaign" / "champion_final_report.md"
    assert json.loads(champion_json.read_text(encoding="utf-8")) == {"verdict": "partial", "n": 2}
    assert champion_md.read_text(encoding="utf-8") == "# report for attempt 2\n"
    # A real attempt produced final_report.json --> no plan-only overwrite.
    assert json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))["verdict"] == "partial"


def test_write_reports_no_champion_copy_when_champion_not_latest(tmp_path):
    # v1 limitation, honestly held: only the LATEST assessed attempt's report
    # is still live at run_dir top-level; an older champion's report is
    # already archived, so nothing is copied (never copy the WRONG report).
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "final_report.json").write_text(json.dumps({"n": 2}), encoding="utf-8")
    rows = [
        {"attempt_n": 1, "status": "assessed", "assessment": _assessment_dict(1), "assessed_at": 1.0},
        {"attempt_n": 2, "status": "assessed", "assessment": _assessment_dict(2), "assessed_at": 2.0},
    ]
    decision = {"kind": "EXHAUSTED", "rule": "max_attempts", "stop_reason": "max_attempts",
                "champion_attempt_n": 1}

    cc._write_reports_impl(run_dir, _state(terminal=dict(decision)), rows, decision)

    assert not (run_dir / "campaign" / "champion_final_report.json").exists()


# --------------------------------------------------------------------------- #
# State-machine additions (emit points + steering), via the REAL campaign      #
# --------------------------------------------------------------------------- #


def test_emit_points_fire_in_order(tmp_path):
    emitted: list = []
    campaign = _campaign(tmp_path, _scripted_stages(tmp_path / "run", emitted))

    outcome = campaign.run()

    assert outcome["kind"] == "REPRODUCED"
    assert [name for name, _payload in emitted] == [
        "campaign_started", "attempt_started", "attempt_assessed", "campaign_decision", "campaign_terminal",
    ]
    by_name = dict(emitted)
    assert by_name["campaign_started"]["project_id"] == "prj_t"
    assert by_name["campaign_started"]["budget"] == _budget()
    assert by_name["attempt_started"]["attempt_n"] == 1
    assert by_name["attempt_started"]["directives_sha256"] == "d-1"
    assert by_name["attempt_assessed"]["assessment"] == {"cost": _zero_cost()}
    assert by_name["campaign_decision"]["kind"] == "REPRODUCED"
    assert by_name["campaign_decision"]["fingerprint"] == "d-1"  # the launched row's sha
    assert by_name["campaign_terminal"]["kind"] == "REPRODUCED"


def test_attempt_started_emit_precedes_launch(tmp_path):
    # attempt_started must be observable (dashboard/operator) BEFORE the
    # (possibly slow/expensive) stages.launch call fires, not after.
    order: list = []

    def _launch(payload):
        order.append(("launch", payload["attempt_n"]))
        n = payload["attempt_n"]
        return {"pid": None, "run_dir": str(tmp_path / "run" / f"a{n}"), "lease_ref": None}

    def _emit_event(name, payload):
        if name == "attempt_started":
            order.append(("attempt_started", payload["attempt_n"]))

    stages = _scripted_stages(tmp_path / "run", [], launch=_launch, emit_event=_emit_event)
    outcome = _campaign(tmp_path, stages).run()

    assert outcome["kind"] == "REPRODUCED"
    assert order == [("attempt_started", 1), ("launch", 1)]


def test_emit_failure_is_fail_soft(tmp_path):
    def _broken_emit(name, payload):
        raise OSError("emitter down")

    campaign = _campaign(tmp_path, _scripted_stages(tmp_path / "run", [], emit_event=_broken_emit))
    outcome = campaign.run()

    assert outcome["kind"] == "REPRODUCED"  # a broken emitter never fails the campaign
    persisted = json.loads((tmp_path / "run" / "campaign" / "campaign.json").read_text(encoding="utf-8"))
    assert any(w.startswith("emit_failed:campaign_started") for w in persisted["warnings"])


def test_poll_steering_called_before_plan(tmp_path):
    order: list = []
    emitted: list = []

    def _poll(state):
        order.append("poll")
        state.mode = "checkpoint"  # mutation must be visible to plan_attempt

    def _plan(state, rows):
        order.append(f"plan:mode={state.mode}")
        n = state.next_attempt_n
        return {
            "attempt_n": n, "directives_sha256": f"d-{n}",
            "envelope": {}, "project_id": state.project_id, "run_dir": str(tmp_path / "run" / f"a{n}"),
            "refusal": None, "downgrade_to_checkpoint": False, "launch_payload": {"attempt_n": n},
        }

    stages = _scripted_stages(tmp_path / "run", emitted, plan_attempt=_plan, poll_steering=_poll)
    outcome = _campaign(tmp_path, stages).run()

    assert outcome["kind"] == "REPRODUCED"
    assert order == ["poll", "plan:mode=checkpoint"]


def test_poll_steering_exception_is_fail_soft(tmp_path):
    def _broken_poll(state):
        raise RuntimeError("steering io error")

    stages = _scripted_stages(tmp_path / "run", [], poll_steering=_broken_poll)
    outcome = _campaign(tmp_path, stages).run()

    assert outcome["kind"] == "REPRODUCED"
    persisted = json.loads((tmp_path / "run" / "campaign" / "campaign.json").read_text(encoding="utf-8"))
    assert any(w.startswith("poll_steering_failed:RuntimeError") for w in persisted["warnings"])


# --------------------------------------------------------------------------- #
# build_campaign surface                                                       #
# --------------------------------------------------------------------------- #


def test_build_campaign_rejects_unknown_driver(tmp_path):
    with pytest.raises(ValueError, match="unknown campaign driver"):
        build_campaign("prj_t", _opts(tmp_path, driver="bogus"))


def test_build_campaign_wires_runnable_campaign(tmp_path):
    opts = _opts(tmp_path)
    campaign = build_campaign("prj_t", opts)

    assert isinstance(campaign, ReproductionCampaign)
    assert campaign.project_id == "prj_t"
    assert campaign.run_dir == opts.runs_root / "prj_t"
    assert campaign.budget == _budget()
    assert campaign.stages.poll_steering is not None


def test_build_driver_selects_live(tmp_path):
    driver = cc._build_driver(_opts(tmp_path, driver="live"))
    assert isinstance(driver, LiveCliDriver)


def test_build_driver_selects_unified(tmp_path):
    driver = cc._build_driver(_opts(tmp_path, driver="unified"))
    assert isinstance(driver, UnifiedRunDriver)


def test_build_driver_selects_paired(tmp_path):
    driver = cc._build_driver(_opts(tmp_path, driver="paired"))
    assert isinstance(driver, PairedDriver)
    # F8: constructed with the operator-ack already granted (reached only
    # via an explicit --campaign-driver paired CLI selection).
    assert isinstance(driver._live, LiveCliDriver)  # noqa: SLF001 - white-box construction check
    assert isinstance(driver._unified, UnifiedRunDriver)  # noqa: SLF001


def test_build_campaign_constructs_unified_and_paired_without_raising(tmp_path):
    unified_campaign = build_campaign("prj_u", _opts(tmp_path, driver="unified"))
    assert isinstance(unified_campaign, ReproductionCampaign)
    assert unified_campaign.driver == "unified"

    paired_campaign = build_campaign("prj_p", _opts(tmp_path, driver="paired"))
    assert isinstance(paired_campaign, ReproductionCampaign)
    assert paired_campaign.driver == "paired"


# --------------------------------------------------------------------------- #
# _apply_profile_env (2026-07-02): the profile's OPENRESEARCH_* keys land in  #
# the CAMPAIGN process's own env, not just the launched child's --run-spec,  #
# so campaign-side readers (DISTILL's lesson_distiller/ExperienceMemory,     #
# any flag-gated helper called in-process) see the same flags.               #
# --------------------------------------------------------------------------- #


def test_apply_profile_env_fills_missing_key(tmp_path):
    profile = _write_profile(tmp_path / "profile_env.json", extra={"OPENRESEARCH_DOOMED_KILL": "1"})
    opts = _opts(tmp_path, run_spec_path=str(profile))
    env: dict[str, str] = {}

    applied = cc._apply_profile_env(opts, env=env)

    assert applied >= 1
    assert env["OPENRESEARCH_DOOMED_KILL"] == "1"
    assert env["OPENRESEARCH_REUSE_RUBRIC"] == "1"  # from the default _write_profile keys


def test_apply_profile_env_never_clobbers_existing_key(tmp_path):
    profile = _write_profile(tmp_path / "profile_env.json", extra={"OPENRESEARCH_DOOMED_KILL": "1"})
    opts = _opts(tmp_path, run_spec_path=str(profile))
    env = {"OPENRESEARCH_DOOMED_KILL": "operator_value"}

    applied = cc._apply_profile_env(opts, env=env)

    assert env["OPENRESEARCH_DOOMED_KILL"] == "operator_value"  # explicit operator env wins
    assert env["OPENRESEARCH_REUSE_RUBRIC"] == "1"  # unrelated key still fills in
    assert applied == 2  # REUSE_RUBRIC + EXTERNAL_VALIDATOR; DOOMED_KILL skipped (pre-existing)


def test_apply_profile_env_skips_only_the_preexisting_key(tmp_path):
    """Precise accounting: with one key already set, every OTHER profile key
    still applies -- the guard is per-key, not all-or-nothing."""
    profile = _write_profile(tmp_path / "profile_env.json", extra={"OPENRESEARCH_DOOMED_KILL": "1"})
    opts = _opts(tmp_path, run_spec_path=str(profile))
    env = {"OPENRESEARCH_DOOMED_KILL": "operator_value"}

    cc._apply_profile_env(opts, env=env)

    assert env["OPENRESEARCH_DOOMED_KILL"] == "operator_value"
    assert env["OPENRESEARCH_REUSE_RUBRIC"] == "1"
    assert env["OPENRESEARCH_EXTERNAL_VALIDATOR"] == "1"


def test_apply_profile_env_fail_soft_on_missing_profile(tmp_path):
    opts = _opts(tmp_path, run_spec_path=str(tmp_path / "absent.json"))
    env: dict[str, str] = {}
    assert cc._apply_profile_env(opts, env=env) == 0
    assert env == {}


def test_apply_profile_env_fail_soft_on_bad_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    opts = _opts(tmp_path, run_spec_path=str(bad))
    env: dict[str, str] = {}
    assert cc._apply_profile_env(opts, env=env) == 0


def test_apply_profile_env_driver_owned_key_is_inits_job_not_this_helpers(tmp_path):
    """A profile carrying a driver-owned key is invalid, but that's enforced
    exclusively by _validate_init_impl (F15) -- _apply_profile_env has no
    special-case for it and simply applies whatever apply_run_spec resolves.
    No interaction between the two: build_campaign always calls this helper,
    the invalid campaign is refused later when the INIT stage actually runs
    (validate_init is a lazily-invoked stage callback, not called here)."""
    profile = _write_profile(
        tmp_path / "profile_env.json",
        extra={"OPENRESEARCH_BASELINE_EXTRA_GUIDANCE": "should be caught by INIT, not here"},
    )
    opts = _opts(tmp_path, run_spec_path=str(profile))
    env: dict[str, str] = {}

    cc._apply_profile_env(opts, env=env)
    assert env.get("OPENRESEARCH_BASELINE_EXTRA_GUIDANCE") == "should be caught by INIT, not here"

    with pytest.raises(CampaignInitError, match="driver-owned"):
        cc._validate_init_impl(opts)


def test_build_campaign_applies_profile_env_to_real_os_environ(tmp_path, monkeypatch):
    """Integration: build_campaign wires _apply_profile_env against the real
    process os.environ (no explicit env= override)."""
    monkeypatch.delenv("OPENRESEARCH_DOOMED_KILL", raising=False)
    profile = _write_profile(tmp_path / "profile_env.json", extra={"OPENRESEARCH_DOOMED_KILL": "1"})
    opts = _opts(tmp_path, run_spec_path=str(profile))

    build_campaign("prj_env_fill", opts)

    assert os.environ["OPENRESEARCH_DOOMED_KILL"] == "1"


def test_build_campaign_does_not_clobber_operator_os_environ(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_DOOMED_KILL", "operator_value")
    profile = _write_profile(tmp_path / "profile_env.json", extra={"OPENRESEARCH_DOOMED_KILL": "1"})
    opts = _opts(tmp_path, run_spec_path=str(profile))

    build_campaign("prj_env_noclobber", opts)

    assert os.environ["OPENRESEARCH_DOOMED_KILL"] == "operator_value"


# --------------------------------------------------------------------------- #
# _decide_impl budget-floor lookahead (2026-07-02)                            #
# --------------------------------------------------------------------------- #


def test_decide_impl_gpu_usd_estimate_scales_by_gpu_count(tmp_path, monkeypatch):
    """``ctx.gpu_usd_per_hr`` (here from the operator's ``--gpu-usd-per-hr``
    override) is the PER-GPU $/hr -- the SAME value ``check_enforceability``
    already multiplies by ``ctx.max_gpu_count`` to get a total-$ bound
    (campaign_policy.py:279). The next-attempt gpu_usd lookahead must use the
    identical convention: a 4-GPU plan at $5.25/GPU/hr for a 2.0h estimate is
    $42, not $10.50 (the pre-fix undercount)."""
    run_dir = tmp_path / "runs" / "prj_t"
    (run_dir / "rlm_state").mkdir(parents=True)
    (run_dir / "rlm_state" / "gpu_plan.json").write_text(
        json.dumps({"gpu_count": 4}), encoding="utf-8"
    )
    opts = _opts(tmp_path, est_gpu_hours=2.0, gpu_usd_per_hr=5.25)
    state = _state()

    real_attempt_estimate = cc.attempt_estimate
    captured: dict = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return real_attempt_estimate(**kwargs)

    monkeypatch.setattr(cc, "attempt_estimate", _spy)

    cc._decide_impl(run_dir, opts, state, rows=[])

    assert captured["ctx"].max_gpu_count == 4
    assert captured["est_usd"] == pytest.approx(2.0 * 5.25 * 4)


def test_decide_impl_gpu_usd_estimate_single_gpu_unchanged(tmp_path, monkeypatch):
    """A 1-GPU plan (the common case) is byte-identical to the pre-fix math."""
    run_dir = tmp_path / "runs" / "prj_t"
    (run_dir / "rlm_state").mkdir(parents=True)
    (run_dir / "rlm_state" / "gpu_plan.json").write_text(
        json.dumps({"gpu_count": 1}), encoding="utf-8"
    )
    opts = _opts(tmp_path, est_gpu_hours=2.0, gpu_usd_per_hr=0.5)
    state = _state()

    real_attempt_estimate = cc.attempt_estimate
    captured: dict = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return real_attempt_estimate(**kwargs)

    monkeypatch.setattr(cc, "attempt_estimate", _spy)

    cc._decide_impl(run_dir, opts, state, rows=[])

    assert captured["est_usd"] == pytest.approx(2.0 * 0.5)


# --------------------------------------------------------------------------- #
# Width minting (spec §8.3, F16)                                              #
# --------------------------------------------------------------------------- #


def test_mint_width_project_ids():
    assert cc.mint_width_project_ids("prj_t", 3) == ["prj_t_w1", "prj_t_w2", "prj_t_w3"]
    assert cc.mint_width_project_ids("prj_t", 1) == ["prj_t_w1"]
    assert cc.mint_width_project_ids("prj_t", 0) == []


def _width_decided_row(
    attempt_n: int, run_dir: Path, *, width: int, project_id: str = "prj_t", seed_pointer: str | None = None
) -> dict:
    return {
        "attempt_n": attempt_n, "status": "decided", "decision": {
            "kind": "CONTINUE", "rule": "continue", "stop_reason": None,
            "next_plan": {"lineage": "fresh" if seed_pointer is None else "champion", "seed_attempt_n": None,
                          "seed_pointer": seed_pointer, "scope_rung": 0, "width": width},
            "champion_attempt_n": None,
        },
    }


def test_plan_attempt_width_override_mints_first_child_project_id(tmp_path, neutral_profile):
    opts = _opts(tmp_path, run_spec_path=str(neutral_profile), width=2)
    run_dir = opts.runs_root / "prj_t"
    (run_dir / "campaign").mkdir(parents=True)
    rows = [
        {"attempt_n": 1, "status": "launched", "directives_sha256": "dsha-1", "envelope": {},
         "driver": "live", "project_id": "prj_t", "run_dir": str(run_dir), "launched_at": 1.0},
        {"attempt_n": 1, "status": "assessed", "assessment": _assessment_dict(1), "assessed_at": 2.0},
        _width_decided_row(1, run_dir, width=2),
    ]

    planned = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=2), rows)

    assert planned["refusal"] is None
    assert planned["project_id"] == "prj_t_w1"
    assert planned["run_dir"] == str(opts.runs_root / "prj_t_w1")
    directives = planned["launch_payload"]
    assert directives.project_id == "prj_t_w1"
    assert "[width] candidate angle 1/2: vary the implementation approach." in directives.extra_guidance

    # Re-persisted directives/<n>.json on disk carries the augmented guidance.
    persisted = json.loads((run_dir / "campaign" / "directives" / "2.json").read_text(encoding="utf-8"))
    assert "[width] candidate angle 1/2" in persisted["extra_guidance"]
    assert persisted["project_id"] == "prj_t_w1"


def test_plan_attempt_width_override_increments_across_attempts(tmp_path, neutral_profile):
    opts = _opts(tmp_path, run_spec_path=str(neutral_profile), width=2)
    run_dir = opts.runs_root / "prj_t"
    (run_dir / "campaign").mkdir(parents=True)
    rows = [
        {"attempt_n": 1, "status": "launched", "directives_sha256": "dsha-1", "envelope": {},
         "driver": "live", "project_id": "prj_t", "run_dir": str(run_dir), "launched_at": 1.0},
        {"attempt_n": 1, "status": "assessed", "assessment": _assessment_dict(1), "assessed_at": 2.0},
        _width_decided_row(1, run_dir, width=2),
        # Attempt 2 already launched under the first minted width child.
        {"attempt_n": 2, "status": "launched", "directives_sha256": "dsha-2", "envelope": {},
         "driver": "live", "project_id": "prj_t_w1", "run_dir": str(run_dir), "launched_at": 3.0},
        {"attempt_n": 2, "status": "assessed", "assessment": _assessment_dict(2), "assessed_at": 4.0},
        _width_decided_row(2, run_dir, width=2),
    ]

    planned = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=3), rows)

    assert planned["project_id"] == "prj_t_w2"
    assert planned["run_dir"] == str(opts.runs_root / "prj_t_w2")
    assert "[width] candidate angle 2/2: vary the implementation approach." in planned["launch_payload"].extra_guidance


def test_plan_attempt_width_reoverride_on_orphaned_replan_is_stable(tmp_path, neutral_profile):
    """A re-plan of the SAME attempt_n (orphaned-intent-relaunch, spec §5)
    must recompute the identical child id -- the current attempt's own
    prior 'launched' row must not inflate its own width counter."""
    opts = _opts(tmp_path, run_spec_path=str(neutral_profile), width=2)
    run_dir = opts.runs_root / "prj_t"
    (run_dir / "campaign").mkdir(parents=True)
    base_rows = [
        {"attempt_n": 1, "status": "launched", "directives_sha256": "dsha-1", "envelope": {},
         "driver": "live", "project_id": "prj_t", "run_dir": str(run_dir), "launched_at": 1.0},
        {"attempt_n": 1, "status": "assessed", "assessment": _assessment_dict(1), "assessed_at": 2.0},
        _width_decided_row(1, run_dir, width=2),
    ]

    first = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=2), base_rows)
    assert first["project_id"] == "prj_t_w1"

    # Simulate the orphaned write-ahead intent for THIS SAME attempt_n=2.
    with_own_intent = base_rows + [
        {"attempt_n": 2, "status": "launched", "directives_sha256": first["directives_sha256"], "envelope": {},
         "driver": "live", "project_id": "prj_t_w1", "run_dir": str(run_dir), "launched_at": 5.0},
    ]
    replanned = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=2), with_own_intent)

    assert replanned["project_id"] == "prj_t_w1"  # NOT prj_t_w2


def test_plan_attempt_width_one_is_byte_identical(tmp_path, neutral_profile):
    """width<=1 (the default) never touches project_id/run_dir/extra_guidance."""
    opts = _opts(tmp_path, run_spec_path=str(neutral_profile), width=1)
    run_dir = opts.runs_root / "prj_t"
    (run_dir / "campaign").mkdir(parents=True)

    planned = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=1), [])

    assert planned["project_id"] == "prj_t"
    assert planned["run_dir"] == str(run_dir)
    assert "[width]" not in planned["launch_payload"].extra_guidance


def test_plan_attempt_width_child_seed_marker_and_quarantine_target_child_dir(tmp_path, neutral_profile):
    """Integration: LiveCliDriver.launch (which derives run_dir purely from
    directives.project_id) force-quarantines and stages the seed marker
    under the CHILD run dir, never the campaign's own top-level dir."""
    from backend.agents.rlm.attempt_driver import LiveCliDriver as _LiveCliDriver

    opts = _opts(tmp_path, run_spec_path=str(neutral_profile), width=2)
    run_dir = opts.runs_root / "prj_t"
    (run_dir / "campaign").mkdir(parents=True)
    seed_code_dir = str(tmp_path / "champion_code")
    rows = [
        {"attempt_n": 1, "status": "launched", "directives_sha256": "dsha-1", "envelope": {},
         "driver": "live", "project_id": "prj_t", "run_dir": str(run_dir), "launched_at": 1.0},
        {"attempt_n": 1, "status": "assessed", "assessment": _assessment_dict(1), "assessed_at": 2.0},
        _width_decided_row(1, run_dir, width=2, seed_pointer=seed_code_dir),
    ]
    planned = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=2), rows)
    assert planned["launch_payload"].seed_pointer == seed_code_dir  # sanity: this plan really carries a seed

    # Pre-seed residue under what would be the WRONG (top-level) dir, to
    # prove the driver never quarantines it: the child id is a distinct dir.
    (run_dir / "code").mkdir(parents=True, exist_ok=True)
    (run_dir / "code" / "train.py").write_text("# top-level residue, must survive")

    def _fake_popen(argv, **kwargs):
        class _P:
            pid = 4242
        return _P()

    driver = _LiveCliDriver(runs_root=opts.runs_root, repo_root=opts.repo_root, popen=_fake_popen)
    handle = driver.launch(planned["launch_payload"])

    child_dir = opts.runs_root / "prj_t_w1"
    assert handle.run_dir == str(child_dir)

    # Seed marker landed under the CHILD dir, carrying the seed pointer.
    marker = json.loads((child_dir / "campaign" / "seed_staging.json").read_text(encoding="utf-8"))
    assert marker["source_code_dir"] == seed_code_dir
    assert not (run_dir / "campaign" / "seed_staging.json").exists()  # never at the top-level dir

    # The top-level campaign dir's residue is untouched -- quarantine targeted the child.
    assert (run_dir / "code" / "train.py").exists()


# --------------------------------------------------------------------------- #
# --paper-hint passthrough                                                     #
# --------------------------------------------------------------------------- #


def test_enforcement_mapping_appends_paper_hint_to_cli_args():
    from backend.agents.rlm.campaign_policy import EnforcementPlan

    plan = EnforcementPlan(
        cli_args=(("--max-usd", "5.0"),), env={}, effective_wall_s=100.0,
        vm_ceiling_s=200.0, provision_charged_to_gpu=False, notes=(),
    )

    without_hint = cc._enforcement_mapping(plan)
    assert without_hint["cli_args"] == [["--max-usd", "5.0"]]

    with_hint = cc._enforcement_mapping(plan, paper_hint="2605.15155")
    assert with_hint["cli_args"] == [["--max-usd", "5.0"], ["--paper-hint", "2605.15155"]]


def test_plan_attempt_paper_hint_reaches_child_argv(tmp_path, neutral_profile):
    """End-to-end: opts.paper_hint flows through plan_attempt's directives
    into the argv the driver would actually launch the child with."""
    from backend.agents.rlm.attempt_driver import build_reproduce_argv

    opts = _opts(tmp_path, run_spec_path=str(neutral_profile), paper_hint="2605.15155")
    run_dir = opts.runs_root / "prj_t"
    (run_dir / "campaign").mkdir(parents=True)

    planned = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=1), [])

    argv = build_reproduce_argv(planned["launch_payload"], python_exe="/usr/bin/python3")
    assert "--paper-hint" in argv
    assert argv[argv.index("--paper-hint") + 1] == "2605.15155"


def test_plan_attempt_no_paper_hint_omits_flag(tmp_path, neutral_profile):
    opts = _opts(tmp_path, run_spec_path=str(neutral_profile), paper_hint=None)
    run_dir = opts.runs_root / "prj_t"
    (run_dir / "campaign").mkdir(parents=True)

    planned = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=1), [])

    assert "--paper-hint" not in planned["launch_payload"].enforcement["cli_args"]
    flat = [item for pair in planned["launch_payload"].enforcement["cli_args"] for item in pair]
    assert "--paper-hint" not in flat


# --------------------------------------------------------------------------- #
# quarantine: dead-in-flight resume, width-child targeting (review FIX 1)     #
# --------------------------------------------------------------------------- #


def test_quarantine_targets_width_child_when_in_flight_run_dir_is_runs_root_child(tmp_path, monkeypatch):
    """The dead/orphaned in_flight attempt may be a WIDTH CHILD (run_dir =
    runs_root/<project>_w2, spec §8.3) -- quarantine must target the CHILD
    id, or the child's residue is silently left behind while the (unrelated,
    likely nonexistent) top-level dir is quarantined instead."""
    calls: list[tuple[str, Path, str]] = []
    monkeypatch.setattr(
        cc,
        "force_archive_incomplete",
        lambda project_id, runs_root, *, reason: calls.append((project_id, runs_root, reason)),
    )
    opts = _opts(tmp_path)
    child_dir = opts.runs_root / "prj_t_w2"
    in_flight = InFlight(attempt_n=2, driver="live", run_dir=str(child_dir), pid=None, lease_ref=None, launched_at=1.0)

    cc._quarantine_impl(opts, "prj_t", in_flight)

    assert calls == [("prj_t_w2", opts.runs_root, "campaign_resume_quarantine")]


def test_quarantine_falls_back_to_campaign_project_id_when_run_dir_empty(tmp_path, monkeypatch):
    """The common case (in_flight.run_dir == "", e.g. the tolerant
    reconstructor's default) must keep quarantining under the campaign's own
    top-level project_id -- unchanged from before the fix."""
    calls: list[tuple[str, Path, str]] = []
    monkeypatch.setattr(
        cc,
        "force_archive_incomplete",
        lambda project_id, runs_root, *, reason: calls.append((project_id, runs_root, reason)),
    )
    opts = _opts(tmp_path)
    in_flight = InFlight(attempt_n=1, driver="live", run_dir="", pid=None, lease_ref=None, launched_at=1.0)

    cc._quarantine_impl(opts, "prj_t", in_flight)

    assert calls == [("prj_t", opts.runs_root, "campaign_resume_quarantine")]


def test_quarantine_falls_back_to_campaign_project_id_when_run_dir_outside_runs_root(tmp_path, monkeypatch):
    """Fail-closed guard: a run_dir that does NOT resolve to a direct child
    of opts.runs_root must never redirect the quarantine target -- falls
    back to the campaign's own project_id rather than ever quarantining a
    path outside runs_root."""
    calls: list[tuple[str, Path, str]] = []
    monkeypatch.setattr(
        cc,
        "force_archive_incomplete",
        lambda project_id, runs_root, *, reason: calls.append((project_id, runs_root, reason)),
    )
    opts = _opts(tmp_path)
    outside_dir = tmp_path / "elsewhere" / "prj_t_w2"
    in_flight = InFlight(
        attempt_n=2, driver="live", run_dir=str(outside_dir), pid=None, lease_ref=None, launched_at=1.0
    )

    cc._quarantine_impl(opts, "prj_t", in_flight)

    assert calls == [("prj_t", opts.runs_root, "campaign_resume_quarantine")]


# --------------------------------------------------------------------------- #
# self-edit env merge excludes non-env diagnostic keys (review FIX 2)         #
# --------------------------------------------------------------------------- #


def test_self_edit_env_merge_excludes_dropped_diagnostic_key(tmp_path, neutral_profile, monkeypatch):
    """active_overrides() surfaces a diagnostic "_dropped" list (names of
    out-of-bounds overrides dropped on read) when the overrides file was
    edited out-of-band to a value that no longer validates -- that list is
    NOT a real env var and must never ride into the launched child's env."""
    from backend.agents.rlm import harness_self_edit as hse

    monkeypatch.setenv(hse.SELF_EDIT_ENV, "1")
    surface_path = tmp_path / "self_edit_surface.json"
    surface_path.write_text(
        json.dumps({
            "version": 1,
            "numeric_keys": {
                "OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD": {"min": 2, "max": 6, "kind": "int"},
            },
            "guidance_blocks": {},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(hse, "SURFACE_PATH", surface_path)

    opts = _opts(tmp_path, run_spec_path=str(neutral_profile))
    run_dir = opts.runs_root / "prj_t"
    (run_dir / "campaign").mkdir(parents=True)
    overrides_path = opts.runs_root / hse.OVERRIDES_FILENAME
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    overrides_path.write_text(
        json.dumps({"OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD": 99}),  # out of bounds -> "_dropped"
        encoding="utf-8",
    )
    # Sanity: this override really does produce the "_dropped" diagnostic the
    # fix guards against (else this test would pass vacuously).
    assert hse.active_overrides(opts.runs_root) == {"_dropped": ["OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD"]}

    planned = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=1), [])

    assert "_dropped" not in planned["launch_payload"].enforcement["env"]

def test_plan_attempt_sandbox_reaches_child_argv(tmp_path, neutral_profile):
    """The campaign's --sandbox choice MUST reach the child argv: without it
    the child falls to the repo default sandbox (runpod) — the 2026-07-02
    live SDAR launch died at RunPod preflight on a GCP VM exactly this way."""
    from backend.agents.rlm.attempt_driver import build_reproduce_argv

    opts = _opts(tmp_path, run_spec_path=str(neutral_profile))
    run_dir = opts.runs_root / "prj_t"
    (run_dir / "campaign").mkdir(parents=True)

    planned = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=1), [])

    argv = build_reproduce_argv(planned["launch_payload"], python_exe="/usr/bin/python3")
    assert "--sandbox" in argv
    assert argv[argv.index("--sandbox") + 1] == opts.sandbox
