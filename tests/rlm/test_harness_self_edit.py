"""Tests for backend/agents/rlm/harness_self_edit.py (Unit 11, Phase C).

Hermetic: every write lands under ``tmp_path`` (never ``runs/``); no sockets,
no LLM calls, no subprocess. The whitelist file the module reads
(``self_edit_surface.json``) is monkeypatched per-test via
``harness_self_edit.SURFACE_PATH`` so tests never depend on -- or mutate --
the real checked-in whitelist.

Spec: docs/superpowers/specs/2026-07-01-reproduction-campaign-and-self-improving-harness-design.md
S11.2 (self-edit tier) + S17 (non-goals: no autonomous canary->default flip,
ever) + S20 F11 (dedicated HarnessEditGate, held_out_gate untouched) / F12
(strengthened canary: >=2 papers x >=2 seeds, sigma bound, negative control,
operator-confirmed default only).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.rlm import campaign_composition as cc
from backend.agents.rlm import harness_self_edit as hse
from backend.agents.rlm.campaign_composition import CampaignOptions
from backend.agents.rlm.campaign_policy import directives_fingerprint
from backend.agents.rlm.reproduction_campaign import CampaignState

# --------------------------------------------------------------------------- #
# Shared fixtures / builders                                                  #
# --------------------------------------------------------------------------- #

_DEFAULT_SURFACE: dict = {
    "version": 1,
    "numeric_keys": {
        "OPENRESEARCH_REPAIR_MAX_ITERATIONS": {"min": 2, "max": 8, "kind": "int"},
        "OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD": {"min": 2, "max": 6, "kind": "int"},
    },
    "guidance_blocks": {"campaign_preamble_suffix": {"max_chars": 200}},
}


def _use_surface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: dict | None = None) -> Path:
    """Point ``harness_self_edit.SURFACE_PATH`` at a tmp-owned whitelist file
    so tests fully control (and never mutate) the whitelist under test."""
    path = tmp_path / "self_edit_surface.json"
    path.write_text(json.dumps(content if content is not None else _DEFAULT_SURFACE), encoding="utf-8")
    monkeypatch.setattr(hse, "SURFACE_PATH", path)
    return path


def _on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hse.SELF_EDIT_ENV, "1")


def _off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(hse.SELF_EDIT_ENV, raising=False)


def _proposal(surface_key: str, delta, mined_from: tuple = ()) -> hse.HarnessEditProposal:
    return hse.HarnessEditProposal(surface_key=surface_key, delta=delta, mined_from=tuple(mined_from))


def _seed_proposal_record(runs_root: Path, proposal: hse.HarnessEditProposal, status: str | None) -> str:
    """Directly write a proposal record to disk, bypassing ``propose()`` --
    simulates a hand-crafted/tampered record (or a hypothetically-buggy
    ``propose()``) so a gate method's frozen-tier RECHECK and/or its
    stage-order check can each be exercised in isolation, on an arbitrary
    (or entirely absent, ``status=None``) prior status, without walking
    every real stage to get there."""
    proposal_id = hse._proposal_id(proposal)
    path = runs_root / hse.PROPOSALS_DIRNAME / f"{proposal_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict = {"id": proposal_id, "proposal": hse._proposal_payload(proposal), "history": []}
    if status is not None:
        record["status"] = status
    path.write_text(json.dumps(record), encoding="utf-8")
    return proposal_id


def _write_corpus(runs_root: Path, cases: list[dict], name: str = "corpus") -> Path:
    path = runs_root / "_memory" / "replay" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"project_id": "prj", "run_dir": str(runs_root), "cases": cases}), encoding="utf-8"
    )
    return path


def _passing_fingerprint_case(case_id: str = "fp1") -> dict:
    """A fingerprint_replay case whose ``expected`` is computed via the REAL
    ``directives_fingerprint`` -- guaranteed baseline == expected, and (since
    the function reads no env) guaranteed overlay == baseline too."""
    inputs = {
        "seed_lineage": "fresh:0",
        "scope_rung": 0,
        "repair_action_kinds": [],
        "failure_classes": [],
        "envelope": {"llm_usd": 1.0, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 600.0, "vm_ceiling_s": 900.0},
    }
    expected_sha = directives_fingerprint(
        seed_lineage=inputs["seed_lineage"],
        scope_rung=inputs["scope_rung"],
        repair_action_kinds=inputs["repair_action_kinds"],
        failure_classes=inputs["failure_classes"],
        envelope=inputs["envelope"],
    )
    return {
        "case_id": case_id,
        "kind": "fingerprint_replay",
        "inputs": inputs,
        "expected": {"directives_sha256": expected_sha},
    }


def _bring_to_canary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runs_root: Path,
    *,
    ab_reports: list[dict],
    grader_sigma: float = 0.0067,
    surface_key: str = "OPENRESEARCH_REPAIR_MAX_ITERATIONS",
    delta=5,
) -> tuple[str, hse.HarnessEditGate]:
    """propose -> shadow (single always-clean fingerprint case) -> canary."""
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    proposed = hse.propose(_proposal(surface_key, delta), runs_root=runs_root)
    assert proposed["status"] == "candidate"
    proposal_id = proposed["id"]
    _write_corpus(runs_root, [_passing_fingerprint_case()])
    gate = hse.HarnessEditGate(runs_root=runs_root)
    shadow_result = gate.shadow(proposal_id)
    assert shadow_result == {"status": "shadow_passed"}
    canary_result = gate.promote_to_canary(proposal_id, ab_reports=ab_reports)
    assert canary_result["status"] == "canary", canary_result
    return proposal_id, gate


def _ab_row(paper_id: str, seed: int, arm: str, score: float, report_path: Path) -> dict:
    return {
        "paper_id": paper_id, "seed": seed, "arm": arm, "score": score, "report_path": str(report_path),
    }


def _paired_ab_reports(tmp_path: Path, *, delta: float, papers=("p1", "p2"), seeds=(1, 2)) -> list[dict]:
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    rows: list[dict] = []
    for paper_id in papers:
        for seed in seeds:
            rows.append(_ab_row(paper_id, seed, "control", 0.5, report_path))
            rows.append(_ab_row(paper_id, seed, "edit", 0.5 + delta, report_path))
    return rows


# --------------------------------------------------------------------------- #
# propose() -- flag off                                                       #
# --------------------------------------------------------------------------- #


def _write_profile(path: Path) -> Path:
    path.write_text(
        json.dumps({"OPENRESEARCH_REUSE_RUBRIC": "1", "OPENRESEARCH_EXTERNAL_VALIDATOR": "1"}), encoding="utf-8"
    )
    return path


def _opts(tmp_path: Path, **overrides) -> CampaignOptions:
    base = dict(
        paper_ref="2605.15155", runs_root=tmp_path / "runs", repo_root=tmp_path,
        max_llm_usd=100.0, max_gpu_usd=100.0, max_gpu_hours=10.0, max_attempts=6,
        wall_clock_s=None, mode="unattended", driver="live", width=1, plateau_k=2,
        sandbox="local", billing_sandbox=None, gpu_usd_per_hr=None, est_gpu_hours=2.0,
        run_spec_path=str(_write_profile(tmp_path / "profile.json")),
        scope_spec=None, scope_ladder=("full",), paper_hint=None, require_cpu_tier=False,
        arxiv_id="2605.15155", paper_class="generic", resume=False,
    )
    base.update(overrides)
    return CampaignOptions(**base)


def _state(**overrides) -> CampaignState:
    base = dict(
        project_id="prj_t", paper_ref="2605.15155", state="attempt_loop",
        next_attempt_n=1, mode="unattended", driver="live",
        budget={"max_llm_usd": 100.0, "max_gpu_usd": 100.0, "max_gpu_hours": 10.0,
                "max_attempts": 6, "max_wall_clock_s": None},
        spent={"llm_usd": 0.0, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 0.0},
        scope_rung=0, in_flight=None, understanding_sha256=None, rubric_sha256=None,
        steering_cursor=0, pending_approval=None, warnings=[], terminal=None,
        created_at=1.0, updated_at=1.0,
    )
    base.update(overrides)
    return CampaignState(**base)


def test_flag_off_propose_disabled_and_overrides_empty(tmp_path, monkeypatch):
    _off(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    result = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5), runs_root=runs_root)
    assert result == {"status": "disabled"}
    assert not (runs_root / "_memory" / "harness_proposals").exists()

    # An out-of-band overlay file present on disk must still read as {} while off.
    overrides_path = runs_root / hse.OVERRIDES_FILENAME
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    overrides_path.write_text(json.dumps({"OPENRESEARCH_REPAIR_MAX_ITERATIONS": 7}), encoding="utf-8")
    assert hse.active_overrides(runs_root) == {}


def test_composition_seams_byte_inert_when_flag_off(tmp_path, monkeypatch):
    _off(monkeypatch)
    opts = _opts(tmp_path)
    run_dir = opts.runs_root / "prj_t"
    (run_dir / "campaign").mkdir(parents=True)

    planned_without = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=1), [])
    directives_without = planned_without["launch_payload"]

    overrides_path = opts.runs_root / hse.OVERRIDES_FILENAME
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    overrides_path.write_text(
        json.dumps({
            "OPENRESEARCH_REPAIR_MAX_ITERATIONS": 7,
            "guidance:campaign_preamble_suffix": "an injected suffix",
        }),
        encoding="utf-8",
    )
    # Re-plan attempt 1 into a fresh run_dir so novelty collision never masks
    # a real difference in the enforcement/env or extra_guidance output.
    run_dir2 = opts.runs_root / "prj_t2"
    (run_dir2 / "campaign").mkdir(parents=True)
    planned_with = cc._plan_attempt_impl(run_dir2, opts, "prj_t2", _state(next_attempt_n=1, project_id="prj_t2"), [])
    directives_with = planned_with["launch_payload"]

    assert directives_with.enforcement["env"] == directives_without.enforcement["env"]
    assert directives_with.memory_hints == directives_without.memory_hints
    assert directives_with.extra_guidance == directives_without.extra_guidance


def test_write_reports_harvest_gated_on_flag(tmp_path, monkeypatch):
    _off(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runs_root = tmp_path / "runs"
    decision = {"kind": "EXHAUSTED", "rule": "max_attempts", "stop_reason": "max_attempts", "champion_attempt_n": None}

    cc._write_reports_impl(run_dir, _state(terminal=dict(decision)), [], decision, runs_root)

    assert not (runs_root / hse.REPLAY_DIRNAME).exists()


# --------------------------------------------------------------------------- #
# propose() -- validation                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "surface_key",
    [
        "OPENRESEARCH_EVIDENCE_GATE",
        "OPENRESEARCH_ZERO_METRICS_GUARD",
        "OPENRESEARCH_VALIDATOR_PANEL_N",
        "self_edit_surface.json",
        "harness_self_edit.py",
        "rubric_gen_anything",
        "OPENRESEARCH_MAX_RUN_GPU_USD",
    ],
)
def test_frozen_tier_proposals_rejected(tmp_path, monkeypatch, surface_key):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    result = hse.propose(_proposal(surface_key, "1"), runs_root=runs_root)

    assert result == {"status": "rejected", "reason": "frozen_tier"}
    proposal_id = hse._proposal_id(_proposal(surface_key, "1"))
    persisted = json.loads((runs_root / hse.PROPOSALS_DIRNAME / f"{proposal_id}.json").read_text())
    assert persisted["status"] == "rejected"
    assert persisted["history"][-1]["detail"]["reason"] == "frozen_tier"


@pytest.mark.parametrize(
    "surface_key",
    [
        "OPENRESEARCH_EVIDENCE_FINGERPRINT",
        "OPENRESEARCH_RUBRIC_HASH",
        "OPENRESEARCH_NEW_VALIDATOR_TIMEOUT_S",
        "OPENRESEARCH_BUDGET_HEADROOM",
        "OPENRESEARCH_REUSE_RUBRIC",
    ],
)
def test_frozen_tier_case_insensitive_against_uppercase_real_repo_keys(tmp_path, monkeypatch, surface_key):
    """Regression: the four generic single-word catch-alls in
    FROZEN_TIER_MARKERS (``rubric``/``validator``/``evidence``/``budget``)
    are lowercase, but every real ``OPENRESEARCH_*`` key in this repo is
    UPPERCASE -- a case-sensitive comparison left them dead code against
    the only naming convention this system actually uses. Every one of
    these real-shaped keys must now be rejected as frozen_tier."""
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    result = hse.propose(_proposal(surface_key, "1"), runs_root=runs_root)

    assert result == {"status": "rejected", "reason": "frozen_tier"}


def test_frozen_tier_hit_case_insensitive_direct():
    assert hse._frozen_tier_hit("OPENRESEARCH_EVIDENCE_FINGERPRINT") == "evidence"
    assert hse._frozen_tier_hit("openresearch_evidence_fingerprint") == "evidence"
    assert hse._frozen_tier_hit("OpenResearch_Rubric_Hash") == "rubric"


def test_frozen_tier_hit_does_not_reject_whitelisted_surface_keys():
    """The now-case-insensitive generic markers must not collide with the
    whitelist -- verified directly against both numeric keys and the
    guidance block id in ``_DEFAULT_SURFACE`` (this test module's fixture,
    which mirrors the two live keys + one guidance block actually shipped
    in self_edit_surface.json; kept hermetic, never reads the real file)."""
    for numeric_key in _DEFAULT_SURFACE["numeric_keys"]:
        assert hse._frozen_tier_hit(numeric_key) is None, numeric_key
    for block_id in _DEFAULT_SURFACE["guidance_blocks"]:
        assert hse._frozen_tier_hit(f"guidance:{block_id}") is None, block_id


def test_unknown_key_rejected(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    numeric = hse.propose(_proposal("OPENRESEARCH_NOT_ON_THE_WHITELIST", 3), runs_root=runs_root)
    assert numeric == {"status": "rejected", "reason": "unknown_key"}

    guidance = hse.propose(_proposal("guidance:not_a_real_block", "hi"), runs_root=runs_root)
    assert guidance == {"status": "rejected", "reason": "unknown_key"}


@pytest.mark.parametrize("delta", [1, 9, "5"])
def test_out_of_bounds_numeric_rejected(tmp_path, monkeypatch, delta):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    result = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", delta), runs_root=runs_root)

    assert result == {"status": "rejected", "reason": "out_of_bounds"}


def test_out_of_bounds_rejects_bool_as_int(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    result = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", True), runs_root=runs_root)

    assert result == {"status": "rejected", "reason": "out_of_bounds"}


def test_guidance_over_cap_rejected(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    result = hse.propose(_proposal("guidance:campaign_preamble_suffix", "x" * 201), runs_root=runs_root)

    assert result == {"status": "rejected", "reason": "over_cap"}


def test_guidance_within_cap_is_candidate(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    result = hse.propose(_proposal("guidance:campaign_preamble_suffix", "x" * 200), runs_root=runs_root)

    assert result["status"] == "candidate"


def test_valid_candidate_persisted_with_lineage(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    result = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5, ("evidence.json",)), runs_root=runs_root)

    assert result["status"] == "candidate"
    assert isinstance(result["id"], str) and len(result["id"]) == 12
    persisted = json.loads((runs_root / hse.PROPOSALS_DIRNAME / f"{result['id']}.json").read_text())
    assert persisted["proposal"] == {
        "surface_key": "OPENRESEARCH_REPAIR_MAX_ITERATIONS", "delta": 5, "mined_from": ["evidence.json"],
    }
    assert persisted["status"] == "candidate"
    assert persisted["history"][0]["stage"] == "propose"
    assert "at" in persisted["history"][0]


# --------------------------------------------------------------------------- #
# HarnessEditGate.shadow                                                      #
# --------------------------------------------------------------------------- #


def test_shadow_empty_corpus_stays_candidate(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    proposed = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5), runs_root=runs_root)

    result = hse.HarnessEditGate(runs_root=runs_root).shadow(proposed["id"])

    assert result == {"status": "candidate", "reason": "replay_corpus_empty"}
    persisted = json.loads((runs_root / hse.PROPOSALS_DIRNAME / f"{proposed['id']}.json").read_text())
    assert persisted["status"] == "candidate"
    assert persisted["history"][-1]["stage"] == "shadow"


def test_shadow_replay_error_rejects(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    proposed = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5), runs_root=runs_root)
    corrupt = runs_root / "_memory" / "replay" / "corrupt.json"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_text("{not valid json", encoding="utf-8")

    result = hse.HarnessEditGate(runs_root=runs_root).shadow(proposed["id"])

    assert result["status"] == "rejected"
    assert result["reason"].startswith("replay_error:")


def test_shadow_malformed_case_rejects(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    proposed = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5), runs_root=runs_root)
    _write_corpus(runs_root, [{"case_id": "bad", "kind": "decide_replay"}])  # missing inputs/expected

    result = hse.HarnessEditGate(runs_root=runs_root).shadow(proposed["id"])

    assert result["status"] == "rejected"
    assert result["reason"].startswith("replay_error:")


def test_shadow_corpus_stale_rejects(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    proposed = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5), runs_root=runs_root)
    stale_case = _passing_fingerprint_case()
    stale_case["expected"] = {"directives_sha256": "0" * 64}  # deliberately wrong
    _write_corpus(runs_root, [stale_case])

    result = hse.HarnessEditGate(runs_root=runs_root).shadow(proposed["id"])

    assert result == {"status": "rejected", "reason": "corpus_stale:fp1"}


def test_shadow_negative_control_regression_rejects(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENRESEARCH_REPAIR_MAX_ITERATIONS", raising=False)
    runs_root = tmp_path / "runs"
    proposed = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5), runs_root=runs_root)

    def _knob_sensitive_executor(inputs):
        import os

        return {"value": os.environ.get("OPENRESEARCH_REPAIR_MAX_ITERATIONS", "baseline")}

    monkeypatch.setitem(hse._CASE_EXECUTORS, "knob_sensitive_fake", _knob_sensitive_executor)
    case = {
        "case_id": "sensitive1", "kind": "knob_sensitive_fake", "inputs": {}, "expected": {"value": "baseline"},
    }
    _write_corpus(runs_root, [case])

    result = hse.HarnessEditGate(runs_root=runs_root).shadow(proposed["id"])

    assert result == {"status": "rejected", "reason": "negative_control_regression:sensitive1"}


def test_shadow_real_executors_env_insensitive(tmp_path, monkeypatch):
    """Pins the fact that decide()/directives_fingerprint() read no env --
    the reason every REAL replay is a negative control by construction."""
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENRESEARCH_REPAIR_MAX_ITERATIONS", raising=False)
    runs_root = tmp_path / "runs"
    proposed = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5), runs_root=runs_root)
    _write_corpus(runs_root, [_passing_fingerprint_case()])

    result = hse.HarnessEditGate(runs_root=runs_root).shadow(proposed["id"])

    assert result == {"status": "shadow_passed"}


def test_shadow_unknown_proposal_id_rejects(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    result = hse.HarnessEditGate(runs_root=runs_root).shadow("deadbeefcafe")

    assert result["status"] == "rejected"


# --------------------------------------------------------------------------- #
# HarnessEditGate.promote_to_canary                                           #
# --------------------------------------------------------------------------- #


def test_promote_requires_shadow_passed_first(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    proposed = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5), runs_root=runs_root)

    result = hse.HarnessEditGate(runs_root=runs_root).promote_to_canary(proposed["id"], ab_reports=[])

    assert result == {"status": "rejected", "reason": "shadow_not_passed"}
    persisted = json.loads((runs_root / hse.PROPOSALS_DIRNAME / f"{proposed['id']}.json").read_text())
    # Prerequisite-ordering rejection is retryable -- it must not brick a
    # proposal that simply hasn't been shadowed yet.
    assert persisted["status"] == "candidate"


@pytest.mark.parametrize(
    "papers,seeds,expected_reason",
    [
        (("p1",), (1, 2), "insufficient_papers"),
        (("p1", "p2"), (1,), "insufficient_seeds"),
    ],
)
def test_canary_requires_two_papers_two_seeds(tmp_path, monkeypatch, papers, seeds, expected_reason):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    proposed = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5), runs_root=runs_root)
    _write_corpus(runs_root, [_passing_fingerprint_case()])
    gate = hse.HarnessEditGate(runs_root=runs_root)
    assert gate.shadow(proposed["id"]) == {"status": "shadow_passed"}

    ab_reports = _paired_ab_reports(tmp_path, delta=0.1, papers=papers, seeds=seeds)
    result = gate.promote_to_canary(proposed["id"], ab_reports=ab_reports)

    assert result == {"status": "rejected", "reason": expected_reason}


def test_canary_incomplete_pairing_rejected(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    proposed = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5), runs_root=runs_root)
    _write_corpus(runs_root, [_passing_fingerprint_case()])
    gate = hse.HarnessEditGate(runs_root=runs_root)
    assert gate.shadow(proposed["id"]) == {"status": "shadow_passed"}

    ab_reports = _paired_ab_reports(tmp_path, delta=0.1)
    ab_reports.pop()  # drop one arm of one (paper, seed) pair

    result = gate.promote_to_canary(proposed["id"], ab_reports=ab_reports)

    assert result["status"] == "rejected"
    assert result["reason"].startswith("incomplete_pairing")


def test_canary_fabricated_evidence_rejected(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    proposed = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5), runs_root=runs_root)
    _write_corpus(runs_root, [_passing_fingerprint_case()])
    gate = hse.HarnessEditGate(runs_root=runs_root)
    assert gate.shadow(proposed["id"]) == {"status": "shadow_passed"}

    ab_reports = _paired_ab_reports(tmp_path, delta=0.1)
    ab_reports[0] = dict(ab_reports[0], report_path=str(tmp_path / "does_not_exist.json"))

    result = gate.promote_to_canary(proposed["id"], ab_reports=ab_reports)

    assert result["status"] == "rejected"
    assert result["reason"].startswith("fabricated_evidence:")


def test_canary_requires_sigma_exceedance(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    # Exact binary fractions -- avoids float-drift false positives/negatives
    # in the strict `>` comparison (e.g. 0.5 + 0.01 - 0.5 != 0.01 in IEEE754).
    sigma = 0.25

    proposed_eq = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5), runs_root=runs_root)
    _write_corpus(runs_root, [_passing_fingerprint_case()])
    gate = hse.HarnessEditGate(runs_root=runs_root, grader_sigma=sigma)
    assert gate.shadow(proposed_eq["id"]) == {"status": "shadow_passed"}
    equal_reports = _paired_ab_reports(tmp_path, delta=sigma)
    equal_result = gate.promote_to_canary(proposed_eq["id"], ab_reports=equal_reports)
    assert equal_result["status"] == "rejected"

    proposed_gt = hse.propose(_proposal("OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", 4), runs_root=runs_root)
    assert gate.shadow(proposed_gt["id"]) == {"status": "shadow_passed"}
    gt_reports = _paired_ab_reports(tmp_path, delta=sigma * 2)
    gt_result = gate.promote_to_canary(proposed_gt["id"], ab_reports=gt_reports)
    assert gt_result == {"status": "canary"}


# --------------------------------------------------------------------------- #
# HarnessEditGate.apply_default                                               #
# --------------------------------------------------------------------------- #


def test_default_requires_operator_confirmation(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    ab_reports = _paired_ab_reports(tmp_path, delta=0.1)
    proposal_id, gate = _bring_to_canary(monkeypatch, tmp_path, runs_root, ab_reports=ab_reports)

    denied = gate.apply_default(proposal_id, operator_confirmed=False)
    assert denied == {"status": "rejected", "reason": "operator_confirmation_required"}
    persisted = json.loads((runs_root / hse.PROPOSALS_DIRNAME / f"{proposal_id}.json").read_text())
    assert persisted["status"] == "canary"  # retryable -- not bricked

    confirmed = gate.apply_default(proposal_id, operator_confirmed=True)
    assert confirmed == {"status": "default"}
    overrides = json.loads((runs_root / hse.OVERRIDES_FILENAME).read_text())
    assert overrides == {"OPENRESEARCH_REPAIR_MAX_ITERATIONS": 5}


def test_default_requires_canary_status(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    proposed = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5), runs_root=runs_root)
    gate = hse.HarnessEditGate(runs_root=runs_root)

    result = gate.apply_default(proposed["id"], operator_confirmed=True)

    assert result == {"status": "rejected", "reason": "canary_not_reached"}
    persisted = json.loads((runs_root / hse.PROPOSALS_DIRNAME / f"{proposed['id']}.json").read_text())
    assert persisted["status"] == "candidate"  # retryable -- not bricked


def test_apply_revalidates_bounds_at_apply_time(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    ab_reports = _paired_ab_reports(tmp_path, delta=0.1)
    proposal_id, gate = _bring_to_canary(monkeypatch, tmp_path, runs_root, ab_reports=ab_reports)

    # Whitelist shrinks between canary and apply: the key is removed outright.
    shrunk = {"version": 1, "numeric_keys": {}, "guidance_blocks": {}}
    _use_surface(monkeypatch, tmp_path, shrunk)

    result = gate.apply_default(proposal_id, operator_confirmed=True)

    assert result == {"status": "rejected", "reason": "unknown_key"}
    assert not (runs_root / hse.OVERRIDES_FILENAME).exists()


def test_apply_default_never_auto_flips(tmp_path, monkeypatch):
    """S17: no code path may reach status='default' without the literal
    operator_confirmed=True argument -- there is no other route in."""
    runs_root = tmp_path / "runs"
    ab_reports = _paired_ab_reports(tmp_path, delta=0.1)
    proposal_id, gate = _bring_to_canary(monkeypatch, tmp_path, runs_root, ab_reports=ab_reports)

    for bogus in (False, None, 1, "true", "True"):
        result = gate.apply_default(proposal_id, operator_confirmed=bogus)
        assert result["status"] == "rejected"
    assert not (runs_root / hse.OVERRIDES_FILENAME).exists()


# --------------------------------------------------------------------------- #
# active_overrides()                                                          #
# --------------------------------------------------------------------------- #


def test_active_overrides_drops_out_of_bounds_on_read(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    overrides_path = runs_root / hse.OVERRIDES_FILENAME
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    overrides_path.write_text(
        json.dumps({
            "OPENRESEARCH_REPAIR_MAX_ITERATIONS": 5,          # valid
            "OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD": 99,  # out of bounds
            "OPENRESEARCH_NOT_WHITELISTED": 1,                # unknown key
        }),
        encoding="utf-8",
    )

    result = hse.active_overrides(runs_root)

    assert result["OPENRESEARCH_REPAIR_MAX_ITERATIONS"] == 5
    assert "OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD" not in result
    assert "OPENRESEARCH_NOT_WHITELISTED" not in result
    assert set(result["_dropped"]) == {"OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD", "OPENRESEARCH_NOT_WHITELISTED"}


def test_active_overrides_missing_file_is_empty(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    assert hse.active_overrides(runs_root) == {}


def test_active_overrides_drops_frozen_tier_key_even_under_compromised_whitelist(tmp_path, monkeypatch):
    """A hand-edited overrides file cannot smuggle a frozen key into a child
    env, even when the whitelist itself has been compromised to list that
    key as an ordinary bounded numeric knob -- the frozen-tier check on
    read runs independent of (and in addition to) the bounds check."""
    _on(monkeypatch)
    compromised_surface = {
        "version": 1,
        "numeric_keys": {"OPENRESEARCH_EVIDENCE_GATE": {"min": 0, "max": 1, "kind": "int"}},
        "guidance_blocks": {},
    }
    _use_surface(monkeypatch, tmp_path, compromised_surface)
    runs_root = tmp_path / "runs"
    overrides_path = runs_root / hse.OVERRIDES_FILENAME
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    overrides_path.write_text(json.dumps({"OPENRESEARCH_EVIDENCE_GATE": 0}), encoding="utf-8")

    result = hse.active_overrides(runs_root)

    assert "OPENRESEARCH_EVIDENCE_GATE" not in result
    assert result["_dropped"] == ["OPENRESEARCH_EVIDENCE_GATE"]


# --------------------------------------------------------------------------- #
# Frozen tier / stage order -- defense-in-depth at every gate boundary        #
# (fix pass: propose() alone was never enough -- shadow/promote_to_canary/    #
# apply_default each independently re-derive both checks; see the frozen-    #
# tier-exploit-chain regression tests further down for the full PoC.)        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seeded_status", ["rejected", "shadow_passed", "canary", "default", None])
def test_shadow_stage_order_requires_exact_candidate_status(tmp_path, monkeypatch, seeded_status):
    """shadow() must refuse to run on anything but a fresh "candidate" -- an
    already-rejected, already-advanced, or status-less ("missing") record
    is refused with reason "stage_order" and persisted as terminally
    "rejected", isolated here from the (non-frozen) surface_key so only the
    stage-order defense is under test."""
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    proposal = _proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5)
    proposal_id = _seed_proposal_record(runs_root, proposal, seeded_status)
    _write_corpus(runs_root, [_passing_fingerprint_case()])

    result = hse.HarnessEditGate(runs_root=runs_root).shadow(proposal_id)

    assert result == {"status": "rejected", "reason": "stage_order"}
    persisted = json.loads((runs_root / hse.PROPOSALS_DIRNAME / f"{proposal_id}.json").read_text())
    assert persisted["status"] == "rejected"


def test_shadow_rechecks_frozen_tier_independent_of_stage_order(tmp_path, monkeypatch):
    """Even with the prerequisite status exactly right ("candidate"),
    shadow() must independently re-derive the frozen tier from the
    persisted proposal -- proves the frozen-tier defense does not rely on
    the stage-order check (or propose()) ever having caught anything."""
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    proposal = _proposal("OPENRESEARCH_EVIDENCE_GATE", 0)
    proposal_id = _seed_proposal_record(runs_root, proposal, "candidate")
    _write_corpus(runs_root, [_passing_fingerprint_case()])

    result = hse.HarnessEditGate(runs_root=runs_root).shadow(proposal_id)

    assert result == {"status": "rejected", "reason": "frozen_tier"}
    persisted = json.loads((runs_root / hse.PROPOSALS_DIRNAME / f"{proposal_id}.json").read_text())
    assert persisted["status"] == "rejected"
    assert persisted["history"][-1]["detail"]["marker"] == "OPENRESEARCH_EVIDENCE_GATE"


def test_promote_to_canary_rechecks_frozen_tier_independent_of_stage_order(tmp_path, monkeypatch):
    """Seeded straight to "shadow_passed" (the exact prerequisite
    promote_to_canary demands) -- the frozen-tier recheck must still fire
    first, before the ordinary shadow_not_passed/AB-report logic even
    runs."""
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    proposal = _proposal("OPENRESEARCH_EVIDENCE_GATE", 0)
    proposal_id = _seed_proposal_record(runs_root, proposal, "shadow_passed")

    result = hse.HarnessEditGate(runs_root=runs_root).promote_to_canary(
        proposal_id, ab_reports=_paired_ab_reports(tmp_path, delta=0.1)
    )

    assert result == {"status": "rejected", "reason": "frozen_tier"}
    persisted = json.loads((runs_root / hse.PROPOSALS_DIRNAME / f"{proposal_id}.json").read_text())
    assert persisted["status"] == "rejected"


def test_apply_default_rechecks_frozen_tier_independent_of_stage_order(tmp_path, monkeypatch):
    """Seeded straight to "canary" with operator_confirmed=True (the exact
    prerequisites apply_default demands) -- the frozen-tier recheck must
    still fire first, before operator-confirmation is even consulted, and
    harness_overrides.json must never be written."""
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    proposal = _proposal("OPENRESEARCH_EVIDENCE_GATE", 0)
    proposal_id = _seed_proposal_record(runs_root, proposal, "canary")

    result = hse.HarnessEditGate(runs_root=runs_root).apply_default(proposal_id, operator_confirmed=True)

    assert result == {"status": "rejected", "reason": "frozen_tier"}
    assert not (runs_root / hse.OVERRIDES_FILENAME).exists()
    persisted = json.loads((runs_root / hse.PROPOSALS_DIRNAME / f"{proposal_id}.json").read_text())
    assert persisted["status"] == "rejected"


def test_frozen_tier_exploit_chain_dead_at_every_stage(tmp_path, monkeypatch):
    """Regression for the reviewer's live PoC: propose() correctly rejects a
    frozen-tier key, but its proposal_id is a deterministic sha256 of the
    proposal payload -- computable by any caller without ever seeing that
    rejection. Driving shadow -> promote_to_canary -> apply_default
    directly on that id -- even under a whitelist deliberately compromised
    (AFTER the correct rejection) to list the frozen key as an ordinary
    bounded numeric knob, with the flag ON and a real passing replay corpus
    -- must reject at every single stage, and harness_overrides.json must
    never come into existence."""
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    frozen_key = "OPENRESEARCH_EVIDENCE_GATE"
    proposal = _proposal(frozen_key, 0)
    proposed = hse.propose(proposal, runs_root=runs_root)
    assert proposed == {"status": "rejected", "reason": "frozen_tier"}
    proposal_id = hse._proposal_id(proposal)

    # Compromised whitelist scenario -- exactly what the frozen tier's own
    # docstring says it must survive ("the whitelist file itself... is a
    # frozen tier member").
    compromised_surface = {
        "version": 1,
        "numeric_keys": {frozen_key: {"min": 0, "max": 1, "kind": "int"}},
        "guidance_blocks": {},
    }
    _use_surface(monkeypatch, tmp_path, compromised_surface)
    _write_corpus(runs_root, [_passing_fingerprint_case()])
    gate = hse.HarnessEditGate(runs_root=runs_root)

    shadow_result = gate.shadow(proposal_id)
    assert shadow_result["status"] == "rejected"
    assert shadow_result["reason"] in ("frozen_tier", "stage_order")

    canary_result = gate.promote_to_canary(proposal_id, ab_reports=_paired_ab_reports(tmp_path, delta=0.1))
    assert canary_result["status"] == "rejected"
    assert canary_result["reason"] in ("frozen_tier", "stage_order", "shadow_not_passed")

    apply_result = gate.apply_default(proposal_id, operator_confirmed=True)
    assert apply_result["status"] == "rejected"
    assert apply_result["reason"] in ("frozen_tier", "stage_order", "canary_not_reached")

    assert not (runs_root / hse.OVERRIDES_FILENAME).exists()
    persisted = json.loads((runs_root / hse.PROPOSALS_DIRNAME / f"{proposal_id}.json").read_text())
    assert persisted["status"] == "rejected"


def test_stale_rejected_proposal_never_advances_through_any_stage(tmp_path, monkeypatch):
    """A proposal rejected at propose() time for an ORDINARY (non-frozen)
    reason -- out-of-bounds -- must also stay terminally rejected: walking
    it through shadow -> promote_to_canary -> apply_default must never let
    its persisted status become anything other than "rejected". Isolates
    the stage-order defense from the frozen-tier one (see the exploit-chain
    test above for the frozen case)."""
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    proposal = _proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 999)
    proposed = hse.propose(proposal, runs_root=runs_root)
    assert proposed == {"status": "rejected", "reason": "out_of_bounds"}
    proposal_id = hse._proposal_id(proposal)

    _write_corpus(runs_root, [_passing_fingerprint_case()])
    gate = hse.HarnessEditGate(runs_root=runs_root)

    steps = (
        lambda: gate.shadow(proposal_id),
        lambda: gate.promote_to_canary(proposal_id, ab_reports=_paired_ab_reports(tmp_path, delta=0.1)),
        lambda: gate.apply_default(proposal_id, operator_confirmed=True),
    )
    for step in steps:
        result = step()
        assert result["status"] == "rejected"
        persisted = json.loads((runs_root / hse.PROPOSALS_DIRNAME / f"{proposal_id}.json").read_text())
        assert persisted["status"] == "rejected"

    assert not (runs_root / hse.OVERRIDES_FILENAME).exists()


# --------------------------------------------------------------------------- #
# HarnessEditGate -- OFF-twin (fix pass: shadow/promote_to_canary/            #
# apply_default were previously entirely ungated by OPENRESEARCH_SELF_EDIT)   #
# --------------------------------------------------------------------------- #


def test_gate_shadow_disabled_when_flag_off(tmp_path, monkeypatch):
    _off(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    result = hse.HarnessEditGate(runs_root=runs_root).shadow("deadbeefcafe")

    assert result == {"status": "disabled"}
    assert not runs_root.exists()


def test_gate_promote_to_canary_disabled_when_flag_off(tmp_path, monkeypatch):
    _off(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    result = hse.HarnessEditGate(runs_root=runs_root).promote_to_canary("deadbeefcafe", ab_reports=[])

    assert result == {"status": "disabled"}
    assert not runs_root.exists()


def test_gate_apply_default_disabled_when_flag_off(tmp_path, monkeypatch):
    _off(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"

    result = hse.HarnessEditGate(runs_root=runs_root).apply_default("deadbeefcafe", operator_confirmed=True)

    assert result == {"status": "disabled"}
    assert not runs_root.exists()


def test_gate_methods_refuse_to_advance_seeded_candidate_when_flag_off(tmp_path, monkeypatch):
    """Regression for the reviewer's MODERATE finding: a candidate-status
    proposal record already on disk (as would persist from an earlier ON
    session, or from direct tampering) must NOT be advanceable through
    shadow -> canary -> default while OPENRESEARCH_SELF_EDIT is off -- even
    though ``_load()`` is pure disk IO with no flag check of its own."""
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    proposal = _proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5)
    proposal_id = _seed_proposal_record(runs_root, proposal, "candidate")
    _write_corpus(runs_root, [_passing_fingerprint_case()])
    _off(monkeypatch)

    gate = hse.HarnessEditGate(runs_root=runs_root)
    assert gate.shadow(proposal_id) == {"status": "disabled"}
    assert gate.promote_to_canary(proposal_id, ab_reports=_paired_ab_reports(tmp_path, delta=0.1)) == {
        "status": "disabled"
    }
    assert gate.apply_default(proposal_id, operator_confirmed=True) == {"status": "disabled"}

    assert not (runs_root / hse.OVERRIDES_FILENAME).exists()
    persisted = json.loads((runs_root / hse.PROPOSALS_DIRNAME / f"{proposal_id}.json").read_text())
    assert persisted["status"] == "candidate"  # untouched


# --------------------------------------------------------------------------- #
# Composition seams -- ON path (env injection + guidance suffix)              #
# --------------------------------------------------------------------------- #


def test_composition_env_seam_merges_numeric_override_when_enabled(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    opts = _opts(tmp_path)
    run_dir = opts.runs_root / "prj_t"
    (run_dir / "campaign").mkdir(parents=True)
    overrides_path = opts.runs_root / hse.OVERRIDES_FILENAME
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    overrides_path.write_text(json.dumps({"OPENRESEARCH_REPAIR_MAX_ITERATIONS": 7}), encoding="utf-8")

    planned = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=1), [])

    assert planned["launch_payload"].enforcement["env"]["OPENRESEARCH_REPAIR_MAX_ITERATIONS"] == "7"


def test_composition_guidance_seam_appends_suffix_when_enabled(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    opts = _opts(tmp_path)
    run_dir = opts.runs_root / "prj_t"
    (run_dir / "campaign").mkdir(parents=True)
    overrides_path = opts.runs_root / hse.OVERRIDES_FILENAME
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    overrides_path.write_text(
        json.dumps({"guidance:campaign_preamble_suffix": "injected-suffix-marker"}), encoding="utf-8"
    )

    planned = cc._plan_attempt_impl(run_dir, opts, "prj_t", _state(next_attempt_n=1), [])

    assert "injected-suffix-marker" in planned["launch_payload"].memory_hints
    assert "injected-suffix-marker" in planned["launch_payload"].extra_guidance


# --------------------------------------------------------------------------- #
# harvest_replay_cases                                                        #
# --------------------------------------------------------------------------- #


def test_harvest_writes_executable_cases_and_shadow_replays_them_green(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    from backend.agents.rlm.attempt_assessment import AttemptAssessment, ReportDigest, ValidatorStatus
    from backend.agents.rlm.campaign_directives import synthesize_directives
    from backend.agents.rlm.campaign_policy import (
        AttemptEnvelope,
        CampaignBudget,
        CampaignSpend,
        NextAttemptPlan,
        PolicyConfig,
        decide,
    )

    runs_root = tmp_path / "runs"
    run_dir = runs_root / "prj_h"
    (run_dir / "campaign").mkdir(parents=True)

    envelope = AttemptEnvelope(llm_usd=1.0, gpu_usd=0.0, gpu_hours=0.0, wall_s=600.0, vm_ceiling_s=900.0)
    plan = NextAttemptPlan(lineage="fresh", seed_attempt_n=None, seed_pointer=None, scope_rung=0, width=1)
    directives = synthesize_directives(
        attempt_n=1, project_id="prj_h", paper_ref="2605.15155", plan=plan, envelope=envelope,
        enforcement={"cli_args": [], "env": {}}, run_spec_path=None, understanding_ref=None,
        unresolved_warnings=[], prior_run_artifacts={}, improvement_notes=[], memory_hints=[],
        injected_lesson_signatures=[], failure_classes=[], scope_spec=None, target_floor=None,
        out_dir=run_dir / "campaign",
    )

    cost = CampaignSpend(llm_usd=1.0, gpu_usd=0.0, gpu_hours=0.0, wall_s=100.0)
    assessment = AttemptAssessment(
        attempt_n=1, driver="live", project_id="prj_h", directives_sha256=directives.fingerprint,
        final_report=ReportDigest(
            score=0.3, target=0.7, meets_target=False, implementation_verdict=None, replication_verdict=None,
            verdict="partial", stop_reason=None, exclusions=(), path="final_report.json",
        ),
        evidence_predicates={"backed_by_ledger": True, "run_level_clean": False}, guard_flags={},
        validator=ValidatorStatus(status="clean", fingerprint="f", fresh=True),
        leaf_pass_count=2, leaf_vector_ref=None, failure_class=None, failure_signature=None, failure_scope=None,
        cost=cost, rubric_sha256_ok=None, hard_quarantined=False, soft_quarantined=False, quarantine_reasons=(),
    )

    budget_dict = {
        "max_llm_usd": 100.0, "max_gpu_usd": 100.0, "max_gpu_hours": 10.0,
        "max_attempts": 6, "max_wall_clock_s": None,
    }
    decision = decide(
        [assessment],
        budget=CampaignBudget(**budget_dict), spent=cost,
        config=PolicyConfig(max_attempts=6, plateau_k=2, width=1, width_skip_score=0.5, ladder_len=1),
        next_estimate=CampaignSpend(), lineage_by_attempt={}, scope_rung_by_attempt={1: 0},
        runs_dir_hint={}, current_rung=0, blocking_gap=None,
    )
    assert decision.kind == "CONTINUE"

    rows = [
        {"attempt_n": 1, "status": "launched", "directives_sha256": directives.fingerprint,
         "envelope": envelope.to_dict(), "driver": "live", "project_id": "prj_h",
         "run_dir": str(run_dir), "launched_at": 1000.0},
        {"attempt_n": 1, "status": "assessed", "assessment": assessment.to_dict(), "assessed_at": 1001.0},
        {"attempt_n": 1, "status": "decided", "decision": decision.to_dict()},
    ]
    state = {"budget": budget_dict, "project_id": "prj_h"}

    out_path = hse.harvest_replay_cases(run_dir, runs_root=runs_root, state=state, rows=rows)

    assert out_path is not None and out_path.exists()
    payload = json.loads(out_path.read_text())
    kinds = {c["kind"] for c in payload["cases"]}
    assert kinds == {"decide_replay", "fingerprint_replay"}

    proposed = hse.propose(_proposal("OPENRESEARCH_REPAIR_MAX_ITERATIONS", 5), runs_root=runs_root)
    result = hse.HarnessEditGate(runs_root=runs_root).shadow(proposed["id"])
    assert result == {"status": "shadow_passed"}


def test_harvest_returns_none_on_empty_ledger(tmp_path, monkeypatch):
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "prj_empty"
    run_dir.mkdir(parents=True)

    result = hse.harvest_replay_cases(run_dir, runs_root=runs_root, state={"budget": {}}, rows=[])

    assert result is None


def test_write_reports_harvest_runs_when_flag_on(tmp_path, monkeypatch):
    """The composition seam calls harvest_replay_cases when SELF_EDIT is on;
    an empty/report-only campaign yields no cases, so no file is written --
    proves the call happens (fail-soft) without needing a full ledger."""
    _on(monkeypatch)
    _use_surface(monkeypatch, tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runs_root = tmp_path / "runs"
    decision = {"kind": "EXHAUSTED", "rule": "max_attempts", "stop_reason": "max_attempts", "champion_attempt_n": None}

    # Must not raise even though the ledger has no harvestable rows.
    cc._write_reports_impl(run_dir, _state(terminal=dict(decision)), [], decision, runs_root)


# --------------------------------------------------------------------------- #
# Lineage / history                                                           #
# --------------------------------------------------------------------------- #


def test_promotion_lineage_recorded(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    ab_reports = _paired_ab_reports(tmp_path, delta=0.1)
    proposal_id, gate = _bring_to_canary(monkeypatch, tmp_path, runs_root, ab_reports=ab_reports)
    gate.apply_default(proposal_id, operator_confirmed=True)

    persisted = json.loads((runs_root / hse.PROPOSALS_DIRNAME / f"{proposal_id}.json").read_text())
    stages = [h["stage"] for h in persisted["history"]]
    assert stages == ["propose", "shadow", "canary", "apply_default"]
    timestamps = [h["at"] for h in persisted["history"]]
    assert all(isinstance(t, str) and t for t in timestamps)
    assert persisted["status"] == "default"
