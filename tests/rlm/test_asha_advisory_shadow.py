"""Shadow-mode ASHA advisory wired into the campaign decide path
(`campaign_composition._maybe_attach_asha_advisory`): OFF is byte-identical, ON
attaches advisory metadata WITHOUT changing the decision, and it is fail-soft."""
from types import SimpleNamespace

from backend.agents.rlm.campaign_composition import _maybe_attach_asha_advisory


def _assess(attempt_n, score, failure_class=None):
    report = SimpleNamespace(score=score) if score is not None else None
    return SimpleNamespace(
        attempt_n=attempt_n, final_report=report, failure_class=failure_class
    )


def test_off_is_byte_identical(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_SCHEDULER_TREE", raising=False)
    result = {"kind": "CONTINUE", "rule": "x"}
    _maybe_attach_asha_advisory(result, [_assess(1, 0.9), _assess(2, 0.1)], 0)
    assert result == {"kind": "CONTINUE", "rule": "x"}  # no advisory key added


def test_on_attaches_advisory_without_changing_decision(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "1")
    result = {"kind": "CONTINUE", "rule": "x"}
    _maybe_attach_asha_advisory(result, [_assess(1, 0.9), _assess(2, 0.1)], 0)
    assert result["kind"] == "CONTINUE"  # the live decision is untouched
    assert result["rule"] == "x"
    adv = result["asha_advisory"]
    assert adv["rung"] == 0
    actions = {d["branch_id"]: d["action"] for d in adv["decisions"]}
    assert actions["1"] == "promote"  # higher score promoted
    assert actions["2"] != "kill"  # not broken → freeze, never killed


def test_on_kills_only_broken(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "1")
    result = {"kind": "CONTINUE"}
    _maybe_attach_asha_advisory(
        result, [_assess(1, 0.9), _assess(2, 0.0, "training_diverged")], 0
    )
    actions = {d["branch_id"]: d["action"] for d in result["asha_advisory"]["decisions"]}
    assert actions["2"] == "kill"  # provable breakage → true-kill


def test_on_is_fail_soft(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "1")
    monkeypatch.setattr(
        "backend.agents.rlm.asha_campaign_adapter.asha_decide_for_assessments",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = {"kind": "CONTINUE"}
    _maybe_attach_asha_advisory(result, [_assess(1, 0.9)], 0)
    assert result == {"kind": "CONTINUE"}  # advisory failed → decision untouched


def _cohort4():
    # Four scored branches, spread 0.6 ≫ noise floor → the halving path runs.
    return [_assess(1, 0.9), _assess(2, 0.7), _assess(3, 0.5), _assess(4, 0.3)]


def test_width_meter_a100_cap_binds(monkeypatch):
    # Generous $ budget but a hard 1-GPU cap ⇒ only the single best promotes,
    # DISTINCT from the eta fallback (which would promote ceil(4/3)=2).
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "1")
    result = {"kind": "CONTINUE"}
    _maybe_attach_asha_advisory(
        result, _cohort4(), 0,
        max_gpu_usd=1000.0, gpu_usd_spent=4.0, max_gpu_count=1,
    )
    actions = [d["action"] for d in result["asha_advisory"]["decisions"]]
    assert actions.count("promote") == 1  # A100 cap forces top-1
    assert actions.count("freeze") == 3
    assert "kill" not in actions  # underperformance never kills


def test_width_meter_gpu_usd_budget_binds(monkeypatch):
    # Budget 3.5 at ~1.0 $/branch ⇒ k=3, DISTINCT from the eta fallback's k=2.
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "1")
    result = {"kind": "CONTINUE"}
    _maybe_attach_asha_advisory(
        result, _cohort4(), 0,
        max_gpu_usd=7.5, gpu_usd_spent=4.0, max_gpu_count=100,
    )
    actions = [d["action"] for d in result["asha_advisory"]["decisions"]]
    assert actions.count("promote") == 3  # (7.5-4.0) // (4.0/4) = 3
    assert actions.count("freeze") == 1


def test_width_meter_metadata_surfaced(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "1")
    result = {"kind": "CONTINUE"}
    _maybe_attach_asha_advisory(
        result, _cohort4(), 2,
        max_gpu_usd=10.0, gpu_usd_spent=4.0, max_gpu_count=8,
    )
    wm = result["asha_advisory"]["width_meter"]
    assert wm["gpu_usd_budget"] == 6.0  # 10.0 - 4.0 remaining
    assert wm["a100_cap"] == 8
    assert wm["gpu_usd_spent"] == 4.0
    assert result["asha_advisory"]["rung"] == 2  # fidelity meter kept separate


def test_width_inputs_absent_is_eta_fallback(monkeypatch):
    # No $ ceiling supplied (legacy positional call) ⇒ geometric eta width
    # (ceil(4/3)=2), and the width_meter reports an unmetered budget.
    monkeypatch.setenv("OPENRESEARCH_SCHEDULER_TREE", "1")
    result = {"kind": "CONTINUE"}
    _maybe_attach_asha_advisory(result, _cohort4(), 0)
    actions = [d["action"] for d in result["asha_advisory"]["decisions"]]
    assert actions.count("promote") == 2  # eta fallback, unchanged legacy behaviour
    assert result["asha_advisory"]["width_meter"]["gpu_usd_budget"] is None
