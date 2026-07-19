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
