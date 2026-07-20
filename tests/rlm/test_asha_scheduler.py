"""ASHA successive-halving decision core (pure logic).

Covers every decision path: breakage true-kill, discovery quarantine, safety-
bracket exemption, N=1 degradation, the envelope-shrink gate, top-k halving with
the GPU-$/A100 width meter (+ eta fallback), and the freeze-never-kill /
faithful-kept-by-fidelity invariants.
"""
from backend.agents.rlm.asha_scheduler import (
    BranchObservation,
    RungConfig,
    asha_decide,
)


def _by_id(decisions):
    return {d.branch_id: d for d in decisions}


def test_broken_branch_is_true_killed():
    d = _by_id(asha_decide([BranchObservation("a", broken=True, score=0.9)], RungConfig()))
    assert d["a"].action == "kill"
    assert "breakage" in d["a"].reason


def test_broken_faithful_branch_still_killed():
    # Breakage = provably-broken code → true-delete even for a faithful branch.
    d = _by_id(asha_decide(
        [BranchObservation("f", branch_type="faithful", broken=True, score=0.9)],
        RungConfig(),
    ))
    assert d["f"].action == "kill"


def test_discovery_branch_is_promoted_not_halved():
    d = _by_id(asha_decide([
        BranchObservation("disc", branch_type="discovery", score=0.1),
        BranchObservation("f1", score=0.9, gpu_usd=1.0),
        BranchObservation("f2", score=0.2, gpu_usd=1.0),
    ], RungConfig(gpu_usd_budget=1.0, higher_is_better=True, noise_floor=0.0)))
    assert d["disc"].action == "promote"
    assert d["disc"].reason == "discovery_quarantined"


def test_safety_bracket_exempt_from_halving():
    d = _by_id(asha_decide([
        BranchObservation("sb", is_safety_bracket=True, score=0.05, gpu_usd=1.0),
        BranchObservation("f1", score=0.9, gpu_usd=1.0),
        BranchObservation("f2", score=0.8, gpu_usd=1.0),
    ], RungConfig(gpu_usd_budget=1.0, noise_floor=0.0)))
    assert d["sb"].action == "promote"  # never halved despite the worst score
    assert d["sb"].reason == "safety_bracket_exempt"


def test_n1_single_branch_promotes():
    d = _by_id(asha_decide([BranchObservation("solo", score=0.4)], RungConfig()))
    assert d["solo"].action == "promote"


def test_envelope_not_shrunk_holds_all():
    # spread (0.02) <= noise_floor (0.05) → don't score-cull. The declared
    # width is sufficient for both, so both are allowed to continue.
    d = _by_id(asha_decide([
        BranchObservation("a", score=0.50, gpu_usd=1.0),
        BranchObservation("b", score=0.52, gpu_usd=1.0),
    ], RungConfig(gpu_usd_budget=2.0, noise_floor=0.05)))
    assert d["a"].action == "promote"
    assert d["b"].action == "promote"
    assert d["a"].reason == "envelope_not_shrunk"


def test_explicit_zero_width_freezes_even_a_single_scored_branch():
    d = _by_id(asha_decide(
        [BranchObservation("solo", score=0.4, gpu_usd=1.0)],
        RungConfig(gpu_usd_budget=0.0),
    ))
    assert d["solo"].action == "freeze"
    assert d["solo"].reason == "width_meter_exhausted"


def test_halving_promotes_top_k_freezes_rest():
    # budget 1.0, cost 1.0/branch → k=1; 3 branches → promote best, freeze 2.
    d = _by_id(asha_decide([
        BranchObservation("hi", score=0.9, gpu_usd=1.0),
        BranchObservation("mid", score=0.5, gpu_usd=1.0),
        BranchObservation("lo", score=0.1, gpu_usd=1.0),
    ], RungConfig(gpu_usd_budget=1.0, higher_is_better=True, noise_floor=0.0)))
    assert d["hi"].action == "promote"
    assert d["mid"].action == "freeze"
    assert d["lo"].action == "freeze"
    assert all(x.action != "kill" for x in d.values())  # underperformance never kills


def test_faithful_underperformer_frozen_never_killed():
    d = _by_id(asha_decide([
        BranchObservation("good", branch_type="faithful", score=0.9, gpu_usd=1.0),
        BranchObservation("bad", branch_type="faithful", score=0.01, gpu_usd=1.0),
    ], RungConfig(gpu_usd_budget=1.0, noise_floor=0.0)))
    assert d["bad"].action == "freeze"  # kept by fidelity — never killed for a low score


def test_width_from_a100_cap_clamps():
    # budget allows more, but a100_cap=2 → only the top-2 promote.
    obs = [BranchObservation(f"b{i}", score=float(i), gpu_usd=1.0) for i in range(4)]
    d = _by_id(asha_decide(
        obs, RungConfig(gpu_usd_budget=10.0, a100_cap=2, higher_is_better=True, noise_floor=0.0)
    ))
    promoted = {k for k, v in d.items() if v.action == "promote"}
    assert promoted == {"b3", "b2"}  # top-2 by score, capped by A100 count


def test_eta_fallback_when_no_budget():
    # no budget/cost → geometric: 6 branches, eta 3 → ceil(6/3)=2 promote.
    obs = [BranchObservation(f"b{i}", score=float(i)) for i in range(6)]
    d = _by_id(asha_decide(obs, RungConfig(eta=3.0, higher_is_better=True, noise_floor=0.0)))
    assert sum(1 for v in d.values() if v.action == "promote") == 2


def test_lower_is_better_direction():
    d = _by_id(asha_decide([
        BranchObservation("low_loss", score=0.1, gpu_usd=1.0),
        BranchObservation("high_loss", score=2.0, gpu_usd=1.0),
    ], RungConfig(gpu_usd_budget=1.0, higher_is_better=False, noise_floor=0.0)))
    assert d["low_loss"].action == "promote"
    assert d["high_loss"].action == "freeze"


def test_none_score_branch_frozen():
    d = _by_id(asha_decide([
        BranchObservation("scored", score=0.8, gpu_usd=1.0),
        BranchObservation("noscore", score=None, gpu_usd=1.0),
        BranchObservation("scored2", score=0.2, gpu_usd=1.0),
    ], RungConfig(gpu_usd_budget=1.0, a100_cap=1, higher_is_better=True, noise_floor=0.0)))
    assert d["noscore"].action == "freeze"
    assert d["noscore"].reason == "no_score_at_rung"
    assert d["scored"].action == "promote"


def test_broken_plus_cohort_kills_only_broken():
    d = _by_id(asha_decide([
        BranchObservation("dead", broken=True, score=0.0, gpu_usd=1.0),
        BranchObservation("hi", score=0.9, gpu_usd=1.0),
        BranchObservation("lo", score=0.2, gpu_usd=1.0),
    ], RungConfig(gpu_usd_budget=1.0, higher_is_better=True, noise_floor=0.0)))
    assert d["dead"].action == "kill"
    assert d["hi"].action == "promote"
    assert d["lo"].action == "freeze"
