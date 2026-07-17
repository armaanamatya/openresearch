"""Track E Task 1: the offline reproduction composite is deterministic-dominated
and structurally confined off the verdict surface.

The north-star invariant (CLAUDE.md, spec §3.1/§6.2): the composite is a
report/rank-only stat that lives entirely inside ``backend/evals/`` and can
NEVER reach ``final_report.verdict`` / ``campaign_policy.decide`` — the discrete
evidence verdict (``verdict_authority.decide``) stays the sole gate. This suite
locks the deterministic-dominated re-weight AND the off-verdict-surface
containment.
"""

from __future__ import annotations

from pathlib import Path

from backend.evals.schemas import DEFAULT_COMPOSITE_WEIGHTS, ReproductionScore

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_weights_sum_to_one():
    w = DEFAULT_COMPOSITE_WEIGHTS
    assert abs(w.build + w.run + w.metric_match + w.fidelity - 1.0) < 1e-9


def test_composite_is_deterministic_dominated_fidelity_zero_by_default():
    # A run with a HIGH LLM fidelity but ZERO deterministic evidence must score
    # ~0 — the LLM term cannot lift the default composite.
    s = ReproductionScore(
        build_success=False, run_success=False, metric_match=0.0, fidelity_score=1.0
    )
    assert s.composite_score() == 0.0


def test_composite_dominated_by_metric_match():
    s = ReproductionScore(
        build_success=True, run_success=True, metric_match=1.0, fidelity_score=0.0
    )
    assert abs(s.composite_score() - 1.0) < 1e-9


def test_component_fields_preserved_separately():
    # The re-weight must not collapse the components — every field stays readable.
    s = ReproductionScore(fidelity_score=0.77, metric_match=0.5)
    assert s.fidelity_score == 0.77 and s.metric_match == 0.5


def test_custom_weights_override_for_ab_tuning():
    from backend.evals.schemas import CompositeWeights

    s = ReproductionScore(fidelity_score=1.0)
    # A caller can opt back into a fidelity-weighted blend for A/B tuning; the
    # DEFAULT stays deterministic-dominated.
    assert s.composite_score(CompositeWeights(0.0, 0.0, 0.0, 1.0)) == 1.0


def test_composite_never_reaches_verdict_surface():
    # Structural invariant: the composite lives in ``backend.evals.schemas`` and
    # is persisted only by ``backend.evals.store``. No verdict / campaign-DECIDE
    # module may import EITHER, so ``composite_score`` is structurally unable to
    # reach ``final_report.verdict`` or campaign DECIDE. (``report.py`` legitimately
    # imports ``backend.evals.paperbench.leaf_scorer`` — the rubric grader, a
    # DIFFERENT module whose verdict-minting was already severed in Track A — so we
    # ban the composite's home + store specifically, not all of ``backend.evals``.)
    banned = ("backend.evals.schemas", "backend.evals.store")
    for mod in (
        "backend/agents/rlm/report.py",
        "backend/agents/rlm/verdict_authority.py",
        "backend/agents/rlm/campaign_policy.py",
    ):
        src = (_REPO_ROOT / mod).read_text(encoding="utf-8")
        for banned_mod in banned:
            assert banned_mod not in src, (
                f"{mod} must not import {banned_mod} — the composite is report/rank-only "
                "and can never reach the verdict surface (spec §6.2)"
            )
