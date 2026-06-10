"""Golden-fixture quadrant rail for the two-axis reproducibility verdict (U18).

This is the **deterministic stop condition** for the reproducibility-verdict
fan-out (handoff doc Part A §6 + A.1/A9).  It asserts:

  * the (fidelity x replication) quadrant cells (cases 6-10), and
  * the ADVERSARIAL false-contradiction inputs (A9) — every one must resolve to
    ``inconclusive`` or the correct quadrant, NEVER a false ``contradicted``.

The litmus test (case 7): a faithful build that refutes the paper must read
``(faithful, contradicted)`` and project a legacy verdict of ``reproduced`` —
never ``failed``.  If this file regresses, the design has collapsed back to the
PaperBench single-score assumption.

Pure + deterministic: no GPU, no LLM, no I/O.
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.reproducibility_verdict import (
    ComparisonSpec,
    FidelityCertificate,
    MeasuredClaim,
    ScopeTuple,
    SeedBundle,
    compute_reproducibility_verdict,
    effect_confidence_interval,
    fidelity_rank,
)

# --------------------------------------------------------------------------- #
# Builders — sensible "clean faithful" defaults; each test varies one axis.
# --------------------------------------------------------------------------- #

_SDAR_SCOPE = ScopeTuple(model="Qwen2.5-3B", dataset="ALFWorld", split="test")


def _cert(
    *,
    invariant_tests_passed: bool = True,
    mutation_confirmed: bool = True,
    blinded_extraction_agreed: bool = True,
    profile_satisfied: bool = True,
    obligation_profile: str = "end_to_end",
    has_measured_metrics: bool = True,
) -> FidelityCertificate:
    return FidelityCertificate(
        invariant_tests_passed=invariant_tests_passed,
        mutation_confirmed=mutation_confirmed,
        blinded_extraction_agreed=blinded_extraction_agreed,
        obligation_profile=obligation_profile,  # type: ignore[arg-type]
        profile_satisfied=profile_satisfied,
        has_measured_metrics=has_measured_metrics,
    )


def _spec(
    *,
    claim_id: str = "primary",
    is_primary: bool = True,
    claimed_effect: float = 9.4,
    equivalence_margin: float = 1.0,
    direction: str = "higher_is_better",
    estimate_kind: str = "percentage_points",
    scope: ScopeTuple = _SDAR_SCOPE,
    ambiguous: bool = False,
    ambiguity_reason: str = "",
) -> ComparisonSpec:
    return ComparisonSpec(
        claim_id=claim_id,
        description="SDAR beats GRPO on ALFWorld",
        metric_name="success_rate",
        direction=direction,  # type: ignore[arg-type]
        estimate_kind=estimate_kind,  # type: ignore[arg-type]
        baseline_label="GRPO",
        claimed_effect=claimed_effect,
        equivalence_margin=equivalence_margin,
        scope=scope,
        is_primary=is_primary,
        ambiguous=ambiguous,
        ambiguity_reason=ambiguity_reason,
    )


def _bundle(
    effects: tuple[float, ...] = (9.0, 9.2),
    *,
    seeds: tuple[int, ...] | None = None,
    rng_independent: bool = True,
) -> SeedBundle:
    if seeds is None:
        seeds = tuple(range(42, 42 + len(effects)))
    return SeedBundle(seeds=seeds, per_seed_effect=effects, rng_independent=rng_independent)


def _claim(
    spec: ComparisonSpec | None = None,
    bundle: SeedBundle | None = None,
    measured_scope: ScopeTuple = _SDAR_SCOPE,
) -> MeasuredClaim:
    return MeasuredClaim(
        comparison=spec or _spec(),
        seed_bundle=bundle or _bundle(),
        measured_scope=measured_scope,
    )


def _verdict(claims=None, *, fidelity_score=0.9, certificate=None, **kw):
    return compute_reproducibility_verdict(
        fidelity_score=fidelity_score,
        certificate=certificate or _cert(),
        claims=claims if claims is not None else [_claim()],
        **kw,
    )


# --------------------------------------------------------------------------- #
# §6 quadrant rail — cases 6-10
# --------------------------------------------------------------------------- #

def test_case6_faithful_replicated():
    """Faithful build, metrics match the claim (>=2 seeds) -> (faithful, replicated)."""
    v = _verdict([_claim(bundle=_bundle((9.0, 9.2)))])
    assert v.implementation_verdict == "faithful"
    assert v.replication_verdict == "replicated"
    assert v.replication_credit > 0.9
    assert v.legacy_verdict == "reproduced"


def test_case7_faithful_contradicted_is_NOT_failed():
    """THE LITMUS TEST: faithful build that refutes the paper.

    Must read (faithful, contradicted) and project legacy 'reproduced' — never
    'failed'.  This is the single assertion the whole design exists to make true.
    """
    v = _verdict([_claim(bundle=_bundle((-3.0, -2.8)))])  # method clearly worse
    assert v.implementation_verdict == "faithful"
    assert v.replication_verdict == "contradicted"
    assert v.legacy_verdict == "reproduced"  # <- NOT "failed"
    assert v.legacy_verdict != "failed"


def test_case8_faithful_partially_replicated():
    """Right direction, magnitude far short of the claim -> partially-replicated."""
    v = _verdict([_claim(bundle=_bundle((3.0, 3.2)))])  # ~1/3 of the claimed 9.4
    assert v.implementation_verdict == "faithful"
    assert v.replication_verdict == "partially-replicated"
    # significant credit for correct direction, but clearly < a full replication
    assert 0.5 <= v.replication_credit < 0.9


def test_case9a_broken_certificate_is_inconclusive():
    """Certificate red (invariant tests failed) -> (broken, inconclusive)."""
    v = _verdict(
        [_claim(bundle=_bundle((-3.0, -2.8)))],
        certificate=_cert(invariant_tests_passed=False),
    )
    assert v.implementation_verdict == "broken"
    assert v.replication_verdict == "inconclusive"
    assert v.legacy_verdict == "failed"


def test_case9b_no_metrics_is_broken_inconclusive():
    v = _verdict(
        [_claim(bundle=_bundle((-3.0, -2.8)))],
        certificate=_cert(has_measured_metrics=False),
    )
    assert v.implementation_verdict == "broken"
    assert v.replication_verdict == "inconclusive"


def test_case10_leaderboard_ranks_faithful_above_broken():
    """A faithful-contradicted run (excellent reproduction, negative finding)
    ranks ABOVE a broken-but-high-fidelity-score attempt; replication is a
    badge, not a rank penalty (decision 7 / A5)."""
    faithful_contradicted = _verdict([_claim(bundle=_bundle((-3.0, -2.8)))], fidelity_score=0.9)
    broken = _verdict(
        [_claim(bundle=_bundle((9.0, 9.2)))],
        certificate=_cert(invariant_tests_passed=False),
        fidelity_score=0.95,  # even with a higher raw score, broken sorts below
    )
    assert fidelity_rank(faithful_contradicted) > fidelity_rank(broken)


def test_replication_is_not_a_rank_penalty():
    """Among faithful runs, refuting the paper does not demote you below a
    faithful partial-replication with a lower fidelity score."""
    contradicted_hi_fidelity = _verdict([_claim(bundle=_bundle((-3.0, -2.8)))], fidelity_score=0.92)
    partial_lo_fidelity = _verdict([_claim(bundle=_bundle((3.0, 3.2)))], fidelity_score=0.70)
    assert fidelity_rank(contradicted_hi_fidelity) > fidelity_rank(partial_lo_fidelity)


# --------------------------------------------------------------------------- #
# Decision 1 — replication is inconclusive unless the build is faithful
# --------------------------------------------------------------------------- #

def test_decision1_low_fidelity_blocks_contradiction():
    """A clear contradicting effect from a non-faithful build (low fidelity
    score) -> inconclusive, NOT contradicted."""
    v = _verdict([_claim(bundle=_bundle((-3.0, -2.8)))], fidelity_score=0.40)
    assert v.implementation_verdict == "partial"
    assert v.replication_verdict == "inconclusive"


# --------------------------------------------------------------------------- #
# A9 — adversarial false-contradiction inputs (each must NOT be contradicted)
# --------------------------------------------------------------------------- #

def test_a9_1_ambiguous_spec_is_inconclusive():
    """Relative-vs-absolute (or any) extraction ambiguity -> inconclusive even
    when the numbers look like a contradiction."""
    spec = _spec(ambiguous=True, ambiguity_reason="pp vs relative-% undetermined")
    v = _verdict([_claim(spec=spec, bundle=_bundle((-5.0, -4.8)))])
    assert v.replication_verdict == "inconclusive"


def test_a9_4_scope_mismatch_7B_claim_on_3B_is_inconclusive():
    """THE worst false-contradiction path: a 7B-specific claim evaluated on our
    cost-bounded 3B run must be inconclusive, never contradicted (A2)."""
    spec = _spec(scope=ScopeTuple(model="Qwen2.5-7B", dataset="ALFWorld", split="test"))
    v = _verdict([_claim(
        spec=spec,
        bundle=_bundle((-3.0, -2.8)),  # inverts — but at the WRONG scope
        measured_scope=ScopeTuple(model="Qwen2.5-3B", dataset="ALFWorld", split="test"),
    )])
    assert v.replication_verdict == "inconclusive"
    assert any("scope mismatch" in cv.reason for cv in v.per_claim)


def test_a9_5_duplicated_seeds_cannot_contradict():
    """Two recorded seeds that are the same value -> 1 effective seed -> cannot
    contradict (A3 'two seeds may agree by chance / shared RNG')."""
    v = _verdict([_claim(bundle=_bundle((-3.0, -2.8), seeds=(42, 42)))])
    assert v.replication_verdict == "inconclusive"


def test_a9_5b_uncontrolled_rng_cannot_contradict():
    v = _verdict([_claim(bundle=_bundle((-3.0, -2.8), rng_independent=False))])
    assert v.replication_verdict == "inconclusive"


def test_a9_6_ci_crossing_zero_is_inconclusive():
    """Disagreeing seeds -> wide CI straddling the equivalence region ->
    inconclusive, even with a negative mean."""
    v = _verdict([_claim(bundle=_bundle((-3.0, 2.5)))])
    assert v.replication_verdict == "inconclusive"


def test_a9_7_decoy_implementation_caught_by_mutation():
    """Implementer-authored tests pass on a decoy, but mutation did not bite ->
    certificate not green -> partial -> inconclusive (A6b)."""
    v = _verdict(
        [_claim(bundle=_bundle((-3.0, -2.8)))],
        certificate=_cert(mutation_confirmed=False),
    )
    assert v.implementation_verdict == "partial"
    assert v.replication_verdict == "inconclusive"


def test_a9_7b_blinded_disagreement_blocks_decisive_verdict():
    """Blinded re-extraction disagreed with the frozen spec -> not faithful ->
    inconclusive (A6a)."""
    v = _verdict(
        [_claim(bundle=_bundle((-3.0, -2.8)))],
        certificate=_cert(blinded_extraction_agreed=False),
    )
    assert v.replication_verdict == "inconclusive"


def test_a9_3_lower_is_better_metric_replicates_correctly():
    """For a lower-is-better metric the Extractor folds direction into the sign,
    so a method that genuinely wins (lower loss) replicates — it is not mistaken
    for a contradiction."""
    spec = _spec(direction="lower_is_better", claimed_effect=0.5, equivalence_margin=0.05)
    v = _verdict([_claim(spec=spec, bundle=_bundle((0.48, 0.52)))])  # advantage realised
    assert v.replication_verdict == "replicated"


def test_a9_single_secondary_claim_never_makes_paper_contradicted():
    """A contradicted SECONDARY (non-primary) claim with no eligible primary ->
    paper-level inconclusive, not contradicted (top-K mining can't overgeneralize
    an ablation)."""
    secondary = _spec(claim_id="ablation", is_primary=False)
    v = _verdict([_claim(spec=secondary, bundle=_bundle((-3.0, -2.8)))])
    assert v.replication_verdict == "inconclusive"


# --------------------------------------------------------------------------- #
# effect_confidence_interval — the seed-CI primitive (A3)
# --------------------------------------------------------------------------- #

def test_ci_empty_and_single():
    assert effect_confidence_interval(()) == (0.0, 0.0, 0.0)
    assert effect_confidence_interval((4.2,)) == (4.2, 4.2, 4.2)


def test_ci_two_seeds_is_wide():
    """Two seeds give a deliberately wide interval (t df=1 = 12.7) so a
    confident contradiction needs a large, consistent effect."""
    mean, lo, hi = effect_confidence_interval((9.0, 9.2))
    assert mean == pytest.approx(9.1, abs=1e-9)
    assert lo < 9.0 < 9.2 < hi
    assert (hi - lo) > 2.0  # wide


def test_ci_tight_with_many_consistent_seeds():
    mean, lo, hi = effect_confidence_interval((5.0, 5.0, 5.0, 5.0, 5.0))
    assert mean == pytest.approx(5.0)
    assert lo == pytest.approx(5.0)
    assert hi == pytest.approx(5.0)
