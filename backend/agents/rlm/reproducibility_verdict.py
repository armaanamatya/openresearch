"""Two-axis reproducibility verdict — the deterministic decision spine.

This module is the **pure** core of the reproducibility-verdict design
(handoff doc ``2026-06-08-agent-codegen-tdd-hardening-handoff.md`` Part A + A.1).
It takes already-computed inputs (a fidelity score + certificate, and a list of
measured claims each carrying a typed ``ComparisonSpec`` + a seed bundle) and
decides the two axes:

  * ``implementation_verdict`` — did WE build the method faithfully?
      ``faithful`` | ``partial`` | ``broken``
  * ``replication_verdict``    — given a faithful build, did the PAPER's
      claimed result hold?  ``replicated`` | ``partially-replicated`` |
      ``contradicted`` | ``inconclusive``

It is **stdlib-only and side-effect-free** (mirrors ``cell_matrix.py``), so the
golden-fixture rail (``tests/rlm/test_reproducibility_verdict.py``) can assert
every (fidelity x replication) quadrant — *and every adversarial false-
contradiction input* — deterministically, with no GPU and no LLM.

Why a pure module: a ``contradicted`` verdict is a strong public scientific
claim ("this published paper does not replicate").  Whether the harness is
*entitled* to make it must be decided by code we can read and test, not by an
LLM's say-so.  The LLM surfaces (Extractor, blinded verifier, grader) *produce
the inputs*; this module *adjudicates* them under the locked rules.

Locked rules encoded here (Part A / A.1):
  - A1  typed ``ComparisonSpec``; any ambiguity → the claim is ``inconclusive``.
  - A2  claim-scope eligibility: a result whose run-scope does not match the
        claim's scope can NEVER contradict it (the smallest-two-vs-7B bug).
  - A3  seed bundle: contradiction requires the effect CI to EXCLUDE a
        claim-specific equivalence region; ``>=2`` seeds necessary, not
        sufficient.  Credit is continuous in the recovered-effect fraction.
  - A4  schema-versioned: the legacy ``verdict`` projection is derived from the
        FIDELITY axis only (so a faithful-contradicted run never collapses to
        ``failed`` via the blended-score reconcile).
  - A7  certificate obligation profile gates which evidence is sufficient.
  - decision 1  replication is ``inconclusive`` unless the build is ``faithful``.

The decision is intentionally *conservative*: when the evidence cannot cleanly
distinguish "the paper is wrong" from "we could be wrong", it returns
``inconclusive`` rather than risk a false ``contradicted``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

__all__ = [
    "MetricDirection",
    "EstimateKind",
    "ObligationProfile",
    "ImplementationVerdict",
    "ReplicationVerdict",
    "ScopeTuple",
    "ComparisonSpec",
    "SeedBundle",
    "MeasuredClaim",
    "FidelityCertificate",
    "ClaimVerdict",
    "ReproducibilityVerdict",
    "SCHEMA_VERSION",
    "compute_reproducibility_verdict",
    "effect_confidence_interval",
    "fidelity_rank",
]

# Report schema version that carries the two-axis verdict.  Reports at this
# version SKIP the legacy aggregate-score reconcile (A4); older reports keep the
# current ``reconcile_verdict_with_score`` logic untouched.
SCHEMA_VERSION = 2

MetricDirection = Literal["higher_is_better", "lower_is_better"]
EstimateKind = Literal["percentage_points", "relative_percent", "absolute"]
# A7 — what evidence is sufficient for a decisive verdict on this claim.
ObligationProfile = Literal["static", "forward_pass", "multi_step", "trace", "end_to_end"]
ImplementationVerdict = Literal["faithful", "partial", "broken"]
ReplicationVerdict = Literal["replicated", "partially-replicated", "contradicted", "inconclusive"]

# Default fidelity-score floor for a "faithful" implementation verdict.  The
# certificate (executable invariant tests + mutation + blinded agreement) is the
# *primary* gate; this score floor is a secondary guard so a green certificate
# on an otherwise-threadbare rubric still can't mint "faithful".
DEFAULT_FAITHFUL_MIN_SCORE = 0.60
# Profiles for which a single seed can support a *positive* verdict (the claim
# is deterministic).  Contradiction ALWAYS needs >=2 seeds with a CI (A3).
_DETERMINISTIC_PROFILES: frozenset[str] = frozenset({"static", "forward_pass"})

# Two-sided 95% Student-t critical values by degrees of freedom (n-1).  Embedded
# so the module stays stdlib-only; df>=11 falls back to the normal 1.96.  With
# n=2 (df=1) t=12.71 makes the CI very wide — which is the point: two seeds
# rarely license a confident contradiction (A3).
_T_CRIT_95: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
}
_T_CRIT_INF = 1.96


def effect_confidence_interval(values: tuple[float, ...]) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` for a per-seed effect sample.

    A two-sided 95% interval using the Student-t critical value for ``n-1``
    degrees of freedom (normal 1.96 for ``n>10``).  A single value yields a
    degenerate point interval ``(v, v, v)`` — callers must apply the
    ``>=2`` seed rule (A3) separately; a point interval is never sufficient for
    a contradiction on a non-deterministic profile.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = math.fsum(values) / n
    if n == 1:
        return (mean, mean, mean)
    var = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    sem = math.sqrt(var) / math.sqrt(n)
    t = _T_CRIT_95.get(n - 1, _T_CRIT_INF)
    half = t * sem
    return (mean, mean - half, mean + half)


@dataclass(frozen=True)
class ScopeTuple:
    """The exact scope at which a claim is made / a result was produced.

    Equality is normalized (case/space-insensitive).  ``matches`` is used by the
    A2 eligibility gate: a claim can only be decisively graded by a result whose
    scope matches on every populated axis.  An empty axis ("") is a wildcard on
    BOTH sides (we don't know → don't block), so legacy single-config runs are
    not spuriously ruled ineligible.
    """

    model: str = ""
    dataset: str = ""
    split: str = ""
    protocol: str = ""

    @staticmethod
    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    def _axis_matches(self, a: str, b: str) -> bool:
        na, nb = self._norm(a), self._norm(b)
        return na == "" or nb == "" or na == nb

    def matches(self, other: "ScopeTuple") -> bool:
        return (
            self._axis_matches(self.model, other.model)
            and self._axis_matches(self.dataset, other.dataset)
            and self._axis_matches(self.split, other.split)
            and self._axis_matches(self.protocol, other.protocol)
        )

    def describe(self) -> str:
        parts = [p for p in (self.model, self.dataset, self.split, self.protocol) if p]
        return "/".join(parts) if parts else "(unspecified)"


@dataclass(frozen=True)
class ComparisonSpec:
    """A typed, paper-cited comparison the result will be graded against (A1).

    ``claimed_effect`` is expressed in ``estimate_kind`` units and signed so that
    a POSITIVE value is the paper's claimed advantage for the proposed method
    over ``baseline_label`` (regardless of ``direction`` — the Extractor folds
    higher/lower-is-better into the sign).  ``equivalence_margin`` is the half-
    width of the practical-equivalence region around zero effect: a measured
    effect inside it is "no meaningful difference".

    ``ambiguous=True`` (with a reason) is the Extractor's honest escape hatch:
    if it cannot pin estimate kind / unit / metric / coordinates, the claim is
    forced to ``inconclusive`` rather than risk a misbound false contradiction.
    """

    claim_id: str
    description: str
    metric_name: str
    direction: MetricDirection
    estimate_kind: EstimateKind
    baseline_label: str
    claimed_effect: float
    equivalence_margin: float
    scope: ScopeTuple
    is_primary: bool = False
    table_ref: str = ""
    paper_span: str = ""  # the cited quote/offset the constant traces to (provenance)
    ambiguous: bool = False
    ambiguity_reason: str = ""

    def __post_init__(self) -> None:
        if self.equivalence_margin < 0:
            raise ValueError("equivalence_margin must be >= 0")


@dataclass(frozen=True)
class SeedBundle:
    """Per-seed measured effects + the independence evidence required by A3.

    ``per_seed_effect`` is in the SAME units/sign convention as
    ``ComparisonSpec.claimed_effect``.  ``rng_independent`` records whether the
    seeds were verified to use distinct, controlled RNG states — duplicated or
    uncontrolled seeds do not count toward the ``>=2`` rule.
    """

    seeds: tuple[int, ...] = ()
    per_seed_effect: tuple[float, ...] = ()
    rng_independent: bool = False

    @property
    def n_effective(self) -> int:
        """Number of usable, independent seeds.

        Requires ``rng_independent`` AND distinct seed values AND one effect per
        seed.  Anything short of that collapses to ``<2`` so it cannot support a
        contradiction (the "two seeds agree by chance / shared RNG" hole).
        """
        if not self.rng_independent:
            return 0
        if len(self.seeds) != len(self.per_seed_effect):
            return 0
        return len(set(self.seeds))

    def interval(self) -> tuple[float, float, float]:
        return effect_confidence_interval(self.per_seed_effect)


@dataclass(frozen=True)
class MeasuredClaim:
    """A ``ComparisonSpec`` paired with what we actually ran + measured."""

    comparison: ComparisonSpec
    seed_bundle: SeedBundle
    measured_scope: ScopeTuple


@dataclass(frozen=True)
class FidelityCertificate:
    """The executable fidelity certificate (A6/A7) — the gate on the verdict.

    ``invariant_tests_passed`` — verifier-owned tests against the production
    entry points ran green.  ``mutation_confirmed`` — perturbing the registered
    constants made those tests fail (they bite; not tautological).
    ``blinded_extraction_agreed`` — a blinded re-extraction from raw paper spans
    agreed with the frozen spec (A6a).  ``obligation_profile`` /
    ``profile_satisfied`` — the evidence demanded for this paper's claim type
    was produced (A7).
    """

    invariant_tests_passed: bool
    mutation_confirmed: bool
    blinded_extraction_agreed: bool
    obligation_profile: ObligationProfile
    profile_satisfied: bool
    has_measured_metrics: bool = True
    # Whether the executable invariant tests actually RAN.  Distinguishes
    # "ran and failed" (demonstrably unfaithful → broken) from "did not run"
    # (uncertified → partial).  Defaults True so an explicitly-built green
    # certificate is green without restating it.
    invariant_tests_ran: bool = True

    @property
    def is_green(self) -> bool:
        return (
            self.has_measured_metrics
            and self.invariant_tests_ran
            and self.invariant_tests_passed
            and self.mutation_confirmed
            and self.blinded_extraction_agreed
            and self.profile_satisfied
        )


@dataclass(frozen=True)
class ClaimVerdict:
    """Per-claim outcome.  ``status`` mirrors the paper-level vocabulary."""

    claim_id: str
    status: ReplicationVerdict
    credit: float
    reason: str
    measured_mean: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    eligible: bool = False


@dataclass(frozen=True)
class ReproducibilityVerdict:
    """The full two-axis result."""

    implementation_verdict: ImplementationVerdict
    replication_verdict: ReplicationVerdict
    replication_credit: float
    legacy_verdict: str  # projected from fidelity (A4): reproduced|partial|failed
    schema_version: int
    per_claim: tuple[ClaimVerdict, ...] = ()
    rationale: tuple[str, ...] = field(default=())
    fidelity_score: float = 0.0


# --------------------------------------------------------------------------- #
# Decision logic
# --------------------------------------------------------------------------- #

def _implementation_verdict(
    fidelity_score: float,
    certificate: FidelityCertificate,
    faithful_min: float,
) -> ImplementationVerdict:
    if not certificate.has_measured_metrics:
        # No trustworthy metrics — we cannot assess the implementation at all.
        return "broken"
    if certificate.invariant_tests_ran and not certificate.invariant_tests_passed:
        # Executable invariants RAN and FAILED → the code is demonstrably unfaithful.
        return "broken"
    if certificate.is_green and fidelity_score >= faithful_min:
        return "faithful"
    # Ran + has metrics, but not executably certified faithful (e.g. tests never
    # ran, mutation didn't confirm, blinded re-extraction disagreed).
    return "partial"


def _grade_claim(
    claim: MeasuredClaim,
    *,
    min_seeds_for_contradiction: int,
) -> ClaimVerdict:
    spec = claim.comparison
    bundle = claim.seed_bundle

    # A1 — ambiguous extraction can never decide a claim.
    if spec.ambiguous:
        return ClaimVerdict(
            claim_id=spec.claim_id,
            status="inconclusive",
            credit=0.0,
            reason=f"ambiguous comparison spec: {spec.ambiguity_reason or 'unspecified'}",
            eligible=False,
        )

    # A2 — scope eligibility.  A result at a non-matching scope (e.g. 3B run vs a
    # 7B-specific claim) cannot decide the claim in EITHER direction.
    if not claim.measured_scope.matches(spec.scope):
        return ClaimVerdict(
            claim_id=spec.claim_id,
            status="inconclusive",
            credit=0.0,
            reason=(
                f"scope mismatch: claim@{spec.scope.describe()} "
                f"vs ran@{claim.measured_scope.describe()} — cannot grade"
            ),
            eligible=False,
        )

    n = bundle.n_effective
    mean, lo, hi = bundle.interval()
    margin = spec.equivalence_margin

    # A POSITIVE verdict may stand on a single seed for a deterministic claim
    # (the obligation profile is enforced at the certificate level, A7); a
    # CONTRADICTION always requires >=2 independent seeds (A3, enforced below).
    if n < 1:
        return ClaimVerdict(
            claim_id=spec.claim_id, status="inconclusive", credit=0.0,
            reason="no usable independent seeds (rng_independent false or seeds duplicated)",
            measured_mean=None, eligible=True,
        )

    # Decisive bands (sign convention: positive effect = claimed advantage).
    #   whole CI above +margin              → real effect in claimed direction
    #   whole CI below -margin              → effect inverts (contradiction)
    #   CI straddles the equivalence region → cannot distinguish from null
    recovered = mean / spec.claimed_effect if spec.claimed_effect else 0.0

    if lo > margin:
        # Replicated direction (the whole CI clears the equivalence region, so a
        # real, correctly-directed effect exists).  Credit = a DIRECTION base
        # (0.5, since the qualitative result held) + a continuous MAGNITUDE term
        # (up to 0.5, by recovered-effect fraction).  This gives a correctly-
        # directed result "significant credit" while keeping ordering monotone
        # in magnitude — an honest near-miss outranks a trivial +epsilon effect
        # (Codex finding 10 vs the operator's "~0.75 for right-direction").
        recovered_clamped = max(0.0, min(1.0, recovered))
        credit = 0.5 + 0.5 * recovered_clamped
        status: ReplicationVerdict = (
            "replicated" if recovered_clamped >= 0.90 else "partially-replicated"
        )
        return ClaimVerdict(
            claim_id=spec.claim_id, status=status, credit=round(credit, 4),
            reason=(
                f"effect CI [{lo:.4g}, {hi:.4g}] exceeds equivalence ±{margin:.4g}; "
                f"recovered {recovered:.0%} of claimed {spec.claimed_effect:.4g}"
            ),
            measured_mean=mean, ci_low=lo, ci_high=hi, eligible=True,
        )

    if hi < -margin:
        # Contradiction — but only with enough independent seeds (A3).
        if n < min_seeds_for_contradiction:
            return ClaimVerdict(
                claim_id=spec.claim_id, status="inconclusive", credit=0.0,
                reason=(
                    f"effect CI inverts (≤ -{margin:.4g}) but only {n} independent "
                    f"seed(s) (< {min_seeds_for_contradiction}) — cannot rule out variance"
                ),
                measured_mean=mean, ci_low=lo, ci_high=hi, eligible=True,
            )
        return ClaimVerdict(
            claim_id=spec.claim_id, status="contradicted", credit=0.0,
            reason=(
                f"effect CI [{lo:.4g}, {hi:.4g}] lies beyond the equivalence region "
                f"on the OPPOSITE side of the claim ({n} independent seeds)"
            ),
            measured_mean=mean, ci_low=lo, ci_high=hi, eligible=True,
        )

    # CI overlaps the equivalence region / crosses zero → indistinguishable.
    return ClaimVerdict(
        claim_id=spec.claim_id, status="inconclusive", credit=0.0,
        reason=(
            f"effect CI [{lo:.4g}, {hi:.4g}] overlaps the equivalence region "
            f"±{margin:.4g} — not distinguishable from no-effect"
        ),
        measured_mean=mean, ci_low=lo, ci_high=hi, eligible=True,
    )


def _legacy_verdict_from_fidelity(impl: ImplementationVerdict) -> str:
    # A4 — the legacy `verdict` projection comes from the FIDELITY axis only, so
    # a faithful-contradicted run never collapses to "failed" via the blended
    # `overall_score` reconcile.  Two-axis (schema>=2) reports use this and SKIP
    # `reconcile_verdict_with_score`.
    return {"faithful": "reproduced", "partial": "partial", "broken": "failed"}[impl]


def compute_reproducibility_verdict(
    *,
    fidelity_score: float,
    certificate: FidelityCertificate,
    claims: list[MeasuredClaim],
    min_seeds_for_contradiction: int = 2,
    faithful_min: float = DEFAULT_FAITHFUL_MIN_SCORE,
) -> ReproducibilityVerdict:
    """Adjudicate the two-axis verdict from pre-computed inputs.

    See the module docstring for the locked rules.  This function never raises
    on plausible inputs and is fully deterministic.
    """
    rationale: list[str] = []

    impl = _implementation_verdict(fidelity_score, certificate, faithful_min)
    rationale.append(
        f"implementation={impl} (fidelity_score={fidelity_score:.3f}, "
        f"certificate_green={certificate.is_green}, profile={certificate.obligation_profile})"
    )

    per_claim = tuple(
        _grade_claim(c, min_seeds_for_contradiction=min_seeds_for_contradiction)
        for c in claims
    )

    # Decision 1 — replication is meaningless without a faithful build.  A
    # contradicted/replicated claim from a non-faithful run is downgraded to
    # inconclusive (we can't tell the paper from our own bug).
    if impl != "faithful":
        rationale.append(
            "replication=inconclusive — build is not certified faithful, so no "
            "claim about the paper can be trusted (gating decision 1)"
        )
        return ReproducibilityVerdict(
            implementation_verdict=impl,
            replication_verdict="inconclusive",
            replication_credit=0.0,
            legacy_verdict=_legacy_verdict_from_fidelity(impl),
            schema_version=SCHEMA_VERSION,
            fidelity_score=fidelity_score,
            per_claim=per_claim,
            rationale=tuple(rationale),
        )

    # Paper-level rollup is driven by the PRIMARY claim (A2).  Secondary claims
    # are reported per-claim but never escalate to a paper-level contradiction.
    primary = [cv for cv, c in zip(per_claim, claims) if c.comparison.is_primary]
    if not primary:
        rationale.append("replication=inconclusive — no eligible primary claim to grade")
        replication: ReplicationVerdict = "inconclusive"
        credit = 0.0
    else:
        # If multiple primaries, the weakest decisive outcome governs (a paper
        # is only "replicated" if its headline holds; a single contradicted
        # eligible primary makes the paper contradicted).
        pv = primary[0] if len(primary) == 1 else _rollup_primaries(primary)
        replication = pv.status
        credit = pv.credit
        rationale.append(f"replication={replication} from primary claim {pv.claim_id}: {pv.reason}")

    return ReproducibilityVerdict(
        implementation_verdict=impl,
        replication_verdict=replication,
        replication_credit=round(credit, 4),
        legacy_verdict=_legacy_verdict_from_fidelity(impl),
        schema_version=SCHEMA_VERSION,
        fidelity_score=fidelity_score,
        per_claim=per_claim,
        rationale=tuple(rationale),
    )


# Order of severity for rolling up multiple primary claims: a contradiction is
# the strongest paper-level signal, then inconclusive, then partial, then full.
_ROLLUP_ORDER: dict[str, int] = {
    "contradicted": 0, "inconclusive": 1, "partially-replicated": 2, "replicated": 3,
}


def _rollup_primaries(primaries: list[ClaimVerdict]) -> ClaimVerdict:
    """Combine multiple primary-claim verdicts into one (weakest governs)."""
    return min(primaries, key=lambda cv: _ROLLUP_ORDER.get(cv.status, 1))


_IMPL_RANK: dict[str, int] = {"faithful": 2, "partial": 1, "broken": 0}


def fidelity_rank(verdict: ReproducibilityVerdict) -> tuple[int, float]:
    """Leaderboard / best-attempt sort key (A5).  Higher sorts first.

    Ranks by the FIDELITY axis: a ``faithful`` run outranks any non-faithful one
    regardless of replication outcome — so a faithful-contradicted run (an
    excellent reproduction carrying a negative finding) ranks ABOVE a
    broken-high-rubric-score run, and the replication outcome is a *badge*, not
    a rank penalty (decision 7).  The secondary key is the fidelity score, NOT
    the replication credit, so refuting a paper never demotes a faithful run.
    """
    return (_IMPL_RANK.get(verdict.implementation_verdict, 0), round(verdict.fidelity_score, 6))
