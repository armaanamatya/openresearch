"""WS1 acceptance: the reproduction verdict is now the honest deterministic one
(Task 9, ``docs/superpowers/specs/2026-07-09-eval-integrity-track-a-design.md`` §7,
corrected per review).

**Headline case.** The design doc's own §7 acceptance text says the frozen
``runs/prj_adam_local_1`` should re-grade to ``partial`` (it expected the SFO
primary claim to show up as a genuine, non-ambiguous primary that is simply
``unmeasured``). A review of the actual artifact corrected this: the ONE claim
in ``rlm_state/repro_spec.json`` is marked ``ambiguous=True`` (paper prose gave
no extractable numeric effect — "SFO is 5-10x slower per iteration" carries no
usable delta), so per the locked taxonomy (rule 1 of
``verdict_authority.decide``: no claim is genuinely ``is_primary``, or the
only primaries are ambiguous/unresolved -> ``inconclusive``,
``reason="no_measurable_target"``), the honest re-grade is **inconclusive**,
not ``reproduced`` and not ``partial``.

**A second, more important finding surfaced while building this test (see the
module-level comment on ``test_adam_headline_reground_is_inconclusive_not_reproduced``
below): the verdict comes out ``inconclusive`` for a DIFFERENT mechanical
reason than "the ambiguous flag correctly excluded the one primary claim" —
the on-disk claim shape does not match what ``result_fidelity.evaluate``
reads at all. Both paths land on the same rule-1 branch today, so the
label-level assertion still holds, but this is a real, load-bearing gap, not
just documentation trivia — see the docstring below and the task report.**

**Synthetic taxonomy fixtures.** The three non-Adam cases build a
``repro_spec`` directly in the *flat*, ``kind``-discriminated shape
``result_fidelity.evaluate``/``metric_binding.bind_claims`` actually consume
(reusing ``tests/agents/rlm/test_result_fidelity.py``'s ``_claim``/``_run``
helpers verbatim rather than duplicating them) and drive it through the REAL
``evaluate() -> decide()`` pipeline — this is what distinguishes an
*acceptance* test from the existing per-module unit tests, which either
exercise ``evaluate`` alone (``test_result_fidelity.py``) or hand-construct a
``ResultFidelity`` dict to exercise ``decide`` alone
(``test_verdict_authority.py``'s ``_rf``/``_c``). Those two helpers are
deliberately NOT reused here: they bypass ``evaluate()``, which is exactly the
seam this file needs to prove is wired correctly end-to-end.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from backend.agents.rlm.result_fidelity import evaluate
from backend.agents.rlm.verdict_authority import decide
from tests.agents.rlm.test_result_fidelity import _claim, _run

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ADAM_RUN_DIR = _REPO_ROOT / "runs" / "prj_adam_local_1"


def _copy_adam_artifacts_readonly(tmp_path: Path) -> Path:
    """Copy ONLY the two artifacts ``result_fidelity.evaluate`` reads
    (``rlm_state/repro_spec.json`` + ``code/metrics.json``) out of the frozen
    ``runs/prj_adam_local_1`` into an isolated ``tmp_path``.

    The real run directory is opened for reading exactly twice (here, and
    once more at the end of the headline test to prove it was never touched)
    and is never written to -- this test suite must never mutate a live run
    artifact under ``runs/``.
    """
    dest = tmp_path / "prj_adam_local_1"
    (dest / "rlm_state").mkdir(parents=True)
    (dest / "code").mkdir(parents=True)
    shutil.copy2(
        _ADAM_RUN_DIR / "rlm_state" / "repro_spec.json",
        dest / "rlm_state" / "repro_spec.json",
    )
    shutil.copy2(
        _ADAM_RUN_DIR / "code" / "metrics.json",
        dest / "code" / "metrics.json",
    )
    return dest


# --------------------------------------------------------------------------- #
# 1. ADAM headline -- the actual WS1 acceptance criterion
# --------------------------------------------------------------------------- #


def test_adam_headline_reground_is_inconclusive_not_reproduced(tmp_path):
    """Re-grading the frozen Adam artifact through the severed, grade-free
    pipeline (``result_fidelity.evaluate`` -> ``verdict_authority.decide``,
    no grade/``impl_fidelity`` input exists in ``decide``'s signature) must
    yield ``inconclusive``/``no_measurable_target`` -- never ``reproduced``.
    The shipped ``final_report.json`` on this same run says
    ``verdict="reproduced"`` from ``rubric_score=0.7848`` alone; this is
    exactly the inversion WS1 exists to fix.

    IMPORTANT finding (see the module docstring): ``rlm_state/repro_spec.json``
    on disk is written by the CURRENT ``repro_spec_extractor.build_repro_spec``
    in the OLDER nested shape --
    ``{"claims": [{"comparison": {...}, "seed_bundle": {...}, "measured_scope": {...}}]}``
    -- consumed by ``two_axis_report.load_claims``. ``result_fidelity.evaluate``
    (and ``metric_binding.bind_claims`` underneath it) instead read a FLAT
    per-claim dict (``kind``/``is_primary``/``ambiguous``/``metric_name`` etc.
    directly on each ``claims[i]`` entry -- see
    ``tests/agents/rlm/test_result_fidelity.py``'s own ``_claim()`` fixture,
    and the design doc's own interface block in §6). Nothing adapts one shape
    to the other: ``report.py::_load_repro_spec_for_authority`` (the ONLY
    production call site) reads the raw on-disk dict and passes it straight
    into ``evaluate()`` unmodified, by explicit design
    ("``result_fidelity.evaluate`` wants the raw ReproSpec dict ... not the
    typed ``MeasuredClaim`` list ``two_axis_report.load_claims`` builds").

    Consequence, verified empirically below: ``claim.get("is_primary")``,
    ``claim.get("ambiguous")``, ``claim.get("kind")`` and
    ``claim.get("metric_name")`` are all ``None`` for this claim (they live
    one level down, under ``claim["comparison"]``), so
    ``metric_binding.bind_claims`` fails to bind on a blank ``metric_name``
    (``reason="missing_metric_name"``) and ``result_fidelity.evaluate`` emits
    a SINGLE ``per_claim`` entry with ``is_primary=False``/``ambiguous=False``
    -- NOT because the taxonomy correctly recognised the one real claim as an
    ambiguous primary, but because the whole claim is invisible to this
    reader. ``verdict_authority.decide``'s rule 1 ("no claim is genuinely
    ``is_primary``") fires either way, so the OBSERVED verdict label matches
    what the corrected brief predicts -- but for a different mechanical
    reason than "the ambiguous primary was correctly excluded". Both the
    "genuinely ambiguous primary" case and the "claim shape invisible to the
    reader" case land on the identical rule-1 branch today, which is why this
    assertion is honest and passing is not a false green -- but it also means
    this artifact's ``ambiguous=True`` flag is NOT actually exercised by this
    call path, and (more importantly) EVERY repro_spec.json written by
    today's extractor will read as "no genuine primaries" regardless of its
    real claim content, until either ``repro_spec_extractor`` is updated to
    emit the flat shape or an adapter is inserted ahead of ``evaluate()``.
    This is flagged in the task report as a real, separate finding -- not
    papered over by loosening this assertion.
    """
    if not _ADAM_RUN_DIR.is_dir():
        pytest.skip(f"frozen Adam acceptance artifact not present: {_ADAM_RUN_DIR}")

    run_dir = _copy_adam_artifacts_readonly(tmp_path)
    repro_spec = json.loads(
        (run_dir / "rlm_state" / "repro_spec.json").read_text(encoding="utf-8")
    )

    result_fidelity = evaluate(repro_spec, run_dir)
    verdict = decide(
        result_fidelity=result_fidelity,
        evidence_gate=True,  # most-generous evidence input -- isolates the claim-side rule
        fidelity_certificate=None,
    )

    assert verdict["verdict"] == "inconclusive", (
        "if this says 'reproduced', the WS1 sever failed -- the grade is "
        f"still minting the verdict. observed verdict={verdict!r} "
        f"per_claim={result_fidelity['per_claim']!r}"
    )
    assert verdict["reason"] == "no_measurable_target"

    # The frozen artifact must never be mutated by this test -- re-read the
    # REAL on-disk file (not the tmp_path copy) and confirm it is unchanged.
    untouched = json.loads(
        (_ADAM_RUN_DIR / "rlm_state" / "repro_spec.json").read_text(encoding="utf-8")
    )
    assert untouched == repro_spec


# --------------------------------------------------------------------------- #
# 2-4. Synthetic taxonomy fixtures -- partial / contradicted / reproduced
#
# Each builds a repro_spec in the flat, kind-discriminated shape `evaluate()`
# actually understands (reusing `_claim`/`_run` from test_result_fidelity.py
# verbatim) and drives the real evaluate() -> decide() pipeline, per the
# taxonomy locked in verdict_authority.py's module docstring (§4.3 rules 2-4).
# --------------------------------------------------------------------------- #


def test_synthetic_unmeasured_primary_is_partial(tmp_path):
    """A REAL (non-ambiguous) primary claim whose metric has no match anywhere
    in metrics.json stays unbound -> unmeasured -> decide() -> partial (rule 3:
    "any primary unmeasured, none fail"). Mirrors
    test_result_fidelity.test_unbound_metric_is_unmeasured, layered with decide().
    """
    run = _run(tmp_path, {"other_metric": 1.0})
    claim = _claim(metric_name="totally_absent_metric")

    result_fidelity = evaluate({"claims": [claim]}, run)
    per_claim = result_fidelity["per_claim"][0]
    assert per_claim["is_primary"] is True
    assert per_claim["ambiguous"] is False
    assert per_claim["status"] == "unmeasured"

    verdict = decide(
        result_fidelity=result_fidelity, evidence_gate=True, fidelity_certificate=None
    )
    assert verdict["verdict"] == "partial"
    assert verdict["reason"] == "primary_claim_unmeasured"


def test_synthetic_measured_violation_is_contradicted(tmp_path):
    """A bound primary claim whose measured value violates the claim (outside
    the equivalence margin) -> fail -> decide() -> contradicted (rule 2: "any
    primary fail" dominates). Mirrors
    test_result_fidelity.test_numeric_outside_margin_fails_only_with_verified_bind.
    """
    run = _run(tmp_path, {"accuracy": 0.80})
    claim = _claim()  # claims accuracy ~= 0.99 +/- 0.01

    result_fidelity = evaluate({"claims": [claim]}, run)
    assert result_fidelity["per_claim"][0]["status"] == "fail"
    assert result_fidelity["any_contradicted"] is True

    verdict = decide(
        result_fidelity=result_fidelity, evidence_gate=True, fidelity_certificate=None
    )
    assert verdict["verdict"] == "contradicted"
    assert verdict["reason"] == "primary_claim_failed"


def test_synthetic_measured_pass_with_evidence_gate_is_reproduced(tmp_path):
    """A bound primary claim that passes, WITH a satisfied evidence_gate ->
    decide() -> reproduced (rule 4). Mirrors
    test_result_fidelity.test_numeric_within_margin_passes.

    Also proves the rule-4 asymmetry end-to-end: the identical passing claim
    WITHOUT a satisfied evidence gate must NOT reach ``reproduced`` -- a
    measured claim alone is never sufficient (mirrors
    test_verdict_authority.test_all_pass_but_no_evidence_is_not_reproduced,
    now exercised through the real evaluate() output rather than a
    hand-built ResultFidelity dict).
    """
    run = _run(tmp_path, {"accuracy": 0.991})
    claim = _claim()

    result_fidelity = evaluate({"claims": [claim]}, run)
    assert result_fidelity["per_claim"][0]["status"] == "pass"

    verdict = decide(
        result_fidelity=result_fidelity, evidence_gate=True, fidelity_certificate=None
    )
    assert verdict["verdict"] == "reproduced"
    assert verdict["reason"] == "all_primary_claims_pass"

    verdict_no_gate = decide(
        result_fidelity=result_fidelity, evidence_gate=False, fidelity_certificate=None
    )
    assert verdict_no_gate["verdict"] != "reproduced"
