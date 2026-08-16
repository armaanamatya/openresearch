"""Tests for verdict_authority.decide + freeze_contract (Track A, §4.3/§4.6).

Pure deterministic tests — no LLM, no network, no filesystem beyond the
tmp_path fixtures ``freeze_contract`` exercises. See
docs/history/specs/2026-07-09-eval-integrity-track-a-design.md §4.3 for
the locked taxonomy precedence this locks down: a genuinely-primary claim
that FAILS always wins (``contradicted``), an unmeasured-but-genuine primary
is ``partial``, and only an all-pass primary set WITH a satisfied evidence
gate is ``reproduced`` — the LLM grade never enters this decision (there is
no parameter for it to enter through).
"""

from __future__ import annotations

import json

from backend.agents.rlm.verdict_authority import decide, freeze_contract


# ---------------------------------------------------------------------------
# Verbatim fixtures + tests from the task brief (Step 1)
# ---------------------------------------------------------------------------


def _rf(per_claim, **kw):
    return {"per_claim": per_claim, "result_fidelity_score": kw.get("score", 0.0),
            "primary_all_measured": kw.get("all_measured", False),
            "any_contradicted": any(c["status"] == "fail" for c in per_claim)}


def _c(status, primary=True, ambiguous=False):
    return {"claim_id": "x", "status": status, "is_primary": primary, "ambiguous": ambiguous}


def test_no_measurable_primary_is_inconclusive():
    v = decide(result_fidelity=_rf([_c("unmeasured", ambiguous=True)]),
               evidence_gate=True, fidelity_certificate=None)
    assert v["verdict"] == "inconclusive" and v["reason"] == "no_measurable_target"


def test_any_primary_fail_is_contradicted_even_with_unmeasured_sibling():
    v = decide(result_fidelity=_rf([_c("fail"), _c("unmeasured")]),
               evidence_gate=True, fidelity_certificate=None)
    assert v["verdict"] == "contradicted"


def test_mixed_pass_unmeasured_none_fail_is_partial():
    v = decide(result_fidelity=_rf([_c("pass"), _c("unmeasured")]),
               evidence_gate=True, fidelity_certificate=None)
    assert v["verdict"] == "partial"


def test_all_primary_pass_with_evidence_is_reproduced():
    v = decide(result_fidelity=_rf([_c("pass")], all_measured=True, score=1.0),
               evidence_gate=True, fidelity_certificate=object())
    assert v["verdict"] == "reproduced"


def test_all_pass_but_no_evidence_is_not_reproduced():
    v = decide(result_fidelity=_rf([_c("pass")], all_measured=True, score=1.0),
               evidence_gate=False, fidelity_certificate=None)
    assert v["verdict"] != "reproduced"


def test_claim_gate_cap_only_lowers():
    v = decide(result_fidelity=_rf([_c("pass")], all_measured=True, score=1.0),
               evidence_gate=True, fidelity_certificate=object(), claim_gate_cap="partial")
    assert v["verdict"] == "partial"


# ---------------------------------------------------------------------------
# Additional taxonomy-precedence coverage
# ---------------------------------------------------------------------------


def test_multiple_primaries_fail_beats_pass():
    """`fail` governs a multi-primary rollup even against a passing sibling
    (not just against `unmeasured`, which the verbatim brief test covers)."""
    v = decide(result_fidelity=_rf([_c("pass"), _c("fail")]),
               evidence_gate=True, fidelity_certificate=None)
    assert v["verdict"] == "contradicted"


def test_secondary_claim_failure_never_escalates():
    """A failing SECONDARY (is_primary=False) claim must never contradict the
    paper-level verdict — only PRIMARY claims are ever consulted."""
    v = decide(result_fidelity=_rf([_c("pass"), _c("fail", primary=False)]),
               evidence_gate=True, fidelity_certificate=object())
    assert v["verdict"] == "reproduced"


def test_auto_promoted_marker_ignored_even_when_unambiguous():
    """The extractor's mechanical auto-promotion (repro_spec_extractor.py:
    "ensure exactly one primary claim") sets is_primary=True with no basis in
    the paper's own emphasis. No extractor field distinguishes this today, so
    `ambiguous` is the load-bearing signal per the locked design (covered
    above) -- but IF an `auto_promoted` marker is ever threaded through, it
    must be honoured too, defensively, without another change to this module.
    """
    claim = _c("pass")
    claim["auto_promoted"] = True
    v = decide(result_fidelity=_rf([claim]), evidence_gate=True, fidelity_certificate=object())
    assert v["verdict"] == "inconclusive" and v["reason"] == "no_measurable_target"


def test_primary_source_auto_promoted_ignored():
    claim = _c("pass")
    claim["primary_source"] = "auto_promoted"
    v = decide(result_fidelity=_rf([claim]), evidence_gate=True, fidelity_certificate=object())
    assert v["verdict"] == "inconclusive"


# ---------------------------------------------------------------------------
# Missing / empty result_fidelity (RDR / legacy path) -- never a pass-through
# ---------------------------------------------------------------------------


def test_missing_result_fidelity_is_inconclusive():
    v = decide(result_fidelity=None, evidence_gate=True, fidelity_certificate=object())
    assert v["verdict"] == "inconclusive" and v["reason"] == "no_measurable_target"


def test_empty_dict_result_fidelity_is_inconclusive():
    v = decide(result_fidelity={}, evidence_gate=True, fidelity_certificate=object())
    assert v["verdict"] == "inconclusive"


def test_result_fidelity_missing_per_claim_key_is_inconclusive():
    v = decide(result_fidelity={"result_fidelity_score": 0.9}, evidence_gate=True,
               fidelity_certificate=None)
    assert v["verdict"] == "inconclusive"


# ---------------------------------------------------------------------------
# evidence_gate: bool AND struct (never blind non-emptiness truthiness)
# ---------------------------------------------------------------------------


class _FakeGateStruct:
    def __init__(self, satisfied: bool) -> None:
        self.satisfied = satisfied
        self.some_other_field = "populated"  # proves attribute presence alone isn't enough


def test_evidence_gate_struct_dict_satisfied_true_is_reproduced():
    v = decide(result_fidelity=_rf([_c("pass")], all_measured=True, score=1.0),
               evidence_gate={"satisfied": True, "note": "ledger row ok"},
               fidelity_certificate=object())
    assert v["verdict"] == "reproduced"


def test_evidence_gate_struct_dict_satisfied_false_is_not_reproduced_despite_nonempty():
    """A non-empty dict with satisfied=False must NOT read as truthy just
    because plain `bool(dict)` would be True for any non-empty dict."""
    v = decide(result_fidelity=_rf([_c("pass")], all_measured=True, score=1.0),
               evidence_gate={"satisfied": False, "note": "no ledger row"},
               fidelity_certificate=None)
    assert v["verdict"] != "reproduced"


def test_evidence_gate_object_with_satisfied_attribute():
    v_true = decide(result_fidelity=_rf([_c("pass")], all_measured=True, score=1.0),
                     evidence_gate=_FakeGateStruct(True), fidelity_certificate=object())
    v_false = decide(result_fidelity=_rf([_c("pass")], all_measured=True, score=1.0),
                      evidence_gate=_FakeGateStruct(False), fidelity_certificate=None)
    assert v_true["verdict"] == "reproduced"
    assert v_false["verdict"] != "reproduced"


# ---------------------------------------------------------------------------
# claim_gate_cap: downward-only clamp
# ---------------------------------------------------------------------------


def test_claim_gate_cap_does_not_raise_a_lower_verdict():
    """A cap weaker than the computed verdict must never PROMOTE it -- "caps
    the result downward only" is bidirectional: it can't lift a bad verdict
    either."""
    v = decide(result_fidelity=_rf([_c("fail"), _c("unmeasured")]),
               evidence_gate=True, fidelity_certificate=None, claim_gate_cap="reproduced")
    assert v["verdict"] == "contradicted"


def test_unknown_claim_gate_cap_value_is_noop():
    v = decide(result_fidelity=_rf([_c("pass")], all_measured=True, score=1.0),
               evidence_gate=True, fidelity_certificate=object(),
               claim_gate_cap="not_a_real_verdict_word")
    assert v["verdict"] == "reproduced"


def test_claim_gate_cap_none_is_noop():
    v = decide(result_fidelity=_rf([_c("pass")], all_measured=True, score=1.0),
               evidence_gate=True, fidelity_certificate=object(), claim_gate_cap=None)
    assert v["verdict"] == "reproduced"


# ---------------------------------------------------------------------------
# ruler_quality: Spec-B seam, inert in Track A
# ---------------------------------------------------------------------------


def test_ruler_quality_default_and_explicit_trusted_agree():
    kwargs = dict(result_fidelity=_rf([_c("pass")], all_measured=True, score=1.0),
                  evidence_gate=True, fidelity_certificate=object())
    v_unset = decide(**kwargs)
    v_none = decide(**kwargs, ruler_quality=None)
    v_trusted = decide(**kwargs, ruler_quality="trusted")
    assert v_unset == v_none == v_trusted == {"verdict": "reproduced", "reason": "all_primary_claims_pass"}


# ---------------------------------------------------------------------------
# freeze_contract (Spec-C seam)
# ---------------------------------------------------------------------------


def test_freeze_contract_marks_existing_spec_frozen(tmp_path):
    rlm_state = tmp_path / "rlm_state"
    rlm_state.mkdir()
    spec_path = rlm_state / "repro_spec.json"
    spec_path.write_text(json.dumps({"claims": []}))

    out = freeze_contract(tmp_path)

    assert out == spec_path
    data = json.loads(spec_path.read_text())
    assert data["contract_status"] == "frozen"
    assert isinstance(data["frozen_at"], str) and data["frozen_at"]
    assert data["claims"] == []  # non-contract keys preserved


def test_freeze_contract_is_idempotent(tmp_path):
    rlm_state = tmp_path / "rlm_state"
    rlm_state.mkdir()
    spec_path = rlm_state / "repro_spec.json"
    spec_path.write_text(json.dumps({"claims": []}))

    freeze_contract(tmp_path)
    first_frozen_at = json.loads(spec_path.read_text())["frozen_at"]

    freeze_contract(tmp_path)
    second_frozen_at = json.loads(spec_path.read_text())["frozen_at"]

    assert first_frozen_at == second_frozen_at


def test_freeze_contract_missing_spec_is_noop(tmp_path):
    out = freeze_contract(tmp_path)
    assert out == tmp_path / "rlm_state" / "repro_spec.json"
    assert not out.exists()


def test_freeze_contract_malformed_json_left_untouched(tmp_path):
    rlm_state = tmp_path / "rlm_state"
    rlm_state.mkdir()
    spec_path = rlm_state / "repro_spec.json"
    spec_path.write_text("not valid json{")

    out = freeze_contract(tmp_path)

    assert out == spec_path
    assert spec_path.read_text() == "not valid json{"


def test_freeze_contract_non_dict_json_left_untouched(tmp_path):
    rlm_state = tmp_path / "rlm_state"
    rlm_state.mkdir()
    spec_path = rlm_state / "repro_spec.json"
    spec_path.write_text(json.dumps([1, 2, 3]))

    out = freeze_contract(tmp_path)

    assert out == spec_path
    assert json.loads(spec_path.read_text()) == [1, 2, 3]
