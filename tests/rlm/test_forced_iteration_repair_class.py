"""P3.1 — the distinct fix-first repair-refusal class in ForcedIterationPolicy.

All new behavior is gated on ``evidence_fingerprint`` being wired (the fix-first
loop is "engaged"); with it unwired the policy is byte-identical to the hardened
2026-06-17 behavior. These tests exercise BOTH states.

Hermetic: no network, no filesystem; the policy is a plain dataclass.
"""

from __future__ import annotations

from backend.agents.rlm.forced_iteration import ForcedIterationPolicy


# ---------------------------------------------------------------------------
# B1 — a repair refusal must NOT accumulate toward the root_degenerate_loop trip
# ---------------------------------------------------------------------------


def test_repair_refusal_excluded_from_degenerate_counter_when_engaged():
    p = ForcedIterationPolicy(min_iterations=0, evidence_fingerprint=lambda: "fp")
    for _ in range(5):
        p.register_refusal("repair_floor")
    # Excluded — a stuck-but-trying repair never trips the degenerate detector.
    assert p._noprogress_refusals == 0


def test_repair_refusal_counts_when_not_engaged():
    # evidence_fingerprint unset → legacy behavior: repair_floor accumulates.
    p = ForcedIterationPolicy(min_iterations=0)
    p.register_refusal("repair_floor")
    p.register_refusal("repair_floor")
    assert p._noprogress_refusals == 2


def test_non_repair_signature_still_counts_when_engaged():
    # Only the repair_floor signature is excluded; a real no-progress signature
    # (e.g. no_experiment) still feeds the degenerate detector.
    p = ForcedIterationPolicy(min_iterations=0, evidence_fingerprint=lambda: "fp")
    p.register_refusal("no_experiment")
    p.register_refusal("no_experiment")
    assert p._noprogress_refusals == 2


# ---------------------------------------------------------------------------
# Evidence-fingerprint progress → bounded repair loop
# ---------------------------------------------------------------------------


def test_fingerprint_progress_keeps_noprogress_zero():
    fps = iter(["a", "b", "c"])
    p = ForcedIterationPolicy(min_iterations=0, evidence_fingerprint=lambda: next(fps))
    p.record_repair_attempt("fabrication_suspected")  # fp=a (first)
    p.record_repair_attempt("fabrication_suspected")  # fp=b (changed → progress)
    assert p._repair_noprogress_count == 0
    assert p._terminal_failure_class != "repair_exhausted"


def test_no_progress_triggers_repair_exhausted():
    p = ForcedIterationPolicy(min_iterations=0, evidence_fingerprint=lambda: "stuck")
    p.record_repair_attempt("fabrication_suspected")  # first → noprogress 0
    p.record_repair_attempt("fabrication_suspected")  # same class + same fp → noprogress 1
    p.record_repair_attempt("fabrication_suspected")  # noprogress 2 >= limit(2) → exhausted
    assert p._repair_noprogress_count >= 2
    assert p._terminal_failure_class == "repair_exhausted"


def test_repair_max_ceiling_triggers_exhausted(monkeypatch):
    # The ceiling fires even when every attempt makes progress (changing fps).
    monkeypatch.setenv("OPENRESEARCH_REPAIR_MAX_ITERATIONS", "3")
    fps = iter(["a", "b", "c", "d"])
    p = ForcedIterationPolicy(min_iterations=0, evidence_fingerprint=lambda: next(fps))
    p.record_repair_attempt("fc")  # count 1
    p.record_repair_attempt("fc")  # count 2 (progress)
    p.record_repair_attempt("fc")  # count 3 >= max 3 → exhausted
    assert p._terminal_failure_class == "repair_exhausted"


def test_repair_loop_inert_when_not_engaged():
    # Without evidence_fingerprint the new fields never change — byte-identical.
    p = ForcedIterationPolicy(min_iterations=0)
    for _ in range(10):
        p.record_repair_attempt("fc")
    assert p._repair_noprogress_count == 0
    assert p._terminal_failure_class is None  # never auto-set to repair_exhausted


# ---------------------------------------------------------------------------
# clear_repair_trigger — a success clears the loop (engaged only)
# ---------------------------------------------------------------------------


def test_clear_repair_trigger_resets_when_engaged():
    p = ForcedIterationPolicy(min_iterations=0, evidence_fingerprint=lambda: "x")
    p.record_repair_attempt("fc")
    assert p._repair_iter_count == 1
    p.clear_repair_trigger()
    assert p._repair_iter_count == 0
    assert p._last_repair_failure_class is None
    assert p._repair_noprogress_count == 0


def test_clear_repair_trigger_noop_when_not_engaged():
    p = ForcedIterationPolicy(min_iterations=0)
    p.record_repair_attempt("fc")
    p.clear_repair_trigger()
    # Legacy: the count-based floor bounds the loop, so clear is a no-op.
    assert p._repair_iter_count == 1
    assert p._last_repair_failure_class == "fc"


# ---------------------------------------------------------------------------
# should_refuse integration
# ---------------------------------------------------------------------------


def test_should_refuse_accepts_on_repair_exhausted():
    p = ForcedIterationPolicy(min_iterations=0, evidence_fingerprint=lambda: "stuck")
    for _ in range(3):
        p.record_repair_attempt("fc")
    assert p._terminal_failure_class == "repair_exhausted"
    refuse, msg = p.should_refuse()
    assert refuse is False  # the terminal check accepts the next FINAL_VAR
    assert msg is None


def test_repair_window_extends_past_min_repair_when_engaged(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_MIN_REPAIR_ITERATIONS", "1")
    monkeypatch.setenv("OPENRESEARCH_REPAIR_MAX_ITERATIONS", "4")
    fps = iter(["a", "b", "c", "d", "e"])
    p = ForcedIterationPolicy(
        min_iterations=0,
        evidence_fingerprint=lambda: next(fps),
        current_iteration=lambda: 1,
        remaining_s=lambda: 9999.0,
    )
    p._total_run_experiments = 1  # bypass the no-experiment refusal
    p.record_repair_attempt("fc")  # count 1
    p.record_repair_attempt("fc")  # count 2 — past MIN_REPAIR(1), under MAX(4), progressing
    refuse, msg = p.should_refuse()
    assert refuse is True  # the extended window forces another repair
    assert "repair" in msg.lower()


# ---------------------------------------------------------------------------
# Validator gate — a machine-verified veto feeds the same fix-first loop
# ---------------------------------------------------------------------------


def test_validator_gate_vetoed_refuses_final_var():
    p = ForcedIterationPolicy(
        min_iterations=0,
        evidence_fingerprint=lambda: "ev",
        validator_gate=lambda: (True, "fix the reward wiring to real env outcomes"),
        current_iteration=lambda: 1,
        remaining_s=lambda: 9999.0,
    )
    p._total_run_experiments = 1
    refuse, msg = p.should_refuse()
    assert refuse is True
    assert "reward" in msg.lower()


def test_validator_gate_inert_when_none():
    # No validator_gate → byte-identical (no veto path); min_iterations 0 accepts.
    p = ForcedIterationPolicy(
        min_iterations=0, current_iteration=lambda: 1, remaining_s=lambda: 9999.0
    )
    p._total_run_experiments = 1
    refuse, _ = p.should_refuse()
    assert refuse is False


def test_validator_gate_exhausts_to_repair_exhausted():
    # A persistently-vetoing validator on UNCHANGED evidence stops honestly.
    p = ForcedIterationPolicy(
        min_iterations=0,
        evidence_fingerprint=lambda: "stuck",
        validator_gate=lambda: (True, "still fabricated"),
        current_iteration=lambda: 1,
        remaining_s=lambda: 9999.0,
    )
    p._total_run_experiments = 1
    results = [p.should_refuse()[0] for _ in range(4)]
    assert results[-1] is False  # eventually accepts (repair_exhausted)
    assert p._terminal_failure_class == "repair_exhausted"


# ---------------------------------------------------------------------------
# Claim gate (§4.4 B) — an ungrounded-report veto feeds the SAME fix-first loop
# ---------------------------------------------------------------------------


def test_claim_gate_vetoed_refuses_final_var():
    # An ungrounded report claim → refuse FINAL_VAR with the gate's directive.
    p = ForcedIterationPolicy(
        min_iterations=0,
        evidence_fingerprint=lambda: "ev",
        claim_gate=lambda: (True, "report claims accuracy 0.84 not in metrics.json"),
        current_iteration=lambda: 1,
        remaining_s=lambda: 9999.0,
    )
    p._total_run_experiments = 1
    refuse, msg = p.should_refuse()
    assert refuse is True
    assert "accuracy" in msg.lower()


def test_claim_gate_inert_when_none():
    # No claim_gate → byte-identical (no claim veto path); min_iterations 0 accepts.
    p = ForcedIterationPolicy(
        min_iterations=0, current_iteration=lambda: 1, remaining_s=lambda: 9999.0
    )
    p._total_run_experiments = 1
    refuse, _ = p.should_refuse()
    assert refuse is False


def test_claim_gate_clean_does_not_refuse():
    # claim_gate returning None (claims grounded / unverifiable) → no refusal.
    p = ForcedIterationPolicy(
        min_iterations=0,
        evidence_fingerprint=lambda: "ev",
        claim_gate=lambda: None,
        current_iteration=lambda: 1,
        remaining_s=lambda: 9999.0,
    )
    p._total_run_experiments = 1
    refuse, _ = p.should_refuse()
    assert refuse is False


def test_claim_gate_exhausts_to_repair_exhausted():
    # A persistently-ungrounded report on UNCHANGED evidence stops honestly as
    # repair_exhausted (bounded by REPAIR_MAX), never an infinite refusal loop.
    p = ForcedIterationPolicy(
        min_iterations=0,
        evidence_fingerprint=lambda: "stuck",
        claim_gate=lambda: (True, "still ungrounded"),
        current_iteration=lambda: 1,
        remaining_s=lambda: 9999.0,
    )
    p._total_run_experiments = 1
    results = [p.should_refuse()[0] for _ in range(4)]
    assert results[-1] is False  # eventually accepts (repair_exhausted)
    assert p._terminal_failure_class == "repair_exhausted"


def test_claim_gate_exception_treated_as_clean():
    # A claim_gate that raises must never crash the policy; treated as clean.
    def _boom():
        raise RuntimeError("claim gate blew up")

    p = ForcedIterationPolicy(
        min_iterations=0,
        evidence_fingerprint=lambda: "ev",
        claim_gate=_boom,
        current_iteration=lambda: 1,
        remaining_s=lambda: 9999.0,
    )
    p._total_run_experiments = 1
    refuse, _ = p.should_refuse()
    assert refuse is False


def test_claim_gate_clean_after_veto_clears_stale_trigger():
    # codex Area-5: after a report_claim veto, if the report is corrected so the gate
    # goes clean, the stale repair trigger must clear so FINAL_VAR is no longer refused
    # — a report TEXT fix is valid progress without a new run_experiment. Without the
    # fix, the repair floor kept refusing the corrected report until repair_exhausted.
    state = {"veto": True}

    def _gate():
        return (True, "report claims accuracy 0.84 not in metrics.json") if state["veto"] else None

    p = ForcedIterationPolicy(
        min_iterations=0,
        evidence_fingerprint=lambda: "ev",
        claim_gate=_gate,
        current_iteration=lambda: 1,
        remaining_s=lambda: 9999.0,
    )
    p._total_run_experiments = 1
    # First attempt: ungrounded → refuse + sets the report_claim repair trigger.
    refuse1, _ = p.should_refuse()
    assert refuse1 is True
    assert p._last_repair_failure_class == "report_claim"
    # Root corrects the report text → gate now clean.
    state["veto"] = False
    refuse2, _ = p.should_refuse()
    assert refuse2 is False, "a corrected report must not stay stuck in the repair floor"
    assert p._last_repair_failure_class is None
