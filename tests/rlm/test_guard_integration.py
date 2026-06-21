"""Regression tests: G1 arg-contract guard integration through wrap_primitive.

Drives wrap_primitive directly (no RLM completion loop). Four focused cases:
1. Guard fires + short-circuits (underlying primitive NOT called).
2. Flag off → guard absent, primitive runs normally.
3. Flag on + clean args → no false-positive, primitive runs.
4. Primitive not in the contract table → no guard even with sentinel kwargs.
"""

from __future__ import annotations



# ---------------------------------------------------------------------------
# Test 1: G1 fires in-loop + short-circuits
# ---------------------------------------------------------------------------

def test_g1_fires_and_short_circuits(tmp_path, make_context, monkeypatch):
    """With OPENRESEARCH_ARG_CONTRACTS=1 and a sentinel in method_spec, the guard
    returns success=False + failure_class='arg_contract' WITHOUT calling the
    underlying primitive."""
    monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")

    ctx = make_context(tmp_path)

    called: list[int] = []

    def fake_plan_reproduction(
        method_spec=None,
        paper_claim_map=None,
        compute_scope=None,
        ctx=None,
    ):
        called.append(1)
        return {"ok": True, "plan": "real"}

    from backend.agents.rlm.binding import wrap_primitive

    wrapped = wrap_primitive("plan_reproduction", fake_plan_reproduction, ctx)
    result = wrapped(method_spec={"method_name": "unknown", "components": ["real", "tbd"]})

    assert result["success"] is False, f"Expected success=False, got {result}"
    assert result["failure_class"] == "arg_contract", f"Got {result.get('failure_class')!r}"
    assert result.get("contract_violations"), "Expected non-empty contract_violations"
    assert called == [], "Underlying primitive should NOT have been invoked (guard short-circuited)"


# ---------------------------------------------------------------------------
# Test 2: Flag OFF → no guard, primitive runs byte-for-byte
# ---------------------------------------------------------------------------

def test_flag_off_primitive_runs(tmp_path, make_context, monkeypatch):
    """Without OPENRESEARCH_ARG_CONTRACTS set, sentinel args do NOT trigger the
    guard; the underlying primitive runs and its result is returned unchanged."""
    monkeypatch.delenv("OPENRESEARCH_ARG_CONTRACTS", raising=False)

    ctx = make_context(tmp_path)

    called: list[int] = []

    def fake_plan_reproduction(
        method_spec=None,
        paper_claim_map=None,
        compute_scope=None,
        ctx=None,
    ):
        called.append(1)
        return {"ok": True, "plan": "real"}

    from backend.agents.rlm.binding import wrap_primitive

    wrapped = wrap_primitive("plan_reproduction", fake_plan_reproduction, ctx)
    result = wrapped(method_spec={"method_name": "unknown"})

    assert called == [1], "Primitive should have been called when flag is off"
    assert result == {"ok": True, "plan": "real"}, f"Result should be unchanged, got {result}"


# ---------------------------------------------------------------------------
# Test 3: Flag ON + clean args → primitive runs, no false-positive
# ---------------------------------------------------------------------------

def test_flag_on_clean_args_primitive_runs(tmp_path, make_context, monkeypatch):
    """With OPENRESEARCH_ARG_CONTRACTS=1 and legitimate (non-sentinel) args, the
    guard fires no violation and the underlying primitive runs normally."""
    monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")

    ctx = make_context(tmp_path)

    called: list[int] = []

    def fake_plan_reproduction(
        method_spec=None,
        paper_claim_map=None,
        compute_scope=None,
        ctx=None,
    ):
        called.append(1)
        return {"ok": True, "plan": "real"}

    from backend.agents.rlm.binding import wrap_primitive

    wrapped = wrap_primitive("plan_reproduction", fake_plan_reproduction, ctx)
    result = wrapped(method_spec={"method_name": "SDAR", "lambda_": 0.1})

    assert called == [1], "Primitive should have been called with clean args"
    assert result == {"ok": True, "plan": "real"}, f"Result should be unchanged, got {result}"


# ---------------------------------------------------------------------------
# Test 4: Flag ON + primitive NOT in contract table → no guard
# ---------------------------------------------------------------------------

def test_flag_on_undeclared_primitive_no_guard(tmp_path, make_context, monkeypatch):
    """With OPENRESEARCH_ARG_CONTRACTS=1, a primitive not in PRIMITIVE_ARG_CONTRACTS
    (e.g. understand_section) passes through even when kwargs contain sentinels."""
    monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")

    ctx = make_context(tmp_path)

    called: list[int] = []

    def fake_understand_section(section=None, ctx=None):
        called.append(1)
        return {"ok": True, "content": "some content"}

    from backend.agents.rlm.binding import wrap_primitive

    wrapped = wrap_primitive("understand_section", fake_understand_section, ctx)
    result = wrapped(section="unknown")  # sentinel value — but undeclared primitive

    assert called == [1], "Primitive should have been called (not in contract table)"
    assert result == {"ok": True, "content": "some content"}, f"Unexpected result: {result}"
