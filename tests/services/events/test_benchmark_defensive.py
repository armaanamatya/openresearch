"""Pins the defensive numeric coercion on BenchmarkSummary.

2026-05-26 regression: live_runs.py finalize_benchmark picked
next(iter(baseline_metrics)) as reproducedValue. When baseline_metrics
started with a dict-valued key (Adam's `scope`), Pydantic 500'd on every
GET /runs/<id> and the lab UI's saved-run loader silently fell back
to upload view.

These tests pin: invalid types degrade to None; valid scalars pass through.
"""
from __future__ import annotations
import pytest
from backend.services.events.live_runs import BenchmarkSummary, _coerce_to_float_or_none


def _stub_kwargs(**overrides):
    return {
        "benchmarkName": "x", "paperbenchTaskId": "x", "overallScore": 0.0,
        "targetMetric": "x", "targetValue": 0.0, "deltaValue": 0.0,
        "verdict": "x", "reportPath": "x", "comparisonPath": "x", "logPath": "x",
        **overrides,
    }


class TestCoercion:
    def test_int_passes(self):
        assert _coerce_to_float_or_none(42) == 42.0

    def test_float_passes(self):
        assert _coerce_to_float_or_none(0.413) == 0.413

    def test_none_passes(self):
        assert _coerce_to_float_or_none(None) is None

    def test_bool_becomes_none(self):
        assert _coerce_to_float_or_none(True) is None
        assert _coerce_to_float_or_none(False) is None

    def test_numeric_string_coerces(self):
        assert _coerce_to_float_or_none("0.413") == 0.413
        assert _coerce_to_float_or_none("42") == 42.0

    def test_non_numeric_string_becomes_none(self):
        assert _coerce_to_float_or_none("hello") is None

    def test_dict_becomes_none(self):
        # the Adam regression — scope dict was being assigned to reproducedValue
        assert _coerce_to_float_or_none({"models_run": [], "gaps": []}) is None

    def test_list_becomes_none(self):
        assert _coerce_to_float_or_none([1, 2, 3]) is None


class TestBenchmarkSummary:
    def test_dict_in_reproduced_value_does_not_raise(self):
        b = BenchmarkSummary(**_stub_kwargs(reproducedValue={"scope": "garbage"}))
        assert b.reproducedValue is None

    def test_dict_in_delta_does_not_raise(self):
        b = BenchmarkSummary(**_stub_kwargs(deltaValue={"foo": "bar"}))
        assert b.deltaValue is None

    def test_dict_in_our_rubric_score_does_not_raise(self):
        b = BenchmarkSummary(**_stub_kwargs(ourRubricScore={"x": 1}))
        assert b.ourRubricScore is None

    def test_valid_floats_round_trip(self):
        b = BenchmarkSummary(**_stub_kwargs(reproducedValue=0.413, deltaValue=-0.1))
        assert b.reproducedValue == pytest.approx(0.413)
        assert b.deltaValue == pytest.approx(-0.1)

    def test_string_float_coerces(self):
        b = BenchmarkSummary(**_stub_kwargs(reproducedValue="0.535"))
        assert b.reproducedValue == pytest.approx(0.535)
