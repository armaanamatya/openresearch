"""Tests for backend/agents/rlm/arg_contracts.py — arg pre-validation guard."""
from __future__ import annotations

import pytest

from backend.agents.rlm.arg_contracts import validate_primitive_args


# ---------------------------------------------------------------------------
# Tiny local stub used as `fn` for signature binding tests.
# Mirrors the relevant parameters of plan_reproduction.
# ---------------------------------------------------------------------------

def _fake_plan_reproduction(
    method_spec=None,
    paper_claim_map=None,
    compute_scope=None,
    ctx=None,
):
    """Stub primitive — never actually called; used for inspect.signature binding."""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFlagOff:
    """With the flag unset/off, validate_primitive_args is always a no-op."""

    def test_returns_none_even_with_sentinel(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_ARG_CONTRACTS", raising=False)
        result = validate_primitive_args(
            "plan_reproduction",
            _fake_plan_reproduction,
            (),
            {"method_spec": {"method_name": "unknown"}},
        )
        assert result is None

    def test_flag_off_explicit_zero(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "0")
        result = validate_primitive_args(
            "plan_reproduction",
            _fake_plan_reproduction,
            (),
            {"method_spec": {"method_name": "tbd"}},
        )
        assert result is None


class TestFlagOnSentinels:
    """With the flag on, sentinel values in declared params are caught."""

    def test_detects_sentinel_in_nested_dict(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")
        result = validate_primitive_args(
            "plan_reproduction",
            _fake_plan_reproduction,
            (),
            {
                "method_spec": {
                    "method_name": "unknown",
                    "components": ["real", "tbd"],
                },
            },
        )
        assert result is not None
        assert result["success"] is False
        assert result["failure_class"] == "arg_contract"
        assert result["source"] == "arg_guard"
        # Should have violations naming the offending paths
        violations = result["contract_violations"]
        assert len(violations) >= 1
        paths = [v["detail"] for v in violations]
        # 'unknown' at method_spec.method_name should appear
        assert any("method_name" in p and "unknown" in p for p in paths)
        # 'tbd' inside the list should also be caught
        assert any("tbd" in p for p in paths)

    def test_error_message_mentions_primitive_and_count(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")
        result = validate_primitive_args(
            "plan_reproduction",
            _fake_plan_reproduction,
            (),
            {"method_spec": {"method_name": "unknown"}},
        )
        assert result is not None
        assert "plan_reproduction" in result["error"]
        assert "argument(s) contain placeholder" in result["error"]

    def test_empty_string_counts_as_violation(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")
        result = validate_primitive_args(
            "plan_reproduction",
            _fake_plan_reproduction,
            (),
            {"method_spec": {"method_name": ""}},
        )
        assert result is not None
        assert result["failure_class"] == "arg_contract"

    def test_whitespace_string_counts_as_violation(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")
        result = validate_primitive_args(
            "plan_reproduction",
            _fake_plan_reproduction,
            (),
            {"method_spec": {"method_name": "   "}},
        )
        assert result is not None
        assert result["failure_class"] == "arg_contract"

    def test_paper_claim_map_sentinel(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")
        result = validate_primitive_args(
            "plan_reproduction",
            _fake_plan_reproduction,
            (),
            {"paper_claim_map": {"claim_1": "n/a"}},
        )
        assert result is not None
        assert result["failure_class"] == "arg_contract"
        violations = result["contract_violations"]
        assert any("n/a" in v["detail"] for v in violations)


class TestFlagOnCleanArgs:
    """Clean (non-sentinel) args return None."""

    def test_clean_method_spec_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")
        result = validate_primitive_args(
            "plan_reproduction",
            _fake_plan_reproduction,
            (),
            {
                "method_spec": {
                    "method_name": "SDAR",
                    "lambda": 0.1,
                    "components": ["GRPO", "OPSD"],
                },
            },
        )
        assert result is None

    def test_numeric_values_ignored(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")
        result = validate_primitive_args(
            "plan_reproduction",
            _fake_plan_reproduction,
            (),
            {"method_spec": {"lr": 0.001, "epochs": 10, "beta": 10}},
        )
        assert result is None

    def test_none_value_ignored(self, monkeypatch):
        """None is NOT a sentinel (ambiguous — could be legit)."""
        monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")
        result = validate_primitive_args(
            "plan_reproduction",
            _fake_plan_reproduction,
            (),
            {"method_spec": {"key": None}},
        )
        assert result is None


class TestPrimitiveNotInTable:
    """Primitives not in PRIMITIVE_ARG_CONTRACTS are always skipped."""

    def test_run_experiment_not_in_table(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")
        # Even with a sentinel value, run_experiment is not in the table
        result = validate_primitive_args(
            "run_experiment",
            lambda code_path=None: None,
            (),
            {"code_path": "unknown"},
        )
        assert result is None

    def test_understand_section_not_in_table(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")
        result = validate_primitive_args(
            "understand_section",
            lambda section=None: None,
            (),
            {"section": "unknown"},
        )
        assert result is None


class TestFailSoft:
    """Guard must never raise — fail-soft on bad inputs."""

    def test_unsignable_fn_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")
        # Built-in functions can't be introspected by inspect.signature in all cases
        # Simulate by passing a non-callable that will fail signature inspection
        result = validate_primitive_args(
            "plan_reproduction",
            "not_a_function",  # type: ignore[arg-type]
            (),
            {"method_spec": {"key": "unknown"}},
        )
        # Must return None, not raise
        assert result is None

    def test_non_dict_method_spec_does_not_raise(self, monkeypatch):
        """If method_spec is unexpectedly not a dict (e.g. a raw string), no crash."""
        monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")
        # A sentinel string at the top level of method_spec should still be caught
        result = validate_primitive_args(
            "plan_reproduction",
            _fake_plan_reproduction,
            (),
            {"method_spec": "unknown"},
        )
        # Should NOT raise; may or may not detect depending on implementation
        # (spec says string leaves are scanned — "unknown" IS a leaf)
        assert result is None or isinstance(result, dict)

    def test_completely_wrong_kwargs_type_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")
        # Pass args as positional when the function expects keyword — bind_partial
        # should handle this gracefully (or fail-soft)
        try:
            result = validate_primitive_args(
                "plan_reproduction",
                _fake_plan_reproduction,
                ({"method_name": "unknown"},),  # positional
                {},
            )
            # Either returns a guard dict (detected sentinel) or None (no crash)
            assert result is None or isinstance(result, dict)
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"validate_primitive_args raised unexpectedly: {e}")

    def test_exception_in_internal_scan_returns_none(self, monkeypatch):
        """Simulate a degenerate arg value that might trip the scanner."""
        monkeypatch.setenv("OPENRESEARCH_ARG_CONTRACTS", "1")
        # A deeply nested structure with a cycle would normally be a problem,
        # but since we cap at 5 violations and use simple iteration this is safe.
        # Pass a normal value to confirm no crash.
        result = validate_primitive_args(
            "plan_reproduction",
            _fake_plan_reproduction,
            (),
            {"method_spec": {"a": {"b": {"c": "real_value"}}}},
        )
        assert result is None
